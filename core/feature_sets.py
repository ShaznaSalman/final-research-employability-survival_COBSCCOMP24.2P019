"""Pre-registered comparison of the CORE vs EXTENDED predictor sets (schema.FEATURE_SETS).

Every set is fitted on the TRAINING set and scored on the VALIDATION set (core/splits.py; no data reuse, the test
set is never touched). The comparison is paired: the same validation graduates score both sets, and a bootstrap of
the validation set (no model is refitted) gives the share of resamples in which the challenger is better.
The rule below was fixed before looking at any result:

  Adopt a challenger only if, for Cox PH,
     improvement in validation Uno's C >= MIN_DELTA_C   AND   it is better in >= MIN_WIN_SHARE of bootstrap resamples,
  and either EPV(challenger, training set) >= 10, or Elastic Net Cox (penalised, safer at low EPV) meets the same
  criteria.

Each candidate added alone to the core set is also reported (Cox PH), for interpretation only.
This module only RECOMMENDS; the set in use is schema.FEATURE_SET, changed by hand and documented.

Run:  python -m core.feature_sets
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import evaluate, preprocess, schema, splits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
INFO_PATH = ROOT / "data" / "processed" / "dataset_info.json"

MODELS = ("cox_ph", "cox_elasticnet")
CHALLENGERS = ("extended", "transformed")
MIN_DELTA_C = 0.01
MIN_WIN_SHARE = 0.70
CANDIDATES = [f for f in schema.EXTENDED_FEATURES if f not in schema.CORE_FEATURES]
LABELS = {"has_portfolio": "Portfolio before completion (Q14)", "prep_years": "Years of preparation (Q20)",
          "self_learning_freq": "Self-learning frequency (Q21)", "hackathon_count": "Hackathons / events (Q23)",
          "open_source": "Open-source contribution (Q24)", "sdlc_familiarity": "SDLC familiarity (Q29)"}


N_BOOT = 1000


def _paired(base, other, model: str, y_train, tau) -> dict:
    """Paired comparison on the validation set: point difference + bootstrap of the validation graduates."""
    (bm, bp), (om, op) = base, other
    a, b = bm[bm["model"] == model].iloc[0], om[om["model"] == model].iloc[0]
    pa = bp[bp["model"] == model].set_index("respondent_id")
    pb = op[op["model"] == model].set_index("respondent_id").loc[pa.index]
    y = preprocess.sksurv_y(pa.reset_index())
    rng = np.random.default_rng(evaluate.SEED)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y["event"][i].sum() < 2:
            continue
        try:
            d.append(evaluate.uno_c(y_train, y[i], pb["risk"].to_numpy()[i], tau)
                     - evaluate.uno_c(y_train, y[i], pa["risk"].to_numpy()[i], tau))
        except ValueError:
            continue
    d = np.array(d)
    return {"model": model, "c_base": float(a["uno_c"]), "c_other": float(b["uno_c"]),
            "delta_c": float(b["uno_c"] - a["uno_c"]), "delta_sd": float(d.std(ddof=1)), "win_share": float((d > 0).mean()),
            "ibs_base": float(a["ibs"]), "ibs_other": float(b["ibs"]), "folds": int(len(d))}


def _eval(train, valid, features, models, params):
    rows, preds, _ = evaluate.holdout(train, valid, features=features, models_to_run=models, fixed_params=params)
    return pd.DataFrame(rows), preds


def _meets(row: dict) -> bool:
    return row["delta_c"] >= MIN_DELTA_C and row["win_share"] >= MIN_WIN_SHARE


def run(repeats: int = evaluate.REPEATS) -> dict:   # `repeats` unused (no CV)
    start = time.time()
    train, valid = splits.load("train"), splits.load("validation")
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    events = int(train["event"].sum())
    y_train = preprocess.sksurv_y(train)
    tau = evaluate.truncation_tau(y_train["time"])

    core = _eval(train, valid, schema.CORE_FEATURES, MODELS, None)
    epv = {k: events / schema.model_parameter_count(v) for k, v in schema.FEATURE_SETS.items()}

    results, passed = {}, {}
    for name in CHALLENGERS:
        other = _eval(train, valid, schema.FEATURE_SETS[name], MODELS, None)
        rows = [_paired(core, other, m, y_train, tau) for m in MODELS]
        by_model = {r["model"]: r for r in rows}
        results[name] = rows
        if _meets(by_model["cox_ph"]) and (epv[name] >= schema.EPV_WARNING_THRESHOLD
                                           or _meets(by_model["cox_elasticnet"])):
            passed[name] = by_model["cox_ph"]["delta_c"]
    recommendation = max(passed, key=passed.get) if passed else "core"

    singles = []
    for cand in CANDIDATES:
        feats = {**schema.CORE_FEATURES, cand: schema.EXTENDED_FEATURES[cand]}
        m = _eval(train, valid, feats, ("cox_ph",), None)
        singles.append({"candidate": cand, **_paired(core, m, "cox_ph", y_train, tau)})

    comparison = pd.DataFrame([dict(r, challenger=k) for k, rows in results.items() for r in rows])
    comparison.to_csv(ARTIFACTS / "feature_set_comparison.csv", index=False)
    pd.DataFrame(singles).to_csv(ARTIFACTS / "feature_candidates.csv", index=False)
    md = render(results, singles, epv, events, recommendation, passed, info, repeats, time.time() - start)
    (ARTIFACTS / "feature_set_comparison.md").write_text(md, encoding="utf-8")
    cox = {k: next(r for r in rows if r["model"] == "cox_ph") for k, rows in results.items()}
    return {"recommendation": recommendation, "in_use": schema.FEATURE_SET,
            "delta_c_cox": {k: round(v["delta_c"], 3) for k, v in cox.items()},
            "win_share_cox": {k: round(v["win_share"], 2) for k, v in cox.items()},
            "epv": {k: round(v, 1) for k, v in epv.items()}}


SET_DESCRIPTIONS = {
    "extended": lambda: "core + " + ", ".join(LABELS[c] for c in CANDIDATES),
    "transformed": lambda: ("core information with log(1 + count) for internship months, certifications and "
                            "projects, plus an indicator for GPA >= 3.0 (functional forms fixed in advance from "
                            "human capital and signalling theory)"),
}


def render(results, singles, epv, events, recommendation, passed, info, repeats, runtime) -> str:
    names = {"cox_ph": "Cox PH", "cox_elasticnet": "Elastic Net Cox"}
    L = ["# Feature-set comparison (core vs pre-specified alternatives)", "",
         "_Generated by `python -m core.feature_sets` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        L += ["> **SYNTHETIC DATA - DEMO ONLY.** The synthetic data were generated with known functional forms "
              "(including log-shaped effects and a GPA threshold), so results here partly reflect how the fake data "
              "were made. On real data this comparison must be re-run and may come out differently.", ""]
    L += [f"Every set is fitted on the training set ({info['training_n']} graduates, {events} events) and scored on "
          f"the validation set ({info['validation_n']} graduates); the locked test set is not used. Hyperparameters "
          f"are tuned on validation as in `core/evaluate.py`; differences come with a {N_BOOT}-resample bootstrap of "
          "the validation graduates (no refitting).", "",
          "## Sets", "",
          f"- **Core** ({schema.model_parameter_count(schema.CORE_FEATURES)} columns, EPV {epv['core']:.1f}): "
          + ", ".join(schema.CORE_FEATURES) + "."]
    for name in CHALLENGERS:
        L.append(f"- **{name.capitalize()}** ({schema.model_parameter_count(schema.FEATURE_SETS[name])} columns, "
                 f"EPV {epv[name]:.1f}): {SET_DESCRIPTIONS[name]()}.")
    L += ["", "## Pre-registered decision rule", "",
          f"A set replaces **core** only if, for Cox PH, the improvement in validation Uno's C is at least {MIN_DELTA_C} "
          f"**and** it is better in at least {MIN_WIN_SHARE:.0%} of bootstrap resamples; and either its EPV is at least "
          f"{schema.EPV_WARNING_THRESHOLD} or Elastic Net Cox meets the same criteria. If more than one set passes, "
          "the one with the larger improvement is recommended. Combinations of sets are not tested.", "",
          "## Results (paired, validation set)", "",
          "| Set | Model | Uno's C core | Uno's C set | Difference (bootstrap SD) | Set better in | IBS core | IBS set |",
          "|---|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        L += [f"| {name} | {names[r['model']]} | {r['c_base']:.3f} | {r['c_other']:.3f} | {r['delta_c']:+.3f} "
              f"({r['delta_sd']:.3f}) | {r['win_share']:.0%} of {r['folds']} resamples | {r['ibs_base']:.3f} | "
              f"{r['ibs_other']:.3f} |" for r in rows]
    verdict = ", ".join(passed) if passed else "none"
    L += ["", f"Sets meeting the rule: **{verdict}**.", "",
          f"**Recommendation: {recommendation.upper()}.** Currently in use (`schema.FEATURE_SET`): "
          f"**{schema.FEATURE_SET}**." + ("" if recommendation == schema.FEATURE_SET else
                                          " The set in use differs from the recommendation - review and update "
                                          "`FEATURE_SET` in core/schema.py, then re-run `python -m core.run_all`."), "",
          "## Each extended candidate added alone to the core set (Cox PH, for interpretation)", "",
          "| Candidate | Uno's C with it | Difference vs core (SD) | Better in |", "|---|---|---|---|"]
    L += [f"| {LABELS[s['candidate']]} | {s['c_other']:.3f} | {s['delta_c']:+.3f} ({s['delta_sd']:.3f}) | "
          f"{s['win_share']:.0%} |" for s in sorted(singles, key=lambda s: -s["delta_c"])]
    L += ["", "Single-candidate differences are noisy at this sample size; they are not used for the decision "
          "(choosing among many single additions would overfit the comparison).", "",
          f"_Runtime: {runtime / 60:.1f} minutes._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
