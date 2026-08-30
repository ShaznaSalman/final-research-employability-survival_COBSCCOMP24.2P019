"""Exploratory accuracy experiments (REPORT ONLY - they do not change the selected model or the app).

Asks how much discrimination more flexible methods add on the same data, using the SAME fixed split as the main
analysis (core/splits.py; no data reuse): every method is fitted on the TRAINING set and tuned and scored on the
VALIDATION set; the locked test set is never used. Intervals come from a bootstrap of the validation graduates:
  - cox_core      : Cox PH on the core features (the main analysis' model family, penalizer tuned)
  - cox_flexible  : Cox PH with non-linear and interaction terms (GPA^2, GPA >= 3.0, log counts,
                    technical skills x degree group), penalizer tuned
  - gbm           : gradient-boosted Cox model (scikit-survival), tuned
  - rsf           : random survival forest (scikit-survival), tuned
  - neural nets   : DeepSurv, DeepHit, Cox-Time and N-MTLR (pycox; core/models/neural.py), tuned
  - ensemble      : average of the rank-normalised risks of cox_core, gbm and rsf
and the theoretical ceiling: the C-index of the generator's TRUE risk score (synthetic data only).

The main model family list and selection rule were fixed in advance (core/models/__init__.py); these results are
exploratory and reported as such. Writes artifacts/accuracy_experiments.{json,md}.
Run:  python -m core.accuracy_experiments
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

from core import evaluate, preprocess, schema, synthetic
from core.models.deepsurv import DeepSurvModel
from core.models.neural import CoxTimeModel, DeepHitModel, NMTLRModel

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
LABELS = {"cox_core": "Cox PH (core features)", "cox_flexible": "Cox PH + non-linear & interaction terms",
          "gbm": "Gradient-boosted Cox", "rsf": "Random survival forest",
          "deepsurv": "DeepSurv (neural net)", "deephit": "DeepHit (neural net)", "coxtime": "Cox-Time (neural net)",
          "nmtlr": "N-MTLR (neural net)", "ensemble": "Ensemble (Cox + GBM + RSF)"}
NEURAL = {"deepsurv": DeepSurvModel, "deephit": DeepHitModel, "coxtime": CoxTimeModel, "nmtlr": NMTLRModel}
GRIDS = {
    "cox": [{"penalizer": p} for p in (0.1, 0.5, 1.0)],
    "gbm": [{"n_estimators": n, "learning_rate": 0.05, "max_depth": d, "subsample": 0.8}
            for n in (100, 250) for d in (1, 2)],
    "rsf": [{"n_estimators": 300, "min_samples_leaf": m, "max_features": "sqrt"} for m in (10, 25)],
    "neural": [{"width": w, "dropout": d, "lr": 1e-3} for w in (16, 32) for d in (0.1, 0.3)],
}
FLEX_FEATURES = {**schema.CORE_FEATURES, "gpa_3plus": "numeric", "log_internship_months": "numeric",
                 "log_certifications": "numeric", "log_projects": "numeric"}


def _flex(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["gpa_sq"] = X["gpa"] ** 2
    for g in ("degree_group_it_is", "degree_group_specialised"):
        X[f"technical_x_{g}"] = X["technical_mean"] * X[g]
    return X


def _design(train, test, kind):
    feats = FLEX_FEATURES if kind == "cox_flexible" else None
    Xa, Xb, _ = preprocess.fit_transform_fold(train, test, feats)
    if kind == "cox_flexible":
        Xa, Xb = _flex(Xa), _flex(Xb)
    return Xa, Xb


def _fit_predict(kind, params, Xa, ya, Xb, seed):
    if kind in ("cox_core", "cox_flexible"):
        df = Xa.assign(_T=ya["time"], _E=ya["event"].astype(int))
        m = CoxPHFitter(penalizer=params["penalizer"]).fit(df, "_T", "_E")
        return np.asarray(m.predict_log_partial_hazard(Xb), float).ravel()
    if kind in NEURAL:
        m = NEURAL[kind](seed=seed, **params).fit(Xa, ya["time"], ya["event"])
        return m.predict_risk(Xb)
    if kind == "gbm":
        return GradientBoostingSurvivalAnalysis(random_state=seed, **params).fit(Xa, ya).predict(Xb)
    return RandomSurvivalForest(random_state=seed, n_jobs=1, **params).fit(Xa, ya).predict(Xb)


def _one(kind, train, valid, seed):
    """Tune on validation (fit on train, score on validation); return the validation risk of the best setting."""
    warnings.simplefilter("ignore")
    grid = GRIDS["neural" if kind in NEURAL else "cox" if kind.startswith("cox") else kind]
    Xa, Xb = _design(train, valid, kind)
    ya, yb = preprocess.sksurv_y(train), preprocess.sksurv_y(valid)
    tau = evaluate.truncation_tau(ya["time"])
    best, best_s, best_risk = grid[0], -np.inf, None
    for p in grid:
        try:
            risk = _fit_predict(kind, p, Xa, ya, Xb, seed)
            sc = evaluate.uno_c(ya, yb, risk, tau)
        except Exception:
            continue
        if sc > best_s:
            best, best_s, best_risk = p, sc, risk
    return kind, best, best_risk


def ceiling(n_surveys: int = 20) -> dict:
    """C-index of the generator's true risk score on fresh synthetic surveys of the training size."""
    warnings.simplefilter("ignore")
    vals = []
    for s in range(n_surveys):
        _, t, _ = synthetic.generate_graduates(seed=90_000 + s, defects=False)
        m = t[t["in_market"]]
        vals.append(concordance_index(m["duration_months"], -m["linear_predictor"], m["event"]))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "surveys": n_surveys}


N_BOOT = 500


def run(repeats: int = evaluate.REPEATS, out_dir: Path = ARTIFACTS) -> dict:   # `repeats` unused (no CV)
    from core import splits
    train, valid = splits.load("train"), splits.load("validation")
    kinds = ("cox_core", "cox_flexible", "gbm", "rsf", *NEURAL)
    res = Parallel(n_jobs=-1)(delayed(_one)(k, train, valid, evaluate.SEED) for k in kinds)
    risks = {k: r for k, _, r in res}
    params = {k: p for k, p, _ in res}
    risks["ensemble"] = np.mean([rankdata(risks[m]) / len(valid) for m in ("cox_core", "gbm", "rsf")], axis=0)
    params["ensemble"] = {}
    ya, yb = preprocess.sksurv_y(train), preprocess.sksurv_y(valid)
    tau = evaluate.truncation_tau(ya["time"])
    point = {k: evaluate.uno_c(ya, yb, r, tau) for k, r in risks.items()}
    rng = np.random.default_rng(evaluate.SEED)
    boot = {k: [] for k in risks}
    for _ in range(N_BOOT):                                    # bootstrap of the validation graduates only
        i = rng.integers(0, len(yb), len(yb))
        if yb["event"][i].sum() < 2:
            continue
        try:
            vals = {k: evaluate.uno_c(ya, yb[i], r[i], tau) for k, r in risks.items()}
        except ValueError:
            continue
        for k, v in vals.items():
            boot[k].append(v)
    boot = {k: np.array(v) for k, v in boot.items()}
    m = pd.DataFrame([{"model": k, "params": json.dumps(params[k]), "uno_c": point[k]} for k in risks])
    summary = []
    for name in LABELS:
        d = boot[name] - boot["cox_core"]
        summary.append({"model": name, "label": LABELS[name], "uno_c_mean": float(point[name]),
                        "uno_c_sd": float(boot[name].std(ddof=1)),
                        "delta_vs_cox": float(point[name] - point["cox_core"]), "delta_sd": float(d.std(ddof=1)),
                        "folds_better": float((d > 0).mean()) if name != "cox_core" else None})
    info = json.loads(evaluate.INFO_PATH.read_text(encoding="utf-8"))
    result = {"folds": 1, "n_validation": len(valid), "summary": summary,
              "is_synthetic": bool(info.get("is_synthetic", True)),
              "ceiling": ceiling() if info.get("is_synthetic", True) else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_dir / "accuracy_experiments_validation.csv", index=False)
    (out_dir / "accuracy_experiments_folds.csv").unlink(missing_ok=True)
    (out_dir / "accuracy_experiments.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "accuracy_experiments.md").write_text(to_markdown(result), encoding="utf-8")
    return result


def to_markdown(r: dict) -> str:
    L = ["# Exploratory accuracy experiments (report only)", "",
         "_Generated by `python -m core.accuracy_experiments` - do not edit by hand._", "",
         "Same fixed split as the main analysis: every method fitted on the training set, tuned and scored on the "
         f"validation set ({r.get('n_validation', '?')} graduates); the test set is not used. \u00b1 = bootstrap SD "
         "over the validation graduates. These do **not** change the selected model: the model families and "
         "selection rule were fixed in advance.", "",
         "| Method | Validation Uno's C (\u00b1 SD) | Change vs Cox PH | Better in (bootstrap) |", "|---|---|---|---|"]
    for s in r["summary"]:
        fb = "-" if s["folds_better"] is None else f"{100 * s['folds_better']:.0f}%"
        dv = "-" if s["model"] == "cox_core" else f"{s['delta_vs_cox']:+.3f} ± {s['delta_sd']:.3f}"
        L.append(f"| {s['label']} | {s['uno_c_mean']:.3f} ± {s['uno_c_sd']:.3f} | {dv} | {fb} |")
    if r["ceiling"]:
        c = r["ceiling"]
        L += ["", "## Theoretical ceiling (synthetic data)", "",
              f"A model that knew every graduate's TRUE risk would reach C = **{c['mean']:.3f} ± {c['sd']:.3f}** "
              f"(Harrell's C, {c['surveys']} fresh surveys). The remaining variation - luck, networks, interviews, and the gap "
              "between self-rated and true skill - is not in the data, so no method can recover it."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    for s in r["summary"]:
        print(f"{s['label']:45s} {s['uno_c_mean']:.3f}  {s['delta_vs_cox']:+.3f}")
    if r["ceiling"]:
        print("ceiling", round(r["ceiling"]["mean"], 3))
