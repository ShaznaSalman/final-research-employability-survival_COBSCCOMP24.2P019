"""Monte Carlo simulation study: does the analysis recover effects that are KNOWN to exist?

The synthetic generator (core/synthetic.py) fixes the true effect of every predictor. This study repeatedly
draws a fresh synthetic survey, runs it through the real cleaning step, fits an unpenalised Cox model on the
model features and compares the estimates with the truth. For each sample size it reports:
  - direction recovery: share of replications whose estimate has the true sign
  - power: share with p < 0.05 in the true direction (how many respondents a real survey needs)
  - false-positive rate for predictors whose true effect is ZERO (university type) - should be near 5%
  - mean estimate vs truth and 95% CI coverage, for the predictors the model observes exactly and linearly
    (internship months, projects); others are measured with error (self-ratings) or enter non-linearly, so
    their estimates target a different quantity and coverage is not defined
  - Harrell's C on the same data
Unobserved heterogeneity (gamma frailty) attenuates Cox hazard ratios towards zero, so coverage below 95% for
correctly specified predictors is expected and is itself a finding about real survey data.

Writes artifacts/simulation_study.json and artifacts/simulation_study.md.
Run:  python -m core.simulation_study [--reps 100]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from core import clean, preprocess, schema, synthetic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SAMPLE_SIZES = (150, 300, 600, 1200)
REPS = 100
SEED0 = 50_000

LABELS = {"gpa": "GPA", "internship_months": "Internship months", "certifications_count": "Certifications",
          "project_count": "Projects", "technical_mean": "Technical skills (self-rated)",
          "soft_mean": "Soft skills (self-rated)", "degree_group[it_is]": "Degree: IT/IS vs computing",
          "degree_group[specialised]": "Degree: specialised vs computing",
          "university_type[non_state]": "University: non-state vs state",
          "university_type[foreign_affiliated]": "University: foreign-affiliated vs state"}
EXACT = ("internship_months", "project_count")            # observed exactly, enter linearly in the truth
NULL = ("university_type[non_state]", "university_type[foreign_affiliated]")


def design(df: pd.DataFrame) -> pd.DataFrame:
    """The model features exactly as the pipeline derives them (same recodes and block means), one-hot encoded."""
    feats = list(schema.MODEL_FEATURES)
    df = preprocess.BlockMeans().transform(preprocess.DerivedFeatures().transform(df))
    X = df[feats + ["duration_months", "event"]].copy()
    for c, t in schema.MODEL_FEATURES.items():
        if t != "categorical":
            X[c] = X[c].astype(float).fillna(X[c].median())
    X = X.dropna(subset=[c for c, t in schema.MODEL_FEATURES.items() if t == "categorical"])
    for c, t in schema.MODEL_FEATURES.items():
        if t == "categorical":
            levels = list(schema.CATEGORICAL_LEVELS[c])
            X[c] = pd.Categorical(X[c], categories=levels)
            for lv in levels[1:]:
                X[f"{c}[{lv}]"] = (X[c] == lv).astype(float)
            X = X.drop(columns=c)
    return X


def one_rep(n: int, seed: int, lookup: dict) -> dict | None:
    raw, _, _ = synthetic.generate_graduates(n_graduates=n, n_final_year=0, seed=seed, defects=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, _ = clean.clean_graduates(raw, lookup, clean.Audit())
        X = design(df)
        try:
            cph = CoxPHFitter().fit(X, "duration_months", "event")
        except Exception:
            return None
    s = cph.summary
    return {"n_used": len(X), "events": int(X["event"].sum()), "c_index": float(cph.concordance_index_),
            **{f: {"coef": float(s.loc[f, "coef"]), "lo": float(s.loc[f, "coef lower 95%"]),
                   "hi": float(s.loc[f, "coef upper 95%"]), "p": float(s.loc[f, "p"])} for f in LABELS if f in s.index}}


def summarise(reps: list[dict]) -> dict:
    truth = synthetic.TRUE_LOG_HAZARD_RATIOS
    out = {"reps": len(reps), "mean_n_used": float(np.mean([r["n_used"] for r in reps])),
           "mean_events": float(np.mean([r["events"] for r in reps])),
           "c_index_mean": float(np.mean([r["c_index"] for r in reps])),
           "c_index_sd": float(np.std([r["c_index"] for r in reps])), "effects": {}}
    for f, label in LABELS.items():
        rows = [r[f] for r in reps if f in r]
        if not rows:
            continue
        t = truth[f]
        coef = np.array([x["coef"] for x in rows])
        p = np.array([x["p"] for x in rows])
        lo, hi = np.array([x["lo"] for x in rows]), np.array([x["hi"] for x in rows])
        e = {"label": label, "true": t, "mean_estimate": float(coef.mean()), "sd_estimate": float(coef.std())}
        if t == 0:
            e["false_positive_rate"] = float((p < 0.05).mean())
        else:
            e["direction_recovery"] = float((np.sign(coef) == np.sign(t)).mean())
            e["power"] = float(((p < 0.05) & (np.sign(coef) == np.sign(t))).mean())
        if f in EXACT:
            e["bias"] = float(coef.mean() - t)
            e["coverage"] = float(((lo <= t) & (t <= hi)).mean())
        out["effects"][f] = e
    return out


def run(reps: int = REPS, sizes=SAMPLE_SIZES, out_dir: Path = ARTIFACTS) -> dict:
    lookup = synthetic.lookup_dict()
    result = {"sample_sizes": list(sizes), "reps_per_size": reps, "true_log_hazard_ratios": synthetic.TRUE_LOG_HAZARD_RATIOS,
              "frailty_variance": synthetic.FRAILTY_VARIANCE, "by_size": {}}
    for n in sizes:
        rows = [r for i in range(reps) if (r := one_rep(n, SEED0 + 10_000 * sizes.index(n) + i, lookup))]
        result["by_size"][str(n)] = summarise(rows)
        print(f"    n={n}: {len(rows)} reps, C={result['by_size'][str(n)]['c_index_mean']:.3f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "simulation_study.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "simulation_study.md").write_text(to_markdown(result), encoding="utf-8")
    return result


def to_markdown(r: dict) -> str:
    sizes = r["sample_sizes"]
    L = ["# Simulation study: recovery of known effects", "",
         "_Generated by `python -m core.simulation_study` - do not edit by hand._", "",
         f"{r['reps_per_size']} synthetic surveys per sample size, each cleaned by the real pipeline and fitted with an "
         "unpenalised Cox model on the model features. True effects are fixed in `core/synthetic.py`; gamma frailty "
         f"variance {r['frailty_variance']} (unobserved luck, networks) attenuates hazard ratios towards zero.", "",
         "## Power: share of surveys where the effect is detected (p < 0.05, right direction)", "",
         "| Predictor | True log-HR | " + " | ".join(f"n={n}" for n in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    first = r["by_size"][str(sizes[0])]["effects"]
    for f, e in first.items():
        if e["true"] == 0:
            continue
        L.append(f"| {e['label']} | {e['true']:+.2f} | " +
                 " | ".join(f"{100 * r['by_size'][str(n)]['effects'][f]['power']:.0f}%" for n in sizes) + " |")
    L += ["", "## False positives for predictors with NO true effect (expected about 5%)", "",
          "| Predictor | " + " | ".join(f"n={n}" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for f in NULL:
        L.append(f"| {first[f]['label']} | " +
                 " | ".join(f"{100 * r['by_size'][str(n)]['effects'][f]['false_positive_rate']:.0f}%" for n in sizes) + " |")
    L += ["", "## Bias and 95% CI coverage (predictors observed exactly and linearly)", "",
          "| Predictor | True | " + " | ".join(f"n={n}: mean (coverage)" for n in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    for f in EXACT:
        L.append(f"| {first[f]['label']} | {first[f]['true']:+.3f} | " +
                 " | ".join(f"{r['by_size'][str(n)]['effects'][f]['mean_estimate']:+.3f} "
                            f"({100 * r['by_size'][str(n)]['effects'][f]['coverage']:.0f}%)" for n in sizes) + " |")
    L += ["", "## Discrimination", "", "| n | Mean events | Harrell's C (mean +/- SD) |", "|---|---|---|"]
    L += [f"| {n} | {r['by_size'][str(n)]['mean_events']:.0f} | {r['by_size'][str(n)]['c_index_mean']:.3f} +/- "
          f"{r['by_size'][str(n)]['c_index_sd']:.3f} |" for n in sizes]
    L += ["", "## Reading this", "",
          "- Power tells a real survey how many respondents it needs to detect an effect of a given size.",
          "- Self-rated skills are a noisy, inflated proxy for true skill, so their estimates are attenuated: the analysis "
          "can detect that skills matter, but under-states by how much.",
          "- Coverage below 95% for correctly specified predictors reflects frailty attenuation (the Cox model estimates a "
          "population-averaged effect smaller than the individual-level truth), not a coding error.",
          "- C-index here is in-sample; the locked test-set figure in model_comparison.md is the honest estimate."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    run(reps=a.reps)
