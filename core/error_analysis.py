"""Error analysis of the selected model (REPORT ONLY - it does not change the selected model or the app).

Where does the selected AFT Log-Normal model go wrong? Same fixed split as the main analysis (core/splits.py; no data
reuse): the model is fitted on the TRAINING set exactly as in core/evaluate.py and its predicted median time to the
first IT job is compared with what happened on the VALIDATION set. The locked test set is never used.

  - graduates who found work: predicted median vs actual months, by how long they actually took
  - graduates still searching: share whose predicted median had already passed (the model expected a job by now)
  - the worst-predicted tenth vs the rest: how they differ on the model inputs

Writes artifacts/error_analysis.{json,md} (group-level summaries only) and data/processed/error_analysis_cases.csv
(the worst individual cases, for the researcher; git-ignored with the other respondent-level files).
Run:  python -m core.error_analysis
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from core import evaluate, preprocess, splits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CASES_PATH = ROOT / "data" / "processed" / "error_analysis_cases.csv"
DISTRIBUTION = "lognormal"                      # the selected model (artifacts/selection.json)
HORIZON = float(evaluate.CURVE_TIMES[-1])       # medians beyond the prediction grid are reported as "> 24"
BANDS = [(-np.inf, 0.5, "at or before completion"), (0.5, 3, "1-3 months"), (3, 6, "4-6 months"),
         (6, 12, "7-12 months"), (12, np.inf, "more than 12 months")]
PROFILE = {"gpa": "GPA", "internship_months": "Internship (months)", "certifications_count": "Certifications",
           "project_count": "Projects", "technical_mean": "Technical skills (1-5)"}


def predictions(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Validation graduates with the selected model's predicted median (months), capped at the grid horizon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre, model = evaluate.fit_all(train, {}, DISTRIBUTION, "aft_best", evaluate.SEED)
        curve = model.predict_survival(pre.transform(valid), evaluate.CURVE_TIMES)
    median = evaluate.median_from_curve(curve)
    out = preprocess.raw_feature_frame(valid)[list(PROFILE) + ["degree_group", "university_type"]].copy()
    out["respondent_id"] = valid["respondent_id"].to_numpy()
    out["actual_months"] = valid["duration_months"].to_numpy(float)
    out["event"] = valid["event"].to_numpy(int)
    out["predicted_median"] = np.where(np.isnan(median), HORIZON, median)
    out["beyond_horizon"] = np.isnan(median)
    return out


def summarise(p: pd.DataFrame) -> dict:
    ev, cens = p[p["event"] == 1].copy(), p[p["event"] == 0]
    ev["error"] = ev["predicted_median"] - ev["actual_months"]
    bands = []
    for lo, hi, label in BANDS:
        g = ev[(ev["actual_months"] > lo) & (ev["actual_months"] <= hi)]
        if len(g):
            bands.append({"band": label, "n": len(g), "actual_mean": float(g["actual_months"].mean()),
                          "predicted_mean": float(g["predicted_median"].mean()),
                          "mean_error": float(g["error"].mean()), "mae": float(g["error"].abs().mean())})
    worst = ev["error"].abs() >= ev["error"].abs().quantile(0.9)
    compare = []
    for col, label in PROFILE.items():
        compare.append({"input": label, "worst": float(ev.loc[worst, col].mean()),
                        "rest": float(ev.loc[~worst, col].mean())})
    early = ev["actual_months"] <= 0.5
    compare.append({"input": "Found work at or before completion (%)", "worst": float(100 * early[worst].mean()),
                    "rest": float(100 * early[~worst].mean())})
    return {
        "n_validation": len(p), "events": len(ev), "censored": len(cens),
        "mae_events": float(ev["error"].abs().mean()), "median_abs_error": float(ev["error"].abs().median()),
        "mean_error_events": float(ev["error"].mean()),
        "sd_actual": float(ev["actual_months"].std()), "sd_predicted": float(ev["predicted_median"].std()),
        "predicted_range": [float(p["predicted_median"].min()), float(p["predicted_median"].max())],
        "censored_past_median": int((cens["predicted_median"] < cens["actual_months"]).sum()),
        "bands": bands, "worst_n": int(worst.sum()), "worst_vs_rest": compare,
        "worst_over": int((ev.loc[worst, "error"] > 0).sum()), "worst_under": int((ev.loc[worst, "error"] < 0).sum()),
    }


def run(out_dir: Path = ARTIFACTS, cases_path: Path = CASES_PATH) -> dict:
    train, valid = splits.load("train"), splits.load("validation")
    p = predictions(train, valid)
    res = summarise(p)
    ev = p[p["event"] == 1].assign(error=lambda d: d["predicted_median"] - d["actual_months"])
    ev.reindex(ev["error"].abs().sort_values(ascending=False).index).head(res["worst_n"]).to_csv(cases_path, index=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "error_analysis.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def to_markdown(r: dict) -> str:
    L = ["# Error analysis of the selected model (report only)", "",
         "_Generated by `python -m core.error_analysis` - do not edit by hand._", "",
         f"AFT Log-Normal fitted on the training set, predicted median time to the first IT job compared with the "
         f"outcome on the validation set ({r['n_validation']} graduates: {r['events']} found work, {r['censored']} "
         f"still searching). The test set is not used. Only group summaries are shown; individual cases are kept in "
         f"the git-ignored `data/processed/error_analysis_cases.csv`.", "",
         "## Graduates who found work, by how long they actually took", "",
         "| Actual time to first IT job | n | Mean actual (months) | Mean predicted median | Mean error | Mean absolute error |",
         "|---|---|---|---|---|---|"]
    for b in r["bands"]:
        L.append(f"| {b['band']} | {b['n']} | {b['actual_mean']:.1f} | {b['predicted_mean']:.1f} | "
                 f"{b['mean_error']:+.1f} | {b['mae']:.1f} |")
    L += ["", f"Overall: mean absolute error **{r['mae_events']:.1f} months** (median {r['median_abs_error']:.1f}); "
          f"mean error {r['mean_error_events']:+.1f} months (positive = predicted too slow). Predicted medians range "
          f"from {r['predicted_range'][0]:.1f} to {r['predicted_range'][1]:.1f} months; their SD "
          f"({r['sd_predicted']:.1f}) is much smaller than that of the actual times ({r['sd_actual']:.1f}), so the "
          f"model pulls every graduate towards the typical search length.", "",
          f"Still searching: {r['censored_past_median']} of {r['censored']} had already been looking longer than "
          f"their predicted median (the model expected them to be working by then).", "",
          f"## The worst-predicted tenth ({r['worst_n']} graduates) vs the rest", "",
          f"{r['worst_over']} were predicted too slow and {r['worst_under']} too fast.", "",
          "| Input | Worst-predicted tenth (mean) | Rest (mean) |", "|---|---|---|"]
    for c in r["worst_vs_rest"]:
        L.append(f"| {c['input']} | {c['worst']:.2f} | {c['rest']:.2f} |")
    L += ["", "The largest errors are graduates whose search was much shorter or much longer than their profile "
          "suggests. The model sees only the questionnaire inputs; contacts, search effort and interview "
          "performance are not measured, so it cannot place these graduates at the extremes. The validation set is "
          "small, so treat the comparison as descriptive."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    print(f"MAE {r['mae_events']:.2f} months; censored past median {r['censored_past_median']}/{r['censored']}")
    for b in r["bands"]:
        print(f"{b['band']:25s} n={b['n']:3d} actual {b['actual_mean']:5.1f} predicted {b['predicted_mean']:5.1f}")
