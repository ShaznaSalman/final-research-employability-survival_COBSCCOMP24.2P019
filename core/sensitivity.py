"""Sensitivity analyses: do the results depend on data decisions made during cleaning?

Each variant changes ONE decision in the training and validation sets, fits the selected model and Cox PH on the
changed TRAINING set, scores them on the changed VALIDATION set (core/splits.py; the test set is never used), and
fits an unpenalised Cox model on the training set for hazard ratios. Variants were fixed before looking at any
result:

  job_0.25 / job_1.0   job found on/before completion counted as 0.25 or 1.0 months instead of 0.5
  no_heaped            graduates whose job-start time is a common "round" recall value (3, 6, 9, 12, 18, 24
                       months) excluded - recall rounding (heaping) is likely there; genuine values are lost too
  no_careless          straight-liners (identical answers to all skill items) and erratic responders
                       (Mahalanobis distance on the skill items, p < 0.001) excluded
  complete_case        only graduates who answered every model input (no imputation)
  multiple_imputation  missing inputs imputed m = 20 times (chained equations, outcome included in the
                       imputation model), Cox fitted on each, pooled with Rubin's rules (hazard ratios only)

The main analysis is unchanged; this module only reports. Run:  python -m core.sensitivity
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer

from core import evaluate, inference, preprocess, schema, splits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
INFO_PATH = ROOT / "data" / "processed" / "dataset_info.json"

HEAPED_MONTHS = (3, 6, 9, 12, 18, 24)
MAHALANOBIS_P = 0.001
N_IMPUTATIONS = 20
LIKERT = list(schema.TECH_COLUMNS + schema.SOFT_COLUMNS)
RAW_INPUTS = ["gpa", "degree_type", "university_type", "certifications_count", "internship_months",
              "project_count", *LIKERT]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _job_rule(months: float):
    def f(df):
        df = df.copy()
        df.loc[df["job_before_graduation"].astype(bool), "duration_months"] = months
        return df
    return f


def heaped(df: pd.DataFrame) -> pd.Series:
    return (df["event"] == 1) & df["duration_months"].isin(HEAPED_MONTHS)


def careless(df: pd.DataFrame) -> pd.Series:
    """Straight-lining (no variation across answered skill items) or Mahalanobis outlier on complete items."""
    items = df[LIKERT].astype(float)
    straight = (items.std(axis=1) == 0) & (items.notna().sum(axis=1) >= 2)
    complete = items.dropna()
    d2 = pd.Series(0.0, index=df.index)
    if len(complete) > len(LIKERT) + 1:
        diff = complete - complete.mean()
        inv = np.linalg.pinv(np.cov(complete.to_numpy(), rowvar=False))
        d2.loc[complete.index] = np.einsum("ij,jk,ik->i", diff.to_numpy(), inv, diff.to_numpy())
    erratic = pd.Series(chi2.sf(d2, len(LIKERT)) < MAHALANOBIS_P, index=df.index) & items.notna().all(axis=1)
    return straight | erratic


def incomplete(df: pd.DataFrame) -> pd.Series:
    return df[RAW_INPUTS].isna().any(axis=1)


VARIANTS = {
    "main": ("Main analysis", lambda df: df),
    "job_0.25": ("Job on/before completion = 0.25 months", _job_rule(0.25)),
    "job_1.0": ("Job on/before completion = 1.0 month", _job_rule(1.0)),
    "no_heaped": ("Excluding 'round number' job-start times", lambda df: df[~heaped(df)]),
    "no_careless": ("Excluding straight-liners and erratic responders", lambda df: df[~careless(df)]),
    "complete_case": ("Complete cases only (no imputation)", lambda df: df[~incomplete(df)]),
}


# ---------------------------------------------------------------------------
# Multiple imputation
# ---------------------------------------------------------------------------
def multiple_imputation(df: pd.DataFrame, pre, m: int = N_IMPUTATIONS, seed: int = evaluate.SEED) -> pd.DataFrame:
    """Pooled Cox log-HRs over m imputations of the model inputs (after derivation and block means)."""
    base = preprocess.derive_and_block(df)
    num, cat = preprocess.NUMERIC_FEATURES, preprocess.CATEGORICAL_FEATURES
    t, e = preprocess.outcome(df)
    helper = np.column_stack([np.log(t), e])                       # outcome in the imputation model
    obs = base[num].to_numpy(float)
    lo, hi = np.nanmin(obs, axis=0), np.nanmax(obs, axis=0)
    tables = []
    for k in range(m):
        rng = np.random.default_rng(seed + k)
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + k,
                               min_value=np.r_[lo, -np.inf, -np.inf], max_value=np.r_[hi, np.inf, np.inf])
        filled = base.copy()
        filled[num] = imp.fit_transform(np.column_stack([obs, helper]))[:, :len(num)]
        for c in cat:                                              # draw from the observed distribution
            miss = filled[c].isna()
            if miss.any():
                freq = filled[c].value_counts(normalize=True)
                filled.loc[miss, c] = rng.choice(freq.index.to_numpy(), size=int(miss.sum()), p=freq.to_numpy())
        X = pre.named_steps["columns"].transform(filled)
        tables.append(inference.hr_table(inference.fit_cox(X, t, e)))
    return inference.rubin(tables)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def selected_model() -> str:
    path = ARTIFACTS / "selection.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected"] if path.exists() else "cox_ph"


def run(repeats: int = evaluate.REPEATS) -> dict:
    start = time.time()
    df, valid = splits.load("train"), splits.load("validation")       # df = the training set
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    selected = selected_model()
    models = tuple(dict.fromkeys(["cox_ph", selected]))
    pre, _ = inference.design(df)                                  # fixed scaling (training set) for every variant

    rows, hrs = [], {}
    for key, (label, fn) in VARIANTS.items():
        d, v = fn(df).reset_index(drop=True), fn(valid).reset_index(drop=True)
        mrows, _, _ = evaluate.holdout(d, v, models_to_run=models)
        metrics = pd.DataFrame(mrows)
        for mname in models:
            mm = metrics[metrics["model"] == mname].iloc[0]
            rows.append({"variant": key, "label": label, "model": mname, "n": len(d), "events": int(d["event"].sum()),
                         "n_validation": len(v), "uno_c": float(mm["uno_c"]), "uno_c_sd": float(mm["uno_c_se"]),
                         "ibs": float(mm["ibs"])})
        t, e = preprocess.outcome(d)
        hrs[key] = inference.finish(inference.hr_table(inference.fit_cox(pre.transform(d), t, e)))
    hrs["multiple_imputation"] = inference.finish(multiple_imputation(df, pre))

    table = pd.DataFrame(rows)
    table.to_csv(ARTIFACTS / "sensitivity.csv", index=False)
    long = pd.concat([h.assign(variant=k) for k, h in hrs.items()], ignore_index=True)
    long.to_csv(ARTIFACTS / "sensitivity_hazard_ratios.csv", index=False)
    stable = direction_agreement(hrs)
    md = render(table, hrs, stable, selected, info, df, repeats, time.time() - start)
    (ARTIFACTS / "sensitivity.md").write_text(md, encoding="utf-8")
    main_c = table.query("variant == 'main' and model == @selected")["uno_c"].iloc[0]
    return {"selected": selected, "uno_c_range": [round(table.query("model == @selected")["uno_c"].min(), 3),
                                                  round(table.query("model == @selected")["uno_c"].max(), 3)],
            "uno_c_main": round(main_c, 3), "directions_stable": stable["stable"].tolist().count(True),
            "features": len(stable)}


def direction_agreement(hrs: dict) -> pd.DataFrame:
    """Per column: does the direction of the association agree with the main analysis in every variant?"""
    main = hrs["main"].set_index("column")
    out = []
    for col in main.index:
        signs = {k: np.sign(h.set_index("column").loc[col, "log_hr"]) for k, h in hrs.items()}
        clear_main = not (main.loc[col, "lo"] <= 1 <= main.loc[col, "hi"])
        flips = [k for k, s in signs.items() if s != signs["main"]]
        out.append({"column": col, "label": main.loc[col, "label"], "clear_in_main": clear_main,
                    "flips": flips, "stable": not flips})
    return pd.DataFrame(out)


def render(table, hrs, stable, selected, info, df, repeats, runtime) -> str:
    names = {**evaluate.LABELS}
    L = ["# Sensitivity analyses", "", "_Generated by `python -m core.sensitivity` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        L += ["> **SYNTHETIC DATA - DEMO ONLY.** Re-run on the real survey data before reporting.", ""]
    n_heap, n_care, n_inc = int(heaped(df).sum()), int(careless(df).sum()), int(incomplete(df).sum())
    L += ["Each variant changes one data decision (all fixed in advance, see `core/sensitivity.py`). Models are "
          "fitted on the (changed) training set and scored on the (changed) validation set - no graduate is reused "
          "and the test set is not used. Hazard ratios come from an unpenalised Cox model on the training set with "
          "the main analysis' scaling, so they are comparable across variants. N and events refer to the training "
          "set.", "",
          f"Rows affected in the main data: {n_heap} 'round number' job-start times, {n_care} straight-lining or "
          f"erratic responders, {n_inc} graduates with at least one missing input "
          f"({N_IMPUTATIONS} imputations for multiple imputation).", "",
          "## Discrimination by variant", "",
          "| Variant | Model | N | Events | Validation Uno's C (SE) | IBS |", "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        L.append(f"| {r['label']} | {names.get(r['model'], r['model'])} | {r['n']} | {r['events']} | "
                 f"{r['uno_c']:.3f} ({r['uno_c_sd']:.3f}) | {r['ibs']:.3f} |")
    sel = table[table["model"] == selected]
    L += ["", f"For the selected model ({names.get(selected, selected)}), Uno's C ranges from "
          f"{sel['uno_c'].min():.3f} to {sel['uno_c'].max():.3f} across variants (main {sel.iloc[0]['uno_c']:.3f}). "
          "The validation set is small; differences smaller than the bootstrap SE should not be interpreted.", "",
          "## Hazard ratios by variant (95% CI)", "",
          "HR > 1: associated with finding work sooner. Numeric features: per one SD of the training data.", ""]
    keys = list(hrs)
    labels = {**{k: v[0] for k, v in VARIANTS.items()}, "multiple_imputation": "Multiple imputation"}
    L += ["| Feature | " + " | ".join(labels[k] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for col in hrs["main"]["column"]:
        cells = [inference.fmt_hr(hrs[k].set_index("column").loc[col]) for k in keys]
        L.append(f"| {hrs['main'].set_index('column').loc[col, 'label']} | " + " | ".join(cells) + " |")
    flips = stable[~stable["stable"]]
    L += ["", f"**Direction of association unchanged in every variant for {int(stable['stable'].sum())} of "
          f"{len(stable)} model columns.**"]
    for _, r in flips.iterrows():
        L.append(f"- {r['label']}: direction differs in {', '.join(labels[k] for k in r['flips'])}"
                 + (" (its main-analysis CI excludes 1 - worth discussing)." if r["clear_in_main"]
                    else " (its main-analysis CI includes 1, so the direction was uncertain anyway)."))
    L += ["", "Excluding 'round number' times also removes genuine values, and excluding careless responders "
          "assumes the detection rules are right; these variants test robustness, they are not better analyses.",
          "", f"_Runtime: {runtime / 60:.1f} minutes._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
