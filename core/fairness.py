"""Fairness check (backend only - never shown to users).

For the SELECTED model, using its predictions on the LOCKED TEST SET (artifacts/oof_predictions.csv; the model was
fitted on the training set only and never saw these graduates - core/splits.py):
  - Uno's C-index by subgroup (with a bootstrap 95% interval over the test graduates)
  - 6-month calibration by subgroup: mean predicted P(employed by 6 months) vs Kaplan-Meier observed
Subgroups: gender (fairness-only field, never a model feature) and university_type.
Groups with n < MIN_GROUP_N are suppressed.

Risk ranking uses predicted P(employed by 6 months). The censoring distribution for Uno's C is estimated on the
training set (no test outcome is used for it).

Run:  python -m core.fairness     (after core.evaluate)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from core import evaluate, preprocess, splits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
INFO_PATH = ROOT / "data" / "processed" / "dataset_info.json"

MIN_GROUP_N = 10            # the test set is small; groups below this are suppressed
HORIZON = 6
N_BOOT = 200
SEED = 20261006
GROUP_VARS = ("gender", "university_type")
C_GAP_FLAG = 0.05            # flag a C-index gap between groups larger than this
CAL_GAP_FLAG = 0.10          # flag a calibration error larger than this (absolute, probability scale)


def _km_employed(sub: pd.DataFrame, h: float = HORIZON) -> float:
    if sub["duration_months"].max() < h:
        return np.nan
    km = KaplanMeierFitter().fit(sub["duration_months"], sub["event"])
    return 1 - float(km.survival_function_at_times(h).iloc[0])


def _uno(y_all, sub: pd.DataFrame, tau: float) -> float:
    y = preprocess.sksurv_y(sub)
    if y["event"].sum() < 2:
        return np.nan
    try:
        return evaluate.uno_c(y_all, y, 1 - sub[f"S_{HORIZON}"].to_numpy(), tau)
    except ValueError:
        return np.nan


def subgroup_metrics(oof: pd.DataFrame, train: pd.DataFrame, groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (group variable, level) - including 'All' - with discrimination and calibration.
    `train` is the training set (censoring distribution); `groups` holds the group variables of the scored graduates."""
    y_all = preprocess.sksurv_y(train)
    tau = evaluate.truncation_tau(y_all["time"])
    groups = train if groups is None else groups
    oof = oof.merge(groups[["respondent_id", *GROUP_VARS]], on="respondent_id", how="left")
    for g in GROUP_VARS:
        oof[g] = oof[g].fillna("missing")
    first = oof[oof["repeat"] == oof["repeat"].min()]
    rng = np.random.default_rng(SEED)

    rows = []
    levels = [("all", "All")] + [(g, lv) for g in GROUP_VARS for lv in sorted(first[g].unique())]
    for g, level in levels:
        mask_fn = (lambda d: np.ones(len(d), bool)) if g == "all" else (lambda d, g=g, lv=level: d[g] == lv)
        sub0 = first[mask_fn(first)]
        row = {"group": g, "level": level, "n": len(sub0), "events": int(sub0["event"].sum()),
               "suppressed": len(sub0) < MIN_GROUP_N}
        if not row["suppressed"]:
            per_repeat = [(_uno(y_all, r[mask_fn(r)], tau), float((1 - r.loc[mask_fn(r), f"S_{HORIZON}"]).mean()),
                           _km_employed(r[mask_fn(r)])) for _, r in oof.groupby("repeat")]
            c, pred, obs = map(np.array, zip(*per_repeat))
            boot = []
            for _ in range(N_BOOT):
                b = sub0.iloc[rng.integers(0, len(sub0), len(sub0))]
                boot.append(_uno(y_all, b, tau))
            boot = np.array(boot)
            row.update({
                "uno_c": float(np.nanmean(c)), "uno_c_sd_repeats": float(np.nanstd(c, ddof=1)),
                "uno_c_lower": float(np.nanpercentile(boot, 2.5)), "uno_c_upper": float(np.nanpercentile(boot, 97.5)),
                "pred_employed_6m": float(pred.mean()), "obs_employed_6m": float(np.nanmean(obs)),
            })
            row["cal_diff_6m"] = row["pred_employed_6m"] - row["obs_employed_6m"]
        rows.append(row)
    return pd.DataFrame(rows)


def flags(table: pd.DataFrame) -> list[str]:
    out = []
    shown = table[~table["suppressed"]]
    for g in GROUP_VARS:
        t = shown[shown["group"] == g]
        if len(t) >= 2:
            gap = t["uno_c"].max() - t["uno_c"].min()
            if gap > C_GAP_FLAG:
                hi, lo = t.loc[t["uno_c"].idxmax(), "level"], t.loc[t["uno_c"].idxmin(), "level"]
                overlap = t["uno_c_lower"].max() <= t["uno_c_upper"].min()
                out.append(f"{g}: C-index gap of {gap:.3f} between `{hi}` and `{lo}`"
                           + (" (bootstrap intervals overlap - consistent with chance)." if overlap else
                              " (intervals do not overlap - investigate)."))
        for _, r in t.iterrows():
            if abs(r["cal_diff_6m"]) > CAL_GAP_FLAG:
                direction = "over" if r["cal_diff_6m"] > 0 else "under"
                out.append(f"{g} = `{r['level']}`: the model {direction}-predicts employment by {HORIZON} months by "
                           f"{abs(r['cal_diff_6m']):.2f}.")
    return out


def render_report(table: pd.DataFrame, selected: dict, info: dict, flag_list: list[str]) -> str:
    lines = ["# Fairness report (backend only - not shown to users)", "",
             "_Generated by `python -m core.fairness` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        lines += ["> **SYNTHETIC DATA - DEMO ONLY.** In the synthetic data neither gender nor university type has a "
                  "true effect, so any differences below reflect sampling noise.", ""]
    lines += [f"Model: **{selected['label']}** (fitted on the training set, chosen on the validation set). Scored "
              "on the **locked test set** - graduates the model never saw and that were never used to tune or choose "
              "it.", "",
              "## Method", "",
              f"- **Discrimination:** Uno's C within each subgroup, ranking by predicted P(employed by {HORIZON} months); "
              f"with a {N_BOOT}-resample bootstrap 95% interval over the test graduates. The censoring distribution "
              "is estimated on the training set.",
              f"- **Calibration at {HORIZON} months:** mean predicted P(employed by {HORIZON} months) vs Kaplan-Meier "
              "observed in the subgroup. Difference > 0 = model too optimistic.",
              f"- Groups with fewer than {MIN_GROUP_N} graduates are suppressed. Missing gender is its own group.",
              "- Gender is used **only** here; it is never a model input.", "",
              "## Results", "",
              "| Group | Level | n | Events | Uno's C (95% CI) | Predicted employed by 6m | Observed (KM) | Difference |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        if r["suppressed"]:
            lines.append(f"| {r['group']} | {r['level']} | {r['n']} | {r['events']} | suppressed (n < {MIN_GROUP_N}) "
                         "| - | - | - |")
        else:
            lines.append(f"| {r['group']} | {r['level']} | {r['n']} | {r['events']} | {r['uno_c']:.3f} "
                         f"({r['uno_c_lower']:.3f}-{r['uno_c_upper']:.3f}) | {r['pred_employed_6m']:.2f} | "
                         f"{r['obs_employed_6m']:.2f} | {r['cal_diff_6m']:+.2f} |")
    lines += ["", "## Flags", "",
              f"Rules: C-index gap between levels > {C_GAP_FLAG}; |calibration difference| > {CAL_GAP_FLAG}.", ""]
    lines += [f"- {f}" for f in flag_list] or ["- None."]
    lines += ["", "## Interpretation limits", "",
              "- Subgroups are small; wide intervals mean a gap of a few C-index points cannot be distinguished "
              "from chance.",
              "- This checks whether the model works similarly across groups. It does not show that the job "
              "market itself is fair, and the model is decision support only - not a screening tool.", ""]
    return "\n".join(lines)


def run() -> dict:
    oof = pd.read_csv(ARTIFACTS / "oof_predictions.csv")
    selected = json.loads((ARTIFACTS / "selection.json").read_text(encoding="utf-8"))
    oof = oof[oof["model"] == selected["selected"]]
    selected["repeats"] = int(oof["repeat"].nunique())
    train, test = splits.load("train"), splits.load("test")
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    oof = oof.assign(respondent_id=oof["respondent_id"].astype(str))
    table = subgroup_metrics(oof, train, test.assign(respondent_id=test["respondent_id"].astype(str)))
    flag_list = flags(table)
    table.to_csv(ARTIFACTS / "fairness_report.csv", index=False)
    (ARTIFACTS / "fairness_report.md").write_text(render_report(table, selected, info, flag_list), encoding="utf-8")
    return {"groups": len(table), "suppressed": int(table["suppressed"].sum()), "flags": flag_list}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
