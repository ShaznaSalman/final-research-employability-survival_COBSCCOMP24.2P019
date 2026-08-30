"""Competing risks (secondary analysis): some graduates move onto another path - a non-IT job, self-employment or
further studies - before any IT role. The main analysis treats them as still looking for IT work (censored), which
assumes they would find IT work at the same rate as similar graduates who are still looking.

Here, starting another path (optional item Q16a) is a competing event:
  - Aalen-Johansen cumulative incidence of "first IT role" and of "another path" (vs 1 - Kaplan-Meier, which
    overstates the IT share when competing events are common);
  - cause-specific Cox models for each event, compared with the main model's hazard ratios.
A Fine-Gray model is not fitted (no installed package provides it; no new dependency was added).

Run:  python -m core.competing      (needs data/processed/graduates_competing.csv from `python -m core.clean`)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import AalenJohansenFitter, KaplanMeierFitter  # noqa: E402

from core import evaluate, inference, preprocess, schema, splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"
TIMES = (3, 6, 12, 24)
TIE_TOLERANCE = 0.25                 # > lifelines' tie jitter, < the 0.5-month data resolution
MIN_EVENTS = 10                      # fewer competing events: descriptive results only


def cif(df: pd.DataFrame, cause: int, times=TIMES) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                         # tied times are jittered (seeded)
        aj = AalenJohansenFitter(calculate_variance=False, seed=evaluate.SEED)
        aj.fit(df["cr_time"], df["cr_cause"], event_of_interest=cause)
    curve = aj.cumulative_density_.iloc[:, 0]
    # Tied times are jittered by a tiny amount; times are whole months (or 0.5), so read just after each month.
    at = [t + TIE_TOLERANCE for t in times]
    return [float(curve[curve.index <= a].iloc[-1]) if (curve.index <= a).any() else 0.0 for a in at], curve


def run(write: bool = True) -> dict:
    # training set only (core/splits.py): the competing-risks models never see validation or test graduates.
    # Graduates in the competing-risks file but not in the modelling set (e.g. further studies) are kept unless
    # they belong to the validation or test set.
    train = splits.load("train")
    held_out = set(splits.attach(pd.read_csv(PROCESSED / "graduates_train.csv"))
                   .query("split != 'train'")["respondent_id"].astype(str))
    cr = pd.read_csv(PROCESSED / "graduates_competing.csv")
    cr = cr[~cr["respondent_id"].astype(str).isin(held_out)].reset_index(drop=True)
    info = json.loads((PROCESSED / "dataset_info.json").read_text(encoding="utf-8"))
    counts = cr["cr_cause"].value_counts().reindex([0, 1, 2], fill_value=0).astype(int).to_dict()
    params = schema.model_parameter_count()

    cif_it, curve_it = cif(cr, 1)
    cif_alt, curve_alt = cif(cr, 2)
    km = KaplanMeierFitter().fit(cr["cr_time"], cr["cr_cause"] == 1)
    naive = [1 - float(km.survival_function_at_times(t).iloc[0]) for t in TIMES]

    pre, X_main = inference.design(train)
    t, e = preprocess.outcome(train)
    tables = {"main": inference.finish(inference.hr_table(inference.fit_cox(X_main, t, e)))}
    X = pre.transform(cr)
    tables["it"] = inference.finish(inference.hr_table(inference.fit_cox(X, cr["cr_time"], cr["cr_cause"] == 1)))
    if counts[2] >= MIN_EVENTS:
        tables["alt"] = inference.finish(inference.hr_table(inference.fit_cox(X, cr["cr_time"], cr["cr_cause"] == 2)))

    out = {"n": len(cr), "events": counts, "alt_date_missing": int(cr["alt_date_missing"].sum()),
           "cif_it": dict(zip(TIMES, map(lambda v: round(v, 3), cif_it))),
           "one_minus_km": dict(zip(TIMES, map(lambda v: round(v, 3), naive))),
           "cif_other_path": dict(zip(TIMES, map(lambda v: round(v, 3), cif_alt)))}
    if write:
        pd.concat([v.assign(model=k) for k, v in tables.items()]).to_csv(ARTIFACTS / "competing_risks.csv",
                                                                          index=False)
        plot(curve_it, curve_alt, km, ARTIFACTS / "figures" / "competing_cif.png")
        (ARTIFACTS / "competing_risks.md").write_text(render(out, tables, params, info), encoding="utf-8")
    return out


def plot(curve_it, curve_alt, km, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(curve_it.index, curve_it.values, where="post", color="#2a78d6", lw=2, label="First IT role (CIF)")
    ax.step(curve_alt.index, curve_alt.values, where="post", color="#eb6834", lw=2, label="Another path (CIF)")
    sf = km.survival_function_
    ax.step(sf.index, 1 - sf.iloc[:, 0], where="post", color="#2a78d6", lw=1.2, ls="--",
            label="IT role, 1 - Kaplan-Meier (other path censored)")
    ax.set(xlabel="Months after degree completion", ylabel="Cumulative share", ylim=(0, 1),
           xlim=(0, schema.PREDICTION_HORIZON_MONTHS))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render(out, tables, params, info) -> str:
    ev = out["events"]
    L = ["# Competing risks: IT role vs another path", "",
         "_Generated by `python -m core.competing` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        L += ["> **SYNTHETIC DATA - DEMO ONLY.** In the synthetic data, the timing of non-IT paths was generated "
              "independently of the IT-job process, so these results only show that the analysis works.", ""]
    L += [f"{out['n']} graduates: {ev[1]} found a first IT role, {ev[2]} started another path first (non-IT job, "
          f"self-employment or further studies, with the month from Q16a), {ev[0]} were still looking at the "
          f"survey. {out['alt_date_missing']} graduates on another path did not give the month and are treated "
          "as still looking, as in the main analysis.", "",
          "## Share who had reached each outcome", "",
          "| Months | IT role (cumulative incidence) | IT role (1 - Kaplan-Meier) | Another path (cumulative incidence) |",
          "|---|---|---|---|"]
    for t in TIMES:
        L.append(f"| {t} | {out['cif_it'][t]:.1%} | {out['one_minus_km'][t]:.1%} | {out['cif_other_path'][t]:.1%} |")
    L += ["", "1 - Kaplan-Meier answers 'what share would find IT work if nobody took another path'; the cumulative "
          "incidence answers 'what share actually found IT work first'. The gap shows how much the competing "
          "path matters.", "", "![Cumulative incidence](figures/competing_cif.png)", "",
          "## Cause-specific hazard ratios (95% CI)", "",
          "HR > 1: associated with reaching that outcome sooner (numeric features per one SD of the training data)."
          " The IT-role model treats another path as censored, like the main model, but also includes graduates in "
          "further studies who gave a start month.", ""]
    keys = list(tables)
    head = {"main": "Main model (IT role)", "it": "Cause-specific: IT role", "alt": "Cause-specific: another path"}
    L += ["| Feature | " + " | ".join(head[k] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for col in tables["main"]["column"]:
        L.append(f"| {tables['main'].set_index('column').loc[col, 'label']} | "
                 + " | ".join(inference.fmt_hr(tables[k].set_index("column").loc[col]) for k in keys) + " |")
    if "alt" not in tables:
        L += ["", f"Only {ev[2]} graduates started another path (fewer than {MIN_EVENTS}): no model was fitted for it."]
    elif ev[2] / params < schema.EPV_WARNING_THRESHOLD:
        L += ["", f"**The 'another path' model has {ev[2] / params:.1f} events per parameter (below "
              f"{schema.EPV_WARNING_THRESHOLD}): treat it as exploratory.**"]
    L += ["", "Cause-specific hazard ratios describe the rate of each outcome among graduates who have reached "
          "neither yet. They are associations, not effects of the features.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
