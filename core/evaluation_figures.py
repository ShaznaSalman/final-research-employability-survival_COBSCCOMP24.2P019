"""Evaluation figures for the thesis appendix (REPORT ONLY - they draw existing results; nothing is re-selected).

  - km_at_risk          : Kaplan-Meier time to first IT job for all modelling graduates (descriptive, as in Chapter 5),
                          with the number still at risk, to 24 months
  - aft_fit             : fit of the three AFT distributions on the TRAINING set - mean predicted share employed vs
                          Kaplan-Meier (the AIC choice, shown visually)
  - schoenfeld_labelled : proportional-hazards check of the Cox model on the TRAINING set (same method as
                          core/evaluate.py, readable labels)
  - fairness            : Uno's C and calibration by subgroup, drawn from artifacts/fairness_report.md - the TEST-set
                          figures already reported; the test set is not evaluated again
  - robustness          : validation Uno's C of the selected model across the sensitivity variants, on later cohorts
                          and with each institution type held out (from the existing reports)

Writes artifacts/figures/{km_at_risk,aft_fit,schoenfeld_labelled,fairness,robustness}.png and
artifacts/evaluation_figures.{json,md}.
Run:  python -m core.evaluation_figures
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from core import evaluate, preprocess, splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
HORIZON = 24
AT_RISK_TIMES = (0, 3, 6, 12, 18, 24)
# Categorical slots 1-3 of the reference palette (validated); each also has its own dash style (secondary encoding)
DIST_STYLE = {"lognormal": ("#2a78d6", "-", "Log-normal"), "loglogistic": ("#eb6834", "--", "Log-logistic"),
              "weibull": ("#1baf7a", ":", "Weibull")}
LABELS = {"gpa": "GPA", "internship_months": "Internship months", "certifications_count": "Certifications",
          "project_count": "Projects", "technical_mean": "Technical skill score", "soft_mean": "Soft-skill score",
          "has_portfolio": "Portfolio", "prep_years": "Years of preparation",
          "self_learning_freq": "Self-learning frequency", "hackathon_count": "Hackathons",
          "open_source": "Open source", "sdlc_familiarity": "SDLC familiarity",
          "degree_group_it_is": "Degree: IT / IS (vs computing)",
          "degree_group_specialised": "Degree: specialised (vs computing)",
          "university_type_non_state": "Institution: non-state (vs state)",
          "university_type_foreign_affiliated": "Institution: foreign-affiliated (vs state)"}
GROUP_LABELS = {"female": "Women", "male": "Men", "state": "State", "non_state": "Non-state",
                "foreign_affiliated": "Foreign-affiliated", "All": "All test graduates"}


# ---------------------------------------------------------------------------------------------- 1. Kaplan-Meier
def km_at_risk(path: Path) -> dict:
    df = pd.read_csv(splits.TRAIN_PATH)                     # all modelling graduates (descriptive, as in Chapter 5)
    km = KaplanMeierFitter().fit(df["duration_months"], df["event"])
    t = km.survival_function_.index.to_numpy()
    keep = t <= HORIZON
    s = km.survival_function_.iloc[:, 0].to_numpy()[keep]
    ci = km.confidence_interval_survival_function_.to_numpy()[keep]
    fig, (ax, tab) = plt.subplots(2, 1, figsize=(7.5, 4.6), facecolor=evaluate.SURFACE,
                                  gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
    evaluate._style(ax)
    ax.fill_between(t[keep], 1 - ci[:, 0], 1 - ci[:, 1], step="post", color=evaluate.ACCENT_LIGHT, alpha=0.6,
                    linewidth=0, label="95% confidence interval")
    ax.step(t[keep], 1 - s, where="post", color=evaluate.ACCENT, linewidth=2, label="Share who had found an IT job")
    med = float(km.median_survival_time_)
    ax.axhline(0.5, color=evaluate.INK_MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"median {med:g} months", (med, 0.5), textcoords="offset points", xytext=(8, -14), fontsize=8,
                color=evaluate.INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share who found an IT job", fontsize=9, color=evaluate.INK_MUTED)
    ax.set_title(f"Time to first IT job (Kaplan-Meier, n = {len(df)}, {int(df['event'].sum())} found work)",
                 fontsize=11, color=evaluate.INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    at_risk = [int((df["duration_months"] >= x).sum()) for x in AT_RISK_TIMES]
    tab.set_facecolor(evaluate.SURFACE)
    for s_ in tab.spines.values():
        s_.set_visible(False)
    tab.set_yticks([])
    for x, n in zip(AT_RISK_TIMES, at_risk):
        tab.text(x, 0.5, str(n), ha="center", va="center", fontsize=9, color=evaluate.INK)
    tab.text(-0.6, 0.5, "At risk", ha="right", va="center", fontsize=8, color=evaluate.INK_MUTED)
    tab.set_xticks(AT_RISK_TIMES)
    tab.tick_params(colors=evaluate.INK_MUTED, labelsize=9, length=0)
    tab.set_xlim(-0.5, HORIZON + 0.5)
    tab.set_xlabel("Months after degree completion", fontsize=9, color=evaluate.INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)
    return {"n": len(df), "events": int(df["event"].sum()), "median_months": med,
            "employed_by": {h: round(1 - float(km.predict(h)), 3) for h in (3, 6, 12, 24)},
            "at_risk": dict(zip(AT_RISK_TIMES, at_risk))}


# ---------------------------------------------------------------------------------------------- 2. AFT fit
def aft_fit(train: pd.DataFrame, path: Path) -> dict:
    pre = preprocess.build_preprocessor().fit(train)
    X = pre.transform(train)
    t, e = preprocess.outcome(train)
    grid = np.arange(0, HORIZON + 0.01, 0.25)
    km = KaplanMeierFitter().fit(t, e)
    km_emp = 1 - km.survival_function_at_times(grid).to_numpy()
    ci = km.confidence_interval_survival_function_
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=evaluate.SURFACE)
    evaluate._style(ax)
    ax.fill_between(ci.index, 1 - ci.iloc[:, 0], 1 - ci.iloc[:, 1], step="post", color=evaluate.GRID, linewidth=0)
    ax.step(km.survival_function_.index, 1 - km.survival_function_.iloc[:, 0], where="post", color=evaluate.INK,
            linewidth=1.5, label="Kaplan-Meier (observed, 95% CI)")
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d, (color, ls, label) in DIST_STYLE.items():
            m = evaluate.fit_model(X, t, e, f"aft_{d}", {}, evaluate.SEED)
            pred = 1 - m.predict_survival(X, grid).mean(axis=0)
            gap = np.abs(pred - km_emp)
            res[d] = {"aic": float(m.aic), "max_abs_gap": float(gap.max()), "mean_abs_gap": float(gap.mean())}
            ax.plot(grid, pred, color=color, linestyle=ls, linewidth=2,
                    label=f"{label} AFT (AIC {m.aic:,.1f})")
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Months after degree completion", fontsize=9, color=evaluate.INK_MUTED)
    ax.set_ylabel("Share who found an IT job", fontsize=9, color=evaluate.INK_MUTED)
    ax.set_title(f"Fit of the AFT distributions (training set, n = {len(train)})", fontsize=11, color=evaluate.INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------------------------- 3. Schoenfeld
def schoenfeld(train: pd.DataFrame, path: Path) -> dict:
    comp = pd.read_csv(ARTIFACTS / "model_comparison.csv").set_index("model")
    params = json.loads(comp.loc["cox_ph", "tuned_params"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph, _, resid = evaluate.ph_check(train, params)
    resid = resid.rename(columns=LABELS)
    ph = ph.rename(index=LABELS)
    evaluate.plot_schoenfeld(resid, ph, path)
    return {k: round(float(v), 3) for k, v in ph["p"].items()}


# ---------------------------------------------------------------------------------------------- 4. fairness
def fairness_rows(path: Path = ARTIFACTS / "fairness_report.md") -> pd.DataFrame:
    """Rows of the results table in the existing fairness report (test set; not recomputed)."""
    rows = []
    pat = re.compile(r"^\| (\w+) \| (\w+) \| (\d+) \| (\d+) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \| ([\d.]+) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            g, lvl, n, ev, c, lo, hi, pred, obs = m.groups()
            rows.append({"group": g, "level": lvl, "n": int(n), "events": int(ev), "uno_c": float(c),
                         "lo": float(lo), "hi": float(hi), "predicted_6m": float(pred), "observed_6m": float(obs)})
    return pd.DataFrame(rows)


def plot_fairness(f: pd.DataFrame, path: Path):
    f = f.iloc[::-1].reset_index(drop=True)                 # "All" on top
    labels = [f"{GROUP_LABELS.get(r.level, r.level)} (n = {r.n})" for r in f.itertuples()]
    y = np.arange(len(f))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=evaluate.SURFACE, sharey=True)
    for ax in (a1, a2):
        evaluate._style(ax)
        ax.grid(axis="y", visible=False)
    a1.hlines(y, f["lo"], f["hi"], color=evaluate.ACCENT_LIGHT, linewidth=3)
    a1.scatter(f["uno_c"], y, s=50, color=evaluate.ACCENT, edgecolors=evaluate.SURFACE, linewidths=1.5, zorder=3)
    a1.axvline(0.5, color=evaluate.INK_MUTED, linestyle="--", linewidth=1)
    a1.set_yticks(y, labels, fontsize=8, color=evaluate.INK)
    a1.set_xlabel("Uno's C (95% CI); 0.5 = chance", fontsize=9, color=evaluate.INK_MUTED)
    a1.set_title("Ranking by subgroup", fontsize=10, color=evaluate.INK, loc="left")
    a2.hlines(y, f["predicted_6m"], f["observed_6m"], color=evaluate.GRID, linewidth=2)
    a2.scatter(f["predicted_6m"], y, s=45, color=evaluate.ACCENT, label="Predicted", zorder=3,
               edgecolors=evaluate.SURFACE, linewidths=1.5)
    a2.scatter(f["observed_6m"], y, s=45, color=evaluate.INK, marker="D", label="Observed (Kaplan-Meier)", zorder=3,
               edgecolors=evaluate.SURFACE, linewidths=1.5)
    a2.set_xlim(0.4, 1.0)
    a2.set_xlabel("Share employed by 6 months", fontsize=9, color=evaluate.INK_MUTED)
    a2.set_title("Calibration by subgroup", fontsize=10, color=evaluate.INK, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Fairness check on the locked test set (from the fairness report)", x=0.01, ha="left", fontsize=11,
                 color=evaluate.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- 5. robustness
def robustness_rows() -> pd.DataFrame:
    sens = pd.read_csv(ARTIFACTS / "sensitivity.csv")
    sens = sens[sens["model"] == "aft_best"]
    rows = [{"check": f"Sensitivity: {r.label}", "group": "Sensitivity (validation)", "uno_c": r.uno_c,
             "sd": r.uno_c_sd, "n": int(r.n_validation)} for r in sens.itertuples()]
    temp = pd.read_csv(ARTIFACTS / "temporal_validation.csv").set_index("model")
    later_n = json.loads((ARTIFACTS / "run_summary.json").read_text(encoding="utf-8"))["robustness"]["temporal"]["test_n"]
    rows.append({"check": "Later cohorts (trained on earlier cohorts)", "group": "Temporal",
                 "uno_c": float(temp.loc["aft_best", "uno_c"]), "sd": np.nan, "n": int(later_n)})
    tr = json.loads((ARTIFACTS / "thesis_results.json").read_text(encoding="utf-8"))["institution_transfer"]
    for r in tr:
        rows.append({"check": f"Held-out institution type: {GROUP_LABELS.get(r['held_out_type'], r['held_out_type'])}",
                     "group": "Institution transfer", "uno_c": r["uno_c"], "sd": np.nan, "n": r["validation_n"]})
    return pd.DataFrame(rows)


def plot_robustness(r: pd.DataFrame, main_c: float, path: Path):
    r = r.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(r))
    labels = [c + (f" (n = {int(n)})" if pd.notna(n) and not c.startswith("Sensitivity") else "")
              for c, n in zip(r["check"], r["n"])]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(r) + 1.6), facecolor=evaluate.SURFACE)
    evaluate._style(ax)
    ax.grid(axis="y", visible=False)
    has_sd = r["sd"].notna()
    ax.hlines(y[has_sd], (r["uno_c"] - r["sd"])[has_sd], (r["uno_c"] + r["sd"])[has_sd],
              color=evaluate.ACCENT_LIGHT, linewidth=3)
    ax.scatter(r["uno_c"], y, s=50, color=evaluate.ACCENT, edgecolors=evaluate.SURFACE, linewidths=1.5, zorder=3)
    for yi, c in zip(y, r["uno_c"]):
        ax.annotate(f"{c:.3f}", (c, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8,
                    color=evaluate.INK)
    ax.axvline(main_c, color=evaluate.ACCENT, linestyle=":", linewidth=1)
    ax.axvline(0.5, color=evaluate.INK_MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels, fontsize=8, color=evaluate.INK)
    ax.set_xlim(0.45, 0.8)
    ax.set_xlabel(f"Uno's C (bars: +/- 1 SD; dotted line: main result {main_c:.3f}; dashed line: chance)",
                  fontsize=9, color=evaluate.INK_MUTED)
    ax.set_title("Robustness of the selected model's discrimination", fontsize=11, color=evaluate.INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- run
def run(out_dir: Path = ARTIFACTS) -> dict:
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    train = splits.load("train")
    res = {"km": km_at_risk(figs / "km_at_risk.png"), "aft_fit": aft_fit(train, figs / "aft_fit.png"),
           "schoenfeld_p": schoenfeld(train, figs / "schoenfeld_labelled.png")}
    f = fairness_rows()
    plot_fairness(f, figs / "fairness.png")
    res["fairness"] = f.to_dict(orient="records")
    rb = robustness_rows()
    main_c = float(rb.loc[rb["check"] == "Sensitivity: Main analysis", "uno_c"].iloc[0])
    plot_robustness(rb, main_c, figs / "robustness.png")
    res["robustness"] = rb.replace({np.nan: None}).to_dict(orient="records")
    (out_dir / "evaluation_figures.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out_dir / "evaluation_figures.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def to_markdown(r: dict) -> str:
    k, a = r["km"], r["aft_fit"]
    best = min(a, key=lambda d: a[d]["aic"])
    rb = pd.DataFrame(r["robustness"])
    L = ["# Evaluation figures (report only)", "",
         "_Generated by `python -m core.evaluation_figures` - do not edit by hand._", "",
         f"- `figures/km_at_risk.png`: all {k['n']} modelling graduates (descriptive); median {k['median_months']:g} "
         f"months; employed by 3/6/12/24 months: " + " / ".join(f"{100 * v:.1f}%" for v in k["employed_by"].values())
         + f"; at risk at {', '.join(f'{t} m: {n}' for t, n in k['at_risk'].items())}.",
         f"- `figures/aft_fit.png`: training set. Largest gap between mean predicted and Kaplan-Meier share employed "
         "(0-24 months): " + "; ".join(f"{d} {100 * v['max_abs_gap']:.1f} points (AIC {v['aic']:,.1f})"
                                       for d, v in a.items()) + f". Lowest AIC: {best}.",
         f"- `figures/schoenfeld_labelled.png`: training set; smallest p = {min(r['schoenfeld_p'].values()):.3f} "
         "(no evidence against proportional hazards).",
         "- `figures/fairness.png`: drawn from `fairness_report.md` (test set; not recomputed).",
         f"- `figures/robustness.png`: selected model's Uno's C from {rb['uno_c'].min():.3f} to "
         f"{rb['uno_c'].max():.3f} across {len(rb)} checks (sensitivity, temporal and institution-transfer reports)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    print("KM:", r["km"]["median_months"], r["km"]["employed_by"])
    print("AFT:", {d: (round(v["aic"], 1), round(v["max_abs_gap"], 3)) for d, v in r["aft_fit"].items()})
    print("min Schoenfeld p:", min(r["schoenfeld_p"].values()))
    print("fairness rows:", len(r["fairness"]), "| robustness rows:", len(r["robustness"]))
