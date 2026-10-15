"""Build notebooks/research_walkthrough.ipynb (unexecuted). Execute it afterwards with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/research_walkthrough.ipynb \
        --ExecutePreprocessor.timeout=1800

The notebook calls the project's own code in core/, so every number it shows comes from the same functions as the
thesis. Training and tuning run live on the training and validation sets; the locked test-set result is LOADED from
artifacts/ (it was evaluated once, in core/evaluate.py) and is never recomputed here.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("research_walkthrough.ipynb")
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# =====================================================================================================================
md("""
# Time to first IT job: research walkthrough

**A Survival-Analysis Framework for Predicting and Explaining Time-to-Employment for Final-Year IT Undergraduates in
Sri Lanka** - M. S. F. Shazna

This notebook walks through the whole analysis behind the dissertation, from the cleaned survey data to the final
results, with every table and chart produced in place.

**How it works**

* Every step calls the project's own code in `core/`, so the numbers here are produced by the same functions as the
  thesis and the app (no logic is copied into the notebook).
* **Training and tuning run live** on the training set (620 graduates) and are scored on the validation set (133).
* **The locked test set (132) is not evaluated here.** It was used once, in `core/evaluate.py`, after the model was
  chosen; its results are loaded from `artifacts/`. This keeps the rule "test set used once" intact.
* Only counts, summaries and charts are shown - never individual respondents.
* Reports that are regenerated here are written to a temporary folder, so `artifacts/` is not changed.

| Section | Content |
|---|---|
| 1 | Setup and data overview |
| 2 | Cleaning and outcome construction |
| 3 | Exploratory analysis: correlations, skill ratings, reliability, factor structure |
| 4 | Time to first IT job: Kaplan-Meier, groups, competing paths |
| 5 | The fixed train / validation / test split |
| 6 | Training and hyperparameter tuning |
| 7 | Model comparison and the pre-set selection rule |
| 8 | Model checks: AFT fit, proportional hazards, time ratios |
| 9 | Diagnostics: learning curve, baselines, time-dependent AUC, Brier score, error analysis |
| 10 | Final result on the locked test set (loaded) and fairness |
| 11 | Robustness |
| 12 | Skill gaps: graduates, employers and job advertisements (RQ2) |
| 13 | Usability and the LLM benchmark (RQ4) |
| 14 | Summary and consistency check against the reported results |
""")

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 1. Setup and data overview

The first cell finds the project folder, imports the project's modules and defines three small display helpers.
""")
code(r'''
# ---- Locate the project and make its code importable ------------------------------------------------------------
import io, json, os, shutil, sys, tempfile, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()   # works from notebooks/ or the root
os.chdir(ROOT)                      # the project's code reads data/ and artifacts/ relative to the root
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")   # keep the outputs readable (convergence notes from the survival libraries)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # the project's plotting code writes PNG files; we display them below
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

# The project's own modules - the notebook only calls them, it re-implements nothing
from core import (baselines, deepsurv_curves, eda_figures, error_analysis, evaluate, evaluation_figures,
                  learning_curve, preprocess, sample_comparison, schema, splits, thesis_results,
                  time_dependent_metrics, tuning_results)

ART = ROOT / "artifacts"            # saved results of the main pipeline (python -m core.run_all)
FIG = ART / "figures"

# Scratch folder: reports regenerated in this notebook go here, so artifacts/ is never overwritten
TMP = Path(tempfile.mkdtemp(prefix="walkthrough_"))
(TMP / "figures").mkdir()
shutil.copy(ART / "model_comparison.csv", TMP / "model_comparison.csv")   # some reports read the tuned settings

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 80)


def show_png(path, width=900):
    """Display a PNG file (charts written by the project's plotting code)."""
    display(Image(filename=str(path), width=width))


def show_fig(fig, width=900):
    """Display a matplotlib figure made in this notebook."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    display(Image(data=buf.getvalue(), width=width))


print("Project folder:", ROOT)
''')
md("""
**Data overview.** The cleaning step (`python -m core.clean`) records the size of every dataset. All inputs are real:
the main-study graduate and employer surveys and 324 entry-level IT job advertisements.
""")
code(r'''
# ---- Size of every dataset, as recorded by the cleaning step ----------------------------------------------------
info = json.loads((ROOT / "data" / "processed" / "dataset_info.json").read_text(encoding="utf-8"))
print("Synthetic data used:", info["is_synthetic"])          # False: every reported result uses the real data

overview = pd.DataFrame([
    ("Graduate survey responses (raw)", info["raw_n"]),
    ("Final-year students set aside (no outcome yet)", info["final_year_n"]),
    ("Modelling graduates", info["modelling_n"]),
    ("  ... who had found an IT job (events)", info["modelling_events"]),
    ("  ... still searching at the survey (censored)", info["modelling_n"] - info["modelling_events"]),
    ("Employer responses (after cleaning)", info["employers_n"]),
    ("Job advertisements", info["job_ads_n"]),
], columns=["Dataset", "Count"])
display(overview)
print(f"Share still searching (censored): {1 - info['modelling_events'] / info['modelling_n']:.1%}")
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 2. Cleaning and outcome construction

Each excluded response is recorded with its reason (`data/processed/excluded.csv`). Below, the reasons are counted -
no respondent identifiers are shown.

**Outcome.** The time origin is the month of degree completion. A graduate in a paid IT role is an *event*, with
duration = months from completion to that role; a graduate still searching is *censored* at the survey month.
Survival analysis uses both: the censored graduates tell us they had not found work *by* a given month.
""")
code(r'''
# ---- Why responses were excluded (counts only) ------------------------------------------------------------------
excluded = pd.read_csv(ROOT / "data" / "processed" / "excluded.csv")
reasons = (excluded.groupby(["step", "reason"]).size().rename("Responses").reset_index()
           .sort_values("Responses", ascending=False))
display(reasons)
print("Responses excluded or set aside in total:", len(excluded))
''')
code(r'''
# ---- The modelling graduates: missing inputs and outcome summary ------------------------------------------------
grads = pd.read_csv(splits.TRAIN_PATH)        # all 885 modelling graduates (train + validation + test)

inputs = ["gpa", "degree_type", "university_type", "internship_months", "certifications_count", "project_count",
          "has_portfolio", "prep_start", "self_learning_freq", "hackathon_count", "open_source", "sdlc_familiarity"]
missing = grads[inputs].isna().sum().rename("Missing").to_frame()
missing["Share"] = (missing["Missing"] / len(grads)).map("{:.1%}".format)
display(missing[missing["Missing"] > 0])      # only hackathons has gaps; they are imputed inside the pipeline

print("Graduates hired at or before completion (duration set to 0.5 months):",
      int((grads["duration_months"] <= schema.JOB_BEFORE_GRADUATION_MONTHS).sum()))
display(grads.groupby("event")["duration_months"].describe().rename(index={0: "still searching", 1: "found IT job"}))
''')
md("""
### 2.1 Class balance and the distribution of the outcome

In survival analysis the two "classes" are **found an IT job (event)** and **still searching (censored)**. About one
in four graduates was still searching, a moderate level of censoring that survival models handle directly. The
histogram shows how long each group had been followed: months to the first job for those who found one, months of
searching so far for those still looking.
""")
code(r'''
# ---- Class balance: found an IT job vs still searching (all 885 modelling graduates) ----------------------------
FOUND, SEARCHING = "#2a78d6", "#eb6834"          # one colour per outcome, used in every chart below
counts = grads["event"].map({1: "Found an IT job", 0: "Still searching"}).value_counts()
counts = counts.reindex(["Found an IT job", "Still searching"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.6]})
bars = ax1.bar(counts.index, counts.values, color=[FOUND, SEARCHING], width=0.6)
for bar, n in zip(bars, counts.values):           # label each bar with its count and share
    ax1.annotate(f"{n}\n({n / counts.sum():.1%})", (bar.get_x() + bar.get_width() / 2, n),
                 ha="center", va="bottom", fontsize=10)
ax1.set_ylim(0, counts.max() * 1.25)
ax1.set_ylabel("Graduates")
ax1.set_title(f"Class balance (n = {counts.sum()})", loc="left")

# ---- Distribution of the outcome time, by group (0-24 months; a few longer follow-ups are clipped at 24) --------
bins = np.arange(0, 25, 1)
for value, label, colour in ((1, "Found an IT job: months to the first job", FOUND),
                             (0, "Still searching: months searched so far", SEARCHING)):
    times = grads.loc[grads["event"] == value, "duration_months"].clip(upper=24)
    ax2.hist(times, bins=bins, color=colour, alpha=0.75, label=label, edgecolor="white")
ax2.set_xlabel("Months after degree completion (longer times grouped at 24)")
ax2.set_ylabel("Graduates")
ax2.set_title("Distribution of the outcome time", loc="left")
ax2.legend(frameon=False)
for ax in (ax1, ax2):
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
show_fig(fig)
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 3. Exploratory analysis

All relationships below are calculated on the **training set only** (620 graduates), so the validation and test sets
stay untouched.
""")
code(r'''
# ---- Load the training set and summarise the predictors in their original units --------------------------------
train = splits.load("train")
valid = splits.load("validation")

raw = preprocess.raw_feature_frame(train)          # predictors before imputation and scaling
display(raw.select_dtypes("number").describe().T.round(2))
''')
md("""
### 3.0 Distribution of each predictor

Histograms of the numeric predictors and bar charts of the categorical ones (training set). Skewed counts (most
graduates with few certifications or hackathons) and heaping (many internships of exactly 6 months) are visible here.
""")
code(r'''
# ---- Histograms of the numeric predictors (training set, original units) ----------------------------------------
numeric = {"gpa": "GPA", "internship_months": "Internship (months)", "certifications_count": "Certifications",
           "project_count": "Projects", "technical_mean": "Technical skill score (1-5)",
           "soft_mean": "Soft-skill score (1-5)", "hackathon_count": "Hackathons / events", "prep_years": "Years of preparation"}
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, (col, label) in zip(axes.ravel(), numeric.items()):
    values = pd.to_numeric(raw[col], errors="coerce").dropna()
    discrete = bool(np.allclose(values, values.round()))   # whole-number counts: one bar per value; others: 15 bins
    ax.hist(values, bins=np.arange(values.min(), values.max() + 2) - 0.5 if discrete else 15,
            color="#2a78d6", edgecolor="white")
    ax.set_title(f"{label}\nmean {values.mean():.2f}, median {values.median():.2f} (n = {len(values)})",
                 fontsize=9, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Distribution of the numeric predictors (training set)", x=0.01, ha="left")
fig.tight_layout()
show_fig(fig)

# ---- Bar charts of the categorical predictors, plus gender (used only in the fairness check, never as an input) ---
categorical = {"university_type": "Institution type", "degree_group": "Degree group",
               "has_portfolio": "Portfolio before completion", "open_source": "Open-source contribution",
               "gender": "Gender (fairness check only)"}
READABLE = {"foreign_affiliated": "Foreign-affiliated", "non_state": "Non-state", "state": "State",
            "computing": "Computing", "it_is": "IT / IS", "specialised": "Specialised", "yes": "Yes", "no": "No",
            "male": "Male", "female": "Female", "prefer_not_to_say": "Prefer not to say", "missing": "Missing"}
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, (col, label) in zip(axes, categorical.items()):
    shares = train[col].fillna("missing").astype(str).replace(READABLE).value_counts()
    ax.barh(shares.index[::-1], shares.values[::-1], color="#2a78d6")
    for i, n in enumerate(shares.values[::-1]):    # write the count and share at the end of each bar
        ax.annotate(f"{n} ({n / shares.sum():.0%})", (n, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_xlim(0, shares.max() * 1.45)
    ax.set_title(label, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Distribution of the categorical predictors (training set)", x=0.01, ha="left")
fig.tight_layout()
show_fig(fig)
''')
md("""
### 3.1 Correlation between predictors

Spearman correlations (pairwise complete). Strong correlations between predictors would make individual effects hard
to separate (multicollinearity).
""")
code(r'''
# ---- Correlation heatmap (core/eda_figures.py) --------------------------------------------------------------------
corr = eda_figures.spearman(train)
eda_figures.plot_correlation(corr, len(train), TMP / "figures" / "correlation.png")
show_png(TMP / "figures" / "correlation.png", width=760)

# The strongest correlation between two different predictors
off_diagonal = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
a, b = off_diagonal.idxmax()
print(f"Strongest correlation: {a} and {b}, rho = {corr.loc[a, b]:.2f} - moderate, so no predictor duplicates another")
''')
md("""
### 3.2 Skill self-ratings

How graduates rated the 17 technical and 9 soft skills at degree completion (1-5), sorted by mean.
""")
code(r'''
# ---- Distribution of the 26 skill ratings ---------------------------------------------------------------------
shares = eda_figures.likert_shares(train)
eda_figures.plot_likert(shares, len(train), TMP / "figures" / "likert.png")
show_png(TMP / "figures" / "likert.png", width=760)
''')
md("""
### 3.3 Reliability and factor structure of the skill scores

The model averages the technical items into one score and the soft items into another. That is justified if each
block is reliable (Cronbach's alpha >= 0.70) and dominated by one underlying factor (scree plot with parallel
analysis: factors above the dashed random-data line are retained).
""")
code(r'''
# ---- Cronbach's alpha for each skill block (complete cases, training set) --------------------------------------
def cronbach_alpha(items: pd.DataFrame) -> float:
    """alpha = k/(k-1) * (1 - sum of item variances / variance of the total score)"""
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))

skills = train[list(schema.TECH_COLUMNS + schema.SOFT_COLUMNS)].apply(pd.to_numeric, errors="coerce").dropna()
alpha = {"Technical (17 items)": cronbach_alpha(skills[list(schema.TECH_COLUMNS)]),
         "Soft (9 items)": cronbach_alpha(skills[list(schema.SOFT_COLUMNS)])}
display(pd.Series(alpha, name="Cronbach's alpha").round(3).to_frame())

# ---- Scree plots with parallel analysis (same method and seed as thesis Table E.3) ------------------------------
scree = eda_figures.scree(train)
eda_figures.plot_scree(scree, TMP / "figures" / "scree.png")
show_png(TMP / "figures" / "scree.png")
print({block: f"{v['factors']} factor(s) retained; first factor explains {100 * v['eigenvalues'][0] / v['items']:.1f}%"
       for block, v in scree.items()})
''')
md("""
### 3.4 How representative is the sample?

The sample compared with published Sri Lankan figures (University of Kelaniya, 2022; ICTA, 2019). The published
figures describe other populations, so this shows direction only.
""")
code(r'''
# ---- Sample vs published figures (core/sample_comparison.py) ---------------------------------------------------
comparison = pd.DataFrame(sample_comparison.compare(grads))
comparison["measure"] = comparison["measure"].map(sample_comparison.LABELS)
display(comparison[["measure", "sample", "published", "difference", "source"]])
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 4. Time to first IT job

### 4.1 Kaplan-Meier curve

The Kaplan-Meier estimator gives the share who had found an IT job by each month, using the graduates still
searching correctly. The row beneath the chart is the number still being followed at each time.
""")
code(r'''
# ---- All evaluation figures at once (core/evaluation_figures.py), written to the scratch folder -----------------
# This draws: Kaplan-Meier with numbers at risk, AFT fit, Schoenfeld plot, fairness and robustness charts.
# Fairness and robustness are drawn from the saved reports - the test set is not evaluated again.
ev = evaluation_figures.run(out_dir=TMP)

show_png(TMP / "figures" / "km_at_risk.png")
km = ev["km"]
print(f"Median time to first IT job: {km['median_months']:g} months")
print("Share employed by 3 / 6 / 12 / 24 months:",
      " / ".join(f"{100 * v:.1f}%" for v in km["employed_by"].values()))
''')
md("""
### 4.2 Differences between groups (log-rank tests, training set)
""")
code(r'''
# ---- Log-rank tests by group (core/thesis_results.py) -----------------------------------------------------------
rows = []
for g in thesis_results.km_groups(train):
    for level, v in g["levels"].items():
        rows.append({"Group": g["group"], "Level": level, "n": v["n"], "Employed by 6 m": f"{v['employed_by_6m']:.1%}",
                     "Median (months)": v.get("median_months"), "Log-rank p": round(g["p"], 4)})
display(pd.DataFrame(rows))

# ---- Kaplan-Meier curves for the two clearest differences: GPA and internship -----------------------------------
from lifelines import KaplanMeierFitter
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
splits_by = {"GPA": np.where(train["gpa"] >= 3.0, "GPA 3.0 or higher", "GPA below 3.0"),
             "Internship": np.where(train["internship_completed"] == "yes", "Completed an internship", "No internship")}
for ax, (title, groups) in zip(axes, splits_by.items()):
    for level in sorted(set(groups)):
        sel = groups == level
        kmf = KaplanMeierFitter().fit(train.loc[sel, "duration_months"], train.loc[sel, "event"], label=f"{level} (n = {sel.sum()})")
        ax.step(kmf.survival_function_.index, 1 - kmf.survival_function_.iloc[:, 0], where="post", linewidth=2,
                label=kmf._label)
    ax.set_xlim(0, 24); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_title(f"By {title} (training set)", loc="left"); ax.set_xlabel("Months after degree completion")
    ax.legend(frameon=False, loc="lower right")
axes[0].set_ylabel("Share who found an IT job")
show_fig(fig)
''')
md("""
### 4.3 Competing paths

Some graduates took another path first (a non-IT job, self-employment or further studies). The cumulative incidence
treats that as a competing outcome; 1 - Kaplan-Meier (dashed) would treat it as censoring.
""")
code(r'''
# ---- Competing risks (core/competing.py; saved result) ----------------------------------------------------------
summary = json.loads((ART / "run_summary.json").read_text(encoding="utf-8"))
comp_risks = summary["robustness"]["competing"]
display(pd.DataFrame({"IT role (cumulative incidence)": comp_risks["cif_it"],
                      "IT role (1 - Kaplan-Meier)": comp_risks["one_minus_km"],
                      "Another path (cumulative incidence)": comp_risks["cif_other_path"]}).rename_axis("Months").round(3))
show_png(FIG / "competing_cif.png", width=760)
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 5. The fixed train / validation / test split

Each graduate has exactly **one** role, assigned once before any modelling (`core/splits.py`), stratified by event and
institution type with a fixed seed:

* **train (70%)** - every model is fitted here, including preprocessing;
* **validation (15%)** - hyperparameters are tuned and models compared here;
* **test (15%)** - locked; the selected model was evaluated on it **once**.

No graduate is reused across roles (the supervisor's rule), so cross-validation is not used.
""")
code(r'''
# ---- Size, events and censoring per split ----------------------------------------------------------------------
with_split = splits.attach(grads)
by_split = splits.summary(with_split)
by_split["still searching"] = by_split["graduates"] - by_split["events"]
by_split["share censored"] = (by_split["still searching"] / by_split["graduates"]).map("{:.1%}".format)
display(by_split)

# ---- Class balance in each split: the stratified split keeps about the same share still searching ---------------
fig, ax = plt.subplots(figsize=(9, 3.2))
y = np.arange(len(by_split))
found, searching = by_split["events"].to_numpy(), by_split["still searching"].to_numpy()
ax.barh(y, found, color="#2a78d6", label="Found an IT job")
ax.barh(y, searching, left=found, color="#eb6834", label="Still searching")
for i, (f, s) in enumerate(zip(found, searching)):        # share still searching at the end of each bar
    ax.annotate(f"{s / (f + s):.1%} still searching", (f + s, i), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=9)
ax.set_yticks(y, [f"{r} (n = {n})" for r, n in zip(by_split["split"], by_split["graduates"])])
ax.invert_yaxis()
ax.set_xlim(0, by_split["graduates"].max() * 1.35)
ax.set_xlabel("Graduates")
ax.set_title("Class balance by split (stratified by event and institution type)", loc="left")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
show_fig(fig)
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 6. Training and hyperparameter tuning

Seven models from four families are trained on the training set:

| Family | Models | Tuned |
|---|---|---|
| Cox proportional hazards | Cox PH | ridge penalty |
| Penalised Cox | Elastic Net Cox | penalty strength and L1 ratio |
| Accelerated failure time | Weibull, log-normal, log-logistic | distribution chosen by AIC on the training set |
| Neural network | DeepSurv | width, depth, dropout, learning rate |

For each tuned model, every setting is fitted on the training set and scored on the validation set; the best is
kept. This cell runs the full training and tuning, so it takes a few minutes.
""")
code(r'''
# ---- Train and tune every model on the training set, score on validation (core/evaluate.py) --------------------
metrics_rows, valid_predictions, chosen_settings = evaluate.holdout(train, valid)

display(pd.DataFrame([{"Model": evaluate.LABELS[m], "Chosen setting": json.dumps(p)}
                      for m, p in chosen_settings.items()]))
''')
md("""
### 6.1 Every tuning configuration

The full grid behind the chosen settings: 4 Cox, 15 Elastic Net and 24 DeepSurv settings, plus the three AFT
distributions. The last column checks that the setting chosen here is the one recorded by the main analysis.
""")
code(r'''
# ---- Every setting with its validation Uno's C (core/tuning_results.py) -----------------------------------------
tuning = tuning_results.run(out_dir=TMP)
display(tuning[tuning["chosen"]][["label", "params", "aic", "uno_c", "matches_main_analysis"]].round(3))

# Spread of the validation C across settings: small spreads mean the comparison does not hinge on one setting
spread = tuning.groupby("model")["uno_c"].agg(["min", "max", "count"])
spread["range"] = spread["max"] - spread["min"]
display(spread.round(3))
''')
md("""
### 6.2 DeepSurv training and early stopping

DeepSurv holds out 20% of the training graduates to monitor its loss and stops when that loss stops improving,
keeping the best weights. The two losses are on different scales, so compare their **trends**.
""")
code(r'''
# ---- DeepSurv training and early-stopping loss (core/deepsurv_curves.py) ----------------------------------------
ds = deepsurv_curves.run(out_dir=TMP)
show_png(TMP / "figures" / "deepsurv_loss.png")
print(f"Best epoch {ds['best_epoch']} of {ds['epochs_run']}; training stopped {ds['patience']} epochs later "
      "and the best weights were kept (overfitting control)")
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 7. Model comparison and selection

Metrics on the **validation set**:

* **Uno's C** (primary) - how well the model ranks who finds work first (0.5 = chance, 1 = perfect);
* **integrated Brier score (IBS)** - overall error of the predicted probabilities (lower is better);
* **calibration error** - predicted vs observed share employed by 3, 6 and 12 months.

**Selection rule, fixed in advance:** highest validation Uno's C; if another candidate is within 0.02, prefer the
simpler model (Cox PH = AFT < Elastic Net Cox < DeepSurv), then the better calibrated.
""")
code(r'''
# ---- Summarise the validation results and apply the pre-set selection rule -------------------------------------
comparison_table, calibration = evaluate.summarise_validation(pd.DataFrame(metrics_rows), valid_predictions)
cols = [c for c in ["label", "candidate", "uno_c_mean", "uno_c_lo", "uno_c_hi", "harrell_c_mean", "ibs_mean",
                    "cal_err_3m_mean", "cal_err_6m_mean", "cal_err_12m_mean"] if c in comparison_table]
display(comparison_table[cols].round(3))

selected, reason = evaluate.select_model(comparison_table)
print("Selected:", evaluate.LABELS[selected], "-", reason)
''')
code(r'''
# ---- Validation C-index of every model, and calibration of the selected model -----------------------------------
evaluate.plot_comparison(comparison_table, selected, TMP / "figures" / "model_comparison.png")
show_png(TMP / "figures" / "model_comparison.png", width=760)

evaluate.plot_calibration(calibration, selected, TMP / "figures" / "calibration_selected.png")
show_png(TMP / "figures" / "calibration_selected.png")
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 8. Model checks

### 8.1 Does the log-normal distribution fit?

The mean predicted share employed under each AFT distribution, against the observed Kaplan-Meier curve (training set).
""")
code(r'''
# ---- AFT distribution fit (drawn by evaluation_figures.run in section 4) -----------------------------------------
show_png(TMP / "figures" / "aft_fit.png")
display(pd.DataFrame(ev["aft_fit"]).T.rename(columns={"aic": "AIC", "mean_abs_gap": "mean gap", "max_abs_gap": "largest gap"})
        .sort_values("AIC").round(3))
''')
md("""
### 8.2 Proportional-hazards check (Cox model)

Scaled Schoenfeld residuals: a flat smoothed line means a factor's effect does not change over time. All tests have
p > 0.10, so the assumption holds for the Cox models. (The selected AFT model does not need this assumption.)
""")
code(r'''
# ---- Schoenfeld residuals (drawn by evaluation_figures.run) -----------------------------------------------------
show_png(TMP / "figures" / "schoenfeld_labelled.png")
print("Smallest Schoenfeld p-value:", round(min(ev["schoenfeld_p"].values()), 3))
''')
md("""
### 8.3 Which factors are associated with a shorter search? (RQ1)

The selected model is refitted on the training set with the chosen settings. Its coefficients are shown as **time
ratios** per one standard deviation: below 1 means a shorter search. p-values are also corrected for testing 16
coefficients (Holm).
""")
code(r'''
# ---- Fit the final model on the training set and extract time ratios --------------------------------------------
final = evaluate.fit_final(train, selected, chosen_settings)
coef = final["model"].coefficients.copy()            # AFT coefficients on log(time), per SD of each input

ratios = pd.DataFrame({"time ratio": np.exp(coef["coef"]), "95% CI low": np.exp(coef["lower"]),
                       "95% CI high": np.exp(coef["upper"]), "p": coef["p"]})

# Holm correction for the 16 coefficients
order = ratios["p"].sort_values().index
k = len(order)
holm = np.minimum(1, np.maximum.accumulate(ratios.loc[order, "p"].to_numpy() * (k - np.arange(k))))
ratios.loc[order, "Holm p"] = holm
ratios = ratios.sort_values("time ratio")
display(ratios.round(3))

# ---- Forest plot of the time ratios -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(ratios))
clear = (ratios["95% CI high"] < 1) | (ratios["95% CI low"] > 1)
ax.hlines(y, ratios["95% CI low"], ratios["95% CI high"], color="#9ec5f4", linewidth=3)
ax.scatter(ratios["time ratio"], y, color=np.where(clear, "#2a78d6", "#52514e"), zorder=3)
ax.axvline(1, color="#52514e", linestyle="--", linewidth=1)
ax.set_yticks(y, [evaluation_figures.LABELS.get(f, f) for f in ratios.index], fontsize=8)   # readable names
ax.set_xlabel("Time ratio per SD (95% CI); < 1 = shorter search")
ax.set_title("Adjusted time ratios of the selected model (training set)", loc="left")
show_fig(fig, width=760)
print("Clearly associated (CI excludes 1):", ", ".join(evaluation_figures.LABELS.get(f, f) for f in ratios.index[clear]))
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 9. Diagnostics (training and validation sets only)

### 9.1 Learning curve - would more respondents help?
""")
code(r'''
# ---- Selected model fitted on 25-100% of the training set (core/learning_curve.py) ------------------------------
lc = learning_curve.run(out_dir=TMP)
show_png(TMP / "figures" / "learning_curve.png", width=760)
display(pd.DataFrame(lc["curve"]).round(3))
''')
md("""
### 9.2 Simple baselines - how much do the inputs add?

The same model with fewer inputs. If GPA alone did as well, the other inputs would add nothing.
""")
code(r'''
# ---- Baselines (core/baselines.py): paired bootstrap of the validation graduates --------------------------------
bl = baselines.run(out_dir=TMP)
display(pd.DataFrame(bl["rows"])[["label", "uno_c", "lo", "hi", "diff_vs_selected", "diff_lo", "diff_hi",
                                  "selected_better_share"]].round(3))
''')
md("""
### 9.3 Time-dependent AUC and the Brier score over time

Supplementary measures (not used for selection). The time-dependent AUC is the survival version of the ROC-AUC; the
Brier score is compared month by month with the Kaplan-Meier baseline, which predicts the same curve for everyone.
""")
code(r'''
# ---- Time-dependent metrics for every candidate (core/time_dependent_metrics.py) --------------------------------
td = time_dependent_metrics.run(out_dir=TMP)
display(pd.DataFrame(td["models"]).set_index("label")[["auc_3m", "auc_6m", "auc_12m", "auc_mean", "uno_c", "ibs"]]
        .round(3))
print(f"Kaplan-Meier baseline IBS on validation: {td['ibs_km']:.3f}")
show_png(TMP / "figures" / "brier_over_time.png", width=760)
''')
md("""
### 9.4 Error analysis - where does the model go wrong?

Predicted median time against the actual time, for validation graduates who found work. Group summaries only.
""")
code(r'''
# ---- Error analysis (core/error_analysis.py); individual cases go to the scratch folder, not shown -------------
ea = error_analysis.run(out_dir=TMP, cases_path=TMP / "cases.csv")
display(pd.DataFrame(ea["bands"]).round(2))
print(f"Mean absolute error {ea['mae_events']:.1f} months; predicted medians range "
      f"{ea['predicted_range'][0]:.1f}-{ea['predicted_range'][1]:.1f} months (SD {ea['sd_predicted']:.1f} vs "
      f"{ea['sd_actual']:.1f} actual): the model pulls everyone towards the typical search length")
display(pd.DataFrame(ea["worst_vs_rest"]).round(2))
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 10. Final result on the locked test set (loaded, not recomputed)

The selected model was evaluated **once** on the 132 test graduates, in `core/evaluate.py`, after it had been chosen.
Those saved results are loaded here; nothing in this notebook fits, tunes or scores on the test set.
""")
code(r'''
# ---- Load the one test-set evaluation ---------------------------------------------------------------------------
test = json.loads((ART / "test_metrics.json").read_text(encoding="utf-8"))
display(pd.DataFrame([{
    "Model": test["model"], "Graduates": test["n"], "Events": test["events"],
    "Uno's C": f"{test['uno_c']:.3f} ({test['uno_c_lo']:.3f}-{test['uno_c_hi']:.3f})",
    "Harrell's C": round(test["harrell_c"], 3), "IBS": round(test["ibs"], 3),
    "Calibration error 3 / 6 / 12 m": f"{test['cal_err_3m']:.3f} / {test['cal_err_6m']:.3f} / {test['cal_err_12m']:.3f}",
}]).T.rename(columns={0: "Locked test set"}))

model_checks = json.loads((ART / "thesis_results.json").read_text(encoding="utf-8"))["model_checks"]
print(f"Kaplan-Meier baseline IBS on the test set: {model_checks['ibs_test_km_baseline']:.3f} "
      f"(model {model_checks['ibs_test_model']:.3f}) - the model ranks graduates but its probabilities are not better")
show_png(FIG / "calibration_test.png")
''')
md("""
### 10.1 Fairness (test set, from the saved fairness report)

Uno's C and calibration at six months by gender and institution type. Gender is never a model input.
""")
code(r'''
# ---- Fairness chart (drawn from artifacts/fairness_report.md by evaluation_figures.run) --------------------------
show_png(TMP / "figures" / "fairness.png")
display(pd.DataFrame(ev["fairness"])[["group", "level", "n", "events", "uno_c", "lo", "hi", "predicted_6m",
                                      "observed_6m"]])
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 11. Robustness

The selected model's validation Uno's C under alternative data decisions (sensitivity analysis), on later cohorts
when trained on earlier ones (temporal validation), and with each institution type held out of training.
""")
code(r'''
# ---- Robustness chart and the underlying tables (saved reports) -------------------------------------------------
show_png(TMP / "figures" / "robustness.png")
sensitivity = pd.read_csv(ART / "sensitivity.csv")
display(sensitivity[sensitivity["model"] == "aft_best"][["label", "n", "events", "uno_c", "uno_c_sd", "ibs"]].round(3))
display(pd.read_csv(ART / "temporal_validation.csv").round(3))
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 12. Skill gaps: graduates, employers and job advertisements (RQ2)

Gap = employers' importance rating minus graduates' self-rating (both 1-5), tested with the Mann-Whitney U test and
corrected for 26 tests (Benjamini-Hochberg q). The share of job advertisements naming each skill is shown alongside.
""")
code(r'''
# ---- Per-skill alignment (core/thesis_results.py) ---------------------------------------------------------------
align, align_summary = thesis_results.alignment()
display(align[["skill", "kind", "graduate_mean", "employer_mean", "gap", "rank_biserial", "q", "ad_share"]].round(3))
print(f"Employer importance vs share of ads (Spearman): rho = {align_summary['spearman_rho']}, "
      f"p = {align_summary['p']:.3f}")

# ---- Dumbbell chart: graduates vs employers per skill, with the share of ads -------------------------------------
a = align.sort_values("gap")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={"width_ratios": [3, 1]})
y = np.arange(len(a))
ax1.hlines(y, a["graduate_mean"], a["employer_mean"], color="#cccccc", linewidth=2)
ax1.scatter(a["graduate_mean"], y, color="#2a78d6", label="Graduates (self-rating)", zorder=3)
ax1.scatter(a["employer_mean"], y, color="#eb6834", label="Employers (importance)", zorder=3)
ax1.set_yticks(y, a["skill"], fontsize=8); ax1.set_xlabel("Mean rating (1-5)"); ax1.legend(frameon=False, loc="lower right")
ax1.set_title("Graduate self-ratings vs employer importance", loc="left"); ax1.grid(axis="x", alpha=0.3)
ax2.barh(y, 100 * a["ad_share"], color="#9ec5f4"); ax2.set_xlabel("% of job ads"); ax2.set_title("Job-ad demand", loc="left")
show_fig(fig)
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 13. Usability and the LLM benchmark (RQ4)

### 13.1 System Usability Scale (12 participants)
""")
code(r'''
# ---- SUS results (core/sus_study.py; saved result) --------------------------------------------------------------
sus = json.loads((ART / "sus_study.json").read_text(encoding="utf-8"))["summary"]
display(pd.DataFrame([{"Participants": sus["n"], "Mean SUS": sus["mean"],
                       "95% CI": f"{sus['ci_low']}-{sus['ci_high']}", "Benchmark": sus["benchmark"],
                       "t-test p (one-sided)": round(sus["p_one_sided"], 4), "Grade": sus["grade"]}]).T
        .rename(columns={0: "Usability"}))
show_png(FIG / "sus_scores.png", width=760)
''')
md("""
### 13.2 The model against a general-purpose LLM

The same 132 test profiles were given to ChatGPT; both are scored with Uno's C. (Saved result of
`python -m benchmark.score`.)
""")
code(r'''
# ---- LLM benchmark (saved result) -------------------------------------------------------------------------------
llm = json.loads((ART / "llm_benchmark.json").read_text(encoding="utf-8"))["comparisons"][0]
display(pd.DataFrame([{"LLM": llm["llm"], "Profiles": llm["profiles"], "LLM Uno's C": llm["uno_c"],
                       "Model Uno's C": llm["model_uno_c"],
                       "Difference (95% CI)": f"{llm['diff'][0]:+.3f} ({llm['diff'][1]:+.3f} to {llm['diff'][2]:+.3f})"}]))
''')

# ---------------------------------------------------------------------------------------------------------------------
md("""
## 14. Summary and consistency check

The key numbers computed live in this notebook are compared with the results reported in the thesis (saved in
`artifacts/`). Every row should match.
""")
code(r'''
# ---- Compare live results with the saved, reported ones ---------------------------------------------------------
selection = json.loads((ART / "selection.json").read_text(encoding="utf-8"))
sel_row = comparison_table.set_index("model").loc[selected]
gpa_ratio = ratios.loc["gpa", "time ratio"] if "gpa" in ratios.index else np.nan
intern_ratio = ratios.loc["internship_months", "time ratio"] if "internship_months" in ratios.index else np.nan
saved_ratios = json.loads((ART / "thesis_results.json").read_text(encoding="utf-8"))["time_ratios"]

checks = [
    ("Modelling graduates", len(grads), info["modelling_n"]),
    ("Kaplan-Meier median (months)", km["median_months"], 4.0),
    ("Selected model", evaluate.LABELS[selected], "AFT (best by AIC)"),
    ("Validation Uno's C, selected model", round(sel_row["uno_c_mean"], 3), selection["uno_c_validation"]),
    ("Time ratio: GPA", round(gpa_ratio, 2), round(saved_ratios["gpa"]["time_ratio"], 2)),
    ("Time ratio: internship months", round(intern_ratio, 2), round(saved_ratios["internship_months"]["time_ratio"], 2)),
    ("Cronbach's alpha, technical", round(alpha["Technical (17 items)"], 3), 0.925),
    ("Cronbach's alpha, soft", round(alpha["Soft (9 items)"], 3), 0.902),
    ("Learning curve at 100%", round(lc["curve"][-1]["uno_c_mean"], 3), selection["uno_c_validation"]),
    ("Test Uno's C (loaded, evaluated once)", round(test["uno_c"], 3), selection["uno_c"]),
]
check = pd.DataFrame(checks, columns=["Result", "This notebook", "Reported"])
check["Match"] = [str(a) == str(b) for a, b in zip(check["This notebook"], check["Reported"])]
display(check)
print("All key results match the reported ones." if check["Match"].all() else "Some results differ - see above.")
''')
md("""
### What the results say

* **RQ1 - factors.** Most graduates found IT work quickly (median 4 months). After adjustment, only **GPA** and
  **internship experience** were clearly associated with a shorter search; institution type and degree group were not.
* **RQ2 - skill gaps.** Graduates rated themselves below employers' expectations on 25 of 26 skills, most in cloud
  computing, software testing and DevOps; job advertisements asked most for problem solving, teamwork and
  communication.
* **RQ3 - prediction.** The log-normal AFT model was selected on validation (C 0.693) and evaluated once on the
  locked test set (C 0.619): it ranks graduates better than chance, but its probabilities under-predict employment and
  are no better than the Kaplan-Meier baseline. GPA and internship carry most of the ranking (section 9.2); the
  learning curve is levelling off, so the remaining variation most likely reflects factors not measured (networks,
  search effort, interviews).
* **RQ4 - artifact.** Usability was rated good (SUS 76.25, above 68), and the model ranked the same test graduates
  better than ChatGPT.
""")
code(r'''
# ---- Clean up the scratch folder --------------------------------------------------------------------------------
shutil.rmtree(TMP, ignore_errors=True)
print("Done. artifacts/ was not modified by this notebook.")
''')

# =====================================================================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
