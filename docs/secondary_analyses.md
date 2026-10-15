# Secondary analyses (robustness)

These analyses test whether the main results depend on data decisions, on which cohorts were used, on
graduates clustering within institutions, or on graduates leaving for other paths. **None of them changes
the main model or the app**; each writes a report to `artifacts/` and is re-run by `python -m core.run_all`.
All rules were fixed before looking at results.

| Analysis | Question | Command | Report |
|---|---|---|---|
| Sensitivity | Do results change if one cleaning decision changes? | `python -m core.sensitivity` | `artifacts/sensitivity.md` |
| Temporal validation | Does a model trained on earlier cohorts work for later ones? | `python -m core.temporal` | `artifacts/temporal_validation.md` |
| Institution clustering | Are confidence intervals too narrow because graduates share institutions? | `python -m core.clustering` | `artifacts/institution_clustering.md` |
| Competing risks | What if starting a non-IT job / further studies is treated as a competing event? | `python -m core.competing` | `artifacts/competing_risks.md` |
| Job-market study (**real** ads) | What do entry-level IT ads ask for, by role and seniority? | `python -m core.market_study` (also in `run_all`) | `artifacts/job_market_study.md` |
| Simulation study | Does the analysis recover effects known to exist, and how many respondents are needed? | `python -m core.simulation_study --reps 200` (about 5 min, not in `run_all`) | `artifacts/simulation_study.md` |
| Coding agreement | Do the keyword rules code job ads the way a person would? | `python -m core.coding_agreement --sample`, hand-code, then `python -m core.coding_agreement` | `artifacts/coding_agreement.md` |
| Accuracy experiments | Would non-linear terms, boosting, a forest, neural nets or an ensemble discriminate better? | `python -m core.accuracy_experiments` (about 2 min, not in `run_all`) | `artifacts/accuracy_experiments.md` |
| Learning curve | Would more respondents answering the same questions improve discrimination? | `python -m core.learning_curve` (about 1 min, not in `run_all`) | `artifacts/learning_curve.md` |
| Error analysis | Where does the selected model go wrong, and for whom? (validation set; group summaries only) | `python -m core.error_analysis` (not in `run_all`) | `artifacts/error_analysis.md` |
| DeepSurv training curves | Did early stopping stop DeepSurv before it overfitted? | `python -m core.deepsurv_curves` (not in `run_all`) | `artifacts/deepsurv_curves.md` |
| Tuning results | Every hyperparameter setting and its validation score; do the chosen settings match the main analysis? | `python -m core.tuning_results` (about 2 min, not in `run_all`) | `artifacts/tuning_results.md` |
| Time-dependent metrics | Supplementary AUC at 3/6/12 months and the Brier score over time (validation set; not used for selection) | `python -m core.time_dependent_metrics` (not in `run_all`) | `artifacts/time_dependent_metrics.md` |
| Exploratory figures | Predictor correlations, skill-rating distributions and scree plots (training set) | `python -m core.eda_figures` (not in `run_all`) | `artifacts/eda_figures.md` |
| Evaluation figures | Kaplan-Meier with numbers at risk, AFT fit, Schoenfeld check, fairness and robustness charts (from existing results) | `python -m core.evaluation_figures` (not in `run_all`) | `artifacts/evaluation_figures.md` |
| LLM prompt variants | Does the LLM benchmark depend on the prompt? | `python -m benchmark.score` | `artifacts/llm_benchmark.md` |

## Sensitivity (`core/sensitivity.py`)

Variants: job on/before completion counted as 0.25 or 1.0 months (main: 0.5); excluding job-start times at
common "round" recall values (3, 6, 9, 12, 18, 24 months); excluding straight-liners and erratic responders
(Mahalanobis distance on the 26 skill items, p < 0.001); complete cases only; multiple imputation (m = 20,
chained equations with the outcome in the imputation model, Rubin's rules). Each variant reports Uno's C from
the same train -> validation evaluation and unpenalised Cox hazard ratios (training set) on the main analysis' scale, plus whether the direction of
each association stays the same.

## Temporal validation (`core/temporal.py`)

Train (with inner-CV tuning) on graduates who completed in or before the median completion month, test on those
who completed later. Reports Uno's C, IBS and calibration at 3/6/12 months where at least 20 later-cohort
graduates are still followed. A single split: no fold-to-fold spread.

## Institution clustering (`core/clustering.py`)

Cluster-robust (sandwich) standard errors by institution, and a Cox model stratified by institution. Name
variants are merged with the `institution` column of the university lookup (`data/reference/university_types*.csv`;
add it to the real lookup). Institutions are reported as "Institution 1..k", never by name. With fewer than 30
institutions the sandwich estimator is biased downwards, so a **conservative CI** (larger of the two standard
errors) is reported.

## Competing risks (`core/competing.py`)

Uses the new optional item **Q16a** (start month of a non-IT job, self-employment or further studies). Cleaning
writes `data/processed/graduates_competing.csv` (`cr_cause` 1 = IT role, 2 = another path, 0 = censored); the
main training file is unchanged. Reports Aalen-Johansen cumulative incidence vs 1 - Kaplan-Meier, and
cause-specific Cox models. A Fine-Gray model is not fitted (it would need a new dependency).

## LLM benchmark prompt variants (`benchmark/`)

`standard` (primary, unchanged), `no_country` (no mention of Sri Lanka), `local_context` (adds the 2022
University of Kelaniya tracer-study average of about 8 months - **check this sentence against the source before
use**) and `range` (best estimate plus an 80% range, scored for coverage). Collect a variant with
`python -m benchmark.run_llm --provider <p> --variant <v>`, or by hand using `prompts.csv` (filter by `variant`).
Each variant is compared with `standard` for the same LLM on the same profiles (paired bootstrap).

## Results on the real main-study data (11 Oct 2026)

885 modelling graduates (train 620, validation 133; the test set is not used by any of these analyses).

- **Sensitivity:** validation Uno's C of the selected AFT model ranges from 0.659 to 0.707 across the variants
  (main 0.693). The direction of association is unchanged in every variant for all 16 model columns. No
  straight-liners or erratic responders were detected, and no graduate had a missing model input, so those
  variants equal the main analysis. 118 job-start times are at "round" recall values.
- **Temporal validation:** trained on cohorts completing up to 2025-08 (322 graduates), scored on later cohorts
  (68 graduates, short follow-up): Uno's C 0.705-0.727 across models (AFT 0.721). All models under-predict
  employment in the later cohorts (by about 8-12 percentage points at 3 and 6 months).
- **Institution clustering:** 23 institutions (fewer than 30, so robust standard errors are only a rough guide).
  Cluster-robust intervals are mostly narrower than the usual ones; the conservative interval is reported. GPA
  (HR 1.24, 95% CI 1.10-1.40) and internship experience (HR 1.12, 1.03-1.23) remain clearly associated with
  finding work sooner, also when stratified by institution.
- **Competing risks:** 37 graduates started another path first (with a Q16a month). By 12 months, 77.8% had
  found an IT role first (cumulative incidence) against 81.9% from 1 - Kaplan-Meier, so the competing paths
  matter only a little. The "another path" model has 2.3 events per parameter and is exploratory only.
- **Single survey date:** every response uses the reference survey date 6 Oct 2026 (`docs/deviations.md`,
  section 3). This affects only the follow-up time of graduates still searching.

## Job-market study (`core/market_study.py`) - real data

On the real job ads (`data/raw/job_ads.csv`): skill prevalence with Wilson 95% intervals; demand by role family
(classified from the title by fixed, ordered rules); skill co-occurrence (lift and phi for pairs seen together in
>= 10 ads); interns/trainees vs junior/associate roles (Fisher's exact test per skill, Benjamini-Hochberg
correction across the 26 skills); technical vs soft-skill emphasis per ad; and named technologies
(`data/raw/job_ads_technologies.csv`, rules in `data/reference/job_ad_technology_rules.json`, matched on the full
ad text where available, else on the coded skills). Shown on the app's Job market page.

## Simulation study (`core/simulation_study.py`)

Synthetic by design (it needs effects known in advance), so its results are a method check, not a finding about graduates. 200 fresh synthetic surveys per sample size (150, 300, 600, 1200 graduates), each cleaned by the real pipeline
and fitted with an unpenalised Cox model on the model features. Reports power (p < 0.05 in the true direction)
per effect, the false-positive rate for the predictors with no true effect (university type), and bias and 95% CI
coverage for the predictors observed exactly and linearly (internship months, projects). Gamma frailty attenuates
Cox hazard ratios towards zero, so coverage below 95% is expected; self-rated skills are a noisy proxy, so their
effects are attenuated too (soft skills: only 18% power even at n = 1200). Shown on the Research lab page.

## Coding agreement (`core/coding_agreement.py`)

A blind sample of 40 keyword-coded ads (`data/raw/job_ads_coding_sample.csv`, fixed seed) is hand-coded by the
researcher from the source links; the script then reports per-skill agreement, Cohen's kappa, pooled kappa, and the
keyword rules' precision and recall. **Requires the hand coding to be done first.**

## Accuracy experiments (`core/accuracy_experiments.py`) - exploratory, report only

Same fixed split as the main analysis, each method fitted on train and tuned / scored on validation: Cox PH (core), Cox PH with GPA^2,
GPA >= 3.0, log counts and technical x degree interactions, a gradient-boosted Cox model and a random survival
forest (scikit-survival), four established neural survival models from pycox - DeepSurv (Katzman et al. 2018), DeepHit (Lee et al. 2018), Cox-Time (Kvamme et al. 2019) and N-MTLR (Fotso 2018), wrappers in `core/models/neural.py` - and a rank-average ensemble. Result on the real data (validation set, 133 graduates): Cox PH 0.668; Cox PH with non-linear and
interaction terms 0.666; gradient-boosted Cox 0.687 (+0.019, better in 85% of bootstrap resamples); random
survival forest 0.677; ensemble 0.683; neural networks 0.590 (DeepHit) to 0.674 (DeepSurv, Cox-Time). None of
them reaches the selected AFT Log-Normal model (0.693). The theoretical ceiling can only be computed on
synthetic data, where the true risk is known (about 0.72 there). A learning curve (`core/learning_curve.py`: AFT fitted on 25-100%
of the training set, 20 random subsets at each smaller size, scored on validation) rises from 0.641 to 0.693 and is
flattening, so more respondents answering
the same questions are expected to add only about 0.01-0.02. The main model's family list and selection rule were
fixed in advance and are not changed by this analysis.
