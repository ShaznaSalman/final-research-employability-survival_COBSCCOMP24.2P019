# Deviations from the pre-registered analysis plan

## 1. Evaluation design: fixed train / validation / test split instead of nested cross-validation

**Plan:** nested cross-validation (5 outer folds x 3 repeats, 3 inner folds).

**Change:** the supervisor's instruction is that no data may be reused across roles. Cross-validation rotates every
graduate between training and testing, so it was replaced by a single fixed split, made once before any modelling
(`core/splits.py`, stratified by event and institution type): **train 70%** (every model is fitted here only),
**validation 15%** (tuning, model comparison, feature-set comparison, secondary model comparisons) and **test 15%**
(locked; the selected model is evaluated on it once). The final model used by the app is fitted on the training set
only. Intervals come from bootstrapping the evaluation set, which refits no model.

**Consequences to report:** fewer graduates train the model (620 of 885 on the real data; EPV on the training set is
29.2 with the extended feature set, so this is not a problem) and the validation and test estimates are noisier
(133 and 132 graduates; Uno's C intervals about +/- 0.06). This is the cost of the
strict no-reuse rule; recruiting more respondents is the remedy.

## 2. Feature set: "extended" used instead of the set recommended by the pre-registered rule

**Pre-registered rule** (`core/feature_sets.py`, report `artifacts/feature_set_comparison.md`): a challenger set
replaces *core* only if, for Cox PH, validation Uno's C improves by at least 0.01 **and** it is better in at least 70%
of bootstrap resamples of the validation set, with EPV >= 10 (or Elastic Net Cox meeting the same criteria).
(Before change 1 the rule used "70% of outer folds"; the threshold is unchanged.)

**Result on the real main-study data (885 graduates; train 620, validation 133):** the rule recommends **transformed**
(log counts and a GPA >= 3.0 indicator; Cox PH validation Uno's C +0.011 over core, better in 91% of resamples).
Extended: +0.003, better in 66% of resamples - it does not meet the rule. (On the earlier synthetic data the
recommendation was also transformed, +0.045.)

**Decision:** the researcher uses **extended**, so that the preparation questions - portfolio (Q14), when
preparation started (Q20), self-learning (Q21), hackathons (Q23), open source (Q24) and SDLC familiarity (Q29) -
feed the prediction in the app.

**Consequences to report:** six more inputs for the user; lower events per variable; no evidence of better
discrimination than core on the data analysed. The comparison is re-run automatically by `python -m core.run_all`;
if real data shows the extended set does not help, this decision should be revisited.

## 3. Survey date: one reference date for every respondent

**Plan:** the survey date of each graduate is the Google Forms submission Timestamp. For graduates still searching,
follow-up time = survey month - completion month.

**Change:** the main-study responses were collected through Google Forms. `core/import_responses.py` gives every
response (graduate and employer) the same reference survey date, the end of data collection: **6 October 2026**.
The workbooks are not changed by the import. No per-respondent dates were estimated or invented.

**Consequences to report:** for censored graduates who really answered earlier than 6 October 2026, the follow-up
time is overstated by up to about two months (they were known to be still searching only at the time they
answered). This tends to make censored searches look slightly longer. Checks that compare dates with the survey
date ("completion after survey", "job start after survey", "no follow-up time") are made against the reference date.
The single reference date should be stated in the methodology chapter.
