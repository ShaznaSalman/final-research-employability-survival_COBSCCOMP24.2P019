# Development log

An honest record of how this project was developed. Dates come from the metadata of the dated project documents
(title clearance, proposal, ethics form, questionnaires, chapters) and from the development-session records.
Where a date is not recorded it is marked **[date to confirm]** - fill these in from your own notes or supervisor
meeting records; do not estimate them.

> **Version control:** software development started on **4 August 2026** and was done locally; the project was
> placed under git in October 2026. The history for the earlier work (the commits dated 4 August - 27 September
> 2026, on 13 feature branches) was created afterwards to organise the existing code into reviewable units. Their
> dates were assigned to show the order and period of development; they are **not** the actual times the commits
> were made, and some of those commits contain files as they stood later (for example, the results on the real
> survey data, produced on 11 October 2026). Use the dated documents in the timeline below, not commit dates, as
> evidence of when work was done. All later work follows the branch and pull-request workflow in `CONTRIBUTING.md`.

## Timeline

| Date | Phase | What happened | Evidence |
|---|---|---|---|
| 14 Mar 2026 | Ethics | Ethical clearance form started | `5.1.Ethical Clearance/Submitted-Ethical_Clearance_Form.docx` (created) |
| 10 Apr 2026 | Title | Revised title clearance submitted | `1.Title clearance/TitleClearance_Revised_COBSCCOMP242P-019.docx` |
| 18-19 Apr 2026 | Proposal | Research proposal finalised (Project Discovery) | `2.Project Proposal(Discovery)/...Research_Proposal_COBSCCOMP24.2P019` |
| 1 May 2026 | Ethics | Ethical clearance form submitted (last edit) | `Submitted-Ethical_Clearance_Form.docx` (modified) |
| 5 Jun 2026 | Instruments | Original graduate and employer questionnaires | `5.Questionnare/3.Original Graduate_Questionnaire.docx`, `6.Original Employer_Questionnaire.docx` |
| 7 Aug 2026 | Instruments | Revised questionnaires and Evidence & Justification Matrices | `1.Revised-Graduate...docx`, `4.Revised-Employer...docx`, `2.` and `5.` matrices |
| 31 Aug 2026 | Planning | Research summary / quick reference | `3.Final Research - Dissertation/2. Important Points/Research_Summary_QuickReference.docx` |
| From 4 Aug 2026 | Software | Development started; core pipeline built: schema, Google Forms adapter, cleaning and outcome construction, survival models (Cox PH, Elastic Net Cox, AFT, DeepSurv), nested cross-validation, prediction with bootstrap uncertainty, explanations, what-if, FastAPI service, Streamlit app, LLM benchmark | Code in `core/`, `api/`, `frontend/`, `benchmark/` |
| 8 Oct 2026 | Dissertation | Chapters 1-3 revised for the survival-analysis approach | `1. Chapters - Revised (survival analysis)/Chapter 1-3 (revised).docx` |
| 9 Oct 2026 | Data | Synthetic development data enlarged (390 graduate rows, 280 employers) and made more realistic; fictional institution and company names; labelled synthetic throughout | `core/synthetic.py`, `docs/synthetic_data.md` |
| 9 Oct 2026 | App | Institution dropdown (real Sri Lankan institutions grouped by type) | `core/wording.py`, `frontend/views/predict.py` |
| 9 Oct 2026 | Data | 310 real entry-level IT job ads collected (topjobs.lk, ITPro.lk, LinkedIn, xpress.jobs) and coded | `data/raw/job_ads.csv`, `job_ads_provenance.csv` (not committed) |
| 11 Oct 2026 | Data | Job-ad top-up: 14 more real ads (ITPro.lk, jobhub.lk, xpress.jobs) with the same rules -> 324; 110 more checked ads excluded with reasons. A public Kaggle 'Sri Lanka Pulse 2026' job dataset was examined and rejected (templated text, all rows 'scraped' within 0.13 s, incoherent skills: appears synthetic) | `data/raw/job_ads*.csv` (not committed) |
| 10 Oct 2026 | App | Dashboard redesigned (Home, My prediction, Job market, Research lab) | `frontend/` |
| 10 Oct 2026 | Analysis | Job-market study, simulation study, coding-agreement check, accuracy experiments; neural survival models (DeepHit, Cox-Time, N-MTLR) added | `core/market_study.py`, `simulation_study.py`, `coding_agreement.py`, `accuracy_experiments.py`, `models/neural.py` |
| 10 Oct 2026 | Dissertation | Full Chapters 1-6 draft; references restricted to 2022+ and formatted in APA 7; reference register | `4. Final Chapters (draft)/` |
| 10 Oct 2026 | Dissertation | Chapters 1-3 revised against supervisor feedback (citations, measurable objectives, limitations, analysis techniques linked to objectives, pilot and validation, leakage, metric and algorithm justification) | `Chapters_1-3_revised.docx` |
| 10 Oct 2026 | Instruments | Questionnaires v2 and Evidence Matrix v2 (2022+ evidence, graded A/B/C) | `7.Graduate..._v2.docx`, `8.Employer..._v2.docx`, `Questionnaire_Evidence_Matrix_v2.docx` |
| 10 Oct 2026 | Analysis | Every questionnaire item integrated: questionnaire study (repeated-question checks, graduates vs employers), personalised survey insights in the app | `core/survey_items.py`, `survey_study.py`, `insights.py` |
| 10 Oct 2026 | Model | Extended feature set (Q14, Q20, Q21, Q23, Q24, Q29) adopted as model inputs - a documented deviation from the pre-registered rule | `docs/deviations.md` |
| 11 Oct 2026 | Data | Main-study survey data imported: 1,175 graduate and 245 employer responses, collected through Google Forms (the graduate file's Timestamp column was accidentally changed during file handling, so it is not used as a date). Pilot and technical-test responses kept out. Typed "Other" degree programmes and institutions now mapped | `New Data/Main data/`, `core/import_responses.py`, `core/forms_adapter.py`, `data/reference/university_types.csv` |
| 11 Oct 2026 | Model | Full pipeline re-run on the real data: 885 modelling graduates (train 620 / validation 133 / test 132). AFT Log-Normal selected on validation (Uno's C 0.693); test set used once: Uno's C 0.619 (95% CI 0.557-0.683). Synthetic-run outputs backed up | `artifacts/`, `data/backup_synthetic_run_2026-10-11/` |
| 11 Oct 2026 | Data | One reference survey date (6 Oct 2026, end of collection) used for every response - a documented deviation | `docs/deviations.md` |
| 13 Oct 2026 | App | Grounded chat assistant added (`core/chat.py`, `/chat`, "Ask about your result" on the My prediction page): answers from the system's own results, no language model, nothing stored; 25 tests added (241 in total) | `core/chat.py`, `tests/test_chat.py` |
| 14 Oct 2026 | Data | Import fix: the answer text "None" (Q26, Q28, employer Q6) and the comment "n/a" were read as blank cells by pandas; they are now kept (`schema.SURVEY_NA_VALUES`). Re-ran import, cleaning and the questionnaire study only: model inputs, split and exclusions byte-for-byte identical, so the model and test result are unchanged; descriptive shares for Q26, Q28 and Q6 moved by up to 3 points | `core/import_responses.py`, `core/clean.py` |
| 10 Oct 2026 | Process | Git initialised; history organised into feature branches; contribution workflow defined | `CONTRIBUTING.md` |
| 14 Oct 2026 | Evaluation | Usability evaluation: 12 participants, System Usability Scale scored and analysed (mean 76.25, 95% CI 69.53-82.97, criterion of 68 met) | `core/sus_study.py`, `artifacts/sus_study.md` |
| 15 Oct 2026 | Evaluation | Report-only diagnostics on the training and validation sets: learning curve, error analysis, DeepSurv loss curves, every tuning configuration, time-dependent AUC and Brier score, exploratory and evaluation figures (thesis Appendices E, J and K). Complete-case sensitivity check now includes the extended-set inputs (15 of 16 directions unchanged). No headline result changed; the test set was not used again | `core/learning_curve.py`, `error_analysis.py`, `deepsurv_curves.py`, `tuning_results.py`, `time_dependent_metrics.py`, `eda_figures.py`, `evaluation_figures.py`, `sensitivity.py` |

## Decisions and why

| Decision | Reason | Where documented |
|---|---|---|
| Survival analysis instead of classification | Graduates still searching are censored; classification would discard or misclassify them | Chapter 2-3 |
| No ensemble / forest models in the main comparison | Supervisor requirement; explanations harder to present | Chapter 3 |
| Neural survival models added | Supervisor asked for established neural networks | `core/models/neural.py` |
| Accuracy not artificially increased | On the real data every method reaches validation C of about 0.59-0.69 and the learning curve is flattening; the limit is unmeasured factors (contacts, search effort, interviews). Tuning towards a target number would misreport the model | `artifacts/accuracy_experiments.md` |
| Synthetic data kept clearly labelled | Used to build and test the pipeline before real responses existed; since 11 Oct 2026 every reported result uses the real data (the simulation study and tests stay synthetic by design) | `docs/synthetic_data.md` |
| Single reference survey date | The graduate file's Timestamp column was accidentally changed during file handling, so it no longer holds the submission times; no per-person dates were estimated | `docs/deviations.md` |
| Skills and preparation asked "at degree completion" | Prevents reverse causation / leakage | `docs/questionnaires/questionnaire_revisions.md` |
| Extended feature set | Researcher's choice; deviates from the pre-registered rule | `docs/deviations.md` |

## Use of AI tools

Parts of the code, documentation and dissertation drafting were produced with the assistance of an AI coding
assistant (Claude Code). Declare this according to
the university's policy on AI use.

## Status (15 Oct 2026)

Done: real survey import and results on the real data (11 Oct 2026); LLM benchmark with ChatGPT GPT-5.6 and the
usability study with 12 participants (14 Oct 2026); Chapters 4-6 and the appendices; report-only diagnostics
(15 Oct 2026). The planned branches in `CONTRIBUTING.md` and what became of each are listed there.

Not done: fixes to the artifact from the usability comments - they are reported as recommendations (thesis
Section 6.5) instead, because changing the artifact after the study would mean the SUS score describes a different
version.
