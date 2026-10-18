# Time-to-Employment Decision Support (Research Prototype)

Research prototype for the dissertation **"A Survival-Analysis Framework for Predicting and Explaining Time-to-Employment for Final-Year Information Technology Undergraduates and Graduates in Sri Lanka"**.

Given a graduate's (or final-year student's) profile, the app estimates how many months after graduating
graduates with similar profiles took to find their first IT job, explains which parts of the profile were
associated with shorter or longer searches, lets the user try "what-if" changes, and compares their skills
with employer requirements and job ads. Graduates who are still looking can enter how long they have been
searching; the estimate is then based on graduates who were also still looking at that point.

> **Decision support only - not a hiring or screening tool.** Profile and prediction data are not stored by the prediction API. The optional account/progress service uses a separate SQLite database; phone sign-in stays unavailable until SMS credentials are configured.
> All data are real: 324 job ads, and the main-study graduate (1,175 responses) and employer (245 responses)
> surveys (see the Data statement). The synthetic generator is kept only for the method checks and tests.

---

## 1. Setup on a clean Windows machine

You need **Python 3.11** and (optionally) git. All commands are for **PowerShell**, run from the project
folder (e.g. `D:\EmployabilityPredictor`).

**1. Install Python 3.11** (skip if `py -3.11 --version` already works):

```powershell
winget install --id Python.Python.3.11 -e --scope user
```

**2. Create and activate the virtual environment:**

```powershell
cd D:\EmployabilityPredictor
py -3.11 -m venv venv
venv\Scripts\activate
```

The prompt now starts with `(venv)`. If PowerShell says *"running scripts is disabled on this system"*, run
this once and try again: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**3. Install the requirements** (CPU-only PyTorch is pulled from the PyTorch index automatically):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the exact versions used to produce the thesis results, use `pip install -r requirements-lock.txt` instead.

**4. Create your `.env`:**

```powershell
copy .env.example .env
```

The defaults work as-is: the frontend talks to the API at `http://127.0.0.1:8000`.

**5. Import the survey responses and run the full pipeline:**

```powershell
python -m core.import_responses
python -m core.run_all
```

The first command converts the main-study workbooks in `New Data/Main data/` into `data/raw/graduate_responses.csv`
and `employer_responses.csv`. The second cleans the data, fits every model on the training set, compares them on
the validation set, selects and refits the final model, evaluates it once on the test set, fits 200 bootstrap
models, compares the predictor sets, runs the fairness check and the secondary analyses, builds the reference
database, runs the job-market and questionnaire studies, and scores the LLM benchmark if its responses match the
current data. About 5-8 minutes on a laptop. Every number is reproducible (fixed seeds).

`python -m core.run_all --synthetic` regenerates the synthetic demo files, but they are only used when no real file
is in `data/raw/`. Add `--quick` for a faster check - not for thesis numbers.

**6. Start the API** (leave this terminal open):

```powershell
uvicorn api.main:app --port 8000
```

Check it at http://127.0.0.1:8000/health; interactive API docs at http://127.0.0.1:8000/docs.

**7. Start the frontend** in a **second** PowerShell window:

```powershell
cd D:\EmployabilityPredictor
venv\Scripts\activate
streamlit run frontend/app.py
```

It opens at http://localhost:8501 with pages grouped in the sidebar: **Home** and **My profile** to enter answers; **My dashboard** and **Compare** for results; **Career path**, **Saved profile & progress**, and **Improve my chances** under My plan; plus **Job market**. Career path compares your skill ratings with skills named in the collected entry-level job ads; it is descriptive guidance, not a hiring prediction. Page code is in `frontend/views/`, shared styling in `frontend/ui.py`.

On the **My prediction** dashboard, choose **Generate PDF report** and then **Download PDF report**. The report includes the estimate, uncertainty chart, explanation, suggested next steps and employer skill gaps. It is generated on request and returned directly; the API does not save the profile or PDF. The frontend keeps the generated PDF in temporary session memory until the page session ends.

### Optional profile and progress storage

The profile/progress service is an independent app (`profile_service.main`) on port 8001 with a separate SQLite file (`data/profile_store.sqlite3`). It does not change or connect internally to the prediction API. After phone verification, users can save a display name and choose an avatar from the built-in emoji set; these account settings follow the account across sign-ins. Each profile snapshot saves the original answers, calculated profile, prediction, explanation, personalized suggestions, employer skill alignment, recommended roles/gaps, selected career path, and any active Improve my chances what-if comparison. Career goals and action statuses are stored separately; records are retained for up to one year and can be deleted by the user. Start it in its own terminal:

```powershell
venv\Scripts\activate
uvicorn profile_service.main:app --host 127.0.0.1 --port 8001
```

Copy `.env.example` to the ignored `.env`, then set `SMS_API_KEY` and `SMS_SENDER_ID`. The service uses `SMS_API_ENDPOINT` (defaults to `https://dashboard.smsapi.lk/api/v3/sms/send`) and sends a plain-text message with `recipient`, `sender_id`, `type`, and `message` fields. `SMS_API_AUTH_SCHEME` defaults to `Bearer`; change it only if your SMSAPI.lk account requires a different authorization scheme. The endpoint and request fields follow public third-party examples, not verified provider documentation; no real SMS delivery has been tested. Confirm the request format with SMSAPI.lk and test using your own number. Never put an API token in source control or chat.

OTP codes expire after five minutes, allow at most five verification attempts, and are limited to three requests per phone number per 15 minutes. The SQLite database is free and requires no separate DB account; it persists on the machine where the profile service runs. It is suitable for a local installation. For a public deployment or access across devices, the service and database need to be hosted with a persistent disk or managed database, HTTPS and backups. A free hosted database is not required for local use, and free-tier availability/retention varies by provider.

After sign-in, the profile-service session token is kept in this browser tab's `sessionStorage` so full-page reloads (including theme changes) can restore the session. It is cleared when you sign out or delete the account, is not shared as a persistent login across separate tabs, and remains subject to the profile service's 30-day token expiry.

**8. Run the tests** (optional):

```powershell
pytest
```

412 tests, about 2 minutes. See `docs/testing.md` for the
requirement-to-test map.

---

## 2. Swapping in real data

**No code changes are needed.** Any CSV in `data/raw/` **without** `_SYNTHETIC` in its name automatically
takes priority over the synthetic one. The file type (graduate / employer / job ads) is detected from its
columns, so the file name does not matter.

### Before collecting data
1. Use the **v2 questionnaires** - see `docs/questionnaires/questionnaire_revisions.md` (adds month/year of
   degree completion and of the first IT job, exact counts, and "skills at degree completion" wording).
   Check with your supervisor whether this needs an ethics amendment.
2. In Google Forms, **start every question title with its number** (`Q5a. ...`, `Q17a. ...`, `Q18. ...`).
   Columns are matched by these numbers. Month/year can be a Date question or two dropdowns titled
   `[Month]` / `[Year]` in a grid.
3. Keep the automatic **Timestamp** column in the export (it is the survey date).

### Steps
1. Download each form's responses as CSV (Google Forms -> Responses -> Sheets -> File -> Download -> CSV).
2. Put them in `data/raw/`, e.g. `data/raw/graduate_responses.csv` and `data/raw/employer_responses.csv`.
3. Put your manually collected job ads in `data/raw/job_ads.csv` with columns
   `ad_id, title, company, date, source, skills` - `skills` separated by `;`. Natural names are fine
   ("React; AWS; SQL; teamwork") - they are mapped to the skill vocabulary.
4. **Activate the real institution lookup:** copy `data/reference/real/university_types.csv` to
   `data/reference/university_types.csv`. It already maps every institution in the app's dropdown, plus
   common short forms (UoM, UCSC, USJ, SLIIT, NSBM, IIT, NIBM...), to `state`, `non_state` or
   `foreign_affiliated`. It is kept in the `real/` sub-folder until then because the pipeline uses any real
   lookup in `data/reference/` automatically, and the synthetic survey uses fictional institution names.
   Add any name respondents typed that is missing (the cleaning log lists unmapped names).
5. Run:
   ```powershell
   python -m core.run_all
   ```
6. Open `docs/data_cleaning_log.md` and check:
   - **Sources** table says `synthetic: no` for every file.
   - **Notes** - unmapped university names (add them to step 4's file and re-run), export columns ignored.
   - Exclusions, events, censored count and the **EPV** warning.
7. Restart the API and the frontend so they load the new model. The synthetic-data notices disappear only
   when **all** inputs (graduates, employers, job ads, university types) are real.
8. **LLM benchmark:** the old prompts belong to the synthetic profiles. Re-export and collect new answers:
   ```powershell
   python -m benchmark.export_prompts --force-template
   ```
   (`run_all` refuses to score responses collected for a different dataset.)

---

## 3. Data statement

| Data | Status | Files | Details |
|---|---|---|---|
| Job advertisements | **Real** | `data/raw/job_ads.csv` (324 ads), `job_ads_provenance.csv`, `job_ads_technologies.csv` | Entry-level IT ads (intern, trainee, junior, associate; < 2 years' experience) from 211 employers, posted 12 Jul - 11 Oct 2026: 42 from topjobs.lk and 4 from xpress.jobs coded by hand from the flyers; 262 from ITPro.lk, 8 from jobhub.lk and 8 from LinkedIn coded with the fixed keyword rules in `data/reference/job_ad_coding_rules.json`. Every ad has its source link; 173 checked ads were excluded with reasons (not IT, 2+ years' experience, senior, reposts already collected, no requirements or no ad text). |
| Graduate survey | **Real** | `New Data/Main data/Graduate ... (Responses).xlsx` -> `data/raw/graduate_responses.csv` | Main study, 1,175 responses collected through Google Forms; converted by `python -m core.import_responses`. Pilot and technical-test responses (`New Data/Pilot Test/`, `New Data/Technical test/`) are not used. Every response is given one reference survey date, 6 Oct 2026 (`docs/deviations.md`, section 3). Cleaning: `docs/data_cleaning_log.md` (885 modelling graduates, 230 final-year students set aside). |
| Employer survey | **Real** | `New Data/Main data/Employer ... (Responses).xlsx` -> `data/raw/employer_responses.csv` | Main study, 245 responses (239 after cleaning), same collection method and conversion. |
| Institution lookup | **Real** | `data/reference/university_types.csv` (copy in `data/reference/real/`) | Every institution in the dropdown plus the names typed under "Other (not listed)". |

Job ads were collected only from public pages, at the request rates the sites specify, and only advertisement
content was recorded. Before submission, keep PDF or screenshot copies of the ads, since the links expire.

---

## 4. Additional analyses

Each writes a report to `artifacts/` and is shown on the app's **Job market** or **Research lab** page.
Details: `docs/secondary_analyses.md`.

| Command | What it answers | Report |
|---|---|---|
| `python -m core.market_study` (also in `run_all`) | Skill demand in the real ads, role profiles, skill bundles, interns vs junior roles, named technologies | `job_market_study.md` |
| `python -m core.survey_study` (also in `run_all`) | Every questionnaire item that is not a model input: repeated questions tested against the model question they repeat, graduates vs employers, self-perceived employability vs outcome, artifact requirements, descriptive results for every item | `questionnaire_study.md` |
| `python -m core.sus_study` (also in `run_all`) | Usability (RQ4): System Usability Scale scores from the form export in `New Data/Usability (SUS)/`, mean with 95% CI against the benchmark of 68, t and Wilcoxon tests, Cronbach's alpha, approximate percentile, item means and coded comments | `sus_study.md` |
| `python -m core.simulation_study --reps 200` (about 5 min) | Does the analysis recover known effects? Power by sample size, false positives, coverage | `simulation_study.md` |
| `python -m core.accuracy_experiments` (about 3 min) | Do flexible models (neural networks DeepSurv/DeepHit/Cox-Time/N-MTLR, boosting, forest, ensemble) beat Cox? Accuracy ceiling | `accuracy_experiments.md` |
| `python -m core.learning_curve` (about 1 min) | Would more respondents improve discrimination? Selected model fitted on 25-100% of the training set, scored on validation | `learning_curve.md` |
| `python -m core.error_analysis` | Where does the selected model go wrong? Predicted vs actual time on the validation set, by search length, and the worst-predicted tenth vs the rest | `error_analysis.md` |
| `python -m core.deepsurv_curves` | How DeepSurv trained: training and early-stopping hold-out loss by epoch | `deepsurv_curves.md`, `figures/deepsurv_loss.png` |
| `python -m core.tuning_results` | Every tuning configuration with its validation Uno's C; checks the chosen settings match the main analysis | `tuning_results.md` |
| `python -m core.time_dependent_metrics` | Supplementary: time-dependent AUC at 3/6/12 months for every candidate, and the Brier score over time vs Kaplan-Meier (validation set) | `time_dependent_metrics.md`, `figures/brier_over_time.png` |
| `python -m core.eda_figures` | Exploratory figures (training set): predictor correlation heatmap, distribution of the skill ratings, scree plots with parallel analysis | `eda_figures.md`, `figures/eda_*.png` |
| `python -m core.construct_validity` | Appendix E tables: sample characteristics by split, univariate Cox and VIF, KMO / alpha / omega / AVE / HTMT of the skill blocks, item loadings | `construct_validity.md` |
| `python -m core.baselines` | How much do the inputs add? Selected model vs GPA only, GPA + internship and the core inputs (validation, paired bootstrap) | `baselines.md` |
| `python -m core.sample_comparison` | How representative is the sample? Compared with published Sri Lankan figures (Kelaniya 2022, ICTA 2019) | `sample_comparison.md` |
| `python -m core.evaluation_figures` | Evaluation figures: Kaplan-Meier with numbers at risk, AFT distribution fit, labelled Schoenfeld plot, fairness and robustness charts (drawn from the existing reports; the test set is not re-evaluated) | `evaluation_figures.md`, `figures/{km_at_risk,aft_fit,schoenfeld_labelled,fairness,robustness}.png` |
| `python -m core.coding_agreement --sample`, hand-code, then `python -m core.coding_agreement` | Agreement (Cohen's kappa) between keyword coding and manual coding of 40 ads | `coding_agreement.md` |

The exploratory analyses (simulation, accuracy experiments) are report-only: they never change the selected
model, whose candidate list and selection rule were fixed in advance.

---

## 5. LLM benchmark

1. `python -m benchmark.export_prompts` writes `benchmark/prompts.csv` (60 held-out profiles by default;
   `--n 229` for all) and a pre-filled `benchmark/responses.csv`.
2. Collect answers - **either** way (both write the same `responses.csv` format):
   - **Manual:** paste each prompt into the LLM - **a new chat for every prompt** - and record `llm_name`,
     `model_version`, `date` and the answer. Repeat for each run (default 3).
   - **API mode:** put the key in `.env`, install that provider's SDK, then run, e.g.
     ```powershell
     pip install openai                  # or: pip install -r requirements-benchmark.txt
     python -m benchmark.run_llm --provider openai --model <model-id> --dry-run   # shows how many calls
     python -m benchmark.run_llm --provider openai --model <model-id>
     ```
     Providers: `openai` (ChatGPT; `OPENAI_API_KEY`), `anthropic` (Claude; `ANTHROPIC_API_KEY`, default
     model `claude-opus-5-5`), `gemini` (`GEMINI_API_KEY`; called over REST, no extra package). Every prompt
     is an independent request; the model version that actually answered is recorded. Safe to stop and
     re-run - answered rows are skipped. Running a second provider adds its own rows, so you can compare
     several LLMs. 60 profiles x 3 runs = 180 short requests per provider.
3. `python -m benchmark.score` (or `run_all`) writes `artifacts/llm_benchmark.md`: Uno's C, 6-month
   calibration, consistency across runs, and a paired comparison with the selected model.
4. Optional prompt variants (`prompts.csv` has a `variant` column): `standard` (primary), `no_country`,
   `local_context` and `range` (best estimate + 80% range). Collect one with `--variant <name>` in API mode,
   or add `--variants standard range` to `export_prompts` for a manual template. Each variant is compared
   with `standard` on the same profiles; range answers are scored for coverage.

---

## 6. Project layout

| Path | What it is |
|---|---|
| `core/schema.py` | Single source of truth: fields, item numbers, skill vocabulary & areas, model and actionable features |
| `core/synthetic.py` | Synthetic data generator with known effects |
| `core/forms_adapter.py` | Converts a raw Google Forms export to the schema format |
| `core/clean.py` | Cleaning, outcome construction (duration / event), cleaning log |
| `core/preprocess.py` | Transformers fitted inside CV folds only |
| `core/models/` | Cox PH, Elastic Net Cox, AFT (Weibull / log-normal / log-logistic), DeepSurv; `neural.py`: DeepHit, Cox-Time, N-MTLR (exploratory) |
| `core/splits.py` | Fixed train / validation / test split - every graduate has one role (no data reuse) |
| `core/evaluate.py` | Fit on train, compare on validation, selection, final fit, one test-set evaluation, bootstrap |
| `core/feature_sets.py` | Core vs extended predictor sets on the same folds, with a pre-registered adoption rule |
| `core/predict.py` / `explain.py` / `whatif.py` | Prediction ranges, milestone chances, explanations, what-if |
| `core/suggest.py` | Suggested combinations to explore (small changes to actionable features) |
| `data/reference/learning_resources.csv` | Editable list of free learning resources per skill (loaded into reference.db) |
| `core/market.py` / `peers.py` | Employer + job-ad alignment, Kaplan-Meier peer benchmarks, `reference.db` |
| `core/fairness.py` | Subgroup check (backend only) |
| `core/sensitivity.py` / `temporal.py` / `clustering.py` / `competing.py` | Secondary analyses (see `docs/secondary_analyses.md`) |
| `core/market_study.py` / `simulation_study.py` / `accuracy_experiments.py` / `coding_agreement.py` | Job-market study, simulation study, accuracy experiments, coding agreement |
| `core/survey_items.py` / `survey_study.py` / `synthetic_survey.py` | Questionnaire-study items (not model inputs), their analysis, and their synthetic answers |
| `core/sus_study.py` | Usability evaluation: SUS scoring and analysis (form built by `docs/usability/SUS_Form_Builder.gs`) |
| `core/wording.py` | **Every user-facing sentence** - edit wording here |
| `core/chat.py` | Grounded assistant behind `/chat`: answers general questions without a profile and personal-result questions once a profile is complete (no language model, nothing stored) |
| `core/run_all.py` | The one-command pipeline |
| `benchmark/` | LLM prompt export, optional API mode (`run_llm.py`), manual Excel answer sheet for free chat apps (`answer_sheet.py`), scoring |
| `api/` | FastAPI app (thin layer over `core/`), incl. `/chat` (grounded assistant) and the `/job-market` and `/research` study endpoints |
| `frontend/` | Streamlit app: `app.py` (navigation), `views/` (profile, dashboard, career path, Job market, Research lab, and guidance pages), `ui.py` (shared style), `charts.py` / `charts_study.py` (figures) |
| `data/reference/real/` | Real institution lookup, staged until real survey data is used |
| `docs/` | Data dictionary, cleaning log, architecture, testing, questionnaires |
| `artifacts/` | Generated: models, results tables (`model_comparison.md` = Chapter 4), figures |

See `docs/architecture.md` for the component diagram.

## 7. Key outputs for the dissertation

| File | Content |
|---|---|
| `docs/data_dictionary.md` | Every field, type, valid range, item number and role |
| `docs/data_cleaning_log.md` | Raw N -> exclusions -> final N, events, EPV, Cronbach's alpha |
| `artifacts/model_comparison.md` | Validation results for all models, selection rule, locked test-set result, PH check, results in plain terms |
| `artifacts/feature_set_comparison.md` | Pre-registered tests of two alternatives to the core predictors: 6 extra questionnaire items, and log-transformed counts plus a GPA >= 3.0 term |
| `artifacts/figures/` | Kaplan-Meier, model comparison, calibration plots, Schoenfeld residuals |
| `artifacts/fairness_report.md` | Performance by gender and university type |
| `artifacts/sensitivity.md` | Results under alternative cleaning decisions, complete-case and multiple imputation |
| `artifacts/temporal_validation.md` | Trained on earlier cohorts, tested on later ones |
| `artifacts/institution_clustering.md` | Cluster-robust and institution-stratified Cox |
| `artifacts/competing_risks.md` | IT role vs another path: cumulative incidence and cause-specific Cox |
| `artifacts/llm_benchmark.md` | LLM vs survival model |
| `artifacts/job_market_study.md` | **Real data:** skill demand (with CIs), role profiles, skill bundles, interns vs junior roles, technologies |
| `artifacts/questionnaire_study.md` | Repeated-question consistency checks, graduates vs employers, self-perception vs outcome, artifact requirements, every item |
| `artifacts/sus_study.md` | Usability evaluation: SUS mean, 95% CI, tests against 68, item means, usability issues (figure `figures/sus_scores.png`) |
| `artifacts/simulation_study.md` | Power, false positives and coverage by sample size |
| `artifacts/accuracy_experiments.md` | Neural, boosting, forest and ensemble models vs Cox, and the accuracy ceiling |
| `docs/synthetic_data.md` | The synthetic data-generating process and its true effects |

### Walkthrough notebook

[`notebooks/research_walkthrough.ipynb`](notebooks/research_walkthrough.ipynb) walks through the whole analysis in
14 commented sections - data and cleaning, exploratory analysis (correlations, skill ratings, reliability, scree
plots), Kaplan-Meier curves, the split, training and tuning, model comparison and selection, model checks, diagnostics,
the test result, fairness, robustness, skill gaps, usability and the LLM benchmark - with every table and chart saved
in the notebook, so it can be **read on GitHub without running anything**. It calls the code in `core/` (no copied
logic), trains and tunes live on the training and validation sets, and **loads** the locked test-set result instead of
recomputing it. Its last section checks that its numbers match the reported ones. Only summaries are shown, never
individual respondents.

Each main output is followed by a short "What this shows" note, the tables use readable names, and the overview at the
top links to every section.

**No access to the survey data?** Set `USE_SYNTHETIC = True` in the first code cell. The notebook then copies the code
to a temporary folder, generates synthetic survey data with `core/synthetic.py` and runs the whole pipeline there
(about 3 extra minutes); the real `data/` and `artifacts/` folders are never touched. The SUS, LLM-benchmark and
coding-agreement sections are skipped, because they need real human responses.

To re-run it on the real data you need the survey data (not on GitHub, for privacy) and the notebook tools:

```powershell
pip install -r requirements-notebook.txt
python notebooks/build_walkthrough.py          # rebuilds the notebook from its source
jupyter nbconvert --to notebook --execute --inplace notebooks/research_walkthrough.ipynb --ExecutePreprocessor.timeout=1800
```

### Experiment tracking (reproducibility log)

```powershell
python -m core.tracking --track local     # writes artifacts/runs/<timestamp>/ (no account needed)
python -m core.tracking --track wandb     # Weights & Biases, offline by default (pip install -r requirements-tracking.txt)
python -m core.run_all --track local      # run the pipeline, then log it
```

A run records the data fingerprint, the split sizes, each candidate model's settings and validation metrics, the
selection, DeepSurv's loss per epoch, the learning curve and the figures. It reads the saved artifacts only: nothing is
refitted, the locked test result is loaded (never re-evaluated), and no survey answers are logged. Runs are dated when
they are logged, so they are a reproducibility log rather than records of the original development. Tracking is off by
default.

## 8. Privacy

Respondent-level files (`data/raw/`, `data/processed/`, `reference.db`, out-of-fold predictions, the
benchmark profile key) are git-ignored. The real job-ad files in `data/raw/` contain no personal data; add
them to version control explicitly if you want them kept with the code. The API never writes or logs submitted profiles, and the frontend
does not cache them.

## Development process

How the project was developed (with dated evidence) is in `docs/development_log.md`; the branch, commit and pull-request workflow is in `CONTRIBUTING.md`.
