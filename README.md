# Time-to-Employment Decision Support (Research Prototype)

Research prototype for the dissertation **"A Survival-Analysis Framework for Predicting and Explaining
Time-to-Employment for Final-Year IT Undergraduates in Sri Lanka"**.

Given a graduate's (or final-year student's) profile, the app estimates how many months after graduating
graduates with similar profiles took to find their first IT job, explains which parts of the profile were
associated with shorter or longer searches, lets the user try "what-if" changes, and compares their skills
with employer requirements and job ads. Graduates who are still looking can enter how long they have been
searching; the estimate is then based on graduates who were also still looking at that point.

> **Decision support only - not a hiring or screening tool.** No accounts, nothing stored.
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

It opens at http://localhost:8501 with four pages (top navigation): **Home**, **My prediction** (the predictor), **Job market** (the study of the real job ads) and **Research lab** (model comparison, calibration, effects, fairness, robustness and the simulation study). Page code is in `frontend/views/`, shared styling in `frontend/ui.py`.

**8. Run the tests** (optional):

```powershell
pytest
```

216 tests, about 2 minutes. See `docs/testing.md` for the
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
| Graduate survey | **Real** | `New Data/Main data/Graduate ... (Responses).xlsx` -> `data/raw/graduate_responses.csv` | Main study, 1,175 responses collected on a shared Excel sheet; converted by `python -m core.import_responses`. Pilot and technical-test responses (`New Data/Pilot Test/`, `New Data/Technical test/`) are not used. The Timestamp column was added after collection, so every response is given one reference survey date, 6 Oct 2026 (`docs/deviations.md`, section 3). Cleaning: `docs/data_cleaning_log.md` (885 modelling graduates, 230 final-year students set aside). |
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
| `python -m core.simulation_study --reps 200` (about 5 min) | Does the analysis recover known effects? Power by sample size, false positives, coverage | `simulation_study.md` |
| `python -m core.accuracy_experiments` (about 3 min) | Do flexible models (neural networks DeepSurv/DeepHit/Cox-Time/N-MTLR, boosting, forest, ensemble) beat Cox? Accuracy ceiling | `accuracy_experiments.md` |
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
| `core/wording.py` | **Every user-facing sentence** - edit wording here |
| `core/chat.py` | Grounded assistant behind `/chat`: answers questions about a profile from the system's own results (no language model, nothing stored) |
| `core/run_all.py` | The one-command pipeline |
| `benchmark/` | LLM prompt export, optional API mode (`run_llm.py`), scoring |
| `api/` | FastAPI app (thin layer over `core/`), incl. `/job-market` and `/research` study endpoints |
| `frontend/` | Streamlit app: `app.py` (navigation), `views/` (Home, My prediction, Job market, Research lab), `ui.py` (shared style), `charts.py` / `charts_study.py` (figures) |
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
| `artifacts/simulation_study.md` | Power, false positives and coverage by sample size |
| `artifacts/accuracy_experiments.md` | Neural, boosting, forest and ensemble models vs Cox, and the accuracy ceiling |
| `docs/synthetic_data.md` | The synthetic data-generating process and its true effects |

## 8. Privacy

Respondent-level files (`data/raw/`, `data/processed/`, `reference.db`, out-of-fold predictions, the
benchmark profile key) are git-ignored. The real job-ad files in `data/raw/` contain no personal data; add
them to version control explicitly if you want them kept with the code. The API never writes or logs submitted profiles, and the frontend
does not cache them.

## Development process

How the project was developed (with dated evidence) is in `docs/development_log.md`; the branch, commit and pull-request workflow is in `CONTRIBUTING.md`.
