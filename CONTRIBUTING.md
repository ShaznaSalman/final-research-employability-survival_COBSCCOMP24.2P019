# Contributing and version control

## Branches

| Branch | Purpose |
|---|---|
| `main` | Always working: tests pass and `python -m core.run_all` runs. Never commit to it directly. |
| `feature/<name>` | New functionality |
| `fix/<name>` | Bug fixes |
| `data/<name>` | Data collection, import or cleaning changes |
| `analysis/<name>` | New or re-run analyses and their generated reports |
| `docs/<name>` | Documentation and dissertation material |

## Workflow for every change

```bash
git checkout main && git pull
git checkout -b feature/short-name
# ... work in small steps; commit after each logical step ...
python -m pytest                      # must pass
git push -u origin feature/short-name
# open a pull request on GitHub (the template fills in), review the diff, then "Create a merge commit"
git checkout main && git pull && git branch -d feature/short-name
```

## Commit messages

- One logical change per commit (a module, its test, a doc section) - small commits are easier to review and revert.
- Imperative summary line of at most 72 characters: `Add leave-one-institution-out validation`,
  not `added stuff`.
- If needed, a blank line and a short body saying **why** the change was made.
- Generated reports (`artifacts/`, `docs/data_cleaning_log.md`, `docs/data_dictionary.md`) go in their own commit,
  e.g. `Regenerate results after real-data import`.
- Never commit respondent data, `.env` or model binaries (see `.gitignore`).

## Planned branches for the remaining work

Each entry lists the planned commits and a draft pull-request description. Dates are whenever the work is done.

### 1. `fix/questionnaire-logic`
Commits:
1. `Add Not applicable option to Q31, Q39 and Q43 for final-year students`
2. `Align Q5 graduation-year options with the eligibility window`
3. `Clarify Q16 vs Q17 internship wording`
4. `Define professional certification in Q9`
5. `Update forms adapter label maps for the new options`
6. `Add tests for the new questionnaire options`

PR: *Fixes the logic problems found in the questionnaire review so every respondent can answer every question
truthfully and the outcome is classified consistently. Schema and adapter updated; tests added.*

### 2. `feature/leave-institution-out-cv`
Commits:
1. `Add leave-one-institution-type-out validation`
2. `Add report for institution transfer performance`
3. `Show institution transfer results on the Research page`
4. `Add tests for institution-out validation`

PR: *Implements the "works across institutions" check promised in Chapter 3 (Table 3.6): train on two institution
types, test on the third.*

### 3. `data/pilot-study`
Commits:
1. `Add expert-review ratings and item decisions`
2. `Add pilot response summary (completion time, skipped items, alpha)`
3. `Record questionnaire changes made after the pilot`

PR: *Documents the four-stage pilot and validation (Chapter 3, Table 3.7) and every change it caused.*

### 4. `data/real-survey-import`
Commits:
1. `Map final Google Forms headers to the schema`
2. `Import real graduate responses (data stays outside git)`
3. `Import real employer responses (data stays outside git)`
4. `Regenerate cleaning log and data dictionary`

PR: *Switches the pipeline from synthetic to real responses. Respondent data is not committed; only the cleaning
log with counts is. Synthetic files remain for tests.*

### 5. `analysis/real-data-results`
Commits:
1. `Re-run nested CV and model selection on real data`
2. `Re-run feature-set comparison and record the decision`
3. `Re-run fairness and secondary analyses`
4. `Re-run job-market and questionnaire studies`
5. `Regenerate figures`

PR: *All results now come from real data. Includes the feature-set rule outcome on real data and whether the
extended-set deviation is kept.*

### 6. `feature/usability-fixes`
Commits: one per usability issue found in the SUS sessions, e.g. `Clarify confidence label wording`,
`Make skill sliders easier on mobile`, plus `Add SUS results summary`.

PR: *Fixes from the usability study (n = 8-12); SUS mean and 95% CI reported in docs.*

### 7. `analysis/llm-benchmark`
Commits:
1. `Re-export benchmark prompts for the current feature set`
2. `Add recorded LLM responses`
3. `Score LLM against the selected model`

PR: *Re-runs the RQ4 comparison on held-out profiles with the paired bootstrap.*

### 8. `docs/chapters-4-6`
Commits: one per chapter section, e.g. `Write Chapter 4 data description`, `Write Chapter 4 model results`,
`Write Chapter 5 discussion of RQ2`, `Add appendices: questionnaires and evidence matrix`.

PR: *Dissertation chapters based on the real-data results.*
