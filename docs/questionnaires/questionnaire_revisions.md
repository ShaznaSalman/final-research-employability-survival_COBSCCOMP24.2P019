# Questionnaire revisions (v2) — required before data collection

The application's data pipeline (`core/schema.py`) is built against **version 2** of the
graduate questionnaire described here. Version 1 is the file
`1.Revised-Graduate_Employability_Questionnaire.docx` in this folder.

Existing question numbers are **kept**; new items are added as sub-items (Q5a, Q17a …) so
the Evidence–Justification Matrix and the 47-question count remain valid.

> **Ethics:** these changes alter what is collected (adds month/year dates, exact counts).
> Check with your supervisor whether an ethics amendment is required before using v2.

---

## Why these changes matter

| Problem in v1 | Consequence for a survival model | Fix |
|---|---|---|
| Q18/Q19 rate **current** skills | Employed graduates' skills partly reflect on-the-job learning → reverse causation; skill effects would be inflated | Rate skills **at the time you completed your degree** |
| Graduation is year-only (Q5) | Time origin unknown to within 12 months | Add month + year of degree completion (Q5a) |
| Time-to-job is banded (Q17: 13–15, 16–18, 19–24, 24+) | Coarse, interval-censored outcome; "0" cannot distinguish "before graduation" from "within a month" | Ask month + year the first IT role started (Q17a); duration computed from dates |
| "Still searching" time asked separately (Q17b) | Redundant and error-prone | Use the form's automatic submission timestamp as the survey date |
| Event defined by *current* status (Q16) | Someone who had an IT job and then left/moved to further study is misclassified | Ask directly whether a first paid IT role has ever started (Q17) |
| GPA, certifications, internships, projects are banded | Loses information; internship **count** not collected at all | Ask exact numbers (Q6, Q9, Q11a, Q12, Q13) |
| No GPA for classification-based (e.g. UK-affiliated) degrees | Missing GPA for a whole university type | Q6a: degree class as an alternative |

---

## Graduate questionnaire — changed / added items (exact wording)

**Q5a.** *(Graduates only)* In which month and year did you **complete** your degree?
Use the month your final results were released — **not** the convocation/graduation ceremony.
`Month: [Jan–Dec]  Year: [2023–2027]`

**Q5b.** *(Final-year students only)* In which month and year do you **expect to complete**
your degree (final results released)?
`Month: [Jan–Dec]  Year: [2026–2028]`

**Q6 (replaces bands).** What is your final GPA (graduates) or current cumulative GPA
(final-year students), on a 4.0 scale? Enter to two decimals, e.g. 3.25.
`Number 0.00–4.00` — leave blank if your degree does not use a GPA, and answer Q6a instead.

**Q6a.** *(Only if your degree does not use a GPA)* What is your final / expected degree class?
`First class · Second class upper · Second class lower · Third class / General pass`

**Q9 (replaces bands).** How many professional IT certifications have you obtained?
`Number 0–20`

**Q11a.** *(If Q11 = Yes)* How many separate IT internships have you completed (including
internships that were part of your degree)?
`Number 1–5`

**Q12 (replaces bands).** In total, how many months of IT internship experience do you have?
Enter 0 if none.
`Number 0–36`

**Q13 (replaces bands).** How many personal or academic IT projects have you completed
outside coursework?
`Number 0–30`

**Q17 (replaces Q17 and Q17b).** Have you **ever started** a paid IT role — full-time,
part-time, contract, or a paid trainee/intern position taken **after** your degree?
Do **not** count internships that were part of your degree programme.
`Yes · No`

**Q17a.** *(If Q17 = Yes)* In which month and year did your **first** such paid IT role start?
If it started before you completed your degree, still give the actual month.
`Month: [Jan–Dec]  Year: [2020–2027]`

**Q18 (reworded stem).** Please rate your proficiency in each of the following technical
skills **at the time you completed your degree**. *Final-year students: rate your current level.*
(Items and 1–5 scale unchanged.)

**Q19 (reworded stem).** Please rate yourself on each of the following soft skills **at the
time you completed your degree**. *Final-year students: rate your current level.*
(Items and 1–5 scale unchanged.)

**Q14, Q21, Q23, Q24, Q29 (reworded to the same reference point as Q18/Q19)** - these are candidate
predictors (see `artifacts/feature_set_comparison.md`), so they must describe the situation **before the
degree was completed**, not today:

- **Q14.** Did you have a technical portfolio (e.g. GitHub, personal website) **before you completed your
  degree**? `Yes · No`
- **Q21.** **During your degree**, how frequently did you engage in self-learning outside coursework?
  `1 (never) - 5 (very frequently)`
- **Q23 (replaces the yes/no + bands).** **Before you completed your degree**, how many hackathons, coding
  competitions or technical events did you take part in? `Number 0-30`
- **Q24.** Had you contributed to open-source projects **before you completed your degree**? `Yes · No`
- **Q29.** **At the time you completed your degree**, how familiar were you with software development
  lifecycle (SDLC) practices? `1 (no knowledge) - 5 (advanced)`

Q20 (when you started preparing for IT employment) is already retrospective and stays unchanged.
Not used as predictors because they describe the job search itself or are measured "now" (risk of reverse
causation): Q36 job-search platforms, Q37 CV updates, Q38 LinkedIn completeness.

**Q16** — unchanged. It is now used only to identify graduates who are **not in the IT job
market** (Further studies; "have not started job-seeking"), who are excluded from training
if they have never started an IT role.

**Q16a (new, optional)** — shown only if Q16 is *Employed in a non-IT role*, *Self-employed* or
*Further studies*:
> In which month did you start your current non-IT job, self-employment or further studies?
> `Month / Year` (optional)

Used only in the competing-risks analysis (`core/competing.py`), where starting another path is a
competing event. It is **not** a predictor and does not change the main model. Older exports without
Q16a are still accepted (the column is set to missing).

All other items (Q1–Q4, Q5, Q7–Q8, Q10, Q11, Q14–Q15, Q20–Q47) are unchanged. Q16a is the only
new item added after the first revision.

### Survey date
Use the Google Forms **Timestamp** column as `survey_date`. Keep it in the export.

---

## Employer questionnaire — changes

| Item | Change | Reason |
|---|---|---|
| Title | Replace "An Explainable AI Framework for Predicting and Supporting Employability of Entry-Level IT Graduates in Sri Lanka" with the current dissertation title | Consistency with the graduate form and ethics documents |
| Q8 labels | "SQL / Database Knowledge" → "SQL / Database Management"; "Git / Version Control" → "Git / GitHub"; "Linux" → "Linux / Command Line" | Identical wording on both sides of the skill-gap comparison |
| Q9 | Add a 9th item: **Leadership Ability** | Graduate Q19 has 9 soft skills; without this, Leadership has no employer benchmark |

Item count remains **21** questions (Q1-Q21).

---

## Event definition used by the model

- **Event (found work)** = start of the first paid IT role (Q17 = Yes), dated by Q17a.
- **Time origin** = month of degree completion (Q5a).
- **Still searching (censored)** = graduates with Q17 = No, observed until the survey date.
- **Excluded from training** = graduates with Q17 = No **and** Q16 ∈ {Further studies,
  Not yet graduated / have not started job-seeking}: they are not in the IT job market,
  so treating them as "still searching" would bias times upward.
- Self-employed / freelance IT work does not count as an event (no employer hire).
- **Competing-risks analysis only:** a graduate with Q17 = No who gives a Q16a month has a
  competing event (another path) at that month; graduates in further studies who give the month are
  kept in that analysis instead of being excluded.

---

## Version 2.1 - clarity fixes (no change to what is measured)

No question was added or removed, no model input changed, and every analysis is unchanged: the question numbers,
the stored categories and the model are identical to v2 (verified: re-running the pipeline changes only the
display label of employer Q11).

| Item | Change | Problem it fixes | Justification |
|---|---|---|---|
| Q5a | Note: completion must be within the last two years | Respondents outside the eligibility window could answer | Keeps the sample consistent with the stated scope (Chapter 1, Section 1.5) |
| Q9 | Definition: certifications earned by passing an assessment (vendor or professional exams, verified online professional certificates); not attendance certificates or the degree | "Professional certification" was undefined, so respondents counted different things | A defined construct gives a comparable count across respondents (content validity); matches the evidence for Q9 (Athey & Palikot, 2024, verified online certificates) |
| Q16 | Note: a paid IT internship or trainee role after the degree is "IT internship" here and "Yes" in Q17 | The same person could be classified two ways | Makes the event definition (first paid IT role) unambiguous |
| Q17 | Note: paid internships / trainee roles after the degree count; unpaid ones and degree internships do not | Unclear whether unpaid or degree internships count as the event | Same event definition as Chapter 3 and `core/clean.py` |
| Q31, Q39 | Option "Not applicable - I have not had a job interview yet" | Final-year students and graduates without interviews had no truthful answer | Separates "not applicable" from non-response; recorded as missing, so no analysis changes |
| Q43 | Option "Not applicable - I have not started looking for IT jobs yet" | Same, for respondents not yet job-seeking | As above |
| Employer Q11 | "Less than every year" reworded to "More often than once a year" | The old wording was ambiguous (more or less often?) | Same category (`more_than_yearly`); the old wording is still read correctly |

### Repeated questions are kept deliberately

Q25, Q27, Q28, Q32 and Q33 ask about skills that Q18/Q19 also rate. They are retained as **parallel items for
convergent validity**: self-rated skills are known to be biased (Lai et al., 2025), so asking the same construct a
second way tests whether the self-ratings used by the model are consistent (Spearman correlation; analysed in
`core/survey_study.py`). They are not model inputs. If the pilot shows a completion time above 20 minutes, they are
the first items to remove, and that decision will be recorded here.

### Attention check (added in v2.1)

| Item | Change | Justification |
|---|---|---|
| Q18 (graduate) and Q8 (employer) grids | One extra row, "Attention check - please select 4 for this row", placed mid-grid | Detects careless responding in long rating grids, as promised in Chapter 3 (Section 3.11). Respondents who answer it with anything other than 4 are excluded in cleaning and logged in the audit trail ("Failed attention check"). It is not a numbered question, so question counts are unchanged; it is not a model input. |

---

## Version 2.2 - additions from Sri Lankan public data (graduate questionnaire only)

Added after reviewing the national UGC/USJ Employability Survey 2023 questionnaire, the University of Kelaniya
Graduate Employability 2022 report and the Stack Overflow Developer Survey 2025. All new items are **descriptive /
questionnaire-study items, not model inputs**, so the model and its results are unchanged. Question numbering is
kept (new sub-items Q7a, Q7b).

| Item | Change | Justification |
|---|---|---|
| Q7a (new) | Main medium of instruction (English / Sinhala / Tamil / mixed) | Asked in the national UGC/USJ 2023 employability census; the Kelaniya 2022 report analyses employability by medium of instruction |
| Q7b (new) | English proficiency (speaking and writing) at degree completion, 1-5 | English proficiency is rated a key employability factor by 60% of employed Kelaniya graduates (2022) and is measured in the UGC/USJ 2023 census; asked "at completion" to avoid reverse causation |
| Q22 | New option "AI tools (e.g. ChatGPT, GitHub Copilot)" | 61% of Sri Lankan developers in the Stack Overflow Developer Survey 2025 learn with AI tools; the option was missing |
| Q35 | Same new option | Same evidence |

Item counts: 47 numbered graduate questions (sub-items now Q5a, Q5b, Q6a, Q7a, Q7b, Q11a, Q16a, Q17a, Q34a, Q38a,
Q39a); employer questionnaire unchanged (21).

## Version 2.2 - employer questionnaire

Added after reviewing Sri Lankan employer-side studies (SLASSCOM-Deloitte 2024; national graduate surveys that rate
English as a key factor). Both changes are **questionnaire-study items, not model inputs**; the model is unchanged.

| Item | Change | Justification |
|---|---|---|
| Q9a (new) | Importance of English proficiency for entry-level IT graduates (1-5) | English is highlighted by every Sri Lankan source reviewed (UGC/USJ 2023 census; Kelaniya 2022: 60% of employed graduates name it); pairs with graduate Q7b so the graduate-employer gap can be compared |
| Q13 | New option "Attitude / lack of interest in the job" | Employers seek a strong work ethic (NACE, 2023: about two-thirds of employers); in Sri Lanka, lack of interest was cited by a quarter of employers in the DCS Labour Demand Survey (2017, background only) |

Item count: 21 numbered employer questions plus sub-item Q9a.

### Number questions as dropdowns (v2.2, graduate questionnaire)

| Item | Change | Justification |
|---|---|---|
| Q6 GPA | Short answer replaced by a dropdown in 0.1 steps (1.9 or below, 2.0 ... 4.0) | Easier on phones and removes typing errors; rounding to 0.1 (at most +/-0.05) does not materially affect the model |
| Q9, Q11a, Q12, Q13 | Short answers replaced by dropdowns of exact values with an open top category (10 or more; 5 or more; 25 or more; 16 or more) | Keeps the exact counts the survival model needs (unlike bands), while avoiding typing; the top categories cover the rare high values and are read as their lower bound |

The system reads the number from each option ("10 or more" -> 10), tested in `tests/test_survey_study.py`.
Q23 (hackathons) stays a short number answer.

### Online form settings (v2.2 build)

| Item | Change | Justification |
|---|---|---|
| Required questions (graduate) | Q3, Q4, Q5, Q5a/b, Q6, Q9, Q11, Q11a, Q12, Q13, Q14, Q16, Q17, Q17a, Q18 and Q19 (every row), Q20, Q21, Q23, Q24, Q29 | Every model input and outcome item is answered, so missing data is minimised; all other items stay optional, and respondents can still stop at any time |
| Required questions (employer) | Q1, Q2, Q3, Q7, Q8 and Q9 (every row) | Items used in the market-alignment comparison |
| Q6 | New option "My degree does not use a GPA" | Lets Q6 be required while respondents without a GPA answer Q6a (degree class) instead; read as a missing GPA, so the degree class is used |
| Q16 | Note: "Final-year students: please choose 'Not yet graduated / have not started job-seeking'." | Keeps final-year students out of the graduate-only Q17 / Q17a |
