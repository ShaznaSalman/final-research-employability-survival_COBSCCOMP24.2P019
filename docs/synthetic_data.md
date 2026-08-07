# Synthetic data (DEMO ONLY)

> **Not used for any reported result since 11 Oct 2026.** The real main-study responses in `data/raw/graduate_responses.csv` and `employer_responses.csv` take priority over these files. The synthetic data are kept only for the automated tests and the simulation study, which need effects known in advance.

All files ending in `_SYNTHETIC.csv` are **fake**. They exist so the full pipeline can be built and
tested before real survey data is collected. Regenerate with `python -m core.synthetic` (fixed seed
`20261006`). Generator: `core/synthetic.py`. The data are designed to *behave* like a real Sri Lankan
IT-graduate survey - messy, noisy and imperfect - while every true effect stays known.

| File | Rows | Contents |
|---|---|---|
| `data/raw/graduates_SYNTHETIC.csv` | 390 | 306 graduates + 78 final-year + 6 duplicate submissions |
| `data/raw/employers_SYNTHETIC.csv` | 280 | Employer questionnaire v2 responses |
| `data/raw/job_ads.csv` (**real**) | 324 | Real entry-level IT ads (internships, trainee/associate/junior roles, < 2 years' experience) posted 12 Jul - 11 Oct 2026 by 211 companies: 42 from topjobs.lk (software and hardware/networks categories) and 4 from xpress.jobs coded by hand from the ad flyers; 262 from ITPro.lk (`source` = other; current listings plus the site's sitemap, fetched at its 10-second crawl delay) 8 from jobhub.lk (its job sitemap, filtered to entry-level IT titles; most of its IT ads are reposts of topjobs.lk ads already collected) and 8 from LinkedIn (links chosen by the researcher), coded with the keyword rules in `data/reference/job_ad_coding_rules.json`. Known limitation of the keyword coding: terms in a company's own description (e.g. 'AI', 'analytics', 'cloud') are counted as mentioned, so AI/ML, analytics and cloud are slightly over-counted. The file takes priority over the synthetic one. Links, coding method, experience required and the 173 excluded ads (with reasons) are in `data/raw/job_ads_provenance.csv` |
| `data/raw/job_ads_SYNTHETIC.csv` (unused while the real file exists) | 605 | 600 job ads (98 companies, incl. reposts) + 5 recorded twice; skills written as a person would type them |
| `data/reference/university_types_SYNTHETIC.csv` | 18 | 7 fictional institutions with their common name variants |
| `data/raw/synthetic_manifest_SYNTHETIC.json` | - | Seed, true effects, outcome counts, every injected defect |

## Anchors from published sources and realism scorecard

The generator is calibrated against published Sri Lankan figures, and `core/synthetic.py` checks the
generated data against them every time it runs (`realism_scorecard`, saved in the manifest):

| Measure | Published | Synthetic (seed 20261006) | Accepted range | Within range |
|---|---|---|---|---|
| Women among graduates (%) | 30.3 | 33.3 | 24.0-37.0 | yes |
| Average months to first job after completion (graduates followed >= 12 months) | 8.0 | 6.3 | 6.0-10.0 | yes |
| Employed graduates who found work before completing (%) | 18.1 | 12.6 | 10.0-25.0 | yes |
| Graduates employed at the time of the survey (%) | 60.5 | 61.1 | 55.0-80.0 | yes |

Sources: University Statistics & Data Monitoring Unit, University of Kelaniya (2022), *Graduate
employability 2021* - average 8 months to employment overall and in the Faculty of Computing & Technology,
60.5% employability, 18.1% of employed graduates found work before or while studying
([report](https://units.kln.ac.lk/usdmu/media/attachments/2024/11/07/report_graduate-employability-2021_20230316---final1.pdf));
Information and Communication Technology Agency of Sri Lanka (2019), IT-BPM Workforce Survey 2019 - 30.3%
of Bachelor's-level ICT workers are women ([news release](https://www.icta.lk/media/news/sri-lanka-aiming-200000-ict-workforce-by-2022)).

Notes on comparability: the Kelaniya figures cover all disciplines at one university and a whole cohort
surveyed after graduation, so the synthetic average time is computed only for graduates followed for at
least 12 months; it comes out somewhat shorter (about 6 months) than the published 8 months. The
employment rate is expected to be higher for IT than for all disciplines. Everything not in this table is
an assumption, listed below so it can be revised.

## Population and survey behaviour

- **Institutions:** 3 state, 2 non-state, 2 foreign-affiliated (invented names such as "Mahasena National University"; none is a real institution). Degree mix depends on type
  (e.g. 40% Software Engineering at foreign-affiliated, 35% Computer Science at state).
- **Skills depend on degree:** Data Science / AI-ML rate Data & AI higher, Cyber Security rate
  Quality & Security higher, Software Engineering rate Programming and Testing higher.
- **Self-ratings are not true skill:** each item = true skill + inflation (technical +0.30, soft +0.55)
  + a personal rating style (SD 0.35) + noise, rounded to 1-5; 35% of would-be 1s become 2s;
  6% of respondents straight-line each block.
- **GPA:** state universities grade ~0.1 lower; reported GPA is slightly optimistic (+0.03) and rounded
  (30% to one decimal, 10% to the nearest 0.5); 70% of foreign-affiliated respondents give a degree class
  instead; weaker students (GPA < 2.7) skip the GPA question 5x more often.
- **Experience:** internships are near-universal at state universities (95%), 85% at non-state and
  65% at foreign-affiliated; months heap at 6 (55%), 3 and 12. Certifications are zero-inflated (~55%
  have none). Project counts are over-dispersed and heaped at 5 and 10.
- **Survey fatigue:** technical items skipped 1.2% of the time, soft-skill items (later in the form) 2.5%.
- **Rushed respondents:** 3% click their skill ratings at random.
- **Non-response bias:** a larger pool of graduates is generated and employed graduates are more likely to
  answer (75% vs 55%), so the respondents over-represent people who found work - as voluntary surveys tend to.
- **Recall error:** 25% of employed respondents remember their job start as a round number of months after
  completion (3, 6, 9, 12, 18 or 24), and 8% give their convocation month instead of the results-release
  month (2-6 months later), which shortens - or even reverses - their apparent search.
- **Messy names:** 30% of respondents type a variant of their institution's name ("MNU",
  "mahasena uni"); 3% make a typo that is *not* in the lookup file and shows up in the cleaning log.
- **Age:** state-university graduates finish older (waiting for A/L results and admission): ~30% are 26+
  versus ~10% at non-state and foreign-affiliated institutions.
- **Curriculum emphasis:** small item-level offsets by institution type (state: Java and Linux higher,
  JavaScript and cloud lower; non-state: JavaScript and web higher; foreign-affiliated: Python and cloud
  higher). The per-type average stays near zero, so university type still has no true effect on the outcome.
- **Gender confidence gap:** women rate their technical skills 0.15 lower than men **at the same true level**.
  There is no true skill or outcome difference, so any gender gap in the self-reported `technical_mean` is a
  measurement artefact - which the fairness check should be read against.
- **Certifications by degree:** Cyber Security graduates are more likely to hold certifications (and hold
  more), IT and Data Science slightly more.
- **Industry modules (Q7):** reported more often at non-state and foreign-affiliated institutions.
- **Reply times:** a lunchtime bump and an evening peak (19:00-21:00), with a few replies after midnight.

## Data-generating process for time to first IT job

Weibull proportional hazards (shape 1.2, scale 11.5 months) on **true** characteristics; the model only
sees the self-reported versions:

log-hazard = 0.40 x (GPA - 3.0) + **0.25 if GPA >= 3.0** (shortlisting cut-off)
           + 0.05 x (internship months - 6)
           + 0.20 x (log(1 + certifications) - log 2) (diminishing returns)
           + 0.07 x (min(projects, 10) - 3)
           + 0.50 x (true technical skill - 3.0) + 0.30 x (true soft skill - 3.5)
           - 0.20 [IT/IS degree] + 0.10 [specialised degree]
           **- 0.25 if completed before 2025** (slower market for earlier cohorts)
           + log(frailty), frailty ~ Gamma(mean 1, variance 0.3) (networks, interviews, luck)

| Feature | True effect | Notes |
|---|---|---|
| GPA | +0.40 per point, +0.25 at >= 3.0 | non-linear |
| Internship months | +0.05 per month | |
| Certifications | +0.20 per log(1 + n) | diminishing returns |
| Projects | +0.07 each, capped at 10 | |
| Technical skill | +0.50 per point of **true** skill | self-ratings are a noisy, inflated proxy |
| Soft skill | +0.30 per point of **true** skill | |
| Degree group | IT/IS -0.20, specialised +0.10 (vs computing) | |
| Portfolio before completion (Q14) | +0.20 | candidate predictor (extended set) |
| Years of preparation (Q20) | +0.08 per year | candidate |
| Hackathons / events (Q23) | +0.15 per log(1 + n) | candidate; 55% attended none |
| Open-source contribution (Q24) | +0.10 | candidate; ~13% contributed |
| Self-learning frequency (Q21) | **0** | candidate; correlates with ability only |
| SDLC familiarity (Q29) | **0** | candidate; correlates with ability only |
| University type | **0** | fairness check has a known answer |
| Gender | **0** (not a model feature) | |

Additional mechanisms: ~8% of an average profile (more for low technical skill) never move into IT within
the window; ~4% start their first IT role before or at completion; 6% of graduates are not in the IT job
market (further studies, often abroad) and are excluded from training.

**Dates:** completion months cluster at each institution's two results releases; graduates from the last
26 months respond, recent ones more often; survey responses arrive in three waves (5 Oct, 2 Nov,
30 Nov 2026) and tail off over the following days.

Resulting outcome (seed 20261006): **294 graduates in training, 179 found an IT role, 115 still searching (39%)**.

### Q16a: start month of another path (competing risks only)

Graduates with no IT role whose status is a non-IT job, self-employment or further studies answer the optional
Q16a with probability 0.85; the month is drawn uniformly between completion and the survey. It uses its **own
random stream** (seed + 7), so adding it changed no other value in the data, and it is drawn independently of
the IT-job process - competing-risks results on synthetic data are illustrative only.

## Employers and job ads

- Employers rate by organisation type: banks and government emphasise GPA and degree; startups emphasise
  portfolios, projects and JavaScript/web; telecoms emphasise Linux and security; plus rater leniency,
  5% straight-lining and 3% skipped ratings.
- Organisation size depends on type (banks and telecoms mostly 500+, startups almost all under 50). Large
  employers weight degree, GPA and certifications more; small ones weight portfolios and projects more.
- 4% of employers answer the hiring-factor questions and then leave before the skills grid.
- Job ads: 9 role families with realistic titles ("Trainee QA Engineer", "Intern - Full Stack Developer"),
  sometimes with suffixes (" - Colombo", " (Fresh Graduates)", " | Hybrid") or in all capitals. About one
  company per six ads, with invented names ("Bluecove Technologies", "Nuvara Labs"; none is a real firm);
  a few large firms post most of them. More ads on weekdays and in Aug-Sep.
- Skills are written as a person would code them ("Spring Boot", "MySQL", "team player"), in no fixed order
  and with inconsistent capitalisation; 25% of ads include a role-typical term outside the skill vocabulary
  ("C#", ".NET", "PHP", "Jira", "Figma") which the cleaning log reports.
- 4% of ads are the same vacancy reposted 2-6 weeks later (kept: they are separate postings), and ~0.8%
  are recorded twice by mistake (removed by cleaning).

## Injected data-entry defects (for cleaning tests)

Beyond the realistic missingness above: missing values in gender (~1.3%, at least 5), age group (~0.8%)
and one degree type; 7 out-of-range entries (GPA 4.3 and 34, Likert 6 and 0, 99 certifications,
48 internship months, -1 projects); 4 inconsistent outcome rows (graduation after the survey date,
job-start year typo 2006, two "yes" answers with no start month); and duplicate submissions (~1.5%, at least
6). Each is listed with its `respondent_id` in the manifest.

## What this means for the results

With the realistic noise above - unobserved luck and networks, rating styles, recall error, non-response
bias and graduates who never enter IT - the selected model's C-index is about **0.66** (locked test set)
(it picks the right one of two graduates about 67 times in 100), and predicted employment shares are within
a few percentage points of what happened. This is in the range typically seen when predicting job-search
time from questionnaire data. A low-noise demo setting can push the figure above 0.7, but that only makes
the fake data easier; **the real C-index is whatever the survey data produce**, and only that figure may be
reported as a result.

Effect *directions* are recovered on a large sample (`tests/test_synthetic.py`, `tests/test_pipeline.py`).
At the real sample size, small effects may not be reliably detected, and subgroup checks can flag
differences that are pure chance (the true effect of gender and university type is zero).


## Questionnaire-study items

The items that are not model inputs (graduate Q8, Q10, Q15, Q22, Q25-Q28, Q30-Q47 and employer Q3-Q6, Q10-Q21)
are added by `core/synthetic_survey.py` **after** the files above are generated, from a separate random stream
seeded by each row's other answers. Every existing column - and so every model result - is unchanged, and a
duplicated submission gets identical answers (so it is still removed as a duplicate). The answers are linked to
the respondent's other answers where a real link is expected (for example a language rated 4-5 in Q18 is usually
ticked in Q25; employed graduates rate their employability higher in Q40). The consistency checks in
`artifacts/questionnaire_study.md` therefore recover links that were **built into the generator**; only real
responses can show whether the questions really repeat each other.
