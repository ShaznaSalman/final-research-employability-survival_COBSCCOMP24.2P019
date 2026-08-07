"""Synthetic data generator (graduates, employers, job ads) - realistic demo data, clearly labelled.

Produces files in the exact raw format of core/schema.py, all named *_SYNTHETIC.csv.

Realism (see docs/synthetic_data.md for every assumption and its source):
  - Population: gender mix from ICTA's ICT workforce survey (~30% women among degree holders); degree mix,
    age group and industry modules depend on institution type; skill profiles depend on degree and on each
    institution type's curriculum emphasis.
  - Survey behaviour: inflated self-ratings (soft skills most), personal rating style, a gender confidence
    gap in technical self-ratings (no true skill difference), straight-lining, avoidance of the lowest
    category, rounded GPAs, weaker students skipping GPA, fatigue-driven skipping, evening-heavy reply times.
  - Experience: zero-inflated certifications (more in cyber security / IT / data science), internship months
    heaped at mandatory 6-month placements, heaped project counts.
  - Employers: organisation size depends on type; large firms weight qualifications, small ones portfolios;
    a few respondents leave before the skills grid.
  - Job ads: many firms (a few post most ads), varied titles and capitalisation, unordered requirements,
    role-specific non-vocabulary tokens (C#, PHP, Jira...), reposted vacancies and double-recorded ads.
  - Outcome: Weibull proportional hazards on TRUE (latent) skill - the model only sees self-ratings - with
    a GPA >= 3.0 shortlisting bonus, diminishing returns to certifications, unobserved heterogeneity
    (frailty), a share who never enter IT within the window, and a slower market for earlier cohorts.
  - Dates: completion months cluster at each institution's results releases; recent graduates respond
    more; survey responses arrive in waves.
  - Messy text: university name variants and a few unmapped typos; job ads use natural skill names.
Gender and university type have NO true effect, so the fairness check has a known answer.

Run:  python -m core.synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import forms_adapter, schema

SEED = 20261006
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"

N_GRADUATES = 306
N_FINAL_YEAR = 78
N_EMPLOYERS = 280
N_JOB_ADS = 600

# ---------------------------------------------------------------------------
# Known data-generating process for time to first IT job
# ---------------------------------------------------------------------------
# Effects on the hazard of finding a first IT job (positive = SHORTER search). Linear terms are per unit.
TRUE_LOG_HAZARD_RATIOS = {
    "gpa": 0.40,                    # per GPA point, plus GPA_THRESHOLD_BONUS at >= 3.0
    "internship_months": 0.05,      # per month
    "certifications_count": 0.20,   # per unit of log(1 + certifications): diminishing returns
    "project_count": 0.07,          # per project, capped at PROJECT_CAP
    "technical_mean": 0.50,         # per point of TRUE technical skill (self-ratings are a noisy, inflated proxy)
    "soft_mean": 0.30,              # per point of TRUE soft skill
    "degree_group[it_is]": -0.20,   # vs computing
    "degree_group[specialised]": 0.10,
    "university_type[non_state]": 0.0,
    "university_type[foreign_affiliated]": 0.0,
    "gender": 0.0,
    # Candidate predictors (extended feature set): some carry real signal, some only correlate with ability.
    "has_portfolio": 0.20,          # had a portfolio before completion
    "prep_years": 0.08,             # per year of preparation before completion
    "hackathon_count": 0.15,        # per unit of log(1 + events)
    "open_source": 0.10,
    "self_learning_freq": 0.0,      # no direct effect (correlated with ability only)
    "sdlc_familiarity": 0.0,        # no direct effect (correlated with ability only)
}
GPA_THRESHOLD, GPA_THRESHOLD_BONUS = 3.0, 0.25   # employers' shortlisting cut-off
PROJECT_CAP = 10
EARLY_COHORT_BEFORE = "2025-01"                  # graduates completing before this faced a slower market
EARLY_COHORT_EFFECT = -0.25
FRAILTY_VARIANCE = 0.30                          # gamma frailty: unobserved factors (networks, interviews, luck)
NEVER_IT_BASE = 0.08                             # share who never move into IT within the window (avg profile)
TESTED_EFFECTS = ("gpa", "internship_months", "certifications_count", "project_count",
                  "technical_mean", "soft_mean")
CENTRE = {"gpa": 3.0, "internship_months": 6, "certifications_count": 1, "project_count": 3,
          "technical_mean": 3.0, "soft_mean": 3.5}
WEIBULL_SHAPE = 1.2
WEIBULL_SCALE = 11.5      # months; tuned to the published anchors (see realism_scorecard)
P_JOB_BEFORE_GRADUATION = 0.03   # plus convocation-month confusion -> ~18% of employed (Kelaniya 2021)
P_NOT_IN_MARKET = 0.06    # further studies (often abroad) / not seeking -> excluded from training later

# Survey behaviour
SELF_RATING_INFLATION = {"technical": 0.30, "soft": 0.55}
STYLE_SD = 0.35           # personal rating style (some people rate everything higher)
P_STRAIGHT_LINE = 0.06    # answers one value for a whole block
GPA_ROUNDING = {"none": 0.60, "tenth": 0.30, "half": 0.10}

SURVEY_WAVES = ("2026-10-05", "2026-11-02", "2026-11-30")
# Replies by hour of day (0-23): a lunchtime bump and an evening peak, a few night owls.
HOUR_WEIGHTS = np.array([1.0, 0.7, 0.4, 0.2, 0.1, 0.2, 0.5, 1.2, 2.0, 2.4, 2.5, 2.8, 3.6, 3.3, 2.6, 2.4, 2.6, 3.0,
                         3.8, 4.8, 5.2, 4.8, 3.6, 2.2])
P_MISSING_GENDER, P_MISSING_AGE = 0.013, 0.008
DUPLICATE_RATE = 0.015    # pressed Submit twice

# Real-survey mechanisms
RESPONSE_PROB = {"employed": 0.75, "searching": 0.55}   # non-response bias: employed graduates answer more often
P_RECALL_ROUNDING = 0.25      # job start remembered as "about 3/6/9/12/18/24 months after"
P_COMPLETION_CONFUSION = 0.08 # gives the convocation month instead of the results-release month (2-6 months later)
P_CARELESS = 0.03             # rushed respondents who click skill ratings at random
P_ALT_DATE_ANSWERED = 0.85    # optional Q16a answered by graduates on a non-IT path
ALT_SEED_OFFSET = 7           # Q16a uses its own random stream, so adding it changes no other value
ROUNDED_MONTHS = np.array([3, 6, 9, 12, 18, 24])

# Published anchors used by realism_scorecard() (sources in docs/synthetic_data.md)
ANCHORS = {
    "female_share_pct": (30.3, 24.0, 37.0, "ICTA (2019): women among Bachelor's-level ICT workforce"),
    "mean_months_after_completion": (8.0, 6.0, 10.0, "Univ. of Kelaniya (2022): Faculty of Computing & Technology "
                                     "(synthetic: graduates followed for >= 12 months, for comparability)"),
    "employed_before_completion_pct": (18.1, 10.0, 25.0, "Univ. of Kelaniya (2022): found work before/while studying"),
    "employed_at_survey_pct": (60.5, 55.0, 80.0, "Univ. of Kelaniya (2022): all-faculty rate; IT expected higher"),
}   # link shared three times; replies decay over days

# ---------------------------------------------------------------------------
# Institutions (fictional) - name variants as respondents would type them
# ---------------------------------------------------------------------------
# (canonical name, type, sampling weight, results-release months, [variants...], [unmapped typos...])
UNIVERSITY_SPECS = [
    # Invented names: none is a real institution, so no generated outcome is attributed to a real organisation.
    ("Mahasena National University", "state", 0.15, (3, 9), ["MNU", "mahasena uni"], ["Mahasena Natinal University"]),
    ("Kandula National University", "state", 0.12, (4, 10), ["KNU", "Kandula Uni"], []),
    ("Nilwala University of Technology", "state", 0.08, (6, 12), ["nilwala university of technology", "NUT"],
     ["Nilwala UoT Matara"]),
    ("Bluecove Institute of Computing", "non_state", 0.20, (2, 8), ["BIC", "Bluecove Campus"], []),
    ("Talawa Institute of Information Technology", "non_state", 0.15, (1, 7), ["TIIT"], ["Talawa IIT (Kandy)"]),
    ("Westbridge Metropolitan Campus", "foreign_affiliated", 0.18, (5, 11), ["WMC", "Westbridge Campus"], []),
    ("Ashworth International College", "foreign_affiliated", 0.12, (6, 12), ["Ashworth Colombo"], []),
]
UNIVERSITIES = [(n, t, w) for n, t, w, *_ in UNIVERSITY_SPECS]          # (name, type, weight)
P_NAME_VARIANT, P_NAME_TYPO = 0.30, 0.03

DEGREE_PROBS_BY_TYPE = {
    "state": {"computer_science": .35, "information_technology": .25, "software_engineering": .10,
              "information_systems": .15, "cyber_security": .03, "data_science": .05, "ai_ml": .04, "other": .03},
    "non_state": {"computer_science": .15, "information_technology": .25, "software_engineering": .30,
                  "information_systems": .08, "cyber_security": .08, "data_science": .07, "ai_ml": .05, "other": .02},
    "foreign_affiliated": {"computer_science": .20, "information_technology": .15, "software_engineering": .40,
                           "information_systems": .05, "cyber_security": .08, "data_science": .06, "ai_ml": .04,
                           "other": .02},
}
P_INTERNSHIP_BY_TYPE = {"state": 0.95, "non_state": 0.85, "foreign_affiliated": 0.65}   # mandatory placements
GPA_SHIFT_BY_TYPE = {"state": -0.10, "non_state": 0.05, "foreign_affiliated": 0.05}      # grading differences
# Graduates' age group: state entrants wait for A/L results and admission, so they finish older.
AGE_PROBS_BY_TYPE = {"state": [0.08, 0.62, 0.24, 0.06], "non_state": [0.35, 0.53, 0.09, 0.03],
                     "foreign_affiliated": [0.40, 0.50, 0.08, 0.02]}
INDUSTRY_MODULES_BY_TYPE = {"state": [0.65, 0.22, 0.13], "non_state": [0.80, 0.11, 0.09],
                            "foreign_affiliated": [0.82, 0.10, 0.08]}                  # yes / no / not_sure
CERT_SHIFT_BY_DEGREE = {"cyber_security": 0.5, "information_technology": 0.2, "data_science": 0.2}
# Curriculum emphasis on single skills (small; the per-type average stays near zero).
UNI_ITEM_OFFSET = {"state": {"java": 0.3, "linux": 0.2, "javascript": -0.2, "cloud_computing": -0.2},
                   "non_state": {"javascript": 0.2, "web_development": 0.2, "linux": -0.2},
                   "foreign_affiliated": {"python": 0.2, "cloud_computing": 0.2, "java": -0.2}}
GENDER_RATING_GAP = 0.15   # women rate technical skills lower at the same TRUE level (confidence gap, no true effect)

AREA_OFFSET = {"programming": 0.3, "web_apis": 0.1, "data_ai": -0.1, "cloud_devops": -0.6, "quality_security": -0.3}
DEGREE_AREA_OFFSET = {
    "software_engineering": {"programming": 0.3, "quality_security": 0.3},
    "data_science": {"data_ai": 0.7, "programming": 0.1},
    "ai_ml": {"data_ai": 0.7},
    "cyber_security": {"quality_security": 0.8, "cloud_devops": 0.3},
    "information_technology": {"web_apis": 0.2, "programming": -0.2},
    "information_systems": {"data_ai": 0.2, "programming": -0.3},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _month_index(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month - 1


def _month_str(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _gpa_to_class(g: float) -> str:
    return "first" if g >= 3.7 else "second_upper" if g >= 3.3 else "second_lower" if g >= 3.0 else "third_or_pass"


def _likert_obs(latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Self-reported 1-5 rating: rounded, with people avoiding the lowest category."""
    v = np.clip(np.rint(latent), 1, 5)
    avoid_one = (v == 1) & (rng.random(v.shape) < 0.35)
    return np.where(avoid_one, 2, v).astype(int)


def lookup_dict() -> dict[str, str]:
    """Normalised university name -> type, as data/reference/university_types_SYNTHETIC.csv defines it."""
    return {forms_adapter.norm(n): t for n, t, *_ in university_lookup().itertuples(index=False)}


def linear_predictor(df: pd.DataFrame) -> np.ndarray:
    """True log-hazard from TRUE features (excluding frailty and cohort), centred on CENTRE."""
    b = TRUE_LOG_HAZARD_RATIOS
    eta = (b["gpa"] * (df["true_gpa"] - CENTRE["gpa"])
           + GPA_THRESHOLD_BONUS * (df["true_gpa"] >= GPA_THRESHOLD)
           + b["internship_months"] * (df["internship_months"] - CENTRE["internship_months"])
           + b["certifications_count"] * (np.log1p(df["certifications_count"]) - np.log1p(CENTRE["certifications_count"]))
           + b["project_count"] * (np.minimum(df["project_count"], PROJECT_CAP) - CENTRE["project_count"])
           + b["technical_mean"] * (df["true_technical"] - CENTRE["technical_mean"])
           + b["soft_mean"] * (df["true_soft"] - CENTRE["soft_mean"])
           + b["degree_group[it_is]"] * (df["degree_group"] == "it_is")
           + b["degree_group[specialised]"] * (df["degree_group"] == "specialised")
           + b["has_portfolio"] * df["true_portfolio"]
           + b["prep_years"] * (df["prep_years"] - 2)
           + b["hackathon_count"] * np.log1p(df["hackathon_count"])
           + b["open_source"] * df["true_open_source"])
    return np.asarray(eta, float)


# ---------------------------------------------------------------------------
# Graduates
# ---------------------------------------------------------------------------
def _survey_timestamps(n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    wave = rng.choice(len(SURVEY_WAVES), size=n, p=[0.5, 0.3, 0.2])
    days = np.minimum(rng.exponential(4.0, n), 25)
    hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    secs = hours * 3600 + rng.integers(0, 3600, n)
    starts = pd.to_datetime([SURVEY_WAVES[w] for w in wave])
    return starts + pd.to_timedelta(np.floor(days), unit="D") + pd.to_timedelta(secs, unit="s")


def _completion_month(survey_m: int, release_months: tuple, rng: np.random.Generator) -> int:
    """A results-release month within the last 26 months, weighted towards recent cohorts."""
    options = [m for m in range(survey_m - 26, survey_m) if (m % 12) + 1 in release_months]
    w = np.array([1.0 + (m - (survey_m - 26)) / 26 for m in options])        # recent graduates respond more
    return int(rng.choice(options, p=w / w.sum()))


def generate_graduates(n_graduates: int = N_GRADUATES, n_final_year: int = N_FINAL_YEAR,
                       seed: int = SEED, defects: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (raw_df, truth_df, defect_log).

    raw_df   - exact raw questionnaire format (schema.RAW_FIELDS columns), with realistic survey behaviour
    truth_df - the model's features as observed (self-ratings etc.), the TRUE latent values, true event time
               and the observed outcome (no injected defects) - for tests
    """
    rng = np.random.default_rng(seed)
    n_pool = int(np.ceil(n_graduates * 1.9))           # graduates invited; non-response decides who answers
    n = n_pool + n_final_year
    status = np.array(["graduated"] * n_pool + ["final_year"] * n_final_year)
    is_grad = status == "graduated"

    # -- who they are --------------------------------------------------------------
    z = rng.normal(size=n)                                               # latent ability
    uni_idx = rng.choice(len(UNIVERSITY_SPECS), size=n, p=[u[2] for u in UNIVERSITY_SPECS])
    uni_type = np.array([UNIVERSITY_SPECS[i][1] for i in uni_idx])
    names = []
    for i in uni_idx:
        canon, _, _, _, variants, typos = UNIVERSITY_SPECS[i]
        r = rng.random()
        if typos and r < P_NAME_TYPO:
            names.append(str(rng.choice(typos)))
        elif variants and r < P_NAME_TYPO + P_NAME_VARIANT:
            names.append(str(rng.choice(variants)))
        else:
            names.append(canon)
    degree_type = np.array([rng.choice(list(DEGREE_PROBS_BY_TYPE[t]), p=list(DEGREE_PROBS_BY_TYPE[t].values()))
                            for t in uni_type])
    degree_group = np.array([schema.DEGREE_GROUPS[d] for d in degree_type])
    gender = rng.choice(schema.GENDER, size=n, p=[0.66, 0.32, 0.02])
    grad_age = np.array([rng.choice(schema.AGE_GROUP, p=AGE_PROBS_BY_TYPE[t]) for t in uni_type])
    age_group = np.where(is_grad, grad_age, rng.choice(schema.AGE_GROUP, size=n, p=[0.55, 0.40, 0.05, 0.00]))

    # -- academic --------------------------------------------------------------------
    true_gpa = np.clip(3.0 + 0.35 * z + rng.normal(0, 0.3, n) + np.vectorize(GPA_SHIFT_BY_TYPE.get)(uni_type),
                       1.8, 4.0)
    rounding = rng.choice(list(GPA_ROUNDING), size=n, p=list(GPA_ROUNDING.values()))
    reported = np.clip(true_gpa + 0.03, 0, 4.0)                          # mild self-report optimism
    reported = np.where(rounding == "tenth", np.round(reported, 1),
                        np.where(rounding == "half", np.round(reported * 2) / 2, np.round(reported, 2)))
    uses_class = (uni_type == "foreign_affiliated") & (rng.random(n) < 0.7)
    gpa_obs = np.where(uses_class, np.nan, reported)
    degree_class = np.where(uses_class, [_gpa_to_class(g) for g in true_gpa], None)
    skip_gpa = ~uses_class & (rng.random(n) < np.where(true_gpa < 2.7, 0.10, 0.02))   # weaker students skip more
    gpa_obs = np.where(skip_gpa, np.nan, gpa_obs)

    # -- experience ------------------------------------------------------------------------
    internship_completed = rng.random(n) < np.vectorize(P_INTERNSHIP_BY_TYPE.get)(uni_type)
    second = internship_completed & (rng.random(n) < 0.15 + 0.05 * (z > 1))
    internship_count = np.where(internship_completed, 1 + second, 0)
    first_len = rng.choice([6, 3, 12, 4, 5, 8, 9], size=n, p=[0.55, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05])
    internship_months = np.where(internship_completed,
                                 np.clip(first_len + second * rng.choice([3, 6], size=n), 1, 24), 0)
    cert_shift = np.array([CERT_SHIFT_BY_DEGREE.get(d, 0.0) for d in degree_type])
    zero_certs = rng.random(n) < 1 / (1 + np.exp(-(0.3 - 0.6 * z - cert_shift)))
    certifications_count = np.where(zero_certs, 0,
                                    np.clip(1 + rng.poisson(np.exp(-0.2 + 0.4 * z + cert_shift)), 1, 12))
    projects = rng.negative_binomial(3, 3 / (3 + np.exp(0.9 + 0.35 * z)))
    heap = (projects >= 7) & (rng.random(n) < 0.5)
    project_count = np.clip(np.where(heap, np.where(projects < 8, 5, 10), projects), 0, 25)

    # -- candidate predictors (extended feature set) ------------------------------------------------
    style = rng.normal(0, STYLE_SD, n)
    has_portfolio = rng.random(n) < 1 / (1 + np.exp(-(0.3 + z)))
    prep_latent = -0.6 * z + rng.normal(0, 0.8, n)
    prep_cuts = np.quantile(prep_latent, [0.10, 0.30, 0.65, 0.90, 0.97])        # 10/20/35/25/7/3 %
    prep_start = np.array(schema.PREP_START)[np.searchsorted(prep_cuts, prep_latent)]
    prep_years = np.vectorize(schema.PREP_YEARS.get)(prep_start).astype(float)
    no_events = rng.random(n) < 1 / (1 + np.exp(-(0.4 - 0.7 * z)))
    hackathon_count = np.where(no_events, 0, np.clip(1 + rng.poisson(np.exp(0.2 + 0.4 * z)), 1, 20))
    open_source = rng.random(n) < 1 / (1 + np.exp(-(-2.2 + 0.7 * z)))
    self_learning_freq = _likert_obs(3.2 + 0.5 * z + style + rng.normal(0, 0.7, n), rng)
    sdlc_familiarity = _likert_obs(3.3 + 0.5 * z + style + rng.normal(0, 0.7, n), rng)

    # -- skills: true level, then self-report -----------------------------------------------
    straight = rng.random(n) < P_STRAIGHT_LINE
    tech_true, tech_obs = {}, {}
    for area, spec in schema.SKILL_AREAS.items():
        deg_off = np.array([DEGREE_AREA_OFFSET.get(d, {}).get(area, 0.0) for d in degree_type])
        area_level = 3.0 + AREA_OFFSET[area] + deg_off + 0.7 * z + rng.normal(0, 0.4, n)
        for key in spec["skills"]:
            uni_off = np.array([UNI_ITEM_OFFSET[t].get(key, 0.0) for t in uni_type])
            true_item = area_level + uni_off + rng.normal(0, 0.5, n)
            tech_true[key] = true_item
            tech_obs[f"tech_{key}"] = _likert_obs(true_item + SELF_RATING_INFLATION["technical"] + style
                                                  - GENDER_RATING_GAP * (gender == "female")
                                                  + rng.normal(0, 0.5, n), rng)
    soft_level = 3.4 + 0.4 * z + rng.normal(0, 0.4, n)
    soft_true = {s.key: soft_level + rng.normal(0, 0.5, n) for s in schema.SOFT_SKILLS}
    soft_obs = {s.column: _likert_obs(soft_true[s.key] + SELF_RATING_INFLATION["soft"] + style
                                      + rng.normal(0, 0.5, n), rng) for s in schema.SOFT_SKILLS}
    tech_df, soft_df = pd.DataFrame(tech_obs), pd.DataFrame(soft_obs)
    for block in (tech_df, soft_df):                                       # straight-liners
        flat = np.clip(np.rint(block.mean(axis=1)), 1, 5).astype(int)
        block.loc[straight] = np.repeat(flat[straight].to_numpy()[:, None], block.shape[1], axis=1)

    truth = pd.DataFrame({
        "true_gpa": true_gpa, "gpa": np.where(uses_class, true_gpa, reported),
        "degree_group": degree_group, "university_type": uni_type,
        "internship_months": internship_months, "certifications_count": certifications_count,
        "project_count": project_count,
        "true_technical": pd.DataFrame(tech_true).mean(axis=1).to_numpy(),
        "true_soft": pd.DataFrame(soft_true).mean(axis=1).to_numpy(),
        "technical_mean": tech_df.mean(axis=1).to_numpy(), "soft_mean": soft_df.mean(axis=1).to_numpy(),
        "gender": gender, "respondent_status": status,
        "true_portfolio": has_portfolio.astype(float), "prep_years": prep_years,
        "hackathon_count": hackathon_count, "true_open_source": open_source.astype(float),
        "has_portfolio": has_portfolio.astype(float), "open_source": open_source.astype(float),
        "self_learning_freq": self_learning_freq, "sdlc_familiarity": sdlc_familiarity,
    })

    # -- dates -----------------------------------------------------------------------------------
    survey_ts = _survey_timestamps(n, rng)
    survey_m = np.array([_month_index(t) for t in survey_ts])
    release = [UNIVERSITY_SPECS[i][3] for i in uni_idx]
    grad_m = np.array([_completion_month(s, r, rng) for s, r in zip(survey_m, release)])
    follow_up = survey_m - grad_m
    early_cohort = grad_m < _month_index(pd.Timestamp(EARLY_COHORT_BEFORE + "-01"))

    # -- outcome ----------------------------------------------------------------------------------
    eta = linear_predictor(truth) + EARLY_COHORT_EFFECT * early_cohort
    frailty = rng.gamma(1 / FRAILTY_VARIANCE, FRAILTY_VARIANCE, n) if FRAILTY_VARIANCE > 0 else np.ones(n)
    t_true = WEIBULL_SCALE * (rng.exponential(size=n) / (np.exp(eta) * frailty)) ** (1 / WEIBULL_SHAPE)
    never_logit = np.log(NEVER_IT_BASE / (1 - NEVER_IT_BASE)) - 0.8 * (truth["true_technical"] - 3.0)
    never_it = rng.random(n) < 1 / (1 + np.exp(-never_logit))
    t_true = np.where(never_it, np.inf, t_true)
    pre_grad = ~never_it & (rng.random(n) < P_JOB_BEFORE_GRADUATION * np.exp(0.5 * eta))
    t_month = np.where(pre_grad, -rng.integers(0, 4, n), np.rint(np.minimum(t_true, 999)).astype(int))

    not_in_market = is_grad & (rng.random(n) < P_NOT_IN_MARKET)
    event = is_grad & ~not_in_market & (t_month <= follow_up)
    job_m = np.where(event, grad_m + t_month, -1)

    # Recall error: job start remembered as a round number of months; convocation month given instead of results.
    rounded = event & (t_month > 2) & (rng.random(n) < P_RECALL_ROUNDING)
    nearest = ROUNDED_MONTHS[np.abs(t_month[:, None] - ROUNDED_MONTHS[None, :]).argmin(axis=1)]
    job_m = np.where(rounded, np.minimum(grad_m + nearest, survey_m), job_m)
    confused = rng.random(n) < P_COMPLETION_CONFUSION
    grad_reported = np.where(confused, np.minimum(grad_m + rng.integers(2, 7, n), survey_m - 1), grad_m)

    employment_status = np.where(
        event, rng.choice(["full_time_it", "it_internship", "non_it", "further_studies"], size=n,
                          p=[0.85, 0.08, 0.03, 0.04]),
        rng.choice(["seeking_it", "non_it", "self_employed"], size=n, p=[0.60, 0.30, 0.10]))
    employment_status = np.where(not_in_market, rng.choice(schema.NOT_IN_MARKET_STATUSES, size=n), employment_status)
    employment_status = np.where(status == "final_year", "not_seeking", employment_status)

    # Q16a (competing risks): start month of the non-IT path, drawn independently of the IT-job process,
    # so on synthetic data the competing-risks analysis is illustrative only.
    rng_alt = np.random.default_rng(seed + ALT_SEED_OFFSET)
    alt_u, alt_answered = rng_alt.random(n), rng_alt.random(n) < P_ALT_DATE_ANSWERED
    on_alt = is_grad & ~event & np.isin(employment_status, schema.ALT_PATH_STATUSES) & alt_answered
    alt_m = grad_m + np.floor(alt_u * (np.maximum(follow_up, 0) + 1)).astype(int)

    raw = pd.DataFrame({
        "respondent_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "survey_date": [t.strftime("%Y-%m-%d %H:%M:%S") for t in survey_ts],
        "age_group": age_group,
        "gender": gender,
        "university_name": names,
        "degree_type": degree_type,
        "respondent_status": status,
        "graduation_date": [_month_str(m) if g else None for m, g in zip(grad_reported, is_grad)],
        "expected_graduation_date": [None if g else _month_str(
            min((m for m in range(s + 1, s + 13) if (m % 12) + 1 in r), default=s + 6))
            for s, g, r in zip(survey_m, is_grad, release)],
        "gpa": gpa_obs,
        "degree_class": degree_class,
        "industry_modules": [rng.choice(["yes", "no", "not_sure"], p=INDUSTRY_MODULES_BY_TYPE[t]) for t in uni_type],
        "certifications_count": certifications_count,
        "internship_completed": np.where(internship_completed, "yes", "no"),
        "internship_count": internship_count,
        "internship_months": internship_months,
        "project_count": project_count,
        "has_portfolio": np.where(has_portfolio, "yes", "no"),
        "prep_start": prep_start,
        "self_learning_freq": self_learning_freq,
        "hackathon_count": hackathon_count,
        "open_source": np.where(open_source, "yes", "no"),
        "sdlc_familiarity": sdlc_familiarity,
        "employment_status": employment_status,
        "alt_path_start_date": [_month_str(m) if a else None for m, a in zip(alt_m, on_alt)],
        "ever_it_job": np.where(is_grad, np.where(event, "yes", "no"), None),
        "first_job_start_date": [_month_str(m) if e else None for m, e in zip(job_m, event)],
        **tech_df, **soft_df,
    })[[f.name for f in schema.RAW_FIELDS]]

    # Rushed respondents click skill ratings at random.
    careless = rng.random(n) < P_CARELESS
    skill_cols = list(schema.TECH_COLUMNS + schema.SOFT_COLUMNS)
    raw.loc[careless, skill_cols] = rng.integers(1, 6, (int(careless.sum()), len(skill_cols)))

    # Observed outcome, as the cleaning step will reconstruct it from the REPORTED dates.
    rep_months = np.where(event, job_m - grad_reported, survey_m - grad_reported).astype(float)
    follow_up_rep = (survey_m - grad_reported).astype(float)
    obs_duration = np.where(event & (rep_months <= 0), schema.JOB_BEFORE_GRADUATION_MONTHS, rep_months)

    # Non-response bias: employed graduates are more likely to answer; keep the first n_graduates who do.
    p_respond = np.where(event, RESPONSE_PROB["employed"], RESPONSE_PROB["searching"])
    responded = np.flatnonzero(is_grad & (rng.random(n) < p_respond))
    if len(responded) < n_graduates:                                    # tiny pools only
        extra = np.setdiff1d(np.flatnonzero(is_grad), responded)[: n_graduates - len(responded)]
        responded = np.sort(np.concatenate([responded, extra]))
    keep = np.concatenate([responded[:n_graduates], np.flatnonzero(~is_grad)])
    raw = raw.iloc[keep].reset_index(drop=True)
    truth = truth.iloc[keep].reset_index(drop=True)
    sel = lambda a: np.asarray(a)[keep]  # noqa: E731
    event, t_month, t_true, pre_grad, never_it = sel(event), sel(t_month), sel(t_true), sel(pre_grad), sel(never_it)
    not_in_market, early_cohort, eta, is_grad = sel(not_in_market), sel(early_cohort), sel(eta), sel(is_grad)
    obs_duration, follow_up_rep = sel(obs_duration), sel(follow_up_rep)
    n = len(keep)
    raw["respondent_id"] = [f"S{i:04d}" for i in range(1, n + 1)]

    # Survey fatigue: later blocks are skipped more often (soft skills come after technical skills).
    for cols, p in ((list(schema.TECH_COLUMNS), 0.012), (list(schema.SOFT_COLUMNS), 0.025)):
        mask = rng.random((n, len(cols))) < p
        raw[cols] = raw[cols].astype(float).mask(mask)
    truth["technical_mean"] = raw[list(schema.TECH_COLUMNS)].mean(axis=1).to_numpy()
    truth["soft_mean"] = raw[list(schema.SOFT_COLUMNS)].mean(axis=1).to_numpy()
    truth["respondent_id"] = raw["respondent_id"]
    truth["in_market"] = is_grad & ~not_in_market
    truth["never_it"] = never_it
    truth["early_cohort"] = early_cohort
    truth["follow_up_months"] = follow_up_rep
    truth["true_time"] = np.where(pre_grad, 0.0, t_true)
    truth["duration_months"] = obs_duration
    truth["event"] = event.astype(int)
    truth["linear_predictor"] = eta

    defect_log: list[dict] = []
    if defects:
        raw, defect_log = _inject_defects(raw, rng)
    return raw, truth, defect_log


def realism_scorecard(truth: pd.DataFrame) -> list[dict]:
    """Compare the generated graduates with published Sri Lankan figures (ANCHORS)."""
    grads = truth[truth["respondent_status"] == "graduated"]
    market = truth[truth["in_market"]]
    employed = market[market["event"] == 1]
    after = employed[(employed["duration_months"] > schema.JOB_BEFORE_GRADUATION_MONTHS)
                     & (employed["follow_up_months"] >= 12)]
    values = {
        "female_share_pct": 100 * (grads["gender"] == "female").mean(),
        "mean_months_after_completion": after["duration_months"].mean(),
        "employed_before_completion_pct": 100 * (employed["duration_months"]
                                                 <= schema.JOB_BEFORE_GRADUATION_MONTHS).mean(),
        "employed_at_survey_pct": 100 * market["event"].mean(),
    }
    out = []
    for key, (target, lo, hi, source) in ANCHORS.items():
        v = float(values[key])
        out.append({"measure": key, "synthetic": round(v, 1), "published": target, "accepted_range": [lo, hi],
                    "within_range": bool(lo <= v <= hi), "source": source})
    return out


def _inject_defects(raw: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    """Add data-entry problems so the cleaning code is exercised. Every defect is logged."""
    raw = raw.copy()
    log: list[dict] = []
    grads = raw.index[raw["respondent_status"] == "graduated"].tolist()
    pool = iter(rng.permutation(grads).tolist())

    def hit(kind: str, column: str | None, value, detail: str):
        i = next(pool)
        if column is not None:
            raw[column] = raw[column].astype(object)
            raw.at[i, column] = value
        log.append({"respondent_id": raw.at[i, "respondent_id"], "defect": kind, "column": column,
                    "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else value,
                    "detail": detail})
        return i

    for _ in range(max(5, round(P_MISSING_GENDER * len(raw)))):
        hit("missing", "gender", None, "gender skipped")
    for _ in range(max(1, round(P_MISSING_AGE * len(raw)))):
        hit("missing", "age_group", None, "age group skipped")
    hit("missing", "degree_type", None, "degree programme skipped")

    hit("out_of_range", "gpa", 4.3, "GPA above 4.0")
    hit("out_of_range", "gpa", 34.0, "GPA typed without decimal point")
    hit("out_of_range", "tech_python", 6, "Likert value 6")
    hit("out_of_range", "soft_teamwork", 0, "Likert value 0")
    hit("out_of_range", "certifications_count", 99, "implausible certification count")
    hit("out_of_range", "internship_months", 48, "internship months above 36")
    hit("out_of_range", "project_count", -1, "negative project count")

    i = hit("inconsistent", "graduation_date", None, "graduation date after survey date")
    raw.at[i, "graduation_date"] = _month_str(_month_index(pd.Timestamp(raw.at[i, "survey_date"])) + 2)
    log[-1]["value"] = raw.at[i, "graduation_date"]
    i = hit("inconsistent", "first_job_start_date", "2006-03", "job start year typo (2006)")
    raw.at[i, "ever_it_job"] = "yes"
    for _ in range(2):
        i = hit("inconsistent", "first_job_start_date", None, "ever_it_job = yes but no start month")
        raw.at[i, "ever_it_job"] = "yes"

    # Duplicate submissions (pressing Submit twice: identical answers, new row id)
    dup_src = rng.choice(raw.index, size=max(6, round(DUPLICATE_RATE * len(raw))), replace=False)
    dups = raw.loc[dup_src].copy()
    dups["respondent_id"] = [f"S{len(raw) + k + 1:04d}" for k in range(len(dups))]
    for src, new_id in zip(raw.loc[dup_src, "respondent_id"], dups["respondent_id"]):
        log.append({"respondent_id": new_id, "defect": "duplicate", "column": None, "value": src,
                    "detail": f"duplicate submission of {src}"})
    raw = pd.concat([raw, dups], ignore_index=True)
    raw = raw.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    return raw, log


# ---------------------------------------------------------------------------
# Employers
# ---------------------------------------------------------------------------
EMPLOYER_SKILL_MEANS = {
    "programming_fundamentals": 4.5, "oop": 4.3, "java": 3.5, "python": 3.6, "javascript": 3.6,
    "sql_databases": 4.2, "git": 4.0, "software_testing": 3.4, "rest_apis": 3.6, "web_development": 3.5,
    "cloud_computing": 3.3, "linux": 3.0, "docker": 2.9, "devops_cicd": 2.9, "cybersecurity": 3.0,
    "data_analytics": 3.1, "ai_ml": 3.1,
    "communication": 4.6, "teamwork": 4.4, "problem_solving": 4.7, "critical_thinking": 4.2,
    "adaptability": 4.1, "time_management": 4.0, "professionalism": 4.2, "self_learning": 4.4,
    "leadership": 3.2,
}
EMPLOYER_FACTOR_MEANS = {"degree_qualification": 3.8, "gpa": 3.2, "technical_skills": 4.6,
                         "internship_experience": 4.2, "personal_projects": 4.0, "certifications": 3.1,
                         "portfolio": 3.7, "communication": 4.4, "interview_performance": 4.6}
# What different kinds of organisation emphasise (added to the means above).
ORG_PROFILES = {
    "banking": {"gpa": .6, "degree_qualification": .5, "sql_databases": .4, "java": .3, "cybersecurity": .3,
                "portfolio": -.4, "personal_projects": -.3},
    "government_it": {"degree_qualification": .6, "gpa": .5, "certifications": .3, "portfolio": -.5},
    "startup": {"portfolio": .6, "personal_projects": .6, "javascript": .5, "web_development": .4, "cloud_computing": .3,
                "gpa": -.6, "degree_qualification": -.5, "self_learning": .3},
    "software_dev": {"technical_skills": .2, "git": .3, "oop": .2, "software_testing": .2, "personal_projects": .2},
    "telecom": {"linux": .6, "cloud_computing": .3, "cybersecurity": .4, "devops_cicd": .3},
    "fintech": {"sql_databases": .3, "cybersecurity": .4, "cloud_computing": .2, "rest_apis": .3},
    "bpo_kpo": {"communication": .3, "data_analytics": .3, "time_management": .3, "ai_ml": -.4},
    "it_consulting": {"communication": .2, "data_analytics": .2, "professionalism": .3},
}
# Organisation size depends on the kind of organisation (order: <50, 50-200, 201-500, 500+).
ORG_SIZE_PROBS = {
    "software_dev": [.30, .35, .15, .20], "it_consulting": [.35, .35, .15, .15], "fintech": [.40, .35, .15, .10],
    "banking": [.00, .10, .25, .65], "telecom": [.00, .10, .20, .70], "bpo_kpo": [.10, .30, .30, .30],
    "government_it": [.15, .40, .25, .20], "startup": [.85, .15, .00, .00], "other": [.40, .35, .15, .10],
}
# Large employers screen on paper qualifications; small ones look at what candidates have built.
SIZE_PROFILES = {"<50": {"portfolio": .3, "personal_projects": .3, "degree_qualification": -.3, "gpa": -.2},
                 "500+": {"degree_qualification": .3, "gpa": .3, "certifications": .2, "portfolio": -.2}}
P_EMPLOYER_DROPOUT = 0.04   # answered the hiring factors, then left before the skills grid


def generate_employers(n: int = N_EMPLOYERS, seed: int = SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    org_type = rng.choice(schema.EMPLOYER_ORG_TYPES, size=n, p=[0.35, 0.12, 0.10, 0.10, 0.06, 0.08, 0.04, 0.12, 0.03])
    org_size = np.array([rng.choice(schema.EMPLOYER_ORG_SIZES, p=ORG_SIZE_PROBS[o]) for o in org_type])
    leniency = rng.normal(0, 0.3, n)                                  # some raters score everything higher
    straight = rng.random(n) < 0.05
    dropout = rng.random(n) < P_EMPLOYER_DROPOUT
    df = pd.DataFrame({
        "employer_id": [f"E{i:03d}" for i in range(1, n + 1)],
        "org_type": org_type,
        "org_size": org_size,
    })

    def rate(key: str, mean: float) -> np.ndarray:
        shift = np.array([ORG_PROFILES.get(o, {}).get(key, 0.0) + SIZE_PROFILES.get(s, {}).get(key, 0.0)
                          for o, s in zip(org_type, org_size)])
        v = _likert_obs(mean + shift + leniency + rng.normal(0, 0.6, n), rng).astype(float)
        v = np.where(straight, 4, v)
        return np.where(rng.random(n) < 0.03, np.nan, v)               # occasional skipped rating

    for f, m in EMPLOYER_FACTOR_MEANS.items():
        df[f"factor_{f}"] = rate(f, m)
    for s in schema.SKILL_VOCABULARY:
        df[f"imp_{s.key}"] = np.where(dropout, np.nan, rate(s.key, EMPLOYER_SKILL_MEANS[s.key]))
    return df[[f.name for f in schema.EMPLOYER_FIELDS]]


# ---------------------------------------------------------------------------
# Job ads (as a person would code them by hand: natural skill names)
# ---------------------------------------------------------------------------
JOB_ROLES = {   # role -> (share, title variants, P(skill mentioned))
    "se": (0.30, ["Associate Software Engineer", "Software Engineer - Graduate", "Junior Software Engineer"],
           {"programming_fundamentals": .5, "oop": .8, "java": .5, "javascript": .5, "sql_databases": .6, "git": .6,
            "rest_apis": .5, "web_development": .3, "python": .25, "cloud_computing": .2, "software_testing": .2}),
    "trainee": (0.15, ["Trainee Software Engineer", "Intern - Software Engineering", "Software Engineering Intern"],
                {"programming_fundamentals": .8, "oop": .7, "java": .4, "javascript": .4, "sql_databases": .5,
                 "git": .4, "web_development": .3}),
    "web": (0.12, ["Junior Web Developer", "Intern - Full Stack Developer", "Associate Frontend Engineer"],
            {"javascript": .95, "web_development": .9, "rest_apis": .5, "git": .5, "sql_databases": .4}),
    "qa": (0.12, ["Associate QA Engineer", "Trainee QA Engineer", "Intern - Quality Assurance"],
           {"software_testing": .95, "sql_databases": .4, "programming_fundamentals": .3, "java": .2, "rest_apis": .3}),
    "data": (0.10, ["Junior Data Analyst", "Trainee Data Engineer", "Associate Business Intelligence Analyst"],
             {"sql_databases": .9, "data_analytics": .9, "python": .6, "ai_ml": .2, "cloud_computing": .2}),
    "devops": (0.08, ["Associate DevOps Engineer", "Trainee Cloud Engineer"],
               {"linux": .8, "docker": .8, "devops_cicd": .9, "cloud_computing": .8, "git": .7, "python": .3}),
    "ml": (0.05, ["Graduate ML Engineer", "Associate AI Engineer"],
           {"python": .95, "ai_ml": .95, "data_analytics": .5, "sql_databases": .4, "cloud_computing": .3}),
    "sec": (0.04, ["Junior Security Analyst", "Trainee SOC Analyst"],
            {"cybersecurity": .95, "linux": .6, "cloud_computing": .3}),
    "ba": (0.04, ["Associate Business Analyst", "Trainee Business Analyst"],
           {"sql_databases": .4, "data_analytics": .5}),
}
SOFT_MENTION = {"communication": .65, "teamwork": .55, "problem_solving": .55, "critical_thinking": .15,
                "adaptability": .15, "time_management": .15, "professionalism": .1, "self_learning": .3,
                "leadership": .05}
# How a person coding ads might write each skill (all recognised by the cleaning step via schema aliases).
SURFACE_FORMS = {s.key: [s.label, *s.aliases] for s in schema.SKILL_VOCABULARY}
UNRECOGNISED_TOKENS = ["Jira", "MS Office", "Agile/Scrum", "Figma", "SAP", "C#", ".NET", "PHP", "Laravel",
                       "UI/UX", "English proficiency", "Power Platform"]   # logged as unknown by cleaning
ROLE_EXTRA_TOKENS = {"se": ["C#", ".NET", "Jira", "Agile/Scrum"], "trainee": ["C#", "PHP", "Agile/Scrum"],
                     "web": ["PHP", "Laravel", "Figma", "UI/UX"], "qa": ["Jira", "Agile/Scrum"],
                     "data": ["MS Office", "Power Platform"], "ba": ["MS Office", "Jira", "English proficiency"],
                     "devops": ["Jira", "Agile/Scrum"], "ml": ["MS Office", "Agile/Scrum"], "sec": ["MS Office", "Jira"]}
TITLE_SUFFIXES = [" - Colombo", " (Fresh Graduates)", " | Hybrid", " - Immediate Hiring", " (Remote)"]
P_REPOST = 0.04           # same vacancy posted again a few weeks later
# Invented company names (root + suffix); none refers to a real firm.
COMPANY_ROOTS = ["Bluecove", "Nuvara", "Kelvara", "Orvane", "Tamvi", "Silvaro", "Ambrel", "Corvin", "Velora",
                 "Ranvel", "Pelmora", "Quillon", "Marvex", "Duneline", "Hollowpeak", "Brightreef", "Coralpine",
                 "Tidewell", "Lanthor", "Vimara", "Ostrel", "Kavira"]
COMPANY_SUFFIXES = ["Technologies", "Labs", "Software", "Digital", "Systems", "Solutions", "IT", "Tech",
                    "Innovations"]
AD_DUPLICATE_RATE = 0.008 # the same posting recorded twice


def _ad_title(title: str, rng: np.random.Generator) -> str:
    """Titles as posters write them: suffixes, odd capitalisation, no fixed convention."""
    if rng.random() < 0.25:
        title += str(rng.choice(TITLE_SUFFIXES))
    r = rng.random()
    return title.upper() if r < 0.05 else title.lower() if r < 0.08 else title


def _ad_token(token: str, rng: np.random.Generator) -> str:
    r = rng.random()
    return token.lower() if r < 0.20 else token.upper() if r < 0.25 else token.title() if r < 0.35 else token


def company_names(n: int, seed: int = SEED + 11) -> list[str]:
    """n distinct invented company names, from their own random stream (so no other value changes)."""
    rng = np.random.default_rng(seed)
    roots, suffixes = rng.permutation(COMPANY_ROOTS), rng.permutation(COMPANY_SUFFIXES)
    R, S = len(roots), len(suffixes)
    # Each root is used once before any repeats, so the busiest posters (lowest indices) all look distinct.
    combos = [f"{roots[k % R]} {suffixes[(k // R + k) % S]}" for k in range(R * S)]
    if n > len(combos):
        combos += [f"{c} Lanka" for c in combos]
    return combos[:n]


def generate_job_ads(n: int = N_JOB_ADS, seed: int = SEED + 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_companies = max(60, n // 6)
    company = company_names(n_companies)
    roles = list(JOB_ROLES)
    picks = rng.choice(len(roles), size=n, p=[JOB_ROLES[r][0] for r in roles])
    firm_w = 1 / np.arange(1, n_companies + 1) ** 0.7                  # a few large firms post a lot
    firms = rng.choice(n_companies, size=n, p=firm_w / firm_w.sum()) + 1
    days = pd.date_range("2026-06-01", "2026-09-30", freq="D")
    day_w = np.where(days.dayofweek < 5, 1.0, 0.25) * np.where(days.month >= 8, 1.3, 1.0)   # weekday, busier Aug-Sep
    sources = ["linkedin", "topjobs", "company_site", "other"]
    width = max(3, len(str(n + 99)))
    rows = []
    for i, (k, firm) in enumerate(zip(picks, firms), 1):
        if rows and rng.random() < P_REPOST:                           # repost of an earlier vacancy
            prev = rows[int(rng.integers(len(rows)))]
            date = min(pd.Timestamp(prev["date"]) + pd.Timedelta(days=int(rng.integers(14, 43))), days[-1])
            repost = {**prev, "ad_id": f"J{i:0{width}d}", "date": date.strftime("%Y-%m-%d"),
                      "source": str(rng.choice(sources, p=[.45, .35, .15, .05]))}
            if not any(all(r[c] == repost[c] for c in ("title", "company", "date", "source", "skills")) for r in rows):
                rows.append(repost)
                continue
        role = roles[k]
        _, titles, probs = JOB_ROLES[role]
        keys = [s for s, p in {**probs, **SOFT_MENTION}.items() if rng.random() < p] or ["programming_fundamentals"]
        tokens = [_ad_token(str(rng.choice(SURFACE_FORMS[s])), rng) for s in keys]
        rng.shuffle(tokens)                                            # ads list requirements in no fixed order
        if rng.random() < 0.25:
            tokens.append(str(rng.choice(ROLE_EXTRA_TOKENS.get(role, UNRECOGNISED_TOKENS))))
        rows.append({"ad_id": f"J{i:0{width}d}", "title": _ad_title(str(rng.choice(titles)), rng),
                     "company": company[firm - 1],
                     "date": days[rng.choice(len(days), p=day_w / day_w.sum())].strftime("%Y-%m-%d"),
                     "source": str(rng.choice(sources, p=[.45, .35, .15, .05])),
                     "skills": "; ".join(tokens)})
    df = pd.DataFrame(rows)
    n_dup = max(4, round(AD_DUPLICATE_RATE * n))
    dups = df.sample(n_dup, random_state=int(rng.integers(1 << 31))).copy()   # the same ad recorded twice
    dups["ad_id"] = [f"J{n + j:0{width}d}" for j in range(1, n_dup + 1)]
    return pd.concat([df, dups], ignore_index=True)[[f.name for f in schema.JOB_AD_FIELDS]]


def university_lookup() -> pd.DataFrame:
    """Canonical names and their common variants -> type and institution. Rare typos are deliberately NOT
    included."""
    rows = [(v, t, name) for name, t, _, _, variants, _ in UNIVERSITY_SPECS for v in [name, *variants]]
    return pd.DataFrame(rows, columns=["university_name", "university_type", "institution"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def write_all(raw_dir: Path = RAW_DIR, reference_dir: Path = REFERENCE_DIR) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    graduates, truth, defects = generate_graduates()
    files = {
        "graduates": raw_dir / "graduates_SYNTHETIC.csv",
        "employers": raw_dir / "employers_SYNTHETIC.csv",
        "job_ads": raw_dir / "job_ads_SYNTHETIC.csv",
        "university_types": reference_dir / "university_types_SYNTHETIC.csv",
        "manifest": raw_dir / "synthetic_manifest_SYNTHETIC.json",
    }
    # Questionnaire-study items are added afterwards from their own random streams (core/synthetic_survey.py),
    # so every column above is unchanged by them.
    from core import synthetic_survey
    graduates = synthetic_survey.add_graduate_survey(graduates)
    graduates.to_csv(files["graduates"], index=False)
    synthetic_survey.add_employer_survey(generate_employers()).to_csv(files["employers"], index=False)
    generate_job_ads().to_csv(files["job_ads"], index=False)
    university_lookup().to_csv(files["university_types"], index=False)

    in_market = truth[truth["in_market"]]
    summary = {
        "seed": SEED, "weibull_shape": WEIBULL_SHAPE, "weibull_scale_months": WEIBULL_SCALE,
        "true_log_hazard_ratios": TRUE_LOG_HAZARD_RATIOS, "gpa_threshold_bonus": GPA_THRESHOLD_BONUS,
        "early_cohort_effect": EARLY_COHORT_EFFECT, "frailty_variance": FRAILTY_VARIANCE,
        "never_it_base": NEVER_IT_BASE, "centre": CENTRE,
        "rows_written": len(graduates),
        "graduates": int((truth["respondent_status"] == "graduated").sum()),
        "final_year": int((truth["respondent_status"] == "final_year").sum()),
        "graduates_in_market": int(len(in_market)),
        "events": int(in_market["event"].sum()),
        "censored": int((in_market["event"] == 0).sum()),
        "censored_pct": round(100 * (1 - in_market["event"].mean()), 1),
        "never_it_in_market": int(in_market["never_it"].sum()),
        "realism_scorecard": realism_scorecard(truth),
        "job_before_graduation": int(((in_market["event"] == 1)
                                      & (in_market["duration_months"] == schema.JOB_BEFORE_GRADUATION_MONTHS)).sum()),
        "defects": defects,
    }
    files["manifest"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"files": {k: str(v) for k, v in files.items()}, **{k: v for k, v in summary.items() if k != "defects"},
            "defect_count": len(defects)}


if __name__ == "__main__":
    print(json.dumps(write_all(), indent=2))
