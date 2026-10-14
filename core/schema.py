"""Feature schema: the single source of truth for every field, skill and model feature.

Built against graduate questionnaire v2 and employer questionnaire v2
(docs/questionnaires/questionnaire_revisions.md). Item numbers below refer to those versions.

Run `python -m core.schema` to regenerate docs/data_dictionary.md from this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core import survey_items

QUESTIONNAIRE_VERSION = {"graduate": "v2", "employer": "v2"}

# ---------------------------------------------------------------------------
# Outcome-construction config (documented in docs/data_cleaning_log.md)
# ---------------------------------------------------------------------------
JOB_BEFORE_GRADUATION_MONTHS = 0.5   # duration assigned when first IT role started on/before completion
PREDICTION_HORIZON_MONTHS = 24       # survival curves are reported from 0 to this horizon
EPV_WARNING_THRESHOLD = 10           # warn if events-per-variable falls below this
MIN_ITEMS_ANSWERED_FRACTION = 2 / 3  # a skill block mean needs >= this share of its items answered, else missing
# Cell values read as missing in the survey files. pandas' default list also treats the text "None", "N/A", "n/a" and
# "NA" as missing, but "None" is a real answer option (Q26, Q28, employer Q6) and "n/a" a real comment, so they are
# kept as text. Pass with keep_default_na=False.
SURVEY_NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
                    "<NA>", "NULL", "NaN", "nan", "null"]

# ---------------------------------------------------------------------------
# Field specification
# ---------------------------------------------------------------------------
Kind = Literal["id", "categorical", "binary", "integer", "numeric", "likert", "month", "timestamp", "text", "multi"]
Role = Literal["id", "model_input", "candidate", "outcome", "fairness", "descriptive", "survey", "quality"]
AppliesTo = Literal["all", "graduated", "final_year"]


@dataclass(frozen=True)
class FieldSpec:
    name: str                       # raw CSV column name
    item: str                       # questionnaire item ("Q5a", "Q18.4") or "auto"/"manual"
    kind: Kind
    role: Role
    description: str
    categories: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    nullable: bool = False
    applies_to: AppliesTo = "all"


# ---------------------------------------------------------------------------
# Categorical vocabularies
# ---------------------------------------------------------------------------
RESPONDENT_STATUS = ("graduated", "final_year")
GENDER = ("male", "female", "prefer_not_to_say")
AGE_GROUP = ("20-22", "23-25", "26-28", "28+")
DEGREE_TYPES = (
    "computer_science", "information_technology", "software_engineering", "information_systems",
    "cyber_security", "data_science", "ai_ml", "other",
)
# Q4 has 8 programmes - too many levels for ~150 events. Collapsed for the model:
DEGREE_GROUPS = {
    "computer_science": "computing", "software_engineering": "computing",
    "information_technology": "it_is", "information_systems": "it_is",
    "cyber_security": "specialised", "data_science": "specialised", "ai_ml": "specialised",
    "other": "specialised",
}
UNIVERSITY_TYPES = ("state", "non_state", "foreign_affiliated")  # mapped from Q3 via data/reference/university_types.csv
DEGREE_CLASSES = ("first", "second_upper", "second_lower", "third_or_pass")
# GPA-equivalent for degrees without a GPA (Q6a). Approximate class boundaries on a 4.0 scale.
DEGREE_CLASS_TO_GPA = {"first": 3.85, "second_upper": 3.50, "second_lower": 3.15, "third_or_pass": 2.50}
EMPLOYMENT_STATUS = (
    "full_time_it", "it_internship", "seeking_it", "non_it", "further_studies", "self_employed",
    "not_seeking",
)
# Graduates with no IT role and one of these statuses are not in the IT job market -> excluded from training
NOT_IN_MARKET_STATUSES = ("further_studies", "not_seeking")
# Competing-risks analysis only (core/competing.py): graduates with no IT role who moved onto another path.
ALT_PATH_STATUSES = ("non_it", "self_employed", "further_studies")
# Questions added after the first questionnaire version: older exports without them are still accepted.
LATE_OPTIONAL_FIELDS = ("alt_path_start_date",)
# Q20: when the respondent started preparing for IT employment -> years of preparation before completion
PREP_START = ("before_university", "first_year", "second_third_year", "final_year", "after_graduation",
              "not_started")
PREP_YEARS = {"before_university": 4, "first_year": 3, "second_third_year": 2, "final_year": 1,
              "after_graduation": 0, "not_started": 0}


# ---------------------------------------------------------------------------
# Skill vocabulary (shared by graduates, employers and job ads)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Skill:
    key: str                 # canonical name used everywhere
    kind: Literal["technical", "soft"]
    label: str               # display label (graduate wording)
    graduate_item: str       # graduate questionnaire item
    employer_item: str       # employer questionnaire item
    aliases: tuple[str, ...] = ()   # extra strings that map to this skill when coding job ads

    @property
    def column(self) -> str:
        return f"{'tech' if self.kind == 'technical' else 'soft'}_{self.key}"


_TECH = [
    ("programming_fundamentals", "Programming Fundamentals", ("programming", "coding")),
    ("oop", "Object-Oriented Programming", ("object oriented", "oop")),
    ("java", "Java", ("spring", "spring boot")),
    ("python", "Python", ("django", "flask")),
    ("javascript", "JavaScript", ("typescript", "react", "node.js", "angular")),
    ("sql_databases", "SQL / Database Management", ("sql", "mysql", "postgresql", "database")),
    ("git", "Git / GitHub", ("git", "version control", "github")),
    ("software_testing", "Software Testing", ("qa", "unit testing", "selenium", "test automation")),
    ("rest_apis", "REST APIs", ("rest", "api", "restful")),
    ("web_development", "Web Development", ("html", "css", "frontend", "full stack")),
    ("cloud_computing", "Cloud Computing", ("aws", "azure", "gcp", "cloud")),
    ("linux", "Linux / Command Line", ("linux", "bash", "shell")),
    ("docker", "Docker", ("containers", "kubernetes")),
    ("devops_cicd", "DevOps / CI-CD", ("ci/cd", "jenkins", "devops", "github actions")),
    ("cybersecurity", "Cybersecurity Fundamentals", ("security", "owasp")),
    ("data_analytics", "Data Analytics", ("power bi", "excel", "tableau", "analytics")),
    ("ai_ml", "Artificial Intelligence / Machine Learning", ("machine learning", "ai", "deep learning")),
]
_SOFT = [
    ("communication", "Communication Skills", ("communication",)),
    ("teamwork", "Teamwork", ("team player", "collaboration")),
    ("problem_solving", "Problem Solving", ("problem solving", "analytical")),
    ("critical_thinking", "Critical Thinking", ("critical thinking",)),
    ("adaptability", "Adaptability", ("adaptable", "flexible")),
    ("time_management", "Time Management", ("time management", "deadlines")),
    ("professionalism", "Professionalism", ("professional",)),
    ("self_learning", "Self-learning Ability", ("self-learner", "quick learner", "eager to learn")),
    ("leadership", "Leadership Ability", ("leadership",)),   # employer item Q9.9 added in v2
]
SKILL_VOCABULARY: tuple[Skill, ...] = tuple(
    [Skill(k, "technical", lab, f"Q18.{i}", f"Q8.{i}", al) for i, (k, lab, al) in enumerate(_TECH, 1)]
    + [Skill(k, "soft", lab, f"Q19.{i}", f"Q9.{i}", al) for i, (k, lab, al) in enumerate(_SOFT, 1)]
)
SKILLS_BY_KEY = {s.key: s for s in SKILL_VOCABULARY}
TECHNICAL_SKILLS = tuple(s for s in SKILL_VOCABULARY if s.kind == "technical")
SOFT_SKILLS = tuple(s for s in SKILL_VOCABULARY if s.kind == "soft")

SKILL_AREAS: dict[str, dict] = {
    "programming": {"label": "Programming",
                    "skills": ("programming_fundamentals", "oop", "java", "python", "javascript")},
    "web_apis": {"label": "Web & APIs", "skills": ("web_development", "rest_apis")},
    "data_ai": {"label": "Data & AI", "skills": ("sql_databases", "data_analytics", "ai_ml")},
    "cloud_devops": {"label": "Cloud & DevOps",
                     "skills": ("cloud_computing", "linux", "docker", "devops_cicd", "git")},
    "quality_security": {"label": "Quality & Security", "skills": ("software_testing", "cybersecurity")},
}
# Soft skills form one "area" for Quick mode and what-if.
SOFT_AREA = {"soft": {"label": "Soft skills", "skills": tuple(s.key for s in SOFT_SKILLS)}}


# ---------------------------------------------------------------------------
# Raw graduate fields (questionnaire v2)
# ---------------------------------------------------------------------------
_RAW: list[FieldSpec] = [
    FieldSpec("respondent_id", "auto", "id", "id", "Row identifier assigned at export"),
    FieldSpec("survey_date", "auto", "timestamp", "outcome",
              "Form submission timestamp; end of follow-up for graduates still searching"),
    FieldSpec("age_group", "Q1", "categorical", "descriptive", "Age group", AGE_GROUP, nullable=True),
    FieldSpec("gender", "Q2", "categorical", "fairness",
              "Gender - fairness check ONLY, never a model feature", GENDER, nullable=True),
    FieldSpec("university_name", "Q3", "text", "descriptive",
              "Institution name; mapped to university_type via data/reference/university_types.csv"),
    FieldSpec("degree_type", "Q4", "categorical", "model_input", "Degree programme", DEGREE_TYPES),
    FieldSpec("respondent_status", "Q5", "categorical", "outcome",
              "Graduated vs final-year (derived from Q5)", RESPONDENT_STATUS),
    FieldSpec("graduation_date", "Q5a", "month", "outcome",
              "Month of degree completion (final results released) - TIME ORIGIN",
              applies_to="graduated"),
    FieldSpec("expected_graduation_date", "Q5b", "month", "descriptive",
              "Expected month of degree completion", applies_to="final_year"),
    FieldSpec("gpa", "Q6", "numeric", "model_input", "GPA on a 4.0 scale", min=0.0, max=4.0, nullable=True),
    FieldSpec("degree_class", "Q6a", "categorical", "model_input",
              "Degree class if no GPA; converted via DEGREE_CLASS_TO_GPA", DEGREE_CLASSES, nullable=True),
    FieldSpec("industry_modules", "Q7", "categorical", "descriptive",
              "Degree included industry-oriented modules", ("yes", "no", "not_sure"), nullable=True),
    FieldSpec("certifications_count", "Q9", "integer", "model_input",
              "Number of professional IT certifications", min=0, max=20),
    FieldSpec("internship_completed", "Q11", "binary", "descriptive", "Completed an IT internship (yes/no)"),
    FieldSpec("internship_count", "Q11a", "integer", "descriptive",
              "Number of internships (0 if Q11 = No)", min=0, max=5),
    FieldSpec("internship_months", "Q12", "integer", "model_input",
              "Total months of IT internship experience", min=0, max=36),
    FieldSpec("project_count", "Q13", "integer", "model_input",
              "Personal/academic IT projects outside coursework", min=0, max=30),
    # Candidate predictors (v2 wording: "before you completed your degree"); used only in the extended feature set.
    FieldSpec("has_portfolio", "Q14", "binary", "candidate",
              "Had a technical portfolio (GitHub, website) before completing the degree", nullable=True),
    FieldSpec("prep_start", "Q20", "categorical", "candidate",
              "When they started preparing for IT employment", PREP_START, nullable=True),
    FieldSpec("self_learning_freq", "Q21", "likert", "candidate",
              "Self-learning outside coursework during the degree (1 = never, 5 = very frequently)",
              min=1, max=5, nullable=True),
    FieldSpec("hackathon_count", "Q23", "integer", "candidate",
              "Hackathons / coding competitions / technical events before completing the degree",
              min=0, max=30, nullable=True),
    FieldSpec("open_source", "Q24", "binary", "candidate",
              "Contributed to open-source projects before completing the degree", nullable=True),
    FieldSpec("sdlc_familiarity", "Q29", "likert", "candidate",
              "Familiarity with SDLC practices at degree completion (1-5)", min=1, max=5, nullable=True),
    FieldSpec("employment_status", "Q16", "categorical", "outcome",
              "Current status; used only to identify graduates not in the IT job market",
              EMPLOYMENT_STATUS, nullable=True),
    FieldSpec("alt_path_start_date", "Q16a", "month", "outcome",
              "If no IT role yet: start month of the current non-IT job, self-employment or further studies "
              "(optional; competing-risks analysis only)", nullable=True, applies_to="graduated"),
    FieldSpec("ever_it_job", "Q17", "binary", "outcome",
              "Has ever started a paid IT role (excl. degree internships) - EVENT INDICATOR",
              nullable=True, applies_to="graduated"),
    FieldSpec("first_job_start_date", "Q17a", "month", "outcome",
              "Start month of first paid IT role", nullable=True, applies_to="graduated"),
]
_RAW += [FieldSpec(s.column, s.graduate_item, "likert", "model_input",
                   f"{s.label} - self-rating at degree completion (1-5)", min=1, max=5, nullable=True)
         for s in SKILL_VOCABULARY]
RAW_FIELDS: tuple[FieldSpec, ...] = tuple(_RAW)
RAW_FIELDS_BY_NAME = {f.name: f for f in RAW_FIELDS}


def _survey_spec(name: str, item: str, kind: str, opts, description: str) -> FieldSpec:
    if kind == "rank":
        return FieldSpec(name, f"{item}.{opts[0][0]}", "integer", "survey", description, min=1, max=3, nullable=True)
    return FieldSpec(name, item, kind, "survey", description,
                     categories=tuple(k for k, _ in opts) if opts else None,
                     min=1 if kind == "likert" else None, max=5 if kind == "likert" else None, nullable=True)


# Questionnaire-study items (core/survey_items.py): analysed in core/survey_study.py, never model inputs.
# Optional in the input, so exports and files made before they were added are still accepted.
SURVEY_FIELDS: tuple[FieldSpec, ...] = tuple(_survey_spec(*i) for i in survey_items.GRADUATE_ITEMS)
EMPLOYER_SURVEY_FIELDS: tuple[FieldSpec, ...] = tuple(_survey_spec(*i) for i in survey_items.EMPLOYER_ITEMS)
# Data-quality item: an instructed-response row in the Q18 grid ("Attention check - please select 4 for this row").
# Respondents who answer it with anything other than ATTENTION_CHECK_ANSWER are excluded (core/clean.py).
ATTENTION_CHECK_ANSWER = 4
ATTENTION_FIELD = FieldSpec("attention_check", "Q18.attention", "likert", "quality",
                            "Attention check row in the Q18 grid (instructed answer: 4)", min=1, max=5, nullable=True)
GRADUATE_ALL_FIELDS = RAW_FIELDS + SURVEY_FIELDS + (ATTENTION_FIELD,)

TECH_COLUMNS = tuple(s.column for s in TECHNICAL_SKILLS)
SOFT_COLUMNS = tuple(s.column for s in SOFT_SKILLS)


# ---------------------------------------------------------------------------
# Derived features and the model feature set
# ---------------------------------------------------------------------------
DERIVED_FEATURES = {
    "gpa": "Q6, or DEGREE_CLASS_TO_GPA[Q6a] when Q6 is blank",
    "degree_group": "DEGREE_GROUPS[degree_type]",
    "university_type": "lookup of university_name in data/reference/university_types.csv",
    "technical_mean": "mean of the 17 Q18 items",
    "soft_mean": "mean of the 9 Q19 items",
    **{f"area_{a}": f"mean of {', '.join(v['skills'])}" for a, v in SKILL_AREAS.items()},
    "duration_months": "months from graduation_date to first_job_start_date (event) or survey_date (censored)",
    "event": "1 if ever_it_job, else 0",
    "has_portfolio / open_source": "yes -> 1, no -> 0 (extended feature set only)",
    "prep_years": "PREP_YEARS[prep_start]: years of preparation before completion (extended feature set only)",
    "gpa_3plus": "1 if GPA >= 3.0 else 0 (transformed feature set only)",
    "log_internship_months / log_certifications / log_projects": "log(1 + count) (transformed feature set only)",
}

# Pre-specified predictor sets. Categorical features expand to k-1 dummy columns.
CORE_FEATURES: dict[str, str] = {
    "gpa": "numeric",
    "degree_group": "categorical",        # 3 levels -> 2 columns
    "university_type": "categorical",     # 3 levels -> 2 columns
    "internship_months": "numeric",
    "certifications_count": "numeric",
    "project_count": "numeric",
    "technical_mean": "numeric",
    "soft_mean": "numeric",
}
EXTENDED_FEATURES: dict[str, str] = {
    **CORE_FEATURES,
    "has_portfolio": "numeric",           # 0/1
    "prep_years": "numeric",              # 0-4
    "self_learning_freq": "numeric",      # 1-5
    "hackathon_count": "numeric",
    "open_source": "numeric",             # 0/1
    "sdlc_familiarity": "numeric",        # 1-5
}
# Same information as CORE, with theory-motivated functional forms fixed in advance:
#   log(1 + count) for certifications, projects and internship months (diminishing returns; skewed counts)
#   an indicator for GPA >= 3.0 alongside GPA (employers' shortlisting cut-offs; signalling theory)
TRANSFORMED_FEATURES: dict[str, str] = {
    "gpa": "numeric",
    "gpa_3plus": "numeric",               # 0/1
    "degree_group": "categorical",
    "university_type": "categorical",
    "log_internship_months": "numeric",
    "log_certifications": "numeric",
    "log_projects": "numeric",
    "technical_mean": "numeric",
    "soft_mean": "numeric",
}
GPA_CUTOFF = 3.0
FEATURE_SETS = {"core": CORE_FEATURES, "extended": EXTENDED_FEATURES, "transformed": TRANSFORMED_FEATURES}
# Which set the model uses. The pre-registered rule in core/feature_sets.py (report:
# artifacts/feature_set_comparison.md) recommended "core". "extended" is used instead at the researcher's decision,
# so the preparation questions (Q14, Q20, Q21, Q23, Q24, Q29) feed the prediction. This is a DEVIATION from the
# pre-registered rule and is reported as such (docs/deviations.md). Re-check it when real data arrives.
FEATURE_SET = "extended"
MODEL_FEATURES: dict[str, str] = FEATURE_SETS[FEATURE_SET]
CATEGORICAL_LEVELS = {
    "degree_group": ("computing", "it_is", "specialised"),
    "university_type": UNIVERSITY_TYPES,
}


def model_parameter_count(features: dict | None = None) -> int:
    """Number of model columns after one-hot encoding (used for events-per-variable)."""
    features = MODEL_FEATURES if features is None else features
    return sum(len(CATEGORICAL_LEVELS[f]) - 1 if t == "categorical" else 1 for f, t in features.items())


# ---------------------------------------------------------------------------
# Actionable features (what-if)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Actionable:
    min: float
    max: float
    step: float
    label: str
    applies_to: AppliesTo = "all"
    max_change: float | None = None   # largest allowed change from the user's current value


ACTIONABLE_FEATURES: dict[str, Actionable] = {
    "internship_months": Actionable(0, 24, 1, "Months of internship experience"),
    "certifications_count": Actionable(0, 10, 1, "Professional certifications"),
    "project_count": Actionable(0, 15, 1, "Personal / academic projects"),
    # A final-year student's GPA can still move a little before completion.
    "gpa": Actionable(0.0, 4.0, 0.05, "GPA", applies_to="final_year", max_change=0.3),
    **{f"area_{a}": Actionable(1, 5, 0.5, f"{v['label']} skill rating") for a, v in SKILL_AREAS.items()},
    "area_soft": Actionable(1, 5, 0.5, "Soft skills rating"),
}
NON_ACTIONABLE = ("gpa (graduates)", "degree_group", "university_type", "gender")


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------
POST_EMPLOYMENT_FIELDS = frozenset({
    "salary", "job_title", "employer_name", "current_role", "employment_status",
    "ever_it_job", "first_job_start_date", "duration_months", "event", "survey_date",
})
FAIRNESS_ONLY_FIELDS = frozenset({"gender"})


def assert_no_leakage(features=None) -> None:
    feats = set(features if features is not None else MODEL_FEATURES)
    leaked = feats & (POST_EMPLOYMENT_FIELDS | FAIRNESS_ONLY_FIELDS)
    assert not leaked, f"Leakage: outcome/post-employment/fairness fields in model features: {sorted(leaked)}"


for _features in FEATURE_SETS.values():
    assert_no_leakage(_features)
assert len(TECHNICAL_SKILLS) == 17 and len(SOFT_SKILLS) == 9
assert sorted(k for a in SKILL_AREAS.values() for k in a["skills"]) == sorted(s.key for s in TECHNICAL_SKILLS)
assert model_parameter_count(CORE_FEATURES) <= 12


# ---------------------------------------------------------------------------
# Employer responses and job ads (raw formats)
# ---------------------------------------------------------------------------
EMPLOYER_ORG_TYPES = ("software_dev", "it_consulting", "fintech", "banking", "telecom", "bpo_kpo",
                      "government_it", "startup", "other")
EMPLOYER_ORG_SIZES = ("<50", "50-200", "201-500", "500+")
EMPLOYER_FACTORS = (   # Q7.1-Q7.9
    "degree_qualification", "gpa", "technical_skills", "internship_experience", "personal_projects",
    "certifications", "portfolio", "communication", "interview_performance",
)
EMPLOYER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("employer_id", "auto", "id", "id", "Row identifier"),
    FieldSpec("org_type", "Q1", "categorical", "descriptive", "Organisation type", EMPLOYER_ORG_TYPES),
    FieldSpec("org_size", "Q2", "categorical", "descriptive", "Organisation size", EMPLOYER_ORG_SIZES),
    *[FieldSpec(f"factor_{f}", f"Q7.{i}", "likert", "descriptive", f"Importance of {f} (1-5)", min=1, max=5,
                nullable=True) for i, f in enumerate(EMPLOYER_FACTORS, 1)],
    *[FieldSpec(f"imp_{s.key}", s.employer_item, "likert", "descriptive", f"Importance of {s.label} (1-5)",
                min=1, max=5, nullable=True) for s in SKILL_VOCABULARY],
)
EMPLOYER_ATTENTION_FIELD = FieldSpec("attention_check", "Q8.attention", "likert", "quality",
                                     "Attention check row in the Q8 grid (instructed answer: 4)", min=1, max=5,
                                     nullable=True)
EMPLOYER_ALL_FIELDS = EMPLOYER_FIELDS + EMPLOYER_SURVEY_FIELDS + (EMPLOYER_ATTENTION_FIELD,)
JOB_AD_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("ad_id", "manual", "id", "id", "Row identifier"),
    FieldSpec("title", "manual", "text", "descriptive", "Advertised job title"),
    FieldSpec("company", "manual", "text", "descriptive", "Hiring company"),
    FieldSpec("date", "manual", "month", "descriptive", "Date the ad was collected"),
    FieldSpec("source", "manual", "categorical", "descriptive", "Where the ad was found",
              ("linkedin", "topjobs", "company_site", "other")),
    FieldSpec("skills", "manual", "text", "descriptive",
              "Semicolon-separated SKILL_VOCABULARY keys mentioned in the ad"),
)


# ---------------------------------------------------------------------------
# Data dictionary generation
# ---------------------------------------------------------------------------
_ROLE_LABEL = {"id": "Identifier", "model_input": "Model feature (input)",
               "candidate": "Candidate predictor (extended set only)", "outcome": "Outcome construction",
               "fairness": "Fairness only", "descriptive": "Descriptive only",
               "survey": "Questionnaire study only"}


def _range(f: FieldSpec) -> str:
    if f.categories:
        return ", ".join(f"`{c}`" for c in f.categories)
    if f.min is not None:
        return f"{f.min:g}-{f.max:g}"
    return {"month": "YYYY-MM", "timestamp": "ISO date-time", "binary": "`yes`, `no`"}.get(f.kind, "free text")


def _range_multi(f: FieldSpec) -> str:
    return ("any of " if f.kind == "multi" else "") + _range(f)


def render_data_dictionary() -> str:
    def table(fields):
        rows = ["| Field | Item | Type | Valid values | Role | Applies to | Nullable | Description |",
                "|---|---|---|---|---|---|---|---|"]
        rows += [f"| `{f.name}` | {f.item} | {f.kind} | {_range_multi(f)} | {_ROLE_LABEL[f.role]} | {f.applies_to} | "
                 f"{'yes' if f.nullable else 'no'} | {f.description} |" for f in fields]
        return "\n".join(rows)

    out = [
        "# Data dictionary",
        "",
        "_Generated from `core/schema.py` by `python -m core.schema` - do not edit by hand._",
        "",
        f"Questionnaire versions: graduate **{QUESTIONNAIRE_VERSION['graduate']}**, employer "
        f"**{QUESTIONNAIRE_VERSION['employer']}** (see `docs/questionnaires/questionnaire_revisions.md`).",
        "",
        "**Role** says how each raw field is used: *Model feature* fields feed the predictors (directly or "
        "via a derived feature), *Outcome construction* fields build `duration_months`/`event` and are "
        "never predictors, *Fairness only* fields are used solely in the subgroup check, and *Descriptive* "
        "fields are reported but not modelled.",
        "",
        "## 1. Graduate questionnaire - raw fields",
        "",
        table(RAW_FIELDS),
        "",
        "## 2. Derived features",
        "",
        "| Feature | Construction |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in DERIVED_FEATURES.items()],
        "",
        "## 3. Model features",
        "",
        f"{len(MODEL_FEATURES)} features -> **{model_parameter_count()} model columns** after one-hot encoding "
        f"(reference levels: first listed). Skill blocks enter as **means**, not PCA.",
        "",
        "| Feature | Type | Levels |",
        "|---|---|---|",
        *[f"| `{k}` | {t} | {', '.join(CATEGORICAL_LEVELS.get(k, ())) or '-'} |" for k, t in MODEL_FEATURES.items()],
        "",
        "Never model features: `gender` (fairness only) and every outcome / post-employment field: "
        + ", ".join(f"`{x}`" for x in sorted(POST_EMPLOYMENT_FIELDS)) + ".",
        "",
        "## 4. Skill vocabulary and areas",
        "",
        "| Key | Kind | Label | Graduate item | Employer item | Area |",
        "|---|---|---|---|---|---|",
        *[f"| `{s.key}` | {s.kind} | {s.label} | {s.graduate_item} | {s.employer_item} | "
          f"{next((v['label'] for v in SKILL_AREAS.values() if s.key in v['skills']), 'Soft skills')} |"
          for s in SKILL_VOCABULARY],
        "",
        "## 5. Actionable features (what-if)",
        "",
        "| Feature | Min | Max | Step | Applies to | Max change |",
        "|---|---|---|---|---|---|",
        *[f"| `{k}` | {a.min:g} | {a.max:g} | {a.step:g} | {a.applies_to} | "
          f"{'-' if a.max_change is None else f'{a.max_change:g}'} |" for k, a in ACTIONABLE_FEATURES.items()],
        "",
        "Not actionable: " + ", ".join(NON_ACTIONABLE) + ". Changing a skill-area rating sets every item "
        "in that area to the new value before the block means are recomputed.",
        "",
        "## 6. Employer questionnaire - raw fields",
        "",
        table(EMPLOYER_FIELDS),
        "",
        "## 7. Questionnaire-study items (not model inputs)",
        "",
        "Analysed in `core/survey_study.py` (report: `artifacts/questionnaire_study.md`). `multi` fields hold the "
        "selected options as semicolon-separated keys. Items that repeat a model question in other words are "
        "used as consistency checks.",
        "",
        "### Graduate questionnaire",
        "",
        table(SURVEY_FIELDS),
        "",
        "### Employer questionnaire",
        "",
        table(EMPLOYER_SURVEY_FIELDS),
        "",
        "## 8. Job ads (collected manually) - raw fields",
        "",
        table(JOB_AD_FIELDS),
        "",
        "## 9. Outcome-construction constants",
        "",
        f"- `JOB_BEFORE_GRADUATION_MONTHS` = {JOB_BEFORE_GRADUATION_MONTHS}",
        f"- `PREDICTION_HORIZON_MONTHS` = {PREDICTION_HORIZON_MONTHS}",
        f"- `EPV_WARNING_THRESHOLD` = {EPV_WARNING_THRESHOLD}",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "docs" / "data_dictionary.md"
    path.write_text(render_data_dictionary(), encoding="utf-8")
    print(f"Wrote {path} ({len(RAW_FIELDS)} graduate fields, {model_parameter_count()} model columns)")
