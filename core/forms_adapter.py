"""Convert a raw Google Forms CSV export into the canonical schema format.

Real exports have question text as headers ("Q5a. In which month ...") and option labels as
values ("Female", "Second class upper"). Columns are matched by the leading question number,
so every question title in the form MUST start with its item number (Q5a., Q17a., Q18. ...).
Matrix (grid) questions are matched by their [row label].

`to_canonical(df, kind)` is called by core/clean.py; files already in canonical format pass through.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from core import schema, survey_items

# When dates are ambiguous (e.g. 03/04/2026) and nothing in the column disambiguates them,
# assume day-first (Sri Lankan / en-GB locale). Columns with a value > 12 are detected automatically.
DEFAULT_DAYFIRST = True

ITEM_RE = re.compile(r"^\s*(Q\d+[a-z]?)\s*[.:)]?\s*(.*?)\s*(?:\[([^\]]+)\])?\s*$", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def norm(text) -> str:
    """Normalise an option label for matching: lowercase, plain hyphens, single spaces."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text).strip().lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).rstrip(".")


YES_NO = {"yes": "yes", "no": "no"}
LABEL_MAPS: dict[str, dict[str, str]] = {
    "age_group": {"20-22": "20-22", "23-25": "23-25", "26-28": "26-28", "above 28": "28+"},
    "gender": {"male": "male", "female": "female", "prefer not to say": "prefer_not_to_say"},
    "degree_type": {
        "computer science": "computer_science", "information technology": "information_technology",
        "software engineering": "software_engineering", "information systems": "information_systems",
        "cyber security": "cyber_security", "data science": "data_science",
        "artificial intelligence / machine learning": "ai_ml",
    },
    "degree_class": {"first class": "first", "second class upper": "second_upper",
                     "second class lower": "second_lower", "third class / general pass": "third_or_pass"},
    "industry_modules": {"yes": "yes", "no": "no", "not sure": "not_sure"},
    "internship_completed": YES_NO, "has_portfolio": YES_NO, "ever_it_job": YES_NO, "open_source": YES_NO,
    "prep_start": {"before starting university": "before_university", "during first year of university": "first_year",
                   "during second / third year": "second_third_year", "during final year": "final_year",
                   "after graduation": "after_graduation", "i have not started preparing": "not_started"},
    "employment_status": {
        "full-time it employment": "full_time_it", "it internship": "it_internship",
        "graduated, currently seeking it employment": "seeking_it", "non-it employment": "non_it",
        "further studies": "further_studies", "self-employed": "self_employed",
        "not yet graduated / have not started job-seeking": "not_seeking",
    },
    # employer form
    "org_type": {
        "software development company": "software_dev", "it consulting company": "it_consulting",
        "financial technology company": "fintech", "banking / financial institution": "banking",
        "telecommunications company": "telecom", "bpo / kpo": "bpo_kpo",
        "government it organisation": "government_it", "startup": "startup",
    },
    "org_size": {"less than 50": "<50", "50-200": "50-200", "201-500": "201-500", "more than 500": "500+"},
}
# Questionnaire-study items: option labels -> keys (single-choice items join LABEL_MAPS; select-all items
# are matched label by label in _map_multi).
MULTI_LABELS: dict[str, dict[str, str]] = {}
for _name, _item, _kind, _opts, _ in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS:
    if _kind == "binary":
        LABEL_MAPS[_name] = YES_NO
    elif _kind == "categorical":
        LABEL_MAPS[_name] = {norm(lab): key for key, lab in _opts}
    elif _kind == "multi":
        MULTI_LABELS[_name] = {norm(lab): key for key, lab in _opts}
RANK_LABELS = {norm(lab): f"rank_{key}" for key, lab in survey_items.EMPLOYER_RANK_FACTORS}
LABEL_MAPS["skill_update_freq"]["less than every year"] = "more_than_yearly"   # employer Q11 wording before v2.1
# "Not applicable - ..." options (graduate Q31, Q39, Q43 in v2.1) are recorded as a missing answer.
NOT_APPLICABLE = re.compile(r"not applicable[^,;]*")

# Labels whose free-text "Other: ..." answer maps to a fixed category.
OTHER_CATEGORY = {"degree_type": "other", "org_type": "other",
                  **{n: "other" for n, _, k, o, _ in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS
                     if k in ("categorical", "multi") and o and "other" in dict(o)}}
SURVEY_OTHER = {n for n, _, k, o, _ in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS
                if k == "categorical" and o and "other" in dict(o)} | {"degree_type", "org_type"}
# Graduate Q3 "Other (not listed)": the typed institution name is in an unnumbered follow-up column.
OTHER_UNIVERSITY = "other (not listed)"
OTHER_UNIVERSITY_TEXT = re.compile(r"^if you chose 'other", re.IGNORECASE)

# Employer Q7 row labels -> factor key
EMPLOYER_FACTOR_LABELS = {
    "degree qualification": "degree_qualification", "academic performance / gpa": "gpa",
    "technical skills": "technical_skills", "internship experience": "internship_experience",
    "personal software projects": "personal_projects", "professional certifications": "certifications",
    "github / technical portfolio": "portfolio", "communication skills": "communication",
    "interview performance": "interview_performance",
}
# Skill row labels: graduate wording, employer v1 wording and short forms all map to the key.
SKILL_LABELS = {norm(s.label): s.key for s in schema.SKILL_VOCABULARY}
SKILL_LABELS.update({
    "sql / database knowledge": "sql_databases", "git / version control": "git", "linux": "linux",
    "communication": "communication", "self-learning": "self_learning", "leadership": "leadership",
})


def _status_from_q5(value) -> str | None:
    v = norm(value)
    if v.startswith("final-year") or v.startswith("final year"):
        return "final_year"
    if v.startswith("graduated"):
        return "graduated"
    return None


def _map_label(field: str, value):
    v = norm(value)
    if not v or v.startswith("not applicable"):
        return None
    mapping = LABEL_MAPS.get(field, {})
    if v in mapping:
        return mapping[v]
    if field in OTHER_CATEGORY and v.startswith("other"):
        return OTHER_CATEGORY[field]
    if field in SURVEY_OTHER:   # Google Forms exports a typed "Other" answer as the text itself
        return "other"
    return value   # unknown label: kept as-is so cleaning logs it as an invalid category


def _map_multi(field: str, value):
    """Google Forms select-all answer ("Coursera, Udemy, my own text") -> "coursera;udemy;other".
    Known labels are matched first (longest first, so labels containing commas survive); any text left
    over is an "Other" answer."""
    rest = NOT_APPLICABLE.sub(" ", norm(value))
    if not rest.strip(" ,;"):
        return None
    keys = []
    for lab, key in sorted(MULTI_LABELS[field].items(), key=lambda kv: -len(kv[0])):
        if lab in rest:
            rest = rest.replace(lab, " ")
            if key not in keys:
                keys.append(key)
    if re.sub(r"[\s,;]+", "", rest) and field in OTHER_CATEGORY and "other" not in keys:
        keys.append("other")
    order = list(MULTI_LABELS[field].values())
    return ";".join(sorted(keys, key=order.index)) or None


def _detect_dayfirst(values: pd.Series) -> bool:
    parts = values.dropna().astype(str).str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
    parts = parts.dropna().astype(int)
    if (parts[0] > 12).any():
        return True
    if (parts[1] > 12).any():
        return False
    return DEFAULT_DAYFIRST


def parse_datetime(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(r"\s*GMT.*$", "", regex=True).str.strip()
    iso = cleaned.str.match(r"^\d{4}-\d{2}", na=False)
    out = pd.to_datetime(cleaned.where(iso), errors="coerce", format="mixed")
    rest = cleaned.where(~iso)
    if rest.notna().any():
        out = out.fillna(pd.to_datetime(rest, errors="coerce", dayfirst=_detect_dayfirst(rest), format="mixed"))
    return out


def parse_month(values: pd.Series) -> pd.Series:
    """Any date-like value -> 'YYYY-MM' string (or None)."""
    dt = parse_datetime(values)
    return dt.dt.strftime("%Y-%m").where(dt.notna(), None)


def _month_from_parts(month: pd.Series, year: pd.Series) -> pd.Series:
    def one(m, y):
        mn, yn = norm(m), norm(y)
        mi = MONTHS.get(mn[:3]) if mn[:3] in MONTHS else (int(mn) if mn.isdigit() else None)
        return f"{int(float(yn)):04d}-{mi:02d}" if mi and yn.replace(".0", "").isdigit() else None
    return pd.Series([one(m, y) for m, y in zip(month, year)], index=month.index)


def _first_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def is_canonical(df: pd.DataFrame, kind: str) -> bool:
    fields = {"graduates": schema.RAW_FIELDS, "employers": schema.EMPLOYER_FIELDS}[kind]
    required = {f.name for f in fields if f.role != "id" and f.name not in schema.LATE_OPTIONAL_FIELDS}
    return required <= set(df.columns)


def to_canonical(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (canonical dataframe, notes). Pass-through if already canonical."""
    all_fields = schema.GRADUATE_ALL_FIELDS if kind == "graduates" else schema.EMPLOYER_ALL_FIELDS
    if is_canonical(df, kind):
        out, notes = df.copy(), ["input already in canonical format"]
        for name in schema.LATE_OPTIONAL_FIELDS:
            if kind == "graduates" and name not in out:
                out[name] = None
                notes.append(f"optional column {name} not in input (set to missing)")
        survey = [f.name for f in all_fields if f.role in ("survey", "quality") and f.name not in out]
        for name in survey:
            out[name] = None
        if survey:
            notes.append(f"questionnaire-study columns not in input (set to missing): {len(survey)}")
        return out, notes
    fields = all_fields
    id_col = "respondent_id" if kind == "graduates" else "employer_id"
    simple = {f.item.upper(): f for f in fields if f.item.startswith("Q") and "." not in f.item}

    out = pd.DataFrame(index=df.index)
    notes, unmatched = [], []
    month_parts: dict[str, dict[str, pd.Series]] = {}
    typed_university = None
    for col in df.columns:
        if kind == "graduates" and OTHER_UNIVERSITY_TEXT.match(str(col).strip()):
            typed_university = df[col]
            continue
        if norm(col) == "timestamp":
            out["survey_date"] = parse_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
            continue
        m = ITEM_RE.match(str(col))
        if not m:
            unmatched.append(col)
            continue
        item, row = m.group(1).upper(), m.group(3)
        if row and norm(row).startswith("attention check") and (kind, item) in (("graduates", "Q18"),
                                                                               ("employers", "Q8")):
            out["attention_check"] = _first_number(df[col])
            continue
        if kind == "graduates" and item in ("Q18", "Q19") and row:
            key = SKILL_LABELS.get(norm(row))
            if key:
                out[schema.SKILLS_BY_KEY[key].column] = _first_number(df[col])
                continue
        if kind == "employers" and item in ("Q8", "Q9") and row:
            key = SKILL_LABELS.get(norm(row))
            if key:
                out[f"imp_{key}"] = _first_number(df[col])
                continue
        if kind == "employers" and item == "Q20" and row:
            name = RANK_LABELS.get(norm(row))
            if name:
                out[name] = _first_number(df[col])
                continue
        if kind == "employers" and item == "Q7" and row:
            key = EMPLOYER_FACTOR_LABELS.get(norm(row))
            if key:
                out[f"factor_{key}"] = _first_number(df[col])
                continue
        spec = simple.get(item)
        if spec is None:
            unmatched.append(col)
            continue
        if spec.kind == "month" and row and norm(row) in ("month", "year"):
            month_parts.setdefault(spec.name, {})[norm(row)] = df[col]
            continue
        if spec.name == "respondent_status":
            out[spec.name] = df[col].map(_status_from_q5)
        elif spec.kind == "month":
            out[spec.name] = parse_month(df[col])
        elif spec.kind in ("integer", "numeric", "likert"):
            out[spec.name] = _first_number(df[col])
        elif spec.kind in ("categorical", "binary"):
            out[spec.name] = df[col].map(lambda v, f=spec.name: _map_label(f, v))
        elif spec.kind == "multi":
            out[spec.name] = df[col].map(lambda v, f=spec.name: _map_multi(f, v))
        else:
            out[spec.name] = df[col].where(df[col].notna(), None)
    for name, parts in month_parts.items():
        if {"month", "year"} <= parts.keys():
            out[name] = _month_from_parts(parts["month"], parts["year"])
    if typed_university is not None and "university_name" in out:
        use = out["university_name"].map(norm).eq(OTHER_UNIVERSITY) & typed_university.map(norm).ne("")
        out.loc[use, "university_name"] = typed_university[use].astype(str).str.strip()

    if id_col not in out:
        prefix = "R" if kind == "graduates" else "E"
        out.insert(0, id_col, [f"{prefix}{i:04d}" for i in range(1, len(out) + 1)])
    missing = [f.name for f in fields if f.name not in out]
    for name in missing:
        out[name] = None
    if missing:
        notes.append(f"columns not found in export (set to missing): {', '.join(missing)}")
    if unmatched:
        notes.append(f"export columns ignored (not used by the model): {len(unmatched)}")
    return out[[f.name for f in fields]], notes
