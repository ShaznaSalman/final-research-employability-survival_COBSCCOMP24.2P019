"""Cleaning and outcome construction.

Reads the graduate, employer and job-ad CSVs from data/raw/ (real files take priority over
*_SYNTHETIC files), validates every field against core/schema.py, builds
duration_months / event for GRADUATES ONLY, and writes:

  data/processed/graduates_train.csv     training set (graduates in the IT job market)
  data/processed/final_year.csv          final-year respondents (descriptive / prediction targets only)
  data/processed/excluded.csv            every removed row with its reason
  data/processed/employers.csv, job_ads.csv
  data/processed/dataset_info.json       counts + synthetic flag (read by the API)
  docs/data_cleaning_log.md              the audit trail

Run:  python -m core.clean
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core import forms_adapter, schema, splits

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "docs" / "data_cleaning_log.md"

MAX_MONTHS_JOB_BEFORE_GRADUATION = 24   # earlier start dates are treated as typos
MAX_MONTHS_SINCE_GRADUATION = 36        # eligibility: completed within ~2 years (Q5 allows 2024 graduates)

MISSING_VALUE_RULES = {
    "Outcome fields (graduation month, Q17, Q17a, survey date)":
        "Row excluded from training - outcome cannot be constructed (see exclusions table).",
    "Skill items (Q18, Q19)":
        f"Block mean uses the answered items if at least {schema.MIN_ITEMS_ANSWERED_FRACTION:.0%} of the block "
        "is answered; otherwise the block mean is missing and imputed (training-fold median) in preprocessing.",
    "GPA (Q6)": "Replaced by the degree-class equivalent (Q6a) when blank; if both are blank, imputed with "
                "the training-fold median in preprocessing.",
    "Counts (Q9, Q12, Q13)": "Internship months = 0 when Q11 = No. Otherwise imputed with the training-fold "
                             "median in preprocessing.",
    "Degree type (Q4), university type (Q3)": "Imputed with the training-fold most frequent level in "
                                              "preprocessing. Unmapped university names are listed below.",
    "Gender (Q2)": "Kept as missing; shown as its own group in the fairness check (suppressed if n < 20).",
    "Out-of-range / invalid values": "Set to missing, then handled by the rule for that field. Never "
                                     "silently corrected (e.g. a GPA of 34 is not turned into 3.4).",
    "Duplicate submissions": "Rows identical on every answer (ignoring row id and timestamp) - the earliest submission is kept.",
}


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def _classify_csv(path: Path) -> str | None:
    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return None
    text = [str(c).strip() for c in cols]
    grid = lambda q: any(c.upper().startswith(q) and "[" in c for c in text)  # noqa: E731
    if "respondent_status" in text or grid("Q18"):
        return "graduates"
    if any(c.startswith("imp_") for c in text) or grid("Q8"):
        return "employers"
    if {"skills", "title"} <= set(text):
        return "job_ads"
    if {"university_name", "university_type"} <= set(text):
        return "university_types"
    return None


def find_input(kind: str, directory: Path | None = None) -> tuple[Path, bool]:
    """Return (path, is_synthetic). A real (non-_SYNTHETIC) file always wins over a synthetic one."""
    directory = directory or (REFERENCE_DIR if kind == "university_types" else RAW_DIR)
    matches = [p for p in sorted(directory.glob("*.csv")) if _classify_csv(p) == kind]
    real = [p for p in matches if "_SYNTHETIC" not in p.name]
    if len(real) > 1:
        raise ValueError(f"More than one real {kind} file in {directory}: {[p.name for p in real]}. Keep one.")
    if real:
        return real[0], False
    synth = [p for p in matches if "_SYNTHETIC" in p.name]
    if not synth:
        raise FileNotFoundError(f"No {kind} CSV found in {directory}. Run `python -m core.synthetic` first.")
    return synth[0], True


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
@dataclass
class Audit:
    source: str = ""
    is_synthetic: bool = True
    raw_n: int = 0
    steps: list[dict] = field(default_factory=list)            # removal steps, in order
    field_issues: dict = field(default_factory=lambda: defaultdict(Counter))
    notes: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    competing: pd.DataFrame | None = None                     # competing-risks outcomes (core/competing.py)
    survey: pd.DataFrame | None = None                        # questionnaire-study answers (core/survey_study.py)

    def remove(self, df: pd.DataFrame, mask: pd.Series, step: str, reason: str) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        for rid in df.loc[mask, "respondent_id"]:
            self.excluded.append({"respondent_id": rid, "step": step, "reason": reason})
        kept = df.loc[~mask]
        self.steps.append({"step": step, "reason": reason, "removed": int(mask.sum()), "remaining": len(kept)})
        return kept


def month_index(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values, format="%Y-%m", errors="coerce")
    return (dt.dt.year * 12 + dt.dt.month - 1).astype("Int64")


def cronbach_alpha(items: pd.DataFrame) -> tuple[float, int]:
    """Cronbach's alpha on complete cases. Returns (alpha, n used)."""
    x = items.dropna()
    k = x.shape[1]
    if len(x) < 3 or k < 2:
        return float("nan"), len(x)
    total_var = x.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / total_var)), len(x)


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------
def validate_fields(df: pd.DataFrame, fields, audit: Audit) -> pd.DataFrame:
    """Coerce types; set out-of-range / invalid values to missing and count them per field."""
    df = df.copy()
    for f in fields:
        if f.name not in df or f.kind in ("id", "text"):
            continue
        col = df[f.name]
        present = col.notna() & (col.astype("string").str.strip() != "")
        if f.kind in ("integer", "numeric", "likert"):
            num = pd.to_numeric(col, errors="coerce")
            bad = present & num.isna()
            if f.min is not None:
                bad |= num.notna() & ((num < f.min) | (num > f.max))
            if f.kind in ("integer", "likert"):
                bad |= num.notna() & (num != np.round(num))
            df[f.name] = num.mask(bad).astype(float)
        elif f.kind in ("categorical", "binary"):
            cats = f.categories or ("yes", "no")
            val = col.astype("string").str.strip().str.lower()
            bad = present & ~val.isin(cats)
            df[f.name] = val.mask(bad | ~present).astype(object).where(lambda s: s.notna(), None)
        elif f.kind == "multi":
            # select-all answers: keep only known option keys; an answer with any unknown key is flagged
            def keep_known(v, cats=f.categories):
                keys = [k.strip().lower() for k in str(v).split(";") if k.strip()]
                return ";".join(k for k in keys if k in cats) or None, any(k not in cats for k in keys)
            parsed = [keep_known(v) if p else (None, False) for v, p in zip(col, present)]
            bad = pd.Series([b for _, b in parsed], index=col.index)
            df[f.name] = pd.Series([v for v, _ in parsed], index=col.index, dtype=object)
        elif f.kind == "month":
            idx = month_index(col.astype("string").str.strip())
            bad = present & idx.isna()
            df[f.name] = col.astype("string").str.strip().mask(bad | ~present).astype(object)
        elif f.kind == "timestamp":
            ts = pd.to_datetime(col, errors="coerce")
            bad = present & ts.isna()
            df[f.name] = ts
        else:
            continue
        if bad.any():
            audit.field_issues[f.name]["invalid / out of range -> missing"] += int(bad.sum())
    return df


# ---------------------------------------------------------------------------
# Graduates
# ---------------------------------------------------------------------------
def load_university_lookup(path: Path) -> dict[str, str]:
    lut = pd.read_csv(path)
    return {forms_adapter.norm(n): t for n, t in zip(lut["university_name"], lut["university_type"])}


def load_institution_lookup(path: Path) -> dict[str, str]:
    """Name variant -> canonical institution (optional `institution` column; else the name itself)."""
    lut = pd.read_csv(path)
    canon = lut["institution"] if "institution" in lut else lut["university_name"]
    return {forms_adapter.norm(n): c for n, c in zip(lut["university_name"], canon)}


def clean_graduates(raw: pd.DataFrame, uni_lookup: dict[str, str], audit: Audit
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training graduates, final-year respondents). Exclusions are recorded in `audit`."""
    df, notes = forms_adapter.to_canonical(raw, "graduates")
    audit.notes += notes
    audit.raw_n = len(df)
    df["respondent_id"] = df["respondent_id"].astype(str)

    # 1. Duplicate submissions
    # Keep the earliest submission (by timestamp, then row id) of identical answer sets.
    answer_cols = [c for c in df.columns if c not in ("respondent_id", "survey_date")]
    order = df.assign(_ts=pd.to_datetime(df["survey_date"], errors="coerce"))         .sort_values(["_ts", "respondent_id"]).index
    dup = df.loc[order].astype(str).duplicated(subset=answer_cols, keep="first").reindex(df.index)
    df = audit.remove(df, dup, "Duplicate submission", "identical answers to an earlier submission")

    # 2. Field-level validation (values -> missing, rows kept)
    df = validate_fields(df, schema.GRADUATE_ALL_FIELDS, audit)

    # 2b. Careless responding: wrong answer to the instructed attention-check row (unanswered rows are kept)
    failed = df["attention_check"].notna() & (df["attention_check"] != schema.ATTENTION_CHECK_ANSWER)
    df = audit.remove(df, failed, "Failed attention check",
                      f"answered the attention-check row (Q18) with something other than {schema.ATTENTION_CHECK_ANSWER}")

    # 3. Harmonisation (deterministic recodes; nothing is fitted here)
    from_class = df["gpa"].isna() & df["degree_class"].notna()
    df["gpa_source"] = np.where(df["gpa"].notna(), "gpa", np.where(from_class, "degree_class", "missing"))
    df.loc[from_class, "gpa"] = df.loc[from_class, "degree_class"].map(schema.DEGREE_CLASS_TO_GPA)
    df["degree_group"] = df["degree_type"].map(schema.DEGREE_GROUPS)
    if "university_type" not in df or df["university_type"].isna().all():
        df["university_type"] = df["university_name"].map(lambda n: uni_lookup.get(forms_adapter.norm(n)))
    unmapped = sorted(df.loc[df["university_type"].isna() & df["university_name"].notna(), "university_name"]
                      .astype(str).unique())
    if unmapped:
        audit.notes.append("Unmapped university names (add to data/reference/university_types.csv): "
                           + "; ".join(unmapped))
    no_intern = df["internship_completed"] == "no"
    for c in ("internship_months", "internship_count"):
        filled = no_intern & df[c].isna()
        df.loc[no_intern, c] = 0
        if filled.any():
            audit.field_issues[c]["set to 0 because Q11 = No"] += int(filled.sum())

    # 4. Separate final-year respondents (no valid time-to-employment yet)
    df = audit.remove(df, df["respondent_status"].isna(), "Status unknown", "Q5 missing or unrecognised")
    is_fy = df["respondent_status"] == "final_year"
    final_year = df.loc[is_fy].copy()
    df = df.loc[~is_fy].copy()
    audit.steps.append({"step": "Final-year respondents set aside", "reason": "no time-to-employment yet; "
                        "kept in final_year.csv for descriptive use only",
                        "removed": int(is_fy.sum()), "remaining": len(df)})

    # 5. Outcome-construction exclusions (graduates)
    grad_m = month_index(df["graduation_date"])
    survey_m = (df["survey_date"].dt.year * 12 + df["survey_date"].dt.month - 1).astype("Int64")
    job_m = month_index(df["first_job_start_date"])
    has_job = df["ever_it_job"] == "yes"
    checks = [
        ("Missing completion month", grad_m.isna(), "Q5a missing or invalid - no time origin"),
        ("Missing survey date", survey_m.isna(), "no timestamp - follow-up unknown"),
        ("Completion after survey", grad_m > survey_m, "completion month later than the survey date"),
        ("Outside eligibility window", (survey_m - grad_m) > MAX_MONTHS_SINCE_GRADUATION,
         f"completed more than {MAX_MONTHS_SINCE_GRADUATION} months before the survey"),
        ("Missing job status", df["ever_it_job"].isna(), "Q17 missing - event unknown"),
        ("Missing job start month", has_job & job_m.isna(), "Q17 = yes but Q17a missing or invalid"),
        ("Job start after survey", has_job & (job_m > survey_m), "first job month later than the survey date"),
        ("Implausible job start", has_job & ((grad_m - job_m) > MAX_MONTHS_JOB_BEFORE_GRADUATION),
         f"first job more than {MAX_MONTHS_JOB_BEFORE_GRADUATION} months before completion (likely typo)"),
        ("Not in IT job market", ~has_job & df["employment_status"].isin(schema.NOT_IN_MARKET_STATUSES),
         "no IT role and Q16 = further studies / not seeking"),
        ("No follow-up time", ~has_job & (survey_m - grad_m <= 0),
         "still searching but surveyed in the completion month"),
    ]
    audit.competing = competing_outcomes(df, grad_m, survey_m, job_m,
                                         [m for step, m, _ in checks if step != "Not in IT job market"])
    for step, mask, reason in checks:
        keep = df.index
        df = audit.remove(df, mask.reindex(keep).astype("boolean"), step, reason)

    # 6. Duration and event
    grad_m, survey_m, job_m = grad_m.loc[df.index], survey_m.loc[df.index], job_m.loc[df.index]
    df["event"] = (df["ever_it_job"] == "yes").astype(int)
    months = np.where(df["event"] == 1, (job_m - grad_m).astype(float), (survey_m - grad_m).astype(float))
    df["job_before_graduation"] = (df["event"] == 1) & (months <= 0)
    df["duration_months"] = np.where(df["job_before_graduation"], schema.JOB_BEFORE_GRADUATION_MONTHS, months)

    schema.assert_no_leakage()
    candidates = [f.name for f in schema.RAW_FIELDS if f.role == "candidate"]
    profile_cols = (["respondent_id", "gender", "university_name", "university_type", "degree_type", "degree_group",
                     "gpa", "gpa_source", "certifications_count", "internship_completed", "internship_count",
                     "internship_months", "project_count", *candidates]
                    + list(schema.TECH_COLUMNS) + list(schema.SOFT_COLUMNS))
    train_cols = profile_cols + ["graduation_date", "job_before_graduation", "duration_months", "event"]
    audit.competing = audit.competing[[c for c in profile_cols if c in audit.competing]
                                      + ["graduation_date", "employment_status", "cr_time", "cr_cause",
                                         "alt_date_missing"]].sort_values("respondent_id").reset_index(drop=True)
    fy_cols = profile_cols
    # Questionnaire study: every respondent kept above (training graduates + final-year), with the profile
    # answers that the repeated questions are checked against and the outcome for the graduates.
    survey_cols = [f.name for f in schema.SURVEY_FIELDS]
    context = ["respondent_status", "employment_status", "industry_modules", "internship_completed",
               "internship_months", "certifications_count", "project_count",
               *[f.name for f in schema.RAW_FIELDS if f.role == "candidate"]]
    audit.survey = (pd.concat([df.assign(respondent_status="graduated"),
                               final_year.assign(event=np.nan, duration_months=np.nan)])
                    [["respondent_id", *context, *schema.TECH_COLUMNS, *schema.SOFT_COLUMNS, "event",
                      "duration_months", *survey_cols]]
                    .sort_values("respondent_id").reset_index(drop=True))
    return (df[train_cols].sort_values("respondent_id").reset_index(drop=True),
            final_year[fy_cols + ["expected_graduation_date"]].sort_values("respondent_id").reset_index(drop=True))


def competing_outcomes(df: pd.DataFrame, grad_m, survey_m, job_m, exclusion_masks: list) -> pd.DataFrame:
    """Outcomes for the competing-risks analysis (secondary; the main outcome is unchanged).

    cr_cause 1 = first paid IT role (as in the main analysis); 2 = started a non-IT job, self-employment or further
    studies (Q16a) with no IT role; 0 = censored at the survey. Unlike the main analysis, graduates in further
    studies are kept when Q16a gives the month. Other exclusions are the same as the main analysis.
    """
    if "alt_path_start_date" not in df:
        df = df.assign(alt_path_start_date=None)
    bad = pd.concat([m.astype("boolean").fillna(False) for m in exclusion_masks], axis=1).any(axis=1)
    has_job = df["ever_it_job"] == "yes"
    alt_m = month_index(df["alt_path_start_date"])
    on_alt = ~has_job & df["employment_status"].isin(schema.ALT_PATH_STATUSES)
    alt_ok = (on_alt & alt_m.notna() & (alt_m <= survey_m)
              & ((grad_m - alt_m) <= MAX_MONTHS_JOB_BEFORE_GRADUATION)).fillna(False).astype(bool)
    cause = np.where(has_job, 1, np.where(alt_ok, 2, 0))
    raw_t = np.where(cause == 1, (job_m - grad_m).astype(float),
                     np.where(cause == 2, (alt_m - grad_m).astype(float), (survey_m - grad_m).astype(float)))
    out = df.assign(cr_cause=cause, cr_time=np.where((cause > 0) & (raw_t <= 0),
                                                     schema.JOB_BEFORE_GRADUATION_MONTHS, raw_t),
                    alt_date_missing=(on_alt & ~alt_ok).astype(bool))
    out_of_market = (cause == 0) & df["employment_status"].isin(schema.NOT_IN_MARKET_STATUSES).to_numpy()
    no_follow_up = (cause == 0) & ~(out["cr_time"] > 0).to_numpy()
    return out.loc[~bad.to_numpy() & ~out_of_market & ~no_follow_up].copy()


# ---------------------------------------------------------------------------
# Employers and job ads
# ---------------------------------------------------------------------------
def clean_employers(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, notes = forms_adapter.to_canonical(raw, "employers")
    n0 = len(df)
    df = df.loc[~df.drop(columns="employer_id").astype(str).duplicated()].copy()
    audit = Audit()
    df = validate_fields(df, schema.EMPLOYER_ALL_FIELDS, audit)
    failed = df["attention_check"].notna() & (df["attention_check"] != schema.ATTENTION_CHECK_ANSWER)
    df = df.loc[~failed].copy()
    issues = {k: dict(v) for k, v in audit.field_issues.items()}
    return df.reset_index(drop=True), {"raw_n": n0, "duplicates": n0 - len(df) - int(failed.sum()),
                                       "failed_attention_check": int(failed.sum()), "final_n": len(df),
                                       "invalid_values": sum(sum(v.values()) for v in issues.values()),
                                       "notes": notes}


def _skill_token(token: str) -> str | None:
    t = forms_adapter.norm(token)
    if t in schema.SKILLS_BY_KEY:
        return t
    for s in schema.SKILL_VOCABULARY:
        if t == forms_adapter.norm(s.label) or t in s.aliases:
            return s.key
    return None


def clean_job_ads(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    n0 = len(df)
    unknown: Counter = Counter()

    def parse(cell) -> str:
        keys = []
        for tok in str(cell if pd.notna(cell) else "").split(";"):
            if not tok.strip():
                continue
            key = _skill_token(tok)
            if key is None:
                unknown[tok.strip()] += 1
            elif key not in keys:
                keys.append(key)
        return ";".join(keys)

    df["skills"] = df["skills"].map(parse)
    df = df.loc[~df.drop(columns="ad_id").astype(str).duplicated()]
    no_skills = df["skills"] == ""
    df = df.loc[~no_skills].reset_index(drop=True)
    return df, {"raw_n": n0, "duplicates_or_empty": n0 - len(df), "final_n": len(df),
                "no_recognised_skills": int(no_skills.sum()), "unknown_skill_tokens": dict(unknown)}


# ---------------------------------------------------------------------------
# Cleaning log
# ---------------------------------------------------------------------------
def render_log(audit: Audit, train: pd.DataFrame, final_year: pd.DataFrame, emp: dict, ads: dict,
               sources: dict, roles: pd.DataFrame | None = None) -> str:
    events = int(train["event"].sum())
    censored = int(len(train) - events)
    params = schema.model_parameter_count()
    role = (roles["split"].to_numpy() if roles is not None else np.full(len(train), "train"))
    train_events = int(train.loc[role == "train", "event"].sum())
    epv = train_events / params
    a_tech, n_tech = cronbach_alpha(train[list(schema.TECH_COLUMNS)])
    a_soft, n_soft = cronbach_alpha(train[list(schema.SOFT_COLUMNS)])
    lines = ["# Data cleaning log", "",
             "_Generated by `python -m core.clean` - do not edit by hand._", ""]
    if audit.is_synthetic:
        lines += ["> **SYNTHETIC DATA - DEMO ONLY.** These counts describe the fake dataset from "
                  "`core/synthetic.py`, not real survey responses.", ""]
    lines += ["## Sources", "", "| Data | File | Synthetic |", "|---|---|---|"]
    lines += [f"| {k} | `{Path(p).name}` | {'yes' if s else 'no'} |" for k, (p, s) in sources.items()]
    lines += ["", "## Graduate respondents: raw N -> training N", "",
              "| Step | Reason | Removed | Remaining |", "|---|---|---|---|",
              f"| Raw responses | - | - | {audit.raw_n} |"]
    lines += [f"| {s['step']} | {s['reason']} | {s['removed']} | {s['remaining']} |" for s in audit.steps]
    lines += ["", f"**Training set: {len(train)} graduates** ({events} found an IT role, {censored} still searching "
              f"at the survey = {100 * censored / max(len(train), 1):.1f}% censored). "
              f"Final-year respondents set aside: {len(final_year)}.", ""]
    total_removed = sum(s["removed"] for s in audit.steps)
    lines += [f"Check: {audit.raw_n} raw - {total_removed} removed/set aside = {audit.raw_n - total_removed} "
              f"= training N {len(train)}.", ""]

    jbg = int(train["job_before_graduation"].sum())
    lines += ["## Outcome construction", "",
              "- **Time origin:** month of degree completion (Q5a, final results released).",
              "- **Event = 1:** first paid IT role started (Q17 = yes); duration = Q17a month - Q5a month.",
              "- **Event = 0 (censored):** no IT role yet; duration = survey month - Q5a month.",
              f"- **Job on/before completion:** duration set to `JOB_BEFORE_GRADUATION_MONTHS` = "
              f"{schema.JOB_BEFORE_GRADUATION_MONTHS} months (a positive time is required by survival models; "
              f"half a month places these just after the origin). Applied to **{jbg}** graduates.",
              "- Durations are in whole months because dates are collected at month precision.", "",
              "| Duration (months) | Events | Censored |", "|---|---|---|"]
    for lab, lo, hi in [("<= 1", 0, 1), ("2-3", 1, 3), ("4-6", 3, 6), ("7-12", 6, 12), ("13-24", 12, 24),
                        ("> 24", 24, 1e9)]:
        sel = (train["duration_months"] > lo) & (train["duration_months"] <= hi) if lo else \
            train["duration_months"] <= hi
        lines.append(f"| {lab} | {int((sel & (train['event'] == 1)).sum())} | "
                     f"{int((sel & (train['event'] == 0)).sum())} |")

    lines += ["## Data split (fixed, no reuse)", "",
              "Every modelling graduate has exactly one role, assigned once (stratified by event and institution "
              "type; `core/splits.py`). Models are fitted only on **train**, tuned and compared only on "
              "**validation**, and evaluated once on the locked **test** set.", "",
              "| Role | Graduates | Events | Used for |", "|---|---|---|---|"]
    uses = {"train": "fitting every model", "validation": "tuning and model / feature-set comparison",
            "test": "final evaluation only (once)"}
    for r in splits.ROLES:
        sel = role == r
        lines.append(f"| {r} | {int(sel.sum())} | {int(train.loc[sel, 'event'].sum())} | {uses[r]} |")
    lines += ["", "## Events per variable (training set)", "",
              f"- Model parameters (after one-hot encoding): **{params}** ({', '.join(schema.MODEL_FEATURES)})",
              f"- Events in the training set: **{train_events}** (all modelling graduates: {events})",
              f"- **EPV = {epv:.1f}**", ""]
    if epv < schema.EPV_WARNING_THRESHOLD:
        lines += [f"> **WARNING: EPV {epv:.1f} is below {schema.EPV_WARNING_THRESHOLD}.** Unpenalised Cox estimates "
                  "may be unstable; rely on the penalised models and report wide intervals.", ""]
    else:
        lines += [f"EPV is at or above the rule-of-thumb threshold of {schema.EPV_WARNING_THRESHOLD}.", ""]

    lines += ["## Scale reliability (training set, complete cases)", "",
              "| Block | Items | Cronbach's alpha | n |", "|---|---|---|---|",
              f"| Technical skills (Q18) | {len(schema.TECH_COLUMNS)} | {a_tech:.2f} | {n_tech} |",
              f"| Soft skills (Q19) | {len(schema.SOFT_COLUMNS)} | {a_soft:.2f} | {n_soft} |", "",
              "Alpha >= 0.70 supports averaging each block into a single mean score.", ""]

    lines += ["## Value-level fixes (rows kept)", "", "| Field | Issue | Count |", "|---|---|---|"]
    issues = [(f, k, v) for f, c in sorted(audit.field_issues.items()) for k, v in c.items()]
    lines += [f"| `{f}` | {k} | {v} |" for f, k, v in issues] or ["| - | none | 0 |"]
    model_cols = ["gpa", "degree_type", "university_type", "internship_months", "certifications_count",
                  "project_count"]
    lines += ["", "Missing values remaining in the training set (handled in preprocessing, inside CV folds):", "",
              "| Field | Missing |", "|---|---|"]
    lines += [f"| `{c}` | {int(train[c].isna().sum())} |" for c in model_cols]
    lines += [f"| skill items (any of {len(schema.TECH_COLUMNS) + len(schema.SOFT_COLUMNS)}) | "
              f"{int(train[list(schema.TECH_COLUMNS + schema.SOFT_COLUMNS)].isna().any(axis=1).sum())} rows |",
              f"| `gender` (fairness only) | {int(train['gender'].isna().sum())} |",
              "", f"GPA taken from degree class (Q6a): {int((train['gpa_source'] == 'degree_class').sum())} graduates.",
              ""]

    lines += ["## Missing-value rules", "", "| Field(s) | Rule |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in MISSING_VALUE_RULES.items()]

    lines += ["", "## Leakage check", "",
              "`schema.assert_no_leakage()` passed: none of "
              + ", ".join(f"`{x}`" for x in sorted(schema.POST_EMPLOYMENT_FIELDS | schema.FAIRNESS_ONLY_FIELDS))
              + " is in `MODEL_FEATURES`.", ""]

    lines += ["## Employer responses and job ads", "",
              f"- Employers: {emp['raw_n']} raw -> {emp['duplicates']} duplicates removed -> **{emp['final_n']}**; "
              f"{emp['invalid_values']} invalid ratings set to missing.",
              f"- Job ads: {ads['raw_n']} raw -> {ads['duplicates_or_empty']} duplicate / no recognised skill "
              f"removed -> **{ads['final_n']}**.",
              f"- Unrecognised job-ad skill tokens: {ads['unknown_skill_tokens'] or 'none'}.", ""]
    if audit.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in audit.notes] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(raw_dir: Path = RAW_DIR, reference_dir: Path = REFERENCE_DIR, processed_dir: Path = PROCESSED_DIR,
        log_path: Path = LOG_PATH) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    sources = {k: find_input(k, reference_dir if k == "university_types" else raw_dir)
               for k in ("graduates", "employers", "job_ads", "university_types")}
    audit = Audit(source=sources["graduates"][0].name, is_synthetic=sources["graduates"][1])

    survey_csv = {"keep_default_na": False, "na_values": schema.SURVEY_NA_VALUES}   # keep "None" answers
    train, final_year = clean_graduates(pd.read_csv(sources["graduates"][0], **survey_csv),
                                        load_university_lookup(sources["university_types"][0]), audit)
    employers, emp_info = clean_employers(pd.read_csv(sources["employers"][0], **survey_csv))
    job_ads, ads_info = clean_job_ads(pd.read_csv(sources["job_ads"][0]))

    train.to_csv(processed_dir / "graduates_train.csv", index=False)
    roles = splits.write(train, processed_dir / "splits.csv")
    audit.competing.to_csv(processed_dir / "graduates_competing.csv", index=False)
    final_year.to_csv(processed_dir / "final_year.csv", index=False)
    pd.DataFrame(audit.excluded, columns=["respondent_id", "step", "reason"]) \
        .sort_values("respondent_id").to_csv(processed_dir / "excluded.csv", index=False)
    employers.to_csv(processed_dir / "employers.csv", index=False)
    job_ads.to_csv(processed_dir / "job_ads.csv", index=False)
    audit.survey.to_csv(processed_dir / "questionnaire_graduates.csv", index=False)

    role = roles["split"].to_numpy()
    n_role = {r: int((role == r).sum()) for r in splits.ROLES}
    ev_role = {r: int(train.loc[role == r, "event"].sum()) for r in splits.ROLES}
    info = {
        "is_synthetic": any(s for _, s in sources.values()),
        "sources": {k: p.name for k, (p, _) in sources.items()},
        "raw_n": audit.raw_n,
        "modelling_n": len(train),
        "modelling_events": int(train["event"].sum()),
        # the training SET (the only data any model is fitted on)
        "training_n": n_role["train"],
        "events": ev_role["train"],
        "censored": n_role["train"] - ev_role["train"],
        "validation_n": n_role["validation"], "validation_events": ev_role["validation"],
        "test_n": n_role["test"], "test_events": ev_role["test"],
        "final_year_n": len(final_year),
        "model_parameters": schema.model_parameter_count(),
        "epv": round(ev_role["train"] / schema.model_parameter_count(), 2),
        "employers_n": emp_info["final_n"],
        "job_ads_n": ads_info["final_n"],
    }
    (processed_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(render_log(audit, train, final_year, emp_info, ads_info, sources, roles), encoding="utf-8")
    return info


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
