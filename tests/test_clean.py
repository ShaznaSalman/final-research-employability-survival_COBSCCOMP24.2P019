"""Tests for cleaning, outcome construction and Google Forms import (core/clean.py, core/forms_adapter.py)."""
import numpy as np
import pandas as pd
import pytest

from core import clean, forms_adapter, schema, synthetic


@pytest.fixture(scope="module")
def synthetic_raw():
    raw, _, defects = synthetic.generate_graduates()
    return raw, defects


@pytest.fixture(scope="module")
def cleaned(synthetic_raw):
    raw, _ = synthetic_raw
    audit = clean.Audit()
    lookup = synthetic.lookup_dict()
    train, final_year = clean.clean_graduates(raw, lookup, audit)
    return train, final_year, audit


def _profile(**overrides) -> dict:
    """One canonical graduate row with sensible defaults."""
    row = {f.name: None for f in schema.RAW_FIELDS}
    row.update({"respondent_id": "T1", "survey_date": "2025-11-15 10:00:00", "gender": "female",
                "university_name": "Mahasena National University", "degree_type": "computer_science",
                "respondent_status": "graduated", "graduation_date": "2025-01", "gpa": 3.2,
                "certifications_count": 1, "internship_completed": "yes", "internship_count": 1,
                "internship_months": 6, "project_count": 3, "employment_status": "seeking_it",
                "ever_it_job": "no"})
    row.update({c: 3 for c in schema.TECH_COLUMNS + schema.SOFT_COLUMNS})
    row.update(overrides)
    return row


def _clean_rows(rows):
    audit = clean.Audit()
    train, fy = clean.clean_graduates(pd.DataFrame(rows), {"synthetic state university a": "state"}, audit)
    return train.set_index("respondent_id"), fy, audit


# --- outcome construction ---------------------------------------------------
def test_duration_and_event_for_each_case():
    train, _, _ = _clean_rows([
        _profile(respondent_id="employed", ever_it_job="yes", first_job_start_date="2025-05",
                 employment_status="full_time_it"),
        _profile(respondent_id="searching"),
        _profile(respondent_id="before_grad", ever_it_job="yes", first_job_start_date="2024-11",
                 employment_status="full_time_it"),
        _profile(respondent_id="same_month", ever_it_job="yes", first_job_start_date="2025-01",
                 employment_status="full_time_it"),
    ])
    assert (train.loc["employed", ["duration_months", "event"]] == [4, 1]).all()
    assert (train.loc["searching", ["duration_months", "event"]] == [10, 0]).all()
    for rid in ("before_grad", "same_month"):
        assert train.loc[rid, "duration_months"] == schema.JOB_BEFORE_GRADUATION_MONTHS
        assert train.loc[rid, "event"] == 1 and train.loc[rid, "job_before_graduation"]


def test_exclusion_rules():
    train, fy, audit = _clean_rows([
        _profile(respondent_id="ok"),
        _profile(respondent_id="no_start", ever_it_job="yes"),
        _profile(respondent_id="studying", employment_status="further_studies"),
        _profile(respondent_id="future_grad", graduation_date="2026-02"),
        _profile(respondent_id="typo_start", ever_it_job="yes", first_job_start_date="2006-03"),
        _profile(respondent_id="fy", respondent_status="final_year", graduation_date=None,
                 expected_graduation_date="2026-03", ever_it_job=None, employment_status="not_seeking"),
    ])
    assert list(train.index) == ["ok"]
    assert list(fy["respondent_id"]) == ["fy"]
    reasons = {e["respondent_id"]: e["step"] for e in audit.excluded}
    assert reasons == {"no_start": "Missing job start month", "studying": "Not in IT job market",
                       "future_grad": "Completion after survey", "typo_start": "Implausible job start"}


# --- synthetic dataset --------------------------------------------------------
def test_training_set_is_graduates_only(cleaned, synthetic_raw):
    train, final_year, _ = cleaned
    raw, _ = synthetic_raw
    fy_ids = set(raw.loc[raw["respondent_status"] == "final_year", "respondent_id"])
    assert not set(train["respondent_id"]) & fy_ids
    assert set(final_year["respondent_id"]) <= fy_ids
    assert train["duration_months"].gt(0).all() and train["event"].isin([0, 1]).all()


def test_audit_counts_add_up(cleaned):
    train, _, audit = cleaned
    removed = sum(s["removed"] for s in audit.steps)
    assert audit.raw_n - removed == len(train)
    for before, after in zip(audit.steps, audit.steps[1:]):
        assert before["remaining"] - after["removed"] == after["remaining"]


def test_every_injected_defect_is_handled(cleaned, synthetic_raw):
    train, final_year, audit = cleaned
    _, defects = synthetic_raw
    excluded = {e["respondent_id"] for e in audit.excluded}
    kept = pd.concat([train, final_year]).set_index("respondent_id")
    for d in defects:
        rid = d["respondent_id"]
        if d["defect"] in ("duplicate", "inconsistent"):
            assert rid in excluded, d
        elif d["defect"] == "out_of_range" and rid in kept.index:
            if d["column"] == "gpa" and kept.loc[rid, "gpa_source"] == "degree_class":
                continue   # invalid GPA dropped, then replaced by the Q6a degree-class equivalent
            assert pd.isna(kept.loc[rid, d["column"]]), d


def test_out_of_range_values_become_missing():
    train, _, audit = _clean_rows([_profile(respondent_id="a", gpa=34, tech_python=6, project_count=-1)])
    assert train.loc["a", ["gpa", "tech_python", "project_count"]].isna().all()
    assert audit.field_issues["gpa"]["invalid / out of range -> missing"] == 1


def test_degree_class_replaces_missing_gpa():
    train, _, _ = _clean_rows([_profile(respondent_id="a", gpa=None, degree_class="second_upper")])
    assert train.loc["a", "gpa"] == schema.DEGREE_CLASS_TO_GPA["second_upper"]
    assert train.loc["a", "gpa_source"] == "degree_class"


def test_no_leakage_into_model_features():
    schema.assert_no_leakage()
    with pytest.raises(AssertionError):
        schema.assert_no_leakage(list(schema.MODEL_FEATURES) + ["salary"])
    with pytest.raises(AssertionError):
        schema.assert_no_leakage(list(schema.MODEL_FEATURES) + ["gender"])


# --- input discovery ------------------------------------------------------------
def test_real_file_takes_priority_over_synthetic(tmp_path, synthetic_raw):
    raw, _ = synthetic_raw
    raw.to_csv(tmp_path / "graduates_SYNTHETIC.csv", index=False)
    assert clean.find_input("graduates", tmp_path) == (tmp_path / "graduates_SYNTHETIC.csv", True)
    raw.to_csv(tmp_path / "Graduate Questionnaire (Responses).csv", index=False)
    assert clean.find_input("graduates", tmp_path) == (tmp_path / "Graduate Questionnaire (Responses).csv", False)


# --- Google Forms export ----------------------------------------------------------
_STATUS_LABEL = {"graduated": "Graduated in 2025", "final_year": "Final-year student, expected graduation 2026"}


def _to_forms_export(canonical: pd.DataFrame) -> pd.DataFrame:
    """Render canonical rows the way a Google Forms CSV export looks (question headers, labels, d/m/Y dates)."""
    inverse = {f: {v: k for k, v in m.items()} for f, m in forms_adapter.LABEL_MAPS.items()}
    out = pd.DataFrame(index=canonical.index)
    out["Timestamp"] = pd.to_datetime(canonical["survey_date"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    for f in schema.RAW_FIELDS:
        if not f.item.startswith("Q"):
            continue
        col = canonical[f.name]
        skill = next((s for s in schema.SKILL_VOCABULARY if s.column == f.name), None)
        if skill is not None:
            q = "Q18. Please rate your proficiency" if skill.kind == "technical" else "Q19. Please rate yourself"
            out[f"{q} [{skill.label}]"] = col
            continue
        header = f"{f.item}. {f.description}"
        if f.name == "respondent_status":
            out[header] = col.map(_STATUS_LABEL)
        elif f.name == "degree_type":
            out[header] = col.map(lambda v: "Other: Robotics" if v == "other" else inverse[f.name].get(v, v))
        elif f.kind == "month":
            out[header] = pd.to_datetime(col, format="%Y-%m", errors="coerce").dt.strftime("15/%m/%Y")
        elif f.name in inverse:
            out[header] = col.map(lambda v, m=inverse[f.name]: m.get(v, v).title() if isinstance(v, str) else v)
        else:
            out[header] = col
    return out


def test_google_forms_export_cleans_identically(synthetic_raw):
    raw, _ = synthetic_raw
    raw = raw.copy()
    raw["survey_date"] = pd.to_datetime(raw["survey_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    forms = _to_forms_export(raw)
    lookup = synthetic.lookup_dict()

    canonical_train, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    forms_train, _ = clean.clean_graduates(forms, lookup, clean.Audit())

    cols = [c for c in canonical_train.columns if c != "respondent_id"]
    assert len(forms_train) == len(canonical_train)
    def comparable(df):
        df = df[cols].astype(object).where(df[cols].notna(), "<missing>")
        return df.astype(str).sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(comparable(forms_train), comparable(canonical_train))


def test_forms_month_and_year_dropdowns_are_combined():
    df = pd.DataFrame({"Q5a. Completion [Month]": ["October", "Mar"], "Q5a. Completion [Year]": ["2025", "2026"]})
    out, _ = forms_adapter.to_canonical(df, "graduates")
    assert list(out["graduation_date"]) == ["2025-10", "2026-03"]


def test_ambiguous_dates_follow_day_first_default():
    parsed = forms_adapter.parse_month(pd.Series(["03/04/2026", "05/06/2025"]))
    assert list(parsed) == (["2026-04", "2025-06"] if forms_adapter.DEFAULT_DAYFIRST else ["2026-03", "2025-05"])
    assert list(forms_adapter.parse_month(pd.Series(["10/25/2026"]))) == ["2026-10"]   # month-first detected
