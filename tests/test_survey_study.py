"""Questionnaire-study items: schema, Google Forms parsing, cleaning, synthetic answers and the analysis."""
import numpy as np
import pandas as pd
import pytest

from core import clean, forms_adapter, schema, survey_items, survey_study, synthetic, synthetic_survey


def test_every_questionnaire_item_is_in_the_schema():
    grad_items = {f.item.split(".")[0] for f in schema.GRADUATE_ALL_FIELDS if f.item.startswith("Q")}
    expected = {f"Q{i}" for i in range(1, 48)} - {"Q17b"}
    assert expected <= grad_items
    emp_items = {f.item.split(".")[0] for f in schema.EMPLOYER_ALL_FIELDS if f.item.startswith("Q")}
    assert {f"Q{i}" for i in range(1, 22)} <= emp_items
    assert all(f.role == "survey" for f in schema.SURVEY_FIELDS + schema.EMPLOYER_SURVEY_FIELDS)


def test_survey_items_are_never_model_features():
    survey = {f.name for f in schema.SURVEY_FIELDS + schema.EMPLOYER_SURVEY_FIELDS}
    for features in schema.FEATURE_SETS.values():
        assert not survey & set(features)


def test_forms_export_select_all_and_ranking_are_parsed():
    grad = pd.DataFrame({
        "Q22. Which online learning platforms have you used?": ["Coursera, Udemy, My own blog", None],
        "Q38. Do you have a LinkedIn profile?": ["Yes", "No"],
        "Q40. How employable do you currently consider yourself?": ["4", "2"],
        "Q43. What was the biggest challenge you faced when entering the IT job market?":
            ["Lack of experience", "Something else entirely"],
    })
    out, _ = forms_adapter.to_canonical(grad, "graduates")
    assert out["learning_platforms"].tolist()[0] == "coursera;udemy;other"
    assert pd.isna(out["learning_platforms"].tolist()[1])
    assert out["linkedin_profile"].tolist() == ["yes", "no"]
    assert out["self_employability"].tolist() == [4, 2]
    assert out["biggest_challenge"].tolist() == ["experience", "other"]

    emp = pd.DataFrame({"Q20. Which factors most distinguish a more employable candidate? [Internship experience]": ["1"],
                        "Q20. Which factors most distinguish a more employable candidate? [Cultural fit]": [None],
                        "Q10. Which technologies? (select all that apply)": ["Artificial Intelligence, IoT"]})
    out, _ = forms_adapter.to_canonical(emp, "employers")
    assert out["rank_internship"].iloc[0] == 1 and pd.isna(out["rank_cultural_fit"].iloc[0])
    assert out["future_tech"].iloc[0] == "ai;iot"


def test_multi_validation_drops_unknown_keys():
    audit = clean.Audit()
    df = pd.DataFrame({"languages": ["java;python", "java;cobol", None]})
    out = clean.validate_fields(df, [schema.GRADUATE_ALL_FIELDS[[f.name for f in schema.GRADUATE_ALL_FIELDS]
                                                               .index("languages")]], audit)
    assert out["languages"].tolist()[:2] == ["java;python", "java"]
    assert audit.field_issues["languages"]["invalid / out of range -> missing"] == 1


def test_files_without_survey_columns_are_still_accepted():
    raw, _, _ = synthetic.generate_graduates(40, 10, defects=False)
    out, notes = forms_adapter.to_canonical(raw, "graduates")
    assert all(f.name in out for f in schema.SURVEY_FIELDS)
    assert any("questionnaire-study columns" in n for n in notes)


def test_synthetic_answers_follow_row_content():
    raw, _, _ = synthetic.generate_graduates(60, 10, defects=False)
    a = synthetic_survey.add_graduate_survey(raw)
    b = synthetic_survey.add_graduate_survey(pd.concat([raw, raw.iloc[[0]].assign(respondent_id="DUP")],
                                                       ignore_index=True))
    cols = [f.name for f in schema.SURVEY_FIELDS]
    assert a[raw.columns].equals(raw)                                     # existing answers untouched
    assert b[cols].iloc[0].equals(b[cols].iloc[-1])                       # duplicate -> identical answers
    assert set(cols) <= set(a.columns)


@pytest.fixture(scope="module")
def study():
    raw, _, _ = synthetic.generate_graduates(240, 60, defects=False)
    raw = synthetic_survey.add_graduate_survey(raw)
    audit = clean.Audit()
    clean.clean_graduates(raw, synthetic.lookup_dict(), audit)
    emp = synthetic_survey.add_employer_survey(synthetic.generate_employers(120))
    emp, _ = clean.clean_employers(emp)
    return survey_study.analyse(audit.survey, emp, {"is_synthetic": True})


def test_study_has_every_section(study):
    assert {"repeated", "logic", "graduates_vs_employers", "self_perception", "artifact", "graduate_items",
            "employer_items", "employer_ranking"} <= set(study)
    assert len(study["repeated"]) >= 10
    for c in study["repeated"]:
        assert c["rho"] is None or -1 <= c["rho"] <= 1
        assert c["verdict"] == survey_study.verdict(c["rho"])
    for row in study["graduates_vs_employers"]["skill_gaps"]:
        assert 0 <= row["graduates_pct"] <= 100 and 0 <= row["q"] <= 1


def test_every_item_is_reported(study):
    reported = {x["field"] for x in study["graduate_items"]}
    assert {n for n, _, k, _, _ in survey_items.GRADUATE_ITEMS} == reported
    for x in study["graduate_items"]:
        for s in x.get("shares", []):
            assert s["pct"] is None or s["ci"][0] <= s["pct"] <= s["ci"][1]


def test_verdict_thresholds():
    assert survey_study.verdict(0.62) == "repeats the model question - keep one"
    assert survey_study.verdict(0.4) == "partly overlaps"
    assert survey_study.verdict(0.1) == "measures something different"
    assert survey_study.verdict(None) == "not enough answers"
    assert np.isnan(survey_study.has(pd.Series([None]), "java").iloc[0])


def test_new_study_sections(study):
    assert [f["mean"] for f in study["employer_factors"]] == sorted((f["mean"] for f in study["employer_factors"]),
                                                                   reverse=True)
    ff = study["fast_finders"]
    assert ff["n_fast"] > 0 and all(0 <= r["pct_fast"] <= 100 and 0 <= r["q"] <= 1 for r in ff["rows"])
    assert study["graduate_experience"]["search_platforms_employed"]
    assert study["employer_views"]["future_tech"]


def test_insights_are_personalised_and_never_touch_the_model(study):
    from core import insights
    weak = {"respondent_status": "graduated", "gpa": 3.0, "internship_months": 0, "project_count": 1,
            "certifications_count": 0, **{f"area_{a}": 1.5 for a in schema.SKILL_AREAS}, "area_soft": 2.0}
    strong = {**weak, **{f"area_{a}": 5.0 for a in schema.SKILL_AREAS}, "area_soft": 5.0}
    out_weak = insights.build(weak, {"has_portfolio": False}, study)
    out_strong = insights.build(strong, {}, study)
    assert out_weak["watch_outs"] and not out_strong["watch_outs"]
    assert all(w["your_rating"] <= insights.WEAK_RATING for w in out_weak["watch_outs"])
    portfolio = next(t for t in out_weak["fast_finders"] if t["key"] == "has_portfolio")
    assert portfolio["you"] is False
    assert portfolio["suggest"] == (portfolio["significant"] and portfolio["pct_fast"] > portfolio["pct_other"])
    assert {e for e in insights.EXTRAS} == set(schema_extras())


def schema_extras():
    from api.schemas import Extras
    return Extras.model_fields


def test_insights_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        body = {"profile": {"respondent_status": "graduated", "gpa": 3.2, "internship_months": 6, "project_count": 3,
                            "certifications_count": 1, "area_cloud_devops": 2},
                "extras": {"has_portfolio": True}}
        r = c.post("/insights", json=body)
        assert r.status_code == 200
        j = r.json()
        assert {"watch_outs", "growing", "fast_finders", "employers_value", "where_to_search"} <= set(j)
        assert c.post("/insights", json={**body, "extras": {"salary": 1}}).status_code == 422


def test_five_added_insights(study):
    from core import insights
    prof = {"respondent_status": "graduated", "gpa": 3.0, "internship_months": 3, "project_count": 2,
            "certifications_count": 1, **{f"area_{a}": 4.5 for a in schema.SKILL_AREAS}, "area_soft": 4.5}
    out = insights.build(prof, {}, study)
    assert {"certifications", "interviewers_said_lacking", "roles_hired", "project_areas",
            "underestimated_gaps"} <= set(out)
    growth = [c["employer_growth_pct"] or 0 for c in out["certifications"]]
    assert growth == sorted(growth, reverse=True)
    for u in out["underestimated_gaps"]:
        assert u["employers_pct"] - u["graduates_pct"] >= insights.UNDERESTIMATE_POINTS
    assert all(r["key"] != "none" for r in out["project_areas"])
    assert set(insights.GAP_SKILLS) == {r["label"] for r in study["graduates_vs_employers"]["skill_gaps"]}


def test_not_applicable_answers_are_missing_and_old_wording_still_maps():
    na_int = "Not applicable - I have not had a job interview yet"
    grad = pd.DataFrame({
        "Q31. Which skills were requested by employers / interviewers that you lacked?": [na_int, "Experience, " + na_int],
        "Q39. Have you received feedback from employers after interviews?": [na_int, "Yes"],
        "Q43. What was the biggest challenge you faced when entering the IT job market?":
            ["Not applicable - I have not started looking for IT jobs yet", "Lack of experience"],
    })
    out, _ = forms_adapter.to_canonical(grad, "graduates")
    assert pd.isna(out["lacked_skills"].iloc[0]) and out["lacked_skills"].iloc[1] == "experience"
    assert pd.isna(out["interview_feedback"].iloc[0]) and out["interview_feedback"].iloc[1] == "yes"
    assert pd.isna(out["biggest_challenge"].iloc[0]) and out["biggest_challenge"].iloc[1] == "experience"
    emp = pd.DataFrame({"Q11. How frequently does your organisation update the technical skills?":
                        ["Less than every year", "More often than once a year", "Every 1-2 years"]})
    out, _ = forms_adapter.to_canonical(emp, "employers")
    assert out["skill_update_freq"].tolist() == ["more_than_yearly", "more_than_yearly", "1-2_years"]


def test_attention_check_is_read_and_failures_are_excluded():
    raw, _, _ = synthetic.generate_graduates(60, 10, defects=False)
    audit = clean.Audit()
    base_train, _ = clean.clean_graduates(raw.copy(), synthetic.lookup_dict(), clean.Audit())
    checked = raw.copy()
    checked["attention_check"] = 4.0
    checked.loc[checked.index[:3], "attention_check"] = 2.0             # three careless respondents
    train, fy = clean.clean_graduates(checked, synthetic.lookup_dict(), audit)
    step = next(s for s in audit.steps if s["step"] == "Failed attention check")
    assert step["removed"] == 3
    assert len(train) + len(fy) <= len(base_train) + 10
    out, _ = forms_adapter.to_canonical(pd.DataFrame({
        "Q18. Please rate your proficiency [Attention check - please select 4 for this row]": ["4", "1"],
        "Q18. Please rate your proficiency [Python]": ["3", "5"]}), "graduates")
    assert out["attention_check"].tolist() == [4, 1] and out["tech_python"].tolist() == [3, 5]
    emp = synthetic.generate_employers(20)
    emp["attention_check"] = [4.0] * 18 + [1.0, 5.0]
    cleaned, info = clean.clean_employers(emp)
    assert info["failed_attention_check"] == 2 and len(cleaned) == 18


def test_data_without_attention_column_is_unchanged():
    raw, _, _ = synthetic.generate_graduates(60, 10, defects=False)
    a, _ = clean.clean_graduates(raw.copy(), synthetic.lookup_dict(), clean.Audit())
    assert "attention_check" not in raw and len(a) > 0


def test_dropdown_number_answers_are_read_as_numbers():
    grad = pd.DataFrame({
        "Q6. What is your final GPA?": ["3.2", "1.9 or below", "4.0"],
        "Q9. How many professional IT certifications have you obtained?": ["0", "10 or more", "3"],
        "Q11a. How many separate IT internships?": ["1", "5 or more", None],
        "Q12. In total, how many months of IT internship experience do you have?": ["6", "25 or more", "0"],
        "Q13. How many personal or academic IT projects?": ["16 or more", "2", "0"],
    })
    out, _ = forms_adapter.to_canonical(grad, "graduates")
    assert out["gpa"].tolist() == [3.2, 1.9, 4.0]
    assert out["certifications_count"].tolist() == [0, 10, 3]
    assert out["internship_months"].tolist() == [6, 25, 0]
    assert out["project_count"].tolist() == [16, 2, 0]
    assert out["internship_count"].tolist()[:2] == [1, 5]


def test_no_gpa_option_falls_back_to_degree_class():
    grad = pd.DataFrame({"Q6. What is your final GPA?": ["My degree does not use a GPA", "3.4"],
                         "Q6a. What is your final / expected degree class?": ["Second class upper", None]})
    out, _ = forms_adapter.to_canonical(grad, "graduates")
    assert pd.isna(out["gpa"].iloc[0]) and out["degree_class"].iloc[0] == "second_upper"
    assert out["gpa"].iloc[1] == 3.4
