"""Tests for the job-market study (core/market_study.py) on a small hand-made set of ads."""
import pandas as pd
import pytest

from core import market_study


@pytest.fixture()
def ads(tmp_path):
    rows = [("A1", "Software Engineering Intern", "C1", "2026-08-01", "javascript;git;communication"),
            ("A2", "Intern - QA", "C2", "2026-08-02", "software_testing;communication"),
            ("A3", "Associate Software Engineer", "C3", "2026-09-01", "javascript;web_development;teamwork"),
            ("A4", "UI/UX Designer Intern", "C4", "2026-09-02", "web_development"),
            ("A5", "Junior DevOps Engineer", "C5", "2026-09-03", "docker;devops_cicd;linux")]
    processed = tmp_path / "job_ads.csv"
    pd.DataFrame(rows, columns=["ad_id", "title", "company", "date", "skills"]).assign(source="other").to_csv(processed, index=False)
    return market_study.load(processed, tmp_path)


def test_role_and_seniority_rules():
    assert market_study.role_family("Intern - QA") == "Quality assurance"
    assert market_study.role_family("UI/UX Designer Intern") == "UI/UX design"
    assert market_study.role_family("Junior DevOps Engineer") == "Infrastructure & support"
    assert market_study.role_family("Associate Software Engineer") == "Software engineering"
    assert market_study.seniority("Software Engineering Intern") == "Intern / trainee"
    assert market_study.seniority("Associate Software Engineer") == "Junior / associate"


def test_prevalence_and_intervals(ads):
    r = market_study.analyse(ads)
    js = next(p for p in r["prevalence"] if p["key"] == "javascript")
    assert js["ads"] == 2 and js["pct"] == pytest.approx(40)
    assert js["ci_low"] < 40 < js["ci_high"]
    assert r["n_ads"] == 5 and r["n_companies"] == 5


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = market_study.bh([0.01, 0.04, 0.03, 0.5])
    assert all(0 <= x <= 1 for x in q)
    assert q[0] <= q[2] <= q[1] <= q[3]
