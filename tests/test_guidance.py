"""Tests for the guidance features: milestone probabilities, suggested combinations and learning links."""
import numpy as np
import pandas as pd

from core import clean, market, schema, synthetic
from core.predict import MILESTONE_MONTHS
from core.suggest import MAX_EFFORT, MIN_MONTHS_SOONER, suggest
from core.whatif import apply_changes
from tests.conftest import QUICK_PROFILE

CAUSAL = ("will get", "will find", "causes", "guarantee", "you should")


def test_milestones_are_monotone_with_ranges(cox_predictor):
    ms = cox_predictor.predict(QUICK_PROFILE)["milestones"]
    assert [m["months"] for m in ms] == [t for t in MILESTONE_MONTHS if t <= cox_predictor.horizon]
    pcts = [m["employed_pct"] for m in ms]
    assert pcts == sorted(pcts) and all(0 <= p <= 100 for p in pcts)
    for m in ms:
        assert m["low_pct"] <= m["employed_pct"] <= m["high_pct"] or m["high_pct"] - m["low_pct"] <= 2
        assert f"{m['employed_pct']}%" in m["text"]
        assert not any(w in m["text"].lower() for w in CAUSAL)


def test_milestones_for_graduate_still_looking_skip_past_months(cox_predictor):
    ms = cox_predictor.predict(dict(QUICK_PROFILE, months_since_graduation=6))["milestones"]
    assert ms and all(m["months"] > 6 for m in ms)
    assert all("still looking" in m["text"] for m in ms)


def test_suggestions_respect_rules(large_cox_predictor):
    out = suggest(large_cox_predictor, QUICK_PROFILE)
    assert out["note"]
    used = set()
    base = large_cox_predictor.predict(QUICK_PROFILE)["median"]
    for s in out["suggestions"]:
        assert set(s["changes"]) <= set(schema.ACTIONABLE_FEATURES)
        assert not set(s["changes"]) & used                       # different features in each suggestion
        used |= set(s["changes"])
        assert 1 <= s["effort"] <= MAX_EFFORT and s["months_sooner"] >= MIN_MONTHS_SOONER
        changed = large_cox_predictor.predict(apply_changes(QUICK_PROFILE, s["changes"]))
        assert changed["median"]["months"] == s["new_median_months"]
        if not base["more_than"]:
            assert base["months"] - s["new_median_months"] == s["months_sooner"]
        assert not any(w in s["text"].lower() for w in CAUSAL)
        for f, v in s["changes"].items():                         # only increases, within the documented range
            assert v <= schema.ACTIONABLE_FEATURES[f].max


def test_no_suggestions_when_already_at_the_top(large_cox_predictor):
    top = dict(QUICK_PROFILE, internship_months=36, certifications_count=20, project_count=30, gpa=4.0,
               **{f"area_{a}": 5.0 for a in list(schema.SKILL_AREAS) + ["soft"]})
    out = suggest(large_cox_predictor, top)
    assert out["suggestions"] == [] and out["message"]


def _reference(tmp_path, train_df, with_resources=True):
    processed = tmp_path / "processed"
    processed.mkdir()
    train_df.to_csv(processed / "graduates_train.csv", index=False)
    synthetic.generate_employers().to_csv(processed / "employers.csv", index=False)
    clean.clean_job_ads(synthetic.generate_job_ads())[0].to_csv(processed / "job_ads.csv", index=False)
    (processed / "dataset_info.json").write_text('{"is_synthetic": true, "sources": {}}', encoding="utf-8")
    kw = {} if with_resources else {"resources_csv": tmp_path / "missing.csv"}
    market.build_reference_db(processed, tmp_path / "reference.db", **kw)
    return market.ReferenceData.load(tmp_path / "reference.db")


def test_learning_resources_cover_every_skill():
    res = pd.read_csv(market.LEARNING_RESOURCES)
    assert set(res["skill"]) == {s.key for s in schema.SKILL_VOCABULARY}
    assert res["url"].str.startswith("https://").all()
    assert not res[["title", "provider"]].isna().any().any()


def test_top_gaps_carry_resources(tmp_path, train_df):
    gaps = _reference(tmp_path, train_df).alignment(QUICK_PROFILE)["top_gaps"]
    assert gaps
    for g in gaps:
        assert 1 <= len(g["resources"]) <= market.MAX_RESOURCES
        assert all(set(r) == {"title", "provider", "url"} for r in g["resources"])


def test_resources_optional(tmp_path, train_df):
    gaps = _reference(tmp_path, train_df, with_resources=False).alignment(QUICK_PROFILE)["top_gaps"]
    assert all(g["resources"] == [] for g in gaps)
