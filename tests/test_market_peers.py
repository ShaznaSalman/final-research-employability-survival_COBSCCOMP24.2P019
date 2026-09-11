"""Tests for market alignment, peer benchmarks and reference.db (core/market.py, core/peers.py)."""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

from core import clean, forms_adapter, market, peers, schema, synthetic
from core.clean import clean_job_ads


# --- peers ------------------------------------------------------------------------
def test_peer_medians_are_kaplan_meier_not_raw(train_df):
    stats = peers.compute_peer_stats(train_df)
    overall = stats[stats["group"] == "overall"].iloc[0]
    km = KaplanMeierFitter().fit(train_df["duration_months"], train_df["event"])
    assert overall["median"] == km.median_survival_time_
    raw_median_of_events = train_df.loc[train_df["event"] == 1, "duration_months"].median()
    assert overall["median"] != raw_median_of_events      # censoring matters: the naive median is biased


def test_small_groups_are_suppressed(train_df, quick_profile):
    small = train_df.copy()
    small.loc[small.index[:7], "degree_type"] = "ai_ml"
    small.loc[small.index[7:], "degree_type"] = small.loc[small.index[7:], "degree_type"].replace(
        {"ai_ml": "computer_science", "data_science": "computer_science", "cyber_security": "computer_science",
         "other": "computer_science"})
    small = small.drop(columns="degree_group")
    stats = peers.compute_peer_stats(small)
    spec = stats[(stats["group"] == "degree_group") & (stats["level"] == "specialised")].iloc[0]
    assert spec["n"] == 7 and spec["suppressed"] and pd.isna(spec["median"])

    out = peers.peer_benchmark(stats, dict(quick_profile, degree_type="ai_ml"))
    entry = next(g for g in out["groups"] if g["group"] == "degree_group")
    assert entry["suppressed"] and entry["median"] is None and "Not enough data" in entry["text"]


def test_peer_groups_never_use_gender():
    assert "gender" not in peers.GROUPS


def test_peer_benchmark_returns_profiles_groups(train_df, quick_profile):
    out = peers.peer_benchmark(peers.compute_peer_stats(train_df), quick_profile)
    got = {g["group"]: g["level"] for g in out["groups"]}
    assert got == {"overall": "all", "degree_group": "it_is", "university_type": "non_state", "internship": "yes",
                   "gpa_band": "3.0_to_3.49"}


# --- market ------------------------------------------------------------------------
def test_skill_demand_formula():
    employers = pd.DataFrame({f"imp_{s.key}": [5, 3] for s in schema.SKILL_VOCABULARY})
    ads = pd.DataFrame({"skills": ["python;sql_databases", "python", "git", "teamwork"]})
    d = market.skill_demand(employers, ads).set_index("skill")
    assert d.loc["python", "employer_importance"] == 4.0
    assert d.loc["python", "ads_pct"] == 50.0
    assert d.loc["python", "demand"] == pytest.approx(np.mean([0.75, 0.5]))
    assert d.loc["java", "ads_n"] == 0


def test_alignment_top_gaps_rule(tmp_path, quick_profile):
    ref = _build(tmp_path)
    out = ref.alignment(quick_profile)
    assert len(out["skills"]) == len(schema.SKILL_VOCABULARY)
    median_demand = ref.demand["demand"].median()
    gaps = out["top_gaps"]
    assert len(gaps) <= market.TOP_GAPS
    assert all(g["gap"] > market.GAP_MIN and g["demand"] >= median_demand for g in gaps)
    assert [g["gap"] for g in gaps] == sorted((g["gap"] for g in gaps), reverse=True)
    python = next(s for s in out["skills"] if s["skill"] == "python")
    assert python["your_rating"] == quick_profile["area_programming"]     # Quick mode fills items


def test_strong_profile_has_no_gaps(tmp_path, quick_profile):
    strong = {k: (5 if k.startswith("area_") else v) for k, v in quick_profile.items()}
    out = _build(tmp_path).alignment(strong)
    assert out["top_gaps"] == [] and out["message"]


# --- reference.db --------------------------------------------------------------------
def _build(tmp_path: Path) -> market.ReferenceData:
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    raw, _, _ = synthetic.generate_graduates()
    lookup = synthetic.lookup_dict()
    train, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    train.to_csv(processed / "graduates_train.csv", index=False)
    synthetic.generate_employers().to_csv(processed / "employers.csv", index=False)
    clean_job_ads(synthetic.generate_job_ads())[0].to_csv(processed / "job_ads.csv", index=False)
    (processed / "dataset_info.json").write_text('{"is_synthetic": true, "sources": {}}', encoding="utf-8")
    db = tmp_path / "reference.db"
    market.build_reference_db(processed, db)
    return market.ReferenceData.load(db)


def test_reference_db_contents_and_privacy(tmp_path):
    ref = _build(tmp_path)
    with sqlite3.connect(tmp_path / "reference.db") as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"employers", "job_ads", "job_ad_skills", "skill_demand", "peer_stats", "meta",
                          "learning_resources"}
        cols = {r[1] for t in tables for r in con.execute(f"PRAGMA table_info({t})")}
    # No respondent-level graduate data in the reference database.
    assert not cols & {"respondent_id", "gender", "duration_months", "event", "gpa"}
    assert ref.meta["is_synthetic"] == "true"
    assert set(ref.demand["skill"]) == {s.key for s in schema.SKILL_VOCABULARY}


def test_no_web_scraping_code():
    src = (Path(market.__file__)).read_text(encoding="utf-8")
    assert not re.search(r"import (requests|urllib|bs4|selenium|scrapy)|from (bs4|selenium|scrapy)", src)
