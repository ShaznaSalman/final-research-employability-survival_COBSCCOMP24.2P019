"""Tests for the secondary analyses: sensitivity variants, multiple imputation, temporal split, institution
clustering and competing risks (core/sensitivity.py, temporal.py, clustering.py, competing.py, inference.py)."""
import numpy as np
import pandas as pd
import pytest

from core import clean, clustering, competing, forms_adapter, inference, schema, sensitivity, synthetic, temporal
from core.clean import month_index


@pytest.fixture(scope="module")
def competing_df():
    raw, _, _ = synthetic.generate_graduates()
    audit = clean.Audit()
    train, _ = clean.clean_graduates(raw, synthetic.lookup_dict(), audit)
    return train, audit.competing


def test_q16a_leaves_the_main_training_data_unchanged(competing_df):
    raw, _, _ = synthetic.generate_graduates()
    without, _ = clean.clean_graduates(raw.drop(columns="alt_path_start_date"), synthetic.lookup_dict(), clean.Audit())
    pd.testing.assert_frame_equal(competing_df[0], without)
    assert "alt_path_start_date" not in schema.MODEL_FEATURES


def test_canonical_input_without_late_optional_column_is_accepted():
    raw, _, _ = synthetic.generate_graduates(n_graduates=20, n_final_year=0, defects=False)
    old = raw.drop(columns="alt_path_start_date")
    assert forms_adapter.is_canonical(old, "graduates")
    out, notes = forms_adapter.to_canonical(old, "graduates")
    assert out["alt_path_start_date"].isna().all() and any("alt_path_start_date" in n for n in notes)


def test_competing_outcome_rules(competing_df):
    train, cr = competing_df
    assert set(train["respondent_id"]) <= set(cr["respondent_id"])         # superset of the main data
    merged = cr.merge(train[["respondent_id", "event", "duration_months"]], on="respondent_id", how="left")
    it = merged[merged["cr_cause"] == 1]
    assert (it["event"] == 1).all() and np.allclose(it["cr_time"], it["duration_months"])
    alt = cr[cr["cr_cause"] == 2]
    assert len(alt) and alt["employment_status"].isin(schema.ALT_PATH_STATUSES).all()
    assert (cr["cr_time"] > 0).all()
    assert not ((cr["cr_cause"] == 0) & cr["employment_status"].isin(schema.NOT_IN_MARKET_STATUSES)).any()
    extra = cr[~cr["respondent_id"].isin(train["respondent_id"])]
    assert (extra["cr_cause"] == 2).all()                                  # only further-studies graduates added
    assert cr.loc[cr["alt_date_missing"], "cr_cause"].eq(0).all()


def test_cumulative_incidences_never_exceed_one(competing_df):
    _, cr = competing_df
    it, _ = competing.cif(cr, 1)
    alt, _ = competing.cif(cr, 2)
    assert all(0 <= a + b <= 1 + 1e-9 for a, b in zip(it, alt))
    assert it == sorted(it) and alt == sorted(alt)


def test_careless_and_heaped_detection(train_df):
    df = train_df.copy()
    cols = sensitivity.LIKERT
    df.loc[df.index[0], cols] = 3                                           # straight-liner
    flagged = sensitivity.careless(df)
    assert flagged.iloc[0]
    assert flagged.mean() < 0.15
    heaped = sensitivity.heaped(df)
    assert (df.loc[heaped, "event"] == 1).all() and df.loc[heaped, "duration_months"].isin([3, 6, 9, 12, 18, 24]).all()


def test_job_rule_variant_changes_only_job_before_completion(train_df):
    out = sensitivity.VARIANTS["job_1.0"][1](train_df)
    pre = train_df["job_before_graduation"].astype(bool)
    assert (out.loc[pre, "duration_months"] == 1.0).all()
    assert out.loc[~pre, "duration_months"].equals(train_df.loc[~pre, "duration_months"])


def test_rubin_rules():
    t = pd.DataFrame({"column": ["a", "b"], "log_hr": [0.2, -0.1], "se": [0.1, 0.2]})
    same = inference.rubin([t, t, t])
    assert np.allclose(same["log_hr"], t["log_hr"]) and np.allclose(same["se"], t["se"])
    t2 = t.assign(log_hr=[0.4, -0.1])
    pooled = inference.rubin([t, t2])
    assert pooled.loc[0, "se"] > 0.1 and np.isclose(pooled.loc[1, "se"], 0.2)


def test_multiple_imputation_returns_every_column(train_df):
    pre, X = inference.design(train_df)
    pooled = sensitivity.multiple_imputation(train_df, pre, m=3)
    assert list(pooled["column"]) == list(X.columns) and np.isfinite(pooled[["log_hr", "se"]]).all().all()


def test_temporal_split_is_earlier_vs_later(train_df):
    from core import splits
    role = splits.assign(train_df)
    train, valid = train_df[role == "train"], train_df[role == "validation"]
    early, late, cut = temporal.split(train, valid)
    assert len(early) and len(late)
    assert month_index(early["graduation_date"]).max() < month_index(late["graduation_date"]).min()
    assert set(early["respondent_id"]) <= set(train["respondent_id"])          # fitted on training graduates only
    assert set(late["respondent_id"]) <= set(valid["respondent_id"])           # evaluated on validation graduates only


def test_clustering_conservative_interval_never_narrower(train_df, tmp_path, monkeypatch):
    monkeypatch.setattr(clustering, "INFO_PATH", tmp_path / "missing.json")
    lookup = {forms_adapter.norm(n): i for n, _, i in synthetic.university_lookup().itertuples(index=False)}
    monkeypatch.setattr(clustering, "ARTIFACTS", tmp_path)
    out = clustering.run(train_df, lookup, write=True)
    assert 2 <= out["institutions"] <= 10
    tab = pd.read_csv(tmp_path / "institution_clustering.csv")
    assert (tab["lo_cons"] <= tab["lo"] + 1e-12).all() and (tab["hi_cons"] >= tab["hi"] - 1e-12).all()
    md = (tmp_path / "institution_clustering.md").read_text(encoding="utf-8")
    assert "Institution 1" in md and "Synthetic" not in md                 # institutions never named
    inst = clustering.institutions(train_df, lookup)
    canonical = {name for name, *_ in synthetic.UNIVERSITY_SPECS}
    assert inst.isin(canonical).mean() > 0.9                               # variants merged to canonical names


def test_cif_matches_km_without_competing_events_despite_tied_times(competing_df):
    """Regression: lifelines jitters tied times, which once made CIF(3 months) miss events at exactly 3."""
    from lifelines import KaplanMeierFitter
    _, cr = competing_df
    only_it = cr.assign(cr_cause=cr["cr_cause"].replace(2, 0))
    it, _ = competing.cif(only_it, 1)
    km = KaplanMeierFitter().fit(only_it["cr_time"], only_it["cr_cause"] == 1)
    assert np.allclose(it, [1 - float(km.survival_function_at_times(t).iloc[0]) for t in competing.TIMES], atol=1e-6)
    with_alt, _ = competing.cif(cr, 1)
    assert all(a <= b + 1e-9 for a, b in zip(with_alt, it))
