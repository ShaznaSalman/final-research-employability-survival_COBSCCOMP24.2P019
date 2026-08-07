"""Tests for the synthetic data generator (core/synthetic.py)."""
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from core import schema, synthetic


@pytest.fixture(scope="module")
def generated():
    return synthetic.generate_graduates()


def test_raw_format_matches_schema(generated):
    raw, _, _ = generated
    assert list(raw.columns) == [f.name for f in schema.RAW_FIELDS]
    counts = raw["respondent_status"].value_counts()
    assert counts["graduated"] >= synthetic.N_GRADUATES and counts["final_year"] >= synthetic.N_FINAL_YEAR


def test_final_year_rows_have_no_outcome(generated):
    raw, _, _ = generated
    fy = raw[raw["respondent_status"] == "final_year"]
    assert fy["graduation_date"].isna().all()
    assert fy["first_job_start_date"].isna().all()
    assert fy["expected_graduation_date"].notna().all()


def test_reproducible_with_fixed_seed(generated):
    raw2, _, _ = synthetic.generate_graduates()
    pd.testing.assert_frame_equal(generated[0], raw2)


def test_censoring_and_pre_graduation_jobs(generated):
    _, truth, _ = generated
    in_market = truth[truth["in_market"]]
    censored_pct = 100 * (1 - in_market["event"].mean())
    assert 25 <= censored_pct <= 45
    pre_grad = (in_market["event"] == 1) & (in_market["duration_months"] == schema.JOB_BEFORE_GRADUATION_MONTHS)
    assert pre_grad.sum() > 0


def test_every_defect_is_logged_and_present(generated):
    raw, _, defects = generated
    kinds = {d["defect"] for d in defects}
    assert kinds == {"missing", "out_of_range", "inconsistent", "duplicate"}
    assert set(d["respondent_id"] for d in defects) <= set(raw["respondent_id"])
    non_id = [c for c in raw.columns if c != "respondent_id"]
    assert raw.duplicated(subset=non_id).sum() >= sum(d["defect"] == "duplicate" for d in defects)


def test_cox_recovers_true_effect_directions():
    """With a large clean sample, Cox estimates match the sign of every tested true effect.

    Only directions are required: the model sees noisy, style-affected self-ratings, so some true effects
    (e.g. soft skills) are attenuated and need very large samples to reach significance - as in real surveys.
    """
    _, truth, _ = synthetic.generate_graduates(n_graduates=6000, n_final_year=0, defects=False)
    df = truth[truth["in_market"]]
    X = pd.get_dummies(df[list(schema.MODEL_FEATURES) + ["duration_months", "event"]],
                       columns=["degree_group", "university_type"], drop_first=True, dtype=float)
    cph = CoxPHFitter().fit(X, "duration_months", "event")
    for feat in synthetic.TESTED_EFFECTS:
        true = synthetic.TRUE_LOG_HAZARD_RATIOS[feat]
        est, p = cph.params_[feat], cph.summary.loc[feat, "p"]
        assert np.sign(est) == np.sign(true), f"{feat}: estimated {est:.3f}, true {true} (p={p:.3f})"


def test_employers_and_job_ads_use_skill_vocabulary():
    emp = synthetic.generate_employers()
    assert list(emp.columns) == [f.name for f in schema.EMPLOYER_FIELDS]
    imp = emp.filter(like="imp_")
    assert imp.shape[1] == len(schema.SKILL_VOCABULARY) and imp.min().min() >= 1 and imp.max().max() <= 5

    from core.clean import clean_job_ads
    raw_ads = synthetic.generate_job_ads()
    ads, info = clean_job_ads(raw_ads)                      # ads are coded with natural names, as by a person
    assert len(raw_ads) > synthetic.N_JOB_ADS and len(ads) == synthetic.N_JOB_ADS   # duplicate postings removed
    mentioned = {s for row in ads["skills"] for s in row.split(";")}
    assert mentioned <= set(schema.SKILLS_BY_KEY)
    assert set(info["unknown_skill_tokens"]) <= set(synthetic.UNRECOGNISED_TOKENS)


def test_output_files_are_labelled_synthetic(tmp_path):
    out = synthetic.write_all(raw_dir=tmp_path / "raw", reference_dir=tmp_path / "ref")
    for path in out["files"].values():
        assert "_SYNTHETIC." in path
