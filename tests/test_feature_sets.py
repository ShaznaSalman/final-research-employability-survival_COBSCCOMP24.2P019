"""Tests for the extended (candidate) predictors, the pre-registered comparison rule and plain-language metrics."""
import numpy as np
import pandas as pd

from core import evaluate, feature_sets, preprocess, schema


def test_extended_set_preprocesses_candidates(train_df):
    X = preprocess.build_preprocessor(schema.EXTENDED_FEATURES).fit_transform(train_df)
    assert X.shape[1] == schema.model_parameter_count(schema.EXTENDED_FEATURES) == 16
    assert not X.isna().any().any()
    raw = preprocess.raw_feature_frame(train_df, schema.EXTENDED_FEATURES)
    assert set(raw["has_portfolio"].dropna().unique()) <= {0.0, 1.0}
    assert raw["prep_years"].dropna().between(0, 4).all()


def test_core_set_is_in_use_and_never_leaks():
    assert schema.MODEL_FEATURES == schema.FEATURE_SETS[schema.FEATURE_SET]
    for feats in schema.FEATURE_SETS.values():
        schema.assert_no_leakage(feats)
    assert not {"job_search_platforms", "linkedin_completeness", "cv_update_frequency"} & set(schema.EXTENDED_FEATURES)


def test_rule_requires_both_size_and_consistency():
    # small but consistent gain, large but inconsistent gain (few bootstrap resamples better), and a solid gain
    assert not feature_sets._meets({"delta_c": 0.005, "win_share": 0.95})
    assert not feature_sets._meets({"delta_c": 0.04, "win_share": 0.55})
    assert feature_sets._meets({"delta_c": 0.02, "win_share": 0.80})


def test_plain_language_metrics():
    lines = evaluate.plain_language_metrics(0.664, {6: 0.032})
    assert "about 66 times" in lines[0]
    assert "within 3 percentage points" in lines[1] and "6 months" in lines[1]
    assert "no single 'accuracy %'" in lines[-1]
    assert not any(w in " ".join(lines).lower() for w in ("hazard", "c-index", "censor"))


def test_transformed_set_derives_log_counts_and_gpa_cutoff(train_df):
    X = preprocess.build_preprocessor(schema.TRANSFORMED_FEATURES).fit_transform(train_df)
    assert X.shape[1] == schema.model_parameter_count(schema.TRANSFORMED_FEATURES) == 11
    raw = preprocess.raw_feature_frame(train_df, schema.TRANSFORMED_FEATURES)
    assert set(raw["gpa_3plus"].dropna().unique()) <= {0.0, 1.0}
    expected = np.log1p(train_df["certifications_count"].clip(lower=0))
    assert np.allclose(raw["log_certifications"].dropna(), expected[raw["log_certifications"].notna()])


def test_transformed_explanations_show_original_units(train_df):
    """Local explanations for log features must display the real count, not log(1 + count)."""
    from core import evaluate, predict
    from core.explain import Explainer
    from tests.conftest import QUICK_PROFILE
    pre, model = evaluate.fit_all(train_df, {"penalizer": 0.1}, None, "cox_ph", 0,
                                  features=schema.TRANSFORMED_FEATURES)
    X = pre.transform(train_df)
    bundle = {"name": "cox_ph", "model_name": "cox_ph", "preprocessor": pre, "model": model,
              "features": schema.TRANSFORMED_FEATURES, "X_mean": X.mean(), "background": X.head(20)}
    ex = Explainer(predict.Predictor(bundle, []))
    for profile in (dict(QUICK_PROFILE, certifications_count=0, project_count=1),
                    dict(QUICK_PROFILE, certifications_count=6, project_count=12)):
        items = ex.local(profile)["helping"] + ex.local(profile)["hurting"]
        for i in items:
            if i["feature"] == "log_certifications":
                assert i["value"] == f"{profile['certifications_count']:g}"
            if i["feature"] == "log_projects":
                assert i["value"] == f"{profile['project_count']:g}"
    assert {e["feature"] for e in ex.global_effects()} == set(schema.TRANSFORMED_FEATURES)
