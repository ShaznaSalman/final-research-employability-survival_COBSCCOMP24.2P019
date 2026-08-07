"""Tests for preprocessing (core/preprocess.py)."""
import numpy as np
import pandas as pd
import pytest

from core import clean, forms_adapter, preprocess, schema, synthetic


@pytest.fixture(scope="module")
def train_df():
    raw, _, _ = synthetic.generate_graduates()
    lookup = synthetic.lookup_dict()
    train, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    return train


def test_output_has_exactly_the_model_columns(train_df):
    X = preprocess.build_preprocessor().fit_transform(train_df)
    assert X.shape == (len(train_df), schema.model_parameter_count())
    assert not X.isna().any().any()
    forbidden = schema.POST_EMPLOYMENT_FIELDS | schema.FAIRNESS_ONLY_FIELDS
    assert not any(c.split("_")[0] in forbidden or c in forbidden for c in X.columns)
    assert "gender" not in " ".join(X.columns)


def test_fitted_on_training_fold_only(train_df):
    """Imputation medians and scaling means must come from the training fold, not the full data."""
    fold_train, fold_test = train_df.iloc[:150].copy(), train_df.iloc[150:].copy()
    fold_test["gpa"] = 1.0                       # shift test-fold values: must not affect fitted parameters
    _, X_test, pre = preprocess.fit_transform_fold(fold_train, fold_test)
    fitted = pre.named_steps["columns"].named_transformers_["num"]
    expected_mean = preprocess.raw_feature_frame(fold_train)["gpa"].median()
    assert fitted.named_steps["impute"].statistics_[0] == pytest.approx(expected_mean)
    scale = preprocess.numeric_scale(pre)
    train_gpa = preprocess.raw_feature_frame(fold_train)["gpa"].fillna(expected_mean)
    assert scale.loc["gpa", "mean"] == pytest.approx(train_gpa.mean())
    assert (X_test["gpa"] < -2).all()            # transformed with training statistics


def test_block_mean_requires_enough_answered_items():
    row = {c: 4 for c in preprocess.ITEM_BLOCKS["technical_mean"]}
    sparse = dict(row)
    for c in list(sparse)[:7]:                   # 10 of 17 answered < 2/3
        sparse[c] = np.nan
    mostly = dict(row)
    for c in list(mostly)[:5]:                   # 12 of 17 answered >= 2/3
        mostly[c] = np.nan
    out = preprocess.BlockMeans().transform(pd.DataFrame([row, sparse, mostly]))
    assert out["technical_mean"].tolist()[0] == 4
    assert np.isnan(out["technical_mean"].tolist()[1])
    assert out["technical_mean"].tolist()[2] == 4


def test_single_profile_transforms(train_df):
    pre = preprocess.build_preprocessor().fit(train_df)
    profile = train_df.iloc[[0]].drop(columns=["degree_group", "duration_months", "event"])
    X = pre.transform(profile)
    assert X.shape == (1, schema.model_parameter_count())


def test_unknown_category_maps_to_reference_level(train_df):
    pre = preprocess.build_preprocessor().fit(train_df)
    profile = train_df.iloc[[0]].copy()
    profile["university_type"] = "unknown_type"
    with pytest.warns(UserWarning):
        X = pre.transform(profile)
    assert (X.filter(like="university_type_") == 0).all().all()


def test_degree_class_and_no_internship_are_derived():
    df = pd.DataFrame({"degree_type": ["ai_ml"], "gpa": [np.nan], "degree_class": ["first"],
                       "internship_completed": ["no"], "internship_months": [np.nan]})
    out = preprocess.DerivedFeatures().transform(df)
    assert out.loc[0, "degree_group"] == "specialised"
    assert out.loc[0, "gpa"] == schema.DEGREE_CLASS_TO_GPA["first"]
    assert out.loc[0, "internship_months"] == 0


def test_missing_category_is_imputed_not_treated_as_reference(train_df):
    """None (e.g. an unmapped university name) must be imputed like NaN, never silently encoded as 'state'."""
    import warnings
    train = train_df.copy()
    train["university_type"] = "foreign_affiliated"                    # most frequent level = foreign_affiliated
    pre = preprocess.build_preprocessor().fit(train)
    profile = train.iloc[[0]].copy()
    profile["university_type"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # no 'unknown category' warning allowed
        X = pre.transform(profile)
    assert X.loc[X.index[0], "university_type_foreign_affiliated"] == 1
