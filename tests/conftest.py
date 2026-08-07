"""Shared fixtures: synthetic training data and small self-contained predictors (no artifacts needed)."""
import numpy as np
import pytest

from core import clean, evaluate, forms_adapter, predict, synthetic


@pytest.fixture(scope="session")
def train_df():
    raw, _, _ = synthetic.generate_graduates()
    lookup = synthetic.lookup_dict()
    df, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    return df


def build_predictor(train, name: str, params: dict, n_boot: int = 20) -> predict.Predictor:
    aft = name.split("_", 1)[1] if name.startswith("aft_") else None
    fit_name = "aft_best" if aft else name
    pre, model = evaluate.fit_all(train, params, aft, fit_name, 0)
    boots = []
    for b in range(n_boot):
        idx = np.random.default_rng(b).integers(0, len(train), len(train))
        p, m = evaluate.fit_all(train.iloc[idx], params, aft, fit_name, b)
        boots.append({"preprocessor": p, "model": m})
    X = pre.transform(train)
    bundle = {"name": name, "model_name": name, "preprocessor": pre, "model": model,
              "max_follow_up_months": float(train["duration_months"].max()),
              "X_mean": X.mean(), "background": X.sample(40, random_state=0),
              "training_summary": evaluate.training_summary(train),
              "at_risk_by_month": {m: int((train["duration_months"] >= m).sum()) for m in range(25)}}
    return predict.Predictor(bundle, boots)


@pytest.fixture(scope="session")
def cox_predictor(train_df):
    return build_predictor(train_df, "cox_ph", {"penalizer": 0.1})


@pytest.fixture(scope="session")
def large_cox_predictor():
    """Cox fitted on 1500 synthetic graduates - large enough for the true effect directions to be recovered."""
    raw, _, _ = synthetic.generate_graduates(n_graduates=1500, n_final_year=0)
    df, _ = clean.clean_graduates(raw, synthetic.lookup_dict(), clean.Audit())
    return build_predictor(df, "cox_ph", {"penalizer": 0.1}, n_boot=10)


QUICK_PROFILE = {
    "respondent_status": "graduated", "gpa": 3.1, "degree_type": "information_technology",
    "university_type": "non_state", "certifications_count": 0, "internship_months": 3, "project_count": 2,
    "area_programming": 3.5, "area_web_apis": 3, "area_data_ai": 3, "area_cloud_devops": 2,
    "area_quality_security": 2.5, "area_soft": 4,
}


@pytest.fixture
def quick_profile():
    return dict(QUICK_PROFILE)
