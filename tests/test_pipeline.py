"""End-to-end checks through the real pipeline (clean -> preprocess -> model) and the fixed train / validation /
test split (no graduate reused across roles)."""
import numpy as np
import pandas as pd

from core import clean, evaluate, forms_adapter, preprocess, splits, synthetic
from core.models import make_model


def _cleaned(n_graduates: int, defects: bool = True) -> pd.DataFrame:
    raw, _, _ = synthetic.generate_graduates(n_graduates=n_graduates, n_final_year=20, defects=defects)
    lookup = synthetic.lookup_dict()
    train, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    return train


def test_cox_through_full_pipeline_recovers_known_directions():
    """Raw synthetic CSV format -> cleaning -> preprocessing -> Cox PH: every tested true effect has the right sign."""
    train = _cleaned(6000)
    X = preprocess.build_preprocessor().fit_transform(train)
    t, e = preprocess.outcome(train)
    coefs = make_model("cox_ph", penalizer=0.0).fit(X, t, e).coefficients
    for feat in synthetic.TESTED_EFFECTS:
        true = synthetic.TRUE_LOG_HAZARD_RATIOS[feat]
        assert np.sign(coefs.loc[feat, "coef"]) == np.sign(true), f"{feat}: {coefs.loc[feat, 'coef']:.3f} vs {true}"


def _spy_fits(monkeypatch) -> list[set]:
    """Record the respondent_ids of every preprocessing fit (every model fit starts with one)."""
    fitted_on: list[set] = []
    original = preprocess.build_preprocessor

    def spy(*args, **kwargs):
        pipe = original(*args, **kwargs)
        real_fit = pipe.fit

        def fit(X, y=None, **kw):
            fitted_on.append(set(X["respondent_id"]))
            return real_fit(X, y, **kw)
        pipe.fit = fit
        return pipe

    monkeypatch.setattr(preprocess, "build_preprocessor", spy)
    return fitted_on


def test_split_gives_every_graduate_exactly_one_role():
    df = _cleaned(400)
    role = splits.assign(df)
    assert role.notna().all() and set(role) == set(splits.ROLES)
    shares = role.value_counts(normalize=True)
    assert abs(shares["train"] - splits.TRAIN_SHARE) < 0.05 and abs(shares["validation"] - splits.VALID_SHARE) < 0.05
    by_role = df.groupby(role)["event"].mean()                  # stratified: similar event share in every role
    assert by_role.max() - by_role.min() < 0.10
    assert (splits.assign(df) == role).all()                    # fixed seed: the same split every time


def test_tuning_and_comparison_never_fit_on_validation_or_test(monkeypatch):
    """Spy on every preprocessing fit in the development step (incl. tuning): only training rows may be used."""
    df = _cleaned(300)
    role = splits.assign(df)
    train, valid = df[role == "train"], df[role == "validation"]
    fitted_on = _spy_fits(monkeypatch)
    monkeypatch.setattr(evaluate, "TUNED", ("cox_ph", "cox_elasticnet"))      # skip DeepSurv to keep the test fast

    rows, preds, params = evaluate.holdout(train, valid)

    train_ids = set(train["respondent_id"])
    assert len(fitted_on) > 1 and all(ids <= train_ids for ids in fitted_on)
    assert set(preds["respondent_id"]) == set(valid["respondent_id"])        # scored only on validation
    assert {r["model"] for r in rows} >= {"cox_ph", "cox_elasticnet", "aft_best"}
    assert "cox_ph" in params


def test_final_model_and_test_evaluation_use_each_set_once(monkeypatch):
    df = _cleaned(300)
    role = splits.assign(df)
    train, test = df[role == "train"].reset_index(drop=True), df[role == "test"].reset_index(drop=True)
    fitted_on = _spy_fits(monkeypatch)
    final = evaluate.fit_final(train, "cox_ph", {"cox_ph": {"penalizer": 0.1}})
    metrics, pred = evaluate.evaluate_test(final, train, test)
    assert fitted_on and all(ids <= set(train["respondent_id"]) for ids in fitted_on)   # never fitted on test
    assert set(pred["respondent_id"]) == set(test["respondent_id"])
    assert 0.4 < metrics["uno_c"] < 1 and metrics["uno_c_lo"] <= metrics["uno_c"] <= metrics["uno_c_hi"]
