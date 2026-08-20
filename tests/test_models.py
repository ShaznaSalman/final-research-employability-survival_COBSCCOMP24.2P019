"""Tests for the model wrappers and evaluation helpers (core/models, core/evaluate.py)."""
import pickle

import numpy as np
import pandas as pd
import pytest

from core import clean, evaluate, forms_adapter, preprocess, synthetic
from core.models import CURVE_TIMES, MODEL_NAMES, make_model, step_survival


@pytest.fixture(scope="module")
def data():
    raw, _, _ = synthetic.generate_graduates()
    lookup = synthetic.lookup_dict()
    train, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    X = preprocess.build_preprocessor().fit_transform(train)
    t, e = preprocess.outcome(train)
    return train, X, t, e


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_interface(name, data):
    _, X, t, e = data
    params = {"epochs": 20} if name == "deepsurv" else {}
    m = make_model(name, **params).fit(X, t, e)
    risk = m.predict_risk(X)
    S = m.predict_survival(X.iloc[:5], CURVE_TIMES)
    assert risk.shape == (len(X),) and np.isfinite(risk).all()
    assert S.shape == (5, len(CURVE_TIMES))
    assert np.allclose(S[:, 0], 1.0)                              # nobody employed at time 0
    assert (np.diff(S, axis=1) <= 1e-9).all()                      # survival never increases
    assert ((S >= 0) & (S <= 1)).all()
    clone = pickle.loads(pickle.dumps(m))
    assert np.allclose(clone.predict_risk(X), risk)


def test_no_forest_or_boosting_models():
    banned = ("forest", "boost", "xgb", "gbm", "ensemble", "bagging")
    assert not any(b in n for n in MODEL_NAMES for b in banned)


def test_step_survival_is_right_continuous():
    times, surv = np.array([1.0, 3.0]), np.array([[0.8, 0.5]])
    out = step_survival(times, surv, np.array([0.0, 0.99, 1.0, 2.0, 3.0, 10.0]))
    assert out.tolist() == [[1.0, 1.0, 0.8, 0.8, 0.5, 0.5]]


def test_median_from_curve_reports_nan_when_never_crossing():
    curves = np.vstack([np.linspace(1, 0.2, len(CURVE_TIMES)), np.full(len(CURVE_TIMES), 0.9)])
    med = evaluate.median_from_curve(curves)
    assert np.isfinite(med[0]) and np.isnan(med[1])


def test_calibration_table_by_tertile(data):
    train, X, t, e = data
    m = make_model("cox_ph").fit(X, t, e)
    S = m.predict_survival(X, np.array(evaluate.CALIBRATION_HORIZONS, float))
    oof = train[["duration_months", "event"]].copy()
    for j, h in enumerate(evaluate.CALIBRATION_HORIZONS):
        oof[f"S_{h}"] = S[:, j]
    cal = evaluate.calibration_table(oof)
    assert len(cal) == 3 * len(evaluate.CALIBRATION_HORIZONS)
    assert cal.groupby("horizon")["n"].sum().eq(len(oof)).all()
    assert cal["abs_error"].max() < 0.25                           # in-sample Cox should be roughly calibrated


def _comparison(rows):
    df = pd.DataFrame(rows, columns=["model", "uno_c_mean", "cal_err_mean"])
    df["candidate"] = True
    df["complexity"] = df["model"].map(evaluate.COMPLEXITY)
    return df


def test_selection_prefers_simpler_model_within_margin():
    comp = _comparison([("deepsurv", 0.70, 0.05), ("cox_ph", 0.69, 0.06), ("cox_elasticnet", 0.66, 0.04)])
    assert evaluate.select_model(comp)[0] == "cox_ph"


def test_selection_keeps_best_model_beyond_margin():
    comp = _comparison([("deepsurv", 0.75, 0.05), ("cox_ph", 0.69, 0.06)])
    assert evaluate.select_model(comp)[0] == "deepsurv"


def test_selection_breaks_complexity_ties_by_calibration():
    comp = _comparison([("cox_ph", 0.70, 0.08), ("aft_best", 0.695, 0.03)])
    assert evaluate.select_model(comp)[0] == "aft_best"


def test_ibs_handles_test_follow_up_beyond_training(data):
    """A test graduate followed longer than anyone in the training fold must not crash the Brier score."""
    train, X, t, e = data
    m = make_model("cox_ph").fit(X, t, e)
    short = t < t.max()                                      # training fold without the longest follow-up
    y_tr = preprocess.sksurv_y(train[short])
    y_te = preprocess.sksurv_y(train)                        # test fold includes it
    value = evaluate.ibs(y_tr, y_te, m, X, evaluate.truncation_tau(y_tr["time"]))
    assert np.isfinite(value) and 0 <= value <= 0.5
