"""Tests for prediction (core/predict.py)."""
import numpy as np
import pytest

from core import clean, evaluate, forms_adapter, predict, preprocess, schema, synthetic
from core.models import CURVE_TIMES


@pytest.fixture(scope="module")
def train():
    raw, _, _ = synthetic.generate_graduates()
    lookup = synthetic.lookup_dict()
    df, _ = clean.clean_graduates(raw, lookup, clean.Audit())
    return df


@pytest.fixture(scope="module")
def predictor(train):
    pre, model = evaluate.fit_all(train, {"penalizer": 0.1}, None, "cox_ph", 0)
    boots = []
    for b in range(20):
        idx = np.random.default_rng(b).integers(0, len(train), len(train))
        p, m = evaluate.fit_all(train.iloc[idx], {"penalizer": 0.1}, None, "cox_ph", b)
        boots.append({"preprocessor": p, "model": m})
    bundle = {"preprocessor": pre, "model": model, "max_follow_up_months": float(train["duration_months"].max())}
    return predict.Predictor(bundle, boots)


QUICK = {"respondent_status": "graduated", "gpa": 3.1, "degree_type": "software_engineering",
         "university_type": "non_state", "certifications_count": 1, "internship_months": 6, "project_count": 3,
         "area_programming": 3.5, "area_web_apis": 3, "area_data_ai": 3, "area_cloud_devops": 2.5,
         "area_quality_security": 2.5, "area_soft": 3.5}


def _months(v):
    return np.inf if v["more_than"] else v["months"]


def test_values_are_ordered_for_many_profiles(predictor, train):
    for r in predictor.predict_frame(train.iloc[:60]):
        assert _months(r["low"]) <= _months(r["median"]) <= _months(r["high"])
        lo, hi = r["uncertainty"]["median_interval_80"]
        assert _months(lo) <= _months(hi)


def test_full_prediction_shape(predictor):
    r = predictor.predict(QUICK)
    assert {"low", "median", "high", "uncertainty", "confidence", "curve", "time_reference"} <= set(r)
    assert r["confidence"] in ("High", "Medium", "Low")
    assert r["curve"]["months"] == list(range(predictor.horizon + 1))
    s = r["curve"]["still_searching"]
    assert s[0] == 1.0 and all(a >= b for a, b in zip(s, s[1:]))
    assert all(isinstance(r[k]["months"], int) or r[k]["more_than"] for k in ("low", "median", "high"))


def test_more_than_x_when_curve_never_reaches_level(predictor):
    flat = np.full((1, len(CURVE_TIMES)), 0.6)
    assert np.isinf(predict.crossing_times(flat, 0.25, 24)[0])
    assert predict._value(np.inf, 24) == {"months": None, "more_than": True, "display": "more than 24"}
    weak = dict(QUICK, gpa=2.0, internship_months=0, certifications_count=0, project_count=0,
                **{k: 1 for k in QUICK if k.startswith("area_")})
    r = predictor.predict(weak)
    assert r["high"]["more_than"] and r["high"]["display"] == f"more than {predictor.horizon}"


def test_crossings_beyond_horizon_are_not_reported():
    S = np.linspace(1, 0, len(CURVE_TIMES))[None, :]       # crosses 0.25 at t = 18
    assert predict.crossing_times(S, 0.25, horizon=12)[0] == np.inf
    assert predict.crossing_times(S, 0.25, horizon=24)[0] == pytest.approx(18.0)


def test_rounding_to_whole_months():
    assert predict.round_month(schema.JOB_BEFORE_GRADUATION_MONTHS) == 0
    assert predict.round_month(0.75) == 1
    assert predict.round_month(4.5) == 5 and predict.round_month(4.25) == 4


def test_final_year_wording(predictor):
    assert predictor.predict(dict(QUICK, respondent_status="final_year"))["time_reference"] == \
        "months after you graduate"
    assert predictor.predict(QUICK)["time_reference"] == "months after graduating"


def test_quick_mode_fills_items_but_detailed_items_win():
    df = predict.profile_frame(dict(QUICK, tech_python=5))
    assert df.loc[0, "tech_java"] == 3.5 and df.loc[0, "tech_python"] == 5
    assert df.loc[0, "soft_leadership"] == 3.5


def test_unknown_profile_field_is_rejected():
    with pytest.raises(ValueError):
        predict.profile_frame(dict(QUICK, salary=100000))


def test_confidence_thresholds():
    assert predict.confidence_label(1.5, True) == "High"
    assert predict.confidence_label(3.0, True) == "Medium"
    assert predict.confidence_label(6.0, True) == "Low"
    assert predict.confidence_label(1.0, False) == "Low"
    assert predict.confidence_label(np.inf, True) == "Low"


def test_fast_bootstrap_path_matches_slow_path(predictor, train):
    df = train.iloc[:15]
    fast = predictor.bootstrap_survival(df)
    for b, boot in enumerate(predictor.bootstraps[:5]):
        slow = boot["model"].predict_survival(boot["preprocessor"].transform(df), CURVE_TIMES)
        assert np.allclose(fast[b], slow, atol=1e-10)
    D = preprocess.derive_and_block(df)
    for boot, fast_cols in zip(predictor.bootstraps[:5], predictor._fast[:5]):
        assert np.allclose(fast_cols(D), boot["preprocessor"].transform(df).to_numpy())


def test_bootstrap_interval_handles_never_reached_medians(predictor):
    """Bootstrap curves that never reach 50% (inf) must give 'more than X', never NaN."""
    import warnings
    n_t = len(CURVE_TIMES)
    S = np.linspace(1, 0.3, n_t)[None, :]
    # 95% of bootstrap curves never reach 50%: both interval ends fall between infinite values.
    S_boot = np.stack([np.linspace(1, 0.3, n_t)[None, :] if b == 0 else np.full((1, n_t), 0.9) for b in range(20)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = predictor.summarise(S, S_boot)[0]
    lo, hi = r["uncertainty"]["median_interval_80"]
    assert hi["more_than"] and r["uncertainty"]["width_months"] is None and r["confidence"] == "Low"
    assert lo["more_than"] and lo["display"] == f"more than {predictor.horizon}"


@pytest.mark.parametrize("dist", ["weibull", "lognormal", "loglogistic"])
def test_fast_aft_bootstrap_matches_lifelines(train, dist):
    """The closed-form AFT bootstrap path must equal lifelines' own predictions."""
    from tests.conftest import build_predictor
    p = build_predictor(train, f"aft_{dist}", {}, n_boot=3)
    assert p._aft is not None and p._aft[0] == dist
    df = train.iloc[:12]
    fast = p.bootstrap_survival(df)
    for b, boot in enumerate(p.bootstraps):
        slow = boot["model"].predict_survival(boot["preprocessor"].transform(df.reset_index(drop=True)), CURVE_TIMES)
        assert np.allclose(fast[b], slow, atol=1e-10)
