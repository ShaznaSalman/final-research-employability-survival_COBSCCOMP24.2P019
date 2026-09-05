"""Tests for the conditional prediction (graduates still looking) and actionable-only local explanations."""
import numpy as np
import pytest

from core import predict, schema
from core.explain import Explainer, actionable_model_features
from core.models import CURVE_TIMES
from core.whatif import what_if


def _months(v):
    return np.inf if v["more_than"] else v["months"]


# --- conditional prediction ----------------------------------------------------------
def test_condition_on_searching_math():
    S = np.linspace(1, 0.2, len(CURVE_TIMES))[None, :]
    m = 6
    C = predict.condition_on_searching(S, m)
    i = int(np.searchsorted(CURVE_TIMES, m))
    assert np.all(C[0, :i + 1] == 1.0)
    assert C[0, -1] == pytest.approx(S[0, -1] / S[0, i])
    assert (np.diff(C, axis=1) <= 1e-12).all()


def test_still_searching_shifts_prediction_later(cox_predictor, quick_profile):
    base = cox_predictor.predict(quick_profile)
    cond = cox_predictor.predict(dict(quick_profile, months_since_graduation=6))
    assert cond["months_since_graduation"] == 6
    for k in ("low", "median", "high"):
        assert _months(cond[k]) >= _months(base[k])
        assert _months(cond[k]) >= 6                                   # never before "now"
    assert _months(cond["low"]) <= _months(cond["median"]) <= _months(cond["high"])
    curve = cond["curve"]["still_searching"]
    assert all(v == 1.0 for v in curve[:7])                            # still looking up to month 6, by definition
    assert "still looking 6 months after graduating" in cond["headline"]


def test_remaining_months_are_relative_to_now(cox_predictor, quick_profile):
    r = cox_predictor.predict(dict(quick_profile, months_since_graduation=4))
    for k in ("low", "median", "high"):
        if not r[k]["more_than"]:
            assert r["remaining"][k]["months"] == max(1, r[k]["months"] - 4)
    assert cox_predictor.predict(quick_profile)["remaining"] is None


def test_few_graduates_still_looking_forces_low_confidence(cox_predictor, quick_profile):
    table = cox_predictor.bundle["at_risk_by_month"]
    m = next(m for m in range(1, 24) if table[m] < predict.MIN_AT_RISK)
    r = cox_predictor.predict(dict(quick_profile, months_since_graduation=m))
    assert r["confidence"] == "Low" and r["cautions"]


def test_invalid_months_since_graduation(cox_predictor, quick_profile):
    with pytest.raises(ValueError, match="cannot estimate"):
        cox_predictor.predict(dict(quick_profile, months_since_graduation=cox_predictor.horizon))
    with pytest.raises(ValueError, match="only applies to graduates"):
        cox_predictor.predict(dict(quick_profile, respondent_status="final_year", months_since_graduation=3))


def test_whatif_keeps_the_condition(cox_predictor, quick_profile):
    r = what_if(cox_predictor, dict(quick_profile, months_since_graduation=5), {"internship_months": 12})
    assert r["original"]["months_since_graduation"] == r["changed"]["months_since_graduation"] == 5


# --- actionable-only local explanations ------------------------------------------------
def test_local_explanations_show_only_actionable_features(cox_predictor, quick_profile):
    ex = Explainer(cox_predictor)
    for status in ("graduated", "final_year"):
        local = ex.local(dict(quick_profile, respondent_status=status))
        shown = {i["feature"] for i in local["helping"] + local["hurting"]}
        assert shown <= actionable_model_features(status)
        assert not shown & {"degree_group", "university_type"}


def test_gpa_is_actionable_only_for_final_year():
    assert "gpa" in actionable_model_features("final_year")
    assert "gpa" not in actionable_model_features("graduated")
    assert {"internship_months", "technical_mean", "soft_mean"} <= actionable_model_features("graduated")


def test_global_effects_still_cover_every_feature(cox_predictor):
    feats = {e["feature"] for e in Explainer(cox_predictor).global_effects()}
    assert feats == set(schema.MODEL_FEATURES)
