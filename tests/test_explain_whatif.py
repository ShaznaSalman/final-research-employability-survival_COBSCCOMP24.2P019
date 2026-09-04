"""Tests for explanations (core/explain.py), what-if (core/whatif.py) and wording (core/wording.py)."""
import re

import numpy as np
import pytest

from core import schema, wording
from core.explain import Explainer, MIN_CONTRIBUTION, TOP_K
from core.whatif import WhatIfError, apply_changes, what_if
from tests.conftest import build_predictor


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)


WORDING = [v for k, v in vars(wording).items() if k.isupper()]
CAUSAL = re.compile(r"\bwill\b|\bif you\b|\bcaus|\bleads? to\b|\b(increase|reduce|improve|boost) your\b|"
                    r"\bguarantee[sd]? (that|you)\b", re.IGNORECASE)
JARGON = re.compile(r"hazard|censor|c-index|concordance|coefficient|schoenfeld", re.IGNORECASE)


# --- wording ---------------------------------------------------------------------
def test_all_wording_is_associational_and_jargon_free():
    for text in _all_strings(WORDING):
        assert not CAUSAL.search(text), f"causal wording: {text!r}"
        assert not JARGON.search(text), f"jargon: {text!r}"


def test_generated_explanations_are_associational(cox_predictor, quick_profile):
    out = Explainer(cox_predictor).explain(quick_profile)
    shown = [i["text"] for i in out["global"] + out["helping"] + out["hurting"]] + [out["note"]]
    for text in shown:                                   # user-facing text only (not machine field values)
        assert not CAUSAL.search(text), text
        assert not JARGON.search(text), text


# --- explanations ------------------------------------------------------------------
def test_local_top3_helping_and_hurting(cox_predictor, quick_profile):
    local = Explainer(cox_predictor).local(quick_profile)
    assert len(local["helping"]) <= TOP_K and len(local["hurting"]) <= TOP_K
    assert all(i["score"] >= MIN_CONTRIBUTION for i in local["helping"])
    assert all(i["score"] <= -MIN_CONTRIBUTION for i in local["hurting"])
    assert [i["score"] for i in local["helping"]] == sorted((i["score"] for i in local["helping"]), reverse=True)
    features = {i["feature"] for i in local["helping"] + local["hurting"]}
    assert features <= set(schema.MODEL_FEATURES) and "gender" not in features


def test_cox_contributions_sum_to_linear_predictor_difference(cox_predictor, quick_profile):
    from core.predict import profile_frame
    ex = Explainer(cox_predictor)
    df = profile_frame(quick_profile)
    X = ex.pre.transform(df).iloc[0]
    coef = ex.model.coefficients["coef"].reindex(ex.columns)
    assert ex.contributions(df).sum() == pytest.approx(float(coef @ (X - ex.x_mean)))


def test_global_effects_have_intervals_containing_estimate(cox_predictor):
    effects = Explainer(cox_predictor).global_effects()
    assert {e["column"] for e in effects} == set(Explainer(cox_predictor).columns)
    for e in effects:
        assert e["kind"] == "hazard_ratio"
        assert e["lower"] <= e["ratio"] <= e["upper"]
        assert (e["ratio"] > 1) == (e["direction"] == "sooner")
        assert e["uncertain"] == (e["lower"] <= 1 <= e["upper"])


def test_elastic_net_uses_bootstrap_intervals(train_df):
    p = build_predictor(train_df, "cox_elasticnet", {"alpha": 0.02, "l1_ratio": 0.5}, n_boot=15)
    for e in Explainer(p).global_effects():
        assert np.isfinite([e["lower"], e["upper"]]).all() and e["lower"] <= e["upper"]


def test_aft_uses_time_ratio_wording(train_df, quick_profile):
    ex = Explainer(build_predictor(train_df, "aft_weibull", {}, n_boot=3))
    effects = ex.global_effects()
    assert all(e["kind"] == "time_ratio" for e in effects)
    for e in effects:                                            # wording follows the estimated direction
        assert ("shorter" in e["text"]) == (e["ratio"] < 1)
        assert ("longer" in e["text"]) == (e["ratio"] >= 1)
    assert ex.local(quick_profile)["method"].startswith("coefficient")


def test_deepsurv_uses_shap(train_df, quick_profile):
    p = build_predictor(train_df, "deepsurv", {"width": 8, "layers": 1, "dropout": 0.1, "lr": 0.01, "epochs": 30},
                        n_boot=2)
    local = Explainer(p).local(quick_profile)
    assert local["method"] == "shap"
    assert all(i["feature"] in schema.MODEL_FEATURES for i in local["helping"] + local["hurting"])


# --- what-if ------------------------------------------------------------------------
@pytest.mark.parametrize("changes", [{"university_type": "state"}, {"gender": "male"}, {"degree_type": "ai_ml"},
                                     {"tech_python": 5}])
def test_whatif_rejects_non_actionable(quick_profile, changes):
    with pytest.raises(WhatIfError, match="not an actionable"):
        apply_changes(quick_profile, changes)


@pytest.mark.parametrize("changes", [{"project_count": 40}, {"internship_months": -1}, {"area_soft": 6},
                                     {"certifications_count": "many"}])
def test_whatif_rejects_out_of_range(quick_profile, changes):
    with pytest.raises(WhatIfError):
        apply_changes(quick_profile, changes)


def test_gpa_change_only_for_final_year_within_limit(quick_profile):
    with pytest.raises(WhatIfError, match="final-year"):
        apply_changes(quick_profile, {"gpa": 3.3})
    fy = dict(quick_profile, respondent_status="final_year")
    assert apply_changes(fy, {"gpa": 3.3})["gpa"] == 3.3
    with pytest.raises(WhatIfError, match="at most"):
        apply_changes(fy, {"gpa": 3.6})


def test_area_change_overrides_its_items(quick_profile):
    p = dict(quick_profile, tech_docker=1)
    new = apply_changes(p, {"area_cloud_devops": 4.5})
    for item in ("tech_docker", "tech_linux", "tech_git", "tech_cloud_computing", "tech_devops_cicd"):
        assert new[item] == 4.5
    assert "tech_python" not in new                                  # other areas untouched


def test_adding_internship_does_not_lengthen_predicted_time(large_cox_predictor, quick_profile):
    for months in (6, 9, 12):
        r = what_if(large_cox_predictor, quick_profile, {"internship_months": months})
        assert r["change_months"]["median"] is None or r["change_months"]["median"] <= 0


def test_whatif_flags_profiles_outside_training_range(cox_predictor, quick_profile):
    r = what_if(cox_predictor, quick_profile, {"certifications_count": 10, "project_count": 15})
    assert any("Nobody in our data" in c for c in r["cautions"])
    assert what_if(cox_predictor, quick_profile, {"internship_months": 6})["cautions"] == []


def test_whatif_reports_change(cox_predictor, quick_profile):
    r = what_if(cox_predictor, quick_profile, {"internship_months": 9, "area_cloud_devops": 3.5})
    o, c = r["original"]["median"], r["changed"]["median"]
    if not (o["more_than"] or c["more_than"]):
        assert r["change_months"]["median"] == c["months"] - o["months"]
    assert r["message"] and r["applied_changes"] == {"internship_months": 9.0, "area_cloud_devops": 3.5}
