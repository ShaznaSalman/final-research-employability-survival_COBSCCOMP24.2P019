"""Tests for the fairness check (core/fairness.py)."""
import numpy as np
import pandas as pd

from core import fairness, preprocess, schema
from core.models import make_model


def _oof(train_df, repeats=2):
    """In-sample Cox predictions shaped like artifacts/oof_predictions.csv (good enough to test the mechanics)."""
    X = preprocess.build_preprocessor().fit_transform(train_df)
    t, e = preprocess.outcome(train_df)
    S6 = make_model("cox_ph").fit(X, t, e).predict_survival(X, np.array([6.0]))[:, 0]
    one = pd.DataFrame({"respondent_id": train_df["respondent_id"], "S_6": S6,
                        "duration_months": t, "event": e})
    return pd.concat([one.assign(repeat=r) for r in range(repeats)], ignore_index=True)


def test_gender_is_never_a_model_feature():
    assert "gender" not in schema.MODEL_FEATURES
    assert "gender" in fairness.GROUP_VARS


def test_subgroup_table_and_suppression(train_df):
    table = fairness.subgroup_metrics(_oof(train_df), train_df)
    assert set(table["group"]) == {"all", "gender", "university_type"}
    small = table[table["n"] < fairness.MIN_GROUP_N]
    assert small["suppressed"].all() and small["uno_c"].isna().all()
    shown = table[~table["suppressed"]]
    assert shown["uno_c"].between(0, 1).all()
    assert (shown["uno_c_lower"] <= shown["uno_c_upper"]).all()
    overall = table[table["group"] == "all"].iloc[0]
    assert overall["n"] == len(train_df)


def test_missing_gender_is_its_own_group(train_df):
    df = train_df.copy()
    df.loc[df.index[:25], "gender"] = np.nan
    table = fairness.subgroup_metrics(_oof(df), df)
    assert "missing" in set(table.loc[table["group"] == "gender", "level"])


def test_flags_trigger_on_large_gaps():
    table = pd.DataFrame([
        {"group": "gender", "level": "a", "suppressed": False, "uno_c": 0.75, "uno_c_lower": 0.72,
         "uno_c_upper": 0.78, "cal_diff_6m": 0.15},
        {"group": "gender", "level": "b", "suppressed": False, "uno_c": 0.60, "uno_c_lower": 0.55,
         "uno_c_upper": 0.65, "cal_diff_6m": 0.0},
    ])
    out = fairness.flags(table)
    assert any("C-index gap" in f and "do not overlap" in f for f in out)
    assert any("over-predicts" in f for f in out)


def test_report_is_marked_backend_only(train_df):
    table = fairness.subgroup_metrics(_oof(train_df), train_df)
    md = fairness.render_report(table, {"label": "Cox PH", "repeats": 2}, {"is_synthetic": True}, [])
    assert "backend only" in md and "SYNTHETIC" in md
