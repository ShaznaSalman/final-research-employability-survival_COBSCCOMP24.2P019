"""System Usability Scale scoring, statistics, data checks and comment coding."""
import numpy as np
import pandas as pd
import pytest

from core import sus_study
from core.sus_study import ITEMS


def answers(*rows):
    return pd.DataFrame(rows, columns=ITEMS)


def form_rows(rows, overrides=None):
    """Rows shaped like load() output: short column names, every check passing unless overridden."""
    base = {"timestamp": "10/14/2026 11:04", "consent": ", ".join(sus_study.CONSENT),
            "eligibility": sus_study.ELIGIBLE, "B1": "Final-year undergraduate", "B2": "Information Technology",
            "B3": "State university", "B4": "Monthly", "C1": "My own details", "C2": "", "C3": "", "C4": ""}
    out = []
    for i, r in enumerate(rows):
        row = {**base, "code": f"P{i + 1:02d}", **dict(zip(ITEMS, r))}
        row.update((overrides or {}).get(i, {}))
        out.append(row)
    return pd.DataFrame(out)


def test_scoring_extremes_and_neutral():
    s = sus_study.sus_score(answers([5, 1] * 5, [1, 5] * 5, [3] * 10))
    assert s.tolist() == [100.0, 0.0, 50.0]


def test_scoring_matches_hand_calculation():
    # P01 in the study: (3+3+3+4+3+3+4+3+3+3) x 2.5 = 80
    assert sus_study.sus_score(answers([4, 2, 4, 1, 4, 2, 5, 2, 4, 2])).iloc[0] == 80.0


def test_grade_and_adjective_boundaries():
    assert sus_study.grade(84.1) == "A+" and sus_study.grade(84.0) == "A" and sus_study.grade(74.1) == "B"
    assert sus_study.grade(68) == "C" and sus_study.grade(51.6) == "F"
    assert sus_study.adjective(71.4) == "Good" and sus_study.adjective(71.3) == "OK"
    assert sus_study.adjective(90) == "Excellent"


def test_summary_interval_and_test_against_benchmark():
    s = sus_study.summary(pd.Series([60, 70, 80, 90.0]))
    assert s["mean"] == 75 and s["sd"] == pytest.approx(12.91, abs=0.01)
    half = 3.182 * 12.91 / 2
    assert s["ci_low"] == pytest.approx(75 - half, abs=0.05) and s["ci_high"] == pytest.approx(75 + half, abs=0.05)
    assert s["criterion_met"] and not s["ci_above_benchmark"]
    assert s["at_or_above_benchmark"] == 3 and 0 < s["p_two_sided"] < 1


def test_cronbach_alpha_perfectly_consistent_items():
    items = pd.DataFrame({f"i{k}": [0, 1, 2, 3, 4] for k in range(10)})
    assert sus_study.cronbach_alpha(items) == pytest.approx(1.0)


def test_checks_exclude_bad_rows():
    df = form_rows([[4, 2] * 5, [4, 2] * 5, [4, 2] * 5, [4, 2] * 5, [3] * 10, [4, 2] * 5],
                   {0: {"eligibility": "One or more of these does NOT apply to me"},
                      1: {"consent": "I agree to take part in this usability evaluation."},
                      2: {"SUS3": np.nan}, 3: {"code": "P05"}})
    c = sus_study.checks(df)
    assert c["duplicate_codes"] == ["P05"]
    assert c["straight_lined"] == ["P05"]
    assert set(c["excluded"]) == {"P01", "P02", "P03", "P05"}
    assert c["analysed"] == 1


def test_acquiescence_flag():
    c = sus_study.checks(form_rows([[5, 5] * 5, [4, 2] * 5]))
    assert c["acquiescent"] == ["P01"]


def test_comment_coding_finds_every_theme_and_reports_uncoded():
    df = pd.DataFrame({"code": ["P01", "P02", "P03"],
                       "C3": ["The confidence label needs a short explanation.", "Profile form is long on mobile.",
                              "Something unrelated"]})
    out = sus_study.code_comments(df, "C3", sus_study.ISSUE_THEMES)
    themes = {t["theme"]: t["participants"] for t in out["themes"]}
    assert themes["Terms and charts need explaining (onboarding)"] == ["P01"]
    assert themes["Mobile layout"] == ["P02"]
    assert out["uncoded"] == ["P03"]


def test_load_renames_form_columns(tmp_path):
    header = (["Timestamp", "Consent", "Participant code", "Eligibility", "B1. Which describes you?"]
              + [f"SUS{i}. Statement {i}" for i in range(1, 11)] + ["C3. What was confusing?"])
    path = tmp_path / "sus.csv"
    pd.DataFrame([["10/14/2026 11:04", "x", " p01 ", "y", "z"] + [3] * 10 + [""]], columns=header).to_csv(path, index=False)
    df = sus_study.load(path)
    assert {"code", "consent", "eligibility", "B1", "C3", *ITEMS} <= set(df.columns)
    assert df.loc[0, "code"] == "P01" and df.loc[0, "SUS1"] == 3


@pytest.mark.skipif(not sus_study.SOURCE.exists(), reason="SUS export not present (participant data is not committed)")
def test_real_export_reproduces_reported_result(tmp_path):
    r = sus_study.run(out_dir=tmp_path)
    s = r["summary"]
    assert r["checks"]["analysed"] == 12 and not r["checks"]["excluded"]
    assert s["mean"] == 76.25 and (s["ci_low"], s["ci_high"]) == (69.53, 82.97)
    assert (s["min"], s["max"]) == (60.0, 92.5) and s["grade"] == "B" and s["criterion_met"]
    assert not r["issues"]["uncoded"] and not r["useful"]["uncoded"]
    assert (tmp_path / "sus_study.md").exists() and (tmp_path / "figures" / "sus_scores.png").exists()
