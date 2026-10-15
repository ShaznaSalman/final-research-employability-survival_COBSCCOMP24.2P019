"""Report-only scripts added for the thesis appendices: they run on synthetic data and produce sane output.

Covers core/learning_curve.py, error_analysis.py, deepsurv_curves.py, tuning_results.py, time_dependent_metrics.py,
eda_figures.py, evaluation_figures.py, sample_comparison.py and baselines.py. The scripts' run() functions read the
real artifacts; these tests exercise their computations and report writers on the synthetic fixture instead.
"""
import json

import numpy as np
import pandas as pd
import pytest

from core import (baselines, deepsurv_curves, eda_figures, error_analysis, evaluate, evaluation_figures,
                  learning_curve, preprocess, sample_comparison, time_dependent_metrics, tuning_results)
from core.models.deepsurv import DeepSurvModel


@pytest.fixture(scope="module")
def halves(train_df):
    """Synthetic graduates split in two: fit on the first half, score on the second."""
    df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cut = len(df) // 2
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- learning curve
def test_learning_curve_score_is_a_c_index(halves):
    tr, va = halves
    c = learning_curve.score(tr, va)
    assert 0.0 < c < 1.0


def test_learning_curve_report_and_figure(tmp_path):
    curve = [{"fraction": f, "n_train": int(100 * f), "subsamples": 1 if f == 1 else 20, "uno_c_mean": 0.6 + 0.05 * f,
              "uno_c_min": 0.55, "uno_c_max": 0.7} for f in learning_curve.FRACTIONS]
    r = {"model": "AFT Log-Normal", "n_train": 100, "n_validation": 30, "curve": curve}
    learning_curve.plot(r, tmp_path / "lc.png")
    assert (tmp_path / "lc.png").stat().st_size > 0
    md = learning_curve.to_markdown(r)
    assert "| 25% | 25 |" in md and "Gain at each step" in md


# ---------------------------------------------------------------------------------------------- error analysis
def test_error_analysis_predictions_and_summary(halves):
    tr, va = halves
    p = error_analysis.predictions(tr, va)
    assert len(p) == len(va)
    assert (p["predicted_median"] > 0).all() and (p["predicted_median"] <= error_analysis.HORIZON).all()
    s = error_analysis.summarise(p)
    assert s["events"] + s["censored"] == len(va)
    assert sum(b["n"] for b in s["bands"]) == s["events"]
    assert s["worst_over"] + s["worst_under"] <= s["worst_n"]
    assert "Error analysis" in error_analysis.to_markdown(s)


def test_error_analysis_summary_on_known_values():
    p = pd.DataFrame({"actual_months": [0.5, 2.0, 5.0, 10.0, 20.0, 6.0], "event": [1, 1, 1, 1, 1, 0],
                      "predicted_median": [4.0, 4.0, 5.0, 6.0, 7.0, 3.0],
                      **{c: [3.0] * 6 for c in error_analysis.PROFILE}})
    s = error_analysis.summarise(p)
    assert s["censored_past_median"] == 1                       # searching 6 months, predicted median 3
    by_band = {b["band"]: b for b in s["bands"]}
    assert by_band["at or before completion"]["mean_error"] == pytest.approx(3.5)
    assert by_band["more than 12 months"]["mean_error"] == pytest.approx(-13.0)


# ---------------------------------------------------------------------------------------------- DeepSurv curves
def test_deepsurv_records_training_and_holdout_loss(train_df):
    pre = preprocess.build_preprocessor().fit(train_df)
    t, e = preprocess.outcome(train_df)
    m = DeepSurvModel(epochs=5, patience=3, seed=0).fit(pre.transform(train_df), t, e)
    log = m.model_.log.to_pandas()
    assert {"train_loss", "val_loss"} <= set(log.columns) and 1 <= len(log) <= 5


def test_deepsurv_report_and_figure(tmp_path):
    log = pd.DataFrame({"train_loss": [2.7, 2.6, 2.55, 2.5], "val_loss": [4.1, 4.05, 4.08, 4.1]})
    r = {"params": {"width": 32, "layers": 1, "dropout": 0.3, "lr": 0.01}, "patience": 15, "max_epochs": 300,
         "epochs_run": 4, "best_epoch": 2, "train_loss_best": 2.6, "val_loss_best": 4.05,
         "train_loss": log["train_loss"].tolist(), "val_loss": log["val_loss"].tolist()}
    deepsurv_curves.plot(log, 1, r, tmp_path / "ds.png")
    assert (tmp_path / "ds.png").stat().st_size > 0
    assert "epoch 2" in deepsurv_curves.to_markdown(r)


# ---------------------------------------------------------------------------------------------- tuning results
def test_tuning_results_report_marks_chosen_settings():
    t = pd.DataFrame([
        {"model": "cox_ph", "label": "Cox PH", "params": '{"penalizer": 0.5}', "aic": np.nan, "uno_c": 0.67,
         "chosen": True, "matches_main_analysis": True},
        {"model": "cox_ph", "label": "Cox PH", "params": '{"penalizer": 0.1}', "aic": np.nan, "uno_c": np.nan,
         "chosen": False, "matches_main_analysis": None},
        {"model": "cox_elasticnet", "label": "Elastic Net Cox", "params": '{"alpha": 0.005, "l1_ratio": 0.1}',
         "aic": np.nan, "uno_c": 0.66, "chosen": True, "matches_main_analysis": False},
        {"model": "deepsurv", "label": "DeepSurv", "params": '{"width": 32}', "aic": np.nan, "uno_c": 0.68,
         "chosen": True, "matches_main_analysis": True},
        {"model": "aft_best", "label": "AFT", "params": '{"distribution": "lognormal"}', "aic": 2724.8, "uno_c": 0.69,
         "chosen": True, "matches_main_analysis": True}])
    md = tuning_results.to_markdown(t, 620, 133)
    assert "| penalizer 0.5 | 0.670 | **yes** |" in md
    assert "failed to fit" in md and "NO - check" in md
    assert tuning_results._fmt_params('{"alpha": 0.005, "l1_ratio": 0.1}') == "alpha 0.005, l1_ratio 0.1"


# ---------------------------------------------------------------------------------------------- time-dependent metrics
def test_time_dependent_metrics_report_and_figure(tmp_path):
    times = list(range(1, 6))
    r = {"n_validation": 50, "n_brier": 48, "times": times,
         "models": [{"model": "aft_best", "label": "AFT Log-Normal", "uno_c": 0.69, "auc_3m": 0.75, "auc_6m": 0.79,
                     "auc_12m": 0.8, "auc_mean": 0.77, "ibs": 0.17}],
         "brier_selected": [0.1, 0.2, 0.19, 0.18, 0.15], "brier_km": [0.12, 0.22, 0.2, 0.17, 0.16], "ibs_km": 0.2}
    time_dependent_metrics.plot(r, tmp_path / "b.png")
    assert (tmp_path / "b.png").stat().st_size > 0
    md = time_dependent_metrics.to_markdown(r)
    assert "below the baseline at 4 of 5 monthly time points" in md


# ---------------------------------------------------------------------------------------------- EDA figures
def test_eda_statistics(train_df):
    cm = eda_figures.spearman(train_df)
    assert np.allclose(np.diag(cm.to_numpy()), 1.0) and (cm.abs() <= 1 + 1e-9).all().all()
    sh = eda_figures.likert_shares(train_df)
    for d in sh.values():
        assert np.allclose(d[[str(k) for k in range(1, 6)]].sum(axis=1), 1.0)
        assert d["mean"].is_monotonic_increasing
    sc = eda_figures.scree(train_df)
    for block in ("technical", "soft"):
        assert sc[block]["factors"] >= 1
        assert sc[block]["eigenvalues"] == sorted(sc[block]["eigenvalues"], reverse=True)


def test_parallel_analysis_thresholds_decrease():
    pa = eda_figures.parallel_analysis(np.random.default_rng(0), 200, 6, iters=50)
    assert pa.shape == (6,) and (np.diff(pa) < 0).all()


def test_eda_figures_are_written(train_df, tmp_path):
    eda_figures.plot_correlation(eda_figures.spearman(train_df), len(train_df), tmp_path / "c.png")
    eda_figures.plot_likert(eda_figures.likert_shares(train_df), len(train_df), tmp_path / "l.png")
    eda_figures.plot_scree(eda_figures.scree(train_df), tmp_path / "s.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("c.png", "l.png", "s.png"))


# ---------------------------------------------------------------------------------------------- evaluation figures
def test_aft_fit_reports_every_distribution(train_df, tmp_path):
    res = evaluation_figures.aft_fit(train_df, tmp_path / "aft.png")
    assert set(res) == set(evaluation_figures.DIST_STYLE)
    assert all(0 <= v["mean_abs_gap"] <= v["max_abs_gap"] <= 1 for v in res.values())


def test_fairness_rows_parse_the_report(tmp_path):
    md = tmp_path / "fairness_report.md"
    md.write_text("| Group | Level |\n|---|---|\n"
                  "| all | All | 132 | 100 | 0.619 (0.559-0.677) | 0.61 | 0.77 | -0.17 |\n"
                  "| gender | female | 48 | 35 | 0.724 (0.597-0.814) | 0.63 | 0.74 | -0.12 |\n"
                  "| gender | missing | 3 | 3 | suppressed (n < 10) | - | - | - |\n", encoding="utf-8")
    f = evaluation_figures.fairness_rows(md)
    assert list(f["level"]) == ["All", "female"]                   # suppressed rows are skipped
    assert f.loc[1, "uno_c"] == pytest.approx(0.724) and f.loc[1, "hi"] == pytest.approx(0.814)
    evaluation_figures.plot_fairness(f, tmp_path / "fair.png")
    assert (tmp_path / "fair.png").stat().st_size > 0


def test_robustness_figure(tmp_path):
    r = pd.DataFrame([{"check": "Sensitivity: Main analysis", "uno_c": 0.69, "sd": 0.03, "n": 133},
                      {"check": "Later cohorts", "uno_c": 0.72, "sd": np.nan, "n": 68}])
    evaluation_figures.plot_robustness(r, 0.69, tmp_path / "rob.png")
    assert (tmp_path / "rob.png").stat().st_size > 0


# ---------------------------------------------------------------------------------------------- sample comparison
def test_sample_comparison_measures(train_df):
    rows = {r["measure"]: r for r in sample_comparison.compare(train_df)}
    assert set(rows) >= {"female_share_pct", "mean_months_after_completion", "employed_before_completion_pct",
                         "employed_at_survey_pct", "km_median_months"}
    for k in ("female_share_pct", "employed_before_completion_pct", "employed_at_survey_pct"):
        assert 0 <= rows[k]["sample"] <= 100
    assert rows["employed_at_survey_pct"]["sample"] == pytest.approx(100 * train_df["event"].mean(), abs=0.05)
    md = sample_comparison.to_markdown(list(rows.values()))
    assert "Published" in md


# ---------------------------------------------------------------------------------------------- baselines
@pytest.mark.parametrize("key", list(baselines.SETS))
def test_each_baseline_fits_and_ranks(halves, key):
    tr, va = halves
    _, feats = baselines.SETS[key]
    pre, m = evaluate.fit_all(tr, {}, baselines.DISTRIBUTION, "aft_best", 0, features=feats)
    risk = m.predict_risk(pre.transform(va))
    y_tr, y_va = preprocess.sksurv_y(tr), preprocess.sksurv_y(va)
    assert len(risk) == len(va)
    assert 0.0 < evaluate.uno_c(y_tr, y_va, risk, evaluate.truncation_tau(y_tr["time"])) < 1.0


def test_baselines_report():
    rows = [{"model": "km", "label": "Kaplan-Meier (no model)", "uno_c": 0.5, "lo": None, "hi": None,
             "diff_vs_selected": -0.19, "diff_lo": None, "diff_hi": None, "selected_better_share": None},
            {"model": "gpa_only", "label": "GPA only", "uno_c": 0.64, "lo": 0.57, "hi": 0.7, "diff_vs_selected": -0.05,
             "diff_lo": -0.09, "diff_hi": -0.02, "selected_better_share": 0.99},
            {"model": "selected", "label": "Selected", "uno_c": 0.69, "lo": 0.63, "hi": 0.75, "diff_vs_selected": 0.0,
             "diff_lo": None, "diff_hi": None, "selected_better_share": None}]
    md = baselines.to_markdown({"n_train": 620, "n_validation": 133, "bootstrap": 1000, "rows": rows})
    assert "GPA alone reaches 0.640" in md and "| -0.050 (-0.090 to -0.020) | 99% |" in md
    json.dumps(rows)                                              # report rows stay JSON-serialisable
