"""Tests for the experiment-tracking reproducibility log (core/tracking.py)."""
import json

import pytest

from core import evaluate, schema, tracking


@pytest.fixture
def fake_artifacts(tmp_path):
    art = tmp_path / "artifacts"; (art / "figures").mkdir(parents=True)
    (art / "selection.json").write_text(json.dumps({"label": "AFT Log-Normal", "uno_c": 0.619, "uno_c_validation": 0.693,
                                                    "uno_c_test_ci": [0.557, 0.683], "bootstrap_models": 200}))
    (art / "test_metrics.json").write_text(json.dumps({"n": 132, "uno_c": 0.619}))
    (art / "learning_curve.json").write_text(json.dumps({"n_train": 620, "n_validation": 133,
                                                         "curve": [{"fraction": 0.25, "uno_c_mean": 0.641}]}))
    (art / "deepsurv_curves.json").write_text(json.dumps({"best_epoch": 2, "train_loss": [2.73, 2.66], "val_loss": [4.08, 4.05]}))
    (art / "model_comparison.csv").write_text(
        "model,uno_c_mean,ibs_mean,cal_err_3m_mean,cal_err_6m_mean,cal_err_12m_mean,tuned_params\n"
        'aft_best,0.693,0.171,0.126,0.118,0.088,"{""distribution"": ""lognormal""}"\n')
    (art / "figures" / "km_overall.png").write_bytes(b"\x89PNG")
    return art


def test_default_is_no_tracking():
    assert tracking.make(None) is None and tracking.make("none") is None


def test_local_run_logs_saved_results(tmp_path, fake_artifacts):
    t = tracking.LocalTracker(root=tmp_path / "runs", name="r1")
    summary = tracking.log_pipeline_run(t, art=fake_artifacts)
    assert summary["test_uno_c"] == 0.619 and summary["test_result"] == tracking.TEST_LABEL
    params = json.loads((t.dir / "params.json").read_text())
    assert params["split"] == {"train": 620, "validation": 133, "test": 132}
    lines = (t.dir / "metrics.jsonl").read_text().splitlines()
    assert any('"deepsurv/holdout_loss"' in ln for ln in lines) and (t.dir / "figures" / "km_overall.png").exists()
    report = (t.dir / "report.html").read_text(encoding="utf-8")
    assert "aft_best" in report and "<polyline" in report and "loaded, not recomputed" in report


def test_tracking_never_evaluates_the_test_set(tmp_path, fake_artifacts, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the locked test set must not be evaluated again")
    for name in ("evaluate_test", "run", "holdout", "fit_final"):
        monkeypatch.setattr(evaluate, name, forbidden)
    tracking.log_pipeline_run(tracking.LocalTracker(root=tmp_path / "runs", name="r2"), art=fake_artifacts)


def test_no_survey_fields_are_logged(tmp_path, fake_artifacts):
    t = tracking.LocalTracker(root=tmp_path / "runs", name="r3")
    tracking.log_pipeline_run(t, art=fake_artifacts)
    logged = "".join(p.read_text(encoding="utf-8") for p in t.dir.glob("*.json*"))
    for field in schema.RAW_FIELDS:
        assert f'"{field.name}"' not in logged
