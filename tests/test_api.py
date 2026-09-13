"""API tests: every endpoint returns its declared schema. A real uvicorn server is started in a thread and
called with `requests` (no extra test dependency). State is injected, so no artifacts are needed."""
import socket
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
import uvicorn

from api import schemas
from api.main import app
from core import clean, market, synthetic
from core.explain import Explainer
from tests.conftest import QUICK_PROFILE


@pytest.fixture(scope="module")
def base_url(tmp_path_factory, cox_predictor, train_df):
    tmp = tmp_path_factory.mktemp("api")
    processed = tmp / "processed"
    processed.mkdir()
    train_df.to_csv(processed / "graduates_train.csv", index=False)
    synthetic.generate_employers().to_csv(processed / "employers.csv", index=False)
    clean.clean_job_ads(synthetic.generate_job_ads())[0].to_csv(processed / "job_ads.csv", index=False)
    (processed / "dataset_info.json").write_text('{"is_synthetic": true, "sources": {}}', encoding="utf-8")
    market.build_reference_db(processed, tmp / "reference.db")

    cox_predictor.bundle.update({"label": "Cox PH", "params": {"penalizer": 0.1}, "selection_reason": "test",
                                 "dataset_info": {"is_synthetic": True, "training_n": len(train_df), "events": 1,
                                                  "censored": 1, "final_year_n": 0, "employers_n": 40,
                                                  "job_ads_n": 200}})
    comparison = pd.DataFrame([{"model": "cox_ph", "label": "Cox PH", "uno_c_mean": 0.66, "uno_c_sd": 0.03,
                                "harrell_c_mean": 0.65, "ibs_mean": 0.19, "cal_err_3m_mean": 0.03,
                                "cal_err_6m_mean": 0.05, "cal_err_12m_mean": 0.04}])
    app.state.core = {"predictor": cox_predictor, "explainer": Explainer(cox_predictor),
                      "reference": market.ReferenceData.load(tmp / "reference.db"), "comparison": comparison,
                      "info": cox_predictor.bundle["dataset_info"]}

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    app.state.core = None


def _check(resp, model):
    assert resp.status_code == 200, resp.text
    body = resp.json()
    model.model_validate(body)
    assert set(body) == set(model.model_fields), f"undeclared or missing fields in {model.__name__}"
    return body


def test_health(base_url):
    _check(requests.get(f"{base_url}/health"), schemas.HealthResponse)


def test_predict(base_url):
    body = _check(requests.post(f"{base_url}/predict", json=QUICK_PROFILE), schemas.PredictionResponse)
    assert body["is_synthetic"] is True and body["curve"]["months"][0] == 0


def test_explain(base_url):
    body = _check(requests.post(f"{base_url}/explain", json=QUICK_PROFILE), schemas.ExplainResponse)
    assert len(body["helping"]) <= 3 and len(body["hurting"]) <= 3


def test_whatif(base_url):
    body = _check(requests.post(f"{base_url}/whatif", json={"profile": QUICK_PROFILE,
                                                            "changes": {"internship_months": 9}}),
                  schemas.WhatIfResponse)
    assert body["applied_changes"] == {"internship_months": 9.0}


def test_market_alignment(base_url):
    body = _check(requests.post(f"{base_url}/market-alignment", json=QUICK_PROFILE), schemas.MarketResponse)
    assert len(body["skills"]) == 26


def test_peer_benchmark(base_url):
    _check(requests.post(f"{base_url}/peer-benchmark", json=QUICK_PROFILE), schemas.PeerResponse)


def test_model_info(base_url):
    body = _check(requests.get(f"{base_url}/model-info"), schemas.ModelInfoResponse)
    assert body["is_synthetic"] is True and body["synthetic_banner"]
    assert body["limitations"] and body["privacy"]


@pytest.mark.parametrize("payload", [dict(QUICK_PROFILE, gender="male"), dict(QUICK_PROFILE, salary=1),
                                     dict(QUICK_PROFILE, gpa=4.5), dict(QUICK_PROFILE, area_soft=7),
                                     dict(QUICK_PROFILE, degree_type="medicine")])
def test_invalid_profiles_rejected(base_url, payload):
    assert requests.post(f"{base_url}/predict", json=payload).status_code == 422


@pytest.mark.parametrize("changes", [{"university_type": 1}, {"project_count": 99}, {"gpa": 3.3}])
def test_invalid_whatif_rejected_with_message(base_url, changes):
    r = requests.post(f"{base_url}/whatif", json={"profile": QUICK_PROFILE, "changes": changes})
    assert r.status_code == 422 and r.json()["detail"]


def test_api_never_stores_profiles():
    """The API source contains no file, database or logging writes."""
    src = (Path(__file__).resolve().parents[1] / "api" / "main.py").read_text(encoding="utf-8")
    for forbidden in ("open(", ".to_csv", ".write_text", "INSERT", "sqlite3.connect", "logging.", "logger."):
        assert forbidden not in src, forbidden


def test_predict_for_graduate_still_looking(base_url):
    body = _check(requests.post(f"{base_url}/predict", json=dict(QUICK_PROFILE, months_since_graduation=5)),
                  schemas.PredictionResponse)
    assert body["months_since_graduation"] == 5 and body["remaining"] is not None


@pytest.mark.parametrize("payload", [dict(QUICK_PROFILE, respondent_status="final_year", months_since_graduation=3),
                                     dict(QUICK_PROFILE, months_since_graduation=30)])
def test_invalid_months_since_graduation_rejected(base_url, payload):
    r = requests.post(f"{base_url}/predict", json=payload)
    assert r.status_code == 422 and r.json()["detail"]


def test_suggestions(base_url):
    body = _check(requests.post(f"{base_url}/suggestions", json=QUICK_PROFILE), schemas.SuggestionsResponse)
    assert len(body["suggestions"]) <= 3 and body["note"]


def test_predict_has_milestones_and_market_has_resources(base_url):
    pred = requests.post(f"{base_url}/predict", json=QUICK_PROFILE).json()
    assert [m["months"] for m in pred["milestones"]] == [3, 6, 12, 24]
    gaps = requests.post(f"{base_url}/market-alignment", json=QUICK_PROFILE).json()["top_gaps"]
    assert all("resources" in g for g in gaps)
