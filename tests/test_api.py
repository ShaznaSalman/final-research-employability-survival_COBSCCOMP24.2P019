"""API tests: every endpoint returns its declared schema. A real uvicorn server is started in a thread and
called with `requests` (no extra test dependency). State is injected, so no artifacts are needed."""
import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import uvicorn

from api import main as api_main, schemas
from api.main import app
from core import clean, market, synthetic
from core.explain import Explainer
from frontend import api_client
from tests.conftest import QUICK_PROFILE


@pytest.fixture(scope="module")
def base_url(tmp_path_factory, cox_predictor, train_df):
    tmp = tmp_path_factory.mktemp("api")
    processed = tmp / "processed"
    processed.mkdir()
    artifacts = tmp / "artifacts"
    artifacts.mkdir()
    (artifacts / "job_market_study.json").write_text(json.dumps({
        "roles": {"Software engineering": 30},
        "by_role": {"Software engineering": {"python": 100, "teamwork": 50}},
    }), encoding="utf-8")
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
                      "info": cox_predictor.bundle["dataset_info"], "artifacts": artifacts}

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


def test_report(base_url, monkeypatch):
    captured = {}
    build_report = api_main.build_report

    def capture_report(*args):
        captured["pathways"] = args[-1]
        return build_report(*args)

    monkeypatch.setattr(api_main, "build_report", capture_report)
    response = requests.post(f"{base_url}/report", json=QUICK_PROFILE)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["cache-control"] == "no-store"
    assert response.content.startswith(b"%PDF-")
    assert "attachment" in response.headers["content-disposition"]
    assert captured["pathways"]["recommendations"][0]["role"] == "Software engineering"


def test_frontend_api_client_accepts_pdf(monkeypatch):
    response = Mock(status_code=200, headers={"content-type": "application/pdf"}, content=b"%PDF-report")
    monkeypatch.setattr(api_client.requests, "post", lambda *args, **kwargs: response)
    assert api_client.post_bytes("/report", QUICK_PROFILE) == b"%PDF-report"


def test_frontend_api_client_rejects_non_pdf(monkeypatch):
    response = Mock(status_code=200, headers={"content-type": "application/json"}, content=b"{}")
    monkeypatch.setattr(api_client.requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(api_client.APIError, match="invalid PDF"):
        api_client.post_bytes("/report", QUICK_PROFILE)


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
    assert body["disclaimer"] == (
        "Decision support only - not a hiring tool. Based on survey data from Sri Lankan IT graduates."
    )
    assert all("620" not in limitation for limitation in body["limitations"])


@pytest.mark.parametrize("payload", [dict(QUICK_PROFILE, gender="male"), dict(QUICK_PROFILE, salary=1),
                                     dict(QUICK_PROFILE, gpa=4.5), dict(QUICK_PROFILE, area_soft=7),
                                     dict(QUICK_PROFILE, degree_type="medicine")])
def test_invalid_profiles_rejected(base_url, payload):
    assert requests.post(f"{base_url}/predict", json=payload).status_code == 422


@pytest.mark.parametrize("changes", [{"university_type": 1}, {"project_count": 99}, {"gpa": 3.3}])
def test_invalid_whatif_rejected_with_message(base_url, changes):
    r = requests.post(f"{base_url}/whatif", json={"profile": QUICK_PROFILE, "changes": changes})
    assert r.status_code == 422 and r.json()["detail"]


@pytest.mark.parametrize("field", ["gpa", "degree_type", "university_type", "certifications_count",
                                   "internship_months", "project_count"])
@pytest.mark.parametrize("endpoint", ["/predict", "/explain", "/peer-benchmark", "/report"])
def test_missing_core_input_rejected_in_plain_words(base_url, endpoint, field):
    r = requests.post(f"{base_url}{endpoint}", json={k: v for k, v in QUICK_PROFILE.items() if k != field})
    assert r.status_code == 422 and "please give" in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("swap", [{"gpa": None, "degree_class": "first"},
                                  {"internship_months": None, "internship_completed": "no"}])
def test_core_input_alternatives_accepted(base_url, swap):
    payload = {k: v for k, v in {**QUICK_PROFILE, **swap}.items() if v is not None}
    assert requests.post(f"{base_url}/explain", json=payload).status_code == 200


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


def test_chat(base_url):
    body = _check(requests.post(f"{base_url}/chat", json={"profile": QUICK_PROFILE, "message": "Which skills should "
                                                                                                  "I learn?"}),
                  schemas.ChatResponse)
    assert body["intent"] == "skills" and body["reply"] and body["follow_ups"]


def test_chat_answers_general_questions_without_a_profile(base_url):
    body = _check(requests.post(f"{base_url}/chat", json={"message": "How does the model work?"}),
                  schemas.ChatResponse)
    assert body["intent"] == "method" and "past IT graduates" in body["reply"]


def test_chat_requires_a_profile_for_personal_estimates(base_url):
    body = _check(requests.post(f"{base_url}/chat", json={"profile": None, "message": "How long will it take me?"}),
                  schemas.ChatResponse)
    assert body["intent"] == "estimate" and "Complete your profile" in body["reply"]


@pytest.mark.parametrize("payload", [{"profile": QUICK_PROFILE, "message": ""},
                                     {"profile": QUICK_PROFILE, "message": "x" * 501},
                                     {"profile": QUICK_PROFILE, "message": "hi", "history": []}])
def test_chat_rejects_invalid_requests(base_url, payload):
    assert requests.post(f"{base_url}/chat", json=payload).status_code == 422
