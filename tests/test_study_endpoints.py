"""The study endpoints (/job-market, /research) and the simulation study's building blocks."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import simulation_study, synthetic


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_research_endpoint_shapes(client):
    r = client.get("/research")
    assert r.status_code == 200
    j = r.json()
    assert j["models"] and any(m["model"] == j["selected_model"] for m in j["models"])
    assert all(0 <= p["employed"] <= 1 for p in j["km"])
    assert {c["horizon"] for c in j["calibration"]} <= {3, 6, 12}


def test_job_market_endpoint(client):
    r = client.get("/job-market")
    assert r.status_code == 200
    j = r.json()
    assert j["n_ads"] == sum(j["roles"].values())
    assert all(p["ci_low"] <= p["pct"] <= p["ci_high"] for p in j["prevalence"])


def test_simulation_rep_and_summary():
    lookup = synthetic.lookup_dict()
    reps = [simulation_study.one_rep(200, 7 + i, lookup) for i in range(3)]
    s = simulation_study.summarise([r for r in reps if r])
    assert s["reps"] == 3 and 0.5 < s["c_index_mean"] < 1
    gpa = s["effects"]["gpa"]
    assert gpa["true"] > 0 and 0 <= gpa["power"] <= 1
    assert "false_positive_rate" in s["effects"]["university_type[non_state]"]
    assert "coverage" in s["effects"]["internship_months"]


def test_accuracy_method_runs_on_train_and_validation():
    from core import accuracy_experiments, splits
    train, valid = splits.load("train"), splits.load("validation")
    kind, params, risk = accuracy_experiments._one("cox_core", train, valid, 0)
    assert kind == "cox_core" and risk is not None and len(risk) == len(valid)
