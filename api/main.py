"""FastAPI app - a thin layer over core/. No logic is duplicated here.

Loads the final model, the bootstrap models and reference.db ONCE at start-up.
No authentication, no rate limiting, and NO storage of submitted profiles (nothing is written or logged).

Run:  uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (ExplainResponse, HealthResponse, InsightsRequest, MarketResponse, ModelInfoResponse,
                         PeerResponse, PredictionResponse, Profile, SuggestionsResponse, WhatIfRequest, WhatIfResponse)
from core import evaluate, insights, predict, wording
from core.explain import Explainer
from core.market import ReferenceData
from core.suggest import suggest
from core.whatif import WhatIfError, what_if

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


def load_state(artifacts: Path | None = None, reference_db: Path | None = None) -> dict:
    artifacts = artifacts or _path("EP_ARTIFACTS_DIR", "artifacts")
    reference_db = reference_db or _path("EP_REFERENCE_DB", "data/reference/reference.db")
    predictor = predict.Predictor.load(artifacts)
    comparison = pd.read_csv(artifacts / "model_comparison.csv") if (artifacts / "model_comparison.csv").exists() \
        else pd.DataFrame()
    study = artifacts / "questionnaire_study.json"
    return {"predictor": predictor, "explainer": Explainer(predictor), "reference": ReferenceData.load(reference_db),
            "comparison": comparison, "info": predictor.bundle.get("dataset_info", {}), "artifacts": artifacts,
            "questionnaire": json.loads(study.read_text(encoding="utf-8")) if study.exists() else None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(app.state, "core", None):          # tests may inject state before start-up
        app.state.core = load_state()
    yield


app = FastAPI(title="Time-to-Employment Decision Support API",
              description="Survival-analysis predictions for Sri Lankan IT graduates. Decision support only - "
                          "not a hiring or screening tool. Submitted profiles are never stored.",
              version="1.0", lifespan=lifespan)


def _core(request: Request) -> dict:
    return request.app.state.core


def _synthetic(state) -> bool:
    return bool(state["info"].get("is_synthetic", True))


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    s = _core(request)
    return {"status": "ok", "model_loaded": s["predictor"] is not None,
            "bootstrap_models": len(s["predictor"].bootstraps), "is_synthetic": _synthetic(s)}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(profile: Profile, request: Request):
    s = _core(request)
    try:
        return {**s["predictor"].predict(profile.to_core()), "is_synthetic": _synthetic(s)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(profile: Profile, request: Request):
    out = _core(request)["explainer"].explain(profile.to_core())
    out["global_effects"] = out.pop("global")
    return out


@app.post("/whatif", response_model=WhatIfResponse)
def whatif_endpoint(body: WhatIfRequest, request: Request):
    s = _core(request)
    try:
        out = what_if(s["predictor"], body.profile.to_core(), body.changes)
    except (WhatIfError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    for k in ("original", "changed"):
        out[k]["is_synthetic"] = _synthetic(s)
    return out


@app.post("/suggestions", response_model=SuggestionsResponse)
def suggestions_endpoint(profile: Profile, request: Request):
    try:
        return suggest(_core(request)["predictor"], profile.to_core())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/market-alignment", response_model=MarketResponse)
def market_endpoint(profile: Profile, request: Request):
    return _core(request)["reference"].alignment(profile.to_core())


@app.post("/insights")
def insights_endpoint(req: InsightsRequest, request: Request):
    """Survey context for a profile: employer and graduate questionnaire answers. Never changes the prediction."""
    study = _core(request).get("questionnaire")
    if not study:
        raise HTTPException(404, "Questionnaire study not found - run `python -m core.survey_study`.")
    return insights.build(req.profile.to_core(), req.extras.model_dump(exclude_none=True), study)


@app.post("/peer-benchmark", response_model=PeerResponse)
def peers_endpoint(profile: Profile, request: Request):
    return _core(request)["reference"].peer_benchmark(profile.to_core())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    s = _core(request)
    bundle, info, comp = s["predictor"].bundle, s["info"], s["comparison"]
    metrics = [{"model": r["model"], "label": r["label"], "uno_c_mean": r["uno_c_mean"], "uno_c_sd": r["uno_c_sd"],
                "harrell_c_mean": r["harrell_c_mean"], "ibs_mean": r["ibs_mean"],
                "cal_err_3m": r["cal_err_3m_mean"], "cal_err_6m": r["cal_err_6m_mean"],
                "cal_err_12m": r["cal_err_12m_mean"], "selected": r["model"] == bundle["name"]}
               for _, r in comp.iterrows()]
    synthetic = _synthetic(s)
    test = bundle.get("test_metrics")
    sel = next((m for m in metrics if m["selected"]), None)
    if test:      # the reported result is the single evaluation on the locked test set
        plain = evaluate.plain_language_metrics(test["uno_c"], {h: test[f"cal_err_{h}m"] for h in (3, 6, 12)})
    else:
        plain = evaluate.plain_language_metrics(sel["uno_c_mean"], {3: sel["cal_err_3m"], 6: sel["cal_err_6m"],
                                                                     12: sel["cal_err_12m"]}) if sel else []
    return {
        "plain_metrics": plain, "feature_set": bundle.get("feature_set", "core"),
        "is_synthetic": synthetic, "training_n": info.get("training_n", 0), "events": info.get("events", 0),
        "censored": info.get("censored", 0), "final_year_n": info.get("final_year_n", 0),
        "employers_n": info.get("employers_n", 0), "job_ads_n": info.get("job_ads_n", 0),
        "selected_model": bundle["name"], "selected_model_label": bundle["label"],
        "selection_reason": bundle.get("selection_reason", ""),
        "hyperparameters": json.loads(json.dumps(bundle.get("params", {}), default=str)),
        "horizon_months": s["predictor"].horizon, "cv_metrics": metrics,
        "test_metrics": json.loads(json.dumps(test, default=float)) if test else None,
        "modelling_n": info.get("modelling_n", info.get("training_n", 0)),
        "validation_n": info.get("validation_n", 0), "test_n": info.get("test_n", 0),
        "confidence_thresholds_months": predict.CONFIDENCE_THRESHOLDS,
        "disclaimer": wording.DISCLAIMER.format(n=info.get("training_n", 0)),
        "synthetic_banner": wording.SYNTHETIC_BANNER if synthetic else None,
        "associational_note": wording.ASSOCIATIONAL_NOTE, "privacy": wording.PRIVACY,
        "limitations": wording.LIMITATIONS,
    }


# ---------------------------------------------------------------------------
# Study results (no profile data): the real job-market study and the research artefacts behind the model
# ---------------------------------------------------------------------------
def _artifacts(request: Request) -> Path:
    return _core(request).get("artifacts") or _path("EP_ARTIFACTS_DIR", "artifacts")


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(pd.read_csv(path).to_json(orient="records"))


@app.get("/job-market")
def job_market(request: Request):
    path = _artifacts(request) / "job_market_study.json"
    if not path.exists():
        raise HTTPException(404, "Job-market study not found - run `python -m core.market_study`.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/questionnaire")
def questionnaire(request: Request):
    path = _artifacts(request) / "questionnaire_study.json"
    if not path.exists():
        raise HTTPException(404, "Questionnaire study not found - run `python -m core.survey_study`.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/research")
def research(request: Request):
    s, art = _core(request), _artifacts(request)
    selected = s["predictor"].bundle["name"]
    cal = pd.read_csv(art / "calibration_table.csv") if (art / "calibration_table.csv").exists() else pd.DataFrame()
    if not cal.empty:
        model = selected if selected in set(cal["model"]) else cal["model"].iloc[0]
        cal = (cal[cal["model"] == model]
               .groupby(["horizon", "tertile"], as_index=False)[["predicted", "observed", "n"]].mean())
    hr = [r for r in _records(art / "competing_risks.csv") if r.get("model") == "main"]
    km = []
    train = ROOT / "data" / "processed" / "graduates_train.csv"
    if train.exists():
        from lifelines import KaplanMeierFitter
        d = pd.read_csv(train)
        kmf = KaplanMeierFitter().fit(d["duration_months"], d["event"])
        ci = kmf.confidence_interval_survival_function_
        km = [{"t": float(t), "employed": float(1 - sv), "lo": float(1 - ci.iloc[i, 1]), "hi": float(1 - ci.iloc[i, 0])}
              for i, (t, sv) in enumerate(kmf.survival_function_.iloc[:, 0].items())]
    sim = art / "simulation_study.json"
    acc = art / "accuracy_experiments.json"
    return {"is_synthetic": _synthetic(s), "selected_model": selected,
            "models": _records(art / "model_comparison.csv"),
            "calibration": json.loads(cal.to_json(orient="records")) if not cal.empty else [],
            "hazard_ratios": hr, "km": km,
            "fairness": _records(art / "fairness_report.csv"), "sensitivity": _records(art / "sensitivity.csv"),
            "temporal": _records(art / "temporal_validation.csv"),
            "feature_sets": _records(art / "feature_set_comparison.csv"),
            "simulation": json.loads(sim.read_text(encoding="utf-8")) if sim.exists() else None,
            "accuracy": json.loads(acc.read_text(encoding="utf-8")) if acc.exists() else None}
