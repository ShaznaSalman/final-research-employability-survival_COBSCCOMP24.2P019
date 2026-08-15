"""Survival model wrappers with one shared interface.

Every model:
  fit(X, time, event)              X = preprocessed feature DataFrame
  predict_risk(X) -> (n,)          higher = higher hazard of finding work = SHORTER expected time
  predict_survival(X, times)       (n, len(times)) P(still searching at t)
  get_params() / set_params(**p)   hyperparameters tuned in the inner CV loop

The main comparison uses the four model families below (no forest / boosting / ensemble methods).
core/models/neural.py adds DeepHit, Cox-Time and N-MTLR for the exploratory comparison in
core/accuracy_experiments.py (report only; they do not enter model selection).
"""
from __future__ import annotations

import numpy as np

CURVE_TIMES = np.round(np.arange(0, 24.001, 0.25), 2)   # months; prediction grid for S(t)


def step_survival(event_times: np.ndarray, surv: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Evaluate right-continuous step survival curves at `times`.

    event_times: (m,) sorted; surv: (n, m) S at those times. Before the first time S = 1.
    """
    idx = np.searchsorted(event_times, times, side="right") - 1
    padded = np.hstack([np.ones((surv.shape[0], 1)), surv])
    return padded[:, idx + 1]


class SurvivalModel:
    name: str = "base"
    label: str = "Base"
    complexity: int = 1          # used by the selection rule: lower = simpler
    param_grid: list[dict] = [{}]

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in self.param_grid[0]} if self.param_grid[0] else {}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, time, event):
        raise NotImplementedError

    def predict_risk(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_survival(self, X, times) -> np.ndarray:
        raise NotImplementedError


def make_model(name: str, **params) -> SurvivalModel:
    from core.models.aft import AFTModel
    from core.models.cox_elasticnet import CoxElasticNetModel
    from core.models.cox_ph import CoxPHModel
    from core.models.deepsurv import DeepSurvModel

    registry = {
        "cox_ph": lambda: CoxPHModel(),
        "cox_elasticnet": lambda: CoxElasticNetModel(),
        "aft_weibull": lambda: AFTModel("weibull"),
        "aft_lognormal": lambda: AFTModel("lognormal"),
        "aft_loglogistic": lambda: AFTModel("loglogistic"),
        "deepsurv": lambda: DeepSurvModel(),
    }
    return registry[name]().set_params(**params)


MODEL_NAMES = ("cox_ph", "cox_elasticnet", "aft_weibull", "aft_lognormal", "aft_loglogistic", "deepsurv")
