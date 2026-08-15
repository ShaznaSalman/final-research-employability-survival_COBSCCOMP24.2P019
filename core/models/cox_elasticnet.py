"""Elastic-net penalised Cox (scikit-survival CoxnetSurvivalAnalysis); alpha and l1_ratio tuned in inner CV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

from core.models import SurvivalModel, step_survival


def _y(time, event):
    y = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = np.asarray(event, bool), np.asarray(time, float)
    return y


class CoxElasticNetModel(SurvivalModel):
    name = "cox_elasticnet"
    label = "Elastic Net Cox"
    complexity = 2
    param_grid = [{"alpha": a, "l1_ratio": r} for a in (0.005, 0.02, 0.05, 0.1, 0.2) for r in (0.1, 0.5, 0.9)]

    def __init__(self, alpha: float = 0.05, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, time, event):
        self.columns_ = list(X.columns)
        self.fitter_ = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alphas=[self.alpha], fit_baseline_model=True,
                                              max_iter=100_000).fit(np.asarray(X, float), _y(time, event))
        return self

    def predict_risk(self, X) -> np.ndarray:
        return self.fitter_.predict(np.asarray(X, float), alpha=self.alpha)

    def predict_survival(self, X, times) -> np.ndarray:
        fns = self.fitter_.predict_survival_function(np.asarray(X, float), alpha=self.alpha)
        surv = np.vstack([f.y for f in fns])
        return step_survival(fns[0].x, surv, np.asarray(times, float))

    @property
    def coefficients(self) -> pd.DataFrame:
        """log-HR per (standardised) feature. No CIs: penalised estimates have no standard errors."""
        return pd.DataFrame({"coef": self.fitter_.coef_[:, 0]}, index=self.columns_)
