"""Cox proportional hazards (lifelines), with a ridge penalizer tuned in the inner CV loop."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from core.models import SurvivalModel, step_survival


class CoxPHModel(SurvivalModel):
    name = "cox_ph"
    label = "Cox PH"
    complexity = 1
    param_grid = [{"penalizer": p} for p in (0.0, 0.01, 0.1, 0.5)]

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer

    def fit(self, X: pd.DataFrame, time, event):
        df = X.assign(_T=np.asarray(time, float), _E=np.asarray(event, int))
        self.fitter_ = CoxPHFitter(penalizer=self.penalizer).fit(df, "_T", "_E")
        return self

    def predict_risk(self, X) -> np.ndarray:
        return np.asarray(self.fitter_.predict_log_partial_hazard(X), float).ravel()

    def predict_survival(self, X, times) -> np.ndarray:
        # Evaluate as a step function: lifelines' own `times=` interpolates between event times.
        surv = self.fitter_.predict_survival_function(X)
        return step_survival(surv.index.to_numpy(float), surv.to_numpy().T, np.asarray(times, float))

    def cox_form(self, times) -> tuple[np.ndarray, float, np.ndarray]:
        """(beta, offset, H0 at times) such that S(t | x) = exp(-H0(t) * exp(x @ beta - offset))."""
        beta = self.fitter_.params_.to_numpy(float)
        offset = float(np.dot(self.fitter_._norm_mean.to_numpy(float), beta))
        H = self.fitter_.baseline_cumulative_hazard_
        h_times, h_vals = H.index.to_numpy(float), H.iloc[:, 0].to_numpy(float)
        idx = np.searchsorted(h_times, np.asarray(times, float), side="right") - 1
        H0 = np.where(idx >= 0, h_vals[np.clip(idx, 0, None)], 0.0)
        return beta, offset, H0

    @property
    def coefficients(self) -> pd.DataFrame:
        """log-HR and 95% CI per (standardised) feature."""
        s = self.fitter_.summary
        return pd.DataFrame({"coef": s["coef"], "se": s["se(coef)"],
                             "lower": s["coef lower 95%"], "upper": s["coef upper 95%"], "p": s["p"]})

    def ph_test(self, X: pd.DataFrame, time, event) -> pd.DataFrame:
        """Schoenfeld-residual test of proportional hazards (rank time transform)."""
        df = X.assign(_T=np.asarray(time, float), _E=np.asarray(event, int))
        res = proportional_hazard_test(self.fitter_, df, time_transform="rank")
        return res.summary[["test_statistic", "p"]]

    def scaled_schoenfeld(self, X: pd.DataFrame, time, event) -> pd.DataFrame:
        df = X.assign(_T=np.asarray(time, float), _E=np.asarray(event, int))
        return self.fitter_.compute_residuals(df, "scaled_schoenfeld")
