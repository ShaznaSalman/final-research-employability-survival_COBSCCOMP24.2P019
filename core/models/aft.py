"""Accelerated failure time models (lifelines): Weibull, log-normal, log-logistic.

Fitted by maximum likelihood (no penalty) so AIC is valid for choosing among the three.
Coefficients are on log-time: exp(coef) is a TIME RATIO (< 1 = shorter search).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

from core.models import SurvivalModel

FITTERS = {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter, "loglogistic": LogLogisticAFTFitter}
LABELS = {"weibull": "AFT Weibull", "lognormal": "AFT Log-Normal", "loglogistic": "AFT Log-Logistic"}
LOCATION_PARAM = {"weibull": "lambda_", "lognormal": "mu_", "loglogistic": "alpha_"}


def aft_survival(distribution: str, location: np.ndarray, log_shape: np.ndarray, times) -> np.ndarray:
    """Closed-form S(t) for many models at once.

    location: (B, n) linear predictor of the location parameter (log scale); log_shape: (B,).
    Returns (B, n, T). Matches lifelines' predict_survival_function for the same parameters.
    """
    from scipy.stats import norm
    t = np.maximum(np.asarray(times, float), 1e-9)[None, None, :]
    loc = location[:, :, None]
    shape = np.exp(log_shape)[:, None, None]
    if distribution == "weibull":
        return np.exp(-(t / np.exp(loc)) ** shape)
    if distribution == "lognormal":
        return norm.sf((np.log(t) - loc) / shape)
    if distribution == "loglogistic":
        return 1.0 / (1.0 + (t / np.exp(loc)) ** shape)
    raise ValueError(distribution)


class AFTModel(SurvivalModel):
    complexity = 1
    param_grid = [{}]

    def __init__(self, distribution: str = "weibull"):
        self.distribution = distribution
        self.name = f"aft_{distribution}"
        self.label = LABELS[distribution]

    def get_params(self) -> dict:
        return {}

    def fit(self, X: pd.DataFrame, time, event):
        df = X.reset_index(drop=True).assign(_T=np.asarray(time, float), _E=np.asarray(event, int))
        self.fitter_ = FITTERS[self.distribution]().fit(df, "_T", "_E")
        return self

    @property
    def aic(self) -> float:
        return float(self.fitter_.AIC_)

    def predict_risk(self, X) -> np.ndarray:
        # Shorter predicted median time = higher risk of finding work.
        return -np.log(np.asarray(self.fitter_.predict_median(X.reset_index(drop=True)), float).ravel())

    def predict_survival(self, X, times) -> np.ndarray:
        t = np.maximum(np.asarray(times, float), 1e-9)
        return self.fitter_.predict_survival_function(X.reset_index(drop=True), times=t).to_numpy().T

    def aft_form(self, columns: list[str]) -> tuple[np.ndarray, float, float]:
        """(beta, intercept, log_shape) of the location parameter, for vectorised survival (see aft_survival)."""
        p = self.fitter_.params_
        loc = LOCATION_PARAM[self.distribution]
        beta = np.array([p[(loc, c)] for c in columns], float)
        shape = {"weibull": "rho_", "lognormal": "sigma_", "loglogistic": "beta_"}[self.distribution]
        return beta, float(p[(loc, "Intercept")]), float(p[(shape, "Intercept")])

    @property
    def coefficients(self) -> pd.DataFrame:
        """Location-parameter coefficients (log time ratios) with 95% CIs."""
        s = self.fitter_.summary.loc[LOCATION_PARAM[self.distribution]].drop(index="Intercept", errors="ignore")
        return pd.DataFrame({"coef": s["coef"], "se": s["se(coef)"],
                             "lower": s["coef lower 95%"], "upper": s["coef upper 95%"], "p": s["p"]})
