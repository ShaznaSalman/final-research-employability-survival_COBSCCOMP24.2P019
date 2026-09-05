"""Shared helpers for the secondary (robustness) analyses: Cox hazard-ratio tables on the model's design matrix,
readable column labels and Rubin's rules. Used by sensitivity.py, clustering.py and competing.py.

Hazard ratios here are UNPENALISED Cox estimates (for inference), per one SD of the full training data for
numeric features (the SD in original units is reported) and versus the reference level for categories.
They describe associations in the data, not causes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from core import preprocess, wording
from core.explain import column_feature

Z95 = 1.959964


def column_label(col: str) -> str:
    feat, level = column_feature(col)
    name = wording.FEATURES.get(feat, {}).get("name", feat)
    return f"{name}: {wording.LEVELS[feat][level]}" if level else name


def fit_cox(X: pd.DataFrame, time_, event, penalizer: float = 0.0, **fit_kw) -> CoxPHFitter:
    data = X.assign(_T=np.asarray(time_, float), _E=np.asarray(event, int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return CoxPHFitter(penalizer=penalizer).fit(data, "_T", "_E", **fit_kw)
        except ConvergenceError:                         # separation in a small subset: tiny ridge for stability
            return CoxPHFitter(penalizer=max(penalizer, 0.01)).fit(data, "_T", "_E", **fit_kw)


def hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary
    return pd.DataFrame({"column": s.index, "log_hr": s["coef"].to_numpy(), "se": s["se(coef)"].to_numpy()})


def finish(t: pd.DataFrame) -> pd.DataFrame:
    """Add HR, 95% CI and labels to a table with log_hr and se."""
    t = t.copy()
    t["hr"] = np.exp(t["log_hr"])
    t["lo"], t["hi"] = np.exp(t["log_hr"] - Z95 * t["se"]), np.exp(t["log_hr"] + Z95 * t["se"])
    t["label"] = t["column"].map(column_label)
    return t


def rubin(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Pool log-HRs over m imputations (Rubin's rules)."""
    est = np.column_stack([t["log_hr"].to_numpy() for t in tables])
    var = np.column_stack([t["se"].to_numpy() ** 2 for t in tables])
    m = est.shape[1]
    within, between = var.mean(axis=1), est.var(axis=1, ddof=1)
    return pd.DataFrame({"column": tables[0]["column"].to_numpy(), "log_hr": est.mean(axis=1),
                         "se": np.sqrt(within + (1 + 1 / m) * between)})


def design(df: pd.DataFrame, pre=None):
    """Model design matrix. `pre` (a fitted preprocessor) fixes imputation/scaling across subsets."""
    pre = pre or preprocess.build_preprocessor().fit(df)
    return pre, pre.transform(df)


def fmt_hr(row) -> str:
    return f"{row['hr']:.2f} ({row['lo']:.2f}-{row['hi']:.2f})"
