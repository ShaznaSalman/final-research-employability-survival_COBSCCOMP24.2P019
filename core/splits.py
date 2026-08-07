"""Fixed three-way split of the modelling graduates: every graduate has exactly ONE role, assigned once.

  train       (TRAIN_SHARE)  - the only data any model is ever fitted on (including the app's final model,
                               the bootstrap refits, hazard-ratio inference and the peer reference data)
  validation  (VALID_SHARE)  - the only data used to tune hyperparameters, compare models and feature sets,
                               and run the secondary model comparisons. Never used to fit a model.
  test        (rest)         - locked: used ONCE, for the final evaluation of the selected model (discrimination,
                               calibration, fairness, LLM benchmark). Never used to fit, tune or choose anything.

No graduate is ever reused across roles, so no cross-validation is used. The split is stratified by event status
and institution type, made with a fixed seed, and written to data/processed/splits.csv by core/clean.py so that
every module reads the same assignment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "processed" / "splits.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
ROLES = ("train", "validation", "test")
TRAIN_SHARE, VALID_SHARE = 0.70, 0.15          # test = the remaining 0.15
SEED = 20261006


def assign(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Role per graduate, stratified by event x institution type (largest-remainder rounding per stratum)."""
    strata = df["event"].astype(str) + "|" + df.get("university_type", pd.Series("all", index=df.index)).fillna("missing")
    rng = np.random.default_rng(seed)
    role = pd.Series(index=df.index, dtype=object)
    for _, idx in df.groupby(strata, sort=True).groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(n * TRAIN_SHARE)), int(round(n * VALID_SHARE))
        n_va = min(n_va, n - n_tr)
        role.loc[idx[:n_tr]] = "train"
        role.loc[idx[n_tr:n_tr + n_va]] = "validation"
        role.loc[idx[n_tr + n_va:]] = "test"
    return role


def write(df: pd.DataFrame, path: Path = SPLITS_PATH) -> pd.DataFrame:
    out = pd.DataFrame({"respondent_id": df["respondent_id"].astype(str), "split": assign(df).to_numpy()})
    out.to_csv(path, index=False)
    return out


def attach(df: pd.DataFrame, path: Path = SPLITS_PATH) -> pd.DataFrame:
    """df with a 'split' column. Graduates missing from the split file are an error (the data changed)."""
    roles = pd.read_csv(path, dtype={"respondent_id": str})
    out = df.assign(respondent_id=df["respondent_id"].astype(str)).merge(roles, on="respondent_id", how="left")
    if out["split"].isna().any():
        raise ValueError("Some graduates have no split - re-run `python -m core.clean` (it rewrites splits.csv).")
    return out


def load(role: str | tuple[str, ...], train_path: Path = TRAIN_PATH, splits_path: Path = SPLITS_PATH) -> pd.DataFrame:
    """Graduates with the given role(s), in their original order, index reset."""
    roles = (role,) if isinstance(role, str) else tuple(role)
    df = attach(pd.read_csv(train_path), splits_path)
    return df[df["split"].isin(roles)].drop(columns="split").reset_index(drop=True)


def summary(df_with_split: pd.DataFrame) -> pd.DataFrame:
    return (df_with_split.groupby("split")
            .agg(graduates=("event", "size"), events=("event", "sum"))
            .reindex(list(ROLES)).reset_index())
