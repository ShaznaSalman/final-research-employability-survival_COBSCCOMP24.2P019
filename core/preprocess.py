"""Preprocessing as scikit-learn transformers.

Everything that learns from data (imputation values, scaling) is fitted on TRAINING FOLDS ONLY.
`build_preprocessor()` returns an unfitted pipeline; evaluation code must call `fit` on the
training fold and only `transform` on the test fold. The same fitted pipeline transforms a single
user profile at prediction time, so training and serving share one code path.

Pipeline:  DerivedFeatures -> BlockMeans -> ColumnTransformer
             numeric:      median imputation -> standardisation
             categorical:  most-frequent imputation -> one-hot (fixed levels, first level dropped)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import schema

NUMERIC_FEATURES = [f for f, t in schema.MODEL_FEATURES.items() if t == "numeric"]
CATEGORICAL_FEATURES = [f for f, t in schema.MODEL_FEATURES.items() if t == "categorical"]


def _split(features: dict | None) -> tuple[list[str], list[str]]:
    features = schema.MODEL_FEATURES if features is None else features
    return ([f for f, t in features.items() if t == "numeric"], [f for f, t in features.items() if t == "categorical"])

ITEM_BLOCKS = {
    "technical_mean": [s.column for s in schema.TECHNICAL_SKILLS],
    "soft_mean": [s.column for s in schema.SOFT_SKILLS],
    **{f"area_{a}": [schema.SKILLS_BY_KEY[k].column for k in v["skills"]] for a, v in schema.SKILL_AREAS.items()},
    "area_soft": [s.column for s in schema.SOFT_SKILLS],
}


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Deterministic recodes (no fitting): degree_group, GPA from degree class, zero internship months."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "degree_group" not in X or X["degree_group"].isna().any():
            mapped = X["degree_type"].map(schema.DEGREE_GROUPS) if "degree_type" in X else np.nan
            X["degree_group"] = X["degree_group"].fillna(mapped) if "degree_group" in X else mapped
        if "degree_class" in X:
            gpa = X["gpa"] if "gpa" in X else pd.Series(np.nan, index=X.index)
            X["gpa"] = pd.to_numeric(gpa, errors="coerce").fillna(X["degree_class"].map(schema.DEGREE_CLASS_TO_GPA))
        if "internship_completed" in X:
            X.loc[X["internship_completed"] == "no", "internship_months"] = 0
        # Candidate predictors (extended feature set): yes/no -> 1/0, preparation start -> years.
        for f in ("has_portfolio", "open_source"):
            if f in X:
                X[f] = X[f].map({"yes": 1.0, "no": 0.0, 1: 1.0, 0: 0.0, 1.0: 1.0, 0.0: 0.0})
        # Transformed feature set: log(1 + count) and the GPA shortlisting cut-off.
        for src, dst in (("internship_months", "log_internship_months"), ("certifications_count", "log_certifications"),
                         ("project_count", "log_projects")):
            if src in X:
                X[dst] = np.log1p(pd.to_numeric(X[src], errors="coerce").clip(lower=0))
        if "gpa" in X:
            g = pd.to_numeric(X["gpa"], errors="coerce")
            X["gpa_3plus"] = (g >= schema.GPA_CUTOFF).astype(float).where(g.notna())
        if "prep_start" in X:
            X["prep_years"] = X["prep_start"].map(schema.PREP_YEARS).astype(float)
        # SimpleImputer only treats np.nan as missing: turn None / pd.NA into np.nan so a missing category
        # (e.g. an unmapped university name) is imputed instead of being encoded as the reference level.
        for f in CATEGORICAL_FEATURES:
            if f in X:
                X[f] = X[f].astype(object).where(X[f].notna(), np.nan)
        return X


class BlockMeans(BaseEstimator, TransformerMixin):
    """Average each Likert block over answered items (>= MIN_ITEMS_ANSWERED_FRACTION answered, else NaN).

    If a block's items are absent but the mean column is already present (e.g. a profile sent with
    area ratings only), the existing mean is kept.
    """

    def __init__(self, min_fraction: float = schema.MIN_ITEMS_ANSWERED_FRACTION):
        self.min_fraction = min_fraction

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for name, cols in ITEM_BLOCKS.items():
            present = [c for c in cols if c in X]
            if not present:
                continue
            items = X[present].apply(pd.to_numeric, errors="coerce")
            answered = items.notna().sum(axis=1) / len(cols)
            X[name] = items.mean(axis=1).where(answered >= self.min_fraction)
        return X


def build_preprocessor(features: dict | None = None) -> Pipeline:
    """Unfitted preprocessing pipeline producing exactly schema.model_parameter_count(features) columns."""
    num_feats, cat_feats = _split(features)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=[list(schema.CATEGORICAL_LEVELS[f]) for f in cat_feats],
                                 drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [("num", numeric, num_feats), ("cat", categorical, cat_feats)],
        remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    # DerivedFeatures and BlockMeans already return DataFrames.
    return Pipeline([("derive", DerivedFeatures()), ("blocks", BlockMeans()), ("columns", columns)])


def feature_names(pre: Pipeline) -> list[str]:
    return list(pre.named_steps["columns"].get_feature_names_out())


def _fitted_columns(pre: Pipeline) -> tuple[list[str], list[str]]:
    cols = {name: list(c) for name, _, c in pre.named_steps["columns"].transformers_ if name in ("num", "cat")}
    return cols.get("num", []), cols.get("cat", [])


def numeric_scale(pre: Pipeline) -> pd.DataFrame:
    """Training-fold mean and SD of each numeric feature (to express effects per original unit)."""
    scaler = pre.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    return pd.DataFrame({"mean": scaler.mean_, "sd": scaler.scale_}, index=_fitted_columns(pre)[0])


def raw_feature_frame(df: pd.DataFrame, features: dict | None = None) -> pd.DataFrame:
    """Model features in original units (derived + block means, NOT imputed or scaled) - for descriptives."""
    X = BlockMeans().transform(DerivedFeatures().transform(df))
    feats = list(schema.MODEL_FEATURES if features is None else features)
    for f in feats:                                     # candidate fields may be absent (e.g. older inputs)
        if f not in X:
            X[f] = np.nan
    return X[feats]


def outcome(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(duration_months, event) as float / int arrays."""
    return df["duration_months"].to_numpy(float), df["event"].to_numpy(int)


def sksurv_y(df: pd.DataFrame) -> np.ndarray:
    """Structured array for scikit-survival: (event: bool, time: float)."""
    t, e = outcome(df)
    y = np.empty(len(t), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = e.astype(bool), t
    return y


class FastColumns:
    """Numpy copy of a FITTED pipeline's column step (impute, scale, one-hot) - for fast bootstrap prediction.

    Apply to the output of `derive_and_block(df)`; gives the same matrix as `pre.transform(df)`.
    """

    def __init__(self, pre: Pipeline):
        ct = pre.named_steps["columns"]
        num, cat = ct.named_transformers_["num"], ct.named_transformers_["cat"]
        self.medians = num.named_steps["impute"].statistics_.astype(float)
        self.means = num.named_steps["scale"].mean_
        self.scales = num.named_steps["scale"].scale_
        self.cat_fill = cat.named_steps["impute"].statistics_
        self.cat_levels = [list(c) for c in cat.named_steps["onehot"].categories_]
        self.columns = feature_names(pre)
        self.numeric_features, self.categorical_features = _fitted_columns(pre)

    @staticmethod
    def arrays(D: pd.DataFrame, features: dict | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        """Extract once, then reuse across many FastColumns: (numeric matrix, categorical arrays)."""
        num_feats, cat_feats = _split(features)
        return D[num_feats].to_numpy(float), [D[f].to_numpy(object) for f in cat_feats]

    def __call__(self, D) -> np.ndarray:
        if isinstance(D, tuple):
            x, cats = D
        else:
            x = D[self.numeric_features].to_numpy(float)
            cats = [D[f].to_numpy(object) for f in self.categorical_features]
        x = np.where(np.isnan(x), self.medians, x)
        parts = [(x - self.means) / self.scales]
        for j, v in enumerate(cats):
            v = np.where(pd.isna(v), self.cat_fill[j], v)
            parts.append(np.column_stack([(v == lev).astype(float) for lev in self.cat_levels[j][1:]]))
        return np.hstack(parts)


def derive_and_block(df: pd.DataFrame) -> pd.DataFrame:
    """The deterministic (unfitted) steps of the pipeline: derived fields + block means."""
    return BlockMeans().transform(DerivedFeatures().transform(df))


def fit_transform_fold(train: pd.DataFrame, test: pd.DataFrame, features: dict | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """Fit preprocessing on the training fold only; transform both folds."""
    pre = build_preprocessor(features).fit(train)
    return pre.transform(train), pre.transform(test), pre
