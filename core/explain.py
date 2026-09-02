"""Explanations: global effects (whole model) and local contributions (one profile).

Global
  Cox-type models  -> hazard ratios per meaningful step (e.g. +3 internship months) with 95% intervals
                      (model-based for Cox PH; bootstrap percentiles for Elastic Net Cox).
  AFT models       -> time ratios ("~25% shorter search") with 95% intervals.
  DeepSurv         -> mean |SHAP| ranking with direction, no intervals.
Local
  Linear models    -> coefficient x (value - training mean), in model units, summed per feature.
  DeepSurv         -> Kernel SHAP on the log-risk output against a small background sample.
Contributions are relative to the AVERAGE training profile. Top 3 helping and top 3 hurting are returned,
restricted to features the user can change when LOCAL_ACTIONABLE_ONLY is True (global effects show all).

All sentences come from core/wording.py and are associational, never causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import preprocess, schema, wording
from core.predict import Predictor, profile_frame

MIN_CONTRIBUTION = 0.02     # |log-scale contribution| below this is not listed as helping / hurting
TOP_K = 3
# Show only things the user can change in "What's helping / holding back" (False = all model features).
LOCAL_ACTIONABLE_ONLY = True
# Model features driven by an actionable input (skill means move with the skill-area ratings).
_ACTIONABLE_DRIVERS = {"internship_months": "internship_months", "certifications_count": "certifications_count",
                       "project_count": "project_count", "gpa": "gpa",
                       "technical_mean": "area_programming", "soft_mean": "area_soft",
                       "log_internship_months": "internship_months", "log_certifications": "certifications_count",
                       "log_projects": "project_count", "gpa_3plus": "gpa"}


def actionable_model_features(status: str) -> set[str]:
    """Model features the user can change, respecting each actionable feature's `applies_to` (e.g. GPA)."""
    out = set()
    for feat, driver in _ACTIONABLE_DRIVERS.items():
        spec = schema.ACTIONABLE_FEATURES[driver]
        if spec.applies_to in ("all", status):
            out.add(feat)
    return out
SHAP_BACKGROUND_CLUSTERS = 10
SHAP_NSAMPLES = 256


def model_kind(model_name: str) -> str:
    if model_name in ("cox_ph", "cox_elasticnet"):
        return "cox"
    if model_name.startswith("aft_"):
        return "aft"
    return "deepsurv"


def column_feature(col: str) -> tuple[str, str | None]:
    """Map a model column back to (feature, level). Numeric columns have level None."""
    for f in preprocess.CATEGORICAL_FEATURES:
        if col.startswith(f + "_"):
            return f, col[len(f) + 1:]
    return col, None


def _strength(score: float) -> str:
    return next(word for cut, word in wording.STRENGTH if abs(score) >= cut)


class Explainer:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        b = predictor.bundle
        self.pre, self.model = b["preprocessor"], b["model"]
        self.kind = model_kind(b.get("model_name", b.get("name", "cox_ph")))
        self.columns = list(preprocess.feature_names(self.pre))
        self.x_mean = pd.Series(b["X_mean"]).reindex(self.columns) if "X_mean" in b else None
        self.background = b.get("background")
        self.scale = preprocess.numeric_scale(self.pre)
        self.features = b.get("features")                   # feature set the bundle was trained with
        self._global = None
        self._shap = None

    # -----------------------------------------------------------------------
    # Global
    # -----------------------------------------------------------------------
    def _unit_coefs(self, coef_table: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
        """Coefficient per wording step in original units (numeric) or per level (dummies)."""
        out = coef_table.reindex(self.columns).copy()
        scaled = [c for c in ("coef", "lower", "upper") if c in out]
        for col in self.columns:
            feat, level = column_feature(col)
            if level is None:
                out.loc[col, scaled] = out.loc[col, scaled] * (wording.FEATURES[feat]["step"] / scale.loc[feat, "sd"])
        return out

    def _bootstrap_intervals(self) -> pd.DataFrame:
        rows = []
        for boot in self.predictor.bootstraps:
            coefs = boot["model"].coefficients[["coef"]]
            rows.append(self._unit_coefs(coefs, preprocess.numeric_scale(boot["preprocessor"]))["coef"])
        B = pd.DataFrame(rows)
        return pd.DataFrame({"lower": B.quantile(0.025), "upper": B.quantile(0.975)})

    def global_effects(self) -> list[dict]:
        if self._global is not None:
            return self._global
        if self.kind == "deepsurv":
            self._global = self._global_shap()
            return self._global
        unit = self._unit_coefs(self.model.coefficients, self.scale)
        if "lower" not in unit or unit["lower"].isna().all():
            unit[["lower", "upper"]] = self._bootstrap_intervals().reindex(self.columns).to_numpy()
        effects = []
        for col in self.columns:
            feat, level = column_feature(col)
            c, lo, hi = unit.loc[col, ["coef", "lower", "upper"]].astype(float)
            ratio, r_lo, r_hi = np.exp(c), np.exp(lo), np.exp(hi)
            sooner = (c > 0) if self.kind == "cox" else (c < 0)
            direction = "sooner" if sooner else "later"
            uncertain = bool(lo <= 0 <= hi)
            fill = {"ratio": ratio, "lo": r_lo, "hi": r_hi, "pct": abs(1 - ratio) * 100}
            templates = wording.GLOBAL_HR if self.kind == "cox" else wording.GLOBAL_TR
            if level is None:
                text = templates[direction].format(step_text=wording.FEATURES[feat]["step_text"], **fill)
            else:
                ref = schema.CATEGORICAL_LEVELS[feat][0]
                text = templates[f"level_{direction}"].format(level=wording.LEVELS[feat][level],
                                                              reference=wording.LEVELS[feat][ref], **fill)
            if uncertain:
                text += wording.UNCERTAIN_SUFFIX
            effects.append({"feature": feat, "level": level, "column": col,
                            "kind": "hazard_ratio" if self.kind == "cox" else "time_ratio",
                            "step": None if level else wording.FEATURES[feat]["step"],
                            "ratio": round(float(ratio), 3), "lower": round(float(r_lo), 3),
                            "upper": round(float(r_hi), 3), "direction": direction, "uncertain": uncertain,
                            "text": text})
        self._global = effects
        return effects

    # -----------------------------------------------------------------------
    # SHAP (DeepSurv)
    # -----------------------------------------------------------------------
    def _shap_explainer(self):
        if self._shap is None:
            import shap
            bg = self.background.to_numpy(float)
            summary = shap.kmeans(bg, min(SHAP_BACKGROUND_CLUSTERS, len(bg)))
            f = lambda Z: self.model.predict_risk(pd.DataFrame(Z, columns=self.columns))  # noqa: E731
            self._shap = shap.KernelExplainer(f, summary)
        return self._shap

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        np.random.seed(0)
        vals = self._shap_explainer().shap_values(X.to_numpy(float), nsamples=SHAP_NSAMPLES, silent=True)
        return np.atleast_2d(np.asarray(vals, float))

    def _global_shap(self) -> list[dict]:
        bg = self.background.iloc[:30]
        sv = self._shap_values(bg)
        effects = []
        for j, col in enumerate(self.columns):
            feat, level = column_feature(col)
            corr = np.corrcoef(bg[col], sv[:, j])[0, 1] if bg[col].std() > 0 else 0.0
            direction = "sooner" if corr >= 0 else "later"
            label = wording.FEATURES[feat].get("step_text", wording.LEVELS.get(feat, {}).get(level, col))
            effects.append({"feature": feat, "level": level, "column": col, "kind": "shap",
                            "mean_abs_shap": round(float(np.abs(sv[:, j]).mean()), 4), "direction": direction,
                            "uncertain": True,
                            "text": wording.GLOBAL_NO_INTERVAL.format(step_text=label, direction=direction)})
        return sorted(effects, key=lambda e: -e["mean_abs_shap"])

    # -----------------------------------------------------------------------
    # Local
    # -----------------------------------------------------------------------
    def contributions(self, df: pd.DataFrame) -> pd.Series:
        """Per-feature 'helping' score for one profile (positive = associated with finding work sooner)."""
        X = self.pre.transform(df)
        if self.kind == "deepsurv":
            col_scores = pd.Series(self._shap_values(X)[0], index=self.columns)
        else:
            coef = self.model.coefficients["coef"].reindex(self.columns)
            col_scores = coef * (X.iloc[0] - self.x_mean)
            if self.kind == "aft":
                col_scores = -col_scores          # positive log-time = longer search
        feats = [column_feature(c)[0] for c in self.columns]
        return col_scores.groupby(feats, sort=False).sum()

    def local(self, profile: dict) -> dict:
        status = profile.get("respondent_status", "graduated")
        df = profile_frame({k: v for k, v in profile.items() if k != "months_since_graduation"})
        scores = self.contributions(df)
        raw = preprocess.raw_feature_frame(df, self.features).iloc[0]
        derived = preprocess.derive_and_block(df).iloc[0]
        allowed = actionable_model_features(status) if LOCAL_ACTIONABLE_ONLY else set(scores.index)
        items = []
        for feat, score in scores.items():
            if abs(score) < MIN_CONTRIBUTION or feat not in allowed:
                continue
            side = "helping" if score > 0 else "hurting"
            spec = wording.FEATURES[feat]
            if feat in preprocess.CATEGORICAL_FEATURES:
                level = raw[feat]
                text = wording.LOCAL[f"{side}_level"].format(level=wording.LEVELS[feat].get(level, level),
                                                             strength=_strength(score))
                value = wording.LEVELS[feat].get(level, str(level))
            else:
                v = raw[feat]
                shown = derived.get(spec.get("display", feat), v)
                if pd.isna(shown):
                    value = "not given"
                elif spec["unit"] == "binary":
                    value = "yes" if shown >= 0.5 else "no"
                else:
                    value = spec["unit"].format(v=shown)
                above = pd.notna(v) and v >= self.scale.loc[feat, "mean"]
                text = wording.LOCAL[f"{side}_{'higher' if above else 'lower'}"].format(
                    Name=spec["name"][0].upper() + spec["name"][1:], value=value, higher=spec["higher"],
                    lower=spec["lower"],
                    strength=_strength(score))
            items.append({"feature": feat, "name": spec["name"], "value": value, "score": round(float(score), 4),
                          "strength": _strength(score), "side": side, "text": text})
        helping = sorted([i for i in items if i["side"] == "helping"], key=lambda i: -i["score"])[:TOP_K]
        hurting = sorted([i for i in items if i["side"] == "hurting"], key=lambda i: i["score"])[:TOP_K]
        return {"helping": helping, "hurting": hurting, "note": wording.LOCAL_NOTE,
                "method": "shap" if self.kind == "deepsurv" else "coefficient x (value - average)"}

    def explain(self, profile: dict) -> dict:
        return {"global": self.global_effects(), **self.local(profile)}
