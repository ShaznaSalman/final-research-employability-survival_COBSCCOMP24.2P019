"""Prediction: survival curve -> typical range, median, bootstrap uncertainty, confidence label.

For a profile, S(t) = P(still searching t months after degree completion), t = 0..24.
  low    = first month where S(t) <= 0.75   (25th percentile of time to first IT job)
  median = first month where S(t) <= 0.50
  high   = first month where S(t) <= 0.25   (75th percentile)
If S(t) never reaches a level within follow-up, the value is reported as "more than X months"
(X = the end of follow-up in the training data) - never extrapolated.

Uncertainty: 80% interval (10th-90th percentile) of the median across the bootstrap refits.
Confidence label from the width of that interval (thresholds below).
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core import preprocess, schema, wording
from core.models import CURVE_TIMES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

LEVELS = {"low": 0.75, "median": 0.50, "high": 0.25}
UNCERTAINTY_PERCENTILES = (10, 90)        # 80% bootstrap interval
CONFIDENCE_THRESHOLDS = {"High": 2.0, "Medium": 4.0}   # interval width in months: <= 2 High, <= 4 Medium, else Low
CONFIDENCE_EXPLANATION = wording.CONFIDENCE_EXPLANATION
TIME_REFERENCE = wording.TIME_REFERENCE
MILESTONE_MONTHS = (3, 6, 12, 24)   # "chance of having found work by ..." points
MIN_AT_RISK = 20           # fewer training graduates still followed at month m -> Low confidence + caution

PROFILE_FIELDS = {
    "respondent_status", "gpa", "degree_class", "degree_type", "university_type", "certifications_count",
    "internship_completed", "internship_months", "project_count", "months_since_graduation",
    # extended feature set (preparation before completing the degree)
    "has_portfolio", "prep_start", "self_learning_freq", "hackathon_count", "open_source", "sdlc_familiarity",
    *schema.TECH_COLUMNS, *schema.SOFT_COLUMNS,
    *[f"area_{a}" for a in schema.SKILL_AREAS], "area_soft",
}
AREA_ITEMS = {**{f"area_{a}": [schema.SKILLS_BY_KEY[k].column for k in v["skills"]]
                 for a, v in schema.SKILL_AREAS.items()},
              "area_soft": list(schema.SOFT_COLUMNS)}


# ---------------------------------------------------------------------------
# Profile handling
# ---------------------------------------------------------------------------
def profile_frame(profile: dict) -> pd.DataFrame:
    """One-row DataFrame in the raw schema. Area ratings (Quick mode) fill any item not given individually."""
    unknown = set(profile) - PROFILE_FIELDS
    if unknown:
        raise ValueError(f"Unknown profile fields: {sorted(unknown)}")
    row = {k: v for k, v in profile.items() if not k.startswith("area_")}
    for area, items in AREA_ITEMS.items():
        if profile.get(area) is not None:
            for c in items:
                if row.get(c) is None:
                    row[c] = float(profile[area])
    for c in (*schema.TECH_COLUMNS, *schema.SOFT_COLUMNS):
        row.setdefault(c, np.nan)
    # Optional preparation answers (extended feature set): missing -> imputed like any other missing value.
    for c in (f.name for f in schema.RAW_FIELDS if f.role == "candidate"):
        row.setdefault(c, np.nan)
    return pd.DataFrame([row])


def round_month(t: float) -> int:
    """Whole months; times at or before the 'job before graduation' code (0.5) are reported as 0."""
    if t <= schema.JOB_BEFORE_GRADUATION_MONTHS:
        return 0
    return max(1, int(math.floor(t + 0.5)))


def crossing_times(S: np.ndarray, level: float, horizon: float, times=CURVE_TIMES) -> np.ndarray:
    """First time S <= level within the horizon; +inf if not reached (= 'more than horizon')."""
    ok = (S <= level) & (times[None, :] <= horizon)
    return np.where(ok.any(axis=1), times[np.argmax(ok, axis=1)], np.inf)


def _value(t: float, horizon: int) -> dict:
    if not np.isfinite(t):
        return {"months": None, "more_than": True, "display": f"more than {horizon}"}
    m = round_month(t)
    return {"months": m, "more_than": False, "display": str(m)}


def confidence_label(width: float, median_reached: bool) -> str:
    if not median_reached or not np.isfinite(width):
        return "Low"
    if width <= CONFIDENCE_THRESHOLDS["High"]:
        return "High"
    if width <= CONFIDENCE_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def condition_on_searching(S: np.ndarray, m: float) -> np.ndarray:
    """S(t | still searching at m) = S(t) / S(m) for t >= m, and 1 up to m. Works on (..., T) arrays."""
    idx = int(np.searchsorted(CURVE_TIMES, m, side="right") - 1)
    denom = np.maximum(S[..., idx:idx + 1], 1e-12)
    out = np.minimum(S / denom, 1.0)
    out[..., :idx + 1] = 1.0
    return out


def _remaining(v: dict, m: int, horizon: int) -> dict:
    """A months-after-graduating value re-expressed as months from now (at least 1)."""
    if v["more_than"]:
        return {"months": None, "more_than": True, "display": f"more than {horizon - m}"}
    r = max(1, v["months"] - m)
    return {"months": r, "more_than": False, "display": str(r)}


def headline(res: dict, horizon: int, status: str, m: int | None = None) -> str:
    low, med, high = res["low"], res["median"], res["high"]
    if m:
        rem = res["remaining"]
        if med["more_than"]:
            return wording.HEADLINE_SEARCHING_NO_MEDIAN.format(m=m, low=low["display"], horizon=horizon)
        if high["more_than"]:
            return wording.HEADLINE_SEARCHING_OPEN.format(m=m, low=low["display"], median=med["display"],
                                                          rmedian=rem["median"]["display"], horizon=horizon)
        return wording.HEADLINE_SEARCHING.format(m=m, low=low["display"], high=high["display"],
                                                 median=med["display"], rlow=rem["low"]["display"],
                                                 rhigh=rem["high"]["display"])
    low, med, high = res["low"], res["median"], res["high"]
    status = status if status in wording.HEADLINE else "graduated"
    if med["more_than"]:
        return wording.HEADLINE_NO_MEDIAN[status].format(low=low["display"], horizon=horizon)
    if high["more_than"]:
        return wording.HEADLINE_OPEN[status].format(low=low["display"], median=med["display"], horizon=horizon)
    return wording.HEADLINE[status].format(low=low["display"], high=high["display"], median=med["display"])


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    """Holds the final model and bootstrap refits; loaded once (e.g. at API start-up)."""

    def __init__(self, bundle: dict, bootstraps: list[dict]):
        self.bundle = bundle
        self.bootstraps = bootstraps
        self.horizon = int(min(schema.PREDICTION_HORIZON_MONTHS,
                               bundle.get("max_follow_up_months", schema.PREDICTION_HORIZON_MONTHS)))
        self._fast = [preprocess.FastColumns(b["preprocessor"]) for b in bootstraps]
        # Cox-type bootstrap models collapse to one matrix operation: S = exp(-H0(t) * exp(x @ beta - offset)).
        self._cox = None
        if bootstraps and all(hasattr(b["model"], "cox_form") for b in bootstraps):
            forms = [b["model"].cox_form(CURVE_TIMES) for b in bootstraps]
            self._cox = (np.stack([f[0] for f in forms]), np.array([f[1] for f in forms]),
                         np.stack([f[2] for f in forms]))
        # AFT bootstrap models (one distribution) also have a closed form: see core.models.aft.aft_survival.
        self._aft = None
        dists = {getattr(b["model"], "distribution", None) for b in bootstraps}
        if bootstraps and self._cox is None and all(hasattr(b["model"], "aft_form") for b in bootstraps) \
                and len(dists) == 1:
            forms = [b["model"].aft_form(f.columns) for b, f in zip(bootstraps, self._fast)]
            self._aft = (dists.pop(), np.stack([f[0] for f in forms]), np.array([f[1] for f in forms]),
                         np.array([f[2] for f in forms]))

    @classmethod
    def load(cls, artifacts: Path = ARTIFACTS) -> "Predictor":
        return cls(joblib.load(artifacts / "final_model.pkl"), joblib.load(artifacts / "bootstrap_models.pkl"))

    # -- curves ---------------------------------------------------------------
    def survival(self, df: pd.DataFrame) -> np.ndarray:
        """Final-model S(t) on CURVE_TIMES, shape (n, T)."""
        X = self.bundle["preprocessor"].transform(df)
        return self.bundle["model"].predict_survival(X, CURVE_TIMES)

    def bootstrap_survival(self, df: pd.DataFrame) -> np.ndarray:
        """Bootstrap S(t), shape (B, n, T)."""
        arrays = preprocess.FastColumns.arrays(preprocess.derive_and_block(df), self.bundle.get("features"))
        Xs = np.stack([f(arrays) for f in self._fast])                    # (B, n, p)
        if self._cox is not None:
            beta, offset, H0 = self._cox
            lp = np.einsum("bnp,bp->bn", Xs, beta) - offset[:, None]       # (B, n)
            return np.exp(-H0[:, None, :] * np.exp(lp)[:, :, None])
        if self._aft is not None:
            from core.models.aft import aft_survival
            dist, beta, intercept, log_shape = self._aft
            location = np.einsum("bnp,bp->bn", Xs, beta) + intercept[:, None]
            return aft_survival(dist, location, log_shape, CURVE_TIMES)
        cols = self._fast[0].columns
        return np.stack([b["model"].predict_survival(pd.DataFrame(X, columns=cols), CURVE_TIMES)
                         for b, X in zip(self.bootstraps, Xs)])

    # -- summaries ------------------------------------------------------------
    def summarise(self, S: np.ndarray, S_boot: np.ndarray) -> list[dict]:
        out = []
        crossings = {k: crossing_times(S, lv, self.horizon) for k, lv in LEVELS.items()}
        boot_medians = np.stack([crossing_times(Sb, 0.5, self.horizon) for Sb in S_boot])   # (B, n)
        for i in range(S.shape[0]):
            # "Never reached" (inf) is beyond the horizon: use a finite stand-in so interpolation cannot
            # produce NaN (inf - inf), then map anything beyond the horizon back to inf.
            finite = np.where(np.isfinite(boot_medians[:, i]), boot_medians[:, i], 10.0 * self.horizon)
            lo_q, hi_q = (q if q <= self.horizon else np.inf
                          for q in np.percentile(finite, UNCERTAINTY_PERCENTILES))
            width = hi_q - lo_q
            reached = bool(np.isfinite(crossings["median"][i]))
            label = confidence_label(width, reached)
            w = CONFIDENCE_THRESHOLDS["High"] if label == "High" else CONFIDENCE_THRESHOLDS["Medium"]
            out.append({
                **{k: _value(crossings[k][i], self.horizon) for k in LEVELS},
                "uncertainty": {
                    "median_interval_80": [_value(lo_q, self.horizon), _value(hi_q, self.horizon)],
                    "width_months": round(float(width), 2) if np.isfinite(width) else None,
                    "n_bootstrap": int(S_boot.shape[0]),
                    "share_beyond_horizon": round(float(np.mean(~np.isfinite(boot_medians[:, i]))), 3),
                },
                "confidence": label,
                "confidence_reason": CONFIDENCE_EXPLANATION[label].format(w=f"{w:g}"),
            })
        return out

    def predict_frame(self, df: pd.DataFrame, months_searched: float | None = None) -> list[dict]:
        S, S_boot = self.survival(df), self.bootstrap_survival(df)
        if months_searched:
            S, S_boot = condition_on_searching(S, months_searched), condition_on_searching(S_boot, months_searched)
        out = []
        for i, r in enumerate(self.summarise(S, S_boot)):
            r["curve"] = S[i]
            r["milestones"] = self.milestones(S[i], S_boot[:, i, :], months_searched)
            out.append(r)
        return out

    def milestones(self, curve: np.ndarray, boot_curves: np.ndarray, months_searched: float | None) -> list[dict]:
        """Share expected to have found work by fixed months, with an 80% bootstrap interval."""
        out = []
        for t in MILESTONE_MONTHS:
            if t > self.horizon or (months_searched and t <= months_searched):
                continue
            j = int(np.searchsorted(CURVE_TIMES, t))
            lo, hi = np.percentile(1 - boot_curves[:, j], UNCERTAINTY_PERCENTILES)
            out.append({"months": t, "employed_pct": int(round(100 * (1 - curve[j]))),
                        "low_pct": int(round(100 * lo)), "high_pct": int(round(100 * hi))})
        return out

    def at_risk(self, m: int) -> int | None:
        """Training graduates still followed (no IT job yet, not yet surveyed) at month m."""
        table = self.bundle.get("at_risk_by_month")
        return None if table is None else int(table.get(m, 0))

    def predict(self, profile: dict) -> dict:
        """Full prediction for one profile (raw fields or Quick-mode area ratings).

        `months_since_graduation` (graduates still looking) makes the prediction conditional on not having
        found work by then. Times are still reported as months after graduating, plus months from now.
        """
        status = profile.get("respondent_status", "graduated")
        m = profile.get("months_since_graduation")
        m = int(round(m)) if m else None
        cautions = []
        if m and status != "graduated":
            raise ValueError(wording.SEARCH_ONLY_GRADUATES)
        if m and m >= self.horizon:
            raise ValueError(wording.SEARCH_BEYOND_FOLLOW_UP.format(horizon=self.horizon, m=m))
        df = profile_frame({k: v for k, v in profile.items() if k != "months_since_graduation"})
        res = self.predict_frame(df, months_searched=m)[0]
        if m:
            res["remaining"] = {k: _remaining(res[k], m, self.horizon) for k in LEVELS}
            n_at_risk = self.at_risk(m)
            if n_at_risk is not None and n_at_risk < MIN_AT_RISK:
                cautions.append(wording.CAUTION_FEW_STILL_SEARCHING.format(n=n_at_risk, m=m))
                res["confidence"] = "Low"
                res["confidence_reason"] = CONFIDENCE_EXPLANATION["Low"].format(
                    w=f"{CONFIDENCE_THRESHOLDS['Medium']:g}")
        monthly = np.arange(0, self.horizon + 1)
        curve = res.pop("curve")
        template = wording.MILESTONE_SEARCHING if m else wording.MILESTONE
        for ms in res["milestones"]:
            ms["text"] = template.format(t=ms["months"], pct=ms["employed_pct"], lo=ms["low_pct"],
                                         hi=ms["high_pct"], m=m)
        res.update({
            "status": status,
            "months_since_graduation": m,
            "remaining": res.get("remaining"),
            "cautions": cautions,
            "time_reference": TIME_REFERENCE.get(status, TIME_REFERENCE["graduated"]),
            "headline": headline(res, self.horizon, status, m),
            "horizon_months": self.horizon,
            "curve": {"months": monthly.tolist(),
                      "still_searching": [round(float(curve[np.searchsorted(CURVE_TIMES, m)]), 4)
                                          for m in monthly]},
        })
        return res
