"""Peer benchmarks: Kaplan-Meier median time to first IT job by group.

Medians are Kaplan-Meier estimates on the graduate training data, so graduates still searching at the
survey count correctly (a raw median of durations would ignore them and look too optimistic).
Groups with fewer than MIN_GROUP_N graduates are suppressed. Gender is never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from core import preprocess, schema, wording
from core.predict import profile_frame, round_month

MIN_GROUP_N = 10
GROUPS = ("overall", "degree_group", "university_type", "internship", "gpa_band")


def _gpa_band(g: float) -> str | None:
    if pd.isna(g):
        return None
    return "below_3.0" if g < 3.0 else "3.0_to_3.49" if g < 3.5 else "3.5_plus"


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group level of every row for every peer grouping (works for training rows and a single profile)."""
    raw = preprocess.raw_feature_frame(df)
    return pd.DataFrame({
        "overall": "all",
        "degree_group": raw["degree_group"],
        "university_type": raw["university_type"],
        "internship": np.where(raw["internship_months"].isna(), None,
                               np.where(raw["internship_months"] > 0, "yes", "no")),
        "gpa_band": raw["gpa"].map(_gpa_band),
    }, index=df.index)


def compute_peer_stats(train: pd.DataFrame, min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    groups = assign_groups(train)
    rows = []
    for g in GROUPS:
        for level, idx in groups.groupby(g).groups.items():
            sub = train.loc[idx]
            n = len(sub)
            row = {"group": g, "level": level, "n": n, "events": int(sub["event"].sum()), "suppressed": n < min_n,
                   "median": None, "median_lower": None, "median_upper": None, "employed_by_6m": None}
            if n >= min_n:
                km = KaplanMeierFitter().fit(sub["duration_months"], sub["event"])
                ci = median_survival_times(km.confidence_interval_)
                row.update({
                    "median": float(km.median_survival_time_),
                    "median_lower": float(ci.iloc[0, 0]), "median_upper": float(ci.iloc[0, 1]),
                    "employed_by_6m": round(1 - float(km.survival_function_at_times(6).iloc[0]), 3)
                    if sub["duration_months"].max() >= 6 else None,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _display(t, horizon: int) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "-"
    return f"more than {horizon}" if not np.isfinite(t) else str(round_month(t))


def peer_benchmark(stats: pd.DataFrame, profile: dict, horizon: int = schema.PREDICTION_HORIZON_MONTHS) -> dict:
    """Peer medians for the groups this profile belongs to (plus everyone)."""
    levels = assign_groups(profile_frame(profile)).iloc[0]
    out = []
    for g in GROUPS:
        level = levels[g]
        if level is None:
            continue
        match = stats[(stats["group"] == g) & (stats["level"] == level)]
        label = wording.PEER_LEVELS[g].get(level, str(level))
        if match.empty or bool(match.iloc[0]["suppressed"]):
            n = 0 if match.empty else int(match.iloc[0]["n"])
            out.append({"group": g, "group_label": wording.PEER_GROUPS[g], "level": level, "level_label": label,
                        "n": n, "suppressed": True, "median": None, "text":
                            wording.PEER_SUPPRESSED.format(min_n=MIN_GROUP_N)})
            continue
        r = match.iloc[0]
        med = float(r["median"])
        if np.isfinite(med):
            text = wording.PEER_MEDIAN.format(median=_display(med, horizon), lo=_display(r["median_lower"], horizon),
                                              hi=_display(r["median_upper"], horizon))
        else:
            text = wording.PEER_MEDIAN_OPEN.format(horizon=horizon)
        out.append({"group": g, "group_label": wording.PEER_GROUPS[g], "level": level, "level_label": label,
                    "n": int(r["n"]), "suppressed": False,
                    "median": {"months": round_month(med) if np.isfinite(med) else None,
                               "more_than": not np.isfinite(med), "display": _display(med, horizon)},
                    "median_interval": [_display(r["median_lower"], horizon), _display(r["median_upper"], horizon)],
                    "employed_by_6m": None if pd.isna(r["employed_by_6m"]) else float(r["employed_by_6m"]),
                    "text": text})
    return {"groups": out, "note": wording.PEER_NOTE, "min_group_n": MIN_GROUP_N}
