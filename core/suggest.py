"""Suggested combinations to explore: small, realistic changes to ACTIONABLE features that profiles like the
user's were associated with finding work sooner.

Search space: every actionable feature moved by one or two realistic steps, alone or in pairs, within the
documented ranges (schema.ACTIONABLE_FEATURES) and the what-if rules (core/whatif.py). Each candidate is scored
by the final model's predicted median; candidates are ranked by months sooner per sqrt(effort), the
suggestions shown use different features from one another, and any
candidate whose profile falls outside the range seen in the training data is discarded. Wording is
associational (core/wording.py).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from core import preprocess, schema, wording
from core.models import CURVE_TIMES
from core.predict import AREA_ITEMS, Predictor, condition_on_searching, crossing_times, profile_frame, round_month
from core.whatif import WhatIfError, apply_changes, caution_messages

# (step size, effort units) per actionable feature; effort is a rough "how hard is this" scale.
STEPS = {
    "internship_months": ((3, 1), (6, 2)),
    "certifications_count": ((1, 1), (2, 2)),
    "project_count": ((2, 1), (4, 2)),
    "gpa": ((0.1, 1), (0.2, 2)),
    **{f"area_{a}": ((0.5, 1), (1.0, 2)) for a in list(schema.SKILL_AREAS) + ["soft"]},
}
MAX_EFFORT = 3
MIN_MONTHS_SOONER = 1
TOP_K = 3
AREA_LABELS = {f"area_{a}": v["label"] for a, v in {**schema.SKILL_AREAS, **schema.SOFT_AREA}.items()}


def current_values(profile: dict) -> dict:
    """Current value of every actionable feature (area ratings from items when given in detail)."""
    d = preprocess.derive_and_block(profile_frame({k: v for k, v in profile.items()
                                                   if k != "months_since_graduation"})).iloc[0]
    out = {}
    for feat in schema.ACTIONABLE_FEATURES:
        if feat.startswith("area_"):
            v = profile.get(feat)
            if v is None:
                items = [d.get(c) for c in AREA_ITEMS[feat]]
                items = [x for x in items if pd.notna(x)]
                v = round(2 * float(np.mean(items))) / 2 if items else None
        else:
            v = profile.get(feat, d.get(feat))
        out[feat] = None if v is None or pd.isna(v) else float(v)
    return out


def _phrase(feat: str, old: float, new: float) -> str:
    d = new - old
    if feat.startswith("area_"):
        return wording.SUGGESTION_CHANGE["area"].format(label=AREA_LABELS[feat], new=new, old=old)
    if feat == "gpa":
        return wording.SUGGESTION_CHANGE["gpa"].format(new=new)
    return wording.SUGGESTION_CHANGE[feat].format(d=d, s="" if d == 1 else "s")


def _medians(predictor: Predictor, profiles: list[dict], m: float | None) -> np.ndarray:
    df = pd.concat([profile_frame({k: v for k, v in p.items() if k != "months_since_graduation"})
                    for p in profiles], ignore_index=True)
    S = predictor.survival(df)
    if m:
        S = condition_on_searching(S, m)
    return crossing_times(S, 0.5, predictor.horizon, CURVE_TIMES)


def suggest(predictor: Predictor, profile: dict, top_k: int = TOP_K) -> dict:
    current = current_values(profile)
    m = profile.get("months_since_graduation") or None
    options = []                                                  # (feature, new value, effort)
    for feat, steps in STEPS.items():
        old = current.get(feat)
        spec = schema.ACTIONABLE_FEATURES[feat]
        if old is None or spec.applies_to not in ("all", profile.get("respondent_status", "graduated")):
            continue
        for step, effort in steps:
            new = round(old + step, 2)
            if new <= spec.max and (spec.max_change is None or step <= spec.max_change + 1e-9):
                options.append((feat, new, effort))

    candidates = [((o,), o[2]) for o in options]
    candidates += [((a, b), a[2] + b[2]) for a, b in combinations(options, 2)
                   if a[0] != b[0] and a[2] + b[2] <= MAX_EFFORT]
    changed, kept = [], []
    for combo, effort in candidates:
        changes = {f: v for f, v, _ in combo}
        try:
            changed.append(apply_changes(profile, changes))
            kept.append((changes, effort))
        except WhatIfError:
            continue
    if not kept:
        return {"suggestions": [], "message": wording.NO_SUGGESTIONS, "note": wording.SUGGESTIONS_NOTE}

    base = _medians(predictor, [profile], m)[0]
    new_meds = _medians(predictor, changed, m)
    scored = []
    for (changes, effort), new_med, prof in zip(kept, new_meds, changed):
        if np.isfinite(base) and np.isfinite(new_med):
            sooner = round_month(base) - round_month(new_med)
        elif not np.isfinite(base) and np.isfinite(new_med):
            sooner = predictor.horizon                            # from "more than X" to within follow-up
        else:
            continue
        if sooner >= MIN_MONTHS_SOONER:
            scored.append((sooner / np.sqrt(effort), sooner, effort, changes, new_med, prof))
    scored.sort(key=lambda x: (-x[0], x[2], -x[1]))   # months sooner per sqrt(effort); ties -> simpler change

    suggestions, used = [], []
    for _, sooner, effort, changes, new_med, prof in scored:
        if any(set(changes) & u for u in used):                   # each suggestion uses different features
            continue
        if caution_messages(predictor, prof, changes.keys()):     # outside the data: not suggested
            continue
        phrase = " and ".join(_phrase(f, current[f], v) for f, v in changes.items())
        old_txt = "more than %d" % predictor.horizon if not np.isfinite(base) else str(round_month(base))
        new_txt = str(round_month(new_med))
        if np.isfinite(base):
            text = wording.SUGGESTION.format(changes=phrase, n=sooner, s="" if sooner == 1 else "s",
                                             new=new_txt, old=old_txt)
        else:
            text = wording.SUGGESTION_OPEN.format(changes=phrase, horizon=predictor.horizon, new=new_txt)
        suggestions.append({"changes": changes, "effort": int(effort), "months_sooner": int(sooner),
                            "new_median_months": round_month(new_med), "text": text})
        used.append(set(changes))
        if len(suggestions) == top_k:
            break
    return {"suggestions": suggestions, "message": None if suggestions else wording.NO_SUGGESTIONS,
            "note": wording.SUGGESTIONS_NOTE}
