"""What-if: change ACTIONABLE features only, re-predict, report the change.

Rules (from core/schema.py ACTIONABLE_FEATURES):
  - only actionable features may change; anything else is rejected
  - values must lie within the allowed [min, max]
  - GPA may change only for final-year students, by at most `max_change`
  - changing a skill-area rating sets every item in that area to the new value
The changed profile is checked against the training data: values beyond the observed range, or in the
outer 5% tails, produce a caution message.
"""
from __future__ import annotations

import numpy as np

from core import preprocess, schema, wording
from core.predict import AREA_ITEMS, Predictor, profile_frame

RARE_PERCENTILES = ("5%", "95%")


class WhatIfError(ValueError):
    """Raised for a change that is not allowed (non-actionable feature, out of range, ...)."""


def apply_changes(profile: dict, changes: dict) -> dict:
    status = profile.get("respondent_status", "graduated")
    new = dict(profile)
    for key, value in changes.items():
        spec = schema.ACTIONABLE_FEATURES.get(key)
        if spec is None:
            raise WhatIfError(f"'{key}' cannot be changed in what-if: it is not an actionable feature. "
                              f"Allowed: {', '.join(schema.ACTIONABLE_FEATURES)}.")
        if spec.applies_to != "all" and spec.applies_to != status:
            raise WhatIfError(f"'{key}' can only be changed for {spec.applies_to.replace('_', '-')} respondents.")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise WhatIfError(f"'{key}' must be a number, got {value!r}.") from None
        if not (spec.min <= v <= spec.max):
            raise WhatIfError(f"'{key}' must be between {spec.min:g} and {spec.max:g}, got {v:g}.")
        if spec.max_change is not None:
            current = profile.get(key)
            if current is None:
                raise WhatIfError(f"'{key}' needs a current value before it can be changed.")
            if abs(v - float(current)) > spec.max_change + 1e-9:
                raise WhatIfError(f"'{key}' can change by at most {spec.max_change:g} from its current value.")
        new[key] = v
        for item in AREA_ITEMS.get(key, ()):        # area rating overrides the items in that area
            new[item] = v
    return new


def caution_messages(predictor: Predictor, profile: dict, changed_keys) -> list[str]:
    """Flag values outside the training range (or in its outer 5% tails)."""
    stats = predictor.bundle.get("training_summary", {}).get("numeric", {})
    df = profile_frame(profile)
    values = preprocess.derive_and_block(df).iloc[0]
    to_check = [f for f, t in schema.MODEL_FEATURES.items() if t == "numeric"]
    to_check += [k for k in changed_keys if k.startswith("area_")]
    outside, rare = [], []
    for f in to_check:
        if f not in stats or f not in values or np.isnan(values[f]):
            continue
        s, v = stats[f], float(values[f])
        name = wording.FEATURES.get(f, {}).get("name") or schema.ACTIONABLE_FEATURES[f].label
        if v < s["min"] or v > s["max"]:
            outside.append(wording.CAUTION_OUTSIDE.format(direction="low" if v < s["min"] else "high",
                                                          feature=name))
        elif v < s[RARE_PERCENTILES[0]] or v > s[RARE_PERCENTILES[1]]:
            rare.append(name)
    msgs = outside
    if rare:
        msgs.append(wording.CAUTION_RARE.format(features=", ".join(rare)))
    return msgs


def _months(v: dict) -> float:
    return np.inf if v["more_than"] else v["months"]


def what_if(predictor: Predictor, profile: dict, changes: dict) -> dict:
    new_profile = apply_changes(profile, changes)
    original, changed = predictor.predict(profile), predictor.predict(new_profile)
    delta = {}
    for k in ("low", "median", "high"):
        a, b = _months(original[k]), _months(changed[k])
        delta[k] = int(b - a) if np.isfinite(a) and np.isfinite(b) else None
    d = delta["median"]
    fill = {"new": changed["median"]["display"], "old": original["median"]["display"]}
    if d is None:
        message = wording.WHATIF["open_ended"].format(**fill)
    elif d < 0:
        message = wording.WHATIF["sooner"].format(months=-d, s="" if d == -1 else "s", **fill)
    elif d > 0:
        message = wording.WHATIF["later"].format(months=d, s="" if d == 1 else "s", **fill)
    else:
        message = wording.WHATIF["same"].format(**fill)
    for r in (original, changed):
        r.pop("curve", None)
    return {"original": original, "changed": changed, "change_months": delta, "message": message,
            "applied_changes": {k: float(v) for k, v in changes.items()},
            "cautions": caution_messages(predictor, new_profile, changes.keys())}
