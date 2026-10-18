"""Profile-to-role comparisons using skill mentions in the entry-level job-ad study."""
from __future__ import annotations

import math

from core import schema
from core.predict import profile_frame


def role_pathways(profile: dict, job_market: dict, limit: int = 3) -> dict:
    """Rank role families by weighted self-ratings and return role-specific skill gaps.

    The match is descriptive, not a hiring probability: each role's job-ad skill prevalence weights the
    graduate's corresponding self-rating. Roles with very small ad samples remain visible with their counts.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    frame = profile_frame({k: v for k, v in profile.items() if k != "months_since_graduation"})
    ratings = {
        skill.key: (None if frame.iloc[0][skill.column] is None or
                    not math.isfinite(float(frame.iloc[0][skill.column])) else
                    float(frame.iloc[0][skill.column]))
        for skill in schema.SKILL_VOCABULARY
    }
    labels = {skill.key: skill.label for skill in schema.SKILL_VOCABULARY}
    ranked = []
    for role, skill_percentages in job_market.get("by_role", {}).items():
        if role == "Other":
            continue
        weights = {key: max(0.0, float(pct)) for key, pct in skill_percentages.items()
                   if key in ratings and ratings[key] is not None}
        weight_sum = sum(weights.values())
        if not weight_sum:
            continue
        score = sum(weights[key] * ratings[key] for key in weights) / weight_sum
        gaps = [
            {"skill": key, "label": labels[key], "your_rating": ratings[key],
             "ads_pct": float(pct), "priority": float(pct) * (5.0 - ratings[key])}
            for key, pct in skill_percentages.items()
            if key in ratings and ratings[key] is not None and float(pct) > 0 and ratings[key] < 4
        ]
        gaps.sort(key=lambda item: (-item["priority"], -item["ads_pct"], item["label"]))
        ranked.append({
            "role": role,
            "ads": int(job_market.get("roles", {}).get(role, 0)),
            "match_rating": round(score, 2),
            "gaps": gaps[:5],
        })
    ranked.sort(key=lambda item: (-item["match_rating"], -item["ads"], item["role"]))
    return {"recommendations": ranked[:limit], "roles": ranked}


def role_gaps(pathways: dict, role: str) -> list[dict]:
    """Return skill gaps for an available role family, rejecting unknown roles."""
    selected = next((item for item in pathways["roles"] if item["role"] == role), None)
    if selected is None:
        raise ValueError(f"Unknown role family: {role}")
    return selected["gaps"]
