"""Personalised survey insights for the product (POST /insights).

Uses the questionnaire study (artifacts/questionnaire_study.json) - employer and graduate answers that are not
model inputs - to add context to a prediction: what employers value, which of the user's weaker skills employers
most often see as weak in graduates, which technologies employers expect to grow, what graduates who found work
quickly had done, and where graduates searched and learned. Nothing here changes the prediction.
"""
from __future__ import annotations

from core import schema

WEAK_RATING = 3            # a skill rated at or below this counts as one of the user's weaker skills
MAX_ITEMS = 4

# Employer answer options -> skill keys in the shared vocabulary
TECH_WEAKNESS_SKILLS = {"programming": "programming_fundamentals", "se_practices": "git", "database": "sql_databases",
                        "cloud": "cloud_computing", "testing": "software_testing", "cybersecurity": "cybersecurity",
                        "devops": "devops_cicd", "ai_ml": "ai_ml", "data_analysis": "data_analytics"}
SOFT_WEAKNESS_SKILLS = {"communication": "communication", "teamwork": "teamwork", "problem_solving": "problem_solving",
                        "adaptability": "adaptability", "time_management": "time_management",
                        "professional_behaviour": "professionalism", "confidence": "professionalism",
                        "attitude": "professionalism"}
FUTURE_TECH_SKILLS = {"ai": "ai_ml", "ml": "ai_ml", "cloud": "cloud_computing", "cybersecurity": "cybersecurity",
                      "data_engineering": "data_analytics", "devops": "devops_cicd", "automation": "devops_cicd"}
# Certification areas (graduate Q10) -> employer growth areas (E10)
CERT_GROWTH = {"cloud": ("cloud",), "cybersecurity": ("cybersecurity",), "data_ai": ("ai", "ml", "data_engineering"),
               "database": ("data_engineering",), "programming": ("software_engineering",), "agile_scrum": ("devops",),
               "networking": ()}
# Skill-gap areas (Q30 vs E12/E13, labels from survey_study) -> skill key for the user's rating
GAP_SKILLS = {"Programming": "programming_fundamentals", "Database": "sql_databases", "Cloud": "cloud_computing",
              "Cybersecurity": "cybersecurity", "DevOps": "devops_cicd", "AI / ML": "ai_ml",
              "Communication": "communication", "Problem solving": "problem_solving"}
UNDERESTIMATE_POINTS = 10   # employers flag the gap at least this many percentage points more often than graduates
# Optional answers the user can give (they do not reach the model) -> behaviour key in the study
EXTRAS = {"has_portfolio": "has_portfolio", "hackathons": "hackathons", "open_source": "open_source",
          "self_learning": "self_learning", "early_prep": "early_prep"}
PROFILE_FACTORS = {"internship_experience": ("internship_months", "{:g} months"),
                   "certifications": ("certifications_count", "{:g}"), "personal_projects": ("project_count", "{:g}"),
                   "gpa": ("gpa", "{:.2f}")}


def skill_rating(profile: dict, key: str) -> float | None:
    """The user's rating of a skill: the item if given (Detailed mode), else its area rating (Quick mode)."""
    skill = schema.SKILLS_BY_KEY[key]
    if profile.get(skill.column) is not None:
        return float(profile[skill.column])
    if skill.kind == "soft":
        return profile.get("area_soft")
    area = next(a for a, v in schema.SKILL_AREAS.items() if key in v["skills"])
    return profile.get(f"area_{area}")


def _pct(rows: list[dict], key: str) -> float | None:
    return next((r["pct"] for r in rows if r["key"] == key), None)


def build(profile: dict, extras: dict | None, study: dict) -> dict:
    extras = extras or {}
    ev, ge = study["employer_views"], study["graduate_experience"]

    # Employer-observed weaknesses that match the user's weaker skills
    watch = []
    for rows, mapping in ((ev["tech_weaknesses"], TECH_WEAKNESS_SKILLS), (ev["soft_weaknesses"], SOFT_WEAKNESS_SKILLS)):
        for r in rows:
            key = mapping.get(r["key"])
            rating = skill_rating(profile, key) if key else None
            if rating is not None and rating <= WEAK_RATING and r["pct"]:
                watch.append({"skill": schema.SKILLS_BY_KEY[key].label, "your_rating": rating,
                              "employer_pct": r["pct"],
                              "text": f"{r['pct']:.0f}% of employers say fresh graduates are often weak in "
                                      f"{r['label'].lower()} - you rated {schema.SKILLS_BY_KEY[key].label} "
                                      f"{rating:g}/5."})
    seen, watch_out = set(), []
    for w in sorted(watch, key=lambda w: -w["employer_pct"]):
        if w["skill"] not in seen:
            seen.add(w["skill"])
            watch_out.append(w)

    growing = []
    for r in ev["future_tech"][:6]:
        key = FUTURE_TECH_SKILLS.get(r["key"])
        rating = skill_rating(profile, key) if key else None
        growing.append({"technology": r["label"], "employer_pct": r["pct"], "your_rating": rating,
                        "text": f"{r['pct']:.0f}% of employers expect {r['label']} to grow in importance"
                                + (f" - your rating: {rating:g}/5." if rating is not None else ".")})

    values = []
    for f in study["employer_factors"]:
        mine = PROFILE_FACTORS.get(f["factor"])
        you = mine[1].format(profile[mine[0]]) if mine and profile.get(mine[0]) is not None else None
        values.append({**f, "you": you})

    tips = []
    for r in study["fast_finders"]["rows"]:
        answer = extras.get(next((k for k, v in EXTRAS.items() if v == r["key"]), ""), None)
        more_common = r["pct_fast"] > r["pct_other"]
        text = (f"{r['label']}: {r['pct_fast']:.0f}% of graduates who found an IT job within "
                f"{study['fast_finders']['months']} months vs {r['pct_other']:.0f}% of others.")
        tips.append({**r, "you": answer, "more_common_among_fast": more_common,
                     "suggest": bool(more_common and r["significant"] and answer is False), "text": text})

    # Certifications graduates hold (Q10), next to how many employers expect that area to grow (E10)
    growth = {r["key"]: r["pct"] for r in ev["future_tech"]}
    certs = []
    for r in ge["cert_areas"]:
        keys = CERT_GROWTH.get(r["key"], ())
        g = max((growth.get(k) or 0 for k in keys), default=None) if keys else None
        if r["key"] == "other" or not r["pct"]:
            continue
        certs.append({"area": r["label"], "graduates_pct": r["pct"], "employer_growth_pct": g,
                      "text": f"{r['label']}: held by {r['pct']:.0f}% of certified graduates"
                              + (f"; {g:.0f}% of employers expect this area to grow." if g else ".")})
    certs.sort(key=lambda c: (-(c["employer_growth_pct"] or 0), -c["graduates_pct"]))

    # Gaps employers flag much more often than graduates do (Q30 vs E12/E13)
    underestimated = []
    for r in study["graduates_vs_employers"]["skill_gaps"]:
        if r["employers_pct"] - r["graduates_pct"] < UNDERESTIMATE_POINTS or not r["significant"]:
            continue
        key = GAP_SKILLS.get(r["label"])
        rating = skill_rating(profile, key) if key else None
        mine = (f" You rated it {rating:g}/5 - worth checking against employers' view."
                if rating is not None and rating >= 4 else "")
        underestimated.append({"area": r["label"], "employers_pct": r["employers_pct"],
                               "graduates_pct": r["graduates_pct"], "your_rating": rating,
                               "text": f"{r['label']}: {r['employers_pct']:.0f}% of employers see it as a common "
                                       f"weakness, but only {r['graduates_pct']:.0f}% of graduates think they need to "
                                       f"improve it.{mine}"})

    return {
        "certifications": certs[:MAX_ITEMS],
        "interviewers_said_lacking": ge["lacked_skills"][:MAX_ITEMS],
        "interview_feedback": ge["feedback_areas"][:3],
        "roles_hired": ev["roles_recruited"][:5],
        "project_areas": ge["project_areas"][:5],
        "underestimated_gaps": sorted(underestimated, key=lambda u: u["graduates_pct"] - u["employers_pct"])[:MAX_ITEMS],
        "is_synthetic": study["is_synthetic"],
        "employers_value": values,
        "employer_top3": study["employer_ranking"][:MAX_ITEMS],
        "watch_outs": watch_out[:MAX_ITEMS],
        "growing": growing[:MAX_ITEMS + 1],
        "fast_finders": tips,
        "where_to_search": ge["search_platforms_employed"][:MAX_ITEMS],
        "where_to_learn": ge["learning_platforms"][:MAX_ITEMS],
        "how_they_learn": ge["learning_strategies"][:3],
        "common_challenges": sorted(ge["challenges"], key=lambda r: -(r["pct"] or 0))[:3],
        "note": ("From the graduate and employer questionnaires. These are associations and opinions, not effects "
                 "on your estimate - they never change the prediction above."),
    }
