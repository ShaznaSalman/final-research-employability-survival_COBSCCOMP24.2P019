"""SYNTHETIC answers to the questionnaire-study items (core/survey_items.py) - demo data only.

Added to the synthetic graduate and employer files after they are generated, so every existing column
(and therefore every model result) is unchanged. Each row's answers come from a random stream seeded by
that row's other answers, so a duplicated submission gets identical answers and is still removed as a
duplicate by the cleaning step.

Answers are linked to the respondent's other answers where a real link is expected (e.g. a language
rated 4-5 in Q18 is usually also ticked in Q25; employed graduates feel more employable in Q40), so the
consistency checks and comparisons in core/survey_study.py have something to find. All assumptions are
listed here so they can be revised.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from core import schema, survey_items

SURVEY_SEED = 20261006 + 31
P_ITEM_SKIP = 0.03          # item non-response on optional questionnaire-study items
P_COMMENT = 0.08            # share who write anything in the open comment box

# P(option ticked) for select-all items before any adjustment.
CERT_AREA_W = {"cloud": .30, "programming": .20, "data_ai": .15, "networking": .10, "cybersecurity": .10,
               "database": .08, "agile_scrum": .05, "other": .02}
PLATFORM_P = {"youtube": .85, "udemy": .55, "coursera": .45, "linkedin_learning": .25, "google": .20,
              "microsoft_learn": .20, "aws_training": .15, "ai_tools": .60, "other": .05}
OTHER_LANG_P = {"c_cpp": .40, "csharp": .25, "php": .25, "kotlin": .10, "other": .05}
PROJECT_AREA_P = {"web": .70, "mobile": .40, "desktop": .25, "data_analytics": .30, "machine_learning": .30,
                  "cloud": .15, "cybersecurity": .10, "iot": .10}
STRATEGY_P = {"online_courses": .80, "documentation": .50, "practice_projects": .60, "mentors": .30,
              "community": .35, "university": .30, "ai_tools": .60}
MEDIUM_W = {"english": .70, "mixed": .20, "sinhala": .08, "tamil": .02}
JOB_PLATFORM_P = {"linkedin": .85, "topjobs": .70, "company_sites": .55, "indeed": .20, "university_career": .30,
                  "agencies": .15, "other": .05}
LACKED_P = {"experience": .55, "framework": .40, "cloud": .30, "programming_language": .25, "database": .20,
            "git": .15, "communication": .30, "problem_solving": .20, "certifications": .25}
FEEDBACK_P = {"technical": .45, "communication": .35, "confidence": .30, "experience": .45, "cultural_fit": .10,
              "certifications": .10, "salary": .10, "other": .03}
BELIEVED_W = {"technical_skills": .90, "internship": .70, "projects": .55, "communication": .50,
              "certifications": .45, "interview_prep": .40, "networking": .35, "industry_connections": .30,
              "degree_class": .25, "university_reputation": .25}
CHALLENGE_W = {"experience": .35, "job_opportunities": .20, "technical_skills": .15, "interviews": .12,
               "confidence": .08, "industry_knowledge": .07, "other": .03}
EXPLANATION_P = {"improvements": .80, "learning_paths": .65, "industry_comparison": .60, "strengths": .50,
                 "similar_profiles": .35}
COMMENTS = ["More industry projects during the degree would help.", "Internships should be compulsory.",
            "Hard to find entry-level roles without experience.", "Interview preparation support is needed."]

# Employers
ROLE_W = {"hr_manager": .22, "recruiter": .18, "tech_lead": .20, "engineer_hiring": .14, "eng_manager": .12,
          "project_manager": .06, "cto_cio": .06, "other": .02}
EXPERIENCE_W = {"<1": .08, "1-3": .30, "4-6": .30, "7-10": .20, "10+": .12}
ROLES_P = {"software_engineer": .85, "qa": .50, "business_analyst": .35, "data_analyst": .30, "ui_ux": .20,
           "devops": .20, "cloud": .20, "cybersecurity": .15, "data_scientist": .15, "ai_ml_engineer": .15,
           "network_admin": .15, "other": .03}
HIRES_BY_SIZE = {"<50": [.10, .70, .15, .05, .00], "50-200": [.02, .45, .33, .15, .05],
                 "201-500": [.00, .20, .35, .30, .15], "500+": [.00, .05, .20, .35, .40]}
FUTURE_P = {"ai": .80, "cloud": .65, "ml": .60, "cybersecurity": .55, "data_engineering": .40, "automation": .40,
            "devops": .35, "software_engineering": .30, "iot": .15, "blockchain": .10}
UPDATE_W = {"more_than_yearly": .35, "1-2_years": .45, "3-5_years": .12, "rarely": .08}
TECH_WEAK_W = {"se_practices": .50, "testing": .45, "cloud": .45, "devops": .40, "programming": .35,
               "cybersecurity": .30, "database": .25, "ai_ml": .25, "data_analysis": .20, "other": .03}
SOFT_WEAK_W = {"communication": .55, "confidence": .50, "time_management": .35, "professional_behaviour": .30,
               "problem_solving": .30, "adaptability": .20, "teamwork": .15, "attitude": .30}
IMPROVE_P = {"practical_dev": .70, "industry_projects": .65, "internships": .55, "modern_tech": .50,
             "soft_skills": .50, "career_prep": .30, "certifications": .25}
OUTPUT_P = {"missing_skills": .75, "training_areas": .60, "strengths": .55, "job_roles": .45,
            "suitability_score": .40, "interview_prep": .35}
CHALLENGE_EMP_W = {"experience": .55, "curriculum_gap": .50, "soft_skills": .45, "tech_skills": .45,
                   "salary_expectations": .30, "attrition": .30, "long_process": .10, "other": .03}
RANK_W = {"technical_depth": .90, "problem_solving": .70, "communication": .60, "interview": .55,
          "internship": .55, "portfolio": .45, "cultural_fit": .30, "certifications": .20}


def _row_rng(row: pd.Series) -> np.random.Generator:
    """Random stream determined by the row's answers (ids and timestamps excluded)."""
    content = "|".join(str(v) for k, v in row.items() if k not in ("respondent_id", "employer_id", "survey_date"))
    return np.random.default_rng(SURVEY_SEED + int(hashlib.md5(content.encode()).hexdigest()[:12], 16))


def _num(v, default: float) -> float:
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _likert(mean: float, rng, sd: float = 0.8) -> int:
    return int(np.clip(np.rint(mean + rng.normal(0, sd)), 1, 5))


def _ticks(probs: dict, rng, order: list[str], min_one: bool = False) -> str | None:
    keys = [k for k, p in probs.items() if rng.random() < min(max(p, 0), 1)]
    if min_one and not keys:
        keys = [max(probs, key=probs.get)]
    return ";".join(sorted(keys, key=order.index)) or None


def _pick(weights: dict, rng, k: int, order: list[str]) -> str | None:
    keys, w = list(weights), np.array(list(weights.values()), float)
    chosen = rng.choice(keys, size=min(k, len(keys)), replace=False, p=w / w.sum())
    return ";".join(sorted(chosen, key=order.index)) or None


def _p_from_rating(r: float) -> float:
    return {1: .05, 2: .10, 3: .45, 4: .80, 5: .90}.get(int(np.clip(round(r), 1, 5)), .3)


def _graduate_answers(row: pd.Series) -> dict:
    rng = _row_rng(row)
    opts = lambda name: survey_items.options(survey_items.GRADUATE_ITEMS, name)  # noqa: E731
    tech = lambda k, d=3.0: _num(row.get(f"tech_{k}"), d)  # noqa: E731
    soft = lambda k, d=3.5: _num(row.get(f"soft_{k}"), d)  # noqa: E731
    grad = row.get("respondent_status") == "graduated"
    employed = row.get("ever_it_job") == "yes"
    searched = grad and row.get("employment_status") not in schema.NOT_IN_MARKET_STATUSES
    certs = int(_num(row.get("certifications_count"), 0))
    projects = _num(row.get("project_count"), 2)
    learning = _num(row.get("self_learning_freq"), 3)
    a: dict = {}

    a["instruction_medium"] = _pick(MEDIUM_W, rng, 1, opts("instruction_medium"))
    a["english_proficiency"] = _likert(soft("communication") + 0.2 * (a["instruction_medium"] == "english")
                                       - 0.4 * (a["instruction_medium"] in ("sinhala", "tamil")), rng, 0.8)
    a["degree_relevance"] = _likert(3.2 + 0.5 * (row.get("industry_modules") == "yes"), rng)
    a["cert_areas"] = _pick(CERT_AREA_W, rng, min(certs, 3), opts("cert_areas")) if certs > 0 else None
    has_pf = row.get("has_portfolio") == "yes"
    a["portfolio_update_freq"] = (rng.choice(["rarely", "occasionally", "frequently"], p=[.3, .45, .25]) if has_pf
                                  else rng.choice(["never", "rarely"], p=[.9, .1]))
    a["learning_platforms"] = _ticks({k: p * (0.6 + 0.12 * learning) for k, p in PLATFORM_P.items()}, rng,
                                     opts("learning_platforms"), min_one=True)
    langs = {k: _p_from_rating(tech(k)) for k in ("java", "python", "javascript")} | OTHER_LANG_P
    a["languages"] = _ticks(langs, rng, opts("languages"), min_one=True)
    areas = dict(PROJECT_AREA_P)
    areas["data_analytics"] += 0.3 * (row.get("degree_type") == "data_science")
    areas["machine_learning"] += 0.3 * (row.get("degree_type") in ("data_science", "ai_ml"))
    areas["cybersecurity"] += 0.4 * (row.get("degree_type") == "cyber_security")
    a["project_areas"] = ("none" if projects == 0 and rng.random() < 0.7
                          else _ticks(areas, rng, opts("project_areas"), min_one=True))
    a["problem_confidence"] = _likert(soft("problem_solving"), rng, 0.85)
    practices = {"version_control": _p_from_rating(tech("git")), "testing": _p_from_rating(tech("software_testing")),
                 "cicd": _p_from_rating(tech("devops_cicd")) * 0.8, "agile_scrum": .60, "code_reviews": .40,
                 "documentation": .55}
    a["dev_practices"] = _ticks(practices, rng, opts("dev_practices")) or "none"
    weak = lambda r: 1 / (1 + np.exp(1.6 * (r - 2.8)))  # noqa: E731  low self-rating -> more likely ticked
    a["improvement_areas"] = _ticks({
        "programming": weak(tech("programming_fundamentals")), "database": weak(tech("sql_databases")),
        "cloud": weak(tech("cloud_computing")), "cybersecurity": weak(tech("cybersecurity")),
        "devops": weak(tech("devops_cicd")), "ai_ml": weak(tech("ai_ml")),
        "communication": weak(soft("communication")), "problem_solving": weak(soft("problem_solving")),
        "interview_skills": .45, "professional_experience": .50 if _num(row.get("internship_months"), 0) < 3 else .25,
    }, rng, opts("improvement_areas"), min_one=True)
    a["lacked_skills"] = _ticks(LACKED_P, rng, opts("lacked_skills"), min_one=True) if searched else None
    a["explain_nontechnical"] = _likert(soft("communication") - 0.2, rng, 0.85)
    a["team_confidence"] = _likert(soft("teamwork"), rng, 0.8)
    a["group_projects"] = "yes" if rng.random() < 0.92 else "no"
    a["group_project_count"] = (rng.choice(["1-2", "3-5", "5+"], p=[.35, .45, .20]) if a["group_projects"] == "yes"
                                else None)
    a["learning_strategies"] = _ticks(STRATEGY_P, rng, opts("learning_strategies"), min_one=True)
    platforms = JOB_PLATFORM_P if searched else {k: p * 0.5 for k, p in JOB_PLATFORM_P.items()}
    a["job_platforms"] = _ticks(platforms, rng, opts("job_platforms"), min_one=searched)
    a["cv_update_freq"] = rng.choice(opts("cv_update_freq"), p=[.05, .20, .45, .30] if searched else [.15, .35, .40, .10])
    a["linkedin_profile"] = "yes" if rng.random() < (0.88 if grad else 0.75) else "no"
    a["linkedin_completeness"] = _likert(3.4 + 0.3 * employed, rng) if a["linkedin_profile"] == "yes" else None
    a["interview_feedback"] = ("yes" if rng.random() < 0.45 else "no") if searched else None
    a["feedback_areas"] = (_ticks(FEEDBACK_P, rng, opts("feedback_areas"), min_one=True)
                           if a["interview_feedback"] == "yes" else None)
    tech_mean = np.nanmean([_num(row.get(c), np.nan) for c in schema.TECH_COLUMNS] or [3.0])
    a["self_employability"] = _likert(2.6 + 0.9 * employed + 0.35 * ((tech_mean if not np.isnan(tech_mean) else 3) - 3)
                                      - 0.3 * (searched and not employed), rng)
    a["believed_factors"] = _pick(BELIEVED_W, rng, int(rng.choice([3, 4, 5], p=[.2, .3, .5])), opts("believed_factors"))
    a["university_prepared"] = _likert(3.0 + 0.4 * (row.get("industry_modules") == "yes"), rng, 0.9)
    a["biggest_challenge"] = _pick(CHALLENGE_W, rng, 1, opts("biggest_challenge"))
    a["explanation_importance"] = _likert(4.2, rng, 0.7)
    a["explanation_types"] = _ticks(EXPLANATION_P, rng, opts("explanation_types"), min_one=True)
    a["would_use_system"] = _likert(3.9, rng, 0.8)
    a["comments"] = rng.choice(COMMENTS) if rng.random() < P_COMMENT else None
    for k in a:   # optional items are sometimes skipped
        if k != "comments" and a[k] is not None and rng.random() < P_ITEM_SKIP:
            a[k] = None
    return a


def _employer_answers(row: pd.Series) -> dict:
    rng = _row_rng(row)
    opts = lambda name: survey_items.options(survey_items.EMPLOYER_ITEMS, name)  # noqa: E731
    a: dict = {
        "respondent_role": _pick(ROLE_W, rng, 1, opts("respondent_role")),
        "recruit_experience": _pick(EXPERIENCE_W, rng, 1, opts("recruit_experience")),
        "roles_recruited": _ticks(ROLES_P, rng, opts("roles_recruited"), min_one=True),
        "annual_hires": rng.choice(opts("annual_hires"), p=HIRES_BY_SIZE.get(row.get("org_size"), HIRES_BY_SIZE["50-200"])),
        "future_tech": _ticks(FUTURE_P, rng, opts("future_tech"), min_one=True),
        "skill_update_freq": _pick(UPDATE_W, rng, 1, opts("skill_update_freq")),
        "tech_weaknesses": _pick(TECH_WEAK_W, rng, int(rng.integers(2, 6)), opts("tech_weaknesses")),
        "soft_weaknesses": _pick(SOFT_WEAK_W, rng, int(rng.integers(1, 5)), opts("soft_weaknesses")),
        "grad_alignment": _likert(2.9, rng, 0.8),
        "emp_english_importance": _likert(4.3, rng, 0.7),
        "uni_improvements": _ticks(IMPROVE_P, rng, opts("uni_improvements"), min_one=True),
        "ai_usefulness": _likert(3.6, rng, 0.9),
        "emp_explanation_importance": _likert(4.0, rng, 0.8),
        "ai_outputs": _ticks(OUTPUT_P, rng, opts("ai_outputs"), min_one=True),
        "hiring_challenges": _pick(CHALLENGE_EMP_W, rng, int(rng.integers(1, 4)), opts("hiring_challenges")),
        "emp_comments": rng.choice(COMMENTS) if rng.random() < P_COMMENT / 2 else None,
    }
    top3 = rng.choice(list(RANK_W), size=3, replace=False, p=np.array(list(RANK_W.values())) / sum(RANK_W.values()))
    for k, _ in survey_items.EMPLOYER_RANK_FACTORS:
        a[f"rank_{k}"] = float(list(top3).index(k) + 1) if k in top3 else np.nan
    for k in list(a):
        if not k.startswith("rank_") and k != "emp_comments" and a[k] is not None and rng.random() < P_ITEM_SKIP:
            a[k] = None
    return a


def add_graduate_survey(raw: pd.DataFrame) -> pd.DataFrame:
    answers = pd.DataFrame([_graduate_answers(r) for _, r in raw.iterrows()], index=raw.index)
    return pd.concat([raw, answers[[f.name for f in schema.SURVEY_FIELDS]]], axis=1)


def add_employer_survey(df: pd.DataFrame) -> pd.DataFrame:
    answers = pd.DataFrame([_employer_answers(r) for _, r in df.iterrows()], index=df.index)
    return pd.concat([df, answers[[f.name for f in schema.EMPLOYER_SURVEY_FIELDS]]], axis=1)
