"""Questionnaire study: analyses every questionnaire item that is not a model input.

  python -m core.survey_study      -> artifacts/questionnaire_study.{json,md}

1. Repeated questions - items that ask about a model question again in other words are tested for
   consistency (Spearman correlation, or the contradiction rate for yes/no logic). A strong correlation
   means the two items measure the same thing (one can be dropped); a weak one means they do not.
2. Graduates vs employers - the same idea asked of both groups (preparedness, explanations, usefulness,
   skill gaps, what matters most), compared with Mann-Whitney U or Fisher's exact test (BH-corrected).
3. Self-perceived employability (Q40) vs the observed time to first IT job.
4. Requirements for the artifact (explanation needs, wanted outputs, intention to use).
5. Descriptive results for every item (shares with Wilson 95% intervals; Likert means).

Inputs: data/processed/questionnaire_graduates.csv and employers.csv (written by core/clean.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats

from core import schema, survey_items
from core.market_study import bh, wilson

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
STRONG, MODERATE = 0.5, 0.3     # |rho| thresholds for "repeats" / "partly overlaps"
ALPHA = 0.05

LABELS = {n: dict(o) for n, _, _, o, _ in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS if o}
ITEM = {n: i for n, i, *_ in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS}
DESC = {n: d for n, _, _, _, d in survey_items.GRADUATE_ITEMS + survey_items.EMPLOYER_ITEMS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _keys(v) -> list[str] | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v).split(";")


def has(series: pd.Series, key: str) -> pd.Series:
    """1/0 for 'key ticked' in a select-all column; NaN where the item was not answered."""
    return series.map(lambda v: np.nan if _keys(v) is None else float(key in _keys(v))).astype(float)


def n_ticked(series: pd.Series, exclude=("none",)) -> pd.Series:
    return series.map(lambda v: np.nan if _keys(v) is None
                      else float(sum(k not in exclude for k in _keys(v)))).astype(float)


def spearman(a: pd.Series, b: pd.Series) -> dict:
    d = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    n = len(d)
    if n < 10 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return {"rho": None, "p": None, "n": n, "ci": [None, None]}
    rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    z, se = np.arctanh(np.clip(rho, -0.999, 0.999)), 1 / np.sqrt(max(n - 3, 1))
    return {"rho": round(float(rho), 3), "p": float(p), "n": n,
            "ci": [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]}


def verdict(rho: float | None) -> str:
    if rho is None:
        return "not enough answers"
    if abs(rho) >= STRONG:
        return "repeats the model question - keep one"
    if abs(rho) >= MODERATE:
        return "partly overlaps"
    return "measures something different"


def multi_shares(series: pd.Series, name: str) -> list[dict]:
    answered = series.notna()
    n = int(answered.sum())
    out = []
    for key, label in LABELS[name].items():
        k = int(has(series[answered], key).sum()) if n else 0
        lo, hi = wilson(k, n)
        out.append({"key": key, "label": label, "k": k, "n": n, "pct": round(100 * k / n, 1) if n else None,
                    "ci": [round(100 * lo, 1), round(100 * hi, 1)] if n else [None, None]})
    return sorted(out, key=lambda r: -(r["pct"] or 0))


def likert_summary(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "sd": round(float(x.std(ddof=1)), 2),
            "median": float(x.median()), "pct_agree": round(100 * float((x >= 4).mean()), 1),
            "dist": {str(v): int((x == v).sum()) for v in range(1, 6)}}


def cat_shares(series: pd.Series, name: str | None = None) -> list[dict]:
    x = series.dropna().astype(str)
    n = len(x)
    keys = list(LABELS[name]) if name in LABELS else sorted(x.unique())
    out = []
    for key in keys:
        k = int((x == key).sum())
        lo, hi = wilson(k, n)
        out.append({"key": key, "label": LABELS.get(name, {}).get(key, key), "k": k, "n": n,
                    "pct": round(100 * k / n, 1) if n else None,
                    "ci": [round(100 * lo, 1), round(100 * hi, 1)] if n else [None, None]})
    return out


def mann_whitney(a: pd.Series, b: pd.Series) -> dict:
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return {"n_a": len(a), "n_b": len(b), "p": None}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n_a": len(a), "n_b": len(b), "mean_a": round(float(a.mean()), 2), "mean_b": round(float(b.mean()), 2),
            "median_a": float(a.median()), "median_b": float(b.median()), "U": float(u), "p": float(p),
            "rank_biserial": round(float(1 - 2 * u / (len(a) * len(b))), 3)}


def fisher_rows(pairs, grads: pd.DataFrame, emps: pd.DataFrame) -> list[dict]:
    """pairs: (label, graduate series of 0/1, employer series of 0/1)."""
    rows = []
    for label, g, e in pairs:
        g, e = g.dropna(), e.dropna()
        kg, ke = int(g.sum()), int(e.sum())
        if len(g) == 0 or len(e) == 0:
            continue
        _, p = stats.fisher_exact([[kg, len(g) - kg], [ke, len(e) - ke]])
        rows.append({"label": label, "graduates_pct": round(100 * kg / len(g), 1), "n_graduates": len(g),
                     "employers_pct": round(100 * ke / len(e), 1), "n_employers": len(e), "p": float(p)})
    q = bh(np.array([r["p"] for r in rows])) if rows else []
    for r, qv in zip(rows, q):
        r["q"] = float(qv)
        r["significant"] = bool(qv < ALPHA)
    return rows


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def repeated_questions(q: pd.DataFrame) -> list[dict]:
    """Each item that re-asks a model question, tested against the model question."""
    checks = [
        ("Q25 Java ticked", "languages", has(q["languages"], "java"), "Q18 Java rating", q["tech_java"]),
        ("Q25 Python ticked", "languages", has(q["languages"], "python"), "Q18 Python rating", q["tech_python"]),
        ("Q25 JavaScript ticked", "languages", has(q["languages"], "javascript"), "Q18 JavaScript rating",
         q["tech_javascript"]),
        ("Q27 Unfamiliar-problem confidence", "problem_confidence", q["problem_confidence"],
         "Q19 Problem Solving", q["soft_problem_solving"]),
        ("Q32 Explaining to non-technical people", "explain_nontechnical", q["explain_nontechnical"],
         "Q19 Communication", q["soft_communication"]),
        ("Q33 Team confidence", "team_confidence", q["team_confidence"], "Q19 Teamwork", q["soft_teamwork"]),
        ("Q34 Group projects (yes)", "group_projects", q["group_projects"].map({"yes": 1.0, "no": 0.0}),
         "Q33 Team confidence", q["team_confidence"]),
        ("Q28 Version control used", "dev_practices", has(q["dev_practices"], "version_control"),
         "Q18 Git / GitHub rating", q["tech_git"]),
        ("Q28 Testing practices used", "dev_practices", has(q["dev_practices"], "testing"),
         "Q18 Software Testing rating", q["tech_software_testing"]),
        ("Q28 CI/CD used", "dev_practices", has(q["dev_practices"], "cicd"), "Q18 DevOps / CI-CD rating",
         q["tech_devops_cicd"]),
        ("Q28 Number of practices used", "dev_practices", n_ticked(q["dev_practices"]),
         "Q29 SDLC familiarity", q["sdlc_familiarity"]),
    ]
    out = []
    for a_label, field, a, b_label, b in checks:
        r = spearman(a, b)
        out.append({"repeat": a_label, "item": ITEM[field], "model_question": b_label, **r,
                    "verdict": verdict(r["rho"]), "type": "correlation"})
    return out


def logic_checks(q: pd.DataFrame) -> list[dict]:
    """Yes/no pairs where one answer rules out the other (contradiction rate)."""
    rules = [
        ("Q11 internship = yes but Q12 = 0 months", "Q11 / Q12",
         q["internship_completed"].eq("yes") & q["internship_months"].eq(0), q["internship_completed"].notna()),
        ("Q14 no portfolio but Q15 updates it occasionally / frequently", "Q14 / Q15",
         q["has_portfolio"].eq("no") & q["portfolio_update_freq"].isin(["occasionally", "frequently"]),
         q["has_portfolio"].notna() & q["portfolio_update_freq"].notna()),
        ("Q38 no LinkedIn profile but Q36 searched on LinkedIn", "Q36 / Q38",
         q["linkedin_profile"].eq("no") & has(q["job_platforms"], "linkedin").eq(1),
         q["linkedin_profile"].notna() & q["job_platforms"].notna()),
        ("Q34 no group projects but Q34a gives a number", "Q34 / Q34a",
         q["group_projects"].eq("no") & q["group_project_count"].notna(), q["group_projects"].notna()),
    ]
    out = []
    for label, items, bad, base in rules:
        n, k = int(base.sum()), int((bad & base).sum())
        lo, hi = wilson(k, n)
        out.append({"check": label, "items": items, "contradictions": k, "n": n,
                    "pct": round(100 * k / n, 1) if n else None, "ci": [round(100 * lo, 1), round(100 * hi, 1)],
                    "type": "logic"})
    return out


def graduates_vs_employers(q: pd.DataFrame, emp: pd.DataFrame) -> dict:
    likert = [
        ("Graduates are prepared for industry", "Q42", "university_prepared", "Q14", "grad_alignment"),
        ("Importance of explanations from an AI system", "Q44", "explanation_importance", "Q17",
         "emp_explanation_importance"),
        ("Would use / find useful an AI employability system", "Q46", "would_use_system", "Q16", "ai_usefulness"),
        ("English: graduates' proficiency vs employers' importance", "Q7b", "english_proficiency", "Q9a",
         "emp_english_importance"),
    ]
    rows = [{"question": lab, "graduate_item": gi, "employer_item": ei, **mann_whitney(q[g], emp[e])}
            for lab, gi, g, ei, e in likert]
    ps = [r["p"] for r in rows if r["p"] is not None]
    for r, qv in zip([r for r in rows if r["p"] is not None], bh(np.array(ps)) if ps else []):
        r["q"], r["significant"] = float(qv), bool(qv < ALPHA)

    gaps = [  # graduate Q30 'needs improvement' vs employer Q12/Q13 'commonly weak'
        ("Programming", "programming", "tech_weaknesses", "programming"),
        ("Database", "database", "tech_weaknesses", "database"),
        ("Cloud", "cloud", "tech_weaknesses", "cloud"),
        ("Cybersecurity", "cybersecurity", "tech_weaknesses", "cybersecurity"),
        ("DevOps", "devops", "tech_weaknesses", "devops"),
        ("AI / ML", "ai_ml", "tech_weaknesses", "ai_ml"),
        ("Communication", "communication", "soft_weaknesses", "communication"),
        ("Problem solving", "problem_solving", "soft_weaknesses", "problem_solving"),
    ]
    gap_rows = fisher_rows([(lab, has(q["improvement_areas"], g), has(emp[ef], e)) for lab, g, ef, e in gaps], q, emp)

    top3 = lambda k: emp[f"rank_{k}"].notna().astype(float).where(  # noqa: E731
        emp[[f"rank_{f}" for f, _ in survey_items.EMPLOYER_RANK_FACTORS]].notna().any(axis=1))
    factors = [  # graduate Q41 'matters most' vs employer Q20 'in top 3'
        ("Technical skills", "technical_skills", "technical_depth"), ("Internship experience", "internship", "internship"),
        ("Projects / portfolio", "projects", "portfolio"), ("Certifications", "certifications", "certifications"),
        ("Communication", "communication", "communication"), ("Interview preparation", "interview_prep", "interview"),
    ]
    factor_rows = fisher_rows([(lab, has(q["believed_factors"], g), top3(e)) for lab, g, e in factors], q, emp)
    return {"likert": rows, "skill_gaps": gap_rows, "what_matters": factor_rows}


def self_perception(q: pd.DataFrame) -> dict:
    g = q[q["respondent_status"] == "graduated"].dropna(subset=["self_employability", "duration_months", "event"])
    out = {"n": len(g)}
    if len(g) >= 20 and g["event"].sum() > 0:
        # higher self-rating should mean a shorter time -> negate for lifelines (higher score = longer survival)
        out["harrell_c"] = round(float(concordance_index(g["duration_months"], -g["self_employability"], g["event"])), 3)
    groups = {"Found an IT job": q[(q["respondent_status"] == "graduated") & (q["event"] == 1)],
              "Still searching": q[(q["respondent_status"] == "graduated") & (q["event"] == 0)],
              "Final-year": q[q["respondent_status"] == "final_year"]}
    out["by_group"] = {k: likert_summary(v["self_employability"]) for k, v in groups.items()}
    a, b = groups["Found an IT job"]["self_employability"], groups["Still searching"]["self_employability"]
    out["employed_vs_searching"] = mann_whitney(a, b)
    return out


def artifact_requirements(q: pd.DataFrame, emp: pd.DataFrame) -> dict:
    return {
        "explanation_importance": likert_summary(q["explanation_importance"]),
        "explanation_types": multi_shares(q["explanation_types"], "explanation_types"),
        "would_use_system": likert_summary(q["would_use_system"]),
        "employer_usefulness": likert_summary(emp["ai_usefulness"]),
        "employer_explanation_importance": likert_summary(emp["emp_explanation_importance"]),
        "employer_outputs": multi_shares(emp["ai_outputs"], "ai_outputs"),
    }


def describe_all(df: pd.DataFrame, items) -> list[dict]:
    out = []
    for name, item, kind, _, desc in items:
        if name not in df or kind == "text":
            if kind == "text" and name in df:
                out.append({"field": name, "item": item, "kind": kind, "description": desc,
                            "answered": int(df[name].notna().sum())})
            continue
        if kind == "rank":
            continue
        row = {"field": name, "item": item, "kind": kind, "description": desc, "answered": int(df[name].notna().sum())}
        if kind == "likert":
            row["summary"] = likert_summary(df[name])
        elif kind == "multi":
            row["shares"] = multi_shares(df[name], name)
        else:
            row["shares"] = cat_shares(df[name].map({"yes": "yes", "no": "no"}) if kind == "binary" else df[name],
                                       name if kind != "binary" else None)
        out.append(row)
    return out


def employer_ranking(emp: pd.DataFrame) -> list[dict]:
    cols = [f"rank_{k}" for k, _ in survey_items.EMPLOYER_RANK_FACTORS]
    answered = emp[cols].notna().any(axis=1)
    n = int(answered.sum())
    rows = []
    for k, lab in survey_items.EMPLOYER_RANK_FACTORS:
        r = emp.loc[answered, f"rank_{k}"]
        lo, hi = wilson(int(r.notna().sum()), n)
        rows.append({"factor": lab, "pct_top3": round(100 * r.notna().mean(), 1) if n else None,
                     "ci": [round(100 * lo, 1), round(100 * hi, 1)], "pct_first": round(100 * (r == 1).mean(), 1) if n else None,
                     "mean_rank": round(float(r.mean()), 2) if r.notna().any() else None, "n": n})
    return sorted(rows, key=lambda r: -(r["pct_top3"] or 0))


def employer_factors(emp: pd.DataFrame) -> list[dict]:
    """E7: importance (1-5) of each hiring factor - the employer view of the graduate model's inputs."""
    rows = []
    for f in schema.EMPLOYER_FACTORS:
        x = pd.to_numeric(emp.get(f"factor_{f}"), errors="coerce").dropna()
        if x.empty:
            continue
        se = x.std(ddof=1) / np.sqrt(len(x))
        rows.append({"factor": f, "label": FACTOR_LABELS[f], "mean": round(float(x.mean()), 2),
                     "ci": [round(float(x.mean() - 1.96 * se), 2), round(float(x.mean() + 1.96 * se), 2)],
                     "pct_4_5": round(100 * float((x >= 4).mean()), 1), "n": int(len(x))})
    return sorted(rows, key=lambda r: -r["mean"])


FACTOR_LABELS = {"degree_qualification": "Degree qualification", "gpa": "GPA / academic performance",
                 "technical_skills": "Technical skills", "internship_experience": "Internship experience",
                 "personal_projects": "Personal projects", "certifications": "Certifications",
                 "portfolio": "GitHub / portfolio", "communication": "Communication skills",
                 "interview_performance": "Interview performance"}

FAST_MONTHS = 6
# Preparation before degree completion (asked retrospectively): (key, label, 0/1 indicator from the answers)
BEHAVIOURS = [
    ("has_portfolio", "Had a portfolio (GitHub / website)", lambda q: q["has_portfolio"].map({"yes": 1.0, "no": 0.0})),
    ("hackathons", "Took part in a hackathon or competition",
     lambda q: (pd.to_numeric(q["hackathon_count"], errors="coerce") > 0).astype(float).where(q["hackathon_count"].notna())),
    ("open_source", "Contributed to open source", lambda q: q["open_source"].map({"yes": 1.0, "no": 0.0})),
    ("self_learning", "Self-learned often during the degree (4-5)",
     lambda q: (pd.to_numeric(q["self_learning_freq"], errors="coerce") >= 4).astype(float).where(q["self_learning_freq"].notna())),
    ("early_prep", "Started preparing by 2nd/3rd year",
     lambda q: q["prep_start"].isin(["before_university", "first_year", "second_third_year"]).astype(float)
     .where(q["prep_start"].notna())),
    ("english", "Good English at completion (Q7b 4-5)",
     lambda q: (pd.to_numeric(q["english_proficiency"], errors="coerce") >= 4).astype(float)
     .where(q["english_proficiency"].notna())),
    ("sdlc", "Knew SDLC practices well at completion (4-5)",
     lambda q: (pd.to_numeric(q["sdlc_familiarity"], errors="coerce") >= 4).astype(float).where(q["sdlc_familiarity"].notna())),
]


def fast_finders(q: pd.DataFrame, months: int = FAST_MONTHS) -> dict:
    """Preparation of graduates who found an IT job within `months` vs those who did not (only graduates
    followed up for at least `months`, so nobody is counted as 'slow' just because they were surveyed early).
    Descriptive and associational: these items are not model inputs (see feature_set_comparison.md)."""
    g = q[q["respondent_status"] == "graduated"].copy()
    fast = (g["event"] == 1) & (g["duration_months"] <= months)
    known = fast | (g["duration_months"] >= months)
    g, fast = g[known], fast[known]
    rows = []
    for key, label, fn in BEHAVIOURS:
        x = fn(g)
        a, b = x[fast].dropna(), x[~fast].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        _, p = stats.fisher_exact([[int(a.sum()), len(a) - int(a.sum())], [int(b.sum()), len(b) - int(b.sum())]])
        rows.append({"key": key, "label": label, "pct_fast": round(100 * a.mean(), 1), "n_fast": len(a),
                     "pct_other": round(100 * b.mean(), 1), "n_other": len(b), "p": float(p)})
    for r, qv in zip(rows, bh(np.array([r["p"] for r in rows])) if rows else []):
        r["q"], r["significant"] = float(qv), bool(qv < ALPHA)
    return {"months": months, "n_fast": int(fast.sum()), "n_other": int((~fast).sum()), "rows": rows}


def graduate_experience(q: pd.DataFrame) -> dict:
    """Where graduates who found IT work searched, where graduates learn, and the challenges they report."""
    g = q[q["respondent_status"] == "graduated"]
    employed = g[g["event"] == 1]
    return {
        "search_platforms_employed": multi_shares(employed["job_platforms"], "job_platforms"),
        "learning_platforms": multi_shares(q["learning_platforms"], "learning_platforms"),
        "learning_strategies": multi_shares(q["learning_strategies"], "learning_strategies"),
        "cert_areas": multi_shares(q["cert_areas"], "cert_areas"),
        "challenges": cat_shares(g["biggest_challenge"], "biggest_challenge"),
        "lacked_skills": multi_shares(g["lacked_skills"], "lacked_skills"),
        "feedback_areas": multi_shares(g["feedback_areas"], "feedback_areas"),
        "project_areas": [r for r in multi_shares(q["project_areas"], "project_areas") if r["key"] != "none"],
    }


def employer_views(emp: pd.DataFrame) -> dict:
    return {"future_tech": multi_shares(emp["future_tech"], "future_tech"),
            "tech_weaknesses": multi_shares(emp["tech_weaknesses"], "tech_weaknesses"),
            "soft_weaknesses": multi_shares(emp["soft_weaknesses"], "soft_weaknesses"),
            "uni_improvements": multi_shares(emp["uni_improvements"], "uni_improvements"),
            "hiring_challenges": multi_shares(emp["hiring_challenges"], "hiring_challenges"),
            "roles_recruited": multi_shares(emp["roles_recruited"], "roles_recruited"),
            "skill_update_freq": cat_shares(emp["skill_update_freq"], "skill_update_freq")}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def load(processed: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    q = pd.read_csv(processed / "questionnaire_graduates.csv")
    emp = pd.read_csv(processed / "employers.csv")
    info_path = processed / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return q, emp, info


def analyse(q: pd.DataFrame, emp: pd.DataFrame, info: dict | None = None) -> dict:
    info = info or {}
    for f in schema.SURVEY_FIELDS:
        if f.name not in q:
            q[f.name] = np.nan
    for f in schema.EMPLOYER_SURVEY_FIELDS:
        if f.name not in emp:
            emp[f.name] = np.nan
    graduate_answered = int(q[[f.name for f in schema.SURVEY_FIELDS]].notna().any(axis=1).sum())
    return {
        "is_synthetic": bool(info.get("is_synthetic", True)),
        "n_graduates": int((q["respondent_status"] == "graduated").sum()),
        "n_final_year": int((q["respondent_status"] == "final_year").sum()),
        "n_with_answers": graduate_answered,
        "n_employers": len(emp),
        "repeated": repeated_questions(q),
        "logic": logic_checks(q),
        "graduates_vs_employers": graduates_vs_employers(q, emp),
        "self_perception": self_perception(q),
        "artifact": artifact_requirements(q, emp),
        "employer_ranking": employer_ranking(emp),
        "employer_factors": employer_factors(emp),
        "fast_finders": fast_finders(q),
        "graduate_experience": graduate_experience(q),
        "employer_views": employer_views(emp),
        "graduate_items": describe_all(q, survey_items.GRADUATE_ITEMS),
        "employer_items": describe_all(emp, survey_items.EMPLOYER_ITEMS),
    }


def _fmt_p(p) -> str:
    return "-" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def to_markdown(r: dict) -> str:
    L = ["# Questionnaire study", "",
         "_Generated by `python -m core.survey_study` - do not edit by hand._", ""]
    if r["is_synthetic"]:
        L += ["> **SYNTHETIC DATA - DEMO ONLY.** Every number below comes from generated responses "
              "(`core/synthetic_survey.py`) and shows how the analysis works, not real findings.", ""]
    L += [f"Graduates: {r['n_graduates']} (+ {r['n_final_year']} final-year). Employers: {r['n_employers']}.", "",
          "## 1. Repeated questions", "",
          f"Each item that asks about a model question again is correlated with it (Spearman). "
          f"|rho| >= {STRONG}: the items measure the same thing, so one can be dropped; "
          f"{MODERATE}-{STRONG}: they partly overlap; below {MODERATE}: they measure something different.", "",
          "| Repeat item | Model question | rho (95% CI) | n | Verdict |", "|---|---|---|---|---|"]
    for c in r["repeated"]:
        ci = f"{c['rho']} ({c['ci'][0]} to {c['ci'][1]})" if c["rho"] is not None else "-"
        L.append(f"| {c['repeat']} | {c['model_question']} | {ci} | {c['n']} | {c['verdict']} |")
    L += ["", "**Logic checks** (answers that contradict each other):", "",
          "| Check | Contradictions | % (95% CI) |", "|---|---|---|"]
    for c in r["logic"]:
        L.append(f"| {c['check']} | {c['contradictions']} / {c['n']} | {c['pct']} ({c['ci'][0]}-{c['ci'][1]}) |")

    gve = r["graduates_vs_employers"]
    L += ["", "## 2. Graduates vs employers", "", "| Question | Graduates mean | Employers mean | Rank-biserial | p | q (BH) |",
          "|---|---|---|---|---|---|"]
    for x in gve["likert"]:
        L.append(f"| {x['question']} ({x['graduate_item']} vs E{x['employer_item'][1:]}) | {x.get('mean_a', '-')} | "
                 f"{x.get('mean_b', '-')} | {x.get('rank_biserial', '-')} | {_fmt_p(x['p'])} | {_fmt_p(x.get('q'))} |")
    for title, key, a, b in (("Skill gaps: graduates who say they need to improve (Q30) vs employers who see it as a "
                              "common weakness (E12/E13)", "skill_gaps", "Graduates %", "Employers %"),
                             ("What matters most: graduates who chose it (Q41) vs employers who ranked it in their "
                              "top 3 (E20)", "what_matters", "Graduates %", "Employers %")):
        L += ["", f"**{title}**", "", f"| Area | {a} | {b} | p | q (BH) |", "|---|---|---|---|---|"]
        L += [f"| {x['label']} | {x['graduates_pct']} | {x['employers_pct']} | {_fmt_p(x['p'])} | "
              f"{_fmt_p(x['q'])}{' *' if x['significant'] else ''} |" for x in gve[key]]

    sp = r["self_perception"]
    L += ["", "## 3. Self-perceived employability (Q40) vs the actual outcome", "",
          f"Harrell's C of the Q40 rating alone against time to first IT job: **{sp.get('harrell_c', '-')}** "
          f"(n = {sp['n']}). Q40 is asked at the survey, after the outcome, so it partly reflects whether the "
          "graduate already has a job; it is reported, never used as a predictor.", "",
          "| Group | n | Mean | % rating 4-5 |", "|---|---|---|---|"]
    L += [f"| {k} | {v.get('n', 0)} | {v.get('mean', '-')} | {v.get('pct_agree', '-')} |" for k, v in sp["by_group"].items()]

    ar = r["artifact"]
    L += ["", "## 4. Requirements for the artifact", "",
          f"- Graduates rating explanations important (Q44 = 4-5): {ar['explanation_importance'].get('pct_agree')}% "
          f"(mean {ar['explanation_importance'].get('mean')}); employers (E17): "
          f"{ar['employer_explanation_importance'].get('pct_agree')}%.",
          f"- Graduates who would use the system (Q46 = 4-5): {ar['would_use_system'].get('pct_agree')}%; "
          f"employers who find it useful (E16 = 4-5): {ar['employer_usefulness'].get('pct_agree')}%.", "",
          "| Explanation wanted by graduates (Q45) | % | Output wanted by employers (E18) | % |", "|---|---|---|---|"]
    for a, b in zip(ar["explanation_types"] + [None] * 6, ar["employer_outputs"]):
        L.append(f"| {a['label'] if a else ''} | {a['pct'] if a else ''} | {b['label']} | {b['pct']} |")

    L += ["", "**Employer ranking of what makes a graduate more employable (E20)**", "",
          "| Factor | % ranked in top 3 (95% CI) | % ranked first | Mean rank |", "|---|---|---|---|"]
    L += [f"| {x['factor']} | {x['pct_top3']} ({x['ci'][0]}-{x['ci'][1]}) | {x['pct_first']} | {x['mean_rank']} |"
          for x in r["employer_ranking"]]

    L += ["", "**How important employers rate each hiring factor (E7, 1-5)**", "",
          "| Factor | Mean (95% CI) | % rating 4-5 | n |", "|---|---|---|---|"]
    L += [f"| {x['label']} | {x['mean']} ({x['ci'][0]}-{x['ci'][1]}) | {x['pct_4_5']} | {x['n']} |"
          for x in r["employer_factors"]]

    ff = r["fast_finders"]
    L += ["", f"## 5. What graduates who found an IT job within {ff['months']} months did", "",
          f"{ff['n_fast']} graduates found an IT job within {ff['months']} months; {ff['n_other']} did not (only "
          f"graduates followed up for at least {ff['months']} months). Associational only - these preparation items "
          "are not model inputs.", "",
          "| Preparation before completion | Found job within 6 months % | Others % | q (BH) |", "|---|---|---|---|"]
    L += [f"| {x['label']} | {x['pct_fast']} | {x['pct_other']} | {_fmt_p(x['q'])}{' *' if x['significant'] else ''} |"
          for x in ff["rows"]]
    ge = r["graduate_experience"]
    L += ["", "- Where employed graduates searched (Q36): " + "; ".join(
              f"{s['label']} {s['pct']}%" for s in ge["search_platforms_employed"][:5]),
          "- Learning platforms (Q22): " + "; ".join(f"{s['label']} {s['pct']}%" for s in ge["learning_platforms"][:5]),
          "- Biggest challenge (Q43): " + "; ".join(f"{s['label']} {s['pct']}%" for s in
                                                     sorted(ge["challenges"], key=lambda s: -(s["pct"] or 0))[:4]), ""]

    for title, rows in (("6. Every graduate item", r["graduate_items"]), ("7. Every employer item", r["employer_items"])):
        L += ["", f"## {title}", ""]
        for x in rows:
            L.append(f"**{x['item']} {x['description']}** (answered: {x['answered']})")
            if "summary" in x:
                s = x["summary"]
                L.append(f"Mean {s.get('mean')} (SD {s.get('sd')}), median {s.get('median')}, "
                         f"{s.get('pct_agree')}% rated 4-5.")
            elif "shares" in x:
                L.append("; ".join(f"{s['label']} {s['pct']}%" for s in x["shares"] if s["pct"] is not None))
            L.append("")
    return "\n".join(L)


def run(processed: Path = PROCESSED, out_dir: Path = ARTIFACTS) -> dict:
    q, emp, info = load(processed)
    result = analyse(q, emp, info)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questionnaire_study.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    (out_dir / "questionnaire_study.md").write_text(to_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = run()
    for c in res["repeated"]:
        print(f"{c['repeat']:40s} vs {c['model_question']:28s} rho={c['rho']}  -> {c['verdict']}")
