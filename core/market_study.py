"""Job-market study on the REAL job ads (data/raw/job_ads.csv, collected Jul-Oct 2026).

Descriptive analysis of what Sri Lankan employers ask of entry-level IT applicants:
  - prevalence of each skill in the vocabulary, with Wilson 95% intervals
  - demand by role family (classified from the job title by fixed rules)
  - skill co-occurrence: lift and phi for every pair seen together in >= MIN_PAIR_ADS ads
  - interns vs junior / associate roles: two-proportion tests with Benjamini-Hochberg correction
  - technical vs soft-skill emphasis per ad
  - named technologies (data/raw/job_ads_technologies.csv, rules in data/reference/job_ad_technology_rules.json)

Writes artifacts/job_market_study.json (read by the API) and artifacts/job_market_study.md.
Run:  python -m core.market_study
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core import schema

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "job_ads.csv"
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
MIN_PAIR_ADS = 10
ALPHA = 0.05

# Checked in order; the first match wins. "Software engineering" is the fallback for developer titles.
ROLE_RULES = [
    ("Quality assurance", r"\bqa\b|quality|test"),
    ("UI/UX design", r"\bui\b|\bux\b|ui/ux|design"),
    ("Cybersecurity", r"cyber|security|\bsoc\b"),
    ("Data & AI", r"\bdata\b|\bai\b|\bml\b|machine learning|analytic|\bbi\b|\bmis\b|intelligence"),
    ("Business analysis", r"business analy|analayst|\bba\b|product owner|business consultant|implementation|consultant|erp"),
    ("Infrastructure & support", r"devops|cloud|system|network|infrastructure|server|support|help ?desk|\bit\b|operations|"
                                 r"observability|platform|technician|edi"),
    ("Software engineering", r"software|developer|engineer|full.?stack|front.?end|back.?end|programmer|mern|python|flutter|"
                             r"mobile|web|game|intern"),
]
OTHER_ROLE = "Other"


def role_family(title: str) -> str:
    t = str(title).lower()
    for name, pattern in ROLE_RULES:
        if re.search(pattern, t):
            return name
    return OTHER_ROLE


def seniority(title: str) -> str:
    t = str(title).lower()
    return "Intern / trainee" if re.search(r"intern|trainee|undergraduate|graduate placement", t) else "Junior / associate"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def load(processed: Path = PROCESSED, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ads = pd.read_csv(processed)
    tech_path = raw_dir / "job_ads_technologies.csv"
    if tech_path.exists():
        ads = ads.merge(pd.read_csv(tech_path)[["ad_id", "technologies"]], on="ad_id", how="left")
    else:
        ads["technologies"] = ""
    ads["technologies"] = ads["technologies"].fillna("")
    ads["role"] = ads["title"].map(role_family)
    ads["seniority"] = ads["title"].map(seniority)
    for s in schema.SKILL_VOCABULARY:
        ads[s.key] = ads["skills"].fillna("").str.split(";").map(lambda ks, k=s.key: k in ks).astype(int)
    return ads


def analyse(ads: pd.DataFrame) -> dict:
    n = len(ads)
    keys = [s.key for s in schema.SKILL_VOCABULARY]
    label = {s.key: s.label for s in schema.SKILL_VOCABULARY}
    kind = {s.key: s.kind for s in schema.SKILL_VOCABULARY}

    prevalence = []
    for k in keys:
        c = int(ads[k].sum())
        lo, hi = wilson(c, n)
        prevalence.append({"key": k, "label": label[k], "kind": kind[k], "ads": c, "pct": 100 * c / n,
                           "ci_low": 100 * lo, "ci_high": 100 * hi})
    prevalence.sort(key=lambda r: -r["pct"])

    roles = ads["role"].value_counts()
    by_role = {r: {k: 100 * ads.loc[ads.role == r, k].mean() for k in keys} for r in roles.index}

    pairs = []
    for a, b in itertools.combinations(keys, 2):
        both = int((ads[a] & ads[b]).sum())
        if both < MIN_PAIR_ADS:
            continue
        pa, pb = ads[a].mean(), ads[b].mean()
        lift = (both / n) / (pa * pb)
        phi = np.corrcoef(ads[a], ads[b])[0, 1]
        pairs.append({"a": label[a], "b": label[b], "ads": both, "lift": lift, "phi": float(phi)})
    pairs.sort(key=lambda r: -r["lift"])

    groups = ads["seniority"].value_counts().to_dict()
    tests = []
    g1, g2 = ads[ads.seniority == "Intern / trainee"], ads[ads.seniority == "Junior / associate"]
    for k in keys:
        x1, x2 = int(g1[k].sum()), int(g2[k].sum())
        table = [[x1, len(g1) - x1], [x2, len(g2) - x2]]
        p = stats.fisher_exact(table)[1] if min(len(g1), len(g2)) else 1.0
        tests.append({"key": k, "label": label[k], "intern_pct": 100 * x1 / max(len(g1), 1),
                      "junior_pct": 100 * x2 / max(len(g2), 1), "p": p})
    adj = bh([t["p"] for t in tests])
    for t, q in zip(tests, adj):
        t["q"] = float(q)
        t["significant"] = bool(q < ALPHA)
    tests.sort(key=lambda t: t["q"])

    tech_keys = [k for k in keys if kind[k] == "technical"]
    soft_keys = [k for k in keys if kind[k] == "soft"]
    per_ad_tech, per_ad_soft = ads[tech_keys].sum(axis=1), ads[soft_keys].sum(axis=1)
    emphasis = {
        "mean_technical": float(per_ad_tech.mean()), "mean_soft": float(per_ad_soft.mean()),
        "pct_any_soft": 100 * float((per_ad_soft > 0).mean()), "pct_any_technical": 100 * float((per_ad_tech > 0).mean()),
        "pct_soft_only": 100 * float(((per_ad_soft > 0) & (per_ad_tech == 0)).mean()),
        "spearman_tech_soft": float(stats.spearmanr(per_ad_tech, per_ad_soft)[0]),
    }

    techs = ads["technologies"].str.split("; ").explode()
    techs = techs[techs.str.len() > 0].value_counts()
    technologies = [{"name": t, "ads": int(c), "pct": 100 * c / n} for t, c in techs.items()]

    dates = pd.to_datetime(ads["date"], errors="coerce")
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    return {
        "n_ads": n, "n_companies": int(ads["company"].nunique()),
        "date_from": str(dates.min().date()), "date_to": str(dates.max().date()),
        "sources": ads["source"].value_counts().to_dict(), "roles": roles.to_dict(), "seniority": groups,
        "monthly": {str(k): int(v) for k, v in monthly.items()},
        "prevalence": prevalence, "by_role": by_role, "pairs": pairs[:40], "seniority_tests": tests,
        "emphasis": emphasis, "technologies": technologies[:30],
        "notes": [
            "Real ads from topjobs.lk, ITPro.lk and LinkedIn; skills coded by hand (topjobs flyers) or by fixed keyword "
            "rules (others). A skill counts as 'asked for' when the ad mentions it.",
            "Keyword coding also counts terms in the company's own description, so AI/ML, analytics and cloud are slightly "
            "over-counted.",
            "Role family and intern vs junior/associate are classified from the job title by fixed rules (core/market_study.py).",
            f"Intern vs junior/associate differences use Fisher's exact test with Benjamini-Hochberg correction (q < {ALPHA}).",
        ],
    }


def to_markdown(r: dict) -> str:
    L = ["# Job-market study (real job ads)", "",
         "_Generated by `python -m core.market_study` - do not edit by hand._", "",
         f"**{r['n_ads']} real entry-level IT job ads** from **{r['n_companies']} companies**, posted "
         f"{r['date_from']} to {r['date_to']}. Sources: " + ", ".join(f"{k} {v}" for k, v in r["sources"].items()) + ".", "",
         "## Skill demand (share of ads, Wilson 95% CI)", "", "| Skill | Type | Ads | % | 95% CI |", "|---|---|---|---|---|"]
    L += [f"| {p['label']} | {p['kind']} | {p['ads']} | {p['pct']:.0f}% | {p['ci_low']:.0f}-{p['ci_high']:.0f}% |"
          for p in r["prevalence"]]
    e = r["emphasis"]
    L += ["", "## Technical vs soft skills", "",
          f"- An ad names on average **{e['mean_technical']:.1f} technical** and **{e['mean_soft']:.1f} soft** skills.",
          f"- {e['pct_any_soft']:.0f}% of ads ask for at least one soft skill; {e['pct_any_technical']:.0f}% for at least one "
          f"technical skill; {e['pct_soft_only']:.0f}% ask only for soft skills.",
          f"- Ads that list more technical skills also list more soft skills (Spearman rho = {e['spearman_tech_soft']:.2f}).",
          "", "## Demand by role family (% of ads in the role)", ""]
    roles = list(r["roles"])
    top = [p["key"] for p in r["prevalence"][:10]]
    lab = {p["key"]: p["label"] for p in r["prevalence"]}
    L += ["| Skill | " + " | ".join(f"{x} (n={r['roles'][x]})" for x in roles) + " |", "|---" * (len(roles) + 1) + "|"]
    L += [f"| {lab[k]} | " + " | ".join(f"{r['by_role'][x][k]:.0f}%" for x in roles) + " |" for k in top]
    L += ["", f"## Skills that go together (pairs in >= {MIN_PAIR_ADS} ads, by lift)", "",
          "| Skill A | Skill B | Ads with both | Lift | Phi |", "|---|---|---|---|---|"]
    L += [f"| {p['a']} | {p['b']} | {p['ads']} | {p['lift']:.2f} | {p['phi']:.2f} |" for p in r["pairs"][:15]]
    s = r["seniority"]
    L += ["", "## Interns / trainees vs junior / associate roles", "",
          f"Interns/trainees n = {s.get('Intern / trainee', 0)}, junior/associate n = {s.get('Junior / associate', 0)}.", "",
          "| Skill | Interns | Junior/associate | q (BH) |", "|---|---|---|---|"]
    L += [f"| {t['label']} | {t['intern_pct']:.0f}% | {t['junior_pct']:.0f}% | {t['q']:.3f}{' *' if t['significant'] else ''} |"
          for t in r["seniority_tests"][:10]]
    L += ["", "## Most-named technologies", "", "| Technology | Ads | % |", "|---|---|---|"]
    L += [f"| {t['name']} | {t['ads']} | {t['pct']:.0f}% |" for t in r["technologies"][:20]]
    L += ["", "## Notes", ""] + [f"- {x}" for x in r["notes"]]
    return "\n".join(L) + "\n"


def run(processed: Path = PROCESSED, raw_dir: Path = RAW_DIR, out_dir: Path = ARTIFACTS) -> dict:
    result = analyse(load(processed, raw_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job_market_study.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "job_market_study.md").write_text(to_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print(f"{r['n_ads']} ads, {r['n_companies']} companies; roles {r['roles']}")
