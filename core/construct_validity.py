"""Construct validity, reliability and predictor checks behind thesis Appendix E (REPORT ONLY).

Moved here from the private evidence pack so the numbers in Tables E.1-E.4 can be reproduced from the public code.
Same formulas and data as before; the TRAINING set only, except Table E.1, which describes every split.
  - Table E.1  sample characteristics by split
  - Table E.2  each predictor and the outcome (univariate Cox per SD) and variance inflation factors
  - Table E.3  suitability (KMO, Bartlett), factors by parallel analysis, Cronbach's alpha (95% Feldt CI),
               McDonald's omega, average variance extracted (AVE), loading range; HTMT and Fornell-Larcker
  - Table E.4  item loadings on one factor per block (principal-axis factoring)
Pearson correlations of the 1-5 ratings are used (polychoric correlations would be slightly higher).

Writes artifacts/construct_validity.{json,md}.
Run:  python -m core.construct_validity
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy import stats

from core import eda_figures, schema, splits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PREP_ORDER = {"not_started": 0, "after_graduation": 1, "final_year": 2, "second_third_year": 3, "first_year": 4,
              "before_university": 5}
PREDICTORS = {"gpa": "GPA", "internship_months": "Internship months", "certifications_count": "Certifications",
              "project_count": "Projects", "tech_score": "Technical skill score", "soft_score": "Soft-skill score",
              "portfolio": "Portfolio", "prep_start_n": "Years of preparation",
              "self_learning_freq": "Self-learning frequency", "hackathon_count": "Hackathons",
              "open_src": "Open source", "sdlc_familiarity": "SDLC familiarity"}


# ---------------------------------------------------------------------------------------------- formulas
def kmo(R: np.ndarray) -> float:
    """Kaiser-Meyer-Olkin measure of sampling adequacy (overall)."""
    inv = np.linalg.inv(R)
    P = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))          # partial correlations
    np.fill_diagonal(P, 0)
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    r2, p2 = R0 ** 2, P ** 2
    return float(r2.sum() / (r2.sum() + p2.sum()))


def bartlett(R: np.ndarray, n: int) -> tuple[float, int, float]:
    """Bartlett's test of sphericity: chi-square, degrees of freedom, p."""
    p = R.shape[0]
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df = p * (p - 1) // 2
    return float(chi2), df, float(stats.chi2.sf(chi2, df))


def paf(R: np.ndarray, k: int = 1, iters: int = 200) -> np.ndarray:
    """Principal-axis factoring: communalities iterated from the squared multiple correlations."""
    h2 = 1 - 1 / np.diag(np.linalg.inv(R))
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        vals, vecs = np.linalg.eigh(Rr)
        idx = np.argsort(vals)[::-1][:k]
        L = vecs[:, idx] * np.sqrt(np.clip(vals[idx], 0, None))
        new = (L ** 2).sum(1)
        if np.max(np.abs(new - h2)) < 1e-6:
            break
        h2 = np.clip(new, 0.001, 0.999)
    return L * np.sign(L.sum(0))                                        # loadings mostly positive


def alpha(X: pd.DataFrame) -> float:
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(1).var(ddof=1)))


def alpha_ci(X: pd.DataFrame) -> tuple[float, float, float]:
    """Cronbach's alpha with the Feldt 95% confidence interval."""
    a, (n, k) = alpha(X), X.shape
    lo = 1 - (1 - a) * stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))
    hi = 1 - (1 - a) * stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1))
    return a, float(lo), float(hi)


def omega_ave(lam: np.ndarray) -> tuple[float, float]:
    """McDonald's omega (composite reliability) and average variance extracted from one-factor loadings."""
    s = lam.sum() ** 2
    return float(s / (s + (1 - lam ** 2).sum())), float((lam ** 2).mean())


def htmt(X: pd.DataFrame, a: list, b: list) -> float:
    """Heterotrait-monotrait ratio of correlations between two item blocks."""
    R = X.corr().abs()
    het = R.loc[a, b].to_numpy().mean()
    mono = lambda cols: R.loc[cols, cols].to_numpy()[np.triu_indices(len(cols), 1)].mean()  # noqa: E731
    return float(het / np.sqrt(mono(a) * mono(b)))


# ---------------------------------------------------------------------------------------------- tables
def with_scores(g: pd.DataFrame) -> pd.DataFrame:
    """Add the block scores and numeric codes used in Appendix E."""
    g = g.copy()
    g["tech_score"] = g[list(schema.TECH_COLUMNS)].mean(1)
    g["soft_score"] = g[list(schema.SOFT_COLUMNS)].mean(1)
    g["portfolio"] = (g["has_portfolio"] == "yes").astype(float)
    g["open_src"] = (g["open_source"] == "yes").astype(float)
    g["prep_start_n"] = g["prep_start"].map(PREP_ORDER)
    return g


def split_characteristics(g: pd.DataFrame) -> pd.DataFrame:
    """Table E.1: sample characteristics by split (g must have a 'split' column)."""
    def describe(d):
        km = KaplanMeierFitter().fit(d["duration_months"], d["event"])
        pct = lambda m: f"{100 * m:.1f}%"                              # noqa: E731
        q = lambda c: f"{d[c].median():.0f} ({d[c].quantile(.25):.0f}-{d[c].quantile(.75):.0f})"  # noqa: E731
        return {"n": len(d), "events": int(d["event"].sum()), "censored": pct(1 - d["event"].mean()),
                "women": pct((d["gender"] == "female").mean()),
                "state / non-state / foreign": " / ".join(pct((d["university_type"] == t).mean())
                                                          for t in ("state", "non_state", "foreign_affiliated")),
                "GPA mean (SD)": f"{d['gpa'].mean():.2f} ({d['gpa'].std():.2f})",
                "internship completed": pct((d["internship_completed"] == "yes").mean()),
                "internship months median (IQR)": q("internship_months"),
                "certifications median (IQR)": q("certifications_count"),
                "projects median (IQR)": q("project_count"),
                "technical score mean (SD)": f"{d['tech_score'].mean():.2f} ({d['tech_score'].std():.2f})",
                "soft score mean (SD)": f"{d['soft_score'].mean():.2f} ({d['soft_score'].std():.2f})",
                "Kaplan-Meier median (months)": f"{km.median_survival_time_:g}"}
    t = pd.DataFrame({s: describe(g[g["split"] == s]) for s in splits.ROLES} | {"all": describe(g)})
    t.index.name = "Characteristic"
    return t


def univariate_and_vif(train: pd.DataFrame) -> pd.DataFrame:
    """Table E.2: univariate Cox hazard ratio per SD for each predictor, and its variance inflation factor."""
    num = list(PREDICTORS)
    X = train[num].dropna()
    X = (X - X.mean()) / X.std()
    vif = pd.Series(np.diag(np.linalg.inv(np.corrcoef(X.to_numpy(), rowvar=False))), index=num)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in num:
            d = train[[c, "duration_months", "event"]].dropna().copy()
            d[c] = (d[c] - d[c].mean()) / d[c].std()
            m = CoxPHFitter().fit(d, "duration_months", "event").summary.loc[c]
            rows.append({"predictor": PREDICTORS[c], "n": len(d), "hr": float(m["exp(coef)"]),
                         "lo": float(m["exp(coef) lower 95%"]), "hi": float(m["exp(coef) upper 95%"]),
                         "p": float(m["p"]), "vif": float(vif[c])})
    return pd.DataFrame(rows).sort_values("hr", ascending=False).reset_index(drop=True)


def block_validity(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tables E.3 and E.4 and the discriminant-validity figures (complete cases of the 26 items)."""
    tech, soft = list(schema.TECH_COLUMNS), list(schema.SOFT_COLUMNS)
    X = train[tech + soft].apply(pd.to_numeric, errors="coerce").dropna()
    labels = {s.column: s.label for s in schema.TECHNICAL_SKILLS + schema.SOFT_SKILLS}
    factors = eda_figures.scree(train)                                  # parallel analysis, same seed as Table E.3
    rows, loads, aves = [], [], {}
    for name, cols, key in (("Technical", tech, "technical"), ("Soft", soft, "soft")):
        R = X[cols].corr().to_numpy()
        chi2, df, p = bartlett(R, len(X))
        lam = paf(R, 1)[:, 0]
        om, ave = omega_ave(lam)
        a, lo, hi = alpha_ci(X[cols])
        aves[name] = ave
        rows.append({"block": f"{name} ({len(cols)} items)", "n": len(X), "kmo": kmo(R),
                     "bartlett_chi2": chi2, "bartlett_df": df, "bartlett_p": p,
                     "factors_parallel_analysis": factors[key]["factors"], "alpha": a, "alpha_lo": lo, "alpha_hi": hi,
                     "omega": om, "ave": ave, "loading_min": float(lam.min()), "loading_max": float(lam.max())})
        loads.append(pd.DataFrame({"item": [labels[c] for c in cols], "block": name, "loading": lam,
                                   "communality": lam ** 2}).sort_values("loading", ascending=False))
    r_ts = float(np.corrcoef(X[tech].mean(1), X[soft].mean(1))[0, 1])
    disc = {"htmt": htmt(X, tech, soft), "sqrt_ave_technical": float(np.sqrt(aves["Technical"])),
            "sqrt_ave_soft": float(np.sqrt(aves["Soft"])), "score_correlation": r_ts}
    return pd.DataFrame(rows), pd.concat(loads, ignore_index=True), disc


# ---------------------------------------------------------------------------------------------- run
def run(out_dir: Path = ARTIFACTS) -> dict:
    g = with_scores(splits.attach(pd.read_csv(splits.TRAIN_PATH)))
    train = g[g["split"] == "train"]
    e1 = split_characteristics(g)
    e2 = univariate_and_vif(train)
    e3, e4, disc = block_validity(train)
    res = {"table_e1": e1.to_dict(), "table_e2": e2.to_dict(orient="records"), "table_e3": e3.to_dict(orient="records"),
           "table_e4": e4.to_dict(orient="records"), "discriminant": disc}
    (out_dir / "construct_validity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "construct_validity.md").write_text(to_markdown(e1, e2, e3, e4, disc), encoding="utf-8")
    return res


def _md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def to_markdown(e1, e2, e3, e4, disc) -> str:
    t2 = e2.assign(**{"HR per SD (95% CI)": e2.apply(lambda r: f"{r.hr:.2f} ({r.lo:.2f}-{r.hi:.2f})", axis=1),
                      "p": e2["p"].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"), "VIF": e2["vif"].round(2)}
                   )[["predictor", "n", "HR per SD (95% CI)", "p", "VIF"]]
    t3 = e3.assign(**{"KMO": e3["kmo"].round(3), "alpha (95% CI)": e3.apply(
        lambda r: f"{r.alpha:.3f} ({r.alpha_lo:.3f}-{r.alpha_hi:.3f})", axis=1), "omega": e3["omega"].round(3),
        "AVE": e3["ave"].round(3), "loadings": e3.apply(lambda r: f"{r.loading_min:.2f}-{r.loading_max:.2f}", axis=1)}
                   )[["block", "KMO", "factors_parallel_analysis", "alpha (95% CI)", "omega", "AVE", "loadings"]]
    t4 = e4.assign(loading=e4["loading"].round(3), communality=e4["communality"].round(3))
    L = ["# Construct validity, reliability and predictor checks (Appendix E)", "",
         "_Generated by `python -m core.construct_validity` - do not edit by hand._", "",
         "## Table E.1 - sample characteristics by split", "", *_md(e1.reset_index()), "",
         "## Table E.2 - each predictor and the outcome (univariate Cox, training set) and VIF", "", *_md(t2), "",
         "## Table E.3 - suitability, reliability and convergent validity of the skill blocks (training set)", "",
         *_md(t3), "",
         f"Discriminant validity: HTMT = {disc['htmt']:.3f} (below 0.85); sqrt(AVE) {disc['sqrt_ave_technical']:.3f} "
         f"technical and {disc['sqrt_ave_soft']:.3f} soft, against a correlation of {disc['score_correlation']:.3f} "
         "between the two scores (Fornell-Larcker).", "",
         "## Table E.4 - item loadings on one factor per block (training set)", "", *_md(t4)]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    for row in r["table_e3"]:
        print(f"{row['block']}: KMO {row['kmo']:.3f}, factors {row['factors_parallel_analysis']}, alpha {row['alpha']:.3f} "
              f"({row['alpha_lo']:.3f}-{row['alpha_hi']:.3f}), omega {row['omega']:.3f}, AVE {row['ave']:.3f}, "
              f"loadings {row['loading_min']:.2f}-{row['loading_max']:.2f}")
    d = r["discriminant"]
    print(f"HTMT {d['htmt']:.3f}; sqrt AVE {d['sqrt_ave_technical']:.3f} / {d['sqrt_ave_soft']:.3f}; r {d['score_correlation']:.3f}")
    for row in r["table_e2"]:
        print(f"{row['predictor']:24s} n={row['n']} HR {row['hr']:.2f} ({row['lo']:.2f}-{row['hi']:.2f}) p={row['p']:.4f} VIF {row['vif']:.2f}")
