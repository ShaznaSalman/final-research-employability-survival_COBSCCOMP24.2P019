"""Results promised in Chapter 3 that the other modules do not produce (REPORT ONLY - nothing here changes a model).

  python -m core.thesis_results

  RQ1  Kaplan-Meier time to first IT job (all modelling graduates) and log-rank tests by group (training set);
       time ratios of the selected AFT model (fitted on the training set).
  RQ3  pre-set conditions of Table 3.6: the selected model vs the Kaplan-Meier baseline on the test set (the baseline
       is a no-model reference; the model itself was already scored once), training vs validation vs test Uno's C
       (overfitting), and transfer across institution types (fit on train minus one type, score on the VALIDATION
       graduates of that type - the test set is not used).
  RQ2  per-skill alignment: employer importance (E8/E9) minus graduate self-rating (Q18/Q19), Mann-Whitney U with
       Benjamini-Hochberg correction over 26 skills, share of job ads naming each skill, and Spearman rank correlation
       between employers' stated importance and advertised demand.

Writes artifacts/thesis_results.{json,md}.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr
from sksurv.metrics import integrated_brier_score


from core import evaluate, preprocess, schema, splits

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROC = ROOT / "data" / "processed"
HORIZONS = (3, 6, 12, 24)


def _km_at(df: pd.DataFrame) -> dict:
    km = KaplanMeierFitter().fit(df["duration_months"], df["event"])
    out = {f"employed_by_{h}m": round(float(1 - km.predict(h)), 3) for h in HORIZONS}
    med = km.median_survival_time_
    out["median_months"] = None if np.isinf(med) else float(med)
    return out


def km_groups(train: pd.DataFrame) -> list[dict]:
    groups = {
        "Internship completed": train["internship_completed"].map({"yes": "yes", "no": "no"}),
        "Any certification": np.where(train["certifications_count"] > 0, "1 or more", "none"),
        "GPA 3.0 or higher": np.where(train["gpa"] >= 3.0, "3.0+", "below 3.0"),
        "Portfolio before completion": train["has_portfolio"],
        "Institution type": train["university_type"],
        "Degree group": train["degree_group"],
    }
    rows = []
    for name, g in groups.items():
        g = pd.Series(g, index=train.index)
        ok = g.notna()
        d, g = train[ok], g[ok]
        test = multivariate_logrank_test(d["duration_months"], g, d["event"])
        levels = {lvl: {"n": int((g == lvl).sum()), **_km_at(d[g == lvl])} for lvl in sorted(g.unique())}
        rows.append({"group": name, "p": float(test.p_value), "levels": levels})
    return rows


def time_ratios(bundle: dict) -> pd.DataFrame:
    c = bundle["model"].coefficients
    return pd.DataFrame({"time_ratio": np.exp(c["coef"]), "lower": np.exp(c["lower"]), "upper": np.exp(c["upper"]),
                         "p": c["p"]}).round(3)


def _km_curve(train: pd.DataFrame, times: np.ndarray, n: int) -> np.ndarray:
    km = KaplanMeierFitter().fit(train["duration_months"], train["event"])
    return np.tile(km.predict(times).to_numpy(), (n, 1))


def model_checks(bundle: dict, train, valid, test) -> dict:
    pre, model = bundle["preprocessor"], bundle["model"]
    y_tr = preprocess.sksurv_y(train)
    tau = evaluate.truncation_tau(train["duration_months"].to_numpy())
    c = {role: evaluate.uno_c(y_tr, preprocess.sksurv_y(d), model.predict_risk(pre.transform(d)), tau)
         for role, d in (("train", train), ("validation", valid), ("test", test))}
    # Integrated Brier score of the model and of the Kaplan-Meier baseline on the same test graduates
    y_te = preprocess.sksurv_y(test)
    keep = y_te["time"] < y_tr["time"].max()
    times = np.arange(1.0, min(tau, y_te["time"][keep].max()))
    X_te = pre.transform(test)[keep]
    ibs_model = integrated_brier_score(y_tr, y_te[keep], model.predict_survival(X_te, times), times)
    ibs_km = integrated_brier_score(y_tr, y_te[keep], _km_curve(train, times, int(keep.sum())), times)
    return {"uno_c": {k: round(v, 3) for k, v in c.items()}, "tau": round(tau, 1),
            "ibs_test_model": round(float(ibs_model), 3), "ibs_test_km_baseline": round(float(ibs_km), 3)}


def institution_transfer(bundle: dict, train, valid) -> list[dict]:
    rows = []
    for t in schema.UNIVERSITY_TYPES:
        tr, va = train[train["university_type"] != t], valid[valid["university_type"] == t]
        if va["event"].sum() < 5:
            continue
        pre, m = evaluate.fit_all(tr, {}, bundle["aft_distribution"], "aft_best", evaluate.SEED)
        y_tr = preprocess.sksurv_y(tr)
        tau = evaluate.truncation_tau(tr["duration_months"].to_numpy())
        c = evaluate.uno_c(y_tr, preprocess.sksurv_y(va), m.predict_risk(pre.transform(va)), tau)
        rows.append({"held_out_type": t, "train_n": len(tr), "validation_n": len(va),
                     "validation_events": int(va["event"].sum()), "uno_c": round(c, 3)})
    return rows


def alignment() -> tuple[pd.DataFrame, dict]:
    grads = pd.read_csv(PROC / "questionnaire_graduates.csv")
    emp = pd.read_csv(PROC / "employers.csv")
    ads = pd.read_csv(PROC / "job_ads.csv")
    rows = []
    for s in schema.SKILL_VOCABULARY:
        g, e = grads[s.column].dropna(), emp[f"imp_{s.key}"].dropna()
        u = mannwhitneyu(e, g, alternative="two-sided")
        share = float(ads["skills"].fillna("").str.split(";").apply(lambda x, k=s.key: k in x).mean())
        rows.append({"skill": s.label, "kind": s.kind, "graduate_mean": g.mean(), "employer_mean": e.mean(),
                     "gap": e.mean() - g.mean(), "rank_biserial": 2 * u.statistic / (len(e) * len(g)) - 1,
                     "p": u.pvalue, "ad_share": share, "n_grad": len(g), "n_emp": len(e)})
    t = pd.DataFrame(rows)
    t["q"] = false_discovery_control(t["p"], method="bh")
    rho, p = spearmanr(t["employer_mean"], t["ad_share"])
    rho_t, p_t = spearmanr(t.loc[t.kind == "technical", "employer_mean"], t.loc[t.kind == "technical", "ad_share"])
    t = t.sort_values("gap", ascending=False).reset_index(drop=True)
    return t, {"spearman_rho": round(float(rho), 3), "p": float(p),
               "spearman_rho_technical": round(float(rho_t), 3), "p_technical": float(p_t),
               "n_graduates": int(t["n_grad"].max()), "n_employers": int(t["n_emp"].max()), "n_ads": len(ads)}


def render(res: dict, align: pd.DataFrame) -> str:
    L = ["# Thesis results (report only)", "", "_Generated by `python -m core.thesis_results` - do not edit by hand._", ""]
    k = res["km_all"]
    L += ["## RQ1: time to first IT job (all modelling graduates)", "",
          f"n = {res['n_modelling']}; employed by 3 / 6 / 12 / 24 months: "
          + " / ".join(f"{100 * k[f'employed_by_{h}m']:.1f}%" for h in HORIZONS)
          + f"; median {k['median_months']} months.", "",
          "## RQ1: log-rank tests by group (training set)", "",
          "| Group | Level | n | Employed by 6 m | Median (months) | Log-rank p |", "|---|---|---|---|---|---|"]
    for g in res["km_groups"]:
        for i, (lvl, v) in enumerate(g["levels"].items()):
            L.append(f"| {g['group'] if i == 0 else ''} | {lvl} | {v['n']} | {100 * v['employed_by_6m']:.1f}% | "
                     f"{v['median_months']} | {'<0.001' if g['p'] < 0.001 else round(g['p'], 3) if i == 0 else ''} |")
    L += ["", "## RQ1: time ratios of the selected model (training set)", "",
          "Time ratio < 1: associated with a SHORTER time to the first IT job (numeric features per one SD).", "",
          "| Feature | Time ratio (95% CI) | p |", "|---|---|---|"]
    for f, r in res["time_ratios"].items():
        L.append(f"| {f} | {r['time_ratio']:.2f} ({r['lower']:.2f}-{r['upper']:.2f}) | "
                 f"{'<0.001' if r['p'] < 0.001 else round(r['p'], 3)} |")
    m = res["model_checks"]
    L += ["", "## RQ3: pre-set conditions (Table 3.6)", "",
          f"- Uno's C: train {m['uno_c']['train']}, validation {m['uno_c']['validation']}, test {m['uno_c']['test']} "
          f"(tau {m['tau']} months).",
          f"- Integrated Brier score on the test set: model {m['ibs_test_model']}, Kaplan-Meier baseline "
          f"{m['ibs_test_km_baseline']}.", "",
          "| Held-out institution type | Train n | Validation n (events) | Uno's C |", "|---|---|---|---|"]
    for r in res["institution_transfer"]:
        L.append(f"| {r['held_out_type']} | {r['train_n']} | {r['validation_n']} ({r['validation_events']}) | {r['uno_c']} |")
    a = res["alignment_summary"]
    L += ["", f"## RQ2: per-skill alignment ({a['n_graduates']} graduates, {a['n_employers']} employers, {a['n_ads']} ads)",
          "", "Gap = employer importance mean - graduate self-rating mean (both 1-5).", "",
          "| Skill | Type | Graduates | Employers | Gap | Rank-biserial | q (BH) | % of ads |", "|---|---|---|---|---|---|---|---|"]
    for r in align.itertuples():
        q = "<0.001" if r.q < 0.001 else f"{r.q:.3f}"
        L.append(f"| {r.skill} | {r.kind} | {r.graduate_mean:.2f} | {r.employer_mean:.2f} | {r.gap:+.2f} | "
                 f"{r.rank_biserial:+.2f} | {q} | {100 * r.ad_share:.0f}% |")
    L += ["", f"Spearman rank correlation between employer importance and share of ads, 26 skills: rho = "
          f"{a['spearman_rho']} (p = {a['p']:.3f}); technical skills only: rho = {a['spearman_rho_technical']} "
          f"(p = {a['p_technical']:.3f})."]
    return "\n".join(L) + "\n"


def run() -> dict:
    warnings.filterwarnings("ignore")
    bundle = joblib.load(ART / "final_model.pkl")
    train, valid, test = evaluate.load_roles()
    allm = pd.concat([train, valid, test])
    tr_full = splits.load("train")
    align, a_sum = alignment()
    res = {"n_modelling": len(allm), "km_all": _km_at(allm), "km_groups": km_groups(tr_full),
           "time_ratios": time_ratios(bundle).to_dict(orient="index"),
           "model_checks": model_checks(bundle, train, valid, test),
           "institution_transfer": institution_transfer(bundle, train, valid),
           "alignment_summary": a_sum, "alignment": align.round(4).to_dict(orient="records")}
    (ART / "thesis_results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (ART / "thesis_results.md").write_text(render(res, align), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
    print((ART / "thesis_results.md").read_text(encoding="utf-8"))
