"""Usability evaluation: System Usability Scale (SUS) scoring and analysis (thesis Sections 3.9 and 5.8, RQ4).

  python -m core.sus_study      -> artifacts/sus_study.{json,md}, artifacts/sus_scores.csv, artifacts/figures/sus_scores.png

Input: the Google Forms responses of the usability questionnaire (docs/usability/SUS_Form_Builder.gs), copied from the
response sheet into "New Data/Usability (SUS)/SUS_evaluations.csv". The file is never edited; every number is calculated
here.

1. Data checks - consent, eligibility, unique participant codes, ten answers 1-5, straight-lining.
2. Scoring (Brooke, 1996) - odd items contribute (answer - 1), even items (5 - answer); sum x 2.5 gives 0-100.
3. Against the pre-set criterion (thesis Table 3.6: mean of at least 68, Hyzy et al., 2022) - mean with t and
   bootstrap 95% intervals, one-sample t-test and Wilcoxon signed-rank test against 68, effect size, share >= 68.
4. Interpretation - curved grade (Lewis & Sauro, 2018), adjective (Bangor et al., 2009), approximate percentile.
5. Reliability (Cronbach's alpha of the ten scored items) and item means.
6. Descriptive subgroups (profile used, background, experience with career sites) - too small to test, reported only.
7. Open comments coded into themes with fixed keyword rules (every comment must receive at least one theme).

artifacts/sus_scores.csv holds one row per participant and is not committed (participant-level data).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from core.market_study import wilson

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "New Data" / "Usability (SUS)" / "SUS_evaluations.csv"
ARTIFACTS = ROOT / "artifacts"
BENCHMARK = 68.0           # pre-set criterion (thesis Table 3.6)
POPULATION_SD = 12.5       # SD of SUS scores across studies, used only for the approximate percentile
SEED, N_BOOT = 2026, 10_000

ITEMS = [f"SUS{i}" for i in range(1, 11)]
ITEM_TEXT = {
    "SUS1": "I think that I would like to use this system frequently.",
    "SUS2": "I found the system unnecessarily complex.",
    "SUS3": "I thought the system was easy to use.",
    "SUS4": "I think that I would need the support of a technical person to be able to use this system.",
    "SUS5": "I found the various functions in this system were well integrated.",
    "SUS6": "I thought there was too much inconsistency in this system.",
    "SUS7": "I would imagine that most people would learn to use this system very quickly.",
    "SUS8": "I found the system very cumbersome to use.",
    "SUS9": "I felt very confident using the system.",
    "SUS10": "I needed to learn a lot of things before I could get going with this system.",
}
CONSENT = ("I have read and understood the information above.",
           "I understand that taking part is voluntary",
           "I understand that my name and personal details are not recorded",
           "I agree to take part in this usability evaluation.")
ELIGIBLE = "All of these apply to me"
CODE = re.compile(r"^P\d{2}$")

# Lewis & Sauro (2018) curved grading scale: (lower bound, grade)
GRADES = [(84.1, "A+"), (80.8, "A"), (78.9, "A-"), (77.2, "B+"), (74.1, "B"), (72.6, "B-"), (71.1, "C+"),
          (65.0, "C"), (62.7, "C-"), (51.7, "D"), (0.0, "F")]
# Bangor, Kortum & Miller (2009): mean SUS score of each adjective rating
ADJECTIVES = [(85.5, "Excellent"), (71.4, "Good"), (50.9, "OK"), (35.7, "Poor"), (20.3, "Awful"),
              (0.0, "Worst imaginable")]

# Fixed keyword rules for the open comments (lower case, substring match). Each comment must match at least one.
ISSUE_THEMES = {
    "Too much on one page / main result not obvious": ["too many sections", "much information", "page was long",
                                                       "which number was the main result"],
    "Terms and charts need explaining (onboarding)": ["confidence", "range", "what the chart meant", "short guide",
                                                      "needed help to start"],
    "Navigation and consistency between pages": ["disconnected", "inconsistent", "find the chat"],
    "Chart readability": ["axis labels"],
    "Mobile layout": ["mobile"],
    "Trust in and currency of the numbers": ["apply to me", "data source date"],
    "Feature request": ["dark mode"],
    "No problem reported": ["nothing really", "nothing major"],
}
USEFUL_THEMES = {
    "What-if sliders": ["what-if"],
    "Skill gaps and in-demand skills (job market)": ["skill", "job market"],
    "Chat assistant": ["chat"],
    "Suggestions and overall presentation": ["suggestions", "all together", "estimate was easy", "layout"],
}
EXPERIENCE_ORDER = {"Never": 0, "A few times a year": 1, "Monthly": 2, "Weekly or more": 3}


# ---------------------------------------------------------------------------
# Loading and checks
# ---------------------------------------------------------------------------
def load(path: Path = SOURCE) -> pd.DataFrame:
    """Read the form export and give the columns short names (SUS1-SUS10, B1-B4, C1-C4)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = {}
    for col in raw.columns:
        m = re.match(r"^(SUS\d+|B\d|C\d)\.", col)
        names[col] = m[1] if m else {"Participant code": "code"}.get(col, col.lower())
    df = raw.rename(columns=names)
    df["code"] = df["code"].str.strip().str.upper()
    for item in ITEMS:
        df[item] = pd.to_numeric(df[item], errors="coerce")
    return df


def checks(df: pd.DataFrame) -> dict:
    """Data-quality checks; rows failing consent, eligibility or completeness are excluded from scoring."""
    consent_ok = df["consent"].map(lambda s: all(c in s for c in CONSENT))
    eligible = df["eligibility"].str.strip() == ELIGIBLE
    code_ok = df["code"].map(lambda c: bool(CODE.match(c)))
    complete = df[ITEMS].notna().all(axis=1) & df[ITEMS].isin([1, 2, 3, 4, 5]).all(axis=1)
    duplicated = df["code"].duplicated(keep=False)
    straight = df[ITEMS].nunique(axis=1) == 1
    pos, neg = df[ITEMS[0::2]].mean(axis=1), df[ITEMS[1::2]].mean(axis=1)
    acquiescent = (pos >= 4) & (neg >= 4)        # agreeing with both positive and negative wording
    keep = consent_ok & eligible & code_ok & complete & ~duplicated
    times = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M", errors="coerce")
    return {
        "rows": int(len(df)), "consent_all_four": int(consent_ok.sum()), "eligible": int(eligible.sum()),
        "valid_code": int(code_ok.sum()), "complete_1_to_5": int(complete.sum()),
        "duplicate_codes": sorted(df.loc[duplicated, "code"].unique().tolist()),
        "straight_lined": df.loc[straight, "code"].tolist(), "acquiescent": df.loc[acquiescent, "code"].tolist(),
        "excluded": df.loc[~keep, "code"].tolist(), "analysed": int(keep.sum()), "keep": keep,
        "first_session": str(times.min()), "last_session": str(times.max()),
        "session_dates": sorted({t.date().isoformat() for t in times.dropna()}),
    }


# ---------------------------------------------------------------------------
# Scoring and statistics
# ---------------------------------------------------------------------------
def contributions(answers: pd.DataFrame) -> pd.DataFrame:
    """0-4 contribution per item: odd (positive) items answer - 1, even (negative) items 5 - answer."""
    out = answers[ITEMS].copy()
    out[ITEMS[0::2]] = out[ITEMS[0::2]] - 1
    out[ITEMS[1::2]] = 5 - out[ITEMS[1::2]]
    return out


def sus_score(answers: pd.DataFrame) -> pd.Series:
    return contributions(answers).sum(axis=1) * 2.5


def grade(score: float) -> str:
    return next(g for lo, g in GRADES if score >= lo)


def adjective(score: float) -> str:
    return next(a for lo, a in ADJECTIVES if score >= lo)


def cronbach_alpha(items: pd.DataFrame) -> float:
    k = items.shape[1]
    return float(k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1)))


def summary(scores: pd.Series, benchmark: float = BENCHMARK) -> dict:
    x = scores.to_numpy(float)
    n, mean, sd = len(x), float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    rng = np.random.default_rng(SEED)
    boot = rng.choice(x, size=(N_BOOT, n), replace=True).mean(axis=1)
    t = stats.ttest_1samp(x, benchmark)
    t_one = stats.ttest_1samp(x, benchmark, alternative="greater")
    w = stats.wilcoxon(x - benchmark, alternative="two-sided")
    k = int((x >= benchmark).sum())
    return {
        "n": n, "mean": round(mean, 2), "sd": round(sd, 2), "se": round(se, 2), "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
        "min": float(x.min()), "max": float(x.max()),
        "t_crit": round(tcrit, 3), "ci_low": round(mean - tcrit * se, 2), "ci_high": round(mean + tcrit * se, 2),
        "boot_ci_low": round(float(np.percentile(boot, 2.5)), 2),
        "boot_ci_high": round(float(np.percentile(boot, 97.5)), 2),
        "benchmark": benchmark, "t": round(float(t.statistic), 3), "df": n - 1, "p_two_sided": float(t.pvalue),
        "p_one_sided": float(t_one.pvalue), "wilcoxon_w": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "cohens_d": round((mean - benchmark) / sd, 2),
        "at_or_above_benchmark": k, "share_at_or_above": round(k / n, 3),
        "share_ci": [round(v, 3) for v in wilson(k, n)],
        "grade": grade(mean), "adjective": adjective(mean),
        "percentile_approx": round(float(stats.norm.cdf((mean - benchmark) / POPULATION_SD)) * 100),
        "criterion_met": bool(mean >= benchmark), "ci_above_benchmark": bool(mean - tcrit * se > benchmark),
    }


def item_table(answers: pd.DataFrame) -> list[dict]:
    contrib = contributions(answers)
    rows = []
    for item in ITEMS:
        rows.append({"item": item, "text": ITEM_TEXT[item], "tone": "positive" if int(item[3:]) % 2 else "negative",
                     "mean_answer": round(float(answers[item].mean()), 2), "sd": round(float(answers[item].std()), 2),
                     "mean_contribution": round(float(contrib[item].mean()), 2)})
    return rows


def subgroups(df: pd.DataFrame) -> dict:
    out = {}
    for col, label in (("C1", "Profile entered in the tasks"), ("B1", "Background"),
                       ("B3", "Type of institution"), ("B4", "Use of career or job-search sites")):
        g = df.groupby(col)["sus"].agg(["count", "mean", "min", "max"]).reset_index()
        out[col] = {"label": label, "rows": [{"group": r[col], "n": int(r["count"]), "mean": round(float(r["mean"]), 2),
                                             "min": float(r["min"]), "max": float(r["max"])} for _, r in g.iterrows()]}
    exp = df["B4"].map(EXPERIENCE_ORDER)
    rho = stats.spearmanr(exp, df["sus"])
    out["experience_spearman"] = {"rho": round(float(rho.statistic), 3), "p": float(rho.pvalue), "n": int(exp.notna().sum())}
    own, example = df.loc[df["C1"] == "My own details", "sus"], df.loc[df["C1"] == "The example profile card", "sus"]
    if len(own) and len(example):
        mw = stats.mannwhitneyu(own, example, alternative="two-sided")
        out["own_vs_example"] = {"n_own": int(len(own)), "n_example": int(len(example)),
                                 "u": float(mw.statistic), "p": float(mw.pvalue)}
    return out


# ---------------------------------------------------------------------------
# Open comments
# ---------------------------------------------------------------------------
def code_comments(df: pd.DataFrame, column: str, themes: dict[str, list[str]]) -> dict:
    """Assign each non-empty comment to every theme whose keywords it contains."""
    coded: dict[str, list[str]] = {t: [] for t in themes}
    uncoded = []
    for _, row in df.iterrows():
        text = str(row[column]).strip()
        if not text:
            continue
        hits = [t for t, words in themes.items() if any(w in text.lower() for w in words)]
        for t in hits:
            coded[t].append(row["code"])
        if not hits:
            uncoded.append(row["code"])
    rows = [{"theme": t, "n": len(c), "participants": c} for t, c in coded.items()]
    return {"themes": sorted(rows, key=lambda r: -r["n"]), "uncoded": uncoded}


# ---------------------------------------------------------------------------
# Figure, report and run
# ---------------------------------------------------------------------------
def plot_scores(df: pd.DataFrame, s: dict, path: Path) -> None:
    from core.evaluate import ACCENT, ACCENT_LIGHT, INK, INK_MUTED, SURFACE, _style
    d = df.sort_values("sus")
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(d))
    ax.axvspan(s["ci_low"], s["ci_high"], color=ACCENT_LIGHT, alpha=0.35, linewidth=0)
    ax.axvline(s["mean"], color=ACCENT, linewidth=1.5)
    ax.axvline(s["benchmark"], color=INK_MUTED, linestyle="--", linewidth=1)
    ax.scatter(d["sus"], y, s=56, c=[ACCENT if v >= s["benchmark"] else INK_MUTED for v in d["sus"]],
               edgecolors=SURFACE, linewidths=2, zorder=3)
    for yi, v in zip(y, d["sus"]):
        ax.annotate(f"{v:g}", (v, yi), textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK_MUTED)
    ax.annotate(f"mean {s['mean']:.2f} (95% CI {s['ci_low']:.2f}-{s['ci_high']:.2f})", (s["mean"], len(d) - 0.2),
                textcoords="offset points", xytext=(4, 0), fontsize=8, color=ACCENT)
    ax.annotate(f"benchmark {s['benchmark']:g}", (s["benchmark"], -0.9), textcoords="offset points",
                xytext=(-4, 0), ha="right", fontsize=8, color=INK_MUTED)
    ax.set_yticks(y, d["code"], fontsize=9, color=INK)
    ax.set_xlim(40, 100)
    ax.set_ylim(-1.3, len(d) + 0.3)
    ax.set_xlabel("SUS score (0-100)", fontsize=9, color=INK_MUTED)
    ax.set_title(f"System Usability Scale scores (n = {s['n']})", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def analyse(df: pd.DataFrame) -> dict:
    chk = checks(df)
    keep = chk.pop("keep")
    d = df[keep].copy()
    d["sus"] = sus_score(d)
    return {
        "source": SOURCE.relative_to(ROOT).as_posix(), "checks": chk, "summary": summary(d["sus"]),
        "alpha": round(cronbach_alpha(contributions(d)), 3), "items": item_table(d),
        "scores": {r["code"]: float(r["sus"]) for _, r in d.iterrows()},
        "participants": {c: d[c].value_counts().to_dict() for c in ("B1", "B2", "B3", "B4", "C1")},
        "subgroups": subgroups(d),
        "issues": code_comments(d, "C3", ISSUE_THEMES), "useful": code_comments(d, "C2", USEFUL_THEMES),
        "_frame": d,
    }


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def to_markdown(r: dict) -> str:
    s, c = r["summary"], r["checks"]
    lines = [
        "# Usability evaluation: System Usability Scale", "",
        "_Generated by `python -m core.sus_study` - do not edit by hand._", "",
        f"Source: `{r['source']}` (Google Forms responses, copied from the response sheet). Sessions on {', '.join(c['session_dates'])}, "
        f"{c['first_session'][11:16]} to {c['last_session'][11:16]}.", "",
        "## Data checks", "",
        "| Check | Result |", "|---|---|",
        f"| Responses in the export | {c['rows']} |",
        f"| Gave all four consent statements | {c['consent_all_four']} |",
        f"| Confirmed eligibility | {c['eligible']} |",
        f"| Valid participant code (P + two digits) | {c['valid_code']} |",
        f"| Duplicate codes | {', '.join(c['duplicate_codes']) or 'none'} |",
        f"| All ten SUS items answered 1-5 | {c['complete_1_to_5']} |",
        f"| Same answer to all ten items (straight-lining) | {', '.join(c['straight_lined']) or 'none'} |",
        f"| Agreed with both positive and negative items (mean >= 4 on both) | {', '.join(c['acquiescent']) or 'none'} |",
        f"| Excluded | {', '.join(c['excluded']) or 'none'} |",
        f"| **Analysed** | **{c['analysed']}** |", "",
        "## Scoring", "",
        "Brooke (1996): odd-numbered (positive) items contribute answer - 1, even-numbered (negative) items 5 - answer; "
        "the ten contributions (0-4 each) are summed and multiplied by 2.5, giving 0-100.", "",
        "## Result against the pre-set criterion (mean SUS of at least 68)", "",
        "| Measure | Value |", "|---|---|",
        f"| n | {s['n']} |",
        f"| **Mean SUS** | **{s['mean']:.2f}** |",
        f"| SD / standard error | {s['sd']:.2f} / {s['se']:.2f} |",
        f"| 95% CI (t, df = {s['df']}, t = {s['t_crit']}) | {s['ci_low']:.2f} to {s['ci_high']:.2f} |",
        f"| 95% CI (bootstrap, {N_BOOT:,} resamples, seed {SEED}) | {s['boot_ci_low']:.2f} to {s['boot_ci_high']:.2f} |",
        f"| Median (IQR) | {s['median']:g} ({s['q1']:g} to {s['q3']:g}) |",
        f"| Minimum / maximum | {s['min']:g} / {s['max']:g} |",
        f"| One-sample t-test against {s['benchmark']:g} | t({s['df']}) = {s['t']:.2f}, {_p(s['p_two_sided'])} "
        f"(two-sided), {_p(s['p_one_sided'])} (one-sided, mean > {s['benchmark']:g}) |",
        f"| Wilcoxon signed-rank test against {s['benchmark']:g} | W = {s['wilcoxon_w']:g}, {_p(s['wilcoxon_p'])} |",
        f"| Effect size (mean - {s['benchmark']:g}) / SD | d = {s['cohens_d']:.2f} |",
        f"| Participants scoring {s['benchmark']:g} or more | {s['at_or_above_benchmark']} of {s['n']} "
        f"({s['share_at_or_above']:.0%}, Wilson 95% CI {s['share_ci'][0]:.0%} to {s['share_ci'][1]:.0%}) |",
        f"| Curved grade (Lewis & Sauro, 2018) | {s['grade']} |",
        f"| Adjective (Bangor et al., 2009) | {s['adjective']} |",
        f"| Approximate percentile (normal, mean 68, SD {POPULATION_SD:g}) | {s['percentile_approx']}th |",
        f"| Internal consistency (Cronbach's alpha, 10 scored items) | {r['alpha']:.2f} |", "",
        f"**Criterion {'met' if s['criterion_met'] else 'not met'}**: the mean is {s['mean']:.2f} and the lower end of the "
        f"95% interval ({s['ci_low']:.2f}) is {'above' if s['ci_above_benchmark'] else 'not above'} {s['benchmark']:g}.", "",
        "## Individual scores", "",
        "| Participant | " + " | ".join(r["scores"]) + " |",
        "|---|" + "---|" * len(r["scores"]),
        "| SUS | " + " | ".join(f"{v:g}" for v in r["scores"].values()) + " |", "",
        "## Items", "",
        "| Item | Wording | Tone | Mean answer (1-5) | SD | Mean contribution (0-4) |", "|---|---|---|---|---|---|",
        *[f"| {i['item']} | {i['text']} | {i['tone']} | {i['mean_answer']:.2f} | {i['sd']:.2f} | {i['mean_contribution']:.2f} |"
          for i in r["items"]], "",
        "Lowest contributions show where usability is weakest.", "",
        "## Participants", "",
    ]
    for col, label in (("B1", "Background"), ("B2", "Degree area"), ("B3", "Type of institution"),
                       ("B4", "Use of career or job-search sites"), ("C1", "Profile entered in the tasks")):
        lines.append(f"- {label}: " + "; ".join(f"{k} {v}" for k, v in r["participants"][col].items()))
    sg = r["subgroups"]
    lines += ["", "## Subgroups (descriptive only - groups are too small for reliable tests)", ""]
    for col in ("C1", "B1", "B3", "B4"):
        lines += [f"**{sg[col]['label']}**", "", "| Group | n | Mean SUS | Min | Max |", "|---|---|---|---|---|",
                  *[f"| {g['group']} | {g['n']} | {g['mean']:.1f} | {g['min']:g} | {g['max']:g} |" for g in sg[col]["rows"]], ""]
    e = sg["experience_spearman"]
    lines.append(f"Use of career sites (Never = 0 ... Weekly = 3) vs SUS: Spearman rho = {e['rho']:.2f}, "
                 f"{_p(e['p'])}, n = {e['n']}.")
    if "own_vs_example" in sg:
        o = sg["own_vs_example"]
        lines.append(f"Own details (n = {o['n_own']}) vs example card (n = {o['n_example']}): Mann-Whitney U = "
                     f"{o['u']:g}, {_p(o['p'])}.")
    for key, title in (("issues", "Usability issues (C3, coded with fixed keyword rules)"),
                       ("useful", "Most useful parts (C2, coded with fixed keyword rules)")):
        lines += ["", f"## {title}", "", "| Theme | Participants | n |", "|---|---|---|",
                  *[f"| {t['theme']} | {', '.join(t['participants']) or '-'} | {t['n']} |" for t in r[key]["themes"]],
                  "", f"Comments not matched by any rule: {', '.join(r[key]['uncoded']) or 'none'}."]
    lines += ["", "## Not recorded in the form", "",
              "Task-by-task completion and time per task were not part of the questionnaire; they are recorded only if "
              "the session sheet (Appendix L) was filled in during the sessions. Comments show that P06 and P11 asked "
              "the researcher for help.", "",
              "## References", "",
              "- Bangor, A., Kortum, P., & Miller, J. (2009). Determining what individual SUS scores mean: Adding an "
              "adjective rating scale. Journal of Usability Studies, 4(3), 114-123.",
              "- Brooke, J. (1996). SUS: A 'quick and dirty' usability scale. In P. W. Jordan et al. (Eds.), Usability "
              "evaluation in industry (pp. 189-194). Taylor & Francis.",
              "- Hyzy, M., et al. (2022). System Usability Scale benchmarking for digital health apps: Meta-analysis. "
              "JMIR mHealth and uHealth, 10(8), e37290.",
              "- Lewis, J. R., & Sauro, J. (2018). Item benchmarks for the System Usability Scale. Journal of Usability "
              "Studies, 13(3), 158-167.", ""]
    return "\n".join(lines)


def run(source: Path = SOURCE, out_dir: Path = ARTIFACTS) -> dict:
    result = analyse(load(source))
    frame = result.pop("_frame")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    cols = ["code", "timestamp", "B1", "B2", "B3", "B4", "C1", *ITEMS, "sus", "C2", "C3", "C4"]
    frame[cols].to_csv(out_dir / "sus_scores.csv", index=False)
    plot_scores(frame, result["summary"], out_dir / "figures" / "sus_scores.png")
    (out_dir / "sus_study.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    (out_dir / "sus_study.md").write_text(to_markdown(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    s = r["summary"]
    print(f"SUS n = {s['n']}: mean {s['mean']:.2f} (95% CI {s['ci_low']:.2f}-{s['ci_high']:.2f}), "
          f"grade {s['grade']}, {s['adjective']}; criterion >= {s['benchmark']:g} {'met' if s['criterion_met'] else 'not met'}")
