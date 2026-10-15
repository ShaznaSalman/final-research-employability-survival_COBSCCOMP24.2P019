"""Exploratory-analysis figures for the thesis appendix (REPORT ONLY - nothing here changes the model or the app).

All three use the TRAINING set only (core/splits.py), so the validation and test sets stay untouched:
  - correlation heatmap : Spearman correlation between the model's predictors (pairwise complete)
  - Likert distribution : share of graduates giving each rating (1-5) to the 17 technical and 9 soft-skill items
  - scree plot          : eigenvalues of each skill block with Horn's parallel analysis (500 random data sets, 95th
                          percentile; same method and seed as the construct-validity analysis reported in Table E.3)

Writes artifacts/figures/eda_{correlation,likert,scree}.png and artifacts/eda_figures.{json,md}.
Run:  python -m core.eda_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from core import evaluate, preprocess, schema, splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
PA_SEED, PA_ITERS = 20261014, 500           # as in the construct-validity analysis (Table E.3)
PREDICTORS = {"gpa": "GPA", "internship_months": "Internship months", "certifications_count": "Certifications",
              "project_count": "Projects", "technical_mean": "Technical skill score", "soft_mean": "Soft-skill score",
              "has_portfolio": "Portfolio", "prep_years": "Years of preparation",
              "self_learning_freq": "Self-learning frequency", "hackathon_count": "Hackathons",
              "open_source": "Open source", "sdlc_familiarity": "SDLC familiarity"}
# Reference palette (dataviz skill): ordinal blue ramp for ratings 1-5 (validated), blue-grey-red for correlation
RATING_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
SCALES = {"technical": ("Technical skills (Q18)", "1 = No knowledge, 5 = Advanced level"),
          "soft": ("Soft skills (Q19)", "1 = Very low, 5 = Excellent")}


# ---------------------------------------------------------------------------------------------- statistics
def spearman(train: pd.DataFrame) -> pd.DataFrame:
    X = preprocess.raw_feature_frame(train)[list(PREDICTORS)].apply(pd.to_numeric, errors="coerce")
    return X.corr(method="spearman").rename(index=PREDICTORS, columns=PREDICTORS)


def likert_shares(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for block, skills in (("technical", schema.TECHNICAL_SKILLS), ("soft", schema.SOFT_SKILLS)):
        rows = []
        for s in skills:
            v = pd.to_numeric(train[s.column], errors="coerce").dropna()
            rows.append({"item": s.label, "n": len(v), "mean": float(v.mean()),
                         **{str(k): float((v == k).mean()) for k in range(1, 6)}})
        out[block] = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    return out


def parallel_analysis(rng, n: int, p: int, iters: int = PA_ITERS) -> np.ndarray:
    eig = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
                    for _ in range(iters)])
    return np.percentile(eig, 95, axis=0)


def scree(train: pd.DataFrame) -> dict:
    tech, soft = list(schema.TECH_COLUMNS), list(schema.SOFT_COLUMNS)
    X = train[tech + soft].apply(pd.to_numeric, errors="coerce").dropna()
    rng = np.random.default_rng(PA_SEED)
    out = {}
    for block, cols in (("technical", tech), ("soft", soft)):        # same order as Table E.3: same random draws
        eig = np.sort(np.linalg.eigvalsh(X[cols].corr().to_numpy()))[::-1]
        pa = parallel_analysis(rng, len(X), len(cols))
        n_fac = int(np.argmax(eig <= pa)) if (eig <= pa).any() else len(cols)
        out[block] = {"n": len(X), "items": len(cols), "eigenvalues": eig.round(4).tolist(),
                      "parallel_95th": pa.round(4).tolist(), "factors": n_fac}
    return out


# ---------------------------------------------------------------------------------------------- figures
def plot_correlation(cm: pd.DataFrame, n: int, path: Path):
    k = len(cm)
    mask = np.triu(np.ones((k, k), bool), 1)                          # lower triangle + diagonal
    fig, ax = plt.subplots(figsize=(8.2, 6.8), facecolor=evaluate.SURFACE)
    ax.set_facecolor(evaluate.SURFACE)
    im = ax.imshow(np.ma.masked_array(cm.to_numpy(), mask), cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(k):
        for j in range(i + 1):
            v = cm.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else evaluate.INK)
    ax.set_xticks(range(k), cm.columns, rotation=50, ha="right", fontsize=8, color=evaluate.INK)
    ax.set_yticks(range(k), cm.index, fontsize=8, color=evaluate.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Spearman correlation", fontsize=9, color=evaluate.INK_MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"Correlation between the predictors (training set, n = {n})", fontsize=11, color=evaluate.INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


def plot_likert(shares: dict[str, pd.DataFrame], n: int, path: Path):
    heights = [len(shares["technical"]), len(shares["soft"])]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 0.32 * sum(heights) + 2.4), facecolor=evaluate.SURFACE,
                             gridspec_kw={"height_ratios": heights})
    for ax, block in zip(axes, ("technical", "soft")):
        d = shares[block]
        y = np.arange(len(d))
        left = np.zeros(len(d))
        for k, color in zip(range(1, 6), RATING_COLORS):
            w = d[str(k)].to_numpy()
            ax.barh(y, w, left=left, color=color, height=0.72, edgecolor=evaluate.SURFACE, linewidth=1.5,
                    label=str(k))
            left += w
        ax.set_yticks(y, [f"{i}  (mean {m:.2f})" for i, m in zip(d["item"], d["mean"])], fontsize=8,
                      color=evaluate.INK)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.tick_params(axis="x", colors=evaluate.INK_MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        title, scale = SCALES[block]
        ax.set_title(f"{title}: {scale}", fontsize=10, color=evaluate.INK, loc="left")
    axes[0].legend(title="Rating", ncols=5, frameon=False, fontsize=8, title_fontsize=8,
                   loc="lower right", bbox_to_anchor=(1.0, 1.08))
    axes[1].set_xlabel("Share of graduates giving each rating", fontsize=9, color=evaluate.INK_MUTED)
    fig.suptitle(f"Self-ratings at degree completion (training set, n = {n}), sorted by mean", x=0.01, ha="left",
                 fontsize=11, color=evaluate.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


def plot_scree(sc: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=evaluate.SURFACE)
    for ax, block in zip(axes, ("technical", "soft")):
        evaluate._style(ax)
        r = sc[block]
        x = np.arange(1, r["items"] + 1)
        ax.plot(x, r["eigenvalues"], "o-", color=evaluate.ACCENT, linewidth=2, markersize=6,
                markeredgecolor=evaluate.SURFACE, label="Observed eigenvalue")
        ax.plot(x, r["parallel_95th"], "--", color=evaluate.INK_MUTED, linewidth=1.5,
                label="Random data, 95th percentile")
        ax.set_title(f"{SCALES[block][0].split(' (')[0]} ({r['items']} items): {r['factors']} factor"
                     f"{'s' if r['factors'] != 1 else ''} retained", fontsize=10, color=evaluate.INK, loc="left")
        ax.set_xlabel("Factor", fontsize=9, color=evaluate.INK_MUTED)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Eigenvalue", fontsize=9, color=evaluate.INK_MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Scree plots with parallel analysis (training set, n = {sc['technical']['n']})", x=0.01,
                 ha="left", fontsize=11, color=evaluate.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- run
def run(out_dir: Path = ARTIFACTS) -> dict:
    train = splits.load("train")
    cm, sh, sc = spearman(train), likert_shares(train), scree(train)
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_correlation(cm, len(train), figs / "eda_correlation.png")
    plot_likert(sh, len(train), figs / "eda_likert.png")
    plot_scree(sc, figs / "eda_scree.png")
    off = cm.where(~np.eye(len(cm), dtype=bool)).abs().stack()
    a, b = off.idxmax()
    res = {"n_train": len(train), "strongest_correlation": {"pair": [a, b], "rho": round(float(cm.loc[a, b]), 3)},
           "correlation": cm.round(3).to_dict(), "scree": sc,
           "likert": {k: v.round(4).to_dict(orient="records") for k, v in sh.items()}}
    (out_dir / "eda_figures.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "eda_figures.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def to_markdown(r: dict) -> str:
    s, sc = r["strongest_correlation"], r["scree"]
    lik = {k: pd.DataFrame(v) for k, v in r["likert"].items()}
    L = ["# Exploratory-analysis figures (report only)", "",
         "_Generated by `python -m core.eda_figures` - do not edit by hand._", "",
         f"Training set only (n = {r['n_train']}); the validation and test sets are not used.", "",
         "## Correlation between predictors (`figures/eda_correlation.png`)", "",
         f"Spearman, pairwise complete. Strongest correlation: {s['pair'][0]} and {s['pair'][1]} "
         f"(rho = {s['rho']:.2f}).", "",
         "## Likert distribution (`figures/eda_likert.png`)", ""]
    for block in ("technical", "soft"):
        d = lik[block]
        L.append(f"- {SCALES[block][0]}: item means {d['mean'].min():.2f} ({d.iloc[0]['item']}) to "
                 f"{d['mean'].max():.2f} ({d.iloc[-1]['item']}).")
    L += ["", "## Scree plots with parallel analysis (`figures/eda_scree.png`)", ""]
    for block in ("technical", "soft"):
        x = sc[block]
        L.append(f"- {SCALES[block][0]}: first eigenvalue {x['eigenvalues'][0]:.2f} "
                 f"({100 * x['eigenvalues'][0] / x['items']:.1f}% of variance); {x['factors']} factor(s) above the "
                 f"95th percentile of {PA_ITERS} random data sets (n = {x['n']}).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    print("strongest:", r["strongest_correlation"])
    print({k: v["factors"] for k, v in r["scree"].items()})
