"""Time-dependent AUC and the Brier score over time (REPORT ONLY - supplementary metrics, not used for selection).

The pre-specified metrics (Uno's C, integrated Brier score, calibration) are in core/evaluate.py. These two add
detail on the VALIDATION set, for every candidate model fitted on the TRAINING set with the settings chosen in the
main analysis (artifacts/model_comparison.csv); the locked test set is never used:
  - cumulative/dynamic AUC at 3, 6 and 12 months: of graduates employed by t and graduates still searching at t, how
    often the employed one had the higher predicted chance (the survival version of ROC-AUC; inverse-probability-of-
    censoring weights from the training set)
  - Brier score at each month for the selected model against the Kaplan-Meier baseline (the same curve for everyone),
    the curve whose average is the integrated Brier score

Writes artifacts/time_dependent_metrics.{json,md} and artifacts/figures/brier_over_time.png.
Run:  python -m core.time_dependent_metrics
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import brier_score, cumulative_dynamic_auc, integrated_brier_score

from core import evaluate, preprocess, splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
HORIZONS = (3.0, 6.0, 12.0)
SELECTED = "aft_best"


def candidates(out_dir: Path) -> dict[str, tuple[dict, str | None]]:
    """Settings chosen in the main analysis: {model: (params, AFT distribution or None)}."""
    comp = pd.read_csv(out_dir / "model_comparison.csv").set_index("model")
    out = {}
    for name in evaluate.CANDIDATES:
        params = json.loads(comp.loc[name, "tuned_params"])
        out[name] = ({}, params["distribution"]) if name == "aft_best" else (params, None)
    return out


def run(out_dir: Path = ARTIFACTS) -> dict:
    train, valid = splits.load("train"), splits.load("validation")
    y_tr, y_va = preprocess.sksurv_y(train), preprocess.sksurv_y(valid)
    tau = evaluate.truncation_tau(y_tr["time"])
    keep = y_va["time"] < y_tr["time"].max()                 # censoring weights undefined beyond training follow-up
    times = np.arange(1.0, min(tau, y_va["time"][keep].max()))
    km = KaplanMeierFitter().fit(y_tr["time"], y_tr["event"])
    km_curve = np.tile(km.survival_function_at_times(times).to_numpy(), (int(keep.sum()), 1))
    _, bs_km = brier_score(y_tr, y_va[keep], km_curve, times)

    models, curves = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (params, dist) in candidates(out_dir).items():
            pre, m = evaluate.fit_all(train, params, dist, name, evaluate.SEED)
            X = pre.transform(valid)
            auc, mean_auc = cumulative_dynamic_auc(y_tr, y_va, m.predict_risk(X), np.array(HORIZONS))
            _, bs = brier_score(y_tr, y_va[keep], m.predict_survival(X[keep], times), times)
            curves[name] = bs
            models.append({"model": name, "label": evaluate.LABELS[name] if dist is None
                           else f"AFT {dist.replace('normal', '-normal').title()}",
                           "uno_c": evaluate.uno_c(y_tr, y_va, m.predict_risk(X), tau),
                           **{f"auc_{int(h)}m": float(a) for h, a in zip(HORIZONS, auc)},
                           "auc_mean": float(mean_auc), "ibs": float(integrated_brier_score(y_tr, y_va[keep], m.predict_survival(X[keep], times), times))})
    res = {"n_validation": len(valid), "n_brier": int(keep.sum()), "times": times.tolist(), "models": models,
           "brier_selected": curves[SELECTED].round(4).tolist(), "brier_km": bs_km.round(4).tolist(),
           "ibs_km": float(integrated_brier_score(y_tr, y_va[keep], km_curve, times))}
    plot(res, out_dir / "figures" / "brier_over_time.png")
    (out_dir / "time_dependent_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "time_dependent_metrics.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def plot(r: dict, path: Path):
    t = np.array(r["times"])
    sel, km = np.array(r["brier_selected"]), np.array(r["brier_km"])
    label = next(m["label"] for m in r["models"] if m["model"] == SELECTED)
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=evaluate.SURFACE)
    evaluate._style(ax)
    ax.plot(t, km, "--", color=evaluate.INK_MUTED, linewidth=2, label="Kaplan-Meier baseline (no model)")
    ax.plot(t, sel, color=evaluate.ACCENT, linewidth=2, label=f"Selected model ({label})")
    ax.set_xlabel("Months after degree completion", fontsize=9, color=evaluate.INK_MUTED)
    ax.set_ylabel("Brier score (lower = better)", fontsize=9, color=evaluate.INK_MUTED)
    ax.set_title("Prediction error over time (validation set)", fontsize=11, color=evaluate.INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=evaluate.SURFACE)
    plt.close(fig)


def to_markdown(r: dict) -> str:
    L = ["# Time-dependent AUC and Brier score over time (report only)", "",
         "_Generated by `python -m core.time_dependent_metrics` - do not edit by hand._", "",
         f"Supplementary metrics on the validation set ({r['n_validation']} graduates; {r['n_brier']} for the Brier "
         "score, whose censoring weights need follow-up within the training range). Every candidate fitted on the "
         "training set with the settings chosen in the main analysis; the test set is not used, and these metrics "
         "were not used to choose the model.", "",
         "| Model | AUC at 3 months | AUC at 6 months | AUC at 12 months | Mean AUC | Uno's C | IBS |",
         "|---|---|---|---|---|---|---|"]
    for m in r["models"]:
        L.append(f"| {m['label']} | {m['auc_3m']:.3f} | {m['auc_6m']:.3f} | {m['auc_12m']:.3f} | "
                 f"{m['auc_mean']:.3f} | {m['uno_c']:.3f} | {m['ibs']:.3f} |")
    L.append(f"| Kaplan-Meier baseline | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | {r['ibs_km']:.3f} |")
    sel, km, t = np.array(r["brier_selected"]), np.array(r["brier_km"]), np.array(r["times"])
    better = t[sel < km]
    L += ["", "AUC 0.5 = no better than chance, 1 = perfect. The Kaplan-Meier baseline predicts the same curve for "
          "everyone, so it cannot rank graduates (0.5 by definition).", "",
          f"Brier score over time (`figures/brier_over_time.png`): the selected model is below the baseline at "
          f"{len(better)} of {len(t)} monthly time points"
          + (f" ({int(better.min())}-{int(better.max())} months)" if len(better) else "") +
          f"; the largest gap is {100 * (km - sel).max():.1f} points at month {int(t[(km - sel).argmax()])}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = run()
    for m in r["models"]:
        print(f"{m['label']:20s} AUC3 {m['auc_3m']:.3f} AUC6 {m['auc_6m']:.3f} AUC12 {m['auc_12m']:.3f} "
              f"C {m['uno_c']:.3f} IBS {m['ibs']:.3f}")
    print("KM IBS", round(r["ibs_km"], 3))
