"""Model development and evaluation on a fixed train / validation / test split (core/splits.py) - NO data reuse.

  train       - every model is fitted here (preprocessing too), including the final model and its bootstrap refits
  validation  - hyperparameters are tuned and candidate models compared here (fit on train, score on validation)
  test        - locked; the selected model is evaluated on it exactly once (the reported result)
No graduate appears in more than one role, so there is no cross-validation.

Run:  python -m core.evaluate        (needs data/processed/graduates_train.csv from core.clean)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score

from core import preprocess, schema
from core.models import CURVE_TIMES, MODEL_NAMES, make_model
from core.models.aft import LABELS as AFT_LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
INFO_PATH = ROOT / "data" / "processed" / "dataset_info.json"

SEED = 20261006
REPEATS = 1                            # kept for call compatibility; there are no CV repeats (no data reuse)
CALIBRATION_HORIZONS = (3, 6, 12)
TAU_QUANTILE = 0.9                     # Uno's C / IBS truncated at the 90th percentile of training follow-up
SELECTION_C_MARGIN = 0.02
N_BOOTSTRAP = 200
N_BOOTSTRAP_DEEPSURV = 50
N_JOBS = -1

AFT_DISTS = ("weibull", "lognormal", "loglogistic")
CANDIDATES = ("cox_ph", "cox_elasticnet", "aft_best", "deepsurv")   # one per model family
TUNED = ("cox_ph", "cox_elasticnet", "deepsurv")
LABELS = {"cox_ph": "Cox PH", "cox_elasticnet": "Elastic Net Cox", "deepsurv": "DeepSurv",
          "aft_best": "AFT (best by AIC)", **{f"aft_{d}": AFT_LABELS[d] for d in AFT_DISTS}}
COMPLEXITY = {"cox_ph": 1, "aft_best": 1, "cox_elasticnet": 2, "deepsurv": 3,
              **{f"aft_{d}": 1 for d in AFT_DISTS}}

# Figure style (reference palette from the dataviz guidance: one accent hue, recessive ink)
ACCENT, ACCENT_LIGHT, INK, INK_MUTED, GRID, SURFACE = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def truncation_tau(train_time: np.ndarray) -> float:
    return float(min(schema.PREDICTION_HORIZON_MONTHS, np.quantile(train_time, TAU_QUANTILE)))


def uno_c(y_train, y_test, risk, tau) -> float:
    return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])


def harrell_c(y_test, risk) -> float:
    return float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])


def ibs(y_train, y_test, model, X_test, tau) -> float:
    # The censoring weights come from the training fold, so they are undefined beyond its longest follow-up.
    # Test graduates followed longer than that lie past tau anyway; they are left out of the Brier score only.
    keep = y_test["time"] < y_train["time"].max()
    y_test, X_test = y_test[keep], X_test[keep]
    if len(y_test) == 0:
        return float("nan")
    upper = min(tau, y_test["time"].max())
    times = np.arange(1.0, upper)
    if len(times) < 2:
        return float("nan")
    return float(integrated_brier_score(y_train, y_test, model.predict_survival(X_test, times), times))


def median_from_curve(surv: np.ndarray, times: np.ndarray = CURVE_TIMES) -> np.ndarray:
    """First time S(t) <= 0.5; NaN if the curve never reaches 0.5 within the grid."""
    below = surv <= 0.5
    return np.where(below.any(axis=1), times[np.argmax(below, axis=1)], np.nan)


def calibration_table(oof: pd.DataFrame, horizons=CALIBRATION_HORIZONS) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed P(employed by h), by tertile of predicted P(employed by h)."""
    rows = []
    for h in horizons:
        pred = 1 - oof[f"S_{h}"]
        tertile = pd.qcut(pred.rank(method="first"), 3, labels=["slowest third", "middle third", "fastest third"])
        for lab in tertile.cat.categories:
            sel = oof[tertile == lab]
            km = KaplanMeierFitter().fit(sel["duration_months"], sel["event"])
            observed = (1 - float(km.survival_function_at_times(h).iloc[0])
                        if sel["duration_months"].max() >= h else np.nan)
            rows.append({"horizon": h, "tertile": lab, "n": len(sel),
                         "predicted": float((1 - sel[f"S_{h}"]).mean()), "observed": observed})
    t = pd.DataFrame(rows)
    t["abs_error"] = (t["predicted"] - t["observed"]).abs()
    return t


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
def fit_model(X, time_, event, name: str, params: dict, seed: int = 0):
    params = dict(params)
    if name == "deepsurv":
        params["seed"] = seed
    return make_model(name, **params).fit(X, time_, event)


def fit_all(train: pd.DataFrame, params: dict, aft_dist: str | None, name: str, seed: int,
            features: dict | None = None):
    """Fit preprocessing + one model on `train`. `name` may be 'aft_best' (uses aft_dist)."""
    train = train.reset_index(drop=True)          # bootstrap resamples repeat row labels
    pre = preprocess.build_preprocessor(features).fit(train)
    X = pre.transform(train)
    t, e = preprocess.outcome(train)
    model_name = f"aft_{aft_dist}" if name == "aft_best" else name
    return pre, fit_model(X, t, e, model_name, params, seed)


# ---------------------------------------------------------------------------
# Fixed train / validation / test design (core/splits.py) - no graduate is reused across roles
# ---------------------------------------------------------------------------
N_EVAL_BOOT = 500                       # resamples of the EVALUATION set for intervals (no model is refitted)


def tune_holdout(train: pd.DataFrame, valid: pd.DataFrame, name: str, seed: int = SEED,
                 features: dict | None = None) -> tuple[dict, float]:
    """Grid search: fit every setting on `train`, score Uno's C on `valid`. Preprocessing fitted on train only."""
    grid = make_model(name).param_grid
    Xa, Xb, _ = preprocess.fit_transform_fold(train, valid, features)
    ya, yb = preprocess.sksurv_y(train), preprocess.sksurv_y(valid)
    tau = truncation_tau(ya["time"])
    if len(grid) == 1:
        return grid[0], float("nan")
    best, best_score = grid[0], -np.inf
    for params in grid:
        try:
            score = uno_c(ya, yb, fit_model(Xa, ya["time"], ya["event"], name, params, seed).predict_risk(Xb), tau)
        except Exception:
            score = -np.inf
        if score > best_score:
            best, best_score = params, score
    return best, float(best_score)


def holdout(train: pd.DataFrame, valid: pd.DataFrame, seed: int = SEED, features: dict | None = None,
            models_to_run: tuple | None = None, fixed_params: dict | None = None):
    """Fit every model on `train` and score it on `valid`.

    Hyperparameters are tuned on `valid` (fit on train, score on valid) unless `fixed_params` gives them; AFT
    distributions are chosen by AIC on train. Returns (metric rows, predictions on valid, params used).
    """
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    pre = preprocess.build_preprocessor(features).fit(train)
    X_tr, X_va = pre.transform(train), pre.transform(valid)
    y_tr, y_va = preprocess.sksurv_y(train), preprocess.sksurv_y(valid)
    t, e = y_tr["time"], y_tr["event"]
    tau = truncation_tau(t)

    models, params_used = {}, {}
    for name in TUNED:
        if models_to_run is not None and name not in models_to_run:
            continue
        params = (fixed_params or {}).get(name) if fixed_params else None
        if params is None:
            params, _ = tune_holdout(train, valid, name, seed, features)
        models[name] = fit_model(X_tr, t, e, name, params, seed)
        params_used[name] = params
    if models_to_run is None or any(m.startswith("aft") for m in models_to_run):
        aics = {}
        for d in AFT_DISTS:
            models[f"aft_{d}"] = fit_model(X_tr, t, e, f"aft_{d}", {}, seed)
            aics[d] = models[f"aft_{d}"].aic
        best_dist = min(aics, key=aics.get)
        models["aft_best"] = models[f"aft_{best_dist}"]
        params_used["aft_best"] = {"distribution": best_dist}

    rows, preds = [], []
    for name, m in models.items():
        risk = m.predict_risk(X_va)
        curve = m.predict_survival(X_va, CURVE_TIMES)
        s_h = m.predict_survival(X_va, np.array(CALIBRATION_HORIZONS, float))
        rows.append({"repeat": 0, "fold": 0, "model": name, "tau": tau, "n_eval": len(valid),
                     "uno_c": uno_c(y_tr, y_va, risk, tau), "harrell_c": harrell_c(y_va, risk),
                     "ibs": ibs(y_tr, y_va, m, X_va, tau), "params": json.dumps(params_used.get(name, {})),
                     **eval_intervals(y_tr, y_va, risk, tau)})
        block = pd.DataFrame({"repeat": 0, "fold": 0, "model": name, "respondent_id": valid["respondent_id"].to_numpy(),
                              "risk": risk, "median_months": median_from_curve(curve),
                              "duration_months": y_va["time"], "event": y_va["event"].astype(int)})
        for j, h in enumerate(CALIBRATION_HORIZONS):
            block[f"S_{h}"] = s_h[:, j]
        preds.append(block)
    return rows, pd.concat(preds, ignore_index=True), params_used


def eval_intervals(y_train, y_eval, risk, tau, n: int = N_EVAL_BOOT, seed: int = SEED) -> dict:
    """Bootstrap of the EVALUATION set only (the fitted model is unchanged): SE and 95% interval of Uno's /
    Harrell's C."""
    rng = np.random.default_rng(seed)
    uc, hc = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_eval), len(y_eval))
        if y_eval["event"][i].sum() < 2:
            continue
        try:
            uc.append(uno_c(y_train, y_eval[i], risk[i], tau))
            hc.append(harrell_c(y_eval[i], risk[i]))
        except ValueError:
            continue
    uc, hc = np.array(uc), np.array(hc)
    return {"uno_c_se": float(uc.std(ddof=1)), "uno_c_lo": float(np.percentile(uc, 2.5)),
            "uno_c_hi": float(np.percentile(uc, 97.5)), "harrell_c_se": float(hc.std(ddof=1))}


def evaluate_test(final: dict, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The ONE evaluation on the locked test set: the final model (fitted on train) scored on test."""
    pre, m = final["preprocessor"], final["model"]
    X_te = pre.transform(test)
    y_tr, y_te = preprocess.sksurv_y(train), preprocess.sksurv_y(test)
    tau = truncation_tau(y_tr["time"])
    risk = m.predict_risk(X_te)
    curve = m.predict_survival(X_te, CURVE_TIMES)
    s_h = m.predict_survival(X_te, np.array(CALIBRATION_HORIZONS, float))
    pred = pd.DataFrame({"repeat": 0, "fold": 0, "model": final["name"], "respondent_id": test["respondent_id"].to_numpy(),
                         "risk": risk, "median_months": median_from_curve(curve),
                         "duration_months": y_te["time"], "event": y_te["event"].astype(int)})
    for j, h in enumerate(CALIBRATION_HORIZONS):
        pred[f"S_{h}"] = s_h[:, j]
    cal = calibration_table(pred)
    metrics = {"n": len(test), "events": int(y_te["event"].sum()), "tau": tau,
               "uno_c": uno_c(y_tr, y_te, risk, tau), "harrell_c": harrell_c(y_te, risk), "ibs": ibs(y_tr, y_te, m, X_te, tau),
               **eval_intervals(y_tr, y_te, risk, tau),
               **{f"cal_err_{h}m": float(cal.loc[cal["horizon"] == h, "abs_error"].mean()) for h in CALIBRATION_HORIZONS}}
    metrics["cal_err_mean"] = float(np.nanmean([metrics[f"cal_err_{h}m"] for h in CALIBRATION_HORIZONS]))
    return metrics, pred


# ---------------------------------------------------------------------------
# Summaries and selection
# ---------------------------------------------------------------------------
def plain_language_metrics(uno_c: float, cal_errors: dict) -> list[str]:
    """Plain-language versions of the headline metrics (pairs out of 100; percentage points)."""
    from core import wording
    out = [wording.PLAIN_METRICS["ranking"].format(pairs=int(round(100 * uno_c)))]
    for h in sorted(cal_errors):
        out.append(wording.PLAIN_METRICS["calibration"].format(h=h, pts=f"{100 * cal_errors[h]:.0f}"))
    out.append(wording.PLAIN_METRICS["why_not_accuracy"])
    return out


def select_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Highest mean Uno's C; among candidates within SELECTION_C_MARGIN of it, prefer simpler, then better calibrated."""
    cand = comparison[comparison["candidate"]].copy()
    best_c = cand["uno_c_mean"].max()
    near = cand[cand["uno_c_mean"] >= best_c - SELECTION_C_MARGIN].sort_values(
        ["complexity", "cal_err_mean", "uno_c_mean"], ascending=[True, True, False])
    chosen = near.iloc[0]
    top = cand.loc[cand["uno_c_mean"].idxmax()]
    if chosen["model"] == top["model"]:
        reason = f"highest mean Uno's C ({top['uno_c_mean']:.3f})"
        if len(near) > 1:
            reason += "; it was also the simplest / best-calibrated of the models within the margin"
    else:
        reason = (f"{LABELS[top['model']]} had the highest mean Uno's C ({top['uno_c_mean']:.3f}); "
                  f"{LABELS[chosen['model']]} ({chosen['uno_c_mean']:.3f}) is within {SELECTION_C_MARGIN} and is "
                  f"simpler (complexity {chosen['complexity']} vs {top['complexity']}) or better calibrated "
                  f"(mean calibration error {chosen['cal_err_mean']:.3f} vs {top['cal_err_mean']:.3f})")
    return str(chosen["model"]), reason


# ---------------------------------------------------------------------------
# Final model, PH check, bootstrap
# ---------------------------------------------------------------------------
def fit_final(train: pd.DataFrame, name: str, params_used: dict) -> dict:
    """Fit the selected model on the TRAINING set only, with the hyperparameters chosen on validation (no re-tuning).
    AFT family: the distribution with the lowest AIC on train."""
    aic_table = None
    if name == "aft_best":
        pre = preprocess.build_preprocessor().fit(train)
        X = pre.transform(train)
        t, e = preprocess.outcome(train)
        aic_table = {d: fit_model(X, t, e, f"aft_{d}", {}).aic for d in AFT_DISTS}
        aft_dist, params = min(aic_table, key=aic_table.get), {}
    else:
        aft_dist, params = None, params_used.get(name, {})
    pre, model = fit_all(train, params, aft_dist, name, SEED)
    return {"name": name, "label": LABELS[name] if name != "aft_best" else AFT_LABELS[aft_dist],
            "model_name": f"aft_{aft_dist}" if aft_dist else name, "params": params, "aft_distribution": aft_dist,
            "aic_table": aic_table, "preprocessor": pre, "model": model}


def _bootstrap_one(df, b, name, params, aft_dist):
    idx = np.random.default_rng(SEED + 1000 + b).integers(0, len(df), len(df))
    try:
        pre, model = fit_all(df.iloc[idx], params, aft_dist, name, SEED + b)
        return {"b": b, "preprocessor": pre, "model": model}
    except Exception as exc:          # a resample can fail to converge; logged, not hidden
        return {"b": b, "error": repr(exc)}


def bootstrap(df: pd.DataFrame, final: dict, n_jobs: int = N_JOBS) -> tuple[list[dict], list[dict]]:
    n = N_BOOTSTRAP_DEEPSURV if final["name"] == "deepsurv" else N_BOOTSTRAP
    out = Parallel(n_jobs=n_jobs)(delayed(_bootstrap_one)(df, b, final["name"], final["params"],
                                                          final["aft_distribution"]) for b in range(n))
    return [o for o in out if "error" not in o], [o for o in out if "error" in o]


def ph_check(train: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """Cox PH on the training set with the penalizer chosen on validation; Schoenfeld test + scaled residuals."""
    pre, cox = fit_all(train, params, None, "cox_ph", SEED)
    X = pre.transform(train)
    t, e = preprocess.outcome(train)
    resid = cox.scaled_schoenfeld(X, t, e)
    resid["_T"] = train["duration_months"].loc[resid.index].to_numpy()   # residuals exist for event rows only
    return cox.ph_test(X, t, e), cox, resid


def training_summary(df: pd.DataFrame) -> dict:
    """Original-unit feature summary of the training data (used by explain / what-if)."""
    raw = preprocess.raw_feature_frame(df)
    area_cols = [c for c in preprocess.ITEM_BLOCKS if c.startswith("area_")]
    areas = preprocess.BlockMeans().transform(df)[area_cols]
    num = pd.concat([raw.select_dtypes("number"), areas], axis=1)
    return {
        "numeric": num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_dict("index"),
        "categorical": {c: raw[c].value_counts(dropna=True).to_dict() for c in preprocess.CATEGORICAL_FEATURES},
        "n": len(df), "events": int(df["event"].sum()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_calibration(cal: pd.DataFrame, name: str, path: Path):
    g = cal[cal["model"] == name].groupby(["horizon", "tertile"], observed=True)[["predicted", "observed"]]
    mean, sd = g.mean(), g.std(ddof=1)
    fig, axes = plt.subplots(1, len(CALIBRATION_HORIZONS), figsize=(10, 3.6), facecolor=SURFACE, sharey=True)
    for ax, h in zip(axes, CALIBRATION_HORIZONS):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1)
        m, s = mean.loc[h], sd.loc[h]
        ax.errorbar(m["predicted"], m["observed"], yerr=s["observed"], fmt="o", color=ACCENT, ecolor=ACCENT_LIGHT,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=2, capsize=0)
        for lab, row in m.iterrows():
            ax.annotate(lab.replace(" third", ""), (row["predicted"], row["observed"]), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color=INK_MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"By {h} months", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Predicted share employed", fontsize=9, color=INK_MUTED)
    axes[0].set_ylabel("Observed share employed (Kaplan-Meier)", fontsize=9, color=INK_MUTED)
    fig.suptitle(f"Calibration - {LABELS[name]} (out-of-fold, mean of {cal['repeat'].nunique()} repeats)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(comp: pd.DataFrame, selected: str, path: Path):
    comp = comp.sort_values("uno_c_mean")
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(comp))
    colors = [ACCENT if m == selected else INK_MUTED for m in comp["model"]]
    ax.errorbar(comp["uno_c_mean"], y, xerr=comp["uno_c_sd"], fmt="none", ecolor=GRID, elinewidth=2)
    ax.scatter(comp["uno_c_mean"], y, s=64, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    for yi, (_, r) in zip(y, comp.iterrows()):
        ax.annotate(f"{r['uno_c_mean']:.3f}" + ("  selected" if r["model"] == selected else ""),
                    (r["uno_c_mean"] + r["uno_c_sd"], yi), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=INK if r["model"] == selected else INK_MUTED)
    ax.axvline(0.5, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y, comp["label"], fontsize=9, color=INK)
    ax.set_xlabel("Uno's C-index on the validation set (+/- bootstrap SE)", fontsize=9, color=INK_MUTED)
    ax.set_title("Model discrimination (fit on train, scored on validation)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_km(df: pd.DataFrame, path: Path):
    km = KaplanMeierFitter().fit(df["duration_months"], df["event"])
    sf, ci = km.survival_function_, km.confidence_interval_
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(ci.index, 1 - ci.iloc[:, 0], 1 - ci.iloc[:, 1], step="post", color=ACCENT_LIGHT, alpha=0.5,
                    linewidth=0)
    ax.step(sf.index, 1 - sf.iloc[:, 0], where="post", color=ACCENT, linewidth=2)
    med = km.median_survival_time_
    if np.isfinite(med):
        ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"median {med:g} months", (med, 0.5), textcoords="offset points", xytext=(6, -14), fontsize=8,
                    color=INK_MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Months after degree completion", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Share who found an IT job", fontsize=9, color=INK_MUTED)
    ax.set_title(f"Time to first IT job, Kaplan-Meier (n = {len(df)}, {int(df['event'].sum())} events)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_schoenfeld(resid: pd.DataFrame, ph: pd.DataFrame, path: Path):
    feats = [c for c in resid.columns if c != "_T"]
    ncol = 5
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.8 * nrow), facecolor=SURFACE)
    rank = resid["_T"].rank(method="first").to_numpy()   # break month ties in order
    order = np.argsort(rank)
    for ax, f in zip(axes.ravel(), feats):
        _style(ax)
        r = resid[f].to_numpy()
        ax.scatter(rank, r, s=8, color=INK_MUTED, alpha=0.35, linewidths=0)
        k = max(5, len(r) // 10)
        smooth = pd.Series(r[order]).rolling(k, center=True, min_periods=3).mean()
        ax.plot(rank[order], smooth, color=ACCENT, linewidth=2)
        ax.axhline(0, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"{f}\np = {ph.loc[f, 'p']:.3f}", fontsize=9, color=INK, loc="left")
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    fig.suptitle("Scaled Schoenfeld residuals vs rank of event time (flat line = proportional hazards holds)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def summarise_validation(metrics: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per model from the single validation evaluation (uncertainty = evaluation-set bootstrap)."""
    cal_rows = []
    for name, g in preds.groupby("model"):
        t = calibration_table(g)
        t.insert(0, "model", name)
        t.insert(0, "repeat", 0)
        cal_rows.append(t)
    cal = pd.concat(cal_rows, ignore_index=True)
    rows = []
    for name in ["cox_ph", "cox_elasticnet", *[f"aft_{d}" for d in AFT_DISTS], "aft_best", "deepsurv"]:
        g = metrics[metrics["model"] == name]
        if g.empty:
            continue
        r = g.iloc[0]
        row = {"model": name, "label": LABELS[name], "candidate": name in CANDIDATES, "complexity": COMPLEXITY[name],
               "n_folds": 1, "n_eval": int(r["n_eval"]),
               "uno_c_mean": r["uno_c"], "uno_c_sd": r["uno_c_se"], "uno_c_lo": r["uno_c_lo"], "uno_c_hi": r["uno_c_hi"],
               "harrell_c_mean": r["harrell_c"], "harrell_c_sd": r["harrell_c_se"], "ibs_mean": r["ibs"],
               "ibs_sd": 0.0}
        ce = cal[cal["model"] == name]
        for h in CALIBRATION_HORIZONS:
            row[f"cal_err_{h}m_mean"] = float(ce.loc[ce["horizon"] == h, "abs_error"].mean())
            row[f"cal_err_{h}m_sd"] = 0.0
        row["cal_err_mean"] = float(np.nanmean([row[f"cal_err_{h}m_mean"] for h in CALIBRATION_HORIZONS]))
        row["tuned_params"] = r["params"]
        rows.append(row)
    return pd.DataFrame(rows), cal


def render_report(comp, selected, reason, final, test_m, ph, info, n_boot, boot_fail, runtime) -> str:
    lines = ["# Model comparison", "", "_Generated by `python -m core.evaluate` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        lines += ["> **SYNTHETIC DATA - DEMO ONLY.** All numbers below come from the fake dataset; they demonstrate "
                  "the pipeline and say nothing about real Sri Lankan graduates.", ""]
    lines += [f"Modelling graduates: **{info['modelling_n']}** ({info['modelling_events']} events), split once into "
              f"**train {info['training_n']}** ({info['events']} events, EPV {info['epv']}), **validation "
              f"{info['validation_n']}** ({info['validation_events']} events) and **test {info['test_n']}** "
              f"({info['test_events']} events).", "",
              "## Evaluation design (no data reuse)", "",
              "- **One role per graduate, assigned once** (stratified by event and institution type; `core/splits.py`). "
              "No cross-validation: no graduate is ever used in more than one role.",
              "- **Train:** the only data any model is fitted on, including the final model used by the app and its "
              "bootstrap refits. Preprocessing (imputation, scaling, encoding) is fitted on train only.",
              "- **Validation:** hyperparameters are tuned and the candidate models compared here (fit on train, score "
              "on validation). AFT distributions are chosen by AIC on train.",
              "- **Test (locked):** used once, after the model was chosen, to evaluate the selected model only. "
              "Nothing about the model was changed after seeing the test result.",
              f"- **Uno's C (primary)** and **integrated Brier score** are truncated at tau = the "
              f"{int(TAU_QUANTILE * 100)}th percentile of training follow-up. Harrell's C is secondary. Intervals come "
              f"from {N_EVAL_BOOT} bootstrap resamples of the evaluation set (no model is refitted).",
              f"- **Calibration** at {', '.join(map(str, CALIBRATION_HORIZONS))} months: tertiles of predicted "
              "P(employed by t), mean predicted vs Kaplan-Meier observed; error = mean absolute difference.",
              "- Classification metrics (accuracy, precision, recall, ROC-AUC) are deliberately not used.", "",
              "## Validation results (model comparison)", "",
              "| Model | Uno's C (95% CI) | Harrell's C | IBS (lower = better) | "
              + " | ".join(f"Cal. error {h}m" for h in CALIBRATION_HORIZONS) + " | Tuned params |",
              "|---|---|---|---|" + "---|" * len(CALIBRATION_HORIZONS) + "---|"]
    for _, r in comp.iterrows():
        name = f"**{r['label']}**" if r["model"] == selected else r["label"]
        lines.append(f"| {name} | {r['uno_c_mean']:.3f} ({r['uno_c_lo']:.3f}-{r['uno_c_hi']:.3f}) | "
                     f"{r['harrell_c_mean']:.3f} | {r['ibs_mean']:.3f} | "
                     + " | ".join(f"{r[f'cal_err_{h}m_mean']:.3f}" for h in CALIBRATION_HORIZONS)
                     + f" | `{r['tuned_params']}` |")
    lines += ["", f"The validation set has {int(comp['n_eval'].iloc[0])} graduates, so differences between models "
              "smaller than the interval widths should not be interpreted.", "",
              "## Model selection", "",
              f"**Rule:** choose the candidate (Cox PH, Elastic Net Cox, AFT best-by-AIC, DeepSurv) with the highest "
              f"validation Uno's C. If another candidate is within {SELECTION_C_MARGIN} of it, prefer the simpler model "
              "(Cox PH = AFT < Elastic Net Cox < DeepSurv), then the better calibrated.", "",
              f"**Selected: {final['label']}** - {reason}.", "",
              f"Final model fitted on the **{info['training_n']} training graduates only**, with the hyperparameters "
              f"chosen on validation: `{json.dumps(final['params'])}`"
              + (f", distribution `{final['aft_distribution']}`" if final["aft_distribution"] else "") + ".", "",
              "## Final evaluation on the locked test set (reported result)", "",
              f"| Metric | Value |", "|---|---|",
              f"| Graduates / events | {test_m['n']} / {test_m['events']} |",
              f"| **Uno's C** | **{test_m['uno_c']:.3f}** (95% CI {test_m['uno_c_lo']:.3f}-{test_m['uno_c_hi']:.3f}) |",
              f"| Harrell's C | {test_m['harrell_c']:.3f} |",
              f"| Integrated Brier score | {test_m['ibs']:.3f} |"]
    lines += [f"| Calibration error at {h} months | {test_m[f'cal_err_{h}m']:.3f} |" for h in CALIBRATION_HORIZONS]
    plain = plain_language_metrics(test_m["uno_c"], {h: test_m[f"cal_err_{h}m"] for h in CALIBRATION_HORIZONS})
    lines += ["", "## In plain terms (test set)", ""] + [f"- {x}" for x in plain] + [""]
    lines += [f"**Bootstrap (prediction uncertainty):** {n_boot} resamples of the training set refitted with the final "
              "hyperparameters" + (f" ({len(boot_fail)} failed to converge and were dropped)" if boot_fail else "") + ".",
              "", "## Proportional-hazards check (Cox PH on the training set, scaled Schoenfeld residuals)", "",
              "| Feature | Test statistic | p |", "|---|---|---|"]
    lines += [f"| `{f}` | {r['test_statistic']:.2f} | {r['p']:.3f} |" for f, r in ph.iterrows()]
    viol = ph.index[ph["p"] < 0.05].tolist()
    lines += ["", (f"Possible PH violation (p < 0.05): {', '.join(f'`{v}`' for v in viol)}. With "
                   f"{len(ph)} tests, about {0.05 * len(ph):.1f} false positives are expected at 5%; inspect "
                   "`figures/schoenfeld_cox_ph.png`. AFT models do not assume proportional hazards."
                   if viol else "No feature shows evidence against proportional hazards at p < 0.05."), "",
              "## Figures", "",
              "- `figures/model_comparison.png` - validation Uno's C by model",
              "- `figures/km_overall.png` - Kaplan-Meier time to first IT job (all modelling graduates, descriptive)",
              "- `figures/calibration_<model>.png` - validation calibration; `figures/calibration_test.png` - test set",
              "- `figures/schoenfeld_cox_ph.png` - PH diagnostics", "",
              f"_Runtime: {runtime / 60:.1f} minutes._", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_roles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from core import splits
    return tuple(splits.load(r) for r in splits.ROLES)


def run(repeats: int = REPEATS) -> dict:          # `repeats` kept for call compatibility; not used
    start = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    train, valid, test = load_roles()
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

    print(f"Train {len(train)} / validation {len(valid)} / test {len(test)} - fitting on train, comparing on "
          "validation ...", flush=True)
    rows, vpred, params_used = holdout(train, valid)
    metrics = pd.DataFrame(rows)
    comp, cal = summarise_validation(metrics, vpred)
    selected, reason = select_model(comp)
    print(f"Selected {selected}: {reason}", flush=True)

    final = fit_final(train, selected, params_used)
    test_m, tpred = evaluate_test(final, train, test)        # the single use of the test set
    print(f"Test set: Uno's C {test_m['uno_c']:.3f} ({test_m['uno_c_lo']:.3f}-{test_m['uno_c_hi']:.3f})", flush=True)
    ph, _, resid = ph_check(train, params_used.get("cox_ph", {}))
    print("Bootstrap ...", flush=True)
    boots, boot_fail = bootstrap(train, final)

    bundle = {k: final[k] for k in ("name", "label", "model_name", "params", "aft_distribution", "preprocessor",
                                    "model")}
    bundle.update({"feature_names": preprocess.feature_names(final["preprocessor"]),
                   "training_summary": training_summary(train), "dataset_info": info,
                   "cv_metrics": comp.set_index("model").loc[selected].drop(["label"]).to_dict(),
                   "test_metrics": test_m, "selection_reason": reason, "curve_times": CURVE_TIMES,
                   "feature_set": schema.FEATURE_SET, "features": dict(schema.MODEL_FEATURES),
                   "max_follow_up_months": float(train["duration_months"].max()),
                   "X_mean": final["preprocessor"].transform(train).mean(),
                   "background": final["preprocessor"].transform(train).sample(min(100, len(train)), random_state=SEED),
                   "last_event_months": float(train.loc[train["event"] == 1, "duration_months"].max()),
                   "at_risk_by_month": {m: int((train["duration_months"] >= m).sum())
                                        for m in range(schema.PREDICTION_HORIZON_MONTHS + 1)}})
    joblib.dump(bundle, ARTIFACTS / "final_model.pkl")
    joblib.dump(boots, ARTIFACTS / "bootstrap_models.pkl")

    for old in ("cv_fold_metrics.csv",):
        (ARTIFACTS / old).unlink(missing_ok=True)
    metrics.to_csv(ARTIFACTS / "validation_metrics.csv", index=False)
    vpred.to_csv(ARTIFACTS / "validation_predictions.csv", index=False)
    tpred.to_csv(ARTIFACTS / "oof_predictions.csv", index=False)          # held-out = locked test set
    cal_test = calibration_table(tpred)
    cal_test.insert(0, "model", selected)
    cal_test.insert(0, "repeat", 0)
    cal_test.to_csv(ARTIFACTS / "calibration_table.csv", index=False)
    comp.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    ph.to_csv(ARTIFACTS / "ph_test.csv")
    (ARTIFACTS / "test_metrics.json").write_text(json.dumps({"model": selected, **test_m}, indent=2), encoding="utf-8")

    for name in comp["model"]:
        plot_calibration(cal, name, FIGURES / f"calibration_{name}.png")
    plot_calibration(cal_test, selected, FIGURES / "calibration_test.png")
    plot_comparison(comp, selected, FIGURES / "model_comparison.png")
    plot_km(pd.concat([train, valid, test]), FIGURES / "km_overall.png")
    plot_schoenfeld(resid, ph, FIGURES / "schoenfeld_cox_ph.png")

    runtime = time.time() - start
    (ARTIFACTS / "model_comparison.md").write_text(
        render_report(comp, selected, reason, final, test_m, ph, info, len(boots), boot_fail, runtime), encoding="utf-8")
    summary = {"selected": selected, "label": final["label"], "params": final["params"],
               "uno_c": round(float(test_m["uno_c"]), 3),
               "uno_c_validation": round(float(comp.set_index("model").loc[selected, "uno_c_mean"]), 3),
               "uno_c_test_ci": [round(test_m["uno_c_lo"], 3), round(test_m["uno_c_hi"], 3)],
               "bootstrap_models": len(boots), "bootstrap_failures": len(boot_fail),
               "runtime_min": round(runtime / 60, 1)}
    (ARTIFACTS / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
