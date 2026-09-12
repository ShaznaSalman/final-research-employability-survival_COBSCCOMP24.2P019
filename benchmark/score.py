"""Score recorded LLM answers with the SAME metrics as the survival models, on the SAME held-out profiles.

Reads benchmark/responses.csv (filled in by hand), benchmark/profile_key.csv, the selected model's
predictions on the locked test set and the test outcomes (censoring weights from the training set). Writes:
  artifacts/llm_benchmark.md        comparison report
  artifacts/llm_benchmark.csv       per (LLM, run) metrics
  artifacts/llm_unparseable.csv     every answer that could not be turned into a number, with the reason

Metrics
  - Uno's C (primary) and Harrell's C: ranking by predicted months (shorter = higher risk of finding work)
  - Calibration at 6 months by tertile: share predicted to be employed by 6 months vs Kaplan-Meier observed.
    The LLM gives a point estimate, so its "predicted share" is the share of answers <= 6 months; the model
    is shown both with its probabilities and with the same point-estimate rule (median <= 6).
  - Consistency across runs: within-profile SD, % identical answers, mean Spearman correlation between runs.
  - Paired bootstrap of the C-index difference (model - LLM) on the profiles both scored.
  - Prompt variants (benchmark/export_prompts.py): each LLM x variant is scored separately; every other variant
    is compared with `standard` on the same profiles (paired bootstrap).
  - Range answers (variant `range`): share of graduates whose time fell inside the stated 80% range (events;
    still-searching graduates count only when their follow-up already passed the upper end) and mean width.

Run:  python -m benchmark.score
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from benchmark.export_prompts import DEFAULT_VARIANT, StaleBenchmarkError, assert_fresh  # noqa: F401
from core import evaluate, preprocess

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
ARTIFACTS = ROOT / "artifacts"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"
INFO_PATH = ROOT / "data" / "processed" / "dataset_info.json"

HORIZON = 6
N_BOOT = 1000
SEED = 20261006
MAX_PLAUSIBLE_MONTHS = 120

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"twenty-four": 24, "twenty four": 24, "eighteen": 18, "a": 1, "an": 1, "half a": 0.5})
NUM = r"(\d+(?:\.\d+)?)"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_months(text) -> tuple[float | None, str]:
    """Return (months, note). months is None when the answer cannot be used; note explains why or how."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        return None, "empty response"
    t = str(text).strip().lower().replace("–", "-").replace("—", "-").replace(",", "")
    for word, val in sorted(NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{word}\b(?=\s*(years?|months?|weeks?|yrs?|mos?)\b)", str(val), t)
    unit = 1.0
    if re.search(r"\b(years?|yrs?)\b", t) and not re.search(r"\bmonths?\b", t):
        unit = 12.0
    elif re.search(r"\bweeks?\b", t) and not re.search(r"\bmonths?\b", t):
        unit = 1 / 4.345
    rng = re.search(rf"{NUM}\s*(?:-|to)\s*{NUM}", t)
    nums = re.findall(NUM, t)
    if rng:
        value, note = (float(rng.group(1)) + float(rng.group(2))) / 2, "range given - midpoint used"
    elif nums:
        value = float(nums[0])
        note = "several numbers - first used" if len(set(nums)) > 1 else ""
    else:
        return None, "no number found"
    value *= unit
    if unit != 1.0:
        note = (note + "; " if note else "") + ("converted from years" if unit == 12 else "converted from weeks")
    if value < 0 or value > MAX_PLAUSIBLE_MONTHS:
        return None, f"implausible value ({value:g} months)"
    return value, note


RANGE_RE = re.compile(rf"{NUM}\s*\(\s*{NUM}\s*(?:-|to)\s*{NUM}\s*\)")


def parse_range(text) -> tuple[float | None, float | None, float | None, str]:
    """'BEST (LOW-HIGH)' -> (best, low, high, note). Falls back to parse_months (no range) for other formats."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    m = RANGE_RE.search(t.replace("–", "-").replace("—", "-"))
    if m:
        best, lo, hi = (float(g) for g in m.groups())
        if lo > hi:
            lo, hi = hi, lo
        if not (0 <= lo <= hi <= MAX_PLAUSIBLE_MONTHS) or not (0 <= best <= MAX_PLAUSIBLE_MONTHS):
            return None, None, None, "implausible range"
        return best, lo, hi, "" if lo <= best <= hi else "best estimate outside its own range"
    value, note = parse_months(text)
    return value, None, None, (note + "; " if note else "") + "no range in BEST (LOW-HIGH) format"


def range_coverage(d: pd.DataFrame) -> dict:
    """d: one row per answer with duration_months, event, low, high."""
    d = d.dropna(subset=["low", "high"])
    ev = d[d["event"] == 1]
    inside_ev = (ev["duration_months"] >= ev["low"]) & (ev["duration_months"] <= ev["high"])
    cens_known = d[(d["event"] == 0) & (d["duration_months"] > d["high"])]       # certainly outside
    known = len(ev) + len(cens_known)
    return {"answers_with_range": len(d), "coverage_events": float(inside_ev.mean()) if len(ev) else np.nan,
            "coverage_known": float(inside_ev.sum() / known) if known else np.nan,
            "mean_width": float((d["high"] - d["low"]).mean()) if len(d) else np.nan}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def c_indices(y_all, df: pd.DataFrame, risk: np.ndarray, tau: float) -> tuple[float, float]:
    y = preprocess.sksurv_y(df)
    if y["event"].sum() < 2:
        return np.nan, np.nan
    return evaluate.uno_c(y_all, y, risk, tau), evaluate.harrell_c(y, risk)


def calibration_6m(df: pd.DataFrame, predicted_employed: np.ndarray, sort_key: np.ndarray) -> pd.DataFrame:
    """Tertiles of `sort_key` (higher = predicted faster); predicted vs KM-observed share employed by 6 months."""
    tert = pd.qcut(pd.Series(sort_key).rank(method="first"), 3, labels=["slowest", "middle", "fastest"])
    rows = []
    for lab in tert.cat.categories:
        m = (tert == lab).to_numpy()
        sub = df[m]
        obs = np.nan
        if sub["duration_months"].max() >= HORIZON:
            km = KaplanMeierFitter().fit(sub["duration_months"], sub["event"])
            obs = 1 - float(km.survival_function_at_times(HORIZON).iloc[0])
        rows.append({"tertile": lab, "n": int(m.sum()), "predicted": float(predicted_employed[m].mean()),
                     "observed": obs})
    t = pd.DataFrame(rows)
    t["abs_error"] = (t["predicted"] - t["observed"]).abs()
    return t


def consistency(wide: pd.DataFrame) -> dict:
    """wide: rows = profiles, columns = runs, values = months."""
    complete = wide.dropna()
    out = {"runs": wide.shape[1], "profiles_all_runs": len(complete)}
    if wide.shape[1] < 2 or len(complete) < 3:
        return out | {"mean_within_profile_sd": None, "pct_identical": None, "mean_spearman": None}
    rhos = [complete[a].corr(complete[b], method="spearman") for a, b in combinations(complete.columns, 2)]
    return out | {"mean_within_profile_sd": round(float(complete.std(axis=1, ddof=1).mean()), 2),
                  "pct_identical": round(100 * float((complete.nunique(axis=1) == 1).mean()), 1),
                  "mean_spearman": round(float(np.nanmean(rhos)), 3)}


def paired_bootstrap(y_all, df, risk_model, risk_llm, tau, seed=SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(df), len(df))
        a, _ = c_indices(y_all, df.iloc[i], risk_model[i], tau)
        b, _ = c_indices(y_all, df.iloc[i], risk_llm[i], tau)
        diffs.append(a - b)
    d = np.array(diffs)
    return float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(bench: Path = BENCH, artifacts: Path = ARTIFACTS, train_path: Path = TRAIN_PATH,
          info_path: Path = INFO_PATH) -> dict:
    assert_fresh(bench, train_path)
    key = pd.read_csv(bench / "profile_key.csv")
    responses = pd.read_csv(bench / "responses.csv", dtype={"raw_response": "string"})
    if "variant" not in responses:                                   # files from before prompt variants
        responses["variant"] = DEFAULT_VARIANT
    responses["variant"] = responses["variant"].fillna(DEFAULT_VARIANT).replace("", DEFAULT_VARIANT)
    train = pd.read_csv(train_path)
    selected = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(artifacts / "oof_predictions.csv")
    oof = oof[(oof["model"] == selected["selected"]) & (oof["repeat"] == 0)]
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"is_synthetic": True}

    from core import splits
    # censoring weights: training set only (the split file sits next to the training file)
    y_all = preprocess.sksurv_y(splits.load("train", train_path, train_path.parent / splits.SPLITS_PATH.name))
    tau = evaluate.truncation_tau(y_all["time"])
    base = key.merge(train[["respondent_id", "duration_months", "event"]], on="respondent_id") \
        .merge(oof[["respondent_id", "risk", f"S_{HORIZON}", "median_months"]], on="respondent_id")

    # Model on the full benchmark set
    model_uno, model_harrell = c_indices(y_all, base, base["risk"].to_numpy(), tau)
    model_cal = calibration_6m(base, 1 - base[f"S_{HORIZON}"].to_numpy(), 1 - base[f"S_{HORIZON}"].to_numpy())
    med = base["median_months"].fillna(np.inf).to_numpy()
    model_cal_point = calibration_6m(base, (med <= HORIZON).astype(float), -med)

    # Parse responses
    answered = responses[responses["raw_response"].fillna("").str.strip() != ""].copy()
    parsed = [parse_range(t) if v == "range" else (*parse_months(t)[:1], None, None, parse_months(t)[1])
              for t, v in zip(answered["raw_response"], answered["variant"])]
    answered["months"] = [p[0] for p in parsed]
    answered["low"] = [p[1] for p in parsed]
    answered["high"] = [p[2] for p in parsed]
    answered["parse_note"] = [p[3] for p in parsed]
    unparseable = answered[answered["months"].isna()]
    unparseable[["profile_id", "variant", "run_number", "llm_name", "model_version", "raw_response", "parse_note"]] \
        .to_csv(artifacts / "llm_unparseable.csv", index=False)
    ok = answered[answered["months"].notna()].copy()
    ok["llm_base"] = (ok["llm_name"].fillna("unnamed LLM").astype(str) + " "
                      + ok["model_version"].fillna("").astype(str)).str.strip()
    ok["llm"] = np.where(ok["variant"] == DEFAULT_VARIANT, ok["llm_base"], ok["llm_base"] + " [" + ok["variant"] + "]")

    run_rows, llm_summaries, agg_by = [], [], {}
    for llm, g in ok.groupby("llm"):
        for run, r in g.groupby("run_number"):
            d = base.merge(r[["profile_id", "months"]], on="profile_id")
            uno, har = c_indices(y_all, d, -d["months"].to_numpy(), tau)
            cal = calibration_6m(d, (d["months"] <= HORIZON).to_numpy(float), -d["months"].to_numpy())
            run_rows.append({"llm": llm, "run": int(run), "n": len(d), "uno_c": uno, "harrell_c": har,
                             "cal_err_6m": float(cal["abs_error"].mean())})
        wide = g.pivot_table(index="profile_id", columns="run_number", values="months", aggfunc="first")
        agg = base.merge(wide.mean(axis=1).rename("months").reset_index(), on="profile_id")
        agg_by[(g["llm_base"].iloc[0], g["variant"].iloc[0])] = agg
        uno, har = c_indices(y_all, agg, -agg["months"].to_numpy(), tau)
        cal = calibration_6m(agg, (agg["months"] <= HORIZON).to_numpy(float), -agg["months"].to_numpy())
        m_uno, _ = c_indices(y_all, agg, agg["risk"].to_numpy(), tau)
        diff = paired_bootstrap(y_all, agg, agg["risk"].to_numpy(), -agg["months"].to_numpy(), tau) \
            if len(agg) >= 10 else (np.nan, np.nan, np.nan)
        coverage = range_coverage(base.merge(g[["profile_id", "low", "high"]], on="profile_id")) \
            if g["variant"].iloc[0] == "range" else None
        llm_summaries.append({"llm": llm, "llm_base": g["llm_base"].iloc[0], "variant": g["variant"].iloc[0],
                              "coverage": coverage, "profiles": len(agg), "uno_c": uno, "harrell_c": har,
                              "cal_err_6m": float(cal["abs_error"].mean()), "calibration": cal,
                              "model_uno_same_profiles": m_uno, "diff": diff, "consistency": consistency(wide)})

    variant_rows = []
    for (llm_base, variant), agg in agg_by.items():
        ref = agg_by.get((llm_base, DEFAULT_VARIANT))
        if variant == DEFAULT_VARIANT or ref is None:
            continue
        both = agg.merge(ref[["profile_id", "months"]].rename(columns={"months": "months_std"}), on="profile_id")
        d = paired_bootstrap(y_all, both, -both["months"].to_numpy(), -both["months_std"].to_numpy(), tau) \
            if len(both) >= 10 else (np.nan, np.nan, np.nan)
        variant_rows.append({"llm": llm_base, "variant": variant, "profiles": len(both), "diff": d})

    runs = pd.DataFrame(run_rows, columns=["llm", "run", "n", "uno_c", "harrell_c", "cal_err_6m"])
    runs.to_csv(artifacts / "llm_benchmark.csv", index=False)
    md = render(base, selected, info, model_uno, model_harrell, model_cal, model_cal_point, runs, llm_summaries,
                len(responses), len(answered), unparseable, variant_rows)
    (artifacts / "llm_benchmark.md").write_text(md, encoding="utf-8")
    return {"profiles": len(base), "responses_answered": len(answered), "unparseable": len(unparseable),
            "llms": [s["llm"] for s in llm_summaries], "model_uno_c": round(model_uno, 3),
            "variant_comparisons": len(variant_rows)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _cal_table(cal: pd.DataFrame) -> list[str]:
    rows = ["| Tertile | n | Predicted employed by 6m | Observed (KM) |", "|---|---|---|---|"]
    return rows + [f"| {r['tertile']} | {r['n']} | {r['predicted']:.2f} | {r['observed']:.2f} |"
                   for _, r in cal.iterrows()]


def render(base, selected, info, m_uno, m_har, m_cal, m_cal_point, runs, llms, n_rows, n_answered, unparseable,
           variant_rows=()):
    L = ["# LLM benchmark", "", "_Generated by `python -m benchmark.score` - do not edit by hand._", ""]
    if info.get("is_synthetic", True):
        L += ["> **SYNTHETIC DATA - DEMO ONLY.** Outcomes are fake; a comparison on this data only demonstrates "
              "the scoring pipeline.", ""]
    L += [f"Benchmark set: **{len(base)} held-out graduates** ({int(base['event'].sum())} found an IT job, "
          f"{int(len(base) - base['event'].sum())} still searching), drawn from the locked test set (the "
          f"selected model, **{selected['label']}**, was fitted on the training set only and chosen on validation). "
          "Neither the model nor the LLM saw these graduates.",
          "", "Prompt: `benchmark/prompt_template.txt`. Responses recorded by hand in `benchmark/responses.csv`.", "",
          "## Selected model on the benchmark set", "",
          f"- Uno's C **{m_uno:.3f}**, Harrell's C {m_har:.3f}",
          f"- Calibration error at 6 months: {m_cal['abs_error'].mean():.3f} (probabilities), "
          f"{m_cal_point['abs_error'].mean():.3f} (point-estimate rule, comparable with the LLM)", ""]
    L += _cal_table(m_cal) + [""]
    L += ["## Responses", "", f"- Rows in responses.csv: {n_rows}; answered: {n_answered}; "
          f"unparseable: {len(unparseable)} (listed in `artifacts/llm_unparseable.csv`)."]
    if len(unparseable):
        L += [f"  - {r}: {n}" for r, n in unparseable["parse_note"].value_counts().items()]
    L.append("")
    if not llms:
        L += ["**No LLM responses recorded yet.** Paste each prompt from `benchmark/prompts.csv` into the LLM, "
              "record the answer in `benchmark/responses.csv`, then re-run `python -m benchmark.score`.", ""]
        return "\n".join(L)

    L += ["## Comparison (same profiles)", "",
          "| Predictor | Profiles | Uno's C | Harrell's C | Cal. error 6m | C difference model - LLM (95% CI) |",
          "|---|---|---|---|---|---|"]
    for s in llms:
        d = s["diff"]
        ci = "-" if np.isnan(d[0]) else f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
        L.append(f"| {s['llm']} (mean of runs) | {s['profiles']} | {s['uno_c']:.3f} | {s['harrell_c']:.3f} | "
                 f"{s['cal_err_6m']:.3f} | {ci} |")
        L.append(f"| {selected['label']} (same profiles) | {s['profiles']} | {s['model_uno_same_profiles']:.3f} | - "
                 f"| {m_cal_point['abs_error'].mean():.3f} | - |")
    L += ["", "A positive difference means the survival model ranked graduates better than the LLM. If the 95% "
          "interval includes 0, the data cannot distinguish them.", "",
          "## Per run", "", "| LLM | Run | Profiles | Uno's C | Harrell's C | Cal. error 6m |", "|---|---|---|---|---|---|"]
    L += [f"| {r['llm']} | {r['run']} | {r['n']} | {r['uno_c']:.3f} | {r['harrell_c']:.3f} | {r['cal_err_6m']:.3f} |"
          for _, r in runs.iterrows()]
    L += ["", "## Consistency across runs", "",
          "| LLM | Runs | Profiles in all runs | Mean within-profile SD (months) | % identical answers | "
          "Mean Spearman between runs |", "|---|---|---|---|---|---|"]
    for s in llms:
        c = s["consistency"]
        L.append(f"| {s['llm']} | {c['runs']} | {c['profiles_all_runs']} | {c['mean_within_profile_sd']} | "
                 f"{c['pct_identical']} | {c['mean_spearman']} |")
    if variant_rows:
        L += ["", "## Prompt variants vs the standard prompt (same LLM, same profiles)", "",
              "| LLM | Variant | Profiles | Uno's C difference variant - standard (95% CI) |", "|---|---|---|---|"]
        for v in variant_rows:
            d = v["diff"]
            ci = "-" if np.isnan(d[0]) else f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
            L.append(f"| {v['llm']} | {v['variant']} | {v['profiles']} | {ci} |")
        L += ["", "A positive difference means the variant ranked graduates better than the standard prompt."]
    ranged = [s for s in llms if s["coverage"]]
    if ranged:
        L += ["", "## 80% ranges (variant `range`)", "",
              "| LLM | Answers with a range | Coverage (graduates who found work) | Coverage (all with a known "
              "result) | Mean width (months) |", "|---|---|---|---|---|"]
        for s in ranged:
            c = s["coverage"]
            fmt = lambda v: "-" if v is None or np.isnan(v) else f"{v:.0%}"  # noqa: E731
            width = "-" if np.isnan(c["mean_width"]) else f"{c['mean_width']:.1f}"
            L.append(f"| {s['llm_base']} | {c['answers_with_range']} | {fmt(c['coverage_events'])} | "
                     f"{fmt(c['coverage_known'])} | {width} |")
        L += ["", "Well-calibrated 80% ranges contain the true time about 80% of the time. Still-searching "
              "graduates count only when their follow-up already passed the upper end (certainly outside)."]
    for s in llms:
        L += ["", f"### Calibration at 6 months - {s['llm']} (mean of runs)", ""] + _cal_table(s["calibration"])
    L += ["", "## Notes", "",
          "- The LLM returns one number, not a probability; its 6-month calibration treats an answer <= 6 as "
          "'predicted employed by 6 months'. The model is shown with the same rule for a like-for-like comparison.",
          "- Uno's C uses the full training set to estimate the censoring distribution and is truncated at the "
          "same tau as in model evaluation.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    print(json.dumps(score(), indent=2))
