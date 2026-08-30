"""One command that regenerates every artifact from whatever is in data/raw/.

  python -m core.run_all                    real data if present, otherwise the *_SYNTHETIC files
  python -m core.run_all --synthetic        (re)generate the synthetic data first
  python -m core.run_all --quick            1 CV repeat instead of 3 (for a fast check; not for the thesis)

Steps: data dictionary -> clean (+ fixed train/validation/test split) -> fit on train, compare on validation,
       select, final fit on train, ONE evaluation on the locked test set, bootstrap -> fairness
       -> reference.db (market + peers) -> job-market study -> questionnaire study -> LLM benchmark (export prompts if none yet, then score)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core import clean, evaluate, fairness, feature_sets, market, schema, synthetic

ROOT = Path(__file__).resolve().parents[1]


def _step(name: str, fn):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    result = fn()
    print(f"    done in {time.time() - start:.0f}s", flush=True)
    return result


def run(make_synthetic: bool = False, quick: bool = False) -> dict:
    summary: dict = {}
    if make_synthetic:
        summary["synthetic"] = _step("Generate synthetic data", synthetic.write_all)["files"]

    def dictionary():
        path = ROOT / "docs" / "data_dictionary.md"
        path.write_text(schema.render_data_dictionary(), encoding="utf-8")
        return str(path)

    _step("Data dictionary", dictionary)
    info = _step("Clean + build outcomes", clean.run)
    print(f"    sources: {info['sources']}  synthetic: {info['is_synthetic']}")
    print(f"    training N {info['training_n']}, events {info['events']}, censored {info['censored']}, "
          f"EPV {info['epv']}" + ("   <-- below 10, see docs/data_cleaning_log.md"
                                  if info["epv"] < schema.EPV_WARNING_THRESHOLD else ""))
    summary["data"] = info

    summary["model"] = _step("Train / validation / test: selection, final fit, test evaluation, bootstrap",
                             lambda: evaluate.run(repeats=1 if quick else evaluate.REPEATS))
    print(f"    selected {summary['model']['label']} (Uno's C {summary['model']['uno_c']})")
    summary["feature_sets"] = _step("Feature-set comparison (core vs extended, report only)",
                                    lambda: feature_sets.run(repeats=1 if quick else evaluate.REPEATS))
    fs = summary["feature_sets"]
    if fs["recommendation"] != fs["in_use"]:
        print(f"    NOTE: the pre-registered rule recommends the '{fs['recommendation']}' feature set but "
              f"'{fs['in_use']}' is in use - see artifacts/feature_set_comparison.md")
    summary["fairness"] = _step("Fairness check", fairness.run)

    def robustness():
        from core import clustering, competing, sensitivity, temporal
        return {"sensitivity": sensitivity.run(repeats=1 if quick else evaluate.REPEATS),
                "temporal": temporal.run(), "clustering": clustering.run(), "competing": competing.run()}

    summary["robustness"] = _step("Secondary analyses (sensitivity, temporal, clustering, competing risks)",
                                  robustness)
    summary["reference_db"] = _step("Reference database (market + peers)", market.build_reference_db)

    def job_market():
        from core import market_study
        r = market_study.run()
        return {"n_ads": r["n_ads"], "n_companies": r["n_companies"], "roles": r["roles"]}

    summary["job_market"] = _step("Job-market study (job ads)", job_market)

    def questionnaire():
        from core import survey_study
        r = survey_study.run()
        return {"repeated": {c["repeat"]: c["verdict"] for c in r["repeated"]}}

    summary["questionnaire"] = _step("Questionnaire study (non-model items)", questionnaire)

    def thesis():
        from core import thesis_results
        r = thesis_results.run()
        return {"model_checks": r["model_checks"], "alignment_summary": r["alignment_summary"]}

    summary["thesis_results"] = _step("Thesis results (RQ1 log-rank, time ratios, RQ2 alignment)", thesis)

    def benchmark():
        from benchmark import export_prompts, score
        bench = ROOT / "benchmark"
        if not (bench / "profile_key.csv").exists():
            export_prompts.export()
        try:
            return score.score()
        except score.StaleBenchmarkError as e:
            print(f"    SKIPPED: {e}")
            return {"skipped": str(e)}

    summary["benchmark"] = _step("LLM benchmark (score recorded responses)", benchmark)

    print("\nAll artifacts regenerated:")
    for p in ("docs/data_dictionary.md", "docs/data_cleaning_log.md", "artifacts/model_comparison.md",
              "artifacts/feature_set_comparison.md", "artifacts/fairness_report.md", "artifacts/llm_benchmark.md",
              "artifacts/sensitivity.md", "artifacts/temporal_validation.md", "artifacts/institution_clustering.md",
              "artifacts/competing_risks.md", "artifacts/job_market_study.md", "artifacts/questionnaire_study.md", "artifacts/thesis_results.md",
              "artifacts/figures/",
              "artifacts/final_model.pkl", "artifacts/bootstrap_models.pkl", "data/reference/reference.db"):
        print(f"  - {p}")
    if info["is_synthetic"]:
        print("\nNOTE: at least one input is SYNTHETIC - the app will show the DEMO banner.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="regenerate the synthetic data first")
    ap.add_argument("--quick", action="store_true", help="1 CV repeat instead of 3 (fast check only)")
    a = ap.parse_args()
    out = run(make_synthetic=a.synthetic, quick=a.quick)
    (ROOT / "artifacts" / "run_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
