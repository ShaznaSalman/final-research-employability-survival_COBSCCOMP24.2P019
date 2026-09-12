"""Tests for the LLM benchmark (benchmark/export_prompts.py, benchmark/score.py). Uses temporary folders only."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmark import export_prompts, score
from core import preprocess, splits
from core.models import make_model


@pytest.mark.parametrize("text, months", [
    ("5", 5), ("5 months", 5), ("About 7 months.", 7), ("3-6 months", 4.5), ("3 to 6", 4.5),
    ("1 year", 12), ("1.5 years", 18), ("eight months", 8), ("6 weeks", 6 / 4.345), ("  12\n", 12),
])
def test_parse_months_valid(text, months):
    value, _ = score.parse_months(text)
    assert value == pytest.approx(months)


@pytest.mark.parametrize("text, reason", [
    ("", "empty"), (None, "empty"), ("It depends on many factors.", "no number"), ("I cannot predict that", "no number"),
    ("500 months", "implausible"),
])
def test_parse_months_unparseable(text, reason):
    value, note = score.parse_months(text)
    assert value is None and reason in note


@pytest.fixture
def bench_env(tmp_path, train_df):
    """Minimal artifacts + processed data in a temp dir."""
    art, bench = tmp_path / "artifacts", tmp_path / "benchmark"
    art.mkdir()
    train_path = tmp_path / "graduates_train.csv"
    train_df.to_csv(train_path, index=False)
    splits.write(train_df, tmp_path / splits.SPLITS_PATH.name)
    X = preprocess.build_preprocessor().fit_transform(train_df)
    t, e = preprocess.outcome(train_df)
    m = make_model("cox_ph").fit(X, t, e)
    S = m.predict_survival(X, np.array([3.0, 6.0, 12.0]))
    oof = pd.DataFrame({"repeat": 0, "fold": 0, "model": "cox_ph", "respondent_id": train_df["respondent_id"],
                        "risk": m.predict_risk(X), "median_months": 5.0, "duration_months": t, "event": e,
                        "S_3": S[:, 0], "S_6": S[:, 1], "S_12": S[:, 2]})
    oof.to_csv(art / "oof_predictions.csv", index=False)
    (art / "selection.json").write_text(json.dumps({"selected": "cox_ph", "label": "Cox PH"}), encoding="utf-8")
    info = tmp_path / "info.json"
    info.write_text('{"is_synthetic": true}', encoding="utf-8")
    return {"art": art, "bench": bench, "train": train_path, "info": info, "oof": oof}


def test_prompts_contain_no_outcome(bench_env, train_df):
    out = export_prompts.export(n=30, out_dir=bench_env["bench"], artifacts=bench_env["art"],
                                train_path=bench_env["train"])
    assert out["profiles"] == 30
    prompts = pd.read_csv(bench_env["bench"] / "prompts.csv")
    key = pd.read_csv(bench_env["bench"] / "profile_key.csv")
    assert list(prompts.columns) == ["profile_id", "variant", "prompt_text"]
    assert len(prompts) == 30 * len(export_prompts.VARIANTS)
    rid_of = dict(zip(key["profile_id"], key["respondent_id"]))
    for pid, variant, text in prompts.itertuples(index=False):
        assert rid_of[pid] not in text
        assert "found" not in text.lower() and "still searching" not in text.lower()
        assert "gender" not in text.lower()
        if variant == "range":
            assert "BEST (LOW-HIGH)" in text
        else:
            assert "Answer with a single whole number of months only" in text
        assert ("Sri Lanka" in text) == (variant != "no_country")


def test_existing_responses_are_never_overwritten(bench_env):
    kw = dict(n=10, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"])
    export_prompts.export(**kw)
    responses = bench_env["bench"] / "responses.csv"
    df = pd.read_csv(responses, dtype={"raw_response": "string"})
    assert list(df.columns) == export_prompts.RESPONSE_COLUMNS and len(df) == 10 * export_prompts.DEFAULT_RUNS
    df.loc[0, "raw_response"] = "7"
    df.to_csv(responses, index=False)
    assert export_prompts.export(**kw)["responses_template_written"] is False
    assert pd.read_csv(responses).loc[0, "raw_response"] == 7


def test_score_with_responses_including_unparseable(bench_env):
    export_prompts.export(n=40, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"])
    key = pd.read_csv(bench_env["bench"] / "profile_key.csv").merge(
        bench_env["oof"][["respondent_id", "duration_months"]], on="respondent_id")
    rng = np.random.default_rng(0)
    rows = []
    for run in (1, 2):
        for i, r in key.iterrows():
            answer = f"{max(1, round(r['duration_months'] + rng.normal(0, 3)))} months"
            if i == 0 and run == 1:
                answer = "It is impossible to say."
            rows.append({"profile_id": r["profile_id"], "run_number": run, "llm_name": "TestLLM",
                         "model_version": "v1", "date": "2026-10-06", "raw_response": answer})
    pd.DataFrame(rows).to_csv(bench_env["bench"] / "responses.csv", index=False)

    out = score.score(bench=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                      info_path=bench_env["info"])
    assert out["unparseable"] == 1 and out["llms"] == ["TestLLM v1"]
    bad = pd.read_csv(bench_env["art"] / "llm_unparseable.csv")
    assert len(bad) == 1 and "no number" in bad.loc[0, "parse_note"]
    runs = pd.read_csv(bench_env["art"] / "llm_benchmark.csv")
    assert set(runs["run"]) == {1, 2} and runs["uno_c"].between(0, 1).all()
    assert runs["uno_c"].min() > 0.6          # answers built from true durations should rank well
    md = (bench_env["art"] / "llm_benchmark.md").read_text(encoding="utf-8")
    assert "Consistency across runs" in md and "C difference model - LLM" in md


def test_score_without_responses_reports_model_only(bench_env):
    export_prompts.export(n=20, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"])
    out = score.score(bench=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                      info_path=bench_env["info"])
    assert out["llms"] == [] and out["responses_answered"] == 0
    assert "No LLM responses recorded yet" in (bench_env["art"] / "llm_benchmark.md").read_text(encoding="utf-8")


def test_score_refuses_stale_responses(bench_env, train_df):
    export_prompts.export(n=20, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"])
    train_df.iloc[:-1].to_csv(bench_env["train"], index=False)          # data changed after export
    with pytest.raises(score.StaleBenchmarkError):
        score.score(bench=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                    info_path=bench_env["info"])


def test_standard_variant_is_the_original_prompt_and_variants_share_the_profile():
    assert export_prompts.template("standard") == export_prompts.PROMPT_TEMPLATE
    for v in export_prompts.VARIANTS:
        assert export_prompts.template(v).count("{profile}") == 1
    assert "8 months" in export_prompts.template("local_context")
    with pytest.raises(ValueError):
        export_prompts.template("unknown")


@pytest.mark.parametrize("text, expected", [
    ("5 (3-8)", (5, 3, 8)), ("6 (4 to 10) months", (6, 4, 10)), ("4 (9-2)", (4, 2, 9)),
])
def test_parse_range(text, expected):
    best, lo, hi, _ = score.parse_range(text)
    assert (best, lo, hi) == expected


def test_parse_range_without_range_keeps_point_estimate():
    best, lo, hi, note = score.parse_range("7 months")
    assert best == 7 and lo is None and "no range" in note


def test_score_variants_and_range_coverage(bench_env):
    export_prompts.export(n=40, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                          template_variants=("standard", "range"))
    template = pd.read_csv(bench_env["bench"] / "responses.csv")
    assert set(template["variant"]) == {"standard", "range"}
    key = pd.read_csv(bench_env["bench"] / "profile_key.csv").merge(
        bench_env["oof"][["respondent_id", "duration_months"]], on="respondent_id")
    rows = []
    for _, r in key.iterrows():
        t = max(1, round(r["duration_months"]))
        base = {"profile_id": r["profile_id"], "run_number": 1, "llm_name": "TestLLM", "model_version": "v1",
                "date": "2026-10-06"}
        rows.append({**base, "variant": "standard", "raw_response": str(t)})
        rows.append({**base, "variant": "range", "raw_response": f"{t} ({max(0, t - 30)}-{t + 30})"})
    pd.DataFrame(rows).to_csv(bench_env["bench"] / "responses.csv", index=False)
    out = score.score(bench=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                      info_path=bench_env["info"])
    assert set(out["llms"]) == {"TestLLM v1", "TestLLM v1 [range]"} and out["variant_comparisons"] == 1
    md = (bench_env["art"] / "llm_benchmark.md").read_text(encoding="utf-8")
    assert "Prompt variants vs the standard prompt" in md and "80% ranges" in md
    cov_line = next(line for line in md.splitlines() if line.startswith("| TestLLM v1 | 40 |"))
    assert "| 100% |" in cov_line                        # +/- 30 months always contains the true time


def test_old_responses_without_variant_column_still_score(bench_env):
    export_prompts.export(n=20, out_dir=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"])
    key = pd.read_csv(bench_env["bench"] / "profile_key.csv")
    pd.DataFrame({"profile_id": key["profile_id"], "run_number": 1, "llm_name": "Old", "model_version": "",
                  "date": "", "raw_response": "6"}).to_csv(bench_env["bench"] / "responses.csv", index=False)
    out = score.score(bench=bench_env["bench"], artifacts=bench_env["art"], train_path=bench_env["train"],
                      info_path=bench_env["info"])
    assert out["llms"] == ["Old"] and out["responses_answered"] == 20
