"""Tests for the LLM benchmark API mode (benchmark/run_llm.py). No network: providers are faked."""
import types

import pandas as pd
import pytest

from benchmark import export_prompts, run_llm, score
from tests.test_benchmark import bench_env  # noqa: F401  (shared fixture)


@pytest.fixture
def exported(bench_env):  # noqa: F811
    export_prompts.export(n=10, runs=2, out_dir=bench_env["bench"], artifacts=bench_env["art"],
                          train_path=bench_env["train"])
    return bench_env


def _run(env, ask, **kw):
    return run_llm.run(kw.pop("provider", "openai"), model="test-model", runs=2, delay=0, bench=env["bench"],
                       train_path=env["train"], ask=ask, **kw)


def _responses(env):
    return pd.read_csv(env["bench"] / "responses.csv", dtype=str, keep_default_na=False)


def test_fills_template_rows_and_records_served_model(exported):
    calls = []
    out = _run(exported, lambda p, m: (calls.append(p) or "6", "test-model-2026-01"))
    df = _responses(exported)
    assert out["answers_recorded"] == len(calls) == 20 and len(df) == 20          # template rows claimed, no extras
    assert set(df["llm_name"]) == {"ChatGPT"} and set(df["model_version"]) == {"test-model-2026-01"}
    assert (df["raw_response"] == "6").all() and (df["date"] != "").all()


def test_resumes_without_repeating_calls(exported):
    _run(exported, lambda p, m: ("6", m), limit=3)                                  # partial run
    calls = []
    _run(exported, lambda p, m: (calls.append(p) or "7", m))
    assert len(calls) == 20 - 6                                                   # 3 prompts x 2 runs already done


def test_second_llm_appends_its_own_rows(exported):
    _run(exported, lambda p, m: ("6", m))
    _run(exported, lambda p, m: ("9", m), provider="gemini")
    df = _responses(exported)
    assert len(df) == 40 and df.groupby("llm_name").size().to_dict() == {"ChatGPT": 20, "Gemini": 20}


def test_failure_saves_progress_and_stops(exported):
    n = {"calls": 0}

    def flaky(p, m):
        n["calls"] += 1
        if n["calls"] == 4:
            raise run_llm.LLMError("rate limited")
        return "5", m

    with pytest.raises(SystemExit, match="Stopped after 3 answers"):
        _run(exported, flaky)
    assert (_responses(exported)["raw_response"] == "5").sum() == 3


def test_refusals_are_recorded_and_scored_as_unparseable(exported):
    _run(exported, lambda p, m: (run_llm.REFUSED, m))
    out = score.score(bench=exported["bench"], artifacts=exported["art"], train_path=exported["train"],
                      info_path=exported["info"])
    assert out["unparseable"] == 20


def test_dry_run_calls_nothing(exported):
    out = _run(exported, lambda p, m: pytest.fail("must not call"), dry_run=True)
    assert out["calls_needed"] == 20


def test_refuses_stale_prompts(exported, train_df):
    train_df.iloc[:-1].to_csv(exported["train"], index=False)
    with pytest.raises(export_prompts.StaleBenchmarkError):
        _run(exported, lambda p, m: ("6", m))


# --- provider adapters (SDK / HTTP stubbed) ------------------------------------------------
def test_gemini_adapter_parses_rest_response(monkeypatch):
    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"modelVersion": "gemini-x-001", "candidates": [{"content": {"parts": [{"text": " 8 months "}]}}]}
    monkeypatch.setattr(run_llm.requests, "post", lambda *a, **k: R())
    assert run_llm.ask_gemini("prompt", "gemini-x") == ("8 months", "gemini-x-001")


def test_gemini_adapter_reports_blocked_prompt(monkeypatch):
    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"promptFeedback": {"blockReason": "SAFETY"}}
    monkeypatch.setattr(run_llm.requests, "post", lambda *a, **k: R())
    assert run_llm.ask_gemini("prompt", "gemini-x")[0] == run_llm.REFUSED


def test_anthropic_adapter_handles_refusal_and_text(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    replies = iter([
        types.SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                              content=[types.SimpleNamespace(type="thinking"),
                                       types.SimpleNamespace(type="text", text="7")]),
        types.SimpleNamespace(stop_reason="refusal", model="claude-opus-4-8", content=[]),
    ])
    sent = {}

    class Fake:
        def __init__(self, *a, **k):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self.create))

        def create(self, **kw):
            sent.update(kw)
            return next(replies)
    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    assert run_llm.ask_anthropic("p", "claude-opus-5-5") == ("7", "claude-opus-5-5")
    assert sent["fallbacks"] == [{"model": "claude-opus-4-8"}] and "server-side-fallback-2026-06-01" in sent["betas"]
    assert run_llm.ask_anthropic("p", "claude-opus-5-5") == (run_llm.REFUSED, "claude-opus-4-8")


def test_openai_adapter_reads_message(monkeypatch):
    openai = pytest.importorskip("openai")

    class Fake:
        def __init__(self, *a, **k):
            msg = types.SimpleNamespace(content="5", refusal=None)
            resp = types.SimpleNamespace(model="gpt-test-2026", choices=[types.SimpleNamespace(message=msg)])
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: resp))
    monkeypatch.setattr(openai, "OpenAI", Fake)
    assert run_llm.ask_openai("p", "gpt-test") == ("5", "gpt-test-2026")


def test_variant_rows_are_kept_separate(exported):
    _run(exported, lambda p, m: ("6", "m"))
    out = _run(exported, lambda p, m: ("5 (3-8)", "m"), variant="range")
    assert out["answers_recorded"] == 10 * 2 and out["variant"] == "range"
    df = _responses(exported)
    assert set(df["variant"]) == {"standard", "range"}
    assert (df.loc[df["variant"] == "range", "raw_response"] == "5 (3-8)").all()
    assert (df.loc[df["variant"] == "standard", "raw_response"] == "6").all()


def test_range_prompts_are_sent_for_range_variant(exported):
    sent = []
    _run(exported, lambda p, m: (sent.append(p) or "5 (3-8)", "m"), variant="range", limit=1)
    assert sent and all("BEST (LOW-HIGH)" in p for p in sent)
