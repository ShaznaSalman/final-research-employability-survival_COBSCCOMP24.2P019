"""API mode for the LLM benchmark: send benchmark/prompts.csv to an LLM and record answers in responses.csv.

Each prompt is a separate, independent request (no conversation history) - the API equivalent of opening a
new chat for every prompt. Answers are written in exactly the format used for manual entry, so
`python -m benchmark.score` treats both the same way. Safe to stop and re-run: rows already answered for the
same LLM are skipped, and the file is saved after every answer.

Providers (choose with --provider):
  openai     ChatGPT models     official `openai` SDK         key: OPENAI_API_KEY      model: --model or LLM_MODEL_OPENAI
  anthropic  Claude models      official `anthropic` SDK      key: ANTHROPIC_API_KEY   model: default claude-opus-5-5
  gemini     Gemini models      REST via `requests`           key: GEMINI_API_KEY      model: --model or LLM_MODEL_GEMINI
(Gemini uses REST because the google-genai SDK pins an older `websockets` than the app's core stack.)

Install only the SDK you need:  pip install -r requirements-benchmark.txt   (or just `pip install openai`)

Run:  python -m benchmark.run_llm --provider openai --model <model-id> [--runs 3] [--variant standard] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from benchmark.export_prompts import BENCH, DEFAULT_VARIANT, RESPONSE_COLUMNS, TRAIN_PATH, VARIANTS, assert_fresh

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REFUSED = "REFUSED"          # recorded when the model declines; score.py logs it as unparseable
MAX_TOKENS = 16000

PROVIDERS = {
    "openai": {"label": "ChatGPT", "key_env": "OPENAI_API_KEY", "model_env": "LLM_MODEL_OPENAI", "default": None},
    "anthropic": {"label": "Claude", "key_env": "ANTHROPIC_API_KEY", "model_env": "LLM_MODEL_ANTHROPIC",
                  "default": "claude-opus-5-5"},
    "gemini": {"label": "Gemini", "key_env": "GEMINI_API_KEY", "model_env": "LLM_MODEL_GEMINI", "default": None},
}


class LLMError(RuntimeError):
    """A provider call failed (after the SDK's own retries)."""


# ---------------------------------------------------------------------------
# One call per provider: (prompt, model) -> (answer text, model version that actually served it)
# ---------------------------------------------------------------------------
def ask_openai(prompt: str, model: str) -> tuple[str, str]:
    import openai
    client = openai.OpenAI()                     # reads OPENAI_API_KEY
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    except openai.APIError as e:
        raise LLMError(f"OpenAI: {e}") from e
    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        return REFUSED, resp.model
    return (msg.content or "").strip(), resp.model


def ask_anthropic(prompt: str, model: str) -> tuple[str, str]:
    import anthropic
    client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)
    try:
        # Server-side refusal fallback (opt-in default for current Claude models): if the requested model
        # declines, the API re-runs the request on the fallback model. `response.model` records which one
        # actually answered, so the benchmark always logs the true model version.
        resp = client.beta.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            betas=["server-side-fallback-2026-06-01"], fallbacks=[{"model": "claude-opus-4-8"}],
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Anthropic: network error ({e})") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic: HTTP {e.status_code} - {e.message}") from e
    if resp.stop_reason == "refusal":
        return REFUSED, resp.model
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.model


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def ask_gemini(prompt: str, model: str) -> tuple[str, str]:
    key = os.getenv("GEMINI_API_KEY", "")
    try:
        r = requests.post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": key}, timeout=120,
                          json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]})
    except requests.RequestException as e:
        raise LLMError(f"Gemini: network error ({e})") from e
    if r.status_code != 200:
        raise LLMError(f"Gemini: HTTP {r.status_code} - {r.text[:300]}")
    body = r.json()
    served = body.get("modelVersion", model)
    candidates = body.get("candidates") or []
    if not candidates or body.get("promptFeedback", {}).get("blockReason"):
        return REFUSED, served
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip(), served


ASKERS = {"openai": ask_openai, "anthropic": ask_anthropic, "gemini": ask_gemini}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _load_responses(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.reindex(columns=RESPONSE_COLUMNS, fill_value="")
    df["variant"] = df["variant"].replace("", DEFAULT_VARIANT)         # files from before prompt variants
    return df


def run(provider: str, model: str | None = None, runs: int = 3, llm_name: str | None = None,
        limit: int | None = None, delay: float = 0.5, dry_run: bool = False, bench: Path = BENCH,
        train_path: Path = TRAIN_PATH, ask=None, variant: str = DEFAULT_VARIANT) -> dict:
    spec = PROVIDERS[provider]
    model = model or os.getenv(spec["model_env"]) or spec["default"]
    if not model:
        raise SystemExit(f"Choose a model: --model <id> or set {spec['model_env']} in .env")
    if not dry_run and ask is None and not os.getenv(spec["key_env"]) and provider != "anthropic":
        raise SystemExit(f"Set {spec['key_env']} in .env first.")
    ask = ask or ASKERS[provider]
    llm_name = llm_name or spec["label"]
    assert_fresh(bench, train_path)

    prompts = pd.read_csv(bench / "prompts.csv", dtype=str)
    if "variant" not in prompts:
        prompts["variant"] = DEFAULT_VARIANT
    prompts = prompts[prompts["variant"] == variant]
    if prompts.empty:
        raise SystemExit(f"No prompts for variant '{variant}'. Re-run `python -m benchmark.export_prompts`.")
    if limit:
        prompts = prompts.head(limit)
    path = bench / "responses.csv"
    df = _load_responses(path)

    todo = []
    for run_number in range(1, runs + 1):
        for pid, text in zip(prompts["profile_id"], prompts["prompt_text"]):
            mine = ((df["profile_id"] == pid) & (df["variant"] == variant) & (df["run_number"] == str(run_number))
                    & (df["llm_name"] == llm_name))
            if (mine & (df["raw_response"] != "")).any():
                continue                                         # already answered for this LLM - resume
            todo.append((pid, run_number, text))
    if dry_run:
        return {"provider": provider, "model": model, "llm_name": llm_name, "variant": variant,
                "calls_needed": len(todo)}

    done = 0
    for pid, run_number, text in todo:
        try:
            answer, served = ask(text, model)
        except LLMError as e:
            df.to_csv(path, index=False)
            raise SystemExit(f"Stopped after {done} answers: {e}\nRe-run the same command to continue.") from None
        row = {"profile_id": pid, "variant": variant, "run_number": str(run_number), "llm_name": llm_name,
               "model_version": served,
               "date": dt.date.today().isoformat(), "raw_response": answer if answer else REFUSED}
        # Fill this LLM's empty row, else an unclaimed template row, else append.
        same = (df["profile_id"] == pid) & (df["variant"] == variant) & (df["run_number"] == str(run_number))
        target = df.index[same & (df["llm_name"] == llm_name) & (df["raw_response"] == "")]
        if not len(target):
            target = df.index[same & (df["llm_name"] == "") & (df["raw_response"] == "")]
        if len(target):
            df.loc[target[0], list(row)] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)                             # crash-safe: save after every answer
        done += 1
        if delay:
            time.sleep(delay)
    return {"provider": provider, "model": model, "llm_name": llm_name, "variant": variant, "answers_recorded": done}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", help="model ID (default: .env LLM_MODEL_<PROVIDER>, or claude-opus-5-5 for anthropic)")
    ap.add_argument("--runs", type=int, default=3, help="independent runs per prompt (default 3)")
    ap.add_argument("--llm-name", help="label in responses.csv (default ChatGPT / Claude / Gemini)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (for a quick test)")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between calls (default 0.5)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=VARIANTS, help="prompt variant (default standard)")
    ap.add_argument("--dry-run", action="store_true", help="show how many calls are needed; call nothing")
    a = ap.parse_args()
    print(json.dumps(run(a.provider, a.model, a.runs, a.llm_name, a.limit, a.delay, a.dry_run,
                         variant=a.variant), indent=2))
