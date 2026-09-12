"""Export held-out profiles as plain-text prompts for a general-purpose LLM.

Profiles come from the LOCKED TEST SET (artifacts/oof_predictions.csv holds the selected model's test-set
predictions; core/splits.py), so the model is scored on graduates it never saw and that were never used to tune or
choose it - the same footing as the LLM. The true outcome is never written into a prompt.

Writes:
  benchmark/prompts.csv          profile_id, variant, prompt_text   (paste these into the LLM)
  benchmark/prompt_template.txt  the fixed templates, one per variant (thesis appendix)
  benchmark/profile_key.csv      profile_id -> respondent_id         (NOT for the LLM; used by score.py)
  benchmark/responses.csv        template to record answers          (only if it does not exist yet)

Prompt variants (fixed in advance; `standard` is the primary comparison):
  standard       the main prompt
  no_country     the same without any mention of Sri Lanka - does the country context matter?
  local_context  adds one published Sri Lankan figure (tracer-study average time to employment)
  range          asks for a best estimate AND an 80% range, so interval coverage can be scored

Run:  python -m benchmark.export_prompts [--n 60] [--runs 3] [--variants standard range] [--force-template]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from core import preprocess, schema

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
ARTIFACTS = ROOT / "artifacts"
TRAIN_PATH = ROOT / "data" / "processed" / "graduates_train.csv"

SEED = 20261006
DEFAULT_N = 60
DEFAULT_RUNS = 3
RESPONSE_COLUMNS = ["profile_id", "variant", "run_number", "llm_name", "model_version", "date", "raw_response"]
DEFAULT_VARIANT = "standard"

PROMPT_TEMPLATE = """You are helping with a research study on graduate employment in Sri Lanka.

Below is the profile of a graduate from an IT / computing degree at a Sri Lankan higher education institution.
"First IT job" means a paid IT role (full-time, part-time, contract, or a paid trainee/intern position taken
after the degree). Internships that were part of the degree do not count.

Graduate profile:
{profile}

Question: How many months after completing their degree will this graduate take to find their first IT job?
Answer with a single whole number of months only. Do not give a range or any explanation."""

# Published figure used by the local_context variant: University of Kelaniya graduate tracer study (2022).
LOCAL_CONTEXT = ("Context: a graduate tracer study by the University of Kelaniya (2022) reported that its graduates "
                 "took about 8 months on average to find employment after graduating.\n\n")
RANGE_QUESTION = ("Question: How many months after completing their degree will this graduate take to find their "
                  "first IT job?\nGive your single best estimate, followed by a range that you are 80% confident "
                  "contains the true value, in this exact format: BEST (LOW-HIGH), all in whole months. "
                  "Do not give any explanation.")
VARIANTS = ("standard", "no_country", "local_context", "range")


def template(variant: str = DEFAULT_VARIANT) -> str:
    """The prompt template for one variant. `standard` is PROMPT_TEMPLATE unchanged."""
    if variant == "standard":
        return PROMPT_TEMPLATE
    if variant == "no_country":
        return (PROMPT_TEMPLATE.replace(" in Sri Lanka", "")
                .replace(" at a Sri Lankan higher education institution", " at a higher education institution"))
    if variant == "local_context":
        return PROMPT_TEMPLATE.replace("Graduate profile:", LOCAL_CONTEXT + "Graduate profile:")
    if variant == "range":
        return PROMPT_TEMPLATE[:PROMPT_TEMPLATE.index("Question:")] + RANGE_QUESTION
    raise ValueError(f"unknown prompt variant: {variant}")

def data_fingerprint(train_path: Path) -> str:
    """Hash of the training file: responses are only valid for the profiles exported from this exact data."""
    return hashlib.sha256(train_path.read_bytes()).hexdigest()[:16]


class StaleBenchmarkError(RuntimeError):
    """Responses were (or would be) collected for profiles exported from different training data."""


def assert_fresh(bench: Path = BENCH, train_path: Path = TRAIN_PATH) -> None:
    meta_path = bench / "benchmark_meta.json"
    if meta_path.exists():
        exported = json.loads(meta_path.read_text(encoding="utf-8")).get("data_fingerprint")
        if exported != data_fingerprint(train_path):
            raise StaleBenchmarkError(
                "The training data changed since the prompts were exported, so recorded responses no longer match "
                "the profiles. Run `python -m benchmark.export_prompts --force-template` and collect new responses.")


DEGREE_LABELS = {
    "computer_science": "Computer Science", "information_technology": "Information Technology",
    "software_engineering": "Software Engineering", "information_systems": "Information Systems",
    "cyber_security": "Cyber Security", "data_science": "Data Science",
    "ai_ml": "Artificial Intelligence / Machine Learning", "other": "Other computing degree",
}
CLASS_LABELS = {"first": "First class", "second_upper": "Second class upper", "second_lower": "Second class lower",
                "third_or_pass": "Third class / general pass"}
INSTITUTION_LABELS = {"state": "State university", "non_state": "Non-state (private) degree-awarding institute",
                      "foreign_affiliated": "Foreign-affiliated institute (degree awarded by an overseas university)"}


def describe(row: pd.Series, areas: pd.Series) -> str:
    """Plain-text profile using the same information the model sees."""
    def fmt(v, f="{:g}"):
        return "not given" if pd.isna(v) else f.format(v)

    gpa_to_class = {v: k for k, v in schema.DEGREE_CLASS_TO_GPA.items()}
    if row.get("gpa_source") == "degree_class" and row["gpa"] in gpa_to_class:
        academic = f"- Degree class: {CLASS_LABELS[gpa_to_class[row['gpa']]]}"
    else:
        academic = f"- GPA: {fmt(row['gpa'], '{:.2f}')} (out of 4.00)"
    lines = [
        f"- Degree programme: {DEGREE_LABELS.get(row['degree_type'], 'not given')}",
        f"- Type of institution: {INSTITUTION_LABELS.get(row['university_type'], 'not given')}",
        academic,
        f"- Internship experience: {fmt(row['internship_months'])} months in total",
        f"- Professional IT certifications: {fmt(row['certifications_count'])}",
        f"- Personal / academic IT projects outside coursework: {fmt(row['project_count'])}",
        "- Self-rated skills at degree completion (1 = no knowledge, 5 = advanced):",
    ]
    for a, spec in schema.SKILL_AREAS.items():
        lines.append(f"    - {spec['label']}: {fmt(areas[f'area_{a}'], '{:.1f}')}")
    lines.append(f"    - Soft skills (communication, teamwork, problem solving, etc.): {fmt(areas['area_soft'], '{:.1f}')}")
    return "\n".join(lines)


def select_profiles(oof: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified (by event) random sample of the selected model's test-set profiles (all of them if n >= test size)."""
    rep0 = oof[oof["repeat"] == 0]
    if n >= len(rep0):
        return rep0.sort_values("respondent_id").reset_index(drop=True)
    frac = n / len(rep0)
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=SEED) for _, g in rep0.groupby("event")]
    return pd.concat(parts).sort_values("respondent_id").reset_index(drop=True)


def export(n: int = DEFAULT_N, runs: int = DEFAULT_RUNS, force_template: bool = False, out_dir: Path = BENCH,
           artifacts: Path = ARTIFACTS, train_path: Path = TRAIN_PATH,
           template_variants: tuple = (DEFAULT_VARIANT,)) -> dict:
    """prompts.csv holds every variant; the responses template has rows for `template_variants` only."""
    selected = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))["selected"]
    oof = pd.read_csv(artifacts / "oof_predictions.csv")
    oof = oof[oof["model"] == selected]
    picked = select_profiles(oof, n)
    train = pd.read_csv(train_path).set_index("respondent_id")
    rows = train.loc[picked["respondent_id"]].reset_index()
    areas = preprocess.BlockMeans().transform(rows)

    picked["profile_id"] = [f"P{i:03d}" for i in range(1, len(picked) + 1)]
    profiles = [describe(r, areas.loc[i]) for i, r in rows.iterrows()]
    prompts = pd.DataFrame([{"profile_id": pid, "variant": v, "prompt_text": template(v).format(profile=p)}
                            for v in VARIANTS for pid, p in zip(picked["profile_id"], profiles)])
    out_dir.mkdir(exist_ok=True)
    prompts.to_csv(out_dir / "prompts.csv", index=False)
    (out_dir / "prompt_template.txt").write_text(
        "\n\n".join(f"===== Variant: {v} =====\n{template(v)}" for v in VARIANTS), encoding="utf-8")
    picked[["profile_id", "respondent_id", "repeat", "fold"]].to_csv(out_dir / "profile_key.csv", index=False)
    (out_dir / "benchmark_meta.json").write_text(json.dumps(
        {"data_fingerprint": data_fingerprint(train_path), "profiles": len(picked), "selected_model": selected},
        indent=2), encoding="utf-8")

    responses = out_dir / "responses.csv"
    wrote_template = False
    if force_template or not responses.exists():
        blank = pd.DataFrame([{"profile_id": p, "variant": v, "run_number": r, "llm_name": "",
                               "model_version": "", "date": "", "raw_response": ""}
                              for v in template_variants for r in range(1, runs + 1) for p in picked["profile_id"]],
                             columns=RESPONSE_COLUMNS)
        blank.to_csv(responses, index=False)
        wrote_template = True
    return {"profiles": len(picked), "events": int(picked["event"].sum()), "selected_model": selected,
            "variants": list(VARIANTS), "responses_template_written": wrote_template, "runs_in_template": runs,
            "template_variants": list(template_variants)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="number of held-out profiles (default 60)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="runs per profile in the responses template")
    ap.add_argument("--variants", nargs="+", default=[DEFAULT_VARIANT], choices=VARIANTS,
                    help="variants to include in the responses template (default: standard)")
    ap.add_argument("--force-template", action="store_true", help="overwrite an existing responses.csv")
    a = ap.parse_args()
    print(json.dumps(export(a.n, a.runs, a.force_template, template_variants=tuple(a.variants)), indent=2))
