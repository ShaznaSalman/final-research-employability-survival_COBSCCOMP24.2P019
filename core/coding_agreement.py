"""Inter-rater check of the keyword skill coding of job ads (Cohen's kappa).

Step 1  python -m core.coding_agreement --sample     writes data/raw/job_ads_coding_sample.csv: 40 randomly chosen
        keyword-coded ads with their links and one empty 0/1 column per skill (the keyword result is NOT shown, so
        the hand coding is blind).
Step 2  Open each link, read the ad and put 1 in every skill column the ad asks for (0 or blank otherwise).
Step 3  python -m core.coding_agreement              compares the hand coding with the keyword rules and writes
        artifacts/coding_agreement.md: per-skill percent agreement, Cohen's kappa, and pooled kappa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import schema

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = RAW / "job_ads_coding_sample.csv"
OUT = ROOT / "artifacts" / "coding_agreement.md"
N_SAMPLE, SEED = 40, 20261010


def make_sample(n: int = N_SAMPLE, seed: int = SEED) -> Path:
    ads = pd.read_csv(RAW / "job_ads.csv")
    prov = pd.read_csv(RAW / "job_ads_provenance.csv")
    keyword = prov[(prov["included"] == "yes") & prov["coding"].fillna("").str.startswith("keyword")]
    pool = ads[ads["ad_id"].isin(keyword["ad_id"])].merge(prov[["ad_id", "url"]], on="ad_id")
    s = pool.sample(min(n, len(pool)), random_state=seed)[["ad_id", "title", "company", "url"]]
    for sk in schema.SKILL_VOCABULARY:
        s[sk.key] = ""
    s.to_csv(SAMPLE, index=False)
    return SAMPLE


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def compare(sample: Path = SAMPLE) -> str:
    hand = pd.read_csv(sample)
    keys = [s.key for s in schema.SKILL_VOCABULARY]
    if hand[keys].isna().all().all():
        raise SystemExit(f"{sample} has no hand coding yet - fill in the skill columns first.")
    uncoded = hand[keys].isna().all(axis=1)          # ad could not be opened: leave its row blank, it is skipped
    skipped = hand.loc[uncoded, "ad_id"].tolist()
    hand = hand[~uncoded].reset_index(drop=True)
    auto = pd.read_csv(RAW / "job_ads.csv").set_index("ad_id")["skills"].fillna("")
    from core.clean import _skill_token
    coded = {aid: {_skill_token(t) for t in auto[aid].split(";") if t.strip()} for aid in hand["ad_id"]}
    rows, all_h, all_a = [], [], []
    for s in schema.SKILL_VOCABULARY:
        h = hand[s.key].fillna(0).astype(int).to_numpy()
        a = np.array([int(s.key in coded[aid]) for aid in hand["ad_id"]])
        all_h.append(h)
        all_a.append(a)
        rows.append((s.label, h.sum(), a.sum(), 100 * (h == a).mean(), kappa(h, a)))
    H, A = np.concatenate(all_h), np.concatenate(all_a)
    tp, fp, fn = int(((H == 1) & (A == 1)).sum()), int(((H == 0) & (A == 1)).sum()), int(((H == 1) & (A == 0)).sum())
    lines = ["# Coding agreement: keyword rules vs hand coding", "",
             f"{len(hand)} randomly sampled keyword-coded ads, hand-coded blind to the keyword result"
             + (f"; {len(skipped)} left uncoded by the coder, for example because the advert could no longer be opened "
                  f"({', '.join(skipped)})."
                if skipped else "."), "",
             f"- Pooled Cohen's kappa: **{kappa(H, A):.2f}** (conventionally, 0.61-0.80 is substantial and above 0.80 "
             f"almost perfect agreement; the pre-set target was at least 0.60)",
             f"- Keyword precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f}",
             "", "| Skill | Hand | Keyword | Agreement | Kappa |", "|---|---|---|---|---|"]
    lines += [f"| {l} | {h} | {a} | {p:.0f}% | {'-' if np.isnan(k) else f'{k:.2f}'} |" for l, h, a, p, k in rows]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines[4]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="write the blind hand-coding sample")
    a = ap.parse_args()
    print(make_sample() if a.sample else compare())
