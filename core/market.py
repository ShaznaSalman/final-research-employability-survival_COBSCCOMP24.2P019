"""Market alignment (employer requirements + job ads) and the reference database.

reference.db (SQLite) holds everything the API needs at run time - no respondent-level graduate data:
  employers        cleaned employer questionnaire responses
  job_ads          manually collected job ads (title, company, date, source)
  job_ad_skills    one row per (ad, SKILL_VOCABULARY key)
  skill_demand     per skill: mean employer importance (1-5), % of ads mentioning it, demand index
  peer_stats       Kaplan-Meier medians by peer group (core/peers.py)
  meta             data provenance (synthetic flag, counts)

Demand index (0-1) = mean of the available components: (employer importance - 1) / 4 and share of ads.
Gap = demand - (user rating - 1) / 4. Top gaps = skills with above-median demand and gap > GAP_MIN.

Run:  python -m core.market      (after core.clean)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from core import peers, schema, wording
from core.predict import profile_frame

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DB = ROOT / "data" / "reference" / "reference.db"
LEARNING_RESOURCES = ROOT / "data" / "reference" / "learning_resources.csv"   # editable list of free resources
MAX_RESOURCES = 2

GAP_MIN = 0.10
TOP_GAPS = 5


def _area_of(key: str) -> str:
    return next((a for a, v in schema.SKILL_AREAS.items() if key in v["skills"]), "soft")


def skill_demand(employers: pd.DataFrame, job_ads: pd.DataFrame) -> pd.DataFrame:
    mentions = job_ads["skills"].fillna("").str.split(";")
    rows = []
    for s in schema.SKILL_VOCABULARY:
        col = f"imp_{s.key}"
        imp = pd.to_numeric(employers[col], errors="coerce") if col in employers else pd.Series(dtype=float)
        ads_n = int(mentions.map(lambda m, k=s.key: k in m).sum())
        ads_share = ads_n / len(job_ads) if len(job_ads) else np.nan
        imp_mean = float(imp.mean()) if imp.notna().any() else np.nan
        parts = [p for p in ((imp_mean - 1) / 4, ads_share) if pd.notna(p)]
        rows.append({"skill": s.key, "label": s.label, "kind": s.kind, "area": _area_of(s.key),
                     "employer_importance": imp_mean, "employer_n": int(imp.notna().sum()),
                     "ads_n": ads_n, "ads_pct": round(100 * ads_share, 1) if pd.notna(ads_share) else np.nan,
                     "demand": float(np.mean(parts)) if parts else np.nan})
    return pd.DataFrame(rows)


def build_reference_db(processed_dir: Path = PROCESSED_DIR, db_path: Path = REFERENCE_DB,
                       resources_csv: Path = LEARNING_RESOURCES) -> dict:
    """(Re)build reference.db from the processed files. The file is replaced so the build is reproducible."""
    employers = pd.read_csv(processed_dir / "employers.csv")
    job_ads = pd.read_csv(processed_dir / "job_ads.csv")
    from core import splits
    # Peer reference data comes from the training set only (no validation / test graduate reaches the app)
    if (processed_dir / "splits.csv").exists():
        train = splits.load("train", processed_dir / "graduates_train.csv", processed_dir / "splits.csv")
    else:                                   # no split file yet: apply the same fixed split rule directly
        allg = pd.read_csv(processed_dir / "graduates_train.csv")
        train = allg[splits.assign(allg) == "train"].reset_index(drop=True)
    info = json.loads((processed_dir / "dataset_info.json").read_text(encoding="utf-8"))

    demand = skill_demand(employers, job_ads)
    ad_skills = job_ads[["ad_id", "skills"]].assign(skill=job_ads["skills"].str.split(";")) \
        .explode("skill")[["ad_id", "skill"]]
    peer_stats = peers.compute_peer_stats(train)
    meta = pd.DataFrame([
        ("is_synthetic", json.dumps(info["is_synthetic"])), ("n_employers", str(len(employers))),
        ("n_job_ads", str(len(job_ads))), ("n_graduates", str(len(train))),
        ("peer_min_group_n", str(peers.MIN_GROUP_N)), ("sources", json.dumps(info["sources"])),
    ], columns=["key", "value"])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        employers.to_sql("employers", con, index=False)
        job_ads.drop(columns="skills").to_sql("job_ads", con, index=False)
        ad_skills.to_sql("job_ad_skills", con, index=False)
        demand.to_sql("skill_demand", con, index=False)
        peer_stats.to_sql("peer_stats", con, index=False)
        meta.to_sql("meta", con, index=False)
        if resources_csv.exists():
            pd.read_csv(resources_csv).to_sql("learning_resources", con, index=False)
    return {"db": str(db_path), "employers": len(employers), "job_ads": len(job_ads),
            "skills": len(demand), "peer_groups": len(peer_stats),
            "peer_groups_suppressed": int(peer_stats["suppressed"].sum()), "is_synthetic": info["is_synthetic"]}


class ReferenceData:
    """Read-only view of reference.db, loaded once at API start-up."""

    def __init__(self, demand: pd.DataFrame, peer_stats: pd.DataFrame, meta: dict,
                 resources: pd.DataFrame | None = None):
        self.demand, self.peer_stats, self.meta = demand, peer_stats, meta
        self.resources = resources if resources is not None else pd.DataFrame(columns=["skill", "title", "provider", "url"])

    @classmethod
    def load(cls, db_path: Path = REFERENCE_DB) -> "ReferenceData":
        with sqlite3.connect(db_path) as con:
            demand = pd.read_sql("SELECT * FROM skill_demand", con)
            peer_stats = pd.read_sql("SELECT * FROM peer_stats", con)
            meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            resources = pd.read_sql("SELECT * FROM learning_resources", con) if "learning_resources" in tables else None
        peer_stats["suppressed"] = peer_stats["suppressed"].astype(bool)
        return cls(demand, peer_stats, meta, resources)

    def alignment(self, profile: dict, top_n: int = TOP_GAPS) -> dict:
        row = profile_frame(profile).iloc[0]
        skills = []
        for _, d in self.demand.iterrows():
            rating = row.get(schema.SKILLS_BY_KEY[d["skill"]].column)
            rating = None if pd.isna(rating) else float(rating)
            gap = None if rating is None or pd.isna(d["demand"]) else float(d["demand"] - (rating - 1) / 4)
            skills.append({"skill": d["skill"], "label": d["label"], "kind": d["kind"], "area": d["area"],
                           "your_rating": rating,
                           "employer_importance": None if pd.isna(d["employer_importance"])
                           else round(float(d["employer_importance"]), 2),
                           "ads_pct": None if pd.isna(d["ads_pct"]) else float(d["ads_pct"]),
                           "demand": None if pd.isna(d["demand"]) else round(float(d["demand"]), 3),
                           "gap": None if gap is None else round(gap, 3)})
        median_demand = float(self.demand["demand"].median())
        gaps = sorted([s for s in skills if s["gap"] is not None and s["gap"] > GAP_MIN
                       and s["demand"] >= median_demand], key=lambda s: -s["gap"])[:top_n]
        for s in gaps:
            fill = {"label": s["label"], "ads_pct": f"{s['ads_pct']:.0f}", "rating": f"{s['your_rating']:g}"}
            s["text"] = (wording.MARKET_GAP.format(importance=f"{s['employer_importance']:.1f}", **fill)
                         if s["employer_importance"] is not None else wording.MARKET_GAP_NO_EMPLOYER.format(**fill))
            res = self.resources[self.resources["skill"] == s["skill"]].head(MAX_RESOURCES)
            s["resources"] = res[["title", "provider", "url"]].to_dict("records")
        return {"skills": skills, "top_gaps": gaps,
                "message": None if gaps else wording.MARKET_NO_GAPS,
                "note": wording.MARKET_NOTE.format(n_employers=self.meta.get("n_employers", "?"),
                                                   n_ads=self.meta.get("n_job_ads", "?"))}

    def peer_benchmark(self, profile: dict) -> dict:
        return peers.peer_benchmark(self.peer_stats, profile)


if __name__ == "__main__":
    print(json.dumps(build_reference_db(), indent=2))
