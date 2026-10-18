"""Experiment tracking: a reproducibility log of a pipeline run (local folder or Weights & Biases).

  python -m core.tracking --track local     write artifacts/runs/<timestamp>/ (no account needed)
  python -m core.tracking --track wandb     log the same run to W&B (offline by default; `wandb sync` to upload)
  python -m core.run_all --track local      run the pipeline, then log it

What is logged is read from the artifacts the pipeline has already written: the data fingerprint, the split sizes,
the settings and validation metrics of every candidate model, the selection, DeepSurv's loss per epoch, the learning
curve and the figures. Nothing is refitted.

Rules:
  * Dates. Runs are dated when they are logged. They are a reproducibility log, not records of the original development.
  * Test set. It is never evaluated again: the stored result is loaded and labelled "locked test result (loaded)".
  * Privacy. Only aggregate metrics, settings and figures are logged, never survey answers or respondent rows.
  * Default off. Without --track the pipeline behaves exactly as before.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RAW = ROOT / "data" / "raw"
TEST_LABEL = "locked test result (loaded, not recomputed)"


class LocalTracker:
    """Writes params.json, metrics.jsonl, the copied figures and a short run.md to artifacts/runs/<timestamp>/."""

    def __init__(self, root: Path | None = None, name: str | None = None):
        self.dir = (root or ART / "runs") / (name or time.strftime("%Y%m%d-%H%M%S"))
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        self.params: dict = {}
        self._metrics = (self.dir / "metrics.jsonl").open("w", encoding="utf-8")

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._metrics.write(json.dumps({"step": step, **metrics}) + "\n")

    def log_figure(self, path: Path) -> None:
        shutil.copy2(path, self.dir / "figures" / path.name)

    def finish(self, summary: dict) -> Path:
        self._metrics.close()
        (self.dir / "params.json").write_text(json.dumps(self.params, indent=2), encoding="utf-8")
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = ["# Reproducibility log", "", f"Logged: {time.strftime('%Y-%m-%d %H:%M')} (date of logging, not of development)", ""]
        lines += [f"- {k}: {v}" for k, v in summary.items()]
        (self.dir / "run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.dir / "report.html").write_text(self._html(summary), encoding="utf-8")
        return self.dir

    def _html(self, summary: dict) -> str:
        """One page that opens in any browser: summary, model table, DeepSurv loss chart and the figures."""
        import html
        rows = [json.loads(x) for x in (self.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x]
        models: dict = {}
        for r in rows:
            for k, v in r.items():
                if k.startswith("validation/"):
                    _, m, metric = k.split("/")
                    models.setdefault(m, {})[metric] = v
        loss = [r for r in rows if "deepsurv/train_loss" in r]
        chart = ""
        if loss:
            w, h, pad = 520, 220, 34
            vals = [r["deepsurv/train_loss"] for r in loss] + [r["deepsurv/holdout_loss"] for r in loss]
            lo, hi, n = min(vals), max(vals), len(loss)

            def pts(key):
                return " ".join(f"{pad + i * (w - 2 * pad) / max(n - 1, 1):.1f},"
                                f"{h - pad - (r[key] - lo) / (hi - lo or 1) * (h - 2 * pad):.1f}" for i, r in enumerate(loss))
            chart = (f'<svg viewBox="0 0 {w} {h}" width="{w}"><rect width="{w}" height="{h}" fill="#fff" stroke="#ccc"/>'
                     f'<polyline fill="none" stroke="#3b66e0" stroke-width="2" points="{pts("deepsurv/train_loss")}"/>'
                     f'<polyline fill="none" stroke="#c2410c" stroke-width="2" points="{pts("deepsurv/holdout_loss")}"/>'
                     f'<text x="{pad}" y="16" font-size="12" fill="#3b66e0">training loss</text>'
                     f'<text x="{pad + 110}" y="16" font-size="12" fill="#c2410c">hold-out loss</text>'
                     f'<text x="{w / 2}" y="{h - 8}" font-size="11" text-anchor="middle">epoch 1 to {n}</text></svg>')
        trs = "".join(f"<tr><td>{html.escape(m)}</td>" + "".join(f"<td>{v.get(k, float('nan')):.3f}</td>" for k in
                      ("uno_c", "ibs", "cal_err_3m", "cal_err_6m", "cal_err_12m")) + "</tr>" for m, v in models.items())
        figs = "".join(f'<figure><img src="figures/{p.name}" loading="lazy"><figcaption>{p.stem}</figcaption></figure>'
                       for p in sorted((self.dir / "figures").glob("*.png")))
        summ = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items())
        style = ("body{font-family:Segoe UI,Arial,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#15203a;"
                 "background:#f5f7fb}section{background:#fff;border:1px solid #e3e8f1;border-radius:12px;padding:16px 20px;"
                 "margin:16px 0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e3e8f1;padding:6px 8px;"
                 "text-align:left;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));"
                 "gap:12px}img{width:100%;border:1px solid #e3e8f1;border-radius:8px}figcaption{font-size:12px;color:#5a6682}"
                 ".note{color:#5a6682;font-size:13px}svg{max-width:100%;height:auto}")
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Reproducibility log</title>'
                f'<style>{style}</style></head><body><h1>Reproducibility log</h1>'
                f'<p class="note">Logged {time.strftime("%Y-%m-%d %H:%M")} from the saved pipeline results. This is a '
                f'reproducibility log, not a record of the original development. Nothing was refitted, the locked test '
                f'result was loaded (not re-evaluated) and no survey answers are included.</p>'
                f'<section><h2>Summary</h2><table>{summ}</table></section>'
                f'<section><h2>Candidate models on the validation set</h2><p class="note">Selected: '
                f'{html.escape(str(summary.get("selected_model", "")))} (highest validation Uno\'s C; within 0.02, the '
                f'simpler and better-calibrated model).</p><table><tr><th>Model</th><th>Uno\'s C</th><th>IBS</th>'
                f'<th>Cal. error 3 m</th><th>6 m</th><th>12 m</th></tr>{trs}</table></section>'
                f'<section><h2>DeepSurv training</h2>{chart}<p class="note">Best epoch: '
                f'{summary.get("deepsurv_best_epoch")}; early stopping kept those weights.</p></section>'
                f'<section><h2>Figures</h2><div class="grid">{figs}</div></section></body></html>')


class WandbTracker:
    """Same interface, backed by Weights & Biases. Offline by default; set WANDB_MODE=online to upload directly."""

    def __init__(self, project: str = "employability-survival"):
        try:
            import wandb
        except ImportError as e:                                        # optional dependency
            raise SystemExit("W&B is not installed: pip install -r requirements-tracking.txt") from e
        os.environ.setdefault("WANDB_MODE", "offline")
        self.wandb = wandb
        self.run = wandb.init(project=project, job_type="reproducibility-log", reinit=True)

    def log_params(self, params: dict) -> None:
        self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.run.log(metrics, step=step) if step is not None else self.run.log(metrics)

    def log_figure(self, path: Path) -> None:
        self.run.log({f"figures/{path.stem}": self.wandb.Image(str(path))})

    def finish(self, summary: dict):
        self.run.summary.update(summary)
        self.run.finish()
        return self.run.dir


def make(kind: str | None):
    """None -> no tracking (the default); 'local' or 'wandb' -> a tracker."""
    if kind in (None, "", "none"):
        return None
    if kind == "local":
        return LocalTracker()
    if kind == "wandb":
        return WandbTracker()
    raise ValueError(f"unknown tracker {kind!r}; use 'local' or 'wandb'")


def data_fingerprint() -> dict:
    """SHA-256 of each real raw file (contents are hashed, never logged)."""
    out = {}
    for p in sorted(RAW.glob("*.csv")):
        if "SYNTHETIC" not in p.name:
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def _read_json(art: Path, name: str) -> dict:
    p = art / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def log_pipeline_run(tracker, art: Path | None = None) -> dict:
    """Log the saved results of the last pipeline run. Reads artifacts only; fits and evaluates nothing."""
    art = art or ART
    selection, test, lc, ds = (_read_json(art, n) for n in ("selection.json", "test_metrics.json", "learning_curve.json",
                                                       "deepsurv_curves.json"))
    tracker.log_params({"data_fingerprint": data_fingerprint(),
                        "split": {"train": lc.get("n_train"), "validation": lc.get("n_validation"), "test": test.get("n")},
                        "selection_rule": "highest validation Uno's C; within 0.02, the simpler and better-calibrated model"})
    comparison = art / "model_comparison.csv"
    if comparison.exists():
        with comparison.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tracker.log_params({f"settings/{row['model']}": json.loads(row["tuned_params"] or "{}")})
                tracker.log_metrics({f"validation/{row['model']}/uno_c": float(row["uno_c_mean"]),
                                     f"validation/{row['model']}/ibs": float(row["ibs_mean"]),
                                     f"validation/{row['model']}/cal_err_3m": float(row["cal_err_3m_mean"]),
                                     f"validation/{row['model']}/cal_err_6m": float(row["cal_err_6m_mean"]),
                                     f"validation/{row['model']}/cal_err_12m": float(row["cal_err_12m_mean"])})
    for epoch, (tr, va) in enumerate(zip(ds.get("train_loss", []), ds.get("val_loss", [])), 1):
        tracker.log_metrics({"deepsurv/train_loss": tr, "deepsurv/holdout_loss": va}, step=epoch)
    for point in lc.get("curve", []):
        tracker.log_metrics({"learning_curve/" + k: v for k, v in point.items() if isinstance(v, (int, float))})
    for fig in sorted((art / "figures").glob("*.png")):
        tracker.log_figure(fig)
    summary = {"selected_model": selection.get("label"),
               "validation_uno_c": selection.get("uno_c_validation"),
               "test_uno_c": selection.get("uno_c"),
               "test_uno_c_ci": selection.get("uno_c_test_ci"),
               "test_result": TEST_LABEL,
               "deepsurv_best_epoch": ds.get("best_epoch"),
               "bootstrap_models": selection.get("bootstrap_models")}
    tracker.finish(summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["local", "wandb"], default="local")
    a = ap.parse_args()
    t = make(a.track)
    s = log_pipeline_run(t)
    print(json.dumps(s, indent=2))
    if isinstance(t, LocalTracker):
        print(f"run logged to {t.dir.relative_to(ROOT)}")
