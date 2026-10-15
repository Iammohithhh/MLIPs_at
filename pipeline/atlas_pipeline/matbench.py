"""Snapshot Matbench Discovery model metadata and metrics into Atlas JSON.

Reads every models/<family>/<model>.yml from a local checkout of
https://github.com/janosh/matbench-discovery and writes one JSON file with the
source commit and date, so every number on the Atlas is traceable to an exact
upstream revision. Numbers are copied, never edited or combined; the Atlas does
not compute rankings.

Usage:
  git clone --depth 1 --filter=blob:none --sparse https://github.com/janosh/matbench-discovery mbd
  git -C mbd sparse-checkout set models
  python -m atlas_pipeline.matbench mbd ../atlas/src/data/benchmarks/matbench-discovery.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DISCOVERY_KEYS = ["F1", "DAF", "Precision", "Recall", "Accuracy", "MAE", "RMSE", "R2"]


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _pick(d: dict | None, keys: list[str]) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    return {k: v for k in keys if (v := _num(d.get(k))) is not None}


def model_record(path: Path) -> dict | None:
    """One model entry, or None if the file has no discovery metrics."""
    y = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = y.get("metrics") or {}
    disc = metrics.get("discovery") or {}
    unique = _pick(disc.get("unique_prototypes"), DISCOVERY_KEYS)
    full = _pick(disc.get("full_test_set"), DISCOVERY_KEYS)
    if not unique and not full:
        return None
    geo = (metrics.get("geo_opt") or {}).get("symprec=1e-5") or {}
    kappa = ((metrics.get("phonons") or {}).get("kappa_103")) or {}
    lic = y.get("license") or {}
    return {
        "key": y.get("model_key") or path.stem,
        "name": y.get("model_name") or path.stem,
        "version": str(y.get("model_version")) if y.get("model_version") is not None else None,
        "family_dir": path.parent.name,
        "added": (y.get("dates") or {}).get("benchmark_added"),
        "openness": y.get("openness"),
        "training_sets": [str(t) for t in (y.get("training_sets") or [])],
        "params": _num(y.get("model_params")),
        "targets": y.get("targets"),
        "licence_code": lic.get("code"),
        "licence_checkpoint": lic.get("checkpoint"),
        "paper": y.get("paper"),
        "repo": y.get("repo"),
        "checkpoint_url": y.get("checkpoint_url"),
        "metrics": {
            "discovery_unique": unique,
            "discovery_full": full,
            "rmsd": _num(geo.get("rmsd")),
            "kappa_srme": _num(kappa.get("κ_SRME")),
        },
    }


def snapshot(repo: Path) -> dict:
    models = []
    for f in sorted((repo / "models").glob("*/*.yml")):
        rec = model_record(f)
        if rec:
            models.append(rec)
    try:
        commit, commit_date = (
            subprocess.check_output(["git", "-C", str(repo), "log", "-1", "--format=%H|%cs"], text=True).strip().split("|")
        )
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        commit, commit_date = "unknown", "unknown"
    return {
        "source": {
            "repo": "https://github.com/janosh/matbench-discovery",
            "commit": commit,
            "commit_date": commit_date,
            "fetched": date.today().isoformat(),
            "files": "models/*/*.yml",
        },
        "models": sorted(models, key=lambda m: (m["added"] or "", m["key"])),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    out = snapshot(Path(argv[1]))
    path = Path(argv[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(out['models'])} models at {out['source']['commit'][:10]} ({out['source']['commit_date']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
