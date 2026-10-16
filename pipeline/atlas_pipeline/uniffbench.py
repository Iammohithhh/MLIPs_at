"""Snapshot the UniFFBench per-mineral density results into Atlas JSON.

UniFFBench (Mannan et al., arXiv:2508.05762) runs 50 ps NPT molecular dynamics of experimentally
determined mineral structures with universal MLIPs and compares the time-averaged density with the
density of the experimental structure. The repository ships, for some models, one CSV row per
completed run with both densities. This module copies those two numbers verbatim, with the
mineral name and the temperature/pressure fields encoded in the run name, and records the
upstream commit. It does not filter, average or rank; analysis happens in the site code
(atlas/src/lib/physics/reference-gap.ts), where it is tested.

Only models with a complete result file are taken (a file with fewer than MIN_RUNS rows is a
partial upload, not a result set).

Usage:
  git clone --depth 1 https://github.com/M3RG-IITD/UniFFBench uniff
  python -m atlas_pipeline.uniffbench uniff ../atlas/src/data/benchmarks/uniffbench-density.json
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_URL = "https://github.com/M3RG-IITD/UniFFBench"

# Result file per model, and what the repository's loader at the snapshot commit uses.
# The CSVs themselves do not record a checkpoint, so `loader` is labelled as such.
MODELS = [
    {
        "key": "orb",
        "name": "Orb",
        "file": "Post_processing/results/orb/orb.csv",
        "loader": "md_simulation/models/_orb.py calls pretrained.orb_v2()",
    },
    {
        "key": "mattersim",
        "name": "MatterSim",
        "file": "Post_processing/results/mattersim/mattersim.csv",
        "loader": 'md_simulation/models/_mattersim.py loads "mattersim-v1.0.0-1m"',
    },
]

MIN_RUNS = 500
RHO_EXP = "Exp_Density (g/cm³)"
RHO_SIM = "Sim_Density (g/cm³)"
RUN_NAME = re.compile(r"^\d+_(?P<mineral>.+)_(?P<t>-?\d+(?:\.\d+)?)_(?P<p>-?\d+(?:\.\d+)?)_$")


def parse_run_name(name: str) -> tuple[str, float, float]:
    """'0_Thorianite_298.00_1.01_' -> ('Thorianite', 298.0, 1.01). Units of the third field are not stated upstream."""
    m = RUN_NAME.match(name)
    if not m:
        raise ValueError(f"unexpected run name {name!r}")
    return m["mineral"], float(m["t"]), float(m["p"])


def read_runs(path: Path) -> list[dict]:
    runs = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mineral, t, p = parse_run_name(row["Filename"])
            if row["Filename"] in seen:
                raise ValueError(f"duplicate run {row['Filename']!r} in {path}")
            seen.add(row["Filename"])
            exp, sim = float(row[RHO_EXP]), float(row[RHO_SIM])
            if not (exp > 0 and sim > 0):
                raise ValueError(f"non-positive density in {row['Filename']!r}")
            runs.append({"mineral": mineral, "T": t, "p_field": p, "rho_exp": round(exp, 5), "rho_sim": round(sim, 5)})
    return runs


def commit_of(repo: Path) -> tuple[str, str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%H %cs"], check=True, capture_output=True, text=True
    ).stdout.split()
    return out[0], out[1]


def build(repo: Path, commit: str, commit_date: str) -> dict:
    models = []
    for m in MODELS:
        runs = read_runs(repo / m["file"])
        if len(runs) < MIN_RUNS:
            raise ValueError(f"{m['file']}: only {len(runs)} runs")
        models.append({"key": m["key"], "name": m["name"], "file": m["file"], "loader": m["loader"], "runs": runs})
    return {
        "source": {
            "paper": "mannan2025uniffbench",
            "repository": REPO_URL,
            "commit": commit,
            "commit_date": commit_date,
            "licence": "MIT",
            "retrieved": date.today().isoformat(),
        },
        "units": {"rho_exp": "g/cm³", "rho_sim": "g/cm³", "T": "K", "p_field": "as encoded in the run name; unit not stated"},
        "definitions": {
            "rho_exp": "density of the experimental crystal structure (CIF) the run starts from",
            "rho_sim": "density averaged over the 50 ps NPT trajectory at the run's temperature and pressure",
        },
        "models": models,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    repo = Path(argv[1])
    commit, commit_date = commit_of(repo)
    data = build(repo, commit, commit_date)
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out}: " + ", ".join(f"{m['name']} {len(m['runs'])} runs" for m in data["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
