"""Snapshot the migration-barrier benchmark of Bheemaguli, Xiao & Sai Gautam into Atlas JSON.

Joins the DFT-NEB barriers of Dataset-2 (Zenodo 10.5281/zenodo.17483476, `Dataset-2.json`, field
`target`) with each foundation model's NEB barrier from the authors' repository
(github.com/sai-mat-group/mlips-migration-barriers, `Dataset-2/<system>/<model>/*.csv`, last line
"Migration Barrier,<eV>"). Numbers are copied, never edited. Summary statistics are recomputed from the
joined data so they can be compared with the paper (Sec. 3.1) as a reproduction check.

Usage (sparse clone keeps only the CSV files):
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/sai-mat-group/mlips-migration-barriers repo
  git -C repo sparse-checkout set --no-cone '/Dataset-2/**/*.csv' && git -C repo checkout
  curl -sL https://zenodo.org/api/records/17483476/files/Dataset-2.json/content -o Dataset-2.json
  python -m atlas_pipeline.migration_barriers Dataset-2.json repo ../atlas/src/data/benchmarks/migration-bheemaguli2025.json       ../atlas/public/data/migration-paths.json   # optional: endpoint geometries for the 3D hop viewer
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

# Folder name in the repository → model label used in the paper.
MODELS = {
    "MACE": "MACE-MP-0",
    "SevenNet_run_ACTUAL": "SevenNet",
    "Orb-v3": "Orb-v3",
    "CHGNET": "CHGNet",
    "M3GNET": "M3GNet",
}
FILES = ("neb_results.csv", "energies.csv")


def read_barrier(path: Path) -> float | None:
    """The last 'Migration Barrier,<value>' line of an NEB results file."""
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("Migration Barrier"):
            try:
                return float(line.split(",", 1)[1])
            except ValueError:
                return None
    return None


def model_barrier(system_dir: Path, folder: str) -> float | None:
    for name in FILES:
        f = system_dir / folder / name
        if f.is_file():
            return read_barrier(f)
    return None


def summary(rows: list[dict], model: str) -> dict:
    pairs = [(r["dft"], r["models"][model]) for r in rows if r["models"].get(model) is not None]
    err = [m - d for d, m in pairs]
    n = len(err)
    return {
        "n": n,
        "mae": sum(abs(e) for e in err) / n if n else None,
        "under": sum(e < 0 for e in err),
        "over": sum(e > 0 for e in err),
    }


def build(dataset: list[dict], repo: Path, commit: str | None = None) -> dict:
    rows = []
    for e in dataset:
        sysdir = repo / "Dataset-2" / e["sys_name"]
        rows.append(
            {
                "id": e["sys_name"],
                "formula": e["formula"],
                "class": e.get("crystal_class"),
                "dft": float(e["target"]),
                "models": {label: model_barrier(sysdir, folder) for folder, label in MODELS.items()},
            }
        )
    return {
        "source": {
            "paper": "bheemaguli2025migration",
            "zenodo": "10.5281/zenodo.17483476",
            "repository": "https://github.com/sai-mat-group/mlips-migration-barriers",
            "commit": commit,
            "retrieved": date.today().isoformat(),
        },
        "models": list(MODELS.values()),
        "summary": {m: summary(rows, m) for m in MODELS.values()},
        "systems": rows,
    }


def _cart(struct: dict) -> list[list[float]]:
    lat = struct["lattice_mat"]
    if struct.get("cartesian"):
        return [list(map(float, c)) for c in struct["coords"]]
    return [[sum(f[k] * lat[k][i] for k in range(3)) for i in range(3)] for f in struct["coords"]]


def hop(entry: dict) -> dict:
    """Endpoint geometry of one migration path: the atom that moves most between the initial and final
    structures (minimum image in fractional coordinates) is taken as the migrating ion."""
    ini, fin = entry["structure_ini"], entry["structure_fin"]
    lat = ini["lattice_mat"]
    if ini["elements"] != fin["elements"]:
        raise ValueError(f"{entry['sys_name']}: initial and final structures list different atoms")
    if ini.get("cartesian") or fin.get("cartesian"):
        raise ValueError(f"{entry['sys_name']}: expected fractional coordinates")
    best, disp = -1, None
    for i, (a, b) in enumerate(zip(ini["coords"], fin["coords"])):
        d = [(b[k] - a[k]) - round(b[k] - a[k]) for k in range(3)]
        dc = [sum(d[k] * lat[k][j] for k in range(3)) for j in range(3)]
        n = sum(x * x for x in dc) ** 0.5
        if best < 0 or n > disp[1]:
            best, disp = i, (dc, n)
    pos = _cart(ini)
    end = [pos[best][j] + disp[0][j] for j in range(3)]
    r = lambda v: round(v, 3)
    return {
        "id": entry["sys_name"],
        "lattice": [[r(x) for x in row] for row in lat],
        "elements": ini["elements"],
        "positions": [[r(x) for x in p] for p in pos],
        "mover": best,
        "end": [r(x) for x in end],
        "hop": r(disp[1]),
    }


def main(argv: list[str]) -> int:
    if len(argv) not in (4, 5):
        print(__doc__)
        return 2
    if len(argv) == 5:
        paths = [hop(e) for e in json.loads(Path(argv[1]).read_text(encoding="utf-8"))]
        Path(argv[4]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[4]).write_text(json.dumps({"source": "Zenodo 10.5281/zenodo.17483476 (Dataset-2.json)", "paths": paths}, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {argv[4]}: {len(paths)} migration paths")
    dataset = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    repo = Path(argv[2])
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    data = build(dataset, repo, commit)
    out = Path(argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for m, s in data["summary"].items():
        print(f"{m:10s} n={s['n']:3d} MAE={s['mae']:.3f} eV under/over={s['under']}/{s['over']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
