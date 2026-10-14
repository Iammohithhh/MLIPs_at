"""Build the Atlas structure gallery with ASE and write Atlas-format JSON.

Every structure records how it was made (the exact ASE call and inputs), so a
viewer of the site can rebuild it. Lattice constants here are *inputs*, stated
as such, not measurements; molecular geometries come from the G2 set in ASE,
whose coordinates are MP2(full)/6-31G(d) optimised geometries (ase/data/g2_1.py).

Usage: python -m atlas_pipeline.structures ../atlas/src/data/structures
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import ase
import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, molecule

G2_NOTE = (
    "Coordinates from the G2 set as tabulated in ASE: MP2(full)/6-31G(d) optimised geometries "
    "(Curtiss et al., J. Chem. Phys. 106, 1063 (1997))."
)


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    source: str
    build: Callable[[], Atoms]
    notes: str = ""


def _au_slab() -> Atoms:
    slab = fcc111("Au", size=(4, 4, 4), a=4.078, vacuum=8.0, orthogonal=False)
    slab.center(axis=2)
    return slab


RECIPES: list[Recipe] = [
    Recipe(
        "si-diamond",
        "Silicon, diamond structure (conventional cell)",
        "ase.build.bulk('Si', 'diamond', a=5.431, cubic=True)",
        lambda: bulk("Si", "diamond", a=5.431, cubic=True),
        "a = 5.431 Å is an input chosen close to the experimental room-temperature value.",
    ),
    Recipe(
        "nacl-rocksalt",
        "Sodium chloride, rock salt (conventional cell)",
        "ase.build.bulk('NaCl', 'rocksalt', a=5.64, cubic=True)",
        lambda: bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        "a = 5.64 Å is an input (nearest-neighbour Na–Cl distance a/2 = 2.82 Å).",
    ),
    Recipe(
        "cu-fcc",
        "Copper, fcc (conventional cell)",
        "ase.build.bulk('Cu', 'fcc', a=3.615, cubic=True)",
        lambda: bulk("Cu", "fcc", a=3.615, cubic=True),
        "a = 3.615 Å is an input.",
    ),
    Recipe(
        "zno-wurtzite",
        "Zinc oxide, wurtzite (hexagonal cell, γ = 120°)",
        "ase.build.bulk('ZnO', 'wurtzite', a=3.25, c=5.21, u=0.382)",
        lambda: bulk("ZnO", "wurtzite", a=3.25, c=5.21, u=0.382),
        "a, c and the internal parameter u are inputs; the cell is not orthogonal.",
    ),
    Recipe(
        "au111-slab",
        "Au(111) slab, 4 × 4 × 4 layers, vacuum along c",
        "ase.build.fcc111('Au', size=(4, 4, 4), a=4.078, vacuum=8.0, orthogonal=False); centred along z",
        _au_slab,
        "Periodic in-plane only; a = 4.078 Å is an input. 16 Å of vacuum separates the slab from its images along z.",
    ),
    Recipe("h2o", "Water (G2 geometry)", "ase.build.molecule('H2O')", lambda: molecule("H2O"), G2_NOTE),
    Recipe("benzene", "Benzene (G2 geometry)", "ase.build.molecule('C6H6')", lambda: molecule("C6H6"), G2_NOTE),
    Recipe(
        "ethanol",
        "Ethanol (G2 geometry)",
        "ase.build.molecule('CH3CH2OH')",
        lambda: molecule("CH3CH2OH"),
        G2_NOTE,
    ),
]


def to_atlas(atoms: Atoms, recipe: Recipe) -> dict:
    """Convert ASE Atoms to the Atlas structure format (Å, Cartesian, rows = lattice vectors)."""
    pbc = [bool(x) for x in atoms.pbc]
    cell = atoms.cell.array.tolist() if any(pbc) else None
    r = lambda v: [round(float(x), 6) for x in v]  # noqa: E731
    return {
        "format": "atlas-structure",
        "version": 1,
        "id": recipe.id,
        "title": recipe.title,
        "symbols": atoms.get_chemical_symbols(),
        "positions": [r(p) for p in atoms.positions],
        "cell": [r(v) for v in cell] if cell is not None else None,
        "pbc": pbc,
        "provenance": {
            "source": recipe.source,
            "generator": f"ase {ase.__version__}",
            "created": date.today().isoformat(),
            "notes": recipe.notes,
        },
    }


def build_all() -> list[dict]:
    out = []
    for recipe in RECIPES:
        atoms = recipe.build()
        out.append(to_atlas(atoms, recipe))
    return out


def validate(s: dict) -> None:
    """Sanity checks shared with the tests: shapes, finite numbers, no overlapping atoms."""
    n = len(s["symbols"])
    pos = np.array(s["positions"])
    assert pos.shape == (n, 3), "positions must be n × 3"
    assert np.isfinite(pos).all()
    if s["cell"] is not None:
        cell = np.array(s["cell"])
        assert cell.shape == (3, 3) and abs(np.linalg.det(cell)) > 1e-6
    a = Atoms(s["symbols"], positions=pos, cell=s["cell"] or np.zeros((3, 3)), pbc=s["pbc"])
    if n > 1:
        d = a.get_all_distances(mic=any(s["pbc"]))
        np.fill_diagonal(d, np.inf)
        assert d.min() > 0.5, f"{s['id']}: atoms closer than 0.5 Å"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for s in build_all():
        validate(s)
        (out / f"{s['id']}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        index.append({"id": s["id"], "title": s["title"], "atoms": len(s["symbols"])})
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(index)} structures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
