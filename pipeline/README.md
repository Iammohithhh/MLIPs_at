# MLIP Atlas data pipeline

Build-time Python that produces the data the static site ships. CPU only.

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
../.venv/bin/python -m pytest -q tests
../.venv/bin/python -m atlas_pipeline.elements ../atlas/src/lib/structure/elements.generated.ts
../.venv/bin/python -m atlas_pipeline.structures ../atlas/src/data/structures
```

- `elements.py` — element table (Cordero 2008 covalent radii, Jmol colours) from ASE.
- `structures.py` — the structure gallery in Atlas format (`atlas-structure` v1: Å, Cartesian
  positions, lattice vectors as rows, `pbc`, provenance with the exact ASE call).
