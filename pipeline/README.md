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

## Benchmark snapshots

`matbench.py` copies Matbench Discovery model metadata and metrics, verbatim, from a
local checkout of the benchmark repository into `atlas/src/data/benchmarks/matbench-discovery.json`,
with the upstream commit hash and date. Re-run to update; never edit the JSON by hand.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/janosh/matbench-discovery mbd
git -C mbd sparse-checkout set models
../.venv/bin/python -m atlas_pipeline.matbench mbd ../atlas/src/data/benchmarks/matbench-discovery.json
```

## Architecture X-ray tensors

`xray.py` passes gallery structures through the MACE-MP-0a medium checkpoint on CPU and
captures every intermediate tensor with forward hooks, writing
`atlas/src/data/xray/mace-mp-0a-medium.json`. The checkpoint is a pickled module whose
`torch.fx` parts would execute stored source if loaded normally, so the script reads it with
a restricted unpickler (tensors only) and loads those tensors into a fresh `ScaleShiftMACE`
built by the installed `mace-torch`; the file's SHA-256 is checked first.

```bash
python3 -m venv ../.venv-xray
../.venv-xray/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
../.venv-xray/bin/pip install -r requirements-xray.txt
curl -LO https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
../.venv-xray/bin/python -m atlas_pipeline.xray 2023-12-03-mace-128-L1_epoch-199.model ../atlas/src/data/structures ../atlas/src/data/xray/mace-mp-0a-medium.json
ATLAS_XRAY_CHECKPOINT=2023-12-03-mace-128-L1_epoch-199.model ../.venv-xray/bin/python -m pytest -q tests/test_xray.py
```

## Model Observatory

`observatory.py` runs four released checkpoints trained on MPtrj (MACE-MP-0a medium, CHGNet
0.3.0, SevenNet-0, Orb v2 MPtrj-only) on exactly the same inputs, built deterministically from
the structure gallery: rattled gallery structures, E(V) scans of Cu and Si, Si₂/Cu₂ dimer
curves, a Na adatom above Au(111), a rigid-rotation test and a closed loop in configuration
space. Each model runs in its own environment (see `requirements-observatory.txt`) and writes
a partial file; `merge` writes `atlas/src/data/observatory/observatory.json`.

Before any package loads a checkpoint, `scan_pickle_globals` lists the globals its pickle would
import, reading opcodes only, and refuses anything outside a list of data constructors. The
MACE file stores Python source and is rebuilt from tensors with `xray.build_model` instead.

```bash
S=../atlas/src/data/structures
curl -LO https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
curl -LO https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt
../.venv-obs-mace/bin/python -m atlas_pipeline.observatory run mace-mp-0a-medium $S mace.json --ckpt 2023-12-03-mace-128-L1_epoch-199.model
../.venv-obs-chgnet/bin/python -m atlas_pipeline.observatory run chgnet-0.3.0 $S chgnet.json
../.venv-obs-sevenn/bin/python -m atlas_pipeline.observatory run sevennet-0 $S sevennet.json
../.venv-obs-orb/bin/python -m atlas_pipeline.observatory run orb-v2-mptrj $S orb.json --ckpt orb-mptraj-only-v2-20241014.ckpt
../.venv/bin/python -m atlas_pipeline.observatory merge ../atlas/src/data/observatory/observatory.json $S mace.json chgnet.json sevennet.json orb.json
```

Run the models one after another: four PyTorch processes on four cores oversubscribe threads
and slow each other down by orders of magnitude.
