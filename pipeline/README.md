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

`uniffbench.py` copies the per-run experimental and simulated densities from the UniFFBench
result files (Mannan et al., arXiv:2508.05762; MIT licence) into
`atlas/src/data/benchmarks/uniffbench-density.json`, with the upstream commit. Only complete
result files are taken; the site computes all errors and statistics (`reference-gap.ts`).

```bash
git clone --depth 1 https://github.com/M3RG-IITD/UniFFBench uniff
../.venv/bin/python -m atlas_pipeline.uniffbench uniff ../atlas/src/data/benchmarks/uniffbench-density.json
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

`xray_nequip.py` does the same for NequIP with the NequIP-OAM-S 0.1 package (Zenodo
10.5281/zenodo.18775904, CC-BY-4.0), writing `atlas/src/data/xray/nequip-oam-s.json`: two message-passing
layers, the gate, the per-type scale and shift and the ZBL repulsion are hooked separately, with a rotated and a
mirrored copy of ethanol and three dimer scans. A `.nequip.zip` is a `torch.package` archive whose normal loading
imports and runs the Python stored inside it; the script instead reads the pickled module tree with a restricted
unpickler (tensors and inert stubs only), reads the architecture off that tree, rebuilds it with the installed
`nequip` (pinned to the packaging version 0.15.0) and loads the tensors with `strict=True`.

```bash
python3 -m venv ../.venv-nequip
../.venv-nequip/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
../.venv-nequip/bin/pip install -r requirements-xray-nequip.txt
curl -L -o NequIP-OAM-S-0.1.nequip.zip https://zenodo.org/records/18775904/files/NequIP-OAM-S-0.1.nequip.zip
../.venv-nequip/bin/python -m atlas_pipeline.xray_nequip NequIP-OAM-S-0.1.nequip.zip ../atlas/src/data/structures ../atlas/src/data/xray/nequip-oam-s.json
ATLAS_NEQUIP_PACKAGE=NequIP-OAM-S-0.1.nequip.zip ../.venv-nequip/bin/python -m pytest -q tests/test_xray_nequip.py
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

### Representation Map

`representation.py` passes every atom of a 98-structure reference set (and its rotated and mirrored
copies) through six models and writes `atlas/src/data/representation/map.json`. Each family runs in
its Observatory environment; the partials (`.npz`, raw features, not committed) are merged with numpy:

```bash
../.venv-obs-mace/bin/python -m atlas_pipeline.representation extract mace 2023-12-03-mace-128-L1_epoch-199.model 2023-12-10-mace-128-L0_energy_epoch-249.model mace.npz
../.venv-obs-chgnet/bin/python -m atlas_pipeline.representation extract chgnet chgnet.npz
../.venv-obs-sevenn/bin/python -m atlas_pipeline.representation extract sevennet sevennet.npz
../.venv-obs-orb/bin/python -m atlas_pipeline.representation extract orb orb-mptraj-only-v2-20241014.ckpt orb.npz
../.venv-obs-orb/bin/python -m atlas_pipeline.representation merge ../atlas/src/data/representation/map.json mace.npz chgnet.npz sevennet.npz orb.npz
```

About ten minutes on four CPU cores (one thread per model).

### Cleavage-energy benchmark tables

`cleavage.py` parses Tables 1–2 of Mehdizadeh & Schindler (arXiv:2508.21663v1) from the arXiv HTML into
`atlas/src/data/benchmarks/cleavage-mehdizadeh2025.json`, copying numbers as printed and failing if the two
tables' model lists do not match:

```bash
curl -sL https://arxiv.org/html/2508.21663v1 -o cleavage.html
../.venv/bin/python -m atlas_pipeline.cleavage cleavage.html ../atlas/src/data/benchmarks/cleavage-mehdizadeh2025.json
```

### Catalysis error tables

`catalysis.py` parses Supplementary Tables S2–S15 of Kempen, Cheula & Andersen (arXiv:2512.16702v1) — the RMSE and
maximum error of 80 foundation potentials (and the paper's task-specific baselines) for 14 catalysis target properties —
into `atlas/src/data/benchmarks/catalysis-kempen2025.json`, expanding row spans and failing unless all 14 tables parse:

```bash
curl -sL https://arxiv.org/html/2512.16702v1 -o catalysis.html
../.venv/bin/python -m atlas_pipeline.catalysis catalysis.html ../atlas/src/data/benchmarks/catalysis-kempen2025.json
```

### Digest of additions

`digest.py` reads the repository history (`git log --diff-filter=A`) for content records and pages and writes
`atlas/src/data/digest.json`, which the "What changed" page renders. It refuses to run on a shallow clone:

```bash
../.venv/bin/python -m atlas_pipeline.digest .. ../atlas/src/data/digest.json
```

### Migration barriers (battery lab)

`migration_barriers.py` joins the DFT-NEB barriers of Bheemaguli et al.'s Dataset-2 (Zenodo) with each model's released
NEB barrier and writes `atlas/src/data/benchmarks/migration-bheemaguli2025.json` plus the hop geometries in
`atlas/public/data/migration-paths.json`; see the module docstring for the sparse-clone commands.

### Citation universe

`citations.py` builds the literature around the Atlas papers from OpenAlex (CC0): every Atlas paper, every work it
cites and the most-cited works citing it, with all citations among them, laid out by a seeded 3D force layout with
rank-equalised radii, into `atlas/public/data/citation-universe.json`. Pass a cache path to avoid refetching:

```bash
../.venv/bin/python -m atlas_pipeline.citations ../atlas/src/data/bibliography.json ../atlas/public/data/citation-universe.json /tmp/oa-works.json
```
