# inside-mace pipeline (CPU)

Generates the precomputed MACE evaluations used by the site. Every number shown on the page comes
from a real forward pass of the **MACE-MP-0a medium** checkpoint
(`2023-12-03-mace-128-L1_epoch-199.model`, MIT, SHA-256
`01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845`).

```bash
python -m venv .venv
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
curl -L -o ckpt.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
.venv/bin/python -m inside_mace.intervention ckpt.model ../../atlas/src/data/structures ../src/data/experiments
#   optional 4th argument: experiment ids to (re)run, e.g. `zno-axial-bond`; the others keep their data
.venv/bin/python -m inside_mace.scales ../src/data/experiments/si-bond-stretch    # fixed heatmap scales
.venv/bin/python -m inside_mace.scales ../src/data/experiments/zno-axial-bond
.venv/bin/python -m pytest -q tests                                   # data checks
INSIDE_MACE_CHECKPOINT=ckpt.model .venv/bin/python -m pytest -q tests  # + re-evaluation
```

- `inside_mace/capture.py` — copied from `pipeline/atlas_pipeline/xray.py` (MLIP Atlas): the restricted
  unpickler (the checkpoint is never unpickled as code), model rebuild, forward hooks.
- `inside_mace/intervention.py` — the Atom Intervention experiments: displacement frames, captured
  intermediates, baseline restoration, finite-difference force checks, rotated-input runs, provenance.

The checkpoint SHA-256 is verified before use; a mismatch aborts the run. A frame whose evaluation raises
is stored with `status: "failed"` and its error message, and no numbers.

## Experiments

| id | Structure | Moved atom and path | Frames |
|---|---|---|---|
| `si-bond-stretch` | Si diamond 3×3×3 (216 atoms, cubic 16.29 Å) | Si 105 (nearest the cell centre) along the [111] bond to Si 106; d > 0 shortens it | 25, d = −0.300 … +0.300 Å |
| `zno-axial-bond` | ZnO wurtzite 5×5×3 (300 atoms; a = 3.25, c = 5.21 Å, u = 0.382; in-plane width 14.07 Å, c 15.63 Å, both > 2 r_max) | the Zn nearest the cell centre along the c axis towards its axial O neighbour; d > 0 shortens that Zn–O bond | 25, d = −0.300 … +0.300 Å |

The ZnO geometry is `ase.build.bulk('ZnO', 'wurtzite', a=3.25, c=5.21, u=0.382)` (from
`atlas/src/data/structures/zno-wurtzite.json`) and was **not relaxed with the model**: unlike the Si
diamond baseline, where every force vanishes by symmetry, the wurtzite baseline carries non-zero model forces
along c (the Zn and O sites have C3v symmetry, which allows a force and l = 1 features along the c axis).
The page must say so and must not describe the ZnO baseline as an equilibrium.
