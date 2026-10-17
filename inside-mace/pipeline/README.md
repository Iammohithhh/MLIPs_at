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
.venv/bin/python -m pytest -q tests                                   # data checks
INSIDE_MACE_CHECKPOINT=ckpt.model .venv/bin/python -m pytest -q tests  # + re-evaluation
```

- `inside_mace/capture.py` — copied from `pipeline/atlas_pipeline/xray.py` (MLIP Atlas): the restricted
  unpickler (the checkpoint is never unpickled as code), model rebuild, forward hooks.
- `inside_mace/intervention.py` — the Atom Intervention experiments: displacement frames, captured
  intermediates, baseline restoration, finite-difference force checks, rotated-input runs, provenance.

The checkpoint SHA-256 is verified before use; a mismatch aborts the run. A frame whose evaluation raises
is stored with `status: "failed"` and its error message, and no numbers.
