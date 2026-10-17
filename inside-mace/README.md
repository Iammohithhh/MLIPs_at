# MACE: Inside the Prediction

**Move one atom. Trace what changes inside the model.**

An interactive, static instrument that follows a real MACE checkpoint (MACE-MP-0a medium) from atomic
geometry to energy and forces, and lets you move one atom of a periodic crystal (silicon, or wurtzite zinc
oxide) along an evaluated path to see what changes inside the model. A second project in this repository, separate from the MLIP Atlas
in `atlas/` (see [BRIEF.md](BRIEF.md) for the owner's brief and [HANDOFF.md](HANDOFF.md) for the build log).

Every number on the page is a **precomputed** model evaluation (label: *Precomputed model evaluations;
interactive inspection*). Nothing is computed live, nothing is interpolated between evaluated frames, and
missing data is shown as missing, never as zero.

## Run it

```bash
cd inside-mace
npm ci                 # on Windows too: never `npm install` (it rewrites the lockfile)
npm run dev            # http://localhost:4321
npm run build          # static site in dist/
npm run check          # astro check (TypeScript strict)
npm test               # Vitest: schema, numerics, URL state, trace, demo, export text
npx astro build && npx playwright test --project=desktop --project=mobile   # browser checks (Chromium)
```

On Linux the Playwright config uses `/opt/pw-browsers/chromium` with SwiftShader WebGL; on Windows use
`--project=edge`.

## What is on the page

| Part | What it shows | Source of numbers |
|---|---|---|
| Experiment picker | Si (bond stretch) or ZnO (axial Zn–O bond); `exp` in the URL; each experiment's manifest is a separate lazily loaded chunk | `index.json`, `manifest.json` |
| Atom Intervention Lab | Moved atom's 6 Å neighbourhood (minimum image), displacement slider snapping to 25 evaluated frames, baseline / perturbed / difference / side by side (synchronised cameras, desktop), force arrows with a stated scale, colour by feature change | `manifest.json` frames |
| Guided walkthrough | Seven stages (species → neighbours → radial/angular → aggregation → higher-order features → site energies → forces), each with plain text, what changed, KaTeX equation, mace-torch 0.3.16 file:line locators and recorded tensors | `manifest.json`, `frames/NN.json` |
| Architecture diagram | The checkpoint's actual blocks and shapes; linked to the walkthrough stage | checkpoint hyperparameters |
| Which parts respond | Per-layer, per-irrep ‖Δh‖ against displacement and ΔEᵢ for the selected atom; 128-channel heatmaps | `manifest.json`, chunks |
| Trace the change | Stage by stage (edges → layer 1 → layer 2 → site energies → forces), every atom's change against its distance from the moved atom, with an optional predict-before-reveal question | `manifest.json` |
| Rotation comparison | Same structure and cell rotated 50° about (1, 2, 3) and evaluated again: E invariant, F → RF, scalars invariant, l = 1 channels → Rh | `checks.rotation` |
| Presentation mode | `?present=1&demo=1…5`: large scene, captions computed from the data, link back | `demo.ts` |
| Still image | "Save image (PNG)": scene(s) with title, status, settings, units, legends, attribution, source URL | `figureText.ts` |

Every state is a URL (`v, exp, ckpt, mode, atom, frame, stage, layer, ch, view, present, demo`); values are
validated and an obsolete or unsupported link is restored as far as possible with a visible note.

## Reproduce the data

The data in `src/data/experiments/si-bond-stretch/` and `src/data/experiments/zno-axial-bond/` was generated on CPU in float64 by
`pipeline/inside_mace/intervention.py`; see [pipeline/README.md](pipeline/README.md). In short:

```bash
cd inside-mace/pipeline
python -m venv .venv
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
curl -L -o ckpt.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
.venv/bin/python -m inside_mace.intervention ckpt.model ../../atlas/src/data/structures ../src/data/experiments
.venv/bin/python -m inside_mace.scales ../src/data/experiments/si-bond-stretch
.venv/bin/python -m inside_mace.scales ../src/data/experiments/zno-axial-bond
INSIDE_MACE_CHECKPOINT=ckpt.model .venv/bin/python -m pytest -q tests
```

The checkpoint (SHA-256 `01bfe221…35845`, verified before use) is loaded with a restricted unpickler copied
from `pipeline/atlas_pipeline/xray.py`; it is never unpickled as code. Versions: mace-torch 0.3.16, e3nn
0.4.4, torch 2.14.1+cpu, ase 3.29.0. The Si run takes about 5–6 minutes on a laptop-class CPU; ZnO about
twice that (300 atoms). A fourth argument to `inside_mace.intervention` regenerates only the named experiment ids.

**Experiment 1 (`si-bond-stretch`).** Si diamond 3×3×3 supercell (216 atoms, cubic cell 16.29 Å, periodic). Atom 105 moves along
the unit vector towards its bonded neighbour 106 (a [111] bond direction); d > 0 shortens that bond.
25 frames, d = −0.300 … +0.300 Å in 0.025 Å steps. Storage: energies in full float64; forces, site energies
and positions to 1e-7; features to 5 significant figures; change norms to 4; |x| < 1e-12 written as 0.
Every Si site has tetrahedral symmetry, so the baseline forces and layer-1 l = 1 features are zero.

**Experiment 2 (`zno-axial-bond`).** Wurtzite ZnO, `ase.build.bulk('ZnO', 'wurtzite', a=3.25, c=5.21,
u=0.382)`, 5×5×3 supercell (300 atoms, periodic). Zn atom 150 (nearest the supercell centre) moves along the
unit vector towards its axial O neighbour 149 (−c, [0001]); d > 0 shortens that Zn–O bond. Same 25 frames and
storage. The geometry was **not relaxed with the model**: the baseline already carries a force of 0.0371 eV/Å
along c on every atom and its layer-1 l = 1 features are not zero; the page derives these statements from the
stored run and never claims symmetry zeros for it.

**Domain.** MACE-MP-0 was trained on MPtrj (Materials Project PBE/PBE+U relaxations), which includes
crystalline silicon and wurtzite ZnO (mp-2133); small displacements of a bulk atom stay close to that domain. Nothing here is compared
with DFT, so no number on the page is an error estimate.

## Interpretation limits

- Feature differences are **observed changes in a fixed model under a specified input perturbation**. They
  are not explanations of chemical reasoning, confidence estimates, prediction errors, proof that a feature
  caused an output, or calibrated importance or uncertainty.
- ‖Δh‖ is the norm of the difference of a whole feature block (128 channels; ×3 components for l = 1), not
  the difference of two norms. Internal feature scales differ between layers and irreps, so each block has its
  own axis and colour scale; heights are not compared across blocks.
- Relative change divides by max(‖h_baseline‖, 0.05 × median baseline norm), so blocks that vanish by
  symmetry (layer-1 l = 1 features in the perfect Si crystal) cannot blow up the ratio.
- Site energies Eᵢ are a model-internal partition of the total energy; they are not measurable. Forces are
  −∂E/∂rᵢ of the **total** energy, not the slope of an atom's own site energy (the trace view shows atoms
  whose force changes while their site energy does not).
- "Trace the change" is a dependency order (geometry → layer 1 → layer 2 → energies → forces), not a timing
  or propagation-speed measurement. Its reach (6 Å for layer 1, up to 12 Å for layer 2) follows from the
  architecture: two message-passing layers with r_max = 6 Å.
- A plotted zero means: below 10⁻¹² for feature norms (pipeline write threshold), equal to 10⁻⁷ for stored
  energies and forces.
- One checkpoint, two structures, one displacement path each. Nothing here establishes the model's accuracy or
  reliability, and the model does not "understand" chemistry.

## Rendering decisions

- **Renderer.** The working WebGL renderer and impostor-sphere shaders from `atlas/src/three` are kept (brief:
  migrating custom shaders to WebGPU is a separate decision). Render on demand, device-pixel ratio capped at 2;
  a scene that is off screen or in a hidden tab stops rendering entirely and redraws once when it returns
  (`data-rendering="paused"` on its container; checked in `e2e/render.spec.ts`). Only the selected experiment's
  manifest is downloaded; frame chunks load per frame.
- **Ambient occlusion (brief §4, optional) — evaluated and not enabled.** Session 5 tried N8AO (n8ao 2.0.1,
  ISC) as a three.js `EffectComposer` pass behind an off-by-default toggle. The prerequisite holds: the impostor
  fragment shader writes `gl_FragDepth` (`src/three/primitives.tsx`), so depth-reconstructed occlusion would see
  the spheres rather than their billboard quads. The measured result did not justify it: on the ZnO scene
  (1440×1000, SwiftShader) mean luminance changed from 40.381 to 40.376 (no visible occlusion), while the
  composer's render target dropped multisample anti-aliasing, so 11 % of pixels changed — almost all of them
  thin graph edges and the selection halo becoming aliased. Muddier lines and an extra full-screen pass on
  phones for no visible depth cue is the case the brief says to disable, so the code and dependency were
  removed. A future attempt should use a multisampled render target and be checked on a real GPU.

## Layout

```
inside-mace/
  src/pages/index.astro        page (hero, instrument, provenance)
  src/islands/                 LabLoader (lazy entry), InterventionLab, TraceChange, ArchitectureDiagram,
                               StagePanel, StageEvidence, FeatureInspector, ResponsePlots, RotationDemo, SceneGuard
  src/lib/                     experiment.ts (schema, metrics), scene.ts, trace.ts, stages.ts, urlState.ts,
                               demo.ts, figureText.ts, store.ts (+ *.test.ts)
  src/three/                   copied from atlas/src/three (Stage, impostor atoms, primitives, shading); cameraSync.ts is new
  src/data/experiments/        precomputed data (manifest + lazy per-frame chunks + fixed colour scales)
  pipeline/                    CPU data pipeline (Python) and pytest checks
  e2e/                         Playwright browser checks
```

## Verification summary and remaining limitations

Kept up to date in [HANDOFF.md](HANDOFF.md) during the build; the final summary is written here by the last
build session.
