# Inside MACE — session handoff log

The build of `inside-mace/` (see [BRIEF.md](BRIEF.md)) runs as a chain of cloud sessions on branch
`claude/inside-mace`. Each session reads this file first and **appends** its entry before it stops.
The whole brief is in scope. Work in dependency order; never drop or water down a brief item.

## Session windows (one session at a time)

| # | Starts (UTC) | Hard stop (UTC) | IST window |
|---|---|---|---|
| 1 | 2026-10-17 18:20 | 19:10 | 23:50 – 00:40 |
| 2 | 19:15 | 20:10 | 00:45 – 01:40 |
| 3 | 20:15 | 21:10 | 01:45 – 02:40 |
| 4 | 21:15 | 22:10 | 02:45 – 03:40 |
| 5 | 22:15 | 23:10 | 03:45 – 04:40 |
| 6 | 23:15 | 2026-10-18 00:25 | 04:45 – 05:55 |

## Suggested dependency order (adjust as you learn; record why)

1. Read root `CLAUDE.md`, `CONTRIBUTING.md`, `MLIP_Atlas_Vision.md`, then everything §1 of the brief names.
   Scaffold `inside-mace/` as its own Astro 7 + React + TS strict app (copy the Atlas config, theme
   tokens, KaTeX setup, `src/three/` primitives; note each file's origin).
2. Pipeline (`inside-mace/pipeline/`, CPU): install the pinned stack from `pipeline/requirements-xray.txt`
   (torch 2.14.1 CPU wheel, mace-torch 0.3.16, e3nn 0.4.4, ase 3.29.0). Fetch MACE-MP-0a medium
   (URL + SHA-256 in `pipeline/atlas_pipeline/xray.py`; load it with the same restricted unpickler; never
   plain `torch.load`). Generate: intervention frame sequences (periodic Si and/or ZnO), all captured
   intermediates per frame, rotated-input runs, finite-difference force checks, full provenance.
   Commit generated JSON (lazy-loadable chunks) so the static site needs no Python.
3. Typed data schema + loaders + Vitest numerics tests (§15 checks).
4. The instrument: scene, guided walkthrough (7 stages), Atom Intervention Lab, trace-the-change
   reveal, rotation demonstration, plots/heatmaps, linked state (Zustand), progressive disclosure.
5. URL state, presentation mode, still-image export, scripted demonstration sequence.
6. Playwright browser checks (Chromium on Linux; `--channel msedge` only on the owner's Windows box),
   screenshots at several viewports, mobile performance, docs (README, interpretation limits,
   reproduction, verification summary, remaining limitations).

## Log

<!-- Each session appends: session #, start/stop time, what was done (with commit hashes),
     verified results (test/build output), open problems, and the exact next steps. -->


### Session 1 — 2026-10-17 18:15 → 18:55 UTC

**Done (commits on `claude/inside-mace`):**
- `8c6a260` Scaffold `inside-mace/` (Astro 7 + React + TS strict; `astro.config.mjs` adapted from Atlas, without
  Tailwind/MDX/content-integrity). Copied with origin notes: `src/three/*` (atlas/src/three), `src/lib/katex-options.mjs`,
  `vitest.config.ts`, `tsconfig.json`, `pipeline/inside_mace/capture.py` (= pipeline/atlas_pipeline/xray.py, restricted
  unpickler + hooks), `pipeline/requirements.txt` (= requirements-xray.txt). `package-lock.json` generated on Linux.
- Pipeline `pipeline/inside_mace/intervention.py` and its **real data** in `src/data/experiments/si-bond-stretch/`:
  MACE-MP-0a medium (SHA-256 verified 01bfe221…5845), float64, CPU, mace-torch 0.3.16 / e3nn 0.4.4 / torch 2.14.1+cpu /
  ase 3.29.0. Si diamond 3×3×3 supercell (216 atoms, cell 16.29 Å > 2·r_max). Moved atom = 105 (nearest the cell centre),
  axis = unit vector to bonded partner 106 (a [111]-type bond direction; d > 0 shortens that bond). 25 frames,
  d = −0.300…+0.300 Å in 0.025 Å steps; all 25 evaluated OK (~4 s each). Per frame: energy (full float64), dE, forces,
  node energies, per-layer readout site terms, per-atom feature-block difference norms for all 216 atoms; lazy chunk
  `frames/NN.json` (~212 KB) with full h (l=0: 128 ch; l=1: 128×3 for layer 1; layer 2 is scalar-only) and message block
  norms A for the 17-atom local set (moved atom + first two shells) and radial (46×10) / SH (46×16) of the 46 edges into the
  moved atom. Checks stored: baseline restoration (re-evaluation: ΔE = 0, ΔF = 0), finite differences (h = 1e-2…3e-4 Å,
  central, error ∝ h², 9.9e-7 eV/Å at h=3e-4 for the moved atom), rotated input (positions + cell rotated by 50° about
  (1,2,3)): |ΔE| = 0.0, force equivariance 3e-14, h0 invariance 8e-16, h1 equivariance 5e-16.
  Observed physics: anharmonic well; ΔE(+0.3) = 0.575 eV vs ΔE(−0.3) = 0.304 eV (compression stiffer); F·û restoring.
  Layer-1 l=1 features are exactly zero (<1e-12) at the baseline for every atom (Td site symmetry) and appear on displacement.
- `pipeline/inside_mace/scales.py` → `scales.json` (fixed heatmap colour scales over all frames, derived from chunks).
- `13263c7` `src/lib/experiment.ts` (zod schema, frame snapping, null-not-zero accessors, ΔE/ΔF/site-energy changes,
  feature-change metric definitions incl. relative change with a floor of 0.05·median baseline norm, MIC geometry),
  `src/lib/urlState.ts` (validated URL state: v, exp, ckpt, mode, atom, frame, stage, layer, ch, view; issues reported).
- `c065a8d`, `a56e7ef`, `7ae111e`, `27b5e2a`, `a6fec2f`: page `src/pages/index.astro` with hero (headline, two actions),
  `InterventionLab.tsx` (3D neighbourhood within 6 Å by minimum image, moved atom halo, displacement path, graph edges
  labelled as model edges ≠ bonds, Si–Si sticks <2.6 Å, force arrows 1 Å = 1 eV/Å, baseline/perturbed/difference,
  colour-by-feature-change with legend, slider + ± buttons snapping to frames, readouts, reset experiment/camera, URL
  sync), `FeatureInspector.tsx` (128-channel heatmaps baseline / frame / difference, channel selection, l=1 readout),
  `SceneGuard.tsx` (WebGL detection + error boundary → static fallback), `StagePanel.tsx` + `src/lib/stages.ts` (seven
  stages: plain sentence, what changed, KaTeX equation on demand, recorded tensor, mace-torch 0.3.16 file:line locators —
  verified against site-packages), "Walk me through it" mode with stage-driven scene overlays.
- `3e3df60`, `7a7e1d6`, plus Playwright suite `e2e/lab.spec.ts` + `playwright.config.ts`.

**Verified (this session):**
- `npx astro check` 0 errors; `npm test` (Vitest) 24/24 pass; `npx astro build` OK.
- pytest `inside-mace/pipeline/tests`: 11 passed, 1 skipped; with `INSIDE_MACE_CHECKPOINT=…` the re-evaluation test
  (stored frame vs fresh model run) passed.
- Playwright (Chromium /opt/pw-browsers/chromium, swiftshader), projects desktop + mobile (Pixel 7): 18/18 pass —
  status visible, displacement snapping + URL, keyboard slider, views + reset, shared URL restore + invalid values,
  no horizontal overflow, WebGL-unavailable fallback, reduced motion, guided walkthrough 1→7 with KaTeX and code locator.
- Screenshots inspected at 1440×1000 and 390×844 (desktop and mobile both render the scene, no overflow).

**Environment notes for next sessions:**
- Python venv: `python3 -m venv /home/user/venv-mace && pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  && pip install -r inside-mace/pipeline/requirements.txt` (~2 min). Checkpoint downloads from the GitHub release URL fine.
  Full pipeline run ≈ 5.5 min on this CPU.
- `astro preview` in Astro 7 detaches into a background server in non-interactive shells (and only one may run); the
  Playwright config therefore serves `dist/` with `python3 -m http.server`. Run `npx astro build` before `npx playwright test
  --project=desktop --project=mobile`.

**Added after the first log write (same session):**
- `6c46037` click-to-pick atoms (`AtomPicker` mounted) + keyboard atom selector for the 17-atom local set; smaller plot text.
- `26c4a3a` camera presets (Overview / Selected environment / Side-on to the path); force arrows 0.5 Å per eV/Å, arrows
  below 0.12 eV/Å hidden, scale note in scene units.
- `2dc80f8` `ResponsePlots.tsx`: per-layer/per-irrep ‖Δh‖ vs d and ΔEᵢ small multiples for the selected atom, own axes.
- `7b4b8d5` `RotationDemo.tsx`: real rotated-input comparison (E invariant to 0.0 eV, F → RF to 3e-14, h0 invariant,
  layer-1 l=1 channel h → Rh), with the camera-vs-input distinction stated. Schema now keeps `checks.rotation.runs[].local`.
- `dcc50e4` + final commit: `StageEvidence.tsx` — walkthrough evidence for stages 2/3 (46 edges into the moved atom, the 6
  shortest with 10 Bessel values and Y₁), 6 (E_i, ΣEᵢ vs E) and 7 (autograd vs finite-difference table); mobile panel
  overflow fixed (`.lab__panel` column minmax(0,1fr)).
- Late commits add stage-1 evidence (input position + the 128-value Si embedding h⁽⁰⁾ strip) and stage-4 evidence
  (layer-1 A block norms by l; verified in data: l = 1, 2 exactly 0 at baseline by Td symmetry, l = 3 non-zero).
- Final state verified at 18:55 UTC: astro check 0 errors, Vitest 24/24, build OK, Playwright desktop+mobile 22/22.

**Open problems / known gaps:**
- Walk mode: scene overlays only toggle edges/shell/arrows per stage; the stage-specific tensor views (radial/SH bars for
  edges into the moved atom, A-norm per l, site-energy bars, force decomposition) are not built yet. Scene-toggle
  checkboxes do not reflect walk-mode overrides.
- No "Trace the change" reveal sequence (brief §9), no "Predict before revealing" prompt, no GSAP transitions yet.
- Rotation demonstration has a text/number panel only; a visual (two small scenes or arrow glyphs for F and an l=1
  channel before/after R) would make it memorable in a recording.
- Walk-mode stage 5 has no stage-specific evidence block yet (h heatmap emphasis / product-basis explanation);
  stage-4 evidence is text/numbers only (no A heatmap).
- No camera presets (overview / selected environment / comparison) or synchronised side-by-side baseline|perturbed
  cameras (§7 asks for both states together with synced cameras; currently a toggle).
- Plot tick text scales with SVG width (large on desktop); plots need axis labels/units in-figure and a linked
  per-atom ΔEᵢ plot; feature-change-vs-d per layer plot ("which parts respond") still missing.
- Presentation mode, still-image export, scripted demo sequence, README (incl. interpretation limits), mobile
  performance measurement, render pausing when hidden, lazy loading of manifest (648 KB, currently bundled) — not done.
- Only one experiment (Si). ZnO (wurtzite) second experiment would show a two-species case; ethanol not planned
  (domain mismatch) — if added, must carry the mismatch caveat.

**Next steps (in dependency order):**
1. Side-by-side synced baseline|perturbed scenes on desktop (single scene with toggle on mobile); GSAP camera
   transitions between presets (respect reduced motion); selected-neighbour highlight.
2. Stage-specific inspectors for walk mode using the chunk data (edges radial/SH, A norms by l, h heatmap, site energy
   bars with ΣEᵢ = E, force = −∂E/∂r with FD table from `checks.finiteDifference`).
3. "Trace the change" sequence (stage-by-stage highlight of per-atom change norms, layer 1 → layer 2 → site energies →
   forces; labelled as a dependency explanation, not timing), optional predict-before-reveal.
4. Visual for the rotation panel (glyphs) and stage evidence for stage 5 and an A-norm heatmap for stage 4.
5. Presentation mode (`?present=1`), PNG export with title/settings/units/status/attribution, demo-sequence script,
   README + interpretation limits + reproduction; move manifest to lazy fetch.
