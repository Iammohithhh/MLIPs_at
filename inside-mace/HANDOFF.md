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


### Session 4 — 2026-10-17 21:16 → 21:57 UTC

**State found:** the log had only session 1's entry, but `git log` shows a second batch of commits (19:20–19:39 UTC,
`ac53b8b` … `c2285ee`: Trace the change with predict-before-reveal, side-by-side synced scenes, architecture diagram,
stage-5 evidence, presentation mode + 5-step demo, "what a zero means" notes, PNG export, lazy loader island, rotation
glyphs, README, GSAP camera transitions, walk-mode toggles). No session-3 commits exist. Re-verified that state first:
astro check 0 errors, Vitest 35/35, Playwright desktop+mobile 34/34.

**Done (commits on `claude/inside-mace`):**
- `8650cd6` pipeline: second experiment `zno-axial-bond` (wurtzite ZnO, two species) + optional 4th CLI argument to
  (re)run selected experiment ids only. Moved atom = the Zn nearest the supercell centre (index 150); axis = unit vector
  to its axial O neighbour (149), i.e. −c ([0001] direction); d > 0 shortens that Zn–O bond. 5×5×3 supercell, 300 atoms
  (in-plane width 14.07 Å, c 15.63 Å, both > 2·r_max). Local set by a 3.4 Å cutoff (18 atoms: 4 O, 12 Zn, 1 far O).
- `85651cb` **real ZnO data** in `src/data/experiments/zno-axial-bond/` (MACE-MP-0a medium, SHA-256 verified, float64,
  CPU, same versions as Si): 25 frames d = −0.300…+0.300 Å, all `ok` (~10–14 s each). Checks: restoration ΔE = 0,
  ΔF = 0; FD (h = 1e-2…3e-4 Å) error 7.3e-7 eV/Å at h = 3e-4 (∝ h²) for the moved atom; rotated input |ΔE| = 0.0,
  F equivariance 3.1e-14, h0 invariance 1.3e-15, h1 equivariance 5.2e-16. `scales.json` generated. `index.json` lists both.
  **Physics observed:** the ideal wurtzite input (a = 3.25, c = 5.21, u = 0.382, not relaxed with the model) has a
  baseline force of 0.0371 eV/Å along c on every Zn and O (C3v sites; x, y ≤ 4e-6) — so, unlike Si, the ZnO baseline is
  **not** force-free and its layer-1 l = 1 features are **not** zero. Anharmonic: ΔE(+0.3) = 0.708 eV (bond compressed)
  vs ΔE(−0.3) = 0.302 eV; F·û = −6.25 eV/Å at +0.3 Å, +1.68 at −0.3 Å. Documented in `pipeline/README.md` (new
  Experiments table). New `src/lib/experiments.test.ts`: generic checks over every committed manifest (restoration,
  dE, energy-slope vs F·û, positive curvature, ΣEᵢ = E, ΣF = 0, FD, rotation, atom identity, local set).
- `372a440` `e2e/perf.spec.ts` — Pixel 7 emulation, 4× CPU throttle (CDP), SwiftShader.
  Measured: interactive 2.5–2.8 s; frame step → updated readout median 0.40–0.44 s, worst 1.38–1.46 s (first step,
  includes lazy chunk); 29 long tasks, max ~0.7 s; JS heap 26–33 MB. Budgets: median < 1.5 s, worst < 4 s, heap < 400 MB.
- `00eddea` `src/lib/experimentInfo.ts` (+ test): element display colours (Si as before, Zn added, O from
  atlas/src/lib/chem/elements.ts), bond-stick criterion per experiment (Si 2.6 Å, ZnO 2.3 Å), geometry source, domain
  sentence, and `baselineFacts(m)` **derived from the stored run** (force-free? l = 1 vectors vanish?).
- `e52035c` frame chunks and colour scales keyed by experiment id (`loadChunk(id, k)`, `scalesFor(id)`); a chunk whose
  `experiment`/`k` does not match is rejected; missing scales → alert, never a blank heatmap.
- `c6df239` InterventionLab takes its manifest via React context (`InterventionLab({ manifest })`, default Si), all
  sub-components use `useLab()`. Behaviour unchanged.
- `edadd9c`, `fd3b1e5`, `93ea6bf`: element colours per atom, legend swatches from the composition, bond label, colour
  option text, demo step-1 caption (zero-force claim only if stored forces are zero; else quotes the largest stored force
  and says the geometry is unrelaxed), step-2 "pushes back" derived from the sign, PNG-export text per experiment.
- Last commit of the session: StageEvidence (embedding labelled with the moved atom's element and hidden when the
  selected atom is another element; E₀(element); Σᵢ Eᵢ over m.structure.n atoms; Td-symmetry sentences only when
  `experimentInfo.vectorsVanishReason` is set **and** the stored baseline vectors are zero) and the FeatureInspector
  l = 1 note gated the same way. Re-verified: check 0 errors, Vitest 66/66, Playwright 35 passed.

**Verified at the end (21:55 UTC, after the last code commit `a646d35`):** `astro check` 0 errors; Vitest 66/66 (7 files); `astro build` OK; Playwright
desktop + mobile 35 passed, 1 skipped (perf runs on mobile only); pytest 22/22 with `INSIDE_MACE_CHECKPOINT` (fresh
re-evaluation of a stored frame matches for **both** experiments). Screenshots 1440×1000 and 390×844 inspected (Si page
unchanged; scene renders, no overflow).

**Open problems / known gaps:**
- ZnO is **not yet reachable on the page.** `LabLoader` still renders the default (Si) manifest and `decodeState` reports
  `exp=zno-axial-bond` as "not available". Before exposing it, these Si-only texts must become per-experiment or
  data-derived (each would be false for ZnO):
  `src/lib/stages.ts:38` ("216 Si atoms"), `:49` ("46 neighbours" — compute from the chunk's edge count), `:97`, `:111`,
  `:122` (shapes `[216]`); `src/islands/StageEvidence.tsx` stage-7 note (~line 260, "zero by symmetry" for the FD row);
  (`ArchitectureDiagram` force shape is now the structure-independent `[N atoms, 3]`); `src/pages/index.astro:36,42–43` (Structure / Domain → `experimentInfo`);
  `InterventionLab.tsx` FORCE_SCALE comment (ZnO max |F| 6.25 eV/Å → 3.1 Å arrow; fine, but the comment is Si-specific);
  scene `label="Silicon neighbourhood…"`; check `TraceChange.tsx`/`trace.ts` "exact zeros" wording for a baseline
  with non-zero vectors; Si-only claims in README "Experiment"/"Domain".
- The hero and the lab both show "Move an atom / Walk me through it" button rows (visible duplication on desktop and
  mobile screenshots) — consider keeping only the hero pair or making the lab pair a compact mode switch.
- HANDOFF had no session-2 entry; its work is summarised above from `git log`.

**Next steps (in dependency order):**
1. Make every text listed above per-experiment (use `experimentInfo`, `baselineFacts`, `composition`, chunk edge
   counts), with Vitest cases for both manifests (pattern: `experimentInfo.test.ts`, `figureText.test.ts`).
2. Experiment selection: `LabLoader` reads `exp` from the URL, `import.meta.glob('../data/experiments/*/manifest.json')`
   (lazy, not eager) → `parseManifest` → `<InterventionLab manifest={m} />`; `decodeState` must accept any id in
   `index.json` (pass the list); an experiment picker (two cards: "Si: bond stretch", "ZnO: axial Zn–O bond") near the
   scene; switching keeps mode/stage/view where valid and resets atom/frame to that experiment's moved atom/baseline.
   index.astro provenance block must follow the selected experiment (render client-side or show both).
3. ZnO-specific explanation: the baseline is not a model minimum (0.037 eV/Å along c) — say it next to the readouts
   and in walk stage 7; the difference view is then the meaningful one. Two-species scene legend already works.
4. Playwright for ZnO (`?exp=zno-axial-bond&frame=24`: readouts quote stored ΔE +0.7076 eV, legend shows Zn/O, no
   "zero by symmetry" text; URL round-trip; switching experiments), screenshots at 1440/1024/390 widths.
5. Then the remaining brief items: final README verification summary + limitations (session 6), optional subtle AO only
   after impostor-depth verification (§4, may be judged not worth it — record why).


### Session 5 — 2026-10-17 22:16 → 22:54 UTC

**State found:** branch at `0fd7591` (session 4 handoff), working tree clean. Re-read BRIEF, HANDOFF, CLAUDE.md.

**Done (commits on `claude/inside-mace`):**
- `fb88044` **ZnO is reachable on the page.** `LabLoader` reads `exp` from the URL (validated against
  `index.json`; unknown id → Si plus the existing "experiment … is not available" alert), loads only that
  experiment's manifest via `import.meta.glob` (lazy), and shows an experiment picker (two cards, `aria-pressed`,
  44 px targets). Switching keeps mode/stage/view/presentation and drops atom/frame/ch/layer (`switchQuery`); the
  lab remounts keyed by id. `stagesFor(m)` in `stages.ts` fills atom count, composition, moved element and the
  moved atom's baseline neighbour count (minimum image; equals the stored edge count: Si 46, ZnO 76 — tested);
  `StageDetail` takes `m`. Stage-7 FD note: "zero at d = 0" only when `baselineFacts(m).forceFree`, otherwise
  quotes the stored baseline force. Scene aria-label per element/crystal. `index.astro` provenance lists every
  committed experiment (structure, geometry source, displacement, frames ok, command, schema, domain).
  New `src/lib/stages.test.ts`, `e2e/experiments.spec.ts`.
- `6b75893` readout note when the baseline is not a model minimum (ZnO: stored F·û = +0.0371 eV/Å at d = 0;
  derived from stored forces; absent for Si — tested both ways).
- `5dda4a7` README covers both experiments (ZnO geometry, unrelaxed baseline, regeneration, domain).
- `7e6bb58` hero actions keep the current exp/atom/frame/view when switching mode (`src/lib/heroLink.ts` + test;
  plain links without JS).
- `d8c31dc` lab mode switch restyled as a joined segmented control (removes the "two identical button rows"
  duplication noted by session 4); light experiment cards, selected one on the stage colour.
- `d8f1ce4` demo-sequence unit tests + presentation-mode browser check for ZnO (no zero-force claim; stored numbers).
- `c13e521` **Ambient occlusion (brief §4) evaluated, not enabled** — N8AO 2.0.1 as an EffectComposer pass:
  impostor depth is correct (gl_FragDepth), but on the ZnO scene mean luminance 40.381 → 40.376 (no visible AO)
  while the composer target lost MSAA (11 % of pixels changed, all thin edges/halo aliasing). Code and
  dependency removed; reasoning recorded in README "Rendering decisions".
- `bb39423` scenes pause (`frameloop="never"`) when off screen (IntersectionObserver) or the tab is hidden, and
  redraw once on return; `data-rendering` attribute; `e2e/render.spec.ts`. Verified: frame changed while paused →
  on return the canvas is pixel-identical to a fresh load of that state (0 of 579 464 px differ).
- `4714672` perf spec runs for both experiments. Pixel 7, 4× CPU, SwiftShader: Si interactive 1.65–1.73 s,
  frame step median 0.28–0.34 s, worst 1.0–1.1 s, heap 33 MB; ZnO interactive 2.0–2.2 s, median 0.27–0.37 s,
  worst 1.0–1.1 s, heap 32 MB.
- `931b27b` the InterventionLab chunk no longer imports the Si manifest (manifest is a required prop).
- `6882d27`, `36b6b42` true minus signs in stage-6 energies, rotation comparison and channel readouts; missing
  stage-6 values shown as "—".
- `40fc854` lab heading sub-line names the selected experiment (title, structure, supercell, n, and
  "baseline forces zero" vs "unrelaxed baseline (carries forces)" from the stored run).

**Verified (22:41 UTC, after `36b6b42`):** `astro check` 0 errors (45 files); Vitest 75/75 (10 files);
`astro build` OK; Playwright desktop + mobile **50 passed, 2 skipped** (perf is mobile-only). Screenshots
inspected: ZnO at 1440×1000, 1024×800, 390×844 (no horizontal overflow at any), Si 1440 picker/head, walk
mode stage 7 desktop + mobile. Visual text dump of ZnO stages 1/4/5/6, rotation, trace and inspector: no Si
assumptions remain.
- `22036de` presentation mode hides the experiment picker (minimal controls; the link fixes the experiment).
- README "Rendering decisions" also documents render pausing and per-experiment lazy data.

**Final verification (22:45 UTC, after `22036de`):** astro check 0 errors; Vitest 75/75; build OK; Playwright
desktop + mobile 50 passed / 2 skipped (perf mobile-only) at 22:40, then `experiments.spec` 12/12 and
`lab.spec` 34/34 again after the last CSS change. **pytest 22/22** with a freshly downloaded checkpoint
(`sha256sum` = 01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845, matches) — includes the fresh
re-evaluation of stored frames for both experiments. Venv recipe as in session 1 (≈2 min; pytest ≈36 s).

**Later commits (same session):**
- `8a11049` rotation panel says which rotated runs are stored (d = 0 and +0.100 Å) and why the displaced one is shown.
- `6dc2bfa` true minus in plot ticks/tooltips and response readouts.
- `5c86385` **colour-by-change bug fixed:** the ramp started from each element's colour, so ZnO's red O atoms
  looked "changed" (amber-ish) at zero change and the grey→amber legend was wrong; now every element starts from
  the neutral grey of the legend and of the PNG-export text ("grey = 0 to amber = largest").
- `5e2ff21` layout: `.lab__stage` no longer stretches to the side panel's height (it left an empty dark band under
  the scene, worst in side by side); side-by-side pane labels moved to the top of each pane (they collided with
  the force-scale caption once the stage stopped stretching). Screenshot-checked at 1440×900 (ZnO split).
- Re-verified at 22:51 after the layout change: astro check 0 errors, Vitest 75/75, Playwright desktop + mobile
  50 passed / 2 skipped.

- `ba8ba7e` desktop (≥ 901 × 640 px): the scene stage is `position: sticky` (top 0.75 rem), so it stays in view while
  the walkthrough text / controls scroll; checked by screenshot (ZnO, stage 7). Playwright desktop + mobile
  50 passed / 2 skipped after this change (22:53 UTC).

**Open problems / known gaps:**
- Baseline ZnO forces (0.037 eV/Å) are below the 0.12 eV/Å arrow threshold, so the d = 0 ZnO scene shows no
  arrows. Not worth a per-experiment threshold: at 0.5 Å per eV/Å such an arrow is 0.02 Å long (invisible);
  the readout note quotes the stored value instead.
- Ambient occlusion: evaluated in SwiftShader only; if retried, use a multisampled target and a real GPU.

**Next steps (session 6):**
1. Re-verify (check, Vitest, build, Playwright desktop+mobile, pytest with checkpoint) and inspect screenshots at
   1440/1024/390 for both `?exp=si-bond-stretch` and `?exp=zno-axial-bond`, walk and lab modes, presentation.
2. Write the **final verification summary and remaining limitations** into `README.md` (brief §16) — replace the
   placeholder section at its end. Use the numbers in the session 4/5 entries (tests, perf, checks: restoration,
   FD h² convergence, rotation tolerances per experiment) and list limitations: precomputed only (no live
   inference), two structures / one path each, no DFT comparison, ZnO geometry unrelaxed, AO not enabled,
   WebGPU not adopted, perf measured on emulation (SwiftShader) not a physical phone.
3. If time remains: a Playwright check that colour-by-change on ZnO renders the bond partner amber (pixel sample)
   and a final screenshot pass of every view (baseline/perturbed/difference/side by side) for ZnO.


### Session 6 (final) — 2026-10-17 23:15 → ~23:50 UTC

**State found:** branch at `0f7c0b7` (session 5 handoff), clean. Re-read BRIEF, HANDOFF, CLAUDE.md, README.
Re-verified first (23:16–23:20): astro check 0 errors, Vitest 75/75, build OK, Playwright desktop + mobile 50 passed /
2 skipped, pytest 22/22 with a freshly downloaded checkpoint (`sha256sum` 01bfe221…35845, matches).

**Done (commits on `claude/inside-mace`):**
- `6bb2e06` `e2e/colour.spec.ts`: pixel classification of the scene canvas proves the session-5 colour-by-change fix —
  ZnO at d = 0 by element: ~29 k brick-red O pixels; by change: 0 red pixels; at d = +0.3 Å amber pixels rise well above
  the baseline's (the amber path marker). Desktop + mobile.
- `ef1503c` README: final verification summary (commands and real results), table of how each brief §15 check is
  covered with the stored numbers (restoration, FD h² convergence per experiment, rotation residuals), an honest
  brief §16 status table (done / partial / not done) and remaining limitations.
- `dd3af52` phones: scene height min(50svh, 420px), so scene, slider, d and ΔE fit one Pixel 7 screen (slider was a
  full screen below the structure). Screenshot-checked.
- `1d9ab09` GSAP now also drives the architecture-diagram stage emphasis and the trace-lane reveal (no stagger by
  distance/layer, so nothing suggests propagation or runtime; off under reduced motion); `e2e/motion.spec.ts` records
  the inline styles GSAP writes (present with motion, none with reduced motion).
- `b8ee000` Difference view: the scale caption says "force-change arrows ΔFᵢ = Fᵢ(d) − Fᵢ(0)" with an amber swatch
  (it said "force arrows" with a teal swatch while the arrows are amber ΔF); same for the PNG export legend (Vitest +
  Playwright assertions). `npm run record:demo` (`record/demo.record.ts`, `playwright.record.config.ts`): drives the
  5-step presentation sequence for both experiments, asserts each step, saves a still per step into `recordings/`
  (git-ignored); `RECORD_VIDEO=1` adds a .webm.
- `2273bd3` stage-7 finite-difference evidence: one card per stored check; the h-columns were hidden behind a
  horizontal scroll in the ~300 px side panel. Browser check reads the Si d = +0.1 Å, h = 3e-4 Å error (9.8e−7).
- `6860305` stage-2/3 edge table fits the panel (Y₁ column was clipped).
- Final commit: README verification numbers from the last full run; this entry.
- `39f738a` **bug fix:** signed readouts dropped the exponent's minus for small positive values (`fmt(2.43e-15)` →
  "+2.43e15"; reachable e.g. for ΔEᵢ of a far atom). `fmtSigned` now in `src/lib/format.ts` with tests; Vitest 79/79
  (10 files), Playwright 56 passed / 2 skipped after the fix.

**Verified at the end (23:38–23:41 UTC, after the last code change):** astro check 0 errors / 0 warnings; Vitest
76/76 (9 files); build OK; Playwright desktop + mobile **56 passed, 2 skipped** (perf mobile-only, passed);
`npm run record:demo` 2/2; pytest 22/22 with the checkpoint (fresh re-evaluation of stored frames, both experiments).
Overflow sweep: 0 px at 320, 768, 844 (landscape), 1024, 1920 widths for walk stage 7, ZnO split, ZnO presentation.

**Open problems / known gaps (also in README "Brief §16 status" and "Remaining limitations"):**
- No video file: Playwright's ffmpeg (`/opt/pw-browsers/ffmpeg-1013/ffmpeg-linux`) is missing in the cloud container
  (exact error: `browserContext.newPage: Executable doesn't exist at /opt/pw-browsers/ffmpeg-1013/ffmpeg-linux`);
  `/usr/bin/ffmpeg` exists but is not what Playwright uses. Owner can run `npx playwright install ffmpeg` then
  `RECORD_VIDEO=1 npm run record:demo`, or screen-record `?present=1&demo=1`.
- Performance measured under emulation (SwiftShader, CPU throttle) only; `edge` Playwright project not run here.
- Precomputed only; two experiments, one path each; no DFT reference; ZnO baseline unrelaxed (stated on the page).

**Next steps (for whoever continues):**
1. Record the demonstration video on the owner's machine (`--project=edge` / `RECORD_VIDEO=1`) and check it frame by frame.
2. Measure on a physical phone; if AO is reconsidered, use a multisampled target on a real GPU (README "Rendering decisions").
3. Possible extensions (each needs real runs, never invented data): a third experiment (e.g. a surface or a defect
   with the same storage format), a relaxed ZnO baseline as a separate experiment, per-edge message contributions in the
   chunks so the trace can animate individual edges, a CPU inference service for arbitrary displacements (brief §8
   "separate extension").
