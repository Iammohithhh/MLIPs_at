# MLIP Atlas — project guide for Claude

## What this is
An interactive, evidence-first website about machine-learning interatomic potentials (MLIPs):
3D Physics Lab experiments, a live model Observatory, architecture explainers (Architecture X-ray),
dataset/benchmark pages, a Failure Atlas, and an electrochemical-interface lab.
The full product vision and plan is in [MLIP_Atlas_Vision.md](MLIP_Atlas_Vision.md); read it before doing design or content work.

## Owner preferences (follow these)
- **Never use more than 2 agents/subagents** at a time, including inside workflows. Do scaffolding and foundational work directly.
- **Rich visuals are the priority.** Every page needs purpose-built figures/interactives; no thin catalogue entries (MLIP Hub was judged a "catalogue dump" — do the opposite).
- **No artificial scoping down.** Order work by dependency, not by resource limits. No schedules/timelines needed.
- **No GPU servers.** Site is static; 3D runs in the visitor's browser; heavy results are precomputed on CPU; any live model inference uses a CPU server.
- Ask before outward-facing actions (publishing, making the repo public, deploying).

## Stack
- `atlas/` — Astro 7 site (TypeScript strict) with React islands. (The vision says Astro 6; 7 is used because 6.x has unpatched critical/XSS advisories. Markdown uses the `unified` processor from `@astrojs/markdown-remark` for remark-math/rehype-katex; `compressHTML: true` because Astro 7's default JSX whitespace rules eat spaces in prose.)
- 3D: Three.js + React Three Fiber + drei (WebGPU where available, WebGL fallback); atoms as impostor spheres, instanced.
- Math: KaTeX (build-time via remark-math/rehype-katex). Plots: Observable Plot / D3. Animation: GSAP. State: Zustand.
- Later: Python (ASE, TorchSim, MACE/UMA/PET-MAD) for precompute and a CPU inference API.

## Scientific rules (non-negotiable)
- Keep architecture ≠ checkpoint ≠ dataset ≠ benchmark result distinct everywhere.
- Every non-trivial claim needs a source (paper + figure/table locator) or an explicit derivation.
- Label every interactive/result: `TOY MODEL`, `LIVE CALCULATION`, `PRECOMPUTED`, `PUBLISHED RESULT`, `REPRODUCED`.
- Never present benchmark numbers from different protocols as one ranking.
- Show assumptions, units, boundary conditions and sign conventions in view.
- A DFT-only quantity (DOS, Fermi level, band alignment) must never be shown as an MLIP prediction.

## Working conventions
- Commit and push small, working steps so desktop and mobile/cloud sessions stay in sync.
- On Windows use `npm ci`, not `npm install`: install rewrites `package-lock.json` without other platforms' optional binaries (esbuild, rollup), breaking Linux CI and cloud sessions. Never commit such a lockfile diff.
- Commands: `cd atlas && npm ci`, `npm run dev`, `npm run build`, `npm run check`, `npm test` (Vitest: physics numerics, contrast, TeX guards).
- In `.astro` files pass TeX as ``tex={String.raw`...`}`` — quoted attributes process escapes (`\v`, `\b`); `assertTexIntact` fails the build if this happens.
- Linked symbols: `\sym{key}{tex}` in TeX ↔ `data-sym="key"` on SVG/HTML ↔ `useLinkedSymbol()` in islands.
- Elements created by client scripts don't get Astro scoped-style attributes; style them with `:global()`.
- 3D: `src/three/` (Stage, impostor atoms, sticks/arrows, shells, lines, picking; shared lighting in `shading.ts`). Scene data lives in pure, tested modules under `src/lib/` (`chem/`, `physics/`, `scenes/`); islands in `src/islands/`. Three.js ShaderMaterial fragment shaders must declare `uniform mat4 projectionMatrix;` themselves.
- The TeX lost-escape guard also fires on plain English words such as "sum" inside `	ext{}`; reword the label.
- Screenshot check: `npx astro preview` then Playwright (`--channel msedge`) — WebGL renders in headless Edge.
- Paper authors: one quoted `'Family, Initials'` per list item (block list). A YAML flow list `[Behler, J., Parrinello, M.]` splits every name; the schema rejects it.
- Data pipelines (CPU, `pipeline/`): `atlas_pipeline.structures` → `atlas/src/data/structures/*.json`; `atlas_pipeline.bibliography` → `atlas/src/data/bibliography.json` (rerun after adding/editing a paper record; the test suite fails if a paper has no cached metadata or disagrees with Crossref/arXiv). Cite papers in pages with `<Cite id="…" locator="Fig. 2" />`.

