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
- `atlas/` — Astro 6 site (TypeScript strict) with React islands.
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
- Commands: `cd atlas && npm install`, `npm run dev`, `npm run build`, `npm run check`.
