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

_No sessions have run yet._
