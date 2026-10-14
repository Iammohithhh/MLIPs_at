# MLIP Atlas — Product Vision & Experience Blueprint (v2)

> **An interactive, evidence-first atlas of machine-learned atomistic science**
> **Vision version:** 2.0 · 14 October 2026 (supersedes v1.0)
> **Status:** Concept / design brief, researched against the field as of October 2026
> **Constraints assumed:** none on ambition, team size or tooling — best-in-class everywhere.

**Tagline:** *Explore the models. Understand the physics. Discover the limitations. Navigate the frontier.*

---

## What changed from v1

| Area | v1 | v2 |
|---|---|---|
| Ambition | 80% 2D / 20% 3D, "avoid building big things early" | **3D wherever geometry exists**, cinematic quality, and *live* calculations — no artificial limits |
| Computation | Toy demos + precomputed results only | Five compute tiers, from in-browser physics up to **live GPU inference of real foundation models** |
| New flagships | — | **Model Observatory** (run many MLIPs on the same structure, live), **Architecture X-ray** (real intermediate features of a real forward pass), **Ewald Lab**, **Receptive-field Lab**, **EPhEct-style electrochemistry lab**, **Representation Map** |
| Entities | Model, paper, concept, application | Adds **Checkpoint, Dataset, Benchmark, Benchmark result, Claim, Experiment, Failure case** as first-class records |
| Positioning | Implicit | Explicit, researched competitor map (MLIP Hub, Matbench Discovery, MLIP Arena, MLIPAudit, chemiscope, Atomify, MatterViz) |
| Stack | Next.js + 3Dmol.js + Three.js | **Astro 6 + React islands + Three.js WebGPU/TSL + R3F v9**, build-time Python structure pipeline, Modal GPU backend, lammps.js, ONNX Runtime Web R&D track |
| Field content | Generic | Seeded with the 2025–2026 state of the art: foundation models, OAM-trained checkpoints, LES/charge-equilibration long-range methods, constant-potential electrochemistry, PES softening, EPhEct |

---

## 0. The central idea

MLIP Atlas is a **living, explorable map of machine-learning interatomic potentials** that connects:

**physical principle ↔ mathematics ↔ architecture ↔ trained checkpoint ↔ training data ↔ benchmark evidence ↔ application ↔ open question.**

It combines five interfaces:

1. **Physics Lab** — manipulable 3D thought-experiments with linked plots and equations.
2. **Model Observatory** — real models running live on real structures, side by side.
3. **Atlas** — a 2D/3D knowledge map of models, datasets, benchmarks, papers and concepts.
4. **Monographs** — research-grade pages for every model family, checkpoint, dataset and benchmark.
5. **Failure Atlas & Application hubs** — evidence-linked limitations and research journeys.

> **Promise:** A researcher can start with a basic physical question and trace it all the way to state-of-the-art methods, the evidence behind them, and the unresolved problems — and *see it happen in 3D, on real models*, without losing context.

**Design thesis:** Make scientific ideas *manipulable*, not merely readable — and make the evidence *inspectable*, not merely cited.

### North-star experience

A graduate student opens a 3D Au(111)/water/Na⁺ interface. They drag the ion toward the electrode and watch a toy image-charge curve; they flip on the **Observatory** and see four foundation models' force arrows on the ion diverge as it approaches the surface; a halo shows where the models disagree. A card explains that local models cannot represent image-charge attraction from first principles, links the EPhEct electrostatics benchmark, the latent-Ewald and charge-equilibration families, and the constant-potential MLIP papers. They leave with a justified shortlist and a concrete validation test.

---

# I. The field as of October 2026 (research summary)

This section is the scientific foundation the Atlas must reflect on day one. Every item becomes Atlas content with claim-level sources. Items marked † come from abstracts or secondary summaries and must be verified against the full papers before publication.

## 1. Model generations

| Generation / paradigm | Representative methods | Atlas angle |
|---|---|---|
| Descriptor + regression | Behler–Parrinello HDNNP, GAP/SOAP, MTP, ACE, DeePMD | Locality, body order, smooth cutoffs — the conceptual root of everything |
| Invariant message passing | SchNet, DimeNet, M3GNet, CHGNet | Receptive field grows with layers; angular information via invariants |
| Equivariant (constrained) | NequIP, Allegro, MACE, SevenNet, GRACE, eSEN, EquiformerV2/V3, UMA | Irreps, tensor products, body order (MACE), strict locality (Allegro) |
| Learned / unconstrained equivariance | Orb, PET / PET-MAD | Symmetry by data augmentation instead of construction; speed vs. guarantees |
| Long-range / charge-aware | 4G-HDNNP, CACE, **Latent Ewald Summation (LES)**, MACE-LES, LOREM, SpookyNet, AIMNet2, charge-equilibration hybrids | The physics the Atlas is uniquely positioned to explain |
| Fast deployment-oriented | DPA family incl. **DPA4C** (claims ~100× MACE-OMAT throughput at near-equal accuracy, billion-atom MD) † | Accuracy–speed frontier; the "empirical-potential speed" story |
| Foundation / universal | MACE-MP-0 → MACE-OMAT-0 → **MACE-MH-1** (multi-head, 89 elements); **UMA** (mixture of linear experts; ~half a billion structures; UMA-medium 1.4B total / ~50M active params); PET-MAD; Orb v3; SevenNet-Omni; GRACE-OAM; NequIP/Allegro-OAM | Architecture ≠ checkpoint ≠ result: the clearest case study for the Atlas's core rule |

**Leaderboard snapshot (Matbench Discovery, retrieved 14 Oct 2026, unique-prototype test set, CPS):** Prophet-OAME-MBD 0.912, TECE-OAM-RRA-1.0 0.908, EquFlashV2 0.907, EquiformerV3+DeNS-OAM 0.902, GRACE-3L-OAM-L 0.900, PET-OAM-XL 0.898 … ORB v3 0.860. *Use only as a dated, protocol-labelled snapshot that is pulled automatically from the source — never as a "best model" verdict.* Note that CPS is itself a weighted composite (F1 50%, κ_SRME 40%, RMSD 10%), and the site lets users re-weight it, which is a lesson in itself.

**Representation convergence:** a 2026 *Nature Machine Intelligence* paper reports that independently developed foundation MLIPs show statistically consistent geometric organisation of atomic environments ("Platonic representation") †, and a separate study compares the latent features of universal MLIPs, including MACE-OMAT-0 vs the OMat head of MACE-MH-1 (multi-head training preserves local representations, changes the readout) †. This directly motivates the **Representation Map** flagship (§13).

## 2. Training data is a first-class subject

| Dataset | Domain | What to teach |
|---|---|---|
| MPtrj (~1.5M) | Inorganic relaxation trajectories (PBE/PBE+U) | Near-equilibrium bias |
| Alexandria (Alex, >30M) | Inorganic, relaxation trajectories | Scale; overlap with test sets |
| OMat24 (>110M) | Inorganic, rattled + AIMD + re-relaxed Alexandria structures | Off-equilibrium sampling; the "train on OMat, fine-tune on MPtrj + sAlex" recipe (the "OAM" suffix) |
| MatPES | PBE and r²SCAN single points | Functional fidelity; multi-fidelity heads |
| OMol25 (>100M) | Molecules, ωB97M-V/def2-TZVPD | Molecular vs materials fidelity mismatch |
| SPICE (2) | Drug-like molecules, biomolecular fragments | Organic chemistry coverage |
| OC20/OC22, ODAC | Catalysis, MOF/sorbents | Surface/adsorbate domains |

The Atlas shows **which checkpoint saw which data at which level of theory**, because checkpoints trained on different functionals are not directly comparable.

## 3. Benchmarks — evidence with protocols

- **Matbench Discovery** — stability/discovery F1, geometry RMSD, phonon κ_SRME, MD, diatomics; per-model YAML and checkpoints.
- **MLIP Arena** (Chiang et al., arXiv:2509.20630) — physics awareness, reactivity, extreme conditions, thermodynamics; argues error metrics alone are leaky and insufficient.
- **MLIPAudit** (InstaDeep, NeurIPS 2025) — organic molecules, molecular liquids, proteins.
- **LAMBench**, **UniFFBench** (vs experimental measurements), migration-barrier, MOF and catalysis benchmarks (2025–2026).
- **EPhEct** (Sumić, …, Neugebauer; arXiv:2608.14153, Aug 2026) — four electrostatic-phenomenology tests for electrochemistry: **image-charge attraction at a metal electrode, LO–TO phonon splitting, interfacial-water dipole, Fermi-level pinning during ion discharge.** Thesis: energy/force errors can hide qualitatively wrong electrostatics. *This is almost a ready-made syllabus for the Atlas's electrochemistry chapter.*

## 4. Long-range electrostatics — the Atlas's intellectual centre

- **Taxonomy** (Grasselli, Rossi, de Gironcoli, Grisafi; arXiv:2602.11071): (1) *local charge models* — explicit charge-density decompositions or implicit auxiliary variables; (2) *deliberately nonlocal models* — self-consistent procedures or nonlocal descriptors/architectures; (3) *finite-field effects* — coupling to polarisation under bias. There is still no general prescription for adding long-range physics while keeping locality. Long-range effects are essential at electrochemical interfaces; ionic transport appears much less sensitive to them.
- **Latent Ewald Summation** (Cheng, *npj Comput. Mater.* 2025): learn a latent charge-like variable from local descriptors and Ewald-sum it. Charges come from energy/force training alone. Polarisation and Born effective charges emerge (*Nat. Commun.* 2025, s41467-025-63852-x).
- **Augmentation framework** (Kim et al., arXiv:2507.14302) and perspective "**Long-range electrostatics for MLIPs is easier than we thought**" (Kim & Cheng, *J. Chem. Phys.* 164, 060901, 2026): essentially any short-range MLIP can be augmented (Ewald, PME, P3M).
- **Charge equilibration** lines: self-consistent electrostatic design-space study (arXiv:2603.14700) †; QEq integrated with equivariant MLIPs (JCTC 2026) †; polarizable atomic multipoles (arXiv:2605.05746) †; equivariant (not scalar) charges for long range (TMLR 2026) †.
- **Defects in SrTiO₃** benchmark of long-range MLIPs (arXiv:2610.00794): MACE-LES best in that comparison †.

## 5. Electrochemical interfaces — the application flagship

- **Constant-potential MLFF** (JCTC 2025, PubMed 40717429): MACE-based model that takes electron count as input and predicts the Fermi level; applied to CO₂ reduction on Ni–N–C †.
- **Ag(111)/water CO₂RR** (*Nat. Commun.* 2025, s41467-025-65630-1): separate MLIPs for the PZC and −1.5 V vs SHE — fitting a separate potential per electrode potential as a practical strategy.
- **TRECI** (arXiv:2511.19338): data-efficient transfer learning for constant-potential metal/water MD.
- **Fragment-constrained charge equilibration** (arXiv:2604.27910, hippynn-based) †.
- **Explicit electric-potential-embedded ML** (arXiv:2604.07322) †.
- Constant-potential NN-MD of cation effects at Au/water (JACS Au 2026) †.

## 6. Documented failure modes (Failure Atlas seed set)

| Case | Evidence |
|---|---|
| **Systematic PES softening** of universal MLIPs (M3GNet, CHGNet, MACE-MP-0): underpredicted energies, forces and phonon frequencies (229 PhononDB compounds); root cause is biased near-equilibrium sampling; largely fixable by fine-tuning on very few points | Deng et al., arXiv:2405.07105; phonon fine-tuning follow-up arXiv:2601.07742 † |
| **Surfaces** — out-of-the-box foundation models struggle; error correlates with distance from training data | ACS Appl. Mater. Interfaces 2024; cleavage-energy benchmark arXiv:2508.21663 |
| **Electrostatic phenomenology** — qualitatively wrong image charge, screening, charge transfer despite good force errors | EPhEct, arXiv:2608.14153 |
| **Alkali-ion kinetics**, **heterogeneous catalysis**, **migration barriers** | arXiv:2601.10938; 2512.16702; 2512.03642 † |
| **Experimental vs DFT-referenced accuracy** | UniFFBench, arXiv:2508.05762 |
| **Small-cell → large-cell transfer at solid–liquid interfaces** | Flagship *investigation* (hypothesis-labelled until supported by calculations) |

## 7. Competitive landscape and positioning

| Resource | What it does well | Gap the Atlas fills |
|---|---|---|
| **MLIP Hub** (mliphub.com) | Breadth: ~128 models, graph/timeline layouts, tag filters, verified-only edges, keyboard navigation | Hands-on review (Oct 2026): largely a **catalogue dump** — short entries, weak visuals, little real explanation. No physics, no 3D, no checkpoint-level evidence, no live computation |
| **Matbench Discovery** | Rigorous materials leaderboard, adjustable CPS, per-model metadata and checkpoints | Not explanatory; one domain |
| **MLIP Arena / MLIPAudit / LAMBench** | Physical-behaviour and domain benchmarks | Not a learning or navigation resource |
| **chemiscope** (lab-cosmo; 1.0 in JOSS 2026) | Linked map + 3D structure viewer; Jupyter/Streamlit/Sphinx embedding | A tool, not an atlas — *reuse its interaction idea* |
| **Atomify / lammps.js** | LAMMPS compiled to WebAssembly, live MD in the browser | Classical MD only — *reuse as a compute tier* |
| **MatterViz** (Svelte + three.js) | Crystal/trajectory/periodic-table web components, many file formats | Component library (pre-production) — *reference implementation* |
| **e3nn docs** | Tensor-product and spherical-harmonic visualisers in notebooks | No browser-native interactive explainer — *gap the Atlas fills* |

**Positioning:** *MLIP Hub lists who exists; leaderboards rank checkpoints; the Atlas explains why, shows how, and lets you test it.*

**Anti-patterns to avoid (lessons from existing directories):**
- **No entry without explanation.** A model is published only when it has a real explanation, at least one original figure and its evidence. Fewer entries, each one deep.
- **No metadata dumps.** Tags and filters support the explanations; they don't replace them.
- **Treat visuals as content, not decoration.** Every page has a purpose-built figure or interactive, not a generic card.
- **Breadth comes second.** Coverage grows only as fast as depth and quality allow; unfinished records stay as visible "in review" stubs, never as thin pages. Interoperate rather than duplicate: ingest leaderboard data with attribution, deep-link MLIP Hub entries, and export Atlas records openly.

---

# II. Experience design

## 8. 3D philosophy (v2)

**3D wherever there is geometry; uncompromising 2D wherever there is text, numbers or comparison.** Every 3D scene must teach a concept, reveal a measurement, or enable an action. Within that rule, the 3D should be the most beautiful scientific rendering on the web — think *Bartosz Ciechanowski's explorables × Mol\* rendering quality × Distill's editorial clarity*.

Use 3D for: atomic scenes, periodic cells, cutoff shells, receptive fields, dipoles, fields and isosurfaces, force vectors and disagreement halos, spherical harmonics, latent-space maps, trajectories, the optional constellation.

Use 2D for: equations, derivations, architecture pipelines with tensor shapes, tables, benchmark plots, citations, search and navigation.

## 9. Signature visual identity

- **Canvas:** warm off-white (~#F8FAFC) light theme and a true deep-slate dark theme (scientific dark mode, not neon).
- **Type:** a serif for long-form reading (e.g. Source Serif 4), a neutral grotesk for UI (e.g. Inter), a monospaced face for code and tensor shapes (e.g. JetBrains Mono), all self-hosted variable fonts.
- **Semantic accents:** deep blue = architecture, teal = physics, restrained purple = literature, amber = uncertainty/caveat, red reserved for documented failures. Element colours follow a CPK-derived palette tuned for contrast in both themes.
- **Atom rendering:** ray-cast sphere and cylinder impostors (pixel-perfect spheres at any zoom, thousands of atoms cheaply), screen-space ambient occlusion, subtle silhouette outlines, depth cueing — the Mol\* look, never glowing "AI orbs".
- **Fields:** translucent isosurfaces and volume slices for potentials and densities; streamlines for fields; arrows with physical-unit scale bars.
- **Motion:** slow, purposeful, interruptible; physics responds to input within one frame.
- **Mathematics:** typeset LaTeX, copyable source, and every symbol hoverable to highlight its counterpart in the 3D scene ("linked symbols").
- **Brand motif:** the atomic neighbourhood shell that expands into a network of ideas.

## 10. Homepage: a scientific portal

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ ◈ MLIP ATLAS   Explore  Physics Lab  Observatory  Models  Data  Failures  … │
├────────────────────────────────────────────────────────────────────────────┤
│  THE MLIP ATLAS                       ┌──────────────────────────────────┐ │
│  Mapping the science of               │  live 3D Au(111)/water interface │ │
│  machine-learned atomistic modeling   │  drag an atom · cutoff shell     │ │
│                                       │  [Local] [Message passing] [Long]│ │
│  [Explore the Atlas →] [Begin]        └──────────────────────────────────┘ │
│                                                                            │
│   BY PHYSICS        BY ARCHITECTURE        BY APPLICATION        BY DATA   │
│   "What changed this month"  ·  "How evidence is checked"  ·  Search ⌘K    │
└────────────────────────────────────────────────────────────────────────────┘
```

Skippable storyboard (≤ 20 s): an atom is selected → the cutoff shell appears → **Message passing** grows the effective receptive field layer by layer → **Long-range** shows schematic Ewald interactions in a distinct colour, labelled *schematic* → "Why does this matter?" opens the locality chapter. The first useful interaction takes less than three seconds.

## 11. Top-level destinations

| Destination | Key question | Signature interface |
|---|---|---|
| **Explore** | How is the field connected? | 2D atlas map, timeline, optional 3D constellation, ⌘K search |
| **Physics Lab** | What physics are we approximating? | 3D experiments + linked plots + linked-symbol equations |
| **Observatory** | What do real models actually predict here? | Live multi-model inference on any structure |
| **Models** | How does a model family work, and what is each checkpoint? | Monographs, Architecture X-ray, checkpoint cards, compare |
| **Data** | What did models learn from? | Dataset atlas, coverage maps, level-of-theory lineage |
| **Benchmarks** | What has been measured, and how? | Protocol-aware result explorer (never universal rankings) |
| **Failure Atlas** | Where and why do models fail? | Evidence-graded case studies with diagnostics |
| **Applications** | What matters for my system? | Scroll-driven 3D research journeys |
| **Papers** | Where did ideas come from? | Typed citation graph, chronology, paper records |
| **Learn** | Where should I begin? | Pathways across three depths |
| **About / Method** | How is the Atlas verified? | Methodology, provenance, changelog, contribute |

---

# III. Flagship experiences

Every experiment follows one template:

> **Question → 3D setup → linked plot → explanation → equations (linked symbols) → assumptions & units → connection to models → evidence → papers.**

Each carries a status badge: `TOY MODEL` · `PRECOMPUTED` · `LIVE CALCULATION` · `PUBLISHED RESULT` · `REPRODUCED`.

## 12. Physics Lab modules

| # | Module | 3D interaction | Linked 2D output | Compute tier |
|---|---|---|---|---|
| P1 | **Can an atom feel beyond the cutoff?** | Two charges → periodic ionic crystal; distance, cutoff, charge sliders | Coulomb vs explicitly-defined smooth-cutoff toy vs hybrid | Browser |
| P2 | **Receptive field** | Select atom; step message-passing layers; shells grow graph-hop by graph-hop (and contrast Allegro's strictly local design) | Atoms influencing *i* vs layers × r_c; information-flow Sankey | Browser |
| P3 | **Smooth cutoffs & body order** | Move a triplet/quadruplet; pairwise vs angular vs 4-body contributions | Energy decomposition by body order (toy ACE-like expansion) | Browser |
| P4 | **What changes under rotation?** | Rotate molecule; force arrows co-rotate; energy fixed | E invariant, F_i(QR)=QF_i(R) live; broken-symmetry counter-example | Browser + live |
| P5 | **Spherical harmonics & tensor products** | Y_l^m lobes on a neighbour; combine l₁⊗l₂ → allowed l (Clebsch–Gordan paths), parity rules | Irrep bookkeeping table, path diagram | Browser |
| P6 | **Ewald Lab** | Periodic point charges; split each into screened Gaussian + compensating charge; tune α | Real-space vs reciprocal-space vs self term; convergence vs α, k_max | Browser (WebGPU) |
| P7 | **Latent charges (LES intuition)** | Atoms coloured by learned latent charge from a real LES-augmented model on a precomputed structure | Latent charge distribution; Born-charge comparison where published | Precomputed |
| P8 | **Charge equilibration** | Edit electronegativity/hardness; charges flow to equalise chemical potential | QEq linear system solved live; total-charge constraint | Browser |
| P9 | **Periodic boundaries & finite size** | Replicate/resize cell; image interactions | Image separation; dipole-correction toggle for slabs | Browser |
| P10 | **PES softening** | Displace an atom along a phonon mode; compare DFT-reference curve and a universal-model curve | Curvature ratio; effect of one-point fine-tuning (published data) | Precomputed |
| P11 | **Transferability** | Morph from training-like to shifted configurations | Error vs descriptor distance (real data, labelled) | Precomputed |
| P12 | **Dynamics & stability** | Play two trajectories side by side, synchronised | Energy drift, RDF, MSD diagnostics | lammps.js / precomputed |

## 13. Flagship A — Electrochemical Interface Lab (EPhEct-inspired)

An illustrative Au(111)/water/ion system with a documented structure source, built around the four EPhEct phenomena plus the potential profile:

1. **Image-charge attraction** — drag an ion toward the metal; the classical image-charge curve vs predictions (live Observatory models and published results). Lesson: a purely local model has no mechanism for this.
2. **Interfacial water dipole** — rotate water layers; the planar-averaged ρ̄(z) and φ̄(z) update from a stated 1D Poisson model, d²φ̄/dz² = −ρ̄(z)/ε₀, with explicit boundary conditions, neutrality and reference conventions.
3. **LO–TO splitting** — a polar crystal; phonon dispersion with and without long-range treatment (precomputed); Born effective charges.
4. **Fermi-level pinning during ion discharge** — a schematic band diagram linked to a precomputed constant-potential trajectory, explicitly labelled as electronic-structure output, never as MLIP output.
5. **Constant potential** — compare strategies: a separate MLIP per potential (Ag/water), electron-count-conditioned models (constant-potential MACE), transfer learning (TRECI).

**Hard rule:** a DOS, band alignment or Fermi level shown here is a DFT (or electronic-structure-model) result on a named snapshot, not an energy/force MLIP prediction, unless the model explicitly outputs it.

## 14. Flagship B — Model Observatory (live)

The feature no other resource has: **load or pick a structure, run several real models on it, and see where they agree.**

- **Inputs:** gallery structures (bulk, surface, interface, molecule, defect, Observatory "stress tests") or upload CIF/extXYZ/POSCAR (validated, size-capped).
- **Models:** curated checkpoints with pinned versions, e.g. MACE-MH-1 heads, UMA, PET-MAD, Orb v3, SevenNet, GRACE-OAM, NequIP/Allegro-OAM, plus long-range variants (MACE-LES) where available — each with licence and checkpoint hash.
- **Outputs:** energies, per-atom forces, stress; 3D force arrows per model in distinct colours; a **disagreement halo** (per-atom force-vector spread) as a visual uncertainty proxy; small relaxations and short MD (batched with TorchSim).
- **Scans:** distance scans (ion–surface, dimer curves), strain scans, rotation tests (equivariance check), cell-size scans (small→large transfer).
- **Provenance panel:** model version, checkpoint hash, dtype, device, neighbour-list settings, timestamp, and reproduction code (Python snippet using ASE/TorchSim that reproduces the exact numbers).
- **Caching:** results keyed by (structure hash, checkpoint hash, settings), so popular scans are instant and permanent.
- **Framing:** "these models disagree here" is the message — never "model X is best".

## 15. Flagship C — Architecture X-ray

Real internals, not cartoon boxes. A small structure is passed through an actual model (MACE first, then NequIP, PET, an LES-augmented model); intermediate tensors are captured with forward hooks, precomputed and served.

- **Left:** 3D environment; atoms coloured by the norm of the chosen feature channel per irrep (l = 0, 1, 2…), with vector/tensor glyphs for l ≥ 1.
- **Centre:** an exact 2D dataflow of the architecture (embedding → radial basis → spherical harmonics → tensor product → symmetric contraction (body order) → update → readout) with tensor shapes; clicking a node selects that tensor.
- **Right:** the equation for that step with linked symbols.
- **Rotate test:** rotate the input and watch l = 0 features stay fixed while l = 1 glyphs co-rotate, from the real network.

## 16. Flagship D — Representation Map (chemiscope-style)

Inspired by chemiscope and the 2026 representation-convergence results: atomic environments from a reference set are embedded using several foundation models' latent features; a 2D/3D map is linked to the 3D structure viewer. Users see that environments cluster similarly across models, see which training dataset covers which region, and see where a user's structure falls — an intuitive out-of-distribution check.

## 17. Flagship E — Lineage & Data Universe

- **Timeline/lineage:** typed edges (*extends*, *reuses mechanism*, *fine-tuned from*, *trained on*, *evaluated on*, *compares against*) with documented vs inferred edges visually distinct.
- **Checkpoint genealogy:** architecture → pretraining dataset → fine-tune → heads (e.g. MACE-MP-0 → MACE-OMAT-0 → MACE-MH-1 heads; eSEN/UMA; OAM recipe).
- **Dataset coverage:** element × level-of-theory × configuration-type heatmaps; overlap and leakage warnings between training and test sets.
- **Optional 3D constellation** with identical 2D and list equivalents.

## 18. Failure Atlas

Each case: **Claim → 3D scene → evidence (with provenance) → competing mechanisms → diagnostics → modelling responses → open questions**, graded `THEORY` / `PUBLISHED RESULT` / `REPRODUCED` / `TOY DEMO` / `HYPOTHESIS`. Where possible, the Observatory reproduces the published behaviour live ("try it yourself"), and the case is labelled **REPRODUCED** only when the Atlas's own run matches the published result.

## 19. Application hubs

Materials · Energy (batteries, electrolytes, electrocatalysis) · Chemistry (reactions, solvation) · Molecular & biomolecular · Fundamental physics. Each follows **system → key physics → modelling requirements → architectures → published validations → limitations → papers**, as a scroll-driven 3D journey. Badges read **"evaluated for related systems"** with the reference, never "recommended".

## 20. Learning modes and pathways

Three depths on one canonical content base: **Intuition** (3D, minimal maths) · **Technical** (equations, pseudocode, caveats) · **Research** (assumptions, alternatives, open problems).

Pathways: (1) PES, energies, forces and stress; (2) descriptors → message passing → equivariance; (3) long-range physics, Ewald, charges and periodic systems; (4) data, fidelity and fine-tuning; (5) reading benchmarks critically; (6) MLIPs at electrochemical interfaces; (7) running and validating a foundation model for your system.

## 21. AI navigator

A grounded assistant that answers **only from Atlas records** and cites **claim IDs**, opens the relevant experiment or Observatory scan, and says "the Atlas has no evidence for this" when that is true. It navigates the Atlas; it never replaces it. Implemented with retrieval over the content graph and an LLM with citation output, evaluated against a held-out question set curated by domain reviewers.

---

# IV. Content model, provenance and rigour

## 22. Entities

```text
Concept ──explains──> PhysicalPrinciple
   │
   ├─relevant_to─> ModelFamily ─has─> Checkpoint ─trained_on─> Dataset ─computed_at─> LevelOfTheory
   │                   │                  │
   │                   │                  └─evaluated_in─> BenchmarkResult ─under─> Benchmark(protocol)
   │                   └─implemented_by─> Implementation(repo, version, licence)
   │
   ├─demonstrated_by─> Experiment(PhysicsLab / Observatory scan)
   ├─associated_with─> Application <── FailureCase
   └─described_in──> Paper ── supports ──> Claim (atomic statement, evidence grade, locator: fig/table/page)
```

**Claims are the atom of provenance.** Every non-trivial sentence on a page references one or more claim IDs; each claim points to a paper locator (figure/table/section) or to an Atlas reproduction.

## 23. Record examples

```yaml
# content/checkpoints/mace-mh-1.yaml
id: mace-mh-1
family: mace
name: "MACE-MH-1"
release: { version: "…", date: "YYYY-MM-DD", source: "ACEsuit/mace v0.3.15 release notes" }
elements: 89
heads: ["omat_pbe", "omol", "…"]
pretraining: [{ dataset: omat24 }]
finetuning: [ … ]
outputs: [energy, forces, stress]
long_range: none
licence: "…"
weights: { url: "https://huggingface.co/mace-foundations/mace-mh-1", sha256: "…" }
benchmark_results: [mbd-2026-xx-mace-mh-1]
known_issues: [claim:mace-mh-1-finetune-head-mismatch]
review: { status: draft, reviewers: [], last_checked: "2026-10-14" }
```

```yaml
# content/claims/softening-phonons.yaml
id: softening-phonons
statement: "M3GNet, CHGNet and MACE-MP-0 systematically underpredict phonon frequencies across 229 PhononDB compounds."
grade: PUBLISHED_RESULT
source: { paper: deng2024softening, locator: "Fig. X" }
scope: { checkpoints: [m3gnet-mp-2021, chgnet-0.3, mace-mp-0-medium] }
reproduction: null      # filled when the Observatory reproduces it
```

## 24. Editorial rules (non-negotiable)

1. Separate **architecture**, **checkpoint**, **implementation**, **dataset** and **evaluation** everywhere.
2. Every non-trivial claim has a claim ID with a locator, or is explicitly derived on the page.
3. Never merge benchmark numbers across incompatible protocols; every number carries its protocol, date and source.
4. Separate proven failure, plausible mechanism and illustrative toy.
5. Every simulation lists assumptions, units, boundary conditions and sign/reference conventions in view.
6. Distinguish what an energy/force MLIP predicts from what electronic-structure methods compute.
7. Records are versioned, timestamped and diffable; the changelog is public.
8. Visuals are original or licensed; reused figures are cited.
9. Corrections are one click away from every claim (pre-filled issue with the claim ID).
10. Live results show exact reproduction code.
11. Leaderboard snapshots are pulled automatically from the source, dated, and never editorialised into rankings.
12. Accuracy over breadth: a record is not published until it passes review.

---

# V. Technology — best-in-class stack

## 25. Architecture overview

```text
                    ┌───────────────────────── Content repo (Git = source of truth) ─────────────────────────┐
                    │ MDX chapters · YAML records (models, checkpoints, datasets, benchmarks, claims, papers) │
                    │ structures/ (CIF, extXYZ) · experiments/ · Zod schemas                                  │
                    └───────────────┬──────────────────────────────────────────────┬─────────────────────────┘
                                    │ CI: schema + link + DOI + physics tests      │ Python pipeline (uv)
                                    ▼                                              ▼
                     Astro 6 static build (SSG)                       ASE/pymatgen: structures → compact binary
                     + React islands (3D, plots, maps)                metadata: OpenAlex / Crossref / arXiv / S2
                     + Pagefind (full text) + Orama (facets)          leaderboards: Matbench Discovery YAML sync
                                    │                                 precompute: X-ray tensors, trajectories
                                    ▼                                              │
               CDN (Cloudflare Pages) ◄────── large assets on R2 (Parquet, trajectories, tensors) ◄─┘
                                    │
                 browser compute:   │  TS physics · WebGPU compute · lammps.js (WASM) · ONNX Runtime Web (R&D)
                                    │
                 live compute:      └──► Observatory API (FastAPI on Modal serverless GPUs)
                                          ASE + TorchSim (batched) + MACE / fairchem-UMA / upet / Orb / SevenNet …
                                          result cache (structure hash × checkpoint hash × settings)
```

## 26. Stack choices and rationale

| Layer | Choice | Why it is the best fit |
|---|---|---|
| Site framework | **Astro 6** (TypeScript strict) | Content-first static HTML with per-component islands, so reading pages ship almost no JS while 3D islands hydrate on demand; Content Collections with typed schemas; can host React *and* Svelte islands (so MatterViz components remain an option) |
| Interactive islands | **React 19** | Needed for the R3F ecosystem |
| 3D engine | **Three.js `WebGPURenderer`** (automatic WebGL2 fallback) + **TSL** shaders | WebGPU is now baseline across Chrome, Edge, Firefox and Safari 26; TSL compiles to WGSL and GLSL; enables GPU compute (Ewald, MD) in the same context. Test the Firefox/Linux fallback |
| 3D in React | **React Three Fiber v9** (async `gl` for WebGPU) + **drei** | Declarative scenes, shared state with plots; confirm against current pmndrs releases at kickoff |
| Atom rendering | Custom **impostor spheres/cylinders** in TSL + instancing; SSAO + outline post-processing | Mol\*-grade quality at thousands of atoms on laptops and phones |
| Biomolecules | **Mol\*** (embedded, only on bio pages) | Reference standard for proteins; avoid duplicating it |
| Structure I/O | **Build-time Python (ASE, pymatgen)** → compact typed binary + JSON metadata; lightweight TS parsers for uploads | Validated lattices/PBC once, offline; tiny client payloads |
| Shared state | **Zustand** (+ URL sync) | One state object drives scene, sliders, plots and equations; every state is shareable as a link |
| Plots | **Observable Plot** + **D3** for bespoke figures | Publication-quality, small, composable |
| Maths | **KaTeX at build time** (rehype-katex, HTML + MathML output) with linked-symbol hover layer | Fastest SSR; MathML keeps screen readers working. Revisit MathJax 4 if its accessibility explorer is needed |
| Scrollytelling & motion | **GSAP + ScrollTrigger** (now fully free) for narrative sequences; **Motion** for UI micro-interactions | Industry-standard timeline control; reduced-motion fallbacks |
| Knowledge map (2D) | **Sigma.js v3 + graphology** | WebGL graph rendering, layout and graph algorithms |
| Large paper graph | **Cosmograph** (GPU force layout) | Tens of thousands of nodes for the full citation universe |
| 3D constellation | **3d-force-graph / r3f-forcegraph** | Optional exploration layer |
| Search | **Pagefind** (static full text) + **Orama** (typed faceted search over records) + ⌘K palette | Fast, offline-capable, no server |
| Client analytics on data | **DuckDB-WASM** over Parquet | SQL-grade filtering of benchmark results in the browser |
| Styling | **Tailwind CSS v4** + **Radix / Ark UI** primitives + design tokens | Accessible primitives, consistent theming |
| In-browser classical MD | **lammps.js** (LAMMPS → WebAssembly, as used by Atomify) | Real MD with real potentials (LJ, EAM, Tersoff) for dynamics labs |
| In-browser MLIP (R&D track) | **ONNX Runtime Web (WebGPU EP)** with a small model exported with forces in the graph (or JAX → StableHLO path) | No existing browser MLIP demo was found; exporting `autograd.grad` forces is non-trivial. A working demo would be a genuine first, worth a short paper |
| Live inference | **FastAPI on Modal** serverless GPUs; **ASE** + **TorchSim** (batched; supports MACE, fairchem/UMA, SevenNet, ORB, MatterSim, metatomic/PET); request queue, rate limits, cache in Postgres/Redis + R2 | Scale-to-zero GPUs; pay per second; batched multi-model runs |
| Precompute | Lab GPU/HPC jobs orchestrated with **Snakemake** or **Prefect**; outputs to Parquet/Zarr with provenance manifests | Reproducible heavy trajectories, X-ray tensors, phonons |
| Bibliography | **OpenAlex** + **Crossref** + **arXiv API** (metadata), **Semantic Scholar** (citation edges), **Zotero** as the human curation front end; CSL for formatting | Open, scriptable, deduplicated |
| AI navigator | Retrieval over Atlas records + an LLM with citation output (current Claude models), with an offline evaluation set | Grounded, auditable answers |
| Hosting | **Cloudflare Pages** (static) + **R2** (assets); **Modal** (compute) | Global CDN, cheap egress, scale-to-zero |
| Feedback | **GitHub Issues** (claim-prefilled) + **Giscus** discussions | Transparent corrections |
| Analytics | **Plausible** or **Umami** | Privacy-respecting |
| Quality | **Vitest** (physics numerics vs analytic solutions), **Playwright** (+ screenshot regression of 3D scenes), **axe-core**, **Lighthouse CI**, schema/DOI/link checks in CI | Scientific and engineering correctness on every commit |
| Archival | **Zenodo** DOI per release, `CITATION.cff`, open content licence (e.g. CC BY 4.0) + MIT/Apache code | Citable and reusable |

## 27. Compute tiers (every result is labelled with its tier)

| Tier | Where | Used for | Badge |
|---|---|---|---|
| T0 | Browser, TypeScript | Analytic toys: Coulomb, QEq solve, 1D Poisson, cutoff functions | `TOY MODEL` |
| T1 | Browser, WebGPU compute | Ewald sums, many-particle toy MD, field volumes | `TOY MODEL` / `LIVE CALCULATION` |
| T2 | Browser, lammps.js | Real classical MD with standard potentials | `LIVE CALCULATION` |
| T3 | Browser, ONNX Runtime Web (R&D) | Small real MLIP on small systems | `LIVE CALCULATION` |
| T4 | Server GPU (Modal) | Foundation-model Observatory, scans, short MD | `LIVE CALCULATION` |
| T5 | Offline HPC | Long trajectories, phonons, X-ray tensors, DFT references | `PRECOMPUTED` |

## 28. Performance and accessibility budgets

- Reading pages: < 50 KB JS before any island hydrates; LCP < 1.5 s on mid-range mobile.
- 3D islands: lazy-loaded on visibility with a static high-quality poster (rendered at build time from the same scene); render-on-demand when idle; pause when hidden.
- Instancing and impostors for atoms; capped replicas with visible "simplified" notices; adaptive resolution on low-end GPUs.
- Full keyboard control of every experiment (sliders, atom selection by list, camera presets); screen-reader descriptions of each scene's state; reduced-motion compliance; static figure and text equivalents for every interactive.
- Every page, model, paper and concept is reachable by conventional navigation and search — the 3D map is never the only route.

---

# VI. Build sequence

Ordering follows dependencies, not resource limits.

**Phase 1 — Foundations.** Design system (both themes, type, colour, linked-symbol maths, captions, badges); Astro project; content schemas for every entity; CI quality gates; impostor-based atomic viewer (rotate, select, measure, cell, replicas, shells, vectors); structure pipeline; bibliography pipeline; Pagefind/Orama search.

**Phase 2 — Physics Lab core.** P1 locality vs Coulomb, P2 receptive field, P4 rotation, P5 spherical harmonics, P6 Ewald Lab, P8 QEq, P9 periodicity — each with linked plots, equations, assumptions and tests against analytic results. Homepage portal scene.

**Phase 3 — Knowledge base.** Model-family monographs (descriptor, invariant GNN, NequIP/Allegro, MACE, PET, Orb, eSEN/UMA, GRACE, SevenNet, DPA, LES/CACE/4G-HDNNP); checkpoint cards for the current foundation models; dataset atlas; benchmark explorer with automated Matbench Discovery sync; paper records with typed edges; the 2D atlas map and timeline.

**Phase 4 — Live science.** Observatory API (Modal + TorchSim) with caching and provenance; Observatory UI with disagreement halos and scans; Architecture X-ray (MACE first); lammps.js dynamics lab; Representation Map.

**Phase 5 — Electrochemistry & failures.** Electrochemical Interface Lab (all five panels); Failure Atlas seed cases (PES softening, surfaces, EPhEct phenomena, ion kinetics, small→large cell transfer); application hubs, starting with electrochemical interfaces and batteries.

**Phase 6 — Frontier features.** In-browser MLIP (ONNX Runtime Web); 3D constellation; Cosmograph citation universe; AI navigator; community contribution and review workflow; open API and data dumps; "what changed this month" digest.

**Definition of done for every flagship:** a newcomer can explain the concept afterwards without help, and a domain expert finds no unsupported claim.

---

# VII. Seed content checklist

- **Model families (monographs):** HDNNP/4G-HDNNP · GAP/SOAP · ACE/MTP · DeePMD/DPA (incl. DPA4C) · SchNet · M3GNet · CHGNet · NequIP · Allegro · MACE (+ MACE-LES) · SevenNet · GRACE · eSEN · EquiformerV2/V3 · UMA · Orb · PET/PET-MAD · CACE · LES · AIMNet2 · SpookyNet.
- **Checkpoints (cards):** MACE-MP-0 (s/m/l), MACE-OMAT-0, MACE-MH-1, UMA (s/m), PET-MAD, PET-OAM-XL, Orb v3, SevenNet-Omni, GRACE-2L/3L-OAM, NequIP-OAM-L/XL, Allegro-OAM-L, eSEN-30M-OAM, DPA-3.1/4, CHGNet, M3GNet.
- **Datasets:** MPtrj, Alexandria/sAlex, OMat24, MatPES, OMol25, SPICE 2, OC20/22, ODAC.
- **Benchmarks:** Matbench Discovery, MLIP Arena, MLIPAudit, LAMBench, UniFFBench, EPhEct.
- **Key papers to read and encode first:** Grasselli et al. 2026 (long-range taxonomy); Kim & Cheng 2026 (JCP perspective); LES (npj 2025) and its Nat. Commun. follow-up; EPhEct 2026; Deng et al. (softening); UMA (arXiv:2506.23971); OMat24 (arXiv:2410.12771); PET-MAD (Nat. Commun. 2025); TorchSim (arXiv:2508.06628); MLIP Arena (arXiv:2509.20630); MLIPAudit (arXiv:2511.20487); constant-potential MLFF (JCTC 2025); Ag/water CO₂RR (Nat. Commun. 2025); TRECI (arXiv:2511.19338).

---

# VIII. Success criteria and the PhD story

Worth doing if: students explain concepts correctly after a chapter; researchers find methods, caveats and sources they had missed; the Observatory is used to make real model-selection decisions; experts contribute corrections; figures are reused in teaching and group meetings (with attribution); and the in-browser MLIP or the Observatory produces a citable software paper (JOSS) or an educational paper.

Narrative (to be used only once it is true): *"While researching MLIPs I found the landscape of architectures, physical approximations and evidence hard to navigate, so I built an open interactive atlas that connects the physics, the models, the data and the evidence — including live multi-model comparisons — to understand the field deeply and help others do the same."*

---

# IX. Product commandments

1. Science first. Visual beauty second. Do both at the highest level.
2. Every interactive has a learning objective and a status badge.
3. 3D for geometry; 2D for clarity and comparison.
4. Assumptions, units and conventions are always visible.
5. Published claims are versioned evidence, not marketing.
6. Architecture ≠ checkpoint ≠ dataset ≠ result.
7. Show disagreement and uncertainty; never invent rankings.
8. Every live number is reproducible from the page.
9. Delight without sacrificing accessibility or speed.
10. A visitor leaves knowing *what* a model does, *why*, *when* it works, and *what remains uncertain*.

---

## Final positioning

**MLIP Atlas — the interactive scientific atlas of machine-learned interatomic potentials: explore atomic systems in 3D, run real models side by side, understand architectures and their mathematics, inspect the evidence and the failures, and follow ideas across the field.**

**Signature visual:** a precise, beautifully rendered 3D atomic environment coupled to living equations and plots.
**Signature intellectual feature:** a navigable path from **physical principle → architecture → checkpoint → data → evidence → application → open question**.
**Signature capability:** the **Observatory** — real foundation models, live, on your structure, with their disagreements made visible.

---

## Appendix — Research sources (gathered 14 Oct 2026)

Field and methods
- Grasselli et al., long-range electrostatics perspective — https://arxiv.org/abs/2602.11071
- Kim & Cheng, "Long-range electrostatics for MLIPs is easier than we thought," JCP 2026 — https://pubs.aip.org/aip/jcp/article/164/6/060901/3379367
- Latent Ewald Summation, npj Comput. Mater. 2025 — https://www.nature.com/articles/s41524-025-01577-7
- Learning charges and long-range interactions from energies and forces, Nat. Commun. 2025 — https://www.nature.com/articles/s41467-025-63852-x
- Universal long-range augmentation framework — https://arxiv.org/abs/2507.14302
- Polarizable atomic multipoles † — https://arxiv.org/abs/2605.05746
- Long-range MLIPs for SrTiO₃ defects † — https://arxiv.org/html/2610.00794v1
- UMA — https://arxiv.org/abs/2506.23971 · OMol25/UMA overview — https://rowansci.substack.com/p/the-omol25uma-release
- OMat24 — https://arxiv.org/abs/2410.12771
- PET-MAD — https://arxiv.org/abs/2503.14118
- MACE v0.3.15 (MACE-MH-1) — https://zenodo.org/records/18736115
- Latent features of universal MLIPs † — https://arxiv.org/abs/2512.05717
- Platonic representation of foundation MLIPs † — https://www.nature.com/articles/s42256-026-01235-7
- DPA4C, MLIP MD at empirical-potential speed † — https://arxiv.org/abs/2608.19041
- Foundation models for atomistic simulation (Nat. Rev. Chem.) — https://www.nature.com/articles/s41570-025-00793-5
- Reactive MLIPs review (Chem. Rev. 2026) — https://pubs.acs.org/chreay/article/126/8/4467/5150336
- TorchSim — https://arxiv.org/abs/2508.06628 · https://github.com/torchsim/torch-sim

Electrochemistry
- EPhEct benchmarks — https://arxiv.org/abs/2608.14153
- Constant-potential MLFF (JCTC 2025) — https://pubmed.ncbi.nlm.nih.gov/40717429/
- Ag/H₂O CO₂RR, Nat. Commun. 2025 — https://www.nature.com/articles/s41467-025-65630-1
- TRECI — https://arxiv.org/abs/2511.19338
- Fragment-constrained QEq † — https://arxiv.org/abs/2604.27910
- Electric-potential-embedded ML † — https://arxiv.org/abs/2604.07322

Failures and benchmarks
- Systematic softening — https://arxiv.org/abs/2405.07105
- Surfaces, ACS AMI 2024 — https://pubs.acs.org/doi/abs/10.1021/acsami.4c03815
- Cleavage-energy benchmark — https://arxiv.org/abs/2508.21663
- Alkali-ion kinetics † — https://arxiv.org/abs/2601.10938
- Heterogeneous catalysis † — https://arxiv.org/abs/2512.16702
- Migration barriers † — https://arxiv.org/abs/2512.03642
- UniFFBench — https://arxiv.org/abs/2508.05762
- MLIP Arena — https://arxiv.org/abs/2509.20630
- MLIPAudit — https://arxiv.org/abs/2511.20487
- Matbench Discovery — https://matbench-discovery.materialsproject.org/

Ecosystem and tools
- MLIP Hub — https://www.mliphub.com/
- chemiscope — https://github.com/lab-cosmo/chemiscope · JOSS 2026 — https://joss.theoj.org/papers/10.21105/joss.10380
- Atomify / lammps.js — https://github.com/lammps/lammps.js · https://www.lammps.org/ecosystem/lammpsjs/
- MatterViz — https://github.com/janosh/matterviz
- Mol\* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8262734/
- e3nn — https://docs.e3nn.org/
- Three.js WebGPU migration (2026) — https://www.utsubo.com/blog/webgpu-threejs-migration-guide
- TSL and WebGPU field guide — https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
- Astro 6 vs Next.js 16 for content sites — https://ecorpit.com/astro-6-vs-nextjs-16-content-sites-2026/
- GSAP (free, all plugins) — https://gsap.com/
- Cosmograph comparison — https://cosmograph.app/docs-lib/comparison/
- OpenAlex — https://arxiv.org/abs/2205.01833

† = seen only via abstract or secondary summary; verify before publishing.
