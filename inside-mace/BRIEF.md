# MACE: Inside the Prediction — owner's brief

This file is the owner's instruction for the `inside-mace/` project. **Follow it as is. Do not simplify,
shorten, or skip anything. The result must be beautiful and good. Consult the web, papers, repositories
and package documentation whenever that helps.**

## Owner's placement decision (the only adaptation to the text below)

The owner asked for this to be built as a **second project in its own subfolder**, `inside-mace/` at the
repository root, separate from the MLIP Atlas site in `atlas/`. Where the brief below says "within the
existing MLIP Atlas project", read it as:

- `inside-mace/` is its own app (Astro 7 + React + TypeScript strict, React Three Fiber/drei, the same
  stack and conventions as `atlas/`), with its own `package.json`, tests, and README.
- Its CPU data pipeline lives in `inside-mace/pipeline/` (Python, own `requirements*.txt`).
- **Reuse** the Atlas work the brief names — the recorded MACE forward pass, equations, checkpoint
  metadata and safe checkpoint loading (`pipeline/atlas_pipeline/xray.py`), the 3D primitives
  (`atlas/src/three/`), the X-ray explorer, tests and verification approach — by reading it and copying or
  adapting it into `inside-mace/` with a note of where it came from.
- **Do not modify anything outside `inside-mace/`.** The owner has uncommitted local work in `atlas/` and
  `pipeline/`. Because `atlas/` is untouched, its existing URLs (including `/models/x-ray`) stay
  functional automatically.
- The Atlas's visual identity is the starting point for this project's visual identity (§3).
- The project rules in the root `CLAUDE.md` and `CONTRIBUTING.md` apply here too: scientific rules,
  result labels (`TOY MODEL`, `LIVE CALCULATION`, `PRECOMPUTED`, `PUBLISHED RESULT`, `REPRODUCED`), no GPU
  servers, at most 2 agents/subagents at a time, TeX-in-Astro conventions, `:global()` for script-created
  elements, ShaderMaterial `projectionMatrix` declaration, Playwright screenshot checks, no publishing or
  deploying.

---

## The brief (verbatim)

es. I'd take it one step beyond the explainer:
MACE: Inside the Prediction
Move one atom. Trace what changes inside the model.

The additional attraction is an Atom Intervention Lab: compare an original structure with a slightly perturbed version, inspect how internal features change, and follow the resulting changes in energy and forces.
That gives people something to investigate themselves. It also builds directly on your existing Architecture X-ray and CPU pipelines. It is a proposed interaction—not a claim of novel interpretability research or guaranteed X reach.
For implementation, these are my recommended tools:

| Purpose | Recommendation |
|---|---|
| Application | Keep Astro, React, and TypeScript |
| Scientific 3D | Three.js, React Three Fiber, drei, and your existing atom-rendering primitives |
| Animation | GSAP for coordinated camera, annotation, and diagram transitions; use ScrollTrigger only where scrolling helps the explanation |
| Plots and architecture diagrams | SVG/D3 or Observable Plot; keep text, equations, and comparisons sharp |
| Rendering performance | Instancing, adaptive resolution, and render-on-demand, following R3F's guidance |
| Optional finishing effects | Subtle ambient occlusion through React Postprocessing, only after verifying compatibility with your custom atom shaders |
| Model calculations | Existing Python, PyTorch, MACE, and ASE pipeline; CPU-precomputed experiments initially |
| Verification | Vitest, Playwright, numerical checks, and visual inspection |

I would retain the working renderer initially. Although Three.js documents WebGPU with a WebGL 2 fallback, migrating your custom shaders should be a separate, justified decision.
Here is the copy-ready implementation prompt:

```
Build "MACE: Inside the Prediction" within the existing MLIP Atlas project.

This is a flagship scientific learning experience and interactive instrument.
It must make a real MACE forward pass understandable, inspectable, and
interesting to explore.

The central invitation is:

"Move one atom. Trace what changes inside the model."

The experience must work for:
- A science/engineering student encountering MLIPs for the first time.
- An ML practitioner curious about equivariant atomistic networks.
- An atomistic researcher who wants to inspect the actual computation.

Do not promise research novelty, model reliability, or virality. Produce
something scientifically defensible that people could use in teaching,
inspect technically, and demonstrate in a short screen recording.


1. WORK WITH THE EXISTING PROJECT

Inspect the repository before implementing. Read its project guide,
contribution instructions, scientific rules, and relevant content schemas.

Preserve unrelated changes in the working tree.

Start by inspecting:
- atlas/src/pages/models/x-ray.astro
- atlas/src/islands/XrayExplorer.tsx
- atlas/src/islands/XrayExplorer.css
- atlas/src/lib/models/xray.ts
- atlas/src/data/xray/mace-mp-0a-medium.json
- pipeline/atlas_pipeline/xray.py
- atlas/src/three/
- Existing model, physics, provenance, and integrity tests.

Reuse the existing recorded MACE forward pass, equations, checkpoint
metadata, scene primitives, and verification work wherever appropriate.

Build a canonical flagship page. Keep existing URLs functional through
direct links or appropriate redirects if the page structure changes.

The rest of the Atlas remains accessible as supporting material.
Do not redesign every model page as part of this release.

Deliver a working local implementation. Do not publish or deploy it.


2. THE PRODUCT EXPERIENCE

The opening screen must present an actual scientific interaction.

Suggested headline:
"Inside an atomic prediction."

Suggested supporting text:
"Follow a real MACE checkpoint from atomic geometry to energy and forces.
Then change one atom and inspect the difference."

Primary action:
"Move an atom"

Secondary action:
"Walk me through it"

Keep checkpoint identity and computation status visible.

Do not require visitors to watch an introduction, create an account,
read a long preamble, or understand irreducible representations before
they can begin.

There are two connected modes:

A. Guided walkthrough
A short, carefully sequenced explanation of one forward pass.

B. Atom Intervention Lab
A controlled comparison between a baseline structure and a perturbed
structure, using real evaluated model outputs.

Both modes use the same structures, visual language, and underlying data.
Switching modes should preserve the visitor's context where possible.


3. VISUAL DIRECTION

Create the feeling of a carefully designed scientific instrument.

Use:
- A deep graphite or mineral-green experiment stage.
- Restrained mint/teal accents for selection and structure.
- Warm amber for changes between baseline and intervention.
- Comfortable, high-contrast reading surfaces for explanations.
- Strong typography with short headings and compact technical labels.
- A clear distinction between controls, explanations, and measured outputs.

Build on the Atlas's established visual identity.

Give the atomic structure generous space. It should be the visual anchor.

Avoid:
- Generic fully connected neural-network diagrams standing in for MACE.
- Decorative particle clouds unrelated to the computation.
- Excessive bloom, glass panels, or floating cards.
- Long cinematic delays.
- Tiny plots and equations surrounding an enormous decorative model.
- Motion that implies physical propagation speed or inference timing
  when it is only a presentation sequence.

Every prominent visual should explain a relationship, expose a value,
or help the visitor perform an action.


4. ATOMIC SCENE QUALITY

Use the existing Three.js/R3F rendering foundation.

Improve the scene through:
- Legible atom silhouettes and consistent material treatment.
- Soft depth cues and controlled lighting.
- Clear distinction between selected atoms and their neighbours.
- Thin, readable graph edges.
- Optional cutoff shells.
- Force arrows with an explicit magnitude scale and units.
- Clean camera presets: overview, selected environment, and comparison.
- Stable labels that remain readable against the scene.
- Sensible periodic-image handling for crystals.

Distinguish graph edges from chemical bonds. Label them accurately.

Use instancing or existing impostors for repeated atoms and edges.

If introducing postprocessing, first verify that custom impostor depth
and normal information behaves correctly. Disable effects that cause
incorrect occlusion, muddy colours, or poor mobile performance.

Selection should use an outline, marker, or halo as well as colour.

Do not use element colours to encode feature values without showing
which colour mode is active.

Camera movement should preserve orientation and help explain a transition.
Provide reset and preset controls. Avoid continuous automatic rotation.


5. THE GUIDED WALKTHROUGH

Follow one selected atom through the actual computation.

Use these conceptual stages, adapting their exact boundaries to the
captured checkpoint and implementation:

1. Atomic species and coordinates.
2. Neighbour construction.
3. Radial and angular information.
4. Neighbour aggregation and message construction.
5. Higher-order feature construction and updates.
6. Atomic energy contributions and total energy.
7. Forces from the total-energy gradient.

At each stage show:

- A one-sentence explanation in plain language.
- The relevant part of the atomic scene.
- A compact view of the actual tensor or selected channels.
- The associated equation, available without overwhelming beginners.
- A link to the corresponding implementation location.
- What changed compared with the previous stage.

Use the exact computational structure of the selected MACE checkpoint.

Higher-order correlations, tensor products, feature updates, and readouts
must not be replaced by a generic network cartoon.

Show selected values from real recorded tensors. State clearly when a
diagram is schematic or shows only a subset of channels.

Important force explanation:
Forces are derivatives of total energy with respect to coordinates.
Do not imply that atom i's force is simply the gradient of its own atomic
energy contribution.


6. COORDINATED INTERACTION

Connect the views through one shared state.

Selecting an atom should update:
- Its highlighted neighbourhood.
- Relevant feature values.
- Its atomic energy contribution, where appropriate.
- Its force.
- The explanatory text.

Selecting a computation stage should update:
- Scene overlays.
- Architecture diagram emphasis.
- Tensor/channel inspector.
- Equation and implementation reference.

Selecting a channel should update:
- The corresponding feature visualisation.
- Its numerical value or components.
- A clear statement of what that visual represents.

Use progressive disclosure:
- Default: intuition and a few meaningful values.
- Technical: equations, tensor shapes, channel selection.
- Provenance: checkpoint, versions, settings, and reproduction instructions.

Do not duplicate scientific content across disconnected beginner and
expert implementations.


7. SIGNATURE ADDITION: THE ATOM INTERVENTION LAB

Build this as the main attention-grabbing interaction.

The visitor chooses a supported atom and moves it along a clearly marked,
constrained displacement path.

Show baseline and perturbed states together, with synchronised cameras.

The visitor should be able to inspect:
- Which geometry descriptors change.
- Changes in selected intermediate features.
- Changes in atomic energy contributions.
- Change in total energy.
- Changes in force vectors.

Use amber to highlight differences.

Offer:
- Baseline.
- Perturbed.
- Difference.

Make the question explicit:
"What changes when this atom moves?"

A second question can be:
"Which parts of the model respond?"

Do not label feature differences as:
- Explanations of chemical reasoning.
- Confidence estimates.
- Prediction errors.
- Proof that a feature caused the output.
- Calibrated importance or uncertainty.

They are observed changes in a fixed model under a specified input
perturbation.

Keep this distinction readable and concise.


8. COMPUTE DESIGN FOR THE INTERVENTION

The first complete version should work on static hosting with
CPU-precomputed model evaluations.

Start with suitable periodic silicon or ZnO examples already supported
by the pipeline. Check and explain the checkpoint's applicable domain.

If using ethanol with a materials-trained checkpoint, explicitly explain
the domain mismatch. Do not use the result as evidence of molecular
accuracy.

For each supported experiment:
- Fix the checkpoint, reference structure, cell, and numerical settings.
- Choose a physically interpretable displacement axis.
- Evaluate a documented sequence of displacement frames.
- Capture the required model outputs and intermediate quantities.
- Preserve atom correspondence and periodic-image conventions.
- Store enough precision for the intended comparisons.

Choose frame spacing and displacement limits after inspecting the
resulting curves and geometries.

Prevent unsupported free-form dragging.

The drag handle or slider must snap to evaluated frames. Display the
selected displacement and units.

If camera or geometry transitions are interpolated for presentation,
identify that animation as interpolation. Numerical outputs must remain
tied to evaluated frames.

Do not show fabricated model outputs between frames.

Clearly label:
"Precomputed model evaluations; interactive inspection."

Arbitrary-geometry live inference is a separate extension requiring an
actual inference service or a verified browser implementation. Do not
silently simulate that capability.

Store provenance including:
- Checkpoint identity and hash.
- Relevant package versions.
- Device and dtype.
- Input geometry and boundary conditions.
- Displacement definition.
- Model settings.
- Data-generation command or script.
- Data/schema version.

Handle failed evaluations explicitly.


9. THE VISUAL REVEAL

Give the intervention an engaging sequence:

- Show the baseline structure.
- Invite the visitor to move the highlighted atom.
- Immediately show the selected evaluated geometry and force changes.
- Offer "Trace the change".
- Highlight the affected representations stage by stage.
- Finish at the energy and force comparison.

The trace animation is an explanation of computational dependencies.
It is not a measurement of runtime or information travel speed.

Animate individual edges or contributions only when the stored data
supports that level of detail. Otherwise highlight the affected stage
or atom-level representation.

An optional "Predict before revealing" prompt may ask whether a selected
quantity will change. Reveal the computed result and explain the scope.

Keep this within the same experience; do not build a separate gamified
system with points, accounts, or leaderboards.


10. ROTATION DEMONSTRATION

Include a compact, real rotation comparison:

"What changes when the coordinate frame changes?"

Distinguish:
- Camera rotation, which only changes the view.
- Rotation of the model input, which requires corresponding model outputs.

Show:
- Energy invariance within a stated numerical tolerance.
- Force-vector transformation.
- Appropriate transformation behaviour of selected feature blocks.

Use recorded model runs or explicitly computed representation
transformations, labelled correctly.

Do not compare arbitrary feature components across rotated frames as if
the coordinates were unchanged.

This should be a scientifically accurate, visually memorable moment
that can be understood in a short screen recording.


11. FEATURE AND PLOT DESIGN

Use:
- Compact channel heatmaps for scalar arrays.
- Vector glyphs for appropriate vector representations.
- Clearly explained representations for higher-order features.
- Linked plots for displacement versus energy or force.
- Stable axes and colour scales during baseline/intervention comparison.

Avoid presenting higher-order features as ordinary arrows.

Define each difference metric. For example, a feature-block difference
norm is not the same as the difference between two feature norms.

Do not compare raw magnitudes across unrelated layers as though their
scales were directly comparable.

Normalisation must be documented and handle near-zero baselines safely.

Show units for physical outputs. Explain the meaning and conventions of
internal features.

Never use model disagreement or feature change as a substitute for error
against a reference calculation.


12. DESKTOP AND MOBILE UX

Desktop:
- Give most space to the atomic scene.
- Keep the selected explanation and essential controls adjacent.
- Use a compact computation-stage navigator.
- Expand advanced tensor inspection on demand.
- Keep baseline/intervention comparisons easy to read.

Mobile:
- Present one main scene at a time.
- Toggle baseline, intervention, and difference.
- Keep the displacement control close to the scene.
- Use a readable inspector below it.
- Preserve essential functionality.
- Provide buttons or a slider as an alternative to dragging.
- Avoid tiny horizontal architecture diagrams and hover-only controls.

Include:
- Keyboard-operable controls.
- Visible focus.
- Clear selected states.
- Reset experiment.
- Reset camera.
- Reduced-motion behaviour.
- Useful loading and error states.
- Static figure and text fallback when WebGL is unavailable.

Scrolling must remain normal. Any pinned sequence must have an equally
usable button-driven alternative.


13. VISUAL AND UI TOOLS

Prefer the existing stack:
- Astro for content and page structure.
- React and TypeScript for the interactive instrument.
- React Three Fiber, Three.js, and drei for scientific scenes.
- Existing custom atom-rendering primitives.
- Zustand or the existing state approach for linked views.
- GSAP for coordinated transitions.
- SVG/D3 or Observable Plot for diagrams and numerical comparisons.
- KaTeX for equations.

Install additional packages only where they solve a concrete need.

Do not migrate the whole renderer just to use newer technology.

Blender may be used for a static cover image or scene study if useful.
Scientific geometry should come from documented structures.

External assets must have appropriate licences and attribution.
Do not use stock molecular imagery to imply a calculated result.

AI-generated imagery can support promotional artwork, but it must not
stand in for atomic structures, model outputs, or scientific figures.


14. SHARING AND THE X RELEASE

Every supported experiment state should have a reproducible URL.

Encode or reference:
- Experiment/data version.
- Structure and checkpoint.
- Selected atom.
- Displacement frame.
- Computation stage.
- Relevant channel or feature selection.
- Comparison mode.

Validate URL values and handle obsolete or unsupported states gracefully.

Add a presentation mode with:
- Large scene.
- Minimal controls.
- Short explanatory captions.
- Visible checkpoint and computation-status labels.
- A clear link back to the full instrument.

Provide high-quality still-image export if feasible, including:
- A descriptive title.
- Experiment settings.
- Units and legends.
- Computation status.
- Attribution and source URL.

Prepare a reproducible short demonstration sequence for recording:
1. Show the structure and predicted forces.
2. Move one atom along the evaluated path.
3. Reveal the changes inside the model.
4. Show the energy/force comparison.
5. End with the invitation to try the same experiment.

The recording must demonstrate actual supported behaviour.

Use launch language such as:
"Move one atom. Inspect what changes inside a real MACE checkpoint."

Do not claim:
- The first such tool without evidence.
- That the model understands chemistry.
- That the demonstration establishes reliability.
- Live computation when the outputs were precomputed.


15. SCIENTIFIC AND ENGINEERING VERIFICATION

Add meaningful tests for new numerical and interaction logic.

Check:
- Baseline restoration reproduces the original outputs.
- Stored outputs agree with source computations within documented
  precision and tolerances.
- Energy changes are calculated consistently.
- Force conventions are correct.
- Selected force components agree with suitable finite differences of
  total energy, with step-size and numerical considerations documented.
- Rotation comparisons use the correct transformation.
- Atom and periodic-image identities remain consistent.
- Every displayed number belongs to the selected evaluated frame.
- Missing data never becomes a fabricated zero.
- Shared URLs restore the same experiment state.

Browser checks must cover:
- Guided navigation.
- Atom selection.
- Displacement controls.
- Baseline/intervention/difference views.
- Reset.
- Sharing.
- Keyboard operation.
- Mobile layouts.
- Reduced motion.
- WebGL failure or static fallback.
- Runtime errors and horizontal overflow.

Inspect screenshots at multiple viewport sizes.

Measure interaction performance on a representative mobile device or
emulation setup. Lazy-load advanced data, cap rendering resolution, and
pause work when the scene is hidden.

Retain existing repository checks and run those appropriate to the changes.


16. DELIVERABLES AND COMPLETION CRITERIA

Deliver:
- The working flagship page.
- A complete guided real-model walkthrough.
- The Atom Intervention Lab with evaluated data.
- A real rotation comparison.
- Reproducible data-generation instructions.
- Documentation of interpretation limits.
- Shared-state URLs.
- Presentation-ready demonstration states.
- Relevant tests and a verification summary.
- An explanation of remaining limitations.

The experience is complete when:
- A newcomer can follow geometry → representation → energy → forces.
- A researcher can identify the checkpoint and reproduce the displayed run.
- Moving the supported atom produces inspectable, real model-output
  changes without misleading interpolation.
- The main interaction is compelling in a short recording.
- The page remains useful after the visual novelty wears off.

Do the implementation and verification, not just the scaffolding.
Preserve scientific correctness when visual ambition and model behaviour
come into conflict.
```

The strongest part of this brief is the baseline → intervention → internal changes → output interaction. It gives your existing X-ray a memorable action and a technical contribution you can discuss, while the guided walkthrough preserves your original teaching goal.
