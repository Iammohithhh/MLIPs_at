// Typed schema, validation and derived quantities for the precomputed Atom Intervention data
// written by pipeline/inside_mace/intervention.py. Pure module (no DOM), covered by experiment.test.ts.
//
// Rules this module enforces:
// - Numbers belong to an evaluated frame. There is no interpolation between frames; a requested
//   displacement snaps to the nearest evaluated frame.
// - A failed frame carries no numbers. Accessors return null for it, never 0.
// - Difference metrics are defined once, here, and documented next to their definition.

import { z } from 'zod';

export const SCHEMA_VERSION = 1;

const vec3 = z.tuple([z.number(), z.number(), z.number()]);

const changeLayer = z.object({
  h0: z.array(z.number()),
  h1: z.array(z.number()).nullable(),
});

const okFrame = z.object({
  k: z.number().int(),
  d: z.number(),
  status: z.literal('ok'),
  error: z.null(),
  seconds: z.number(),
  energy: z.number(),
  dE: z.number(),
  forces: z.array(vec3),
  nodeEnergy: z.array(z.number()),
  site: z.array(z.array(z.number())),
  movedPosition: vec3,
  change: z.array(changeLayer),
  chunk: z.string(),
});

const failedFrame = z.object({
  k: z.number().int(),
  d: z.number(),
  status: z.literal('failed'),
  error: z.string().min(1),
  seconds: z.number(),
});

export const frameSchema = z.discriminatedUnion('status', [okFrame, failedFrame]);

const fdStep = z.object({ h: z.number(), fd: z.number(), absDiff: z.number() });

export const manifestSchema = z.object({
  format: z.literal('inside-mace-experiment'),
  schema: z.literal(SCHEMA_VERSION),
  status: z.literal('PRECOMPUTED'),
  id: z.string(),
  title: z.string(),
  structure: z.object({
    source: z.string(),
    supercell: vec3,
    n: z.number().int().positive(),
    symbols: z.array(z.string()),
    positions: z.array(vec3),
    cell: z.tuple([vec3, vec3, vec3]),
    pbc: z.tuple([z.boolean(), z.boolean(), z.boolean()]),
    periodicImages: z.string(),
  }),
  displacement: z.object({
    atom: z.number().int(),
    partner: z.number().int(),
    axis: vec3,
    axisLabel: z.string(),
    unit: z.literal('Å'),
    values: z.array(z.number()),
    baselineIndex: z.number().int(),
    definition: z.string(),
  }),
  local: z.array(z.number().int()),
  shells: z.array(z.number()),
  rMax: z.number(),
  frames: z.array(frameSchema),
  baseline: z.object({
    e0: z.array(z.number()),
    norms: z.array(changeLayer),
    embed: z.array(z.number()),
  }),
  checks: z.object({
    restoration: z.object({ energyAbsDiff: z.number(), forceMaxAbsDiff: z.number() }),
    finiteDifference: z.array(
      z.object({ frame: z.number().int(), atom: z.number().int(), direction: vec3, label: z.string(), analytic: z.number(), steps: z.array(fdStep) }),
    ),
    rotation: z.object({
      R: z.tuple([vec3, vec3, vec3]),
      axis: vec3,
      angleDeg: z.number(),
      runs: z.array(
        z.object({
          frame: z.number().int(),
          energy: z.number(),
          energyAbsDiff: z.number(),
          forces: z.array(vec3),
          forceEquivarianceMaxAbs: z.number(),
          h0InvarianceMaxAbs: z.array(z.number()),
          h1EquivarianceMaxAbs: z.array(z.number().nullable()),
        }),
      ),
    }),
  }),
  provenance: z.record(z.string(), z.unknown()),
});

export type Manifest = z.infer<typeof manifestSchema>;
export type Frame = z.infer<typeof frameSchema>;
export type OkFrame = z.infer<typeof okFrame>;
export type Vec3 = [number, number, number];

export function parseManifest(raw: unknown): Manifest {
  const m = manifestSchema.parse(raw);
  const n = m.structure.n;
  if (m.structure.symbols.length !== n || m.structure.positions.length !== n) throw new Error('structure arrays disagree with n');
  if (m.frames.length !== m.displacement.values.length) throw new Error('frame count disagrees with displacement values');
  m.frames.forEach((f, k) => {
    if (f.k !== k || f.d !== m.displacement.values[k]) throw new Error(`frame ${k} out of order`);
    if (f.status === 'ok' && (f.forces.length !== n || f.nodeEnergy.length !== n)) throw new Error(`frame ${k} has wrong atom count`);
  });
  const base = m.frames[m.displacement.baselineIndex];
  if (!base || base.status !== 'ok' || base.d !== 0) throw new Error('baseline frame missing or not at d = 0');
  return m;
}

// ---------------------------------------------------------------------------
// Frames

/** Index of the evaluated frame nearest to displacement `d` (Å). Never interpolates. */
export function snapToFrame(m: Manifest, d: number): number {
  let best = 0;
  m.displacement.values.forEach((v, k) => {
    if (Math.abs(v - d) < Math.abs((m.displacement.values[best] ?? Infinity) - d)) best = k;
  });
  return best;
}

export function okFrameAt(m: Manifest, k: number): OkFrame | null {
  const f = m.frames[k];
  return f && f.status === 'ok' ? f : null;
}

export function baseline(m: Manifest): OkFrame {
  const f = okFrameAt(m, m.displacement.baselineIndex);
  if (!f) throw new Error('baseline frame is not available');
  return f;
}

// ---------------------------------------------------------------------------
// Physical comparisons (eV, eV/Å)

/** E(frame) − E(baseline) in eV, or null if the frame failed. */
export function energyChange(m: Manifest, k: number): number | null {
  const f = okFrameAt(m, k);
  return f ? f.energy - baseline(m).energy : null;
}

export function sub(a: readonly number[], b: readonly number[]): Vec3 {
  return [(a[0] ?? NaN) - (b[0] ?? NaN), (a[1] ?? NaN) - (b[1] ?? NaN), (a[2] ?? NaN) - (b[2] ?? NaN)];
}
export function dot(a: readonly number[], b: readonly number[]): number {
  return (a[0] ?? NaN) * (b[0] ?? NaN) + (a[1] ?? NaN) * (b[1] ?? NaN) + (a[2] ?? NaN) * (b[2] ?? NaN);
}
export function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

/** Force on `atom` in frame k (eV/Å), or null. */
export function force(m: Manifest, k: number, atom: number): Vec3 | null {
  const f = okFrameAt(m, k)?.forces[atom];
  return f ? [f[0], f[1], f[2]] : null;
}

/** ΔF = F(frame) − F(baseline) on `atom` (eV/Å), or null. */
export function forceChange(m: Manifest, k: number, atom: number): Vec3 | null {
  const f = force(m, k, atom);
  const b = force(m, m.displacement.baselineIndex, atom);
  return f && b ? sub(f, b) : null;
}

/** Projection of the moved atom's force on the displacement axis, F·û (eV/Å). */
export function axialForce(m: Manifest, k: number): number | null {
  const f = force(m, k, m.displacement.atom);
  return f ? dot(f, m.displacement.axis) : null;
}

/** ΔE_i = E_i(frame) − E_i(baseline), the change in atom i's energy contribution (eV), or null. */
export function siteEnergyChange(m: Manifest, k: number, atom: number): number | null {
  const f = okFrameAt(m, k);
  const v = f?.nodeEnergy[atom];
  const b = baseline(m).nodeEnergy[atom];
  return v === undefined || b === undefined ? null : v - b;
}

// ---------------------------------------------------------------------------
// Feature-change metrics
//
// Definition (computed in the pipeline in float64):
//   change_t,l(i) = ‖ h_t,l(i; frame) − h_t,l(i; baseline) ‖₂
// over all 128 channels (and the three Cartesian components for l = 1). This is a feature-block
// *difference norm*, not the difference of two block norms.
//
// Relative change divides by the baseline block norm with a floor, so a near-zero baseline block
// (for example l = 1 features that vanish by the crystal's symmetry) cannot blow up the ratio:
//   rel = change / max(‖h(baseline)‖, floor · median_i ‖h(baseline)‖)
// Values from different layers or irreps are never put on one colour scale.

export const RELATIVE_FLOOR = 0.05;

export type FeatureBlock = 'h0' | 'h1';

export function featureChange(m: Manifest, k: number, layer: number, block: FeatureBlock, atom: number): number | null {
  const v = okFrameAt(m, k)?.change[layer]?.[block]?.[atom];
  return v === undefined ? null : v;
}

export function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  if (!s.length) return NaN;
  const mid = s.length >> 1;
  return s.length % 2 ? (s[mid] as number) : ((s[mid - 1] as number) + (s[mid] as number)) / 2;
}

export function relativeFeatureChange(m: Manifest, k: number, layer: number, block: FeatureBlock, atom: number): number | null {
  const c = featureChange(m, k, layer, block, atom);
  const norms = m.baseline.norms[layer]?.[block];
  const b = norms?.[atom];
  if (c === null || !norms || b === undefined) return null;
  const floor = RELATIVE_FLOOR * median(norms);
  const denom = Math.max(b, floor);
  return denom > 0 ? c / denom : null;
}

/** Largest change of a block over all atoms and frames: a stable colour-scale maximum for one block. */
export function blockScaleMax(m: Manifest, layer: number, block: FeatureBlock): number | null {
  let max = 0;
  let any = false;
  for (const f of m.frames) {
    if (f.status !== 'ok') continue;
    const arr = f.change[layer]?.[block];
    if (!arr) continue;
    any = true;
    for (const v of arr) if (v > max) max = v;
  }
  return any ? max : null;
}

// ---------------------------------------------------------------------------
// Geometry

/** Minimum-image vector from atom i to atom j in the (orthorhombic or general) cell, Å. */
export function mic(m: Manifest, from: Vec3, to: Vec3): Vec3 {
  const cell = m.structure.cell;
  const inv = invert3(cell);
  const d = sub(to, from);
  // fractional = d · inv(cell) (rows of `cell` are lattice vectors)
  const f: Vec3 = [dot([d[0], d[1], d[2]], [inv[0][0], inv[1][0], inv[2][0]]), dot(d, [inv[0][1], inv[1][1], inv[2][1]]), dot(d, [inv[0][2], inv[1][2], inv[2][2]])];
  const r = f.map((x) => x - Math.round(x)) as Vec3;
  return [
    r[0] * cell[0][0] + r[1] * cell[1][0] + r[2] * cell[2][0],
    r[0] * cell[0][1] + r[1] * cell[1][1] + r[2] * cell[2][1],
    r[0] * cell[0][2] + r[1] * cell[1][2] + r[2] * cell[2][2],
  ];
}

export function invert3(a: readonly (readonly number[])[]): [Vec3, Vec3, Vec3] {
  const g = (i: number, j: number) => a[i]?.[j] ?? NaN;
  const det =
    g(0, 0) * (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) - g(0, 1) * (g(1, 0) * g(2, 2) - g(1, 2) * g(2, 0)) + g(0, 2) * (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0));
  const c = (i: number, j: number) => {
    const r = [0, 1, 2].filter((x) => x !== j);
    const s = [0, 1, 2].filter((x) => x !== i);
    const [r0, r1] = r as [number, number];
    const [s0, s1] = s as [number, number];
    return ((i + j) % 2 ? -1 : 1) * (g(r0, s0) * g(r1, s1) - g(r0, s1) * g(r1, s0));
  };
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(i, j) / det)) as [Vec3, Vec3, Vec3];
}

/** Atom positions for frame k: the baseline geometry with only the moved atom displaced. */
export function framePositions(m: Manifest, k: number): Vec3[] | null {
  const f = okFrameAt(m, k);
  if (!f) return null;
  return m.structure.positions.map((p, i) => (i === m.displacement.atom ? f.movedPosition : p) as Vec3);
}
