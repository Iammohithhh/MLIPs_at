// A small, faithful SchNet forward pass with fixed random weights (a TOY MODEL: the
// architecture is real, the weights are not trained). Used on the SchNet page to show
// what a distance-only, invariant message-passing network can and cannot "see".
//
// Architecture as in Schütt et al. (NeurIPS 2017, Sec. 3–4.1; JCP 2018, Sec. II):
//   embedding        x_i⁰ = a_{Z_i}
//   RBF expansion    e_k(r) = exp(−γ (r − μ_k)²), μ_k on a 0.1 Å grid, γ = 10 Å⁻²
//   filter network   W(r) = ssp(Dense(ssp(Dense(e(r)))))
//   cfconv           y_i = Σ_j (W_in x_j) ∘ W(r_ij) · f_c(r_ij)
//   interaction      v_i = Dense(ssp(Dense(y_i))),   x_i ← x_i + v_i
// The cosine cutoff multiplying the filter follows SchNetPack's implementation
// (representation/schnet.py); the original NeurIPS model used no cutoff for molecules.

export type Vec3 = readonly [number, number, number];

/** Shifted softplus, ssp(x) = ln(0.5 eˣ + 0.5); ssp(0) = 0. Numerically stable form. */
export function ssp(x: number): number {
  return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))) - Math.LN2;
}

/** Cosine cutoff f_c(r) = ½(cos(πr/r_c) + 1) for r < r_c, else 0. */
export function cosineCutoff(r: number, rc: number): number {
  return r < rc ? 0.5 * (Math.cos((Math.PI * r) / rc) + 1) : 0;
}

export interface RbfSpec {
  /** Centre spacing, Å (paper: 0.1 Å). */
  spacing: number;
  /** Width parameter γ, Å⁻² (paper: 10). */
  gamma: number;
  /** Last centre, Å. Centres run from 0 to `max` inclusive. */
  max: number;
}

export const PAPER_RBF: Omit<RbfSpec, 'max'> = { spacing: 0.1, gamma: 10 };

export function rbfCentres(spec: RbfSpec): number[] {
  const n = Math.round(spec.max / spec.spacing) + 1;
  return Array.from({ length: n }, (_, k) => k * spec.spacing);
}

export function rbfExpand(r: number, spec: RbfSpec): Float64Array {
  const mu = rbfCentres(spec);
  const out = new Float64Array(mu.length);
  mu.forEach((m, k) => {
    out[k] = Math.exp(-spec.gamma * (r - m) ** 2);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Fixed random weights
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Dense {
  w: Float64Array; // out × in, row-major
  b: Float64Array;
  nIn: number;
  nOut: number;
}

function dense(rand: () => number, nIn: number, nOut: number, bias = true): Dense {
  // Uniform Glorot-style initialisation keeps activations O(1) through the stack.
  const s = Math.sqrt(6 / (nIn + nOut));
  const w = new Float64Array(nIn * nOut).map(() => (2 * rand() - 1) * s);
  const b = new Float64Array(nOut).map(() => (bias ? (2 * rand() - 1) * 0.1 : 0));
  return { w, b, nIn, nOut };
}

function apply(d: Dense, x: ArrayLike<number>, act = false): Float64Array {
  const y = new Float64Array(d.nOut);
  for (let o = 0; o < d.nOut; o++) {
    let s = d.b[o]!;
    const row = o * d.nIn;
    for (let i = 0; i < d.nIn; i++) s += d.w[row + i]! * x[i]!;
    y[o] = act ? ssp(s) : s;
  }
  return y;
}

interface Interaction {
  inProj: Dense;
  filter1: Dense;
  filter2: Dense;
  out1: Dense;
  out2: Dense;
}

export interface ToySchNet {
  features: number;
  rbf: RbfSpec;
  rc: number;
  embedding: Map<number, Float64Array>;
  blocks: Interaction[];
}

export interface ToyOptions {
  features?: number;
  blocks?: number;
  rc: number;
  seed?: number;
  elements?: number[];
  rbf?: Omit<RbfSpec, 'max'>;
}

export function makeToySchNet(o: ToyOptions): ToySchNet {
  const F = o.features ?? 16;
  const rand = mulberry32(o.seed ?? 2017);
  const rbf: RbfSpec = { ...(o.rbf ?? PAPER_RBF), max: o.rc };
  const nRbf = rbfCentres(rbf).length;
  const embedding = new Map<number, Float64Array>();
  for (const z of o.elements ?? [1, 6, 7, 8, 14]) embedding.set(z, new Float64Array(F).map(() => 2 * rand() - 1));
  const blocks = Array.from({ length: o.blocks ?? 3 }, () => ({
    inProj: dense(rand, F, F, false),
    filter1: dense(rand, nRbf, F),
    filter2: dense(rand, F, F),
    out1: dense(rand, F, F),
    out2: dense(rand, F, F),
  }));
  return { features: F, rbf, rc: o.rc, embedding, blocks };
}

/** The continuous filter of one block at distance r, already multiplied by f_c(r). */
export function filterAt(model: ToySchNet, block: number, r: number): Float64Array {
  const b = model.blocks[block]!;
  const w = apply(b.filter2, apply(b.filter1, rbfExpand(r, model.rbf), true), true);
  const fc = cosineCutoff(r, model.rc);
  return w.map((v) => v * fc);
}

const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * Forward pass. Returns the atom features after every block: layers[t][i] is x_i after t
 * blocks (t = 0 is the embedding). Neighbours are all other atoms with r_ij < r_c.
 */
export function forward(model: ToySchNet, Z: readonly number[], pos: readonly Vec3[]): Float64Array[][] {
  const n = Z.length;
  let x = Z.map((z) => {
    const e = model.embedding.get(z);
    if (!e) throw new Error(`no embedding for Z = ${z}`);
    return Float64Array.from(e);
  });
  const layers = [x.map((v) => Float64Array.from(v))];
  const pairs: Array<{ i: number; j: number; r: number }> = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const r = dist(pos[i]!, pos[j]!);
      if (r < model.rc) pairs.push({ i, j, r });
    }
  model.blocks.forEach((b, t) => {
    const proj = x.map((v) => apply(b.inProj, v));
    const y = x.map(() => new Float64Array(model.features));
    for (const { i, j, r } of pairs) {
      const w = filterAt(model, t, r);
      const yi = y[i]!;
      const pj = proj[j]!;
      for (let f = 0; f < model.features; f++) yi[f]! += pj[f]! * w[f]!;
    }
    x = x.map((xi, i) => {
      const v = apply(b.out2, apply(b.out1, y[i]!, true));
      return xi.map((a, f) => a + v[f]!);
    });
    layers.push(x.map((v) => Float64Array.from(v)));
  });
  return layers;
}

// ---------------------------------------------------------------------------
// The bond-angle experiment
// ---------------------------------------------------------------------------

/** Centre atom i at the origin, j on +x, k in the xy plane at angle θ (degrees) from j. */
export function triatomic(r: number, thetaDeg: number): Vec3[] {
  const t = (thetaDeg * Math.PI) / 180;
  return [
    [0, 0, 0],
    [r, 0, 0],
    [r * Math.cos(t), r * Math.sin(t), 0],
  ];
}

/** Distance between the two outer atoms, r_jk = 2 r sin(θ/2). */
export const outerDistance = (r: number, thetaDeg: number) => 2 * r * Math.sin((thetaDeg * Math.PI) / 360);

/**
 * Largest angle at which j and k are still within each other's cutoff, or 180 if they
 * always are: θ* = 2 arcsin(r_c / 2r).
 */
export function blindAngle(r: number, rc: number): number {
  return rc >= 2 * r ? 180 : (360 / Math.PI) * Math.asin(rc / (2 * r));
}

const norm = (v: ArrayLike<number>) => Math.sqrt(Array.from(v).reduce((s, a) => s + a * a, 0));

export interface AngleScan {
  thetas: number[];
  /** diff[t][k] = ‖x_i^(t+1)(θ_k) − x_i^(t+1)(θ_ref)‖ / ‖x_i^(t+1)(θ_ref)‖ for the centre atom. */
  diff: number[][];
  refTheta: number;
}

/** How much the centre atom's features change with the bond angle, after each block. */
export function angleScan(model: ToySchNet, Z: readonly [number, number, number], r: number, thetas: number[], refTheta = 180): AngleScan {
  const ref = forward(model, Z, triatomic(r, refTheta));
  const diff = model.blocks.map(() => [] as number[]);
  for (const th of thetas) {
    const L = forward(model, Z, triatomic(r, th));
    model.blocks.forEach((_, t) => {
      const a = L[t + 1]![0]!;
      const b = ref[t + 1]![0]!;
      diff[t]!.push(norm(a.map((v, f) => v - b[f]!)) / norm(b));
    });
  }
  return { thetas, diff, refTheta };
}
