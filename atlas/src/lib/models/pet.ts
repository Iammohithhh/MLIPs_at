// A small, faithful Point Edge Transformer (PET) message-passing block with fixed random
// weights (a TOY MODEL: the functional form follows the paper, the weights are not
// trained), plus the Equivariant Coordinate System Ensemble (ECSE) that symmetrises it.
//
// Architecture as in Pozdnyakov & Ceriotti (NeurIPS 2023), Sec. 6, Fig. 2 and App. A:
//   neighbour token   t_j = M(concat(E_n(s_j), σ(L r_ij)))         (Eq. 5, first block)
//   centre token      t_0 = E_c(s_i)
//   transformer       self-attention over {t_0, t_j}, with the attention coefficients
//                     modulated α_ij ← α_ij f_c(r_j | R_c, ΔR_c) and renormalised (Sec. 6)
//   energy            y_i = H_c(x_0) + Σ_j H_n(x_j) f_c(r_ij | R_c, ΔR_c)    (App. A, Eq. 7)
// L acts on the Cartesian components of r_ij, so the block is NOT rotationally invariant.
// ECSE (Sec. 5, Eqs. 3–4; App. F.1) averages it over local frames built from ordered pairs
// of neighbours with smooth weights, which makes it exactly invariant.
// Simplifications (stated on the page): one block, one head, one transformer layer,
// no layer normalisation, 8 features.

import { mulberry32, randomRotation } from '../chem/rng';

export type Vec3 = [number, number, number];
/** Row-major 3×3. */
export type Mat3 = number[];

// ---------------------------------------------------------------------------
// Smooth switching functions (App. F.1, Eqs. 9–10)
// ---------------------------------------------------------------------------

/** f_c(r | R_c, ΔR_c): 1 for r ≤ R_c − ΔR_c, 0 for r ≥ R_c, C^∞ in between (Eq. 9). */
export function petCutoff(r: number, rc: number, width: number): number {
  if (r <= rc - width) return 1;
  if (r >= rc) return 0;
  const x = (2 * (r - rc + 0.5 * width)) / width;
  return 0.5 * (Math.tanh(1 / (x + 1) + 1 / (x - 1)) + 1);
}

/** q_c1(z | ω, Δω): 0 for z ≤ ω, 1 for z ≥ ω + Δω, C^∞ in between (Eq. 10). */
export function collinearSwitch(z: number, omega: number, dOmega: number): number {
  if (z <= omega) return 0;
  if (z >= omega + dOmega) return 1;
  const x = (2 * (z - omega - 0.5 * dOmega)) / dOmega;
  return 0.5 * (-Math.tanh(1 / (x + 1) + 1 / (x - 1)) + 1);
}

// ---------------------------------------------------------------------------
// Vectors and rotations
// ---------------------------------------------------------------------------

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3) => Math.sqrt(dot(a, a));
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function applyMat(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0]! * v[0] + m[1]! * v[1] + m[2]! * v[2],
    m[3]! * v[0] + m[4]! * v[1] + m[5]! * v[2],
    m[6]! * v[0] + m[7]! * v[1] + m[8]! * v[2],
  ];
}

/** Rotation by `angle` (radians) about `axis` (Rodrigues). */
export function axisRotation(axis: Vec3, angle: number): Mat3 {
  const [x, y, z] = scale(axis, 1 / norm(axis));
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

export function det3(m: Mat3): number {
  return (
    m[0]! * (m[4]! * m[8]! - m[5]! * m[7]!) -
    m[1]! * (m[3]! * m[8]! - m[5]! * m[6]!) +
    m[2]! * (m[3]! * m[7]! - m[4]! * m[6]!)
  );
}

/**
 * Local frame of an ordered pair of neighbour directions (App. F.1): v̂1, û1 = v̂1×v̂2/‖·‖,
 * û2 = v̂1×û1. Returned as a row-major matrix whose rows are the axes, so applying it to a
 * vector gives that vector's coordinates in the frame. Always a proper rotation.
 */
export function pairFrame(v1: Vec3, v2: Vec3): Mat3 {
  const a = scale(v1, 1 / norm(v1));
  const u = cross(a, scale(v2, 1 / norm(v2)));
  const b = scale(u, 1 / norm(u));
  const c = cross(a, b);
  return [...a, ...b, ...c];
}

// ---------------------------------------------------------------------------
// Atomic environment
// ---------------------------------------------------------------------------

export interface Neighbour {
  /** Displacement from the centre atom, Å. */
  v: Vec3;
  /** Species index into the model's embedding tables. */
  s: number;
}

export interface Environment {
  centre: number;
  neighbours: Neighbour[];
}

export function rotateEnv(env: Environment, m: Mat3): Environment {
  return { centre: env.centre, neighbours: env.neighbours.map((n) => ({ v: applyMat(m, n.v), s: n.s })) };
}

// ---------------------------------------------------------------------------
// The toy PET block
// ---------------------------------------------------------------------------

interface Dense {
  w: Float64Array; // out × in, row-major
  b: Float64Array;
  nIn: number;
  nOut: number;
}

function dense(rand: () => number, nIn: number, nOut: number): Dense {
  const s = Math.sqrt(6 / (nIn + nOut));
  return {
    w: new Float64Array(nIn * nOut).map(() => (2 * rand() - 1) * s),
    b: new Float64Array(nOut).map(() => (2 * rand() - 1) * 0.1),
    nIn,
    nOut,
  };
}

/** SiLU, the activation used in the paper's experiments (App. A). */
export const silu = (x: number) => x / (1 + Math.exp(-x));

function apply(d: Dense, x: Float64Array, act = false): Float64Array {
  const y = new Float64Array(d.nOut);
  for (let o = 0; o < d.nOut; o++) {
    let s = d.b[o]!;
    const row = o * d.nIn;
    for (let i = 0; i < d.nIn; i++) s += d.w[row + i]! * x[i]!;
    y[o] = act ? silu(s) : s;
  }
  return y;
}

export interface ToyPet {
  d: number;
  embedCentre: Float64Array[];
  embedNeighbour: Float64Array[];
  position: Dense; // L: 3 → d
  compress1: Dense; // M: 2d → d (hidden) → d
  compress2: Dense;
  q: Dense;
  k: Dense;
  v: Dense;
  o: Dense;
  ff1: Dense;
  ff2: Dense;
  headCentre1: Dense;
  headCentre2: Dense;
  headBond1: Dense;
  headBond2: Dense;
}

export function makeToyPet(o: { d?: number; species?: number; seed?: number } = {}): ToyPet {
  const d = o.d ?? 8;
  const rand = mulberry32(o.seed ?? 2023);
  const table = () => Array.from({ length: o.species ?? 4 }, () => new Float64Array(d).map(() => 2 * rand() - 1));
  return {
    d,
    embedCentre: table(),
    embedNeighbour: table(),
    position: dense(rand, 3, d),
    compress1: dense(rand, 2 * d, d),
    compress2: dense(rand, d, d),
    q: dense(rand, d, d),
    k: dense(rand, d, d),
    v: dense(rand, d, d),
    o: dense(rand, d, d),
    ff1: dense(rand, d, 2 * d),
    ff2: dense(rand, 2 * d, d),
    headCentre1: dense(rand, d, d),
    headCentre2: dense(rand, d, 1),
    headBond1: dense(rand, d, d),
    headBond2: dense(rand, d, 1),
  };
}

export type AttentionMode = 'pet' | 'plain';

export interface PetOptions {
  /** Cutoff radius R_c, Å. */
  rc: number;
  /** Width ΔR_c of the switching region, Å. */
  width: number;
  /** 'pet': α_ij ← α_ij f_c(r_j), renormalised (the paper); 'plain': the classical softmax. */
  mode?: AttentionMode;
}

export interface PetResult {
  /** Atomic contribution y_i (arbitrary units: the weights are random). */
  energy: number;
  /** Indices (into env.neighbours) of the neighbours that became tokens, r < R_c. */
  tokens: number[];
  /** The centre token's attention to each neighbour token, in the order of `tokens`. */
  attention: number[];
  /** The centre token's attention to itself. */
  selfAttention: number;
}

/** One PET message-passing block for one atomic environment: the backbone y_0(A_i). */
export function petForward(m: ToyPet, env: Environment, opt: PetOptions): PetResult {
  const d = m.d;
  const mode = opt.mode ?? 'pet';
  const tokens: number[] = [];
  const fc: number[] = [1]; // the centre token sits at r = 0
  const x: Float64Array[] = [Float64Array.from(m.embedCentre[env.centre]!)];
  env.neighbours.forEach((n, j) => {
    const r = norm(n.v);
    if (r >= opt.rc) return;
    tokens.push(j);
    fc.push(petCutoff(r, opt.rc, opt.width));
    const pos = apply(m.position, Float64Array.from(n.v), true);
    const cat = new Float64Array(2 * d);
    cat.set(m.embedNeighbour[n.s]!, 0);
    cat.set(pos, d);
    x.push(apply(m.compress2, apply(m.compress1, cat, true)));
  });
  const T = x.length;
  const q = x.map((t) => apply(m.q, t));
  const k = x.map((t) => apply(m.k, t));
  const v = x.map((t) => apply(m.v, t));
  const inv = 1 / Math.sqrt(d);
  const centreRow = new Float64Array(T);
  const alpha = new Float64Array(T);
  const mix = new Float64Array(d);
  const out: Float64Array[] = [];
  for (let i = 0; i < T; i++) {
    const qi = q[i]!;
    let smax = -Infinity;
    for (let j = 0; j < T; j++) {
      const kj = k[j]!;
      let acc = 0;
      for (let c = 0; c < d; c++) acc += qi[c]! * kj[c]!;
      alpha[j] = acc * inv;
      if (alpha[j]! > smax) smax = alpha[j]!;
    }
    let z = 0;
    for (let j = 0; j < T; j++) {
      alpha[j] = Math.exp(alpha[j]! - smax) * (mode === 'pet' ? fc[j]! : 1);
      z += alpha[j]!;
    }
    mix.fill(0);
    for (let j = 0; j < T; j++) {
      const a = alpha[j]! / z;
      if (i === 0) centreRow[j] = a;
      const vj = v[j]!;
      for (let c = 0; c < d; c++) mix[c] = mix[c]! + a * vj[c]!;
    }
    const h = apply(m.o, mix);
    for (let c = 0; c < d; c++) h[c] = h[c]! + x[i]![c]!;
    const f = apply(m.ff2, apply(m.ff1, h, true));
    for (let c = 0; c < d; c++) h[c] = h[c]! + f[c]!;
    out.push(h);
  }
  let energy = apply(m.headCentre2, apply(m.headCentre1, out[0]!, true))[0]!;
  for (let t = 1; t < T; t++) energy += apply(m.headBond2, apply(m.headBond1, out[t]!, true))[0]! * fc[t]!;
  return { energy, tokens, attention: Array.from(centreRow.subarray(1)), selfAttention: centreRow[0] ?? 1 };
}

// ---------------------------------------------------------------------------
// ECSE: Equivariant Coordinate System Ensemble (Sec. 5, Eqs. 3–4)
// ---------------------------------------------------------------------------

export interface EcseOptions {
  /** Cutoff of the neighbours that define frames, Å (may be smaller than the backbone's, Sec. 5). */
  rin: number;
  /** Width of its switching region, Å. */
  rinWidth: number;
  /** Collinearity switch on |r̂_j × r̂_j′|²: frames with z ≤ ω get zero weight. */
  omega: number;
  dOmega: number;
  /** 'smooth': w_jj′ of Eq. 4; 'plain': every pair inside r_in counts once (the paper's Fig. 1b failure). */
  weights?: 'smooth' | 'plain';
}

export interface EcseResult {
  energy: number;
  /** Ordered pairs (j, j′) with non-zero weight: the backbone evaluations needed. */
  frames: number;
  weightSum: number;
}

export function ecse(m: ToyPet, env: Environment, pet: PetOptions, o: EcseOptions): EcseResult {
  const smooth = (o.weights ?? 'smooth') === 'smooth';
  const inner = env.neighbours.map((n) => n.v).filter((v) => norm(v) < o.rin);
  let num = 0;
  let den = 0;
  let frames = 0;
  for (let a = 0; a < inner.length; a++)
    for (let b = 0; b < inner.length; b++) {
      if (a === b) continue;
      const va = inner[a]!;
      const vb = inner[b]!;
      const c = cross(scale(va, 1 / norm(va)), scale(vb, 1 / norm(vb)));
      const z = dot(c, c);
      const w = smooth
        ? petCutoff(norm(va), o.rin, o.rinWidth) * petCutoff(norm(vb), o.rin, o.rinWidth) * collinearSwitch(z, o.omega, o.dOmega)
        : z > o.omega
          ? 1
          : 0;
      if (w === 0) continue;
      frames++;
      num += w * petForward(m, rotateEnv(env, pairFrame(va, vb)), pet).energy;
      den += w;
    }
  return { energy: den > 0 ? num / den : Number.NaN, frames, weightSum: den };
}

// ---------------------------------------------------------------------------
// Experiments shown on the page
// ---------------------------------------------------------------------------

/** Mean and standard deviation of a prediction over n uniformly random rotations. */
export function rotationSpread(f: (e: Environment) => number, env: Environment, n: number, seed = 9): { mean: number; std: number } {
  const rand = mulberry32(seed);
  const ys = Array.from({ length: n }, () => f(rotateEnv(env, randomRotation(rand))));
  const mean = ys.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(ys.reduce((a, y) => a + (y - mean) ** 2, 0) / n);
  return { mean, std };
}

/** The environment with one extra atom of species `s` at distance `d` along unit direction `u`. */
export function withProbe(env: Environment, u: Vec3, d: number, s: number): Environment {
  return { centre: env.centre, neighbours: [...env.neighbours, { v: scale(u, d / norm(u)), s }] };
}

/**
 * The direction from the centre along which a probe moving from rMin to rMax stays farthest
 * from every neighbour (searched over a Fibonacci sphere). Used to place the probe atom.
 */
export function emptiestDirection(env: Environment, rMin: number, rMax: number, n = 400): Vec3 {
  let best: Vec3 = [0, 0, 1];
  let bestGap = -Infinity;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < n; k++) {
    const y = 1 - (2 * (k + 0.5)) / n;
    const rr = Math.sqrt(1 - y * y);
    const u: Vec3 = [Math.cos(golden * k) * rr, y, Math.sin(golden * k) * rr];
    let gap = Infinity;
    for (let t = 0; t <= 20; t++) {
      const p = scale(u, rMin + ((rMax - rMin) * t) / 20);
      for (const nb of env.neighbours) gap = Math.min(gap, norm([p[0] - nb.v[0], p[1] - nb.v[1], p[2] - nb.v[2]]));
    }
    if (gap > bestGap) {
      bestGap = gap;
      best = u;
    }
  }
  return best;
}
