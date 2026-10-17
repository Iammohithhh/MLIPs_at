// The Deep Potential family (DeepPot-SE, DPA-1, DPA-2, DPA-3) as small, exact calculations.
//
// DeepPot-SE maps each neighbour j of atom i to a row of a generalised coordinate matrix,
// R̃_j = s(r_ij)·(1, x/r, y/r, z/r), embeds s(r_ij) with a small network G, and builds the
// descriptor D = (G¹)ᵀ R̃ R̃ᵀ G² (Zhang et al., NeurIPS 2018, Eqs. 8–11). DPA-1 inserts gated
// self-attention between neighbours before the product (Zhang et al. 2024, Eqs. 6–10), DPA-2
// stacks "repformer" layers and makes every softmax smooth (Eqs. 34–36), and DPA-3 is a message-
// passing network on the line-graph series: atoms, bonds, angles (Zhang et al. 2025, Eqs. 3–20).
//
// Three things on the DPA page are computed here:
//  1. What each descriptor can tell apart. Because R̃_j · R̃_k = s_j s_k (1 + cos θ_jk), the
//     DeepPot-SE descriptor splits exactly into a radial part and a "filtered dipole" part, so it
//     sees angles only through l ≤ 1 moments. DPA-1's gate r̂_j · r̂_k is also linear in each
//     direction. A DPA-3-style angle embedding φ(cos θ) is not. Toy networks with random weights
//     make the consequences visible in pairs of environments (fcc vs hcp, etc.).
//  2. Smoothness at the cutoff: the four switching functions and DPA-2's smooth softmax.
//  3. The energy bias: per-element reference energies fitted by least squares from compositions
//     (DPA-2 Eq. 6; DPA-3 Eq. 3), and when they are not identifiable.
//
// Weights are random and untrained. The structural results (invariance, exact equalities of
// descriptors, which construction can separate which environments) do not depend on them.

import { mulberry32 } from '../chem/rng';

export type Vec3 = [number, number, number];

// ---------------------------------------------------------------------------
// Switching functions
// ---------------------------------------------------------------------------

/** DeepPot-SE's weight s(r) (NeurIPS 2018, Eq. 9): 1/r, cosine-switched to 0 between r_cs and r_c. */
export function sDeepPotSE(r: number, rcs: number, rc: number): number {
  if (r >= rc) return 0;
  if (r < rcs) return 1 / r;
  return (0.5 * Math.cos((Math.PI * (r - rcs)) / (rc - rcs)) + 0.5) / r;
}

/** The C² switch w(r) of DPA-1 (Eqs. 2–3) and DPA-2 (Eq. 13): u³(−6u² + 15u − 10) + 1. */
export function wPoly(r: number, rcs: number, rc: number): number {
  if (r >= rc) return 0;
  if (r < rcs) return 1;
  const u = (r - rcs) / (rc - rcs);
  return u * u * u * (-6 * u * u + 15 * u - 10) + 1;
}

/** DPA-1/DPA-2's switched inverse distance s(r) = w(r)/r (DPA-2 Eq. 12). */
export function sPoly(r: number, rcs: number, rc: number): number {
  return wPoly(r, rcs, rc) / r;
}

/** DPA-3's switch (Eq. 9): exp(−exp(C(r − r_cs)/r_cs)) inside r_c, 0 beyond. C = 20 in the paper. */
export function sDpa3(r: number, rcs: number, rc: number, C = DPA3_C): number {
  if (r > rc) return 0;
  return Math.exp(-Math.exp((C * (r - rcs)) / rcs));
}

/** DPA-3 settings for G(1): C = 20, r_c = 6.0 Å, r_cs = 5.3 Å (Sec. 4.3, after Eq. 9). */
export const DPA3_C = 20;
export const DPA3_G1 = { rc: 6.0, rcs: 5.3 } as const;
/** r_cs/r_c used for every switch in the toys: DPA-3's 5.3/6. */
export const RCS_RATIO = DPA3_G1.rcs / DPA3_G1.rc;
/** DPA-3.1-3M: L = 16 layers, LiGS order K = 2, cutoffs 6.0 Å (bonds) and 4.0 Å (angles) (Table S-9). */
export const DPA31_3M = { layers: 16, order: 2, rcBond: 6.0, rcAngle: 4.0 } as const;

export interface SwitchDef {
  id: 'deeppot-se' | 'poly' | 'dpa3' | 'hard';
  label: string;
  /** The weight as each paper defines it (with the 1/r factor where the paper has one). */
  f: (r: number, rcs: number, rc: number) => number;
}

export const SWITCHES: SwitchDef[] = [
  { id: 'hard', label: 'hard cutoff, 1/r', f: (r, _rcs, rc) => (r < rc ? 1 / r : 0) },
  { id: 'deeppot-se', label: 'DeepPot-SE cosine', f: sDeepPotSE },
  { id: 'poly', label: 'DPA-1/DPA-2 polynomial', f: sPoly },
  { id: 'dpa3', label: 'DPA-3 double exponential', f: (r, rcs, rc) => sDpa3(r, rcs, rc) },
];

/** f, f′ and f″ at r by central differences (use away from r_c and r_cs). */
export function derivs(f: (r: number) => number, r: number, h = 1e-4): [number, number, number] {
  const a = f(r - h);
  const b = f(r);
  const c = f(r + h);
  return [b, (c - a) / (2 * h), (c - 2 * b + a) / (h * h)];
}

export interface Jumps {
  /** Value just inside r_c minus value just outside, for f, f′ and f″. */
  value: number;
  first: number;
  second: number;
}

/** p(0), p′(0), p″(0) of the cubic through (t_k, y_k), t in units of the spacing. */
function cubicAtZero(ts: number[], ys: number[]): [number, number, number] {
  // Solve the 4 × 4 Vandermonde system for p(t) = c0 + c1 t + c2 t² + c3 t³.
  const A = ts.map((t, k) => [1, t, t * t, t * t * t, ys[k]!]);
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[piv]![c]!)) piv = r;
    [A[c], A[piv]] = [A[piv]!, A[c]!];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = A[r]![c]! / A[c]![c]!;
      for (let k = c; k < 5; k++) A[r]![k]! -= f * A[c]![k]!;
    }
  }
  const coef = A.map((row, k) => row[4]! / row[k]!);
  return [coef[0]!, coef[1]!, 2 * coef[2]!];
}

/**
 * Jumps of f, f′, f″ across r = rc (inside minus outside), each side extrapolated to rc from a
 * cubic through four points at rc ∓ δ, rc ∓ 2δ, ... so the discontinuity itself is never sampled. Resolution with the
 * default δ: about 10⁻¹² in f, 10⁻⁹ in f′ and 10⁻⁶ in f″ for the switches here.
 */
export function jumpsAt(f: (r: number) => number, rc: number, delta = 3e-5): Jumps {
  const ts = [1, 2, 3, 4];
  const [vi, di, si] = cubicAtZero(ts.map((t) => -t), ts.map((t) => f(rc - t * delta)));
  const [vo, dout, so] = cubicAtZero(ts, ts.map((t) => f(rc + t * delta)));
  return { value: vi - vo, first: (di - dout) / delta, second: (si - so) / (delta * delta) };
}

// ---------------------------------------------------------------------------
// Small networks
// ---------------------------------------------------------------------------

interface Layer {
  W: Float64Array; // out × in, row-major
  b: Float64Array;
  nin: number;
  nout: number;
}
export interface Mlp {
  layers: Layer[];
}

function normal(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function makeLayer(rand: () => number, nin: number, nout: number, gain = 1): Layer {
  const W = new Float64Array(nin * nout);
  const b = new Float64Array(nout);
  const sc = gain / Math.sqrt(nin);
  for (let k = 0; k < W.length; k++) W[k] = sc * normal(rand);
  for (let k = 0; k < nout; k++) b[k] = 0.3 * normal(rand);
  return { W, b, nin, nout };
}

export function makeMlp(rand: () => number, sizes: number[], gain = 1.5): Mlp {
  const layers: Layer[] = [];
  for (let k = 0; k + 1 < sizes.length; k++) layers.push(makeLayer(rand, sizes[k]!, sizes[k + 1]!, gain));
  return { layers };
}

function affine(l: Layer, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(l.nout);
  for (let o = 0; o < l.nout; o++) {
    let s = l.b[o]!;
    for (let i = 0; i < l.nin; i++) s += l.W[o * l.nin + i]! * x[i]!;
    y[o] = s;
  }
  return y;
}

/** tanh after every layer: the embedding nets of DeePMD-kit use tanh. */
export function runMlp(net: Mlp, x: ArrayLike<number>): Float64Array {
  let h: ArrayLike<number> = x;
  for (const l of net.layers) {
    const y = affine(l, h);
    for (let k = 0; k < y.length; k++) y[k] = Math.tanh(y[k]!);
    h = y;
  }
  return h as Float64Array;
}

function linear(l: Layer, x: ArrayLike<number>): Float64Array {
  const y = affine(l, x);
  for (let o = 0; o < l.nout; o++) y[o]! -= l.b[o]!; // bias-free projection (Q, K, V)
  return y;
}

/** Layer normalisation without learnable gain and bias. */
export function layerNorm(x: ArrayLike<number>, eps = 1e-5): Float64Array {
  const n = x.length;
  let m = 0;
  for (let k = 0; k < n; k++) m += x[k]!;
  m /= n;
  let v = 0;
  for (let k = 0; k < n; k++) v += (x[k]! - m) ** 2;
  v /= n;
  const d = Math.sqrt(v + eps);
  const y = new Float64Array(n);
  for (let k = 0; k < n; k++) y[k] = (x[k]! - m) / d;
  return y;
}

// ---------------------------------------------------------------------------
// 1. Descriptors on one atomic environment
// ---------------------------------------------------------------------------

export interface Toy {
  M1: number;
  M2: number;
  dk: number;
  /** DeepPot-SE/DPA-1 embedding net G(s): 1 → 12 → M1. */
  embed: Mlp;
  /** DPA-1 attention layers: Q, K (M1 → dk), V (M1 → M1). */
  att: Array<{ Q: Layer; K: Layer; V: Layer }>;
  /** DPA-3-style toy: bond embedding φ_e(r) (Eq. 18), angle embedding φ_a(cos θ) (Eq. 19),
   *  angle message φ_c (Eqs. 6–7) and vertex readout. */
  bond: Mlp;
  angle: Mlp;
  msg: Mlp;
  vertex: Mlp;
}

export const TOY_DIMS = { M1: 16, M2: 4, dk: 8, e: 8 } as const;
const TOY_NMAX = 64;

export function makeToy(seed = 11): Toy {
  const rand = mulberry32(seed);
  const { M1, M2, dk, e } = TOY_DIMS;
  return {
    M1,
    M2,
    dk,
    embed: makeMlp(rand, [1, 12, M1]),
    att: [0, 1].map(() => ({ Q: makeLayer(rand, M1, dk), K: makeLayer(rand, M1, dk), V: makeLayer(rand, M1, M1) })),
    bond: makeMlp(rand, [1, e, e]),
    angle: makeMlp(rand, [1, e, e]),
    msg: makeMlp(rand, [3 * e, e, e]),
    vertex: makeMlp(rand, [e, e, e]),
  };
}

export interface EnvOptions {
  /** Cutoff and switch start for the radial weight s(r). */
  rc: number;
  rcs: number;
  /** Angle cutoff of the DPA-3-style toy (DPA-3.1-3M: 4 Å). */
  rcAngle?: number;
}

interface Prepared {
  r: number[];
  u: Vec3[]; // unit vectors
  s: number[];
}

function prepare(env: Vec3[], o: EnvOptions): Prepared {
  const r: number[] = [];
  const u: Vec3[] = [];
  const s: number[] = [];
  for (const p of env) {
    const d = Math.hypot(p[0], p[1], p[2]);
    if (d >= o.rc) continue;
    r.push(d);
    u.push([p[0] / d, p[1] / d, p[2] / d]);
    s.push(sPoly(d, o.rcs, o.rc));
  }
  return { r, u, s };
}

/** T = Gᵀ R̃ (M1 × 4): column 0 is the radial sum Σ G s, columns 1–3 the filtered dipole Σ G s r̂. */
function tMatrix(G: Float64Array[], P: Prepared, M1: number): number[][] {
  const T = Array.from({ length: M1 }, () => [0, 0, 0, 0]);
  P.s.forEach((s, j) => {
    const g = G[j]!;
    const row = [s, s * P.u[j]![0], s * P.u[j]![1], s * P.u[j]![2]];
    for (let a = 0; a < M1; a++) for (let c = 0; c < 4; c++) T[a]![c]! += g[a]! * row[c]!;
  });
  return T;
}

/** D = T T₂ᵀ (M1 × M2) flattened; T₂ is the first M2 rows of T (DeepPot-SE Eq. 11; DPA-1 Eq. 10). */
function dFromT(T: number[][], M1: number, M2: number): Float64Array {
  const D = new Float64Array(M1 * M2);
  for (let a = 0; a < M1; a++)
    for (let b = 0; b < M2; b++) {
      let v = 0;
      for (let c = 0; c < 4; c++) v += T[a]![c]! * T[b]![c]!;
      D[a * M2 + b] = v;
    }
  return D;
}

export interface SeResult {
  D: Float64Array;
  /** D split by the Gram identity: radial part (column 0 of T) and dipole part (columns 1–3). */
  radial: Float64Array;
  dipole: Float64Array;
  /** |Σ_j G_a(s_j) s_j r̂_j| for each filter a. */
  dipoleNorms: number[];
  neighbours: number;
}

/** DeepPot-SE descriptor (NeurIPS 2018, Eqs. 8–11), with its exact radial/dipole split. */
export function seDescriptor(toy: Toy, env: Vec3[], o: EnvOptions): SeResult {
  const P = prepare(env, o);
  const G = P.s.map((s) => runMlp(toy.embed, [s]));
  const T = tMatrix(G, P, toy.M1);
  const D = dFromT(T, toy.M1, toy.M2);
  const radial = new Float64Array(toy.M1 * toy.M2);
  const dipole = new Float64Array(toy.M1 * toy.M2);
  for (let a = 0; a < toy.M1; a++)
    for (let b = 0; b < toy.M2; b++) {
      radial[a * toy.M2 + b] = T[a]![0]! * T[b]![0]!;
      dipole[a * toy.M2 + b] = T[a]![1]! * T[b]![1]! + T[a]![2]! * T[b]![2]! + T[a]![3]! * T[b]![3]!;
    }
  const dipoleNorms = T.map((t) => Math.hypot(t[1]!, t[2]!, t[3]!));
  return { D, radial, dipole, dipoleNorms, neighbours: P.r.length };
}

export interface Dpa1Result {
  D: Float64Array;
  /** Largest |attention update| before layer normalisation, per layer. Zero means the layer did nothing. */
  updateNorm: number[];
  /** Attention weights of the last layer, N × N (row j attends over k), gate included. */
  weights: number[][];
}

/**
 * DPA-1 descriptor with l gated self-attention layers (Eqs. 6–10):
 * φ = softmax(Q Kᵀ/√d_k) ⊙ R̂ R̂ᵀ, G ← G + LayerNorm(φ V), D = Ĝᵀ R̃ R̃ᵀ Ġ.
 * Single species, so the type embedding of Eq. 5 is a constant and is left out.
 */
export function dpa1Descriptor(toy: Toy, env: Vec3[], o: EnvOptions, layers = 2, gate = true): Dpa1Result {
  const P = prepare(env, o);
  const N = P.r.length;
  let G = P.s.map((s) => runMlp(toy.embed, [s]));
  const updateNorm: number[] = [];
  let weights: number[][] = [];
  for (let l = 0; l < layers; l++) {
    const { Q, K, V } = toy.att[l % toy.att.length]!;
    const q = G.map((g) => linear(Q, g));
    const k = G.map((g) => linear(K, g));
    const v = G.map((g) => linear(V, g));
    weights = [];
    let un = 0;
    const next: Float64Array[] = [];
    for (let j = 0; j < N; j++) {
      const logits = k.map((kk) => {
        let d = 0;
        for (let c = 0; c < toy.dk; c++) d += q[j]![c]! * kk[c]!;
        return d / Math.sqrt(toy.dk);
      });
      const m = Math.max(...logits);
      const e = logits.map((x) => Math.exp(x - m));
      const z = e.reduce((a, b) => a + b, 0);
      const w = e.map((x, kk) => {
        const g = gate ? P.u[j]![0] * P.u[kk]![0] + P.u[j]![1] * P.u[kk]![1] + P.u[j]![2] * P.u[kk]![2] : 1;
        return (x / z) * g;
      });
      weights.push(w);
      const A = new Float64Array(toy.M1);
      w.forEach((wk, kk) => {
        for (let a = 0; a < toy.M1; a++) A[a]! += wk * v[kk]![a]!;
      });
      un = Math.max(un, Math.hypot(...A));
      const ln = layerNorm(A);
      const g = new Float64Array(toy.M1);
      for (let a = 0; a < toy.M1; a++) g[a] = G[j]![a]! + ln[a]!;
      next.push(g);
    }
    G = next;
    updateNorm.push(un);
  }
  const D = dFromT(tMatrix(G, P, toy.M1), toy.M1, toy.M2);
  return { D, updateNorm, weights };
}

export interface Dpa3Result {
  v: Float64Array;
  bonds: number;
  angles: number;
}

/**
 * A one-layer toy in the spirit of DPA-3 with LiGS order K = 2. Bonds carry φ_e(r) (Eq. 18), angles
 * carry φ_a(cos θ) (Eq. 19); each bond is updated by a switched sum of angle messages
 * φ_c(e_ij, e_im, a_jim) (Eqs. 5–7, 10), and the atom's feature is a switched sum over its bonds.
 * The switch is DPA-3's (Eq. 9) with r_cs/r_c = 5.3/6 for both orders (the paper gives r_cs only for G(1)).
 */
export function dpa3Descriptor(toy: Toy, env: Vec3[], o: EnvOptions): Dpa3Result {
  const rcA = Math.min(o.rcAngle ?? DPA31_3M.rcAngle, o.rc);
  const P = prepareAll(env, o.rc);
  const N = P.r.length;
  const w1 = P.r.map((r) => sDpa3(r, RCS_RATIO * o.rc, o.rc));
  const w2 = P.r.map((r) => (r <= rcA ? sDpa3(r, RCS_RATIO * rcA, rcA) : 0));
  const e = P.r.map((r) => runMlp(toy.bond, [r / o.rc]));
  const E = TOY_DIMS.e;
  // A fixed normalisation, like the paper's N_m^(−α) with N_m an estimated maximum neighbour count:
  // dividing by the actual count would jump whenever an atom crosses the cutoff.
  const norm = 1 / Math.sqrt(TOY_NMAX);
  let angles = 0;
  const v = new Float64Array(E);
  for (let j = 0; j < N; j++) {
    const upd = new Float64Array(E);
    if (w2[j]! > 0)
      for (let m = 0; m < N; m++) {
        if (m === j || w2[m]! === 0) continue;
        angles++;
        const c = P.u[j]![0] * P.u[m]![0] + P.u[j]![1] * P.u[m]![1] + P.u[j]![2] * P.u[m]![2];
        const a = runMlp(toy.angle, [c]);
        const msg = runMlp(toy.msg, [...e[j]!, ...e[m]!, ...a]);
        const w = w2[j]! * w2[m]!;
        for (let k = 0; k < E; k++) upd[k]! += norm * w * msg[k]!;
      }
    const ej = new Float64Array(E);
    for (let k = 0; k < E; k++) ej[k] = e[j]![k]! + upd[k]!;
    const out = runMlp(toy.vertex, ej);
    for (let k = 0; k < E; k++) v[k]! += norm * w1[j]! * out[k]!;
  }
  return { v, bonds: N, angles: angles / 2 };
}

function prepareAll(env: Vec3[], rc: number): { r: number[]; u: Vec3[] } {
  const r: number[] = [];
  const u: Vec3[] = [];
  for (const p of env) {
    const d = Math.hypot(p[0], p[1], p[2]);
    if (d > rc) continue;
    r.push(d);
    u.push([p[0] / d, p[1] / d, p[2] / d]);
  }
  return { r, u };
}

/** ‖a − b‖ / max(‖a‖, ‖b‖): 0 means the descriptor cannot tell the two environments apart. */
export function relDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let k = 0; k < a.length; k++) {
    d += (a[k]! - b[k]!) ** 2;
    na += a[k]! ** 2;
    nb += b[k]! ** 2;
  }
  const m = Math.sqrt(Math.max(na, nb));
  return m === 0 ? 0 : Math.sqrt(d) / m;
}

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------

export function rotate(env: Vec3[], m: number[]): Vec3[] {
  return env.map((p) => [
    m[0]! * p[0] + m[1]! * p[1] + m[2]! * p[2],
    m[3]! * p[0] + m[4]! * p[1] + m[5]! * p[2],
    m[6]! * p[0] + m[7]! * p[1] + m[8]! * p[2],
  ]);
}

export function shuffle<T>(xs: T[], seed: number): T[] {
  const rand = mulberry32(seed);
  const out = xs.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * Neighbours within R of an atom in an ideal close-packed crystal with nearest-neighbour distance d,
 * built from exact lattice vectors: fcc, or hcp with the ideal c/a = √(8/3). The two first shells
 * hold 12 atoms at d and differ only in their angles.
 */
export function closePacked(kind: 'fcc' | 'hcp', d: number, R: number): Vec3[] {
  const out: Vec3[] = [];
  const n = Math.ceil(R / (d * 0.8)) + 1;
  if (kind === 'fcc') {
    const a = d * Math.SQRT2;
    const basis: Vec3[] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
    for (let i = -n; i <= n; i++)
      for (let j = -n; j <= n; j++)
        for (let k = -n; k <= n; k++)
          for (const b of basis) {
            const p: Vec3 = [a * (i + b[0]), a * (j + b[1]), a * (k + b[2])];
            const r = Math.hypot(...p);
            if (r > 1e-9 && r < R) out.push(p);
          }
  } else {
    const c = d * Math.sqrt(8 / 3);
    const a1: Vec3 = [d, 0, 0];
    const a2: Vec3 = [-d / 2, (d * Math.sqrt(3)) / 2, 0];
    // Atom at the origin (A layer); B layer at (1/3, 2/3, 1/2).
    const basis: Array<[number, number, number]> = [[0, 0, 0], [1 / 3, 2 / 3, 0.5]];
    for (let i = -n; i <= n; i++)
      for (let j = -n; j <= n; j++)
        for (let k = -n; k <= n; k++)
          for (const b of basis) {
            const u = i + b[0];
            const v = j + b[1];
            const p: Vec3 = [u * a1[0] + v * a2[0], u * a1[1] + v * a2[1], (k + b[2]) * c];
            const r = Math.hypot(...p);
            if (r > 1e-9 && r < R) out.push(p);
          }
  }
  return out;
}

/** Copper's nearest-neighbour distance for the comparison: a/√2 with a = 3.615 Å (the Atlas's fcc Cu). */
export const CU_D = 3.615 / Math.SQRT2;

/** Three neighbours at distance r on a cone: cos θ = 1/3 to the axis makes |Σ r̂|² = 1. */
export function pyramid(r: number): Vec3[] {
  const c = 1 / 3;
  const s = Math.sqrt(1 - c * c);
  return [0, 1, 2].map((k) => {
    const p = (2 * Math.PI * k) / 3;
    return [r * s * Math.cos(p), r * s * Math.sin(p), r * c] as Vec3;
  });
}

/** Three neighbours in a T: two opposite, one at right angles, so |Σ r̂|² = 1 as for the pyramid. */
export function tShape(r: number): Vec3[] {
  return [
    [r, 0, 0],
    [-r, 0, 0],
    [0, r, 0],
  ];
}

export function tetrahedron(r: number): Vec3[] {
  const k = r / Math.sqrt(3);
  return [
    [k, k, k],
    [k, -k, -k],
    [-k, k, -k],
    [-k, -k, k],
  ];
}

export function square(r: number): Vec3[] {
  return [
    [r, 0, 0],
    [-r, 0, 0],
    [0, r, 0],
    [0, -r, 0],
  ];
}

// ---------------------------------------------------------------------------
// 2. Smooth attention at the cutoff
// ---------------------------------------------------------------------------

export type SoftmaxKind = 'plain' | 'switched' | 'smooth';

/**
 * Attention weights of one row (atom j attending over neighbours k), for neighbour switches w_k:
 *   plain    — softmax(x) over every neighbour inside r_c;
 *   switched — w_k · softmax(x), the fix DPA-2 says does not work;
 *   smooth   — DPA-2's softmax*: w_k · softmax(w_k (x_k + s*) − s*) (Eq. 35), s* = 20.
 */
export function attentionRow(x: number[], w: number[], kind: SoftmaxKind, inside?: boolean[], sStar = 20): number[] {
  // A neighbour is in the list when r < r_c, even where w(r) has rounded to zero just inside r_c.
  const ins = x.map((_, k) => k).filter((k) => (inside ? inside[k]! : w[k]! > 0));
  if (!ins.length) return x.map(() => 0);
  const arg = (k: number) => (kind === 'smooth' ? w[k]! * (x[k]! + sStar) - sStar : x[k]!);
  const m = Math.max(...ins.map(arg));
  const z = ins.reduce((acc, k) => acc + Math.exp(arg(k) - m), 0);
  const out = x.map(() => 0);
  for (const k of ins) {
    const p = Math.exp(arg(k) - m) / z;
    out[k] = kind === 'plain' ? p : w[k]! * p;
  }
  return out;
}

export interface CutoffScan {
  r: number[];
  energy: Record<SoftmaxKind, number[]>;
  /** Energy jump as the moving atom crosses r_c (inside minus outside). */
  jump: Record<SoftmaxKind, number>;
}

/**
 * A toy attention energy around one atom while neighbour 0 moves out along a ray through r_c:
 * E = Σ_j w_j q · Σ_k A_jk V g_k, with g_k = G(s_k) and logits from Q, K as in DPA-1.
 * Other neighbours sit still. Units are arbitrary.
 */
export function cutoffScan(toy: Toy, fixed: Vec3[], dir: Vec3, rs: number[], rcs: number, rc: number): CutoffScan {
  const kinds: SoftmaxKind[] = ['plain', 'switched', 'smooth'];
  const energy = { plain: [] as number[], switched: [] as number[], smooth: [] as number[] };
  const { Q, K, V } = toy.att[0]!;
  const readout = runMlp(toy.vertex, new Float64Array(TOY_DIMS.e).fill(0.3));
  const evalAt = (r: number): Record<SoftmaxKind, number> => {
    const env: Vec3[] = [[dir[0] * r, dir[1] * r, dir[2] * r], ...fixed];
    const d = env.map((p) => Math.hypot(p[0], p[1], p[2]));
    const w = d.map((x) => wPoly(x, rcs, rc));
    const inside = d.map((x) => x < rc);
    const g = d.map((x) => runMlp(toy.embed, [x < rc ? sPoly(x, rcs, rc) : 0]));
    const q = g.map((x) => linear(Q, x));
    const k = g.map((x) => linear(K, x));
    const v = g.map((x) => linear(V, x));
    const out = {} as Record<SoftmaxKind, number>;
    for (const kind of kinds) {
      let E = 0;
      for (let j = 0; j < env.length; j++) {
        if (!inside[j]) continue;
        const x = k.map((kk) => {
          let s = 0;
          for (let c = 0; c < toy.dk; c++) s += q[j]![c]! * kk[c]!;
          return s / Math.sqrt(toy.dk);
        });
        const A = attentionRow(x, w, kind, inside);
        let e = 0;
        A.forEach((a, kk) => {
          for (let c = 0; c < TOY_DIMS.e; c++) e += a * readout[c]! * v[kk]![c]!;
        });
        // The row of atom j is switched by w_j in the smooth forms only.
        E += (kind === 'plain' ? 1 : w[j]!) * e;
      }
      out[kind] = E;
    }
    return out;
  };
  for (const r of rs) {
    const e = evalAt(r);
    for (const kind of kinds) energy[kind].push(e[kind]);
  }
  const inside = evalAt(rc - 1e-9);
  const outside = evalAt(rc + 1e-9);
  const jump = { plain: 0, switched: 0, smooth: 0 };
  for (const kind of kinds) jump[kind] = inside[kind] - outside[kind];
  return { r: rs, energy, jump };
}

// ---------------------------------------------------------------------------
// 3. Energy bias by least squares (DPA-2 Eq. 6; DPA-3 Eq. 3)
// ---------------------------------------------------------------------------

export interface Frame {
  label: string;
  counts: number[];
  energy: number;
}

export interface BiasFit {
  bias: number[];
  /** Rank of the composition matrix: below the number of elements, the bias is not identifiable. */
  rank: number;
  /** Unit vector along which the bias can move without changing any fitted energy (rank deficient only). */
  nullDir: number[] | null;
  rmsPerAtom: number;
}

/** Eigen-decomposition of a symmetric 2×2 or n×n matrix by cyclic Jacobi rotations. */
export function symEig(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p]![q]! ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p]![q]!) < 1e-300) continue;
        const th = (a[q]![q]! - a[p]![p]!) / (2 * a[p]![q]!);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]![k]!;
          const aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k]![p]!;
          const vkq = V[k]![q]!;
          V[k]![p] = c * vkp - s * vkq;
          V[k]![q] = s * vkp + c * vkq;
        }
      }
  }
  return { values: a.map((r, i) => r[i]!), vectors: V };
}

/** Minimum-norm least-squares solution of Σ_z c_mz e(z) = E_m (pseudo-inverse via the normal equations). */
export function fitBias(frames: Frame[]): BiasFit {
  const n = frames[0]!.counts.length;
  const AtA = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const Atb = new Array<number>(n).fill(0);
  for (const f of frames)
    for (let i = 0; i < n; i++) {
      Atb[i]! += f.counts[i]! * f.energy;
      for (let j = 0; j < n; j++) AtA[i]![j]! += f.counts[i]! * f.counts[j]!;
    }
  const { values, vectors } = symEig(AtA);
  const top = Math.max(...values.map(Math.abs));
  const bias = new Array<number>(n).fill(0);
  let rank = 0;
  let nullDir: number[] | null = null;
  values.forEach((lam, k) => {
    const vec = vectors.map((row) => row[k]!);
    if (Math.abs(lam) > 1e-10 * top) {
      rank++;
      const proj = vec.reduce((s, x, i) => s + x * Atb[i]!, 0) / lam;
      vec.forEach((x, i) => (bias[i]! += proj * x));
    } else nullDir = vec;
  });
  let ss = 0;
  for (const f of frames) {
    const pred = f.counts.reduce((s, c, i) => s + c * bias[i]!, 0);
    const atoms = f.counts.reduce((s, c) => s + c, 0);
    ss += ((pred - f.energy) / atoms) ** 2;
  }
  return { bias, rank, nullDir, rmsPerAtom: Math.sqrt(ss / frames.length) };
}

export interface BiasDataset {
  id: string;
  label: string;
  /** Reference energy per element (H, O) of this toy "DFT setup", eV. */
  mu: [number, number];
}

/**
 * Two toy DFT setups for H and O whose energy zeros differ, as they do between codes and
 * pseudopotentials (DPA-3 Sec. 2.5). Values are invented for the demonstration.
 */
export const BIAS_DATASETS: BiasDataset[] = [
  { id: 'a', label: 'setup A', mu: [-13.6, -432.0] },
  { id: 'b', label: 'setup B', mu: [-3.4, -4.9] },
];

export interface FrameSpec {
  label: string;
  counts: [number, number];
  /** A small interaction energy per atom (eV/atom), the part the network would learn. */
  eInt: number;
}

export const FRAME_SPECS: FrameSpec[] = [
  { label: 'H₂O', counts: [2, 1], eInt: -0.33 },
  { label: '(H₂O)₂', counts: [4, 2], eInt: -0.36 },
  { label: '(H₂O)₄', counts: [8, 4], eInt: -0.4 },
  { label: '(H₂O)₈', counts: [16, 8], eInt: -0.42 },
  { label: 'H₂', counts: [2, 0], eInt: -2.25 },
  { label: 'O₂', counts: [0, 2], eInt: -2.6 },
  { label: 'H₂O₂', counts: [2, 2], eInt: -0.28 },
  { label: 'OH', counts: [1, 1], eInt: -0.22 },
];

export function framesFor(ds: BiasDataset, specs: FrameSpec[]): Frame[] {
  return specs.map((f) => {
    const atoms = f.counts[0] + f.counts[1];
    return { label: f.label, counts: f.counts, energy: f.counts[0] * ds.mu[0] + f.counts[1] * ds.mu[1] + atoms * f.eInt };
  });
}
