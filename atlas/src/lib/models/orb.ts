// Orb (Neumann et al. 2024; Rhodes et al. 2025, Orb-v3) as small, exact toys.
//
// Orb is a Graph Network Simulator: edges carry features built from bond vectors, nodes from
// species; each message-passing step updates edges from [edge, sender, receiver], gates them with
// a sigmoid times a smooth distance envelope, sums them into the nodes, and adds residuals.
// It is not equivariant: Orb-v3 feeds the Cartesian components of real spherical harmonics
// (L_max = 3) into ordinary MLPs, so rotational invariance must be learned.
//
// Two things on the Orb page are computed here:
//  1. The neighbour limit. Orb keeps at most k neighbours per atom (k = 20 for the "-20"
//     checkpoints, 120 for "-inf"), the k nearest inside 6 Å. In a crystal the k-th place usually
//     falls inside a shell of symmetry-equivalent atoms, so the cut is a tie broken by ordering,
//     and the energy is discontinuous when an atom crosses the k-th distance.
//  2. Equigrad. Δ_rot = ∂E(r·exp(G − Gᵀ))/∂G at G = 0 (Orb-v3, Eq. 1) is zero for an invariant
//     energy. For a conservative model it equals the antisymmetric part of Σ_i r_i ⊗ ∂E/∂r_i,
//     i.e. minus the net torque the model exerts. Penalising ‖Δ_rot‖² while fitting the last,
//     linear layer of a frozen toy GNS is a closed-form ridge problem.
//
// Weights are random and untrained (except the fitted readout in part 2): energies are in
// arbitrary units, but continuity, jumps and invariance are exact properties.

import { mulberry32, randomRotation } from '../chem/rng';
import { phi2, phi3 } from '../physics/stillinger-weber';

export type Vec3 = [number, number, number];
export type Mat3 = number[]; // row-major 3×3

/** Orb's cutoff radius and polynomial order (orb_models/common/models/nn_util.py, get_cutoff). */
export const ORB_RMAX = 6.0;
export const ORB_ENVELOPE_P = 4;
/** max_num_neighbors of the "-20" and "-inf" checkpoints (orb_models/forcefield/pretrained.py). */
export const ORB_K20 = 20;
export const ORB_KINF = 120;
/** Orb-v3 edge basis: 8 Bessel functions and real spherical harmonics up to L_max = 3. */
export const ORB_NBESSEL = 8;
export const ORB_LMAX = 3;

const norm = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/**
 * The smooth envelope that multiplies Orb-v3's edge embeddings and attention gates:
 * 1 − (p+1)(p+2)/2 x^p + p(p+2) x^{p+1} − p(p+1)/2 x^{p+2}, x = r / r_max, zero beyond r_max.
 * Value, slope and curvature vanish at r_max.
 */
export function orbEnvelope(r: number, rmax = ORB_RMAX, p = ORB_ENVELOPE_P): number {
  if (r >= rmax) return 0;
  const x = r / rmax;
  return 1 - ((p + 1) * (p + 2) / 2) * x ** p + p * (p + 2) * x ** (p + 1) - (p * (p + 1) / 2) * x ** (p + 2);
}

/** Orb's Bessel radial basis √(2/r_max) sin(nπ r / r_max) / r, n = 1 … N. */
export function besselBasis(r: number, rmax = ORB_RMAX, n = ORB_NBESSEL): number[] {
  const pre = Math.sqrt(2 / rmax);
  return Array.from({ length: n }, (_, k) => (pre * Math.sin(((k + 1) * Math.PI * r) / rmax)) / r);
}

const C0 = 0.5 / Math.sqrt(Math.PI);
const C1 = Math.sqrt(3 / (4 * Math.PI));
const C2a = 0.5 * Math.sqrt(15 / Math.PI);
const C2b = 0.25 * Math.sqrt(5 / Math.PI);
const C2c = 0.25 * Math.sqrt(15 / Math.PI);
const C3a = 0.25 * Math.sqrt(35 / (2 * Math.PI));
const C3b = 0.5 * Math.sqrt(105 / Math.PI);
const C3c = 0.25 * Math.sqrt(21 / (2 * Math.PI));
const C3d = 0.25 * Math.sqrt(7 / Math.PI);
const C3e = 0.25 * Math.sqrt(105 / Math.PI);

/**
 * All 16 real spherical-harmonic components (l ≤ 3) of a direction, as explicit polynomials in
 * the same convention and order as shBlock (l = 0 … 3, m = −l … l; no Condon–Shortley phase).
 */
export function shAll(v: Vec3): number[] {
  const r = norm(v) || 1;
  const x = v[0] / r;
  const y = v[1] / r;
  const z = v[2] / r;
  const z2 = z * z;
  return [
    C0,
    C1 * y, C1 * z, C1 * x,
    C2a * x * y, C2a * y * z, C2b * (3 * z2 - 1), C2a * x * z, C2c * (x * x - y * y),
    C3a * y * (3 * x * x - y * y), C3b * x * y * z, C3c * y * (5 * z2 - 1), C3d * z * (5 * z2 - 3),
    C3c * x * (5 * z2 - 1), C3e * z * (x * x - y * y), C3a * x * (x * x - 3 * y * y),
  ];
}

export type EdgeMode = 'orb' | 'distance';

/**
 * Orb-v3's initial edge feature: envelope(r) · (Bessel ⊗ SH), 8 × 16 = 128 numbers.
 * 'distance' keeps only the l = 0 column (a rotation-invariant control).
 */
export function edgeFeature(v: Vec3, mode: EdgeMode = 'orb'): Float64Array {
  const r = norm(v);
  const env = orbEnvelope(r);
  const sh = mode === 'orb' ? shAll(v) : [1 / Math.sqrt(4 * Math.PI)];
  const ns = sh.length;
  const out = new Float64Array(ORB_NBESSEL * ns);
  const pre = (env * Math.sqrt(2 / ORB_RMAX)) / r;
  for (let i = 0; i < ORB_NBESSEL; i++) {
    const b = pre * Math.sin(((i + 1) * Math.PI * r) / ORB_RMAX);
    for (let j = 0; j < ns; j++) out[i * ns + j] = b * sh[j]!;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Neighbour selection
// ---------------------------------------------------------------------------

/**
 * Orb's neighbour list for one atom: inside r_max, sorted by distance, the first k kept.
 * Ties are broken by input order (a stable sort), as an implementation must break them somehow.
 * k = null keeps everything inside r_max.
 */
export function selectNeighbours(dists: number[], k: number | null, rmax = ORB_RMAX): number[] {
  const inside = dists.map((d, i) => [d, i] as const).filter(([d]) => d < rmax);
  inside.sort((a, b) => a[0] - b[0]);
  return (k === null ? inside : inside.slice(0, k)).map(([, i]) => i);
}

export interface LimitCut {
  /** Neighbours inside r_max. */
  inside: number;
  /** Distance of the k-th neighbour (Infinity if fewer than k). */
  rk: number;
  /** Atoms at the k-th distance (within tol), and how many of them the limit keeps. */
  tied: number;
  tiedKept: number;
  /** Neighbours inside r_max that the limit drops. */
  dropped: number;
}

/** Where a k-neighbour limit cuts an atom's distance list, and whether the cut splits a shell. */
export function limitCut(dists: number[], k: number, rmax = ORB_RMAX, tol = 1e-6): LimitCut {
  const sorted = dists.filter((d) => d < rmax).sort((a, b) => a - b);
  if (sorted.length <= k) return { inside: sorted.length, rk: Infinity, tied: 0, tiedKept: 0, dropped: 0 };
  const rk = sorted[k - 1]!;
  const tied = sorted.filter((d) => Math.abs(d - rk) < tol).length;
  const tiedKept = sorted.slice(0, k).filter((d) => Math.abs(d - rk) < tol).length;
  return { inside: sorted.length, rk, tied, tiedKept, dropped: sorted.length - k };
}

/** Shells of an atom's distance list: [distance, multiplicity] within r_max. */
export function shells(dists: number[], rmax = ORB_RMAX, tol = 1e-6): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (const d of dists.filter((x) => x < rmax).sort((a, b) => a - b)) {
    const last = out.at(-1);
    if (last && Math.abs(last[0] - d) < tol) last[1]++;
    else out.push([d, 1]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// A toy one-step Orb GNS acting on one atom's environment
// ---------------------------------------------------------------------------

interface Dense {
  w: Float64Array;
  b: Float64Array;
  nIn: number;
  nOut: number;
}

function dense(rand: () => number, nIn: number, nOut: number, gain = 1): Dense {
  const s = gain * Math.sqrt(6 / (nIn + nOut));
  return {
    w: new Float64Array(nIn * nOut).map(() => (2 * rand() - 1) * s),
    b: new Float64Array(nOut).map(() => (2 * rand() - 1) * 0.1),
    nIn,
    nOut,
  };
}

export const silu = (x: number) => x / (1 + Math.exp(-x));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function apply(d: Dense, x: ArrayLike<number>, act: boolean): Float64Array {
  const y = new Float64Array(d.nOut);
  for (let o = 0; o < d.nOut; o++) {
    let s = d.b[o]!;
    const row = o * d.nIn;
    for (let i = 0; i < d.nIn; i++) s += d.w[row + i]! * x[i]!;
    y[o] = act ? silu(s) : s;
  }
  return y;
}

function concat(...xs: ArrayLike<number>[]): Float64Array {
  const out = new Float64Array(xs.reduce((a, x) => a + x.length, 0));
  let k = 0;
  for (const x of xs) for (let i = 0; i < x.length; i++) out[k++] = x[i]!;
  return out;
}

export interface ToyOrb {
  mode: EdgeMode;
  d: number;
  edgeEnc: Dense;
  node0: Float64Array;
  gate: Dense;
  edgeMlp: Dense;
  nodeMlp: Dense;
  readout: Dense;
}

/** A one-step GNS with Orb-v3's edge features, sigmoid × envelope gates and residual updates. */
export function makeToyOrb(o: { mode?: EdgeMode; d?: number; seed?: number } = {}): ToyOrb {
  const mode = o.mode ?? 'orb';
  const d = o.d ?? 24;
  const rand = mulberry32(o.seed ?? 11);
  const nEdge = ORB_NBESSEL * (mode === 'orb' ? (ORB_LMAX + 1) ** 2 : 1);
  return {
    mode,
    d,
    edgeEnc: dense(rand, nEdge, d, 3),
    node0: new Float64Array(d).map(() => 2 * rand() - 1),
    gate: dense(rand, d, 1),
    edgeMlp: dense(rand, 3 * d, d),
    nodeMlp: dense(rand, 2 * d, d),
    readout: dense(rand, d, 1),
  };
}

export interface OrbResult {
  energy: number;
  /** Indices of the neighbours that became edges, nearest first. */
  edges: number[];
  /** sigmoid(gate) × envelope for each edge, same order. */
  gates: number[];
  /** The centre node's state after the update (the input of the readout). */
  node: Float64Array;
  /** The aggregated message Σ_j ψ_j ε_j / 10 that entered the update. */
  message: Float64Array;
}

/**
 * The centre atom's energy contribution after one message-passing step (all atoms of one
 * species). For edge j: e_j = MLP(edge feature), ε_j = MLP([e_j, v_i, v_j]),
 * ψ_j = sigmoid(w·e_j) f_env(r_j); v_i ← v_i + MLP([v_i, Σ_j ψ_j ε_j]); E_i = readout(v_i).
 */
export function orbForward(m: ToyOrb, vectors: Vec3[], k: number | null): OrbResult {
  const edges = selectNeighbours(vectors.map(norm), k);
  const d = m.d;
  const msg = new Float64Array(d);
  const e = new Float64Array(d);
  const gates: number[] = [];
  // All atoms share one species, so the [v_i, v_j] part of the edge MLP is a constant bias.
  const epsBias = edgeBias(m);
  const W = m.edgeMlp.w;
  const Ew = m.edgeEnc.w;
  const Eb = m.edgeEnc.b;
  const nIn = m.edgeEnc.nIn;
  for (const j of edges) {
    const v = vectors[j]!;
    const f = edgeFeature(v, m.mode);
    for (let o = 0; o < d; o++) {
      let s = Eb[o]!;
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) s += Ew[row + i]! * f[i]!;
      e[o] = silu(s);
    }
    let g = m.gate.b[0]!;
    for (let i = 0; i < d; i++) g += m.gate.w[i]! * e[i]!;
    const psi = sigmoid(g) * orbEnvelope(norm(v));
    gates.push(psi);
    for (let o = 0; o < d; o++) {
      let s = epsBias[o]!;
      const row = o * 3 * d;
      for (let i = 0; i < d; i++) s += W[row + i]! * e[i]!;
      msg[o] = msg[o]! + psi * silu(s);
    }
  }
  // Sum aggregation; scaled by a typical neighbour count so the toy's activations stay O(1).
  for (let c = 0; c < d; c++) msg[c] = msg[c]! / 10;
  const upd = apply(m.nodeMlp, concat(m.node0, msg), true);
  const node = new Float64Array(d).map((_, c) => m.node0[c]! + upd[c]!);
  return { energy: apply(m.readout, node, false)[0]!, edges, gates, node, message: msg };
}

const biasCache = new WeakMap<ToyOrb, Float64Array>();
function edgeBias(m: ToyOrb): Float64Array {
  let b = biasCache.get(m);
  if (!b) {
    const d = m.d;
    b = new Float64Array(d).map((_, o) => {
      let s = m.edgeMlp.b[o]!;
      for (let i = 0; i < d; i++) s += m.edgeMlp.w[o * 3 * d + d + i]! * m.node0[i]! + m.edgeMlp.w[o * 3 * d + 2 * d + i]! * m.node0[i]!;
      return s;
    });
    biasCache.set(m, b);
  }
  return b;
}

/** −∂E/∂r_j for every neighbour (central differences); the centre's force is minus their sum. */
export function orbForces(f: (v: Vec3[]) => number, vectors: Vec3[], h = 1e-5): Vec3[] {
  return vectors.map((_, j) => {
    const F: Vec3 = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const p = vectors.map((v) => [...v] as Vec3);
      const q = vectors.map((v) => [...v] as Vec3);
      p[j]![a] = p[j]![a]! + h;
      q[j]![a] = q[j]![a]! - h;
      F[a] = -(f(p) - f(q)) / (2 * h);
    }
    return F;
  });
}

/** Move neighbour j radially to distance r. */
export function withRadial(vectors: Vec3[], j: number, r: number): Vec3[] {
  const v = vectors[j]!;
  const s = r / norm(v);
  return vectors.map((u, i) => (i === j ? ([v[0] * s, v[1] * s, v[2] * s] as Vec3) : u));
}

// ---------------------------------------------------------------------------
// Rotations and Equigrad
// ---------------------------------------------------------------------------

/** exp(A) for antisymmetric 3×3 A (Rodrigues), row-major. */
export function expAntisym(A: Mat3): Mat3 {
  // A = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]] ↔ ω = (wx, wy, wz)
  const w: Vec3 = [A[7]!, A[2]!, A[3]!];
  const th = norm(w);
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (th < 1e-14) return I.map((x, i) => x + A[i]!);
  const A2 = Array.from({ length: 9 }, (_, i) => {
    const r = Math.floor(i / 3);
    const c = i % 3;
    return A[3 * r]! * A[c]! + A[3 * r + 1]! * A[3 + c]! + A[3 * r + 2]! * A[6 + c]!;
  });
  const s = Math.sin(th) / th;
  const c = (1 - Math.cos(th)) / (th * th);
  return I.map((x, i) => x + s * A[i]! + c * A2[i]!);
}

/** Row-vector convention of Orb's code: r' = r R. */
export const rowTimes = (v: Vec3, R: Mat3): Vec3 => [
  v[0] * R[0]! + v[1] * R[3]! + v[2] * R[6]!,
  v[0] * R[1]! + v[1] * R[4]! + v[2] * R[7]!,
  v[0] * R[2]! + v[1] * R[5]! + v[2] * R[8]!,
];

export const rotateAll = (vectors: Vec3[], R: Mat3): Vec3[] => vectors.map((v) => rowTimes(v, R));

/** Δ_rot = ∂E(r·exp(G − Gᵀ))/∂G at G = 0, by central differences on the nine entries of G. */
export function deltaRot(f: (v: Vec3[]) => number, vectors: Vec3[], h = 1e-5): Mat3 {
  return Array.from({ length: 9 }, (_, i) => {
    const r = Math.floor(i / 3);
    const c = i % 3;
    const E = (t: number) => {
      const A = new Array<number>(9).fill(0);
      A[3 * r + c]! += t;
      A[3 * c + r]! -= t;
      return f(rotateAll(vectors, expAntisym(A)));
    };
    return (E(h) - E(-h)) / (2 * h);
  });
}

/** Net torque Σ_j r_j × F_j about the centre atom (the centre itself sits at the origin). */
export function netTorque(vectors: Vec3[], forces: Vec3[]): Vec3 {
  return vectors.reduce<Vec3>((t, v, j) => {
    const c = cross(v, forces[j]!);
    return [t[0] + c[0], t[1] + c[1], t[2] + c[2]];
  }, [0, 0, 0]);
}

/** Frobenius norm. */
export const frob = (M: Mat3) => Math.hypot(...M);

/** Rotation by θ about a (unit) axis, as a row-vector matrix R with r' = r R turning r by +θ. */
export function axisRot(axis: Vec3, th: number): Mat3 {
  const n = norm(axis);
  const [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n];
  // Column-vector matrix M (Rodrigues); the row-vector form is Mᵀ.
  const c = Math.cos(th);
  const s = Math.sin(th);
  const t = 1 - c;
  const M = [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
  return [M[0]!, M[3]!, M[6]!, M[1]!, M[4]!, M[7]!, M[2]!, M[5]!, M[8]!];
}

/** Uniform random rotation in the row-vector convention. */
export const randomRowRotation = (rand: () => number): Mat3 => {
  const M = randomRotation(rand);
  return [M[0]!, M[3]!, M[6]!, M[1]!, M[4]!, M[7]!, M[2]!, M[5]!, M[8]!];
};

// ---------------------------------------------------------------------------
// Fitting the readout with and without Equigrad
// ---------------------------------------------------------------------------

/** The Stillinger–Weber site energy of the centre atom: ½ Σ_j φ2 + Σ_{j<k} φ3 (eV). Invariant. */
export function swSite(vectors: Vec3[]): number {
  const rs = vectors.map(norm);
  let e = 0;
  for (let j = 0; j < vectors.length; j++) {
    e += 0.5 * phi2(rs[j]!);
    for (let k = j + 1; k < vectors.length; k++) {
      const a = vectors[j]!;
      const b = vectors[k]!;
      e += phi3(rs[j]!, rs[k]!, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (rs[j]! * rs[k]!));
    }
  }
  return e;
}

/** Features of the frozen toy: the aggregated message, the updated centre node and a constant. */
export function features(m: ToyOrb, vectors: Vec3[]): Float64Array {
  const r = orbForward(m, vectors, null);
  return concat(r.message, r.node, [1]);
}

/**
 * ∂φ/∂G_ab at G = 0 for the three independent antisymmetric generators (a, b) = (0,1), (0,2),
 * (1,2). Since Δ_rot is antisymmetric, ‖Δ_rot‖²_F = 2 Σ over these three of (w·∂φ/∂G_ab)².
 */
export function featureRotJacobian(m: ToyOrb, vectors: Vec3[], h = 1e-5): Float64Array[] {
  return ([[0, 1], [0, 2], [1, 2]] as const).map(([r, c]) => {
    const at = (t: number) => {
      const A = new Array<number>(9).fill(0);
      A[3 * r + c] = t;
      A[3 * c + r] = -t;
      return features(m, rotateAll(vectors, expAntisym(A)));
    };
    // ∂/∂G_rc of the antisymmetrised input: G_rc enters A with +1 at (r,c) and −1 at (c,r).
    const p = at(h);
    const q = at(-h);
    return p.map((x, i) => (x - q[i]!) / (2 * h));
  });
}

export interface Sample {
  phi: Float64Array;
  jac: Float64Array[];
  y: number;
}

export function makeSample(m: ToyOrb, vectors: Vec3[]): Sample {
  return { phi: features(m, vectors), jac: featureRotJacobian(m, vectors), y: swSite(vectors) };
}

/** Solve S x = b for symmetric positive-definite S (Cholesky). */
export function choleskySolve(S: Float64Array, b: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = S[i * n + j]!;
      for (let k = 0; k < j; k++) s -= L[i * n + k]! * L[j * n + k]!;
      if (i === j) {
        if (s <= 0) throw new Error('matrix is not positive definite');
        L[i * n + i] = Math.sqrt(s);
      } else L[i * n + j] = s / L[j * n + j]!;
    }
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= L[i * n + k]! * z[k]!;
    z[i] = s / L[i * n + i]!;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]!;
    for (let k = i + 1; k < n; k++) s -= L[k * n + i]! * x[k]!;
    x[i] = s / L[i * n + i]!;
  }
  return x;
}

/**
 * The readout w minimising  (1/N) Σ_n [(w·φ_n − y_n)² + λ ‖Δ_rot,n(w)‖²_F] + μ ‖w̃‖²,
 * where Δ_rot,n(w) is linear in w and w̃ are the weights on standardised features (zero mean,
 * unit variance over the training set; the constant feature is not penalised). Closed form:
 * (Φ̃ᵀΦ̃ + 2λ Σ_n Σ_g J̃_ng J̃_ngᵀ + N μ I′) w̃ = Φ̃ᵀ y, then mapped back so that E = w·φ.
 */
export function fitReadout(samples: Sample[], lambda: number, mu = 1e-4): Float64Array {
  const n = samples[0]!.phi.length;
  const N = samples.length;
  const mean = new Float64Array(n);
  const sd = new Float64Array(n);
  for (const s of samples) for (let i = 0; i < n; i++) mean[i] = mean[i]! + s.phi[i]! / N;
  for (const s of samples) for (let i = 0; i < n; i++) sd[i] = sd[i]! + (s.phi[i]! - mean[i]!) ** 2 / N;
  // Constant features (the bias) keep their raw value and carry the intercept.
  const isConst = Array.from(sd, (v) => v < 1e-20);
  for (let i = 0; i < n; i++) {
    sd[i] = isConst[i] ? 1 : Math.sqrt(sd[i]!);
    if (isConst[i]) mean[i] = 0;
  }
  const S = new Float64Array(n * n);
  const b = new Float64Array(n);
  const x = new Float64Array(n);
  for (const s of samples) {
    for (let i = 0; i < n; i++) x[i] = (s.phi[i]! - mean[i]!) / sd[i]!;
    for (let i = 0; i < n; i++) {
      b[i] = b[i]! + x[i]! * s.y;
      for (let j = 0; j < n; j++) S[i * n + j] = S[i * n + j]! + x[i]! * x[j]!;
    }
    if (lambda > 0)
      for (const g of s.jac)
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i * n + j] = S[i * n + j]! + (2 * lambda * g[i]! * g[j]!) / (sd[i]! * sd[j]!);
  }
  for (let i = 0; i < n; i++) S[i * n + i] = S[i * n + i]! + N * (isConst[i] ? 1e-12 : mu);
  const wt = choleskySolve(S, b, n);
  const w = new Float64Array(n).map((_, i) => wt[i]! / sd[i]!);
  // Fold the feature means into the constant feature(s).
  const shift = Array.from(wt).reduce((a, v, i) => a + (v * mean[i]!) / sd[i]!, 0);
  const c = isConst.findIndex(Boolean);
  if (c >= 0) w[c] = w[c]! - shift / samples[0]!.phi[c]!;
  return w;
}

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

/** ‖Δ_rot‖_F of the fitted model at one sample. */
export const fittedDeltaRot = (w: Float64Array, s: Sample) => Math.sqrt(2 * s.jac.reduce((a, g) => a + dot(w, g) ** 2, 0));

/** A rattled copy of an environment: every neighbour displaced by Gaussian noise of width σ. */
export function rattle(vectors: Vec3[], sigma: number, rand: () => number): Vec3[] {
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  return vectors.map((v) => [v[0] + sigma * gauss(), v[1] + sigma * gauss(), v[2] + sigma * gauss()] as Vec3);
}

export interface EquigradExperiment {
  train: { fixed: Sample[]; rotated: Sample[] };
  test: Array<{ y: number; vectors: Vec3[]; orientations: Sample[] }>;
}

/**
 * Training environments rattled around one crystal orientation ('fixed'), the same environments
 * each copied in `copies` random orientations ('rotated', i.e. augmentation), and test
 * environments each seen in `testOrientations` random orientations.
 */
export function equigradExperiment(
  m: ToyOrb,
  base: Vec3[],
  o: { nTrain?: number; copies?: number; nTest?: number; testOrientations?: number; sigma?: number; seed?: number } = {},
): EquigradExperiment {
  const rand = mulberry32(o.seed ?? 5);
  const sigma = o.sigma ?? 0.12;
  const envs = Array.from({ length: o.nTrain ?? 120 }, () => rattle(base, sigma, rand));
  const fixed = envs.map((v) => makeSample(m, v));
  const rotated = envs.flatMap((v) => Array.from({ length: o.copies ?? 3 }, () => makeSample(m, rotateAll(v, randomRowRotation(rand)))));
  const test = Array.from({ length: o.nTest ?? 20 }, () => {
    const v = rattle(base, sigma, rand);
    const orientations = Array.from({ length: o.testOrientations ?? 8 }, () => makeSample(m, rotateAll(v, randomRowRotation(rand))));
    return { y: orientations[0]!.y, vectors: v, orientations };
  });
  return { train: { fixed, rotated }, test };
}

export interface FitScore {
  /** RMSE against the reference over every test orientation (eV). */
  rmse: number;
  /** Mean over test environments of the std of the prediction over orientations (eV). */
  spread: number;
  /** Mean ‖Δ_rot‖_F over test orientations (eV). */
  deltaRot: number;
  /** RMSE on the training set (eV). */
  trainRmse: number;
}

export function scoreFit(w: Float64Array, train: Sample[], test: EquigradExperiment['test']): FitScore {
  let se = 0;
  let count = 0;
  let spread = 0;
  let dr = 0;
  for (const t of test) {
    const ys = t.orientations.map((s) => dot(w, s.phi));
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    spread += Math.sqrt(ys.reduce((a, y) => a + (y - mean) ** 2, 0) / ys.length);
    ys.forEach((y) => (se += (y - t.y) ** 2));
    count += ys.length;
    t.orientations.forEach((s) => (dr += fittedDeltaRot(w, s)));
  }
  const trainRmse = Math.sqrt(train.reduce((a, s) => a + (dot(w, s.phi) - s.y) ** 2, 0) / train.length);
  return { rmse: Math.sqrt(se / count), spread: spread / test.length, deltaRot: dr / count, trainRmse };
}

// ---------------------------------------------------------------------------
// The neighbour-limit experiment shown on the page
// ---------------------------------------------------------------------------

/** A neighbour tied at the k-th distance that the limit drops (the atom moved in Fig. 1). */
export function droppedTiedNeighbour(vectors: Vec3[], k: number, tol = 1e-6): number {
  const dists = vectors.map(norm);
  const kept = new Set(selectNeighbours(dists, k));
  const { rk } = limitCut(dists, k);
  const j = dists.findIndex((d, i) => Math.abs(d - rk) < tol && !kept.has(i));
  if (j < 0) throw new Error('the limit does not split a shell here');
  return j;
}

export interface LimitCurve {
  r: number[];
  energy: number[];
  /** Radial force on the moved atom, −∂E/∂r (one-sided differences on each side of the crossing). */
  force: number[];
}

/** The centre's energy and the radial force on neighbour j as j moves radially over rs. */
export function limitCurve(m: ToyOrb, vectors: Vec3[], j: number, k: number | null, rs: number[], h = 1e-5): LimitCurve {
  const E = (r: number) => orbForward(m, withRadial(vectors, j, r), k).energy;
  const energy = rs.map(E);
  const force = rs.map((r) => -(E(r + h) - E(r - h)) / (2 * h));
  return { r: rs, energy, force };
}
