// Equiformer family (Liao & Smidt 2023; EquiformerV2, Liao et al. 2024; EquiformerV3, Liao et al. 2026)
// as two small, exact toys.
//
// 1. Activations on the sphere. A degree-≤L feature x_lm is a band-limited function on the sphere,
//    f(û) = Σ x_lm Y_lm(û). EquiformerV2's S² activation samples f on a latitude–longitude grid (G),
//    applies SiLU pointwise and projects back (G⁻¹): y = G⁻¹(SiLU(G(x))) (V2 Sec. 3.3). SiLU(f) is not
//    band-limited, so the projection aliases and the result is only approximately equivariant; the
//    error falls as the grid is refined (V2 App. A.4: about 0.2 % for L = 6 on an 18 × 18 grid).
//    EquiformerV3's SwiGLU-S² multiplies two grid functions and gates by a scalar,
//    y = G⁻¹(σ(s) · G(x₁) ⊙ G(x₂)) (V3 Sec. 4.3, Eq. 7). The product of two degree-L functions has
//    degree 2L, and projecting onto Y_lm (degree ≤ L) integrates a degree-3L function, which a finite
//    grid can do exactly. The grid here is e3nn's (used by both): β_j = π(j + ½)/N_β, α_k = 2πk/N_α,
//    with Kostelec–Rockmore quadrature weights in β (e3nn o3/_s2grid.py).
//
//    Exactness condition (derived on the page and checked in the tests): after the α sum, only
//    terms whose orders cancel survive if N_α > 3·m_max; each surviving term is a polynomial in cos β of
//    degree ≤ 3L, which the N_β-point Kostelec–Rockmore rule integrates exactly if N_β ≥ 3L + 1.
//
// 2. Attention near the cutoff. Equiformer's attention weights are a softmax over neighbours of an
//    MLP of invariant edge features, z_ij = aᵀ LeakyReLU(LN(f_ij⁽⁰⁾)) (V2 Sec. 3.2). EquiformerV2 uses a
//    hard radius cutoff and no envelope. Multiplying only the message by an envelope (as eSEN does) still
//    leaves a jump: a neighbour that crosses r_c enters the softmax denominator with a finite weight.
//    V3 puts the envelope inside the softmax, a_ij = env(r_ij) e^{z_ij} / Σ_k env(r_ik) e^{z_ik}, and on
//    the value (V3 Sec. 4.2, Eq. 4), which is smooth.
//
// Weights are random and untrained: magnitudes mean nothing; equivariance errors, exactness thresholds
// and the presence or absence of jumps are exact properties of the operations.

import { mulberry32 } from '../chem/rng';
import { realSH } from '../physics/spherical-harmonics';
import { applyBlocks, esenEnvelope, gaussianBasis, idx, nCoef, wignerBlocks, type Mat3, type Vec3 } from './escn';

// ---------------------------------------------------------------- sphere grid

/**
 * Kostelec–Rockmore quadrature weights for the N_β = 2b colatitudes β_j = π(2j + 1)/(4b), so that
 * Σ_j w_j g(β_j) = ∫₀^π g(β) sin β dβ for every polynomial g in cos β of degree < 2b (e3nn
 * _quadrature_weights, before its normalisation).
 */
export function krWeights(nBeta: number): Float64Array {
  if (nBeta % 2 !== 0 || nBeta < 2) throw new Error('N_beta must be even (as e3nn requires)');
  const b = nBeta / 2;
  const w = new Float64Array(nBeta);
  for (let j = 0; j < nBeta; j++) {
    const t = (Math.PI * (2 * j + 1)) / (4 * b);
    let s = 0;
    for (let k = 0; k < b; k++) s += Math.sin((2 * j + 1) * (2 * k + 1) * (Math.PI / (4 * b))) / (2 * k + 1);
    w[j] = (2 / b) * Math.sin(t) * s;
  }
  return w;
}

export interface S2Grid {
  L: number;
  mmax: number;
  nBeta: number;
  nAlpha: number;
  /** Indices (l, m) of the coefficients kept, |m| ≤ m_max. */
  coefs: number[];
  /** Unit vectors of the grid points, β-major. */
  dirs: Vec3[];
  /** Y[p * K + c]: harmonic of coefficient coefs[c] at point p. */
  Y: Float64Array;
  /** Quadrature weight of each point (sums to 4π). */
  W: Float64Array;
}

/** e3nn's grid: β_j = π(j + ½)/N_β, α_k = 2πk/N_α; polar axis z (the Atlas's harmonics). */
export function makeGrid(L: number, nBeta: number, nAlpha: number, mmax = L): S2Grid {
  const wb = krWeights(nBeta);
  const coefs: number[] = [];
  for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) if (Math.abs(m) <= mmax) coefs.push(idx(l, m));
  const K = coefs.length;
  const dirs: Vec3[] = [];
  const W = new Float64Array(nBeta * nAlpha);
  for (let j = 0; j < nBeta; j++) {
    const b = (Math.PI * (j + 0.5)) / nBeta;
    for (let k = 0; k < nAlpha; k++) {
      const a = (2 * Math.PI * k) / nAlpha;
      dirs.push([Math.sin(b) * Math.cos(a), Math.sin(b) * Math.sin(a), Math.cos(b)]);
      W[j * nAlpha + k] = (wb[j]! * 2 * Math.PI) / nAlpha;
    }
  }
  const Y = new Float64Array(dirs.length * K);
  const lm = coefs.map((c) => {
    const l = Math.floor(Math.sqrt(c));
    return [l, c - l * l - l] as const;
  });
  dirs.forEach((u, p) => lm.forEach(([l, m], c) => (Y[p * K + c] = realSH(l, m, u))));
  return { L, mmax, nBeta, nAlpha, coefs, dirs, Y, W };
}

/** G: coefficients (full (L+1)² layout; orders above m_max ignored) → values at the grid points. */
export function toGrid(g: S2Grid, x: ArrayLike<number>): Float64Array {
  const K = g.coefs.length;
  const out = new Float64Array(g.dirs.length);
  for (let p = 0; p < g.dirs.length; p++) {
    let s = 0;
    for (let c = 0; c < K; c++) s += g.Y[p * K + c]! * x[g.coefs[c]!]!;
    out[p] = s;
  }
  return out;
}

/** G⁻¹: grid values → coefficients by quadrature, x_lm = Σ_p W_p Y_lm(û_p) v_p (orders above m_max are 0). */
export function fromGrid(g: S2Grid, v: ArrayLike<number>): Float64Array {
  const K = g.coefs.length;
  const out = new Float64Array(nCoef(g.L));
  for (let p = 0; p < g.dirs.length; p++) {
    const wv = g.W[p]! * v[p]!;
    for (let c = 0; c < K; c++) out[g.coefs[c]!]! += g.Y[p * K + c]! * wv;
  }
  return out;
}

export const silu = (x: number) => x / (1 + Math.exp(-x));
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export type Activation = 'gate' | 's2' | 'sep-s2' | 'swiglu-s2';

export const ACTIVATIONS: Record<Activation, { name: string; model: string }> = {
  gate: { name: 'gate', model: 'Equiformer, NequIP, eSEN' },
  s2: { name: 'S² activation (SiLU on the grid)', model: 'EquiformerV2 ablation' },
  'sep-s2': { name: 'separable S² activation', model: 'EquiformerV2' },
  'swiglu-s2': { name: 'SwiGLU-S²', model: 'EquiformerV3' },
};

/**
 * One channel's activation. Inputs: x (and x2 for SwiGLU-S²) in the full (L+1)² layout, and an
 * invariant scalar s (the extra degree-0 channel of the separable and SwiGLU forms).
 *  gate:      y_0 = SiLU(x_0); y_l = σ(s) x_l for l > 0                     (exactly equivariant)
 *  s2:        y = G⁻¹ SiLU(G x)
 *  sep-s2:    y_0 = SiLU(s); y_{l>0} = [G⁻¹ SiLU(G x)]_{l>0}               (V2 Sec. 3.3)
 *  swiglu-s2: y = G⁻¹ (σ(s) · G x ⊙ G x2)                                  (V3 Eq. 7)
 */
export function activate(kind: Activation, g: S2Grid, x: Float64Array, x2: Float64Array, s: number): Float64Array {
  if (kind === 'gate') {
    const y = new Float64Array(x.length);
    const gt = sigmoid(s);
    y[0] = silu(x[0]!);
    for (let c = 1; c < x.length; c++) y[c] = gt * x[c]!;
    return y;
  }
  if (kind === 'swiglu-s2') {
    const a = toGrid(g, x);
    const b = toGrid(g, x2);
    const gt = sigmoid(s);
    for (let p = 0; p < a.length; p++) a[p] = gt * a[p]! * b[p]!;
    return fromGrid(g, a);
  }
  const v = toGrid(g, x).map(silu);
  const y = fromGrid(g, v);
  if (kind === 'sep-s2') y[0] = silu(s);
  return y;
}

/** Relative equivariance error ‖act(D x) − D act(x)‖ / ‖act(x)‖ for one rotation R. */
export function activationError(kind: Activation, g: S2Grid, x: Float64Array, x2: Float64Array, s: number, R: Mat3): number {
  const D = wignerBlocks(g.L, R);
  const y = activate(kind, g, x, x2, s);
  const Dy = applyBlocks(D, y, g.L);
  const yR = activate(kind, g, keep(g, applyBlocks(D, x, g.L)), keep(g, applyBlocks(D, x2, g.L)), s);
  let num = 0;
  let den = 0;
  for (let c = 0; c < y.length; c++) {
    num += (yR[c]! - Dy[c]!) ** 2;
    den += Dy[c]! ** 2;
  }
  return Math.sqrt(num / (den || 1));
}

function keep(g: S2Grid, x: Float64Array): Float64Array {
  if (g.mmax >= g.L) return x;
  const y = new Float64Array(x.length);
  for (const c of g.coefs) y[c] = x[c]!;
  return y;
}

/**
 * Smallest even N_β and smallest N_α for which SwiGLU-S² is exact (derived: N_β ≥ 3L + 1,
 * N_α ≥ 3·m_max + 1). The tests check that one fewer point breaks exactness and these do not.
 */
export function swigluExactGrid(L: number, mmax = L): { nBeta: number; nAlpha: number } {
  const nb = 3 * L + 1;
  return { nBeta: nb % 2 === 0 ? nb : nb + 1, nAlpha: 3 * mmax + 1 };
}

/** Default resolution of EquiformerV2's SO3_Grid when none is given (nets/equiformer_v2/so3.py). */
export function eqv2DefaultGrid(L: number, mmax = L): { nBeta: number; nAlpha: number } {
  return { nBeta: 2 * (L + 1), nAlpha: L === mmax ? 2 * (mmax + 1) + 1 : 2 * mmax + 1 };
}

/** A random input feature with power decaying by degree, scaled by amp; orders above m_max zero. */
export function randomCoefs(L: number, seed: number, amp = 1, mmax = L): Float64Array {
  const rand = mulberry32(seed);
  const x = new Float64Array(nCoef(L));
  for (let l = 0; l <= L; l++)
    for (let m = -l; m <= l; m++) {
      const v = (amp * (2 * rand() - 1)) / Math.sqrt(l + 1);
      if (Math.abs(m) <= mmax) x[idx(l, m)] = v;
    }
  return x;
}

/** Evaluate a coefficient vector at a direction. */
export function evalAt(x: ArrayLike<number>, L: number, u: Vec3): number {
  let s = 0;
  for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) s += x[idx(l, m)]! * realSH(l, m, u);
  return s;
}

// ---------------------------------------------------------------- attention near the cutoff

export type CutoffMode = 'hard' | 'message' | 'smooth';

export const CUTOFF_MODES: Record<CutoffMode, { name: string; model: string }> = {
  hard: { name: 'hard cutoff, no envelope', model: 'EquiformerV2' },
  message: { name: 'envelope on the message only', model: 'eSEN-style message' },
  smooth: { name: 'envelope inside the softmax and on the value', model: 'EquiformerV3, Eq. 4' },
};

export interface ToyAttention {
  rcut: number;
  /** H hidden channels × K Gaussians, for f⁽⁰⁾ = W φ(r) + b. */
  W: Float64Array;
  b: Float64Array;
  a: Float64Array;
  /** Value weights: v(r) = c · tanh(V φ(r)). */
  V: Float64Array;
  c: Float64Array;
  H: number;
  K: number;
}

/** EquiformerV2's attention nonlinearity: SmoothLeakyReLU(0.2) (nets/equiformer_v2/activation.py). */
export const smoothLeakyRelu = (x: number, a = 0.2) => ((1 + a) / 2) * x + ((1 - a) / 2) * x * (2 * sigmoid(x) - 1);

export function makeToyAttention(o: { rcut?: number; hidden?: number; seed?: number } = {}): ToyAttention {
  const rcut = o.rcut ?? 6;
  const H = o.hidden ?? 8;
  const K = 10;
  const rand = mulberry32(o.seed ?? 7);
  const r = (s: number) => s * (2 * rand() - 1);
  const W = Float64Array.from({ length: H * K }, () => r(1.6));
  const b = Float64Array.from({ length: H }, () => r(0.5));
  const a = Float64Array.from({ length: H }, () => r(1.2));
  const V = Float64Array.from({ length: H * K }, () => r(1.2));
  const c = Float64Array.from({ length: H }, () => r(1));
  return { rcut, W, b, a, V, c, H, K };
}

/** Invariant edge scalars f⁽⁰⁾(r), times a feature scale (to show what LayerNorm removes). */
export function edgeScalars(t: ToyAttention, r: number, scale = 1): Float64Array {
  const phi = gaussianBasis(r, t.rcut, t.K);
  const f = new Float64Array(t.H);
  for (let h = 0; h < t.H; h++) {
    let s = t.b[h]!;
    for (let k = 0; k < t.K; k++) s += t.W[h * t.K + k]! * phi[k]!;
    f[h] = scale * s;
  }
  return f;
}

/** LayerNorm without affine parameters (ε = 1e-5, as torch). */
export function layerNorm(f: Float64Array, eps = 1e-5): Float64Array {
  const n = f.length;
  const mu = f.reduce((s, v) => s + v, 0) / n;
  const varr = f.reduce((s, v) => s + (v - mu) ** 2, 0) / n;
  return f.map((v) => (v - mu) / Math.sqrt(varr + eps));
}

/** Attention logit z = aᵀ SmoothLeakyReLU(LN(f⁽⁰⁾)) (V2 Sec. 3.2); without LN if ln is false (Equiformer). */
export function logit(t: ToyAttention, r: number, o: { ln?: boolean; scale?: number } = {}): number {
  const f0 = edgeScalars(t, r, o.scale ?? 1);
  const f = o.ln === false ? f0 : layerNorm(f0);
  let z = 0;
  for (let h = 0; h < t.H; h++) z += t.a[h]! * smoothLeakyRelu(f[h]!);
  return z;
}

/** An invariant value carried by the edge (stands in for the l = 0 part of v_ij). */
export function value(t: ToyAttention, r: number): number {
  const phi = gaussianBasis(r, t.rcut, t.K);
  let v = 0;
  for (let h = 0; h < t.H; h++) {
    let s = 0;
    for (let k = 0; k < t.K; k++) s += t.V[h * t.K + k]! * phi[k]!;
    v += t.c[h]! * Math.tanh(s);
  }
  return v;
}

export interface AttentionResult {
  /** Weights a_ij for each neighbour (0 outside the cutoff). */
  weights: number[];
  /** Aggregated output y_i = Σ_j a_ij v_ij (with the envelope on v where the mode puts it). */
  y: number;
  /** Shannon entropy of the weights, in nats. */
  entropy: number;
}

export function attend(t: ToyAttention, dists: number[], mode: CutoffMode, o: { ln?: boolean; scale?: number } = {}): AttentionResult {
  const z = dists.map((r) => (r < t.rcut ? logit(t, r, o) : -Infinity));
  const env = dists.map((r) => (r < t.rcut ? (mode === 'smooth' ? esenEnvelope(r / t.rcut) : 1) : 0));
  const zmax = Math.max(...z.filter(Number.isFinite), 0);
  const e = z.map((v, k) => (Number.isFinite(v) ? env[k]! * Math.exp(v - zmax) : 0));
  const sum = e.reduce((s, v) => s + v, 0);
  const weights = e.map((v) => (sum > 0 ? v / sum : 0));
  let y = 0;
  dists.forEach((r, k) => {
    if (r >= t.rcut) return;
    const venv = mode === 'hard' ? 1 : esenEnvelope(r / t.rcut);
    y += weights[k]! * venv * value(t, r);
  });
  const entropy = -weights.reduce((s, w) => s + (w > 0 ? w * Math.log(w) : 0), 0);
  return { weights, y, entropy };
}

/** Jump of y when the moving neighbour (index k) crosses the cutoff: y(r_c − δ) − y(r_c + δ). */
export function cutoffJump(t: ToyAttention, dists: number[], k: number, mode: CutoffMode, o: { ln?: boolean; scale?: number } = {}, delta = 1e-7): number {
  const inside = dists.slice();
  const outside = dists.slice();
  inside[k] = t.rcut - delta;
  outside[k] = t.rcut + delta;
  return attend(t, inside, mode, o).y - attend(t, outside, mode, o).y;
}

/**
 * Neighbours for the attention toy: n fixed points inside the cutoff (deterministic) and one mover
 * along a fixed direction at distance r. Returns positions; the centre atom is at the origin.
 */
export function attentionScene(r: number, n = 6, seed = 11): { positions: Vec3[]; mover: number } {
  const rand = mulberry32(seed);
  const positions: Vec3[] = [];
  for (let k = 0; k < n; k++) {
    const z = 2 * rand() - 1;
    const ph = 2 * Math.PI * rand();
    const s = Math.sqrt(1 - z * z);
    const d = 2.2 + 3.2 * rand();
    positions.push([d * s * Math.cos(ph), d * s * Math.sin(ph), d * z]);
  }
  const u: Vec3 = [0.8, -0.36, 0.48];
  const nu = Math.hypot(...u);
  positions.push(u.map((c) => (c * r) / nu) as Vec3);
  return { positions, mover: n };
}
