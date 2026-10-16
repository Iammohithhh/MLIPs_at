// eSEN's edge message (Fu et al. 2025) as a small, exact toy: the eSCN SO(2) convolution
// (Passaro & Zitnick 2023) on spherical-harmonic node features.
//
// Node features are coefficient vectors x_{lm}, l ≤ L, one per channel, i.e. functions on the
// sphere f(û) = Σ x_lm Y_lm(û). An edge message is built in four steps:
//  1. rotate the source and target features into the edge frame, where the bond lies on the polar
//     axis of the harmonics: x' = D(Q) x. eSCN and fairchem use e3nn's harmonics, whose polar axis is
//     y, so they align the bond with +y (init_edge_rot_euler_angles) and draw the roll γ about it at
//     random on every call. The Atlas's harmonics have their polar axis along z, so here Q r̂ = ẑ;
//     the construction is the same;
//  2. in that frame Y_lm(bond) vanishes unless m = 0, so the SO(3) tensor product with the edge's
//     spherical harmonics reduces to an SO(2) convolution: m = 0 coefficients mix freely across l;
//     each pair (m, −m) mixes through a 2×2 block [[A, −B], [B, A]] of radial weights
//     (eSCN Eq. 14). Orders |m| > M_max are dropped;
//  3. a gated nonlinearity between two such convolutions (as in eSEN's edgewise block);
//  4. rotate back with D(Q)ᵀ and multiply by the polynomial envelope of r / r_cut.
//
// A rotation about the bond acts on each (m, −m) pair as a 2-D rotation by mγ, which commutes with
// [[A, −B], [B, A]], so the random roll changes nothing and the message is exactly equivariant.
// The "dense" control replaces the SO(2) blocks by unrestricted matrices in the edge frame: it is
// no longer roll-equivariant, so its output depends on γ and it is not equivariant.
//
// Weights are random and untrained: magnitudes mean nothing; equivariance, roll independence and
// the term counts are exact.

import { mulberry32 } from '../chem/rng';
import { realSH } from '../physics/spherical-harmonics';

export type Vec3 = [number, number, number];
/** Row-major 3×3 acting on column vectors: v′ = R v. */
export type Mat3 = number[];

/** Cutoff, envelope exponent and radial basis of eSEN/UMA (fairchem escn_md.py; eSEN Table 6). */
export const ESEN_RCUT = 6.0;
export const ESEN_ENVELOPE_P = 5;
export const ESEN_NGAUSS = 10;

export const nCoef = (L: number) => (L + 1) ** 2;
export const idx = (l: number, m: number) => l * l + l + m;

const norm = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const scale = (v: Vec3, s: number): Vec3 => [v[0] * s, v[1] * s, v[2] * s];

export const matVec = (R: Mat3, v: Vec3): Vec3 => [
  R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2],
  R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2],
  R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2],
];
export function matMul(A: Mat3, B: Mat3): Mat3 {
  const C = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[3 * i + j]! += A[3 * i + k]! * B[3 * k + j]!;
  return C;
}
export const transpose = (A: Mat3): Mat3 => [A[0]!, A[3]!, A[6]!, A[1]!, A[4]!, A[7]!, A[2]!, A[5]!, A[8]!];

/** Rotation by angle th about a unit axis (Rodrigues). */
export function axisRotation(axis: Vec3, th: number): Mat3 {
  const [x, y, z] = scale(axis, 1 / norm(axis));
  const c = Math.cos(th);
  const s = Math.sin(th);
  const t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}

/** The polynomial envelope 1 − (p+1)(p+2)/2 x^p + p(p+2) x^{p+1} − p(p+1)/2 x^{p+2}, zero for x ≥ 1. */
export function esenEnvelope(x: number, p = ESEN_ENVELOPE_P): number {
  if (x >= 1) return 0;
  const a = (-(p + 1) * (p + 2)) / 2;
  const b = p * (p + 2);
  const c = (-p * (p + 1)) / 2;
  return 1 + x ** p * (a + x * (b + c * x));
}

/** Gaussian smearing on [0, r_cut] with width equal to the centre spacing (fairchem GaussianSmearing). */
export function gaussianBasis(r: number, rcut = ESEN_RCUT, n = ESEN_NGAUSS): Float64Array {
  const dx = rcut / (n - 1);
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) out[k] = Math.exp((-0.5 * (r - k * dx) ** 2) / dx ** 2);
  return out;
}

/** Real spherical harmonics up to degree L, ordered l = 0…L, m = −l…l. */
export function shVector(L: number, v: Vec3): Float64Array {
  const out = new Float64Array(nCoef(L));
  for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) out[idx(l, m)] = realSH(l, m, v);
  return out;
}

/** n quasi-uniform unit vectors (Fibonacci sphere). */
export function fibonacciDirs(n: number): Vec3[] {
  const g = Math.PI * (3 - Math.sqrt(5));
  return Array.from({ length: n }, (_, k) => {
    const z = 1 - (2 * (k + 0.5)) / n;
    const s = Math.sqrt(1 - z * z);
    return [s * Math.cos(g * k), s * Math.sin(g * k), z] as Vec3;
  });
}

function solveSym(S: Float64Array, B: Float64Array, n: number, cols: number): Float64Array {
  // Gaussian elimination with partial pivoting on [S | B]; S is n×n, B is n×cols.
  const a = Float64Array.from(S);
  const b = Float64Array.from(B);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]!) > Math.abs(a[p * n + c]!)) p = r;
    if (p !== c) {
      for (let k = 0; k < n; k++) [a[c * n + k], a[p * n + k]] = [a[p * n + k]!, a[c * n + k]!];
      for (let k = 0; k < cols; k++) [b[c * cols + k], b[p * cols + k]] = [b[p * cols + k]!, b[c * cols + k]!];
    }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r * n + c]! / a[c * n + c]!;
      if (f === 0) continue;
      for (let k = c; k < n; k++) a[r * n + k]! -= f * a[c * n + k]!;
      for (let k = 0; k < cols; k++) b[r * cols + k]! -= f * b[c * cols + k]!;
    }
  }
  for (let r = 0; r < n; r++) for (let k = 0; k < cols; k++) b[r * cols + k]! /= a[r * n + r]!;
  return b;
}

/**
 * Real Wigner matrices D^l(R), one (2l+1)² row-major block per l, defined by Y_l(R v) = D^l Y_l(v):
 * the coefficients of the rotated function f(Rᵀ û) are D x. Obtained by least squares on sample
 * directions, which is exact because each degree-l space is closed under rotation.
 */
export function wignerBlocks(L: number, R: Mat3): Float64Array[] {
  const dirs = fibonacciDirs(4 * (2 * L + 1) + 8);
  const Rt = transpose(R);
  const blocks: Float64Array[] = [];
  for (let l = 0; l <= L; l++) {
    const d = 2 * l + 1;
    const YtY = new Float64Array(d * d);
    const YtB = new Float64Array(d * d);
    for (const u of dirs) {
      const y = Array.from({ length: d }, (_, k) => realSH(l, k - l, u));
      const yb = Array.from({ length: d }, (_, k) => realSH(l, k - l, matVec(Rt, u)));
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) {
          YtY[i * d + j]! += y[i]! * y[j]!;
          YtB[i * d + j]! += y[i]! * yb[j]!;
        }
    }
    blocks.push(solveSym(YtY, YtB, d, d));
  }
  return blocks;
}

export function applyBlocks(D: Float64Array[], x: ArrayLike<number>, L: number, transposed = false): Float64Array {
  const out = new Float64Array(nCoef(L));
  for (let l = 0; l <= L; l++) {
    const d = 2 * l + 1;
    const B = D[l]!;
    const o = l * l;
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += (transposed ? B[j * d + i]! : B[i * d + j]!) * x[o + j]!;
      out[o + i] = s;
    }
  }
  return out;
}

/** The edge frame: Q r̂ = ẑ (the harmonics' polar axis), followed by a roll γ about ẑ. */
export function edgeFrame(rhat: Vec3, gamma = 0): Mat3 {
  const e3 = scale(rhat, 1 / norm(rhat));
  const helper: Vec3 = Math.abs(e3[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const c = cross(helper, e3);
  const e1 = scale(c, 1 / norm(c));
  const e2 = cross(e3, e1);
  const Q: Mat3 = [...e1, ...e2, ...e3];
  return matMul(axisRotation([0, 0, 1], gamma), Q);
}

export type EdgeMode = 'so2' | 'dense';

interface Conv {
  cin: number;
  cout: number;
  /** so2: per (cout, l', cin, l, m ≥ 0, part) radial coefficients; dense: per (cout, i', cin, i). */
  theta: Float64Array;
}

export interface ToyEscn {
  L: number;
  hidden: number;
  so2: [Conv, Conv];
  dense: [Conv, Conv];
}

function so2Size(L: number, cin: number, cout: number): number {
  return cout * (L + 1) * cin * (L + 1) * (L + 1) * 2 * ESEN_NGAUSS;
}

function randomConv(rand: () => number, size: number, cin: number, cout: number, fanIn: number): Conv {
  const s = 1.6 / Math.sqrt(fanIn);
  return { cin, cout, theta: new Float64Array(size).map(() => (2 * rand() - 1) * s) };
}

/** A two-convolution edge block for features up to degree L: [source, target] → hidden → 1 channel. */
export function makeToyEscn(o: { L: number; hidden?: number; seed?: number }): ToyEscn {
  const L = o.L;
  const hidden = o.hidden ?? 4;
  const rand = mulberry32(o.seed ?? 23);
  const n = nCoef(L);
  return {
    L,
    hidden,
    so2: [randomConv(rand, so2Size(L, 2, hidden), 2, hidden, 2 * (L + 1)), randomConv(rand, so2Size(L, hidden, 1), hidden, 1, hidden * (L + 1))],
    dense: [
      randomConv(rand, hidden * n * 2 * n * ESEN_NGAUSS, 2, hidden, 2 * n),
      randomConv(rand, 1 * n * hidden * n * ESEN_NGAUSS, hidden, 1, hidden * n),
    ],
  };
}

const radialDot = (theta: Float64Array, off: number, g: Float64Array) => {
  let s = 0;
  for (let k = 0; k < g.length; k++) s += theta[off + k]! * g[k]!;
  return s;
};

/** SO(2) convolution in the edge frame (eSCN Eq. 14), orders |m| > mmax dropped. */
export function so2Conv(c: Conv, x: Float64Array[], L: number, mmax: number, g: Float64Array): Float64Array[] {
  const out = Array.from({ length: c.cout }, () => new Float64Array(nCoef(L)));
  const NL = L + 1;
  const at = (co: number, lo: number, ci: number, li: number, m: number, part: number) =>
    ((((((co * NL + lo) * c.cin + ci) * NL + li) * NL + m) * 2 + part) * ESEN_NGAUSS) | 0;
  for (let co = 0; co < c.cout; co++)
    for (let m = 0; m <= Math.min(mmax, L); m++)
      for (let lo = m; lo <= L; lo++) {
        let yp = 0;
        let yn = 0;
        for (let ci = 0; ci < c.cin; ci++)
          for (let li = m; li <= L; li++) {
            const A = radialDot(c.theta, at(co, lo, ci, li, m, 0), g);
            const xp = x[ci]![idx(li, m)]!;
            if (m === 0) {
              yp += A * xp;
              continue;
            }
            const B = radialDot(c.theta, at(co, lo, ci, li, m, 1), g);
            const xn = x[ci]![idx(li, -m)]!;
            yp += A * xp - B * xn;
            yn += B * xp + A * xn;
          }
        out[co]![idx(lo, m)] = yp;
        if (m > 0) out[co]![idx(lo, -m)] = yn;
      }
  return out;
}

/** Control: an unrestricted linear map of all coefficients in the edge frame (orders |m| > mmax still dropped). */
export function denseConv(c: Conv, x: Float64Array[], L: number, mmax: number, g: Float64Array): Float64Array[] {
  const n = nCoef(L);
  const keep = (i: number) => {
    const l = Math.floor(Math.sqrt(i));
    return Math.abs(i - l * l - l) <= mmax;
  };
  const out = Array.from({ length: c.cout }, () => new Float64Array(n));
  for (let co = 0; co < c.cout; co++)
    for (let io = 0; io < n; io++) {
      if (!keep(io)) continue;
      let s = 0;
      for (let ci = 0; ci < c.cin; ci++)
        for (let ii = 0; ii < n; ii++) if (keep(ii)) s += radialDot(c.theta, (((co * n + io) * c.cin + ci) * n + ii) * ESEN_NGAUSS, g) * x[ci]![ii]!;
      out[co]![io] = s;
    }
  return out;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Gate: SiLU on each channel's l = 0 part, its sigmoid scales the l > 0 parts (rotation-equivariant). */
export function gate(x: Float64Array[]): Float64Array[] {
  return x.map((c) => {
    const s = c[0]!;
    const y = c.map((v) => v * sigmoid(s));
    y[0] = s * sigmoid(s);
    return y;
  });
}

export interface MessageOptions {
  mmax: number;
  gamma: number;
  mode: EdgeMode;
}

export interface MessageResult {
  /** Message in the global frame. */
  message: Float64Array;
  /** Source and target features in the edge frame (before the convolutions). */
  edgeIn: [Float64Array, Float64Array];
  /** Message in the edge frame, before rotating back and the envelope. */
  edgeOut: Float64Array;
  envelope: number;
}

/** The edge message from source s to target t; rvec = r_t − r_s. */
export function edgeMessage(m: ToyEscn, xs: Float64Array, xt: Float64Array, rvec: Vec3, o: MessageOptions): MessageResult {
  const L = m.L;
  const r = norm(rvec);
  const D = wignerBlocks(L, edgeFrame(rvec, o.gamma));
  const edgeIn: [Float64Array, Float64Array] = [applyBlocks(D, xs, L), applyBlocks(D, xt, L)];
  const g = gaussianBasis(r);
  const conv = o.mode === 'so2' ? so2Conv : denseConv;
  const [c1, c2] = o.mode === 'so2' ? m.so2 : m.dense;
  const h = gate(conv(c1, edgeIn, L, o.mmax, g));
  const edgeOut = conv(c2, h, L, o.mmax, g)[0]!;
  const envelope = esenEnvelope(r / ESEN_RCUT);
  const message = applyBlocks(D, edgeOut, L, true).map((v) => v * envelope);
  return { message, edgeIn, edgeOut, envelope };
}

const rel = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let d = 0;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    d += (a[i]! - b[i]!) ** 2;
    n += b[i]! ** 2;
  }
  return Math.sqrt(d / (n || 1));
};

/** Relative change of the message when the random roll γ about the bond is redrawn. */
export function rollDependence(m: ToyEscn, xs: Float64Array, xt: Float64Array, rvec: Vec3, o: MessageOptions, gammas = [0.7, 1.9, 3.1, 4.4, 5.6]): number {
  const ref = edgeMessage(m, xs, xt, rvec, { ...o, gamma: 0 }).message;
  return Math.max(...gammas.map((g) => rel(edgeMessage(m, xs, xt, rvec, { ...o, gamma: g }).message, ref)));
}

/** ‖msg(D xs, D xt, R r) − D msg(xs, xt, r)‖ / ‖msg‖ for a rotation R. */
export function equivarianceError(m: ToyEscn, xs: Float64Array, xt: Float64Array, rvec: Vec3, R: Mat3, o: MessageOptions): number {
  const D = wignerBlocks(m.L, R);
  const rotated = edgeMessage(m, applyBlocks(D, xs, m.L), applyBlocks(D, xt, m.L), matVec(R, rvec), o).message;
  const expected = applyBlocks(D, edgeMessage(m, xs, xt, rvec, o).message, m.L);
  return rel(rotated, expected);
}

/** Value of f(û) = Σ x_lm Y_lm(û). */
export function signalAt(x: ArrayLike<number>, L: number, u: Vec3): number {
  const y = shVector(L, u);
  let s = 0;
  for (let i = 0; i < y.length; i++) s += x[i]! * y[i]!;
  return s;
}

/** Deterministic random feature vector, with the l-th block scaled by 1/(l+1). */
export function randomFeature(L: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const x = new Float64Array(nCoef(L));
  for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) x[idx(l, m)] = (2 * rand() - 1) / (l + 1);
  return x;
}

/**
 * Multiply–add counts per pair of channels for one edge, as functions of L.
 * Dense SO(3) tensor product of x (l_i ≤ L) with Y(r̂) (l_f ≤ L) into outputs l_o ≤ L: one term per
 * Clebsch–Gordan entry, Σ (2l_i+1)(2l_f+1)(2l_o+1) over triangle-allowed triples, O(L⁶).
 * SO(2) convolution: (L+1)² for m = 0 and 4(L−m+1)² for each 0 < m ≤ M, plus the two rotations
 * 2 Σ (2l+1)², O(L³).
 */
export function tensorProductTerms(L: number): number {
  let s = 0;
  for (let li = 0; li <= L; li++)
    for (let lf = 0; lf <= L; lf++)
      for (let lo = Math.abs(li - lf); lo <= Math.min(L, li + lf); lo++) s += (2 * li + 1) * (2 * lf + 1) * (2 * lo + 1);
  return s;
}

export function so2Terms(L: number, mmax = L): { conv: number; rotation: number; total: number } {
  let conv = (L + 1) ** 2;
  for (let m = 1; m <= Math.min(mmax, L); m++) conv += 4 * (L - m + 1) ** 2;
  let rotation = 0;
  for (let l = 0; l <= L; l++) rotation += 2 * (2 * l + 1) ** 2;
  return { conv, rotation, total: conv + rotation };
}
