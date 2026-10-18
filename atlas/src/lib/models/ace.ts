// The linear atomic cluster expansion (ACE) for one species, built from scratch the way Dusson et al.
// construct it: couple products of the atomic base A_{nlm} = Σ_j R_n(r_j) Y_lm(r̂_j) to rotation
// invariants with Clebsch–Gordan trees, then remove the linear dependences numerically.
//
//   E_i = Σ_v c_v B_v(i),   B_v = Σ_{m} C^v_{m_1…m_ν} Π_t A_{n_t l_t m_t}
//
// The coupling coefficients are complex Clebsch–Gordan products (Racah's formula, Condon–Shortley
// phase) mapped to the real harmonics of ../physics/spherical-harmonics.ts by a unitary matrix that
// is computed by quadrature, so the two phase conventions never have to be matched by hand. A tree
// gives a complex invariant tensor; its real and imaginary parts are both real invariants. Every
// tree of one ordered (n, l) tuple is tried; random environments then reveal which combinations
// are independent once the product's permutation symmetry is taken into account, and the number
// kept must equal the dimension predicted by the character formula (countInvariants), which the
// tests check.

import { mulberry32, randomRotation } from '../chem/rng';
import { realSH, type Vec3 } from '../physics/spherical-harmonics';
import { gaussLegendre, rotate } from './grace';

export type { Vec3 };

// ---------------------------------------------------------------------------
// Clebsch–Gordan coefficients and the complex → real map
// ---------------------------------------------------------------------------

const logFact: number[] = [0];
for (let k = 1; k <= 200; k++) logFact.push(logFact[k - 1]! + Math.log(k));
const lf = (n: number) => logFact[n]!;

/** ⟨l1 m1 l2 m2 | L M⟩ by Racah's formula (Condon–Shortley phase). */
export function clebschGordan(l1: number, m1: number, l2: number, m2: number, L: number, M: number): number {
  if (m1 + m2 !== M || L < Math.abs(l1 - l2) || L > l1 + l2 || Math.abs(m1) > l1 || Math.abs(m2) > l2 || Math.abs(M) > L) return 0;
  const pre = 0.5 * (Math.log(2 * L + 1) + lf(L + l1 - l2) + lf(L - l1 + l2) + lf(l1 + l2 - L) - lf(l1 + l2 + L + 1)) +
    0.5 * (lf(L + M) + lf(L - M) + lf(l1 - m1) + lf(l1 + m1) + lf(l2 - m2) + lf(l2 + m2));
  let s = 0;
  const kmin = Math.max(0, l2 - L - m1, l1 - L + m2);
  const kmax = Math.min(l1 + l2 - L, l1 - m1, l2 + m2);
  for (let k = kmin; k <= kmax; k++) {
    const den = lf(k) + lf(l1 + l2 - L - k) + lf(l1 - m1 - k) + lf(l2 + m2 - k) + lf(L - l2 + m1 + k) + lf(L - l1 - m2 + k);
    s += (k % 2 ? -1 : 1) * Math.exp(pre - den);
  }
  return s;
}

interface Complex {
  re: number;
  im: number;
}

/** Complex harmonic Y_l^μ (Condon–Shortley phase) as re/im, from the real ones: exact identities. */
function complexSH(l: number, mu: number, v: Vec3): Complex {
  // realSH omits the Condon–Shortley phase: Y_l^{μ}(CS) = (−1)^μ (Y_{lμ} + i Y_{l,−μ}) / √2 for μ > 0.
  if (mu === 0) return { re: realSH(l, 0, v), im: 0 };
  const a = Math.abs(mu);
  const c = realSH(l, a, v) / Math.SQRT2;
  const s = realSH(l, -a, v) / Math.SQRT2;
  const sign = a % 2 ? -1 : 1;
  // Y_l^{−μ} = (−1)^μ conj(Y_l^μ)
  return mu > 0 ? { re: sign * c, im: sign * s } : { re: c, im: -s };
}

const uCache = new Map<number, { re: Float64Array; im: Float64Array }>();

/**
 * U[m][μ] = ∫ Y^real_{lm} conj(Y_l^μ) dΩ, so that Y^real_{lm} = Σ_μ U_{mμ} Y_l^μ. Computed by exact
 * quadrature rather than typed, so it cannot disagree with either convention.
 */
export function realFromComplex(l: number): { re: Float64Array; im: Float64Array } {
  const hit = uCache.get(l);
  if (hit) return hit;
  const d = 2 * l + 1;
  const re = new Float64Array(d * d);
  const im = new Float64Array(d * d);
  const { x, w } = gaussLegendre(l + 2);
  const nphi = 2 * l + 3;
  x.forEach((z, a) => {
    const s = Math.sqrt(Math.max(0, 1 - z * z));
    for (let b = 0; b < nphi; b++) {
      const ph = (2 * Math.PI * b) / nphi;
      const v: Vec3 = [s * Math.cos(ph), s * Math.sin(ph), z];
      const wt = (w[a]! * 2 * Math.PI) / nphi;
      for (let m = -l; m <= l; m++) {
        const yr = realSH(l, m, v);
        for (let mu = -l; mu <= l; mu++) {
          const yc = complexSH(l, mu, v);
          re[(m + l) * d + mu + l]! += wt * yr * yc.re;
          im[(m + l) * d + mu + l]! -= wt * yr * yc.im;
        }
      }
    }
  });
  for (let k = 0; k < d * d; k++) {
    if (Math.abs(re[k]!) < 1e-13) re[k] = 0;
    if (Math.abs(im[k]!) < 1e-13) im[k] = 0;
  }
  const out = { re, im };
  uCache.set(l, out);
  return out;
}

// ---------------------------------------------------------------------------
// Coupling trees: ((l1 l2)L1 l3)L2 … l_ν → 0
// ---------------------------------------------------------------------------

/** Every chain of intermediate L's that couples l_1 … l_ν to a total of zero. */
export function couplingTrees(ls: number[]): number[][] {
  const nu = ls.length;
  if (nu === 1) return ls[0] === 0 ? [[]] : [];
  if (nu === 2) return ls[0] === ls[1] ? [[]] : [];
  const out: number[][] = [];
  const rec = (k: number, Lprev: number, chain: number[]) => {
    // couple Lprev with ls[k]; the last intermediate must equal ls[ν − 1]
    if (k === nu - 1) {
      if (Lprev === ls[nu - 1]) out.push(chain.slice());
      return;
    }
    for (let L = Math.abs(Lprev - ls[k]!); L <= Lprev + ls[k]!; L++) rec(k + 1, L, [...chain, L]);
  };
  rec(1, ls[0]!, []);
  return out;
}

/** A sparse coupling tensor over real m-tuples: flat (m_1 + l_1, …) indices and values. */
export interface Coupling {
  ms: Int8Array; // ν entries per term
  c: Float64Array;
}

/** Complex coefficient of a tree for complex μ-tuples, by chaining Clebsch–Gordan coefficients. */
function treeCoefficient(ls: number[], chain: number[], mus: number[]): number {
  const nu = ls.length;
  if (nu === 1) return 1;
  // the last step couples (L_{ν−2}, M) with l_ν to 0 via ⟨L M l −M | 0 0⟩
  let c = 1;
  let Lp = ls[0]!;
  let Mp = mus[0]!;
  for (let k = 1; k < nu; k++) {
    const L = k === nu - 1 ? 0 : chain[k - 1]!;
    const M = Mp + mus[k]!;
    const g = clebschGordan(Lp, Mp, ls[k]!, mus[k]!, L, M);
    if (g === 0) return 0;
    c *= g;
    Lp = L;
    Mp = M;
  }
  return c;
}

/**
 * The tree's invariant tensor in the real basis: t_m = Σ_μ C(μ) Π_t conj(U_{m_t μ_t}), returned as its
 * real and imaginary parts (each a real invariant tensor; either may vanish).
 */
export function realTreeTensor(ls: number[], chain: number[]): { re: Coupling; im: Coupling } {
  const nu = ls.length;
  const dims = ls.map((l) => 2 * l + 1);
  const Us = ls.map(realFromComplex);
  const total = dims.reduce((a, b) => a * b, 1);
  const tre = new Float64Array(total);
  const tim = new Float64Array(total);
  // loop over complex μ-tuples with Σμ = 0 and nonzero coefficient
  const mus = ls.map((l) => -l);
  const ms = new Array<number>(nu).fill(0);
  for (;;) {
    let sum = 0;
    for (const m of mus) sum += m;
    if (sum === 0) {
      const C = treeCoefficient(ls, chain, mus);
      if (C !== 0) {
        // expand Π_t conj(U[m_t][μ_t]) over real m-tuples; each U column has at most two nonzeros
        const cols = mus.map((mu, t) => {
          const d = dims[t]!;
          const U = Us[t]!;
          const list: Array<[number, number, number]> = [];
          for (let m = 0; m < d; m++) {
            const r = U.re[m * d + mu + ls[t]!]!;
            const i = -U.im[m * d + mu + ls[t]!]!;
            if (r !== 0 || i !== 0) list.push([m, r, i]);
          }
          return list;
        });
        const rec = (t: number, flat: number, pr: number, pi: number) => {
          if (t === nu) {
            tre[flat]! += C * pr;
            tim[flat]! += C * pi;
            return;
          }
          for (const [m, r, i] of cols[t]!) {
            ms[t] = m;
            rec(t + 1, flat * dims[t]! + m, pr * r - pi * i, pr * i + pi * r);
          }
        };
        rec(0, 0, 1, 0);
      }
    }
    let k = nu - 1;
    while (k >= 0 && mus[k] === ls[k]) mus[k--] = -ls[k + 1]!;
    if (k < 0) break;
    mus[k]!++;
  }
  const sparse = (t: Float64Array): Coupling => {
    let scale = 0;
    for (const x of t) scale = Math.max(scale, Math.abs(x));
    const idx: number[] = [];
    for (let f = 0; f < total; f++) if (Math.abs(t[f]!) > 1e-12 * (scale || 1)) idx.push(f);
    const msOut = new Int8Array(idx.length * nu);
    const c = new Float64Array(idx.length);
    idx.forEach((f, q) => {
      let r = f;
      for (let t = nu - 1; t >= 0; t--) {
        msOut[q * nu + t] = r % dims[t]!;
        r = Math.floor(r / dims[t]!);
      }
      c[q] = scale > 1e-12 ? t[f]! : 0;
    });
    return { ms: msOut, c: scale > 1e-12 ? c : new Float64Array(0) };
  };
  return { re: sparse(tre), im: sparse(tim) };
}

// ---------------------------------------------------------------------------
// Counting invariants with characters
// ---------------------------------------------------------------------------

/** One-particle label: radial index n ≥ 1 and degree l. */
export interface NL {
  n: number;
  l: number;
}

const chi = (l: number, th: number) => (Math.abs(Math.sin(th / 2)) < 1e-14 ? 2 * l + 1 : Math.sin((l + 0.5) * th) / Math.sin(th / 2));

/** Character of Sym^k(V_l) at rotation angle θ (Newton's identity for complete symmetric functions). */
function symChar(l: number, k: number, th: number): number {
  const h = [1];
  for (let q = 1; q <= k; q++) {
    let s = 0;
    for (let j = 1; j <= q; j++) s += chi(l, j * th) * h[q - j]!;
    h.push(s / q);
  }
  return h[k]!;
}

/** Group a sorted multiset of (n, l) into distinct labels with multiplicities. */
function groups(nls: NL[]): Array<{ l: number; k: number }> {
  const out: Array<{ n: number; l: number; k: number }> = [];
  for (const v of nls) {
    const last = out[out.length - 1];
    if (last && last.n === v.n && last.l === v.l) last.k++;
    else out.push({ ...v, k: 1 });
  }
  return out;
}

/**
 * Number of linearly independent O(3)-invariant, permutation-symmetric functions Σ_m C_m Π A_{n_t l_t m_t}
 * for one multiset {(n_t, l_t)}: ∫_SO(3) Π_groups χ_{Sym^k V_l}(R) dR, and zero when Σ l is odd
 * (inversion multiplies the whole product by (−1)^{Σl}). The integrand depends only on the rotation
 * angle; with the Haar weight (1 − cos θ)/π it is a trigonometric polynomial, integrated exactly.
 */
export function countInvariants(nls: NL[]): number {
  const lsum = nls.reduce((s, v) => s + v.l, 0);
  if (lsum % 2) return 0;
  const g = groups(nls);
  const deg = g.reduce((s, x) => s + x.k * x.l, 0) + 2;
  const N = 2 * deg + 8;
  let s = 0;
  for (let q = 0; q < N; q++) {
    const th = (Math.PI * (q + 0.5)) / N; // midpoint rule on [0, π]: exact for cosine polynomials of degree < 2N
    let f = 1 - Math.cos(th);
    for (const x of g) f *= symChar(x.l, x.k, th);
    s += f;
  }
  return Math.round(s / N) || 0;
}

// ---------------------------------------------------------------------------
// Basis specification and enumeration
// ---------------------------------------------------------------------------

export interface AceSpec {
  /** Radial functions n = 1 … nmax. */
  nmax: number;
  lmax: number;
  /** Maximum correlation order ν (body order ν + 1). */
  numax: number;
  /** Total-degree budget per order: Σ_t (n_t + wL·l_t) ≤ degree[ν − 1]. */
  degree: number[];
  wL: number;
}

/** Every sorted (n, l) multiset of size ν with Σl even within the degree budget. */
export function multisets(spec: AceSpec, nu: number): NL[][] {
  const labels: NL[] = [];
  for (let n = 1; n <= spec.nmax; n++) for (let l = 0; l <= spec.lmax; l++) labels.push({ n, l });
  const D = spec.degree[nu - 1] ?? 0;
  const out: NL[][] = [];
  const rec = (start: number, cur: NL[], deg: number) => {
    if (cur.length === nu) {
      if (cur.reduce((s, v) => s + v.l, 0) % 2 === 0) out.push(cur.slice());
      return;
    }
    for (let k = start; k < labels.length; k++) {
      const v = labels[k]!;
      const d = deg + v.n + spec.wL * v.l;
      if (d > D + 1e-9) continue;
      cur.push(v);
      rec(k, cur, d);
      cur.pop();
    }
  };
  rec(0, [], 0);
  return out;
}

/** One basis function B_v: its (n, l) multiset and a real coupling tensor over the m's. */
export interface BasisFunction {
  nls: NL[];
  coupling: Coupling;
}

export interface MultisetReport {
  nls: NL[];
  /** Coupling trees of the sorted tuple (each gives a real and an imaginary candidate). */
  trees: number;
  /** Nonzero candidate tensors (before removing dependences). */
  candidates: number;
  /** Independent functions kept by the rank test. */
  kept: number;
  /** Dimension predicted by the character formula. */
  predicted: number;
}

/** Random test environments for the rank test (unit-ish radial values; any generic point set works). */
function randomA(nls: NL[], seed: number, samples: number): Array<Map<string, Float64Array>> {
  const rand = mulberry32(seed);
  const keys = [...new Set(nls.map((v) => `${v.n},${v.l}`))];
  return Array.from({ length: samples }, () => {
    const m = new Map<string, Float64Array>();
    // generic values of A_{nl·}: a few random "neighbours" with random radial weights, so the
    // per-l blocks of different n are independent and rotation acts on them consistently
    const nb = Array.from({ length: 7 }, () => {
      const v: Vec3 = [2 * rand() - 1, 2 * rand() - 1, 2 * rand() - 1];
      return { v, w: keys.map(() => 2 * rand() - 1) };
    });
    keys.forEach((key, q) => {
      const l = Number(key.split(',')[1]);
      const a = new Float64Array(2 * l + 1);
      for (const { v, w } of nb) for (let mm = -l; mm <= l; mm++) a[mm + l]! += w[q]! * realSH(l, mm, v);
      m.set(key, a);
    });
    return m;
  });
}

function evalOnMap(nls: NL[], cp: Coupling, A: Map<string, Float64Array>): number {
  const nu = nls.length;
  const blocks = nls.map((v) => A.get(`${v.n},${v.l}`)!);
  let s = 0;
  for (let q = 0; q < cp.c.length; q++) {
    let p = cp.c[q]!;
    for (let t = 0; t < nu; t++) p *= blocks[t]![cp.ms[q * nu + t]!]!;
    s += p;
  }
  return s;
}

/**
 * The independent invariants of one multiset: all tree candidates evaluated on random environments,
 * then a Gram–Schmidt pass that keeps a candidate only if it adds a new direction.
 */
export function multisetBasis(nls: NL[], seed = 1): { functions: BasisFunction[]; report: MultisetReport } {
  const ls = nls.map((v) => v.l);
  const trees = couplingTrees(ls);
  const cands: Coupling[] = [];
  for (const chain of trees) {
    const t = realTreeTensor(ls, chain);
    if (t.re.c.length) cands.push(t.re);
    if (t.im.c.length) cands.push(t.im);
  }
  const predicted = countInvariants(nls);
  const samples = Math.max(12, 3 * cands.length + 6);
  const envs = randomA(nls, seed + 7919 * nls.length + 31 * ls.reduce((a, b) => a + b, 0), samples);
  const kept: BasisFunction[] = [];
  const basis: Float64Array[] = [];
  for (const cp of cands) {
    const v = Float64Array.from(envs, (A) => evalOnMap(nls, cp, A));
    let n0 = 0;
    for (const x of v) n0 += x * x;
    n0 = Math.sqrt(n0);
    if (n0 < 1e-12) continue;
    for (let pass = 0; pass < 2; pass++)
      for (const b of basis) {
        let d = 0;
        for (let i = 0; i < v.length; i++) d += v[i]! * b[i]!;
        for (let i = 0; i < v.length; i++) v[i]! -= d * b[i]!;
      }
    let n1 = 0;
    for (const x of v) n1 += x * x;
    n1 = Math.sqrt(n1);
    if (n1 > 1e-7 * n0) {
      basis.push(v.map((x) => x / n1));
      kept.push({ nls, coupling: cp });
    }
  }
  return { functions: kept, report: { nls, trees: trees.length, candidates: cands.length, kept: kept.length, predicted } };
}

export interface AceBasis {
  spec: AceSpec;
  functions: BasisFunction[];
  reports: MultisetReport[];
}

const basisCache = new Map<string, AceBasis>();

/** The full basis for a specification, ordered by correlation order. */
export function buildBasis(spec: AceSpec): AceBasis {
  const key = JSON.stringify(spec);
  const hit = basisCache.get(key);
  if (hit) return hit;
  const functions: BasisFunction[] = [];
  const reports: MultisetReport[] = [];
  for (let nu = 1; nu <= spec.numax; nu++)
    for (const nls of multisets(spec, nu)) {
      const r = multisetBasis(nls);
      functions.push(...r.functions);
      reports.push(r.report);
    }
  const out = { spec, functions, reports };
  basisCache.set(key, out);
  return out;
}

/** Counts by order without building tensors: multisets, product-basis size Π(2l+1) and invariants. */
export function countBasis(spec: AceSpec): Array<{ nu: number; multisets: number; products: number; invariants: number }> {
  return Array.from({ length: spec.numax }, (_, k) => {
    const nu = k + 1;
    const ms = multisets(spec, nu);
    let products = 0;
    let invariants = 0;
    for (const nls of ms) {
      // distinct m-products of a multiset: for each group of k equal labels, multisets of k m's
      let p = 1;
      for (const g of groups(nls)) p *= binom(2 * g.l + 1 + g.k - 1, g.k);
      products += p;
      invariants += countInvariants(nls);
    }
    return { nu, multisets: ms.length, products, invariants };
  });
}

function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}

// ---------------------------------------------------------------------------
// Radial basis and the atomic base with gradients
// ---------------------------------------------------------------------------

export interface Radial {
  rc: number;
  /** Exponential scaling of the distance (pacemaker's λ). */
  lambda: number;
}

/**
 * Radial functions g_1 … g_nmax at distance r and their r-derivatives: Chebyshev polynomials of an
 * exponentially scaled distance times a cosine cutoff (the ChebExpCos form; see radialForm on the page).
 */
export function radialBasis(r: number, nmax: number, R: Radial): { g: Float64Array; dg: Float64Array } {
  const g = new Float64Array(nmax);
  const dg = new Float64Array(nmax);
  if (r >= R.rc) return { g, dg };
  const el = Math.exp(R.lambda) - 1;
  const e = Math.exp(-R.lambda * (r / R.rc - 1));
  const x = 1 - (2 * (e - 1)) / el;
  const dx = (2 * R.lambda * e) / (R.rc * el);
  const fc = 0.5 * (1 + Math.cos((Math.PI * r) / R.rc));
  const dfc = (-0.5 * Math.PI * Math.sin((Math.PI * r) / R.rc)) / R.rc;
  // T_k(x) and T_k'(x)
  const T = [1, x];
  const dT = [0, 1];
  for (let k = 2; k <= nmax; k++) {
    T.push(2 * x * T[k - 1]! - T[k - 2]!);
    dT.push(2 * T[k - 1]! + 2 * x * dT[k - 1]! - dT[k - 2]!);
  }
  for (let n = 1; n <= nmax; n++) {
    // g_1 = f_c, g_n = ½(1 − T_{n−1}(x)) f_c for n ≥ 2
    const base = n === 1 ? 1 : 0.5 * (1 - T[n - 1]!);
    const dbase = n === 1 ? 0 : -0.5 * dT[n - 1]! * dx;
    g[n - 1] = base * fc;
    dg[n - 1] = dbase * fc + base * dfc;
  }
  return { g, dg };
}

export const lmIndex = (l: number, m: number) => l * l + l + m;

/** Real harmonics Y_lm(r̂) for l ≤ lmax and their Cartesian gradients ∇_r Y_lm(r/|r|), by central differences. */
function harmonics(v: Vec3, lmax: number): { Y: Float64Array; dY: Float64Array[] } {
  const K = (lmax + 1) ** 2;
  const Y = new Float64Array(K);
  for (let l = 0; l <= lmax; l++) for (let m = -l; m <= l; m++) Y[lmIndex(l, m)] = realSH(l, m, v);
  const r = Math.hypot(v[0], v[1], v[2]);
  const h = 1e-5 * r;
  const dY = [0, 1, 2].map((a) => {
    const p: Vec3 = [...v];
    const q: Vec3 = [...v];
    p[a]! += h;
    q[a]! -= h;
    const d = new Float64Array(K);
    for (let l = 0; l <= lmax; l++) for (let m = -l; m <= l; m++) d[lmIndex(l, m)] = (realSH(l, m, p) - realSH(l, m, q)) / (2 * h);
    return d;
  });
  return { Y, dY };
}

/** A_{nlm} of one environment, flat as [n − 1][lm], and optionally its derivative with respect to each neighbour. */
export interface AtomicBase {
  A: Float64Array;
  /** dA[j][axis] = ∂A/∂r_j along x, y, z (flat like A), only when requested. */
  dA?: Float64Array[][];
}

export function atomicBase(env: Vec3[], nmax: number, lmax: number, R: Radial, withGrad = false): AtomicBase {
  const K = (lmax + 1) ** 2;
  const A = new Float64Array(nmax * K);
  const dA: Float64Array[][] = [];
  for (const v of env) {
    const r = Math.hypot(v[0], v[1], v[2]);
    if (r >= R.rc) {
      if (withGrad) dA.push([new Float64Array(0), new Float64Array(0), new Float64Array(0)]);
      continue;
    }
    const { g, dg } = radialBasis(r, nmax, R);
    const { Y, dY } = harmonics(v, lmax);
    for (let n = 0; n < nmax; n++) for (let k = 0; k < K; k++) A[n * K + k]! += g[n]! * Y[k]!;
    if (withGrad) {
      const grads = [0, 1, 2].map((a) => {
        const d = new Float64Array(nmax * K);
        const ur = v[a]! / r;
        for (let n = 0; n < nmax; n++) for (let k = 0; k < K; k++) d[n * K + k] = dg[n]! * ur * Y[k]! + g[n]! * dY[a]![k]!;
        return d;
      });
      dA.push(grads);
    }
  }
  return withGrad ? { A, dA } : { A };
}

// ---------------------------------------------------------------------------
// Evaluating the basis
// ---------------------------------------------------------------------------

/** Offsets of each basis function's factors into the flat A: A[(n − 1)K + l² + m_index]. */
function offsets(f: BasisFunction, K: number): number[] {
  return f.nls.map((v) => (v.n - 1) * K + v.l * v.l);
}

/** B_v for every basis function. */
export function evaluateBasis(basis: AceBasis, A: Float64Array): Float64Array {
  const K = (basis.spec.lmax + 1) ** 2;
  const out = new Float64Array(basis.functions.length);
  basis.functions.forEach((f, q) => {
    const nu = f.nls.length;
    const off = offsets(f, K);
    const { ms, c } = f.coupling;
    let s = 0;
    for (let t = 0; t < c.length; t++) {
      let p = c[t]!;
      for (let k = 0; k < nu; k++) p *= A[off[k]! + ms[t * nu + k]!]!;
      s += p;
    }
    out[q] = s;
  });
  return out;
}

/** B_v and ∂B_v/∂A (dense rows, one per function): B is a polynomial in A, so this is exact. */
export function evaluateBasisWithDerivative(basis: AceBasis, A: Float64Array): { B: Float64Array; dBdA: Float64Array[] } {
  const K = (basis.spec.lmax + 1) ** 2;
  const B = new Float64Array(basis.functions.length);
  const dBdA: Float64Array[] = [];
  basis.functions.forEach((f, q) => {
    const nu = f.nls.length;
    const off = offsets(f, K);
    const { ms, c } = f.coupling;
    const d = new Float64Array(A.length);
    const fac = new Float64Array(nu);
    let s = 0;
    for (let t = 0; t < c.length; t++) {
      let p = c[t]!;
      for (let k = 0; k < nu; k++) {
        fac[k] = A[off[k]! + ms[t * nu + k]!]!;
        p *= fac[k]!;
      }
      s += p;
      for (let k = 0; k < nu; k++) {
        let pk = c[t]!;
        for (let j = 0; j < nu; j++) if (j !== k) pk *= fac[j]!;
        d[off[k]! + ms[t * nu + k]!]! += pk;
      }
    }
    B[q] = s;
    dBdA.push(d);
  });
  return { B, dBdA };
}

/** Basis values of one environment (convenience). */
export function environmentBasis(basis: AceBasis, env: Vec3[], R: Radial): Float64Array {
  return evaluateBasis(basis, atomicBase(env, basis.spec.nmax, basis.spec.lmax, R).A);
}

/** Random rotation helper for tests and figures. */
export function randomlyRotated(env: Vec3[], seed: number): Vec3[] {
  return rotate(env, randomRotation(mulberry32(seed)));
}
