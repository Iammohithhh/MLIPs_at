// Graph Atomic Cluster Expansion (GRACE): star and tree cluster graphs counted in real
// crystals, and the ACE "density trick" with its self-interaction terms, computed exactly.
//
// Graph ACE builds cluster basis functions on trees whose directed edges each carry a
// single-particle function φ(r_ij) that vanishes beyond r_cut; local ACE keeps only stars,
// all edges leaving the root (Bochkarev, Lysogorskiy & Drautz 2024, arXiv:2311.16326v2,
// Secs. II.4–II.5, IV; Figs. 3–5). A T-layer recursion evaluates trees of depth ≤ T
// (Sec. VII, Eqs. 84–86). The atomic base A_inlm = Σ_j R_n(r_ij) Y_lm(r̂_ij) turns a sum over
// ν-tuples of neighbours into a product of ν single sums, at the price of including tuples
// with repeated neighbours ("self-interactions", Sec. IV.2; Drautz 2019).
//
// Radial functions follow gracemaker's "Cheb" basis (tensorpotential/functions/radial.py,
// ChebSqrRadialBasisFunction): T_n(2r/r_c − 1) × the p = 5 polynomial envelope. The released
// models then pass these through an MLP and a chemical embedding; the toy here does not.

import { realSH, type Vec3 } from '../physics/spherical-harmonics';
import { mulberry32 } from '../chem/rng';

export type { Vec3 };

export const GRACE_RCUT = 6.0;
export const GRACE_ENVELOPE_P = 5;

// ---------------------------------------------------------------------------
// Tree topologies and their embeddings in a crystal
// ---------------------------------------------------------------------------

/**
 * Rooted trees in the paper's notation: the string lists each non-root node's distance from
 * the root (Sec. II.5, Fig. 4). `parent[k]` is the parent of node k + 1 (0 = root).
 */
export interface Topology {
  id: string;
  label: string;
  parent: number[];
  /** Size of the automorphism group (permutations of nodes that preserve the rooted tree). */
  aut: number;
}

export const TOPOLOGIES: readonly Topology[] = [
  { id: '1', label: '(1)', parent: [0], aut: 1 },
  { id: '11', label: '(11)', parent: [0, 0], aut: 2 },
  { id: '12', label: '(12)', parent: [0, 1], aut: 1 },
  { id: '111', label: '(111)', parent: [0, 0, 0], aut: 6 },
  { id: '112', label: '(112)', parent: [0, 0, 1], aut: 1 },
  { id: '122', label: '(122)', parent: [0, 1, 1], aut: 2 },
  { id: '123', label: '(123)', parent: [0, 1, 2], aut: 1 },
];

export const topology = (id: string): Topology => TOPOLOGIES.find((t) => t.id === id)!;

/** Number of nodes including the root (the body order of the cluster function). */
export const nodes = (t: Topology) => t.parent.length + 1;

/** Largest number of edges between the root and a node: the number of GRACE layers needed. */
export function depth(t: Topology): number {
  const d = [0];
  t.parent.forEach((p, k) => (d[k + 1] = d[p]! + 1));
  return Math.max(...d);
}

export const isStar = (t: Topology) => depth(t) === 1;

/** A finite cluster around a root atom (index 0 at the origin) with neighbour lists. */
export interface Cluster {
  positions: Vec3[];
  /** Distance of each atom from the root. */
  radius: number[];
  /** For each atom, neighbours within `maxCut` and their distances (sorted by distance). */
  adj: Array<Array<{ j: number; d: number }>>;
  maxCut: number;
}

/**
 * Build neighbour lists inside a ball of atoms around the root. `positions[0]` must be the root
 * at the origin. Lists are complete for atoms within R − maxCut of the root, where R is the ball
 * radius; embeddings of depth T need complete lists up to (T − 1)·maxCut.
 */
export function buildCluster(positions: Vec3[], maxCut: number): Cluster {
  const n = positions.length;
  const adj: Cluster['adj'] = Array.from({ length: n }, () => []);
  // Cell list with cubes of side maxCut.
  const key = (p: Vec3) => p.map((c) => Math.floor(c / maxCut)).join(',');
  const cells = new Map<string, number[]>();
  positions.forEach((p, i) => {
    const k = key(p);
    (cells.get(k) ?? cells.set(k, []).get(k)!).push(i);
  });
  positions.forEach((p, i) => {
    const c = p.map((x) => Math.floor(x / maxCut));
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++)
        for (let e = -1; e <= 1; e++) {
          const list = cells.get(`${c[0]! + a},${c[1]! + b},${c[2]! + e}`);
          if (!list) continue;
          for (const j of list) {
            if (j === i) continue;
            const q = positions[j]!;
            const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
            if (d < maxCut) adj[i]!.push({ j, d });
          }
        }
    adj[i]!.sort((x, y) => x.d - y.d);
  });
  return { positions, radius: positions.map((p) => Math.hypot(...p)), adj, maxCut };
}

/**
 * Count the distinct embeddings of a rooted tree in the cluster: injective maps of its nodes to
 * atoms, root fixed at atom 0, every edge shorter than `rc`. Ordered embeddings are divided by the
 * tree's automorphisms, so each set of atoms with a given connectivity is counted once.
 * No repeated atoms: these are the self-interaction-free graphs of Sec. IV.2.
 */
export function countEmbeddings(c: Cluster, t: Topology, rc: number): number {
  if (rc > c.maxCut + 1e-12) throw new Error('rc exceeds the cluster neighbour-list cutoff');
  const m = t.parent.length;
  const atom = new Int32Array(m + 1);
  atom[0] = 0;
  let ordered = 0;
  const rec = (k: number) => {
    if (k > m) {
      ordered++;
      return;
    }
    const from = atom[t.parent[k - 1]!]!;
    for (const { j, d } of c.adj[from]!) {
      if (d >= rc) break;
      let used = false;
      for (let q = 0; q < k; q++) if (atom[q] === j) used = true;
      if (used) continue;
      atom[k] = j;
      rec(k + 1);
    }
  };
  rec(1);
  return ordered / t.aut;
}

/** One random embedding (atom indices, root first), or null if none exists. Deterministic in `seed`. */
export function sampleEmbedding(c: Cluster, t: Topology, rc: number, seed: number): number[] | null {
  const rand = mulberry32(seed);
  const m = t.parent.length;
  const atom: number[] = [0];
  const rec = (k: number): boolean => {
    if (k > m) return true;
    const from = atom[t.parent[k - 1]!]!;
    const options = c.adj[from]!.filter(({ j, d }) => d < rc && !atom.includes(j)).map(({ j }) => j);
    for (let i = options.length - 1; i > 0; i--) {
      const s = Math.floor(rand() * (i + 1));
      [options[i], options[s]] = [options[s]!, options[i]!];
    }
    for (const j of options) {
      atom[k] = j;
      if (rec(k + 1)) return true;
    }
    atom.length = k;
    return false;
  };
  return rec(1) ? atom : null;
}

/** Hop distance (in edges shorter than rc) of every atom from the root, up to `maxHops`; −1 beyond. */
export function hopLayers(c: Cluster, rc: number, maxHops: number): Int32Array {
  const hop = new Int32Array(c.positions.length).fill(-1);
  hop[0] = 0;
  let frontier = [0];
  for (let h = 1; h <= maxHops; h++) {
    const next: number[] = [];
    for (const i of frontier)
      for (const { j, d } of c.adj[i]!) {
        if (d >= rc) break;
        if (hop[j] === -1) {
          hop[j] = h;
          next.push(j);
        }
      }
    frontier = next;
  }
  return hop;
}

/** Atoms some tree of depth ≤ T can reach, and the largest root distance among them. */
export function reach(c: Cluster, rc: number, T: number): { atoms: number; rmax: number } {
  const hop = hopLayers(c, rc, T);
  let atoms = 0;
  let rmax = 0;
  hop.forEach((h, i) => {
    if (h > 0) {
      atoms++;
      rmax = Math.max(rmax, c.radius[i]!);
    }
  });
  return { atoms, rmax };
}

// ---------------------------------------------------------------------------
// Radial basis (gracemaker "Cheb") and the atomic base
// ---------------------------------------------------------------------------

/** gracemaker's polynomial cutoff: value, slope and curvature vanish at x = 1 for p = 5. */
export function envelope(x: number, p = GRACE_ENVELOPE_P): number {
  if (x >= 1) return 0;
  return 1 - ((p + 1) * (p + 2) / 2) * x ** p + p * (p + 2) * x ** (p + 1) - (p * (p + 1) / 2) * x ** (p + 2);
}

/** g_n(r) = T_n(2 r/r_c − 1) f_env(r/r_c) for n = 1 … nfunc (Chebyshev polynomials of the first kind). */
export function chebBasis(r: number, nfunc: number, rc = GRACE_RCUT): number[] {
  const x = r / rc;
  if (x >= 1) return new Array(nfunc).fill(0);
  const t = 2 * (1 - Math.abs(1 - x)) - 1;
  const T = [1, t];
  for (let k = 2; k <= nfunc; k++) T.push(2 * t * T[k - 1]! - T[k - 2]!);
  const f = envelope(x);
  return T.slice(1, nfunc + 1).map((v) => v * f);
}

export const lmIndex = (l: number, m: number) => l * l + l + m;

/** Single-particle functions φ_{nlm}(r) = g_n(r) Y_lm(r̂) for one neighbour vector, as [n][lm]. */
export function phi(v: Vec3, nfunc: number, lmax: number, rc = GRACE_RCUT): Float64Array[] {
  const g = chebBasis(Math.hypot(...v), nfunc, rc);
  const Y = new Float64Array((lmax + 1) ** 2);
  for (let l = 0; l <= lmax; l++) for (let m = -l; m <= l; m++) Y[lmIndex(l, m)] = realSH(l, m, v);
  return g.map((gn) => Y.map((y) => gn * y));
}

/** Atomic base A_{nlm} = Σ_j φ_{nlm}(r_j), as [n][lm]. Cost: one pass over the neighbours. */
export function atomicBase(env: Vec3[], nfunc: number, lmax: number, rc = GRACE_RCUT): Float64Array[] {
  const A = Array.from({ length: nfunc }, () => new Float64Array((lmax + 1) ** 2));
  for (const v of env) {
    const p = phi(v, nfunc, lmax, rc);
    p.forEach((row, n) => row.forEach((x, k) => (A[n]![k]! += x)));
  }
  return A;
}

// ---------------------------------------------------------------------------
// Real Gaunt coefficients by exact quadrature
// ---------------------------------------------------------------------------

/** Gauss–Legendre nodes and weights on [−1, 1] (exact for polynomials of degree ≤ 2n − 1). */
export function gaussLegendre(n: number): { x: number[]; w: number[] } {
  const x: number[] = [];
  const w: number[] = [];
  for (let i = 1; i <= n; i++) {
    let z = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5));
    let dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1;
      let p1 = z;
      for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k];
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-15) break;
    }
    x.push(z);
    w.push(2 / ((1 - z * z) * dp * dp));
  }
  return { x, w };
}

const gauntCache = new Map<string, Float64Array>();

/**
 * Real Gaunt coefficients G[m1][m2][m3] = ∫ Y_{l1m1} Y_{l2m2} Y_{l3m3} dΩ, flattened. The integrand
 * is a polynomial of degree l1 + l2 + l3 in cos θ and a trigonometric polynomial of the same degree
 * in φ, so Gauss–Legendre × uniform-φ quadrature with enough points is exact.
 */
export function gaunt(l1: number, l2: number, l3: number): Float64Array {
  const key = `${l1},${l2},${l3}`;
  const hit = gauntCache.get(key);
  if (hit) return hit;
  const L = l1 + l2 + l3;
  const { x, w } = gaussLegendre(Math.floor(L / 2) + 2);
  const nphi = L + 2;
  const d1 = 2 * l1 + 1;
  const d2 = 2 * l2 + 1;
  const d3 = 2 * l3 + 1;
  const G = new Float64Array(d1 * d2 * d3);
  x.forEach((z, a) => {
    const s = Math.sqrt(Math.max(0, 1 - z * z));
    for (let b = 0; b < nphi; b++) {
      const ph = (2 * Math.PI * b) / nphi;
      const v: Vec3 = [s * Math.cos(ph), s * Math.sin(ph), z];
      const wt = (w[a]! * 2 * Math.PI) / nphi;
      const y1 = Array.from({ length: d1 }, (_, k) => realSH(l1, k - l1, v));
      const y2 = Array.from({ length: d2 }, (_, k) => realSH(l2, k - l2, v));
      const y3 = Array.from({ length: d3 }, (_, k) => realSH(l3, k - l3, v));
      for (let i = 0; i < d1; i++)
        for (let j = 0; j < d2; j++)
          for (let k = 0; k < d3; k++) G[(i * d2 + j) * d3 + k]! += wt * y1[i]! * y2[j]! * y3[k]!;
    }
  });
  for (let i = 0; i < G.length; i++) if (Math.abs(G[i]!) < 1e-13) G[i] = 0;
  gauntCache.set(key, G);
  return G;
}

// ---------------------------------------------------------------------------
// Invariants: product basis versus explicit sums over neighbour tuples
// ---------------------------------------------------------------------------

/** One ν-correlation invariant: radial channels n_k (1-based) and degrees l_k. ν = 2 needs l1 = l2. */
export interface Invariant {
  n: number[];
  l: number[];
}

export const correlationOrder = (b: Invariant) => b.n.length;

/** Coupling weights over the m indices: δ for ν = 2, the real Gaunt coefficient for ν = 3. */
function coupling(b: Invariant): (ms: number[]) => number {
  if (b.l.length === 2) {
    if (b.l[0] !== b.l[1]) throw new Error('ν = 2 invariants need l1 = l2');
    return (ms) => (ms[0] === ms[1] ? 1 : 0);
  }
  if (b.l.length === 3) {
    const G = gaunt(b.l[0]!, b.l[1]!, b.l[2]!);
    const d2 = 2 * b.l[1]! + 1;
    const d3 = 2 * b.l[2]! + 1;
    return (ms) => G[((ms[0]! + b.l[0]!) * d2 + ms[1]! + b.l[1]!) * d3 + ms[2]! + b.l[2]!]!;
  }
  throw new Error('only ν = 2 and ν = 3 are implemented');
}

/** Iterate over all m-tuples with |m_k| ≤ l_k. */
function forEachM(l: number[], f: (ms: number[]) => void) {
  const ms = l.map((x) => -x);
  for (;;) {
    f(ms);
    let k = ms.length - 1;
    while (k >= 0 && ms[k] === l[k]) ms[k--] = -l[k + 1]!;
    if (k < 0) return;
    ms[k]!++;
  }
}

/** B = Σ_m C_m Π_k A_{n_k l_k m_k}: the product basis, cost independent of the number of neighbours. */
export function productInvariant(A: Float64Array[], b: Invariant): number {
  const C = coupling(b);
  let s = 0;
  forEachM(b.l, (ms) => {
    const c = C(ms);
    if (c === 0) return;
    let p = c;
    b.n.forEach((n, k) => (p *= A[n - 1]![lmIndex(b.l[k]!, ms[k]!)]!));
    s += p;
  });
  return s;
}

export interface Decomposition {
  /** Σ over ν-tuples with all neighbours distinct: the genuine (ν+1)-body term. */
  distinct: number;
  /** Tuples with exactly one coincidence (ν = 3 only: j = k ≠ l and permutations): (ν)-body. */
  partial: number;
  /** All ν neighbours the same atom: a two-body term. */
  diagonal: number;
  total: number;
  /** Number of neighbour-tuple terms evaluated. */
  terms: number;
}

/**
 * The same invariant as an explicit sum over ν-tuples of neighbours, split by how many of them
 * coincide. distinct + partial + diagonal equals the product-basis value exactly.
 */
export function explicitInvariant(env: Vec3[], b: Invariant, nfunc: number, rc = GRACE_RCUT): Decomposition {
  const lmax = Math.max(...b.l);
  const P = env.map((v) => phi(v, nfunc, lmax, rc));
  const C = coupling(b);
  const mlist: number[][] = [];
  const clist: number[] = [];
  forEachM(b.l, (ms) => {
    const c = C(ms);
    if (c !== 0) {
      mlist.push([...ms]);
      clist.push(c);
    }
  });
  const term = (js: number[]) => {
    let s = 0;
    mlist.forEach((ms, q) => {
      let p = clist[q]!;
      js.forEach((j, k) => (p *= P[j]![b.n[k]! - 1]![lmIndex(b.l[k]!, ms[k]!)]!));
      s += p;
    });
    return s;
  };
  const N = env.length;
  const out: Decomposition = { distinct: 0, partial: 0, diagonal: 0, total: 0, terms: 0 };
  if (b.n.length === 2) {
    for (let j = 0; j < N; j++)
      for (let k = 0; k < N; k++) {
        const t = term([j, k]);
        if (j === k) out.diagonal += t;
        else out.distinct += t;
        out.terms++;
      }
  } else {
    for (let j = 0; j < N; j++)
      for (let k = 0; k < N; k++)
        for (let l = 0; l < N; l++) {
          const t = term([j, k, l]);
          const eq = (j === k ? 1 : 0) + (k === l ? 1 : 0) + (j === l ? 1 : 0);
          if (eq === 0) out.distinct += t;
          else if (eq === 3) out.diagonal += t;
          else out.partial += t;
          out.terms++;
        }
  }
  out.total = out.distinct + out.partial + out.diagonal;
  return out;
}

/**
 * The ν = 2 self-interaction term as a pure two-body sum: Σ_j g_{n1}(r_j) g_{n2}(r_j) (2l + 1)/4π,
 * by the addition theorem. It depends on bond lengths only, so a linear model can absorb it into
 * its two-body coefficients ("renormalisation of lower body-order coefficients", Sec. IV.2).
 */
export function selfTermPairForm(env: Vec3[], b: Invariant, nfunc: number, rc = GRACE_RCUT): number {
  if (b.n.length !== 2) throw new Error('pair form is for ν = 2');
  const l = b.l[0]!;
  return env.reduce((s, v) => {
    const g = chebBasis(Math.hypot(...v), nfunc, rc);
    return s + (g[b.n[0]! - 1]! * g[b.n[1]! - 1]! * (2 * l + 1)) / (4 * Math.PI);
  }, 0);
}

/** Multiply–adds to evaluate one invariant: explicitly (N^ν tuples) or via the atomic base. */
export function costs(N: number, nu: number, nfunc: number, lmax: number): { explicit: number; distinctOnly: number; ace: number } {
  let falling = 1;
  for (let k = 0; k < nu; k++) falling *= N - k;
  return { explicit: N ** nu, distinctOnly: Math.max(0, falling), ace: N * nfunc * (lmax + 1) ** 2 };
}

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------

/** Rotate a set of vectors by a row-major 3×3 matrix. */
export const rotate = (vs: Vec3[], R: number[]): Vec3[] =>
  vs.map((v) => [R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2], R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2], R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2]]);

/** Neighbour vectors displaced by Gaussian noise of `sigma` Å per component (Box–Muller, seeded). */
export function rattle(vs: Vec3[], sigma: number, seed: number): Vec3[] {
  const rand = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  return vs.map((v) => [v[0] + sigma * gauss(), v[1] + sigma * gauss(), v[2] + sigma * gauss()]);
}

/** Signal on the sphere for radial channel n: ρ_n(r̂) = Σ_lm A_nlm Y_lm(r̂), truncated at L (for plotting). */
export function densityCoefficients(A: Float64Array[], n: number, L: number): Float64Array {
  return A[n - 1]!.slice(0, (L + 1) ** 2);
}
