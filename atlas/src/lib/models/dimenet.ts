// DimeNet: directional message passing (Gasteiger, Groß & Günnemann, ICLR 2020).
// The bases are the paper's (Eqs. 6–8); the toy network is a faithful forward pass with
// fixed random weights (a TOY MODEL: real architecture, untrained weights) used on the
// DimeNet/GemNet page to show what messages on directed edges can see that a
// distance-only network cannot.
//
//   radial basis   ẽ_RBF,n(d) = √(2/c) sin(nπd/c) / d                               (Eq. 7)
//   2D basis       ã_SBF,ln(d, α) = √(2 / (c³ j_{l+1}(z_ln)²)) j_l(z_ln d/c) Y_l⁰(α)    (Eq. 6)
//   envelope       u(x) = 1 − (p+1)(p+2)/2 xᵖ + p(p+2) xᵖ⁺¹ − p(p+1)/2 xᵖ⁺²,  x = d/c (Eq. 8)
//   embedding      m_ji = σ([h_j ‖ h_i ‖ e_RBF^(ji)] W + b)                            (Eq. 9)
//   interaction    m_ji ← f_update(m_ji, Σ_{k ∈ N_j \ {i}} f_int(m_kj, e_RBF, a_SBF^(kj,ji)))  (Eq. 4)
//   output         t_i^(l) = Dense(Σ_j (e_RBF^(ji) W) ∘ m_ji),  t = Σ_i Σ_l t_i^(l)
// Block layout (dense_ji, dense_kj, rbf gate, bilinear layer, residual layers 1 before and
// 2 after the skip) follows the reference implementation gasteigerjo/dimenet
// (dimenet/model/layers/interaction_block.py, output_block.py, embedding_block.py).

export type Vec3 = readonly [number, number, number];

// ---------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------

function doubleFactorialOdd(m: number): number {
  // (2m+1)!! = 1·3·5···(2m+1)
  let p = 1;
  for (let k = 3; k <= 2 * m + 1; k += 2) p *= k;
  return p;
}

/** Spherical Bessel function of the first kind j_l(x), l ≥ 0, x ≥ 0. */
export function sphericalBessel(l: number, x: number): number {
  if (x < Math.max(1, 0.5 * l + 1)) {
    // Power series j_l(x) = xˡ Σ_k (−x²/2)ᵏ / (k! (2l+2k+1)!!); converges quickly for small x.
    let term = x ** l / doubleFactorialOdd(l);
    let sum = term;
    for (let k = 1; k < 40; k++) {
      term *= -(x * x) / (2 * k * (2 * l + 2 * k + 1));
      sum += term;
      if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
    }
    return sum;
  }
  // Upward recurrence j_{l+1} = (2l+1)/x j_l − j_{l−1}; stable for x ≳ l.
  let jm = Math.sin(x) / x;
  if (l === 0) return jm;
  let j = Math.sin(x) / (x * x) - Math.cos(x) / x;
  for (let k = 1; k < l; k++) {
    const next = ((2 * k + 1) / x) * j - jm;
    jm = j;
    j = next;
  }
  return j;
}

const rootCache = new Map<string, number[]>();

/** The first `n` positive roots z_l1 < z_l2 < … of j_l. */
export function besselRoots(l: number, n: number): number[] {
  const key = `${l}:${n}`;
  const hit = rootCache.get(key);
  if (hit) return hit;
  const roots: number[] = [];
  const step = 0.05;
  let a = Math.max(l, 0.5);
  let fa = sphericalBessel(l, a);
  while (roots.length < n) {
    const b = a + step;
    const fb = sphericalBessel(l, b);
    if (fa === 0) roots.push(a);
    else if (fa * fb < 0) {
      let lo = a;
      let hi = b;
      let flo = fa;
      for (let it = 0; it < 80; it++) {
        const mid = 0.5 * (lo + hi);
        const fm = sphericalBessel(l, mid);
        if (flo * fm <= 0) hi = mid;
        else {
          lo = mid;
          flo = fm;
        }
      }
      roots.push(0.5 * (lo + hi));
    }
    a = b;
    fa = fb;
  }
  rootCache.set(key, roots);
  return roots;
}

/** Legendre polynomial P_l(x) by the three-term recurrence. */
export function legendre(l: number, x: number): number {
  if (l === 0) return 1;
  let pm = 1;
  let p = x;
  for (let k = 1; k < l; k++) {
    const next = ((2 * k + 1) * x * p - k * pm) / (k + 1);
    pm = p;
    p = next;
  }
  return p;
}

/** Real spherical harmonic with m = 0: Y_l⁰(α) = √((2l+1)/4π) P_l(cos α). */
export const y0 = (l: number, alpha: number) => Math.sqrt((2 * l + 1) / (4 * Math.PI)) * legendre(l, Math.cos(alpha));

/** Associated Legendre P_l^m(x), m ≥ 0, without the Condon–Shortley phase. */
export function assocLegendre(l: number, m: number, x: number): number {
  let pmm = 1;
  const sx = Math.sqrt(Math.max(0, 1 - x * x));
  for (let k = 1; k <= m; k++) pmm *= (2 * k - 1) * sx;
  if (l === m) return pmm;
  let pm1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pm1;
  for (let k = m + 2; k <= l; k++) {
    const next = ((2 * k - 1) * x * pm1 - (k + m - 1) * pmm) / (k - m);
    pmm = pm1;
    pm1 = next;
  }
  return pm1;
}

function factorialRatio(l: number, m: number): number {
  // (l − m)! / (l + m)!
  let r = 1;
  for (let k = l - m + 1; k <= l + m; k++) r /= k;
  return r;
}

/** Real spherical harmonic Y_lm(polar, azimuth), orthonormal on the sphere; m < 0 uses sin(|m| azimuth). */
export function realY(l: number, m: number, polar: number, azimuth: number): number {
  const am = Math.abs(m);
  const n = Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * factorialRatio(l, am));
  const p = assocLegendre(l, am, Math.cos(polar));
  if (m === 0) return n * p;
  return Math.SQRT2 * n * p * (m > 0 ? Math.cos(am * azimuth) : Math.sin(am * azimuth));
}

// ---------------------------------------------------------------------------
// The paper's bases
// ---------------------------------------------------------------------------

/** Polynomial envelope u(x), Eq. 8, with a root of multiplicity 3 at x = 1; 0 for x ≥ 1. */
export function envelope(x: number, p = 6): number {
  if (x >= 1) return 0;
  return 1 - ((p + 1) * (p + 2) / 2) * x ** p + p * (p + 2) * x ** (p + 1) - (p * (p + 1) / 2) * x ** (p + 2);
}

/** Radial Bessel basis ẽ_RBF,n(d) = √(2/c) sin(nπd/c)/d, n = 1…N (Eq. 7), without envelope. */
export function rbfTilde(n: number, d: number, c: number): number {
  if (d < 1e-12) return Math.sqrt(2 / c) * ((n * Math.PI) / c);
  return (Math.sqrt(2 / c) * Math.sin((n * Math.PI * d) / c)) / d;
}

/** 2D spherical Fourier–Bessel basis ã_SBF,ln(d, α), l ≥ 0, n ≥ 1 (Eq. 6), without envelope. */
export function sbfTilde(l: number, n: number, d: number, alpha: number, c: number): number {
  if (d >= c) return 0;
  const z = besselRoots(l, n)[n - 1]!;
  const norm = Math.sqrt(2 / (c ** 3 * sphericalBessel(l + 1, z) ** 2));
  return norm * sphericalBessel(l, (z * d) / c) * y0(l, alpha);
}

export interface BasisSizes {
  /** N_SHBF: number of spherical harmonics, l = 0 … N_SHBF − 1. */
  nSph: number;
  /** N_SRBF: radial functions per l in the 2D basis. */
  nSrbf: number;
  /** N_RBF: functions in the radial basis. */
  nRbf: number;
}

/** App. B: N_SHBF = 7, N_SRBF = N_RBF = 6, c = 5 Å, p = 6. */
export const PAPER_BASIS: BasisSizes = { nSph: 7, nSrbf: 6, nRbf: 6 };
export const PAPER_CUTOFF = 5;

export function rbfVector(d: number, c: number, nRbf: number, p = 6): Float64Array {
  const u = envelope(d / c, p);
  return Float64Array.from({ length: nRbf }, (_, k) => u * rbfTilde(k + 1, d, c));
}

export function sbfVector(d: number, alpha: number, c: number, s: BasisSizes, p = 6): Float64Array {
  const u = envelope(d / c, p);
  const out = new Float64Array(s.nSph * s.nSrbf);
  for (let l = 0; l < s.nSph; l++)
    for (let n = 1; n <= s.nSrbf; n++) out[l * s.nSrbf + n - 1] = u * sbfTilde(l, n, d, alpha, c);
  return out;
}

// ---------------------------------------------------------------------------
// Toy DimeNet with fixed random weights
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Swish σ(x) = x · sigmoid(x), DimeNet's activation (Sec. 6). */
export const swish = (x: number) => x / (1 + Math.exp(-x));

interface Dense {
  w: Float64Array; // out × in, row-major
  b: Float64Array;
  nIn: number;
  nOut: number;
}

function dense(rand: () => number, nIn: number, nOut: number, bias = true): Dense {
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
    y[o] = act ? swish(s) : s;
  }
  return y;
}

/** ResidualLayer of the reference code: x + σ(W₂ σ(W₁ x + b₁) + b₂). */
function residual(r: [Dense, Dense], x: ArrayLike<number>): Float64Array {
  const y = apply(r[1], apply(r[0], x, true), true);
  return y.map((v, f) => v + x[f]!);
}

interface Interaction {
  rbf: Dense; // N_RBF → F, no bias
  sbf: Dense; // N_SHBF·N_SRBF → N_bilinear, no bias
  ji: Dense;
  kj: Dense;
  bilinear: Float64Array; // F × N_bilinear × F
  before: [Dense, Dense][];
  finalBefore: Dense;
  after: [Dense, Dense][];
}

interface Output {
  rbf: Dense; // N_RBF → F, no bias
  layers: Dense[];
  final: Dense; // F → 1, no bias
}

export interface ToyDimeNet {
  features: number;
  nBilinear: number;
  basis: BasisSizes;
  rc: number;
  p: number;
  embedding: Map<number, Float64Array>;
  embRbf: Dense;
  embDense: Dense;
  blocks: Interaction[];
  outputs: Output[]; // one per block plus one for the embedding block
  /** Present only in the toy GemNet-Q: one two-hop (quadruplet) term per block. */
  quad?: Quadruplet[];
}

interface Quadruplet {
  sbf: Dense; // N_SHBF²·N_SRBF → N_bilinear, no bias
  cbf: Dense; // N_SHBF·N_SRBF → F, no bias
  rbf: Dense; // N_RBF → F, no bias
  bilinear: Float64Array; // F × N_bilinear × F
}

export interface ToyDimeNetOptions {
  rc: number;
  features?: number;
  blocks?: number;
  nBilinear?: number;
  basis?: BasisSizes;
  seed?: number;
  elements?: number[];
}

export function makeToyDimeNet(o: ToyDimeNetOptions): ToyDimeNet {
  const F = o.features ?? 16;
  const nB = o.nBilinear ?? 8;
  const basis = o.basis ?? PAPER_BASIS;
  const rand = mulberry32(o.seed ?? 2020);
  const embedding = new Map<number, Float64Array>();
  for (const z of o.elements ?? [1, 6, 7, 8, 14]) embedding.set(z, new Float64Array(F).map(() => (2 * rand() - 1) * Math.sqrt(3)));
  const embRbf = dense(rand, basis.nRbf, F);
  const embDense = dense(rand, 3 * F, F);
  const out = (): Output => ({ rbf: dense(rand, basis.nRbf, F, false), layers: [dense(rand, F, F), dense(rand, F, F), dense(rand, F, F)], final: dense(rand, F, 1, false) });
  const res = (): [Dense, Dense] => [dense(rand, F, F), dense(rand, F, F)];
  const blocks: Interaction[] = Array.from({ length: o.blocks ?? 3 }, () => ({
    rbf: dense(rand, basis.nRbf, F, false),
    sbf: dense(rand, basis.nSph * basis.nSrbf, nB, false),
    ji: dense(rand, F, F),
    kj: dense(rand, F, F),
    bilinear: new Float64Array(F * nB * F).map(() => (2 * rand() - 1) * (2 / F) * Math.sqrt(3)),
    before: [res()],
    finalBefore: dense(rand, F, F),
    after: [res(), res()],
  }));
  const outputs = Array.from({ length: blocks.length + 1 }, out);
  return { features: F, nBilinear: nB, basis, rc: o.rc, p: 6, embedding, embRbf, embDense, blocks, outputs };
}

/**
 * The same toy network plus, in every block, a two-hop update in the form of GemNet's
 * Eq. (11) added to the triplet aggregate. Everything else (embedding, triplet term, output
 * blocks) is shared weight-for-weight with makeToyDimeNet for the same seed, so the two
 * differ only by the quadruplet term. Not the full GemNet block: no symmetric message
 * passing, atom self-interaction or variance-scaling factors, and one cutoff for both
 * the embedding and the interaction graph.
 */
export function makeToyGemNetQ(o: ToyDimeNetOptions): ToyDimeNet {
  const base = makeToyDimeNet(o);
  const F = base.features;
  const nB = base.nBilinear;
  const { nSph, nSrbf, nRbf } = base.basis;
  const rand = mulberry32((o.seed ?? 2020) + 2021);
  base.quad = base.blocks.map(() => ({
    sbf: dense(rand, nSph * nSph * nSrbf, nB, false),
    cbf: dense(rand, nSph * nSrbf, F, false),
    rbf: dense(rand, nRbf, F, false),
    bilinear: new Float64Array(F * nB * F).map(() => (2 * rand() - 1) * (2 / F) * Math.sqrt(3)),
  }));
  return base;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export interface Edge {
  /** Message m_ji goes from j (source) to i (target). */
  j: number;
  i: number;
  d: number;
}

export interface Triplet {
  /** Index of edge k→j (incoming message m_kj). */
  kj: number;
  /** Index of edge j→i (message m_ji being updated). */
  ji: number;
  /** α_(kj,ji) = ∠ x_k x_j x_i, radians. */
  alpha: number;
}

/** Directed edges within the cutoff and the triplets k → j → i with k ≠ i (Eq. 4). */
export function graph(pos: readonly Vec3[], rc: number): { edges: Edge[]; triplets: Triplet[] } {
  const edges: Edge[] = [];
  const n = pos.length;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      if (i === j) continue;
      const d = len(sub(pos[i]!, pos[j]!));
      if (d < rc) edges.push({ j, i, d });
    }
  const into = new Map<number, number[]>(); // atom → indices of edges whose target is that atom
  edges.forEach((e, idx) => {
    const list = into.get(e.i) ?? [];
    list.push(idx);
    into.set(e.i, list);
  });
  const triplets: Triplet[] = [];
  edges.forEach((e, ji) => {
    for (const kj of into.get(e.j) ?? []) {
      const k = edges[kj]!.j;
      if (k === e.i) continue;
      const a = sub(pos[k]!, pos[e.j]!);
      const b = sub(pos[e.i]!, pos[e.j]!);
      const c = Math.min(1, Math.max(-1, dot(a, b) / (len(a) * len(b))));
      triplets.push({ kj, ji, alpha: Math.acos(c) });
    }
  });
  return { edges, triplets };
}

export interface Quad {
  /** Edge c→a (message m_ca, updated). */
  ca: number;
  /** Edge d→b (message m_db, read). */
  db: number;
  /** Distances x_ca, x_ba, x_db (Å). */
  xca: number;
  xba: number;
  xdb: number;
  /** φ_cab: angle at a between c and b. φ_abd: angle at b between a and d. */
  phiCab: number;
  phiAbd: number;
  /** θ_cabd: unsigned dihedral between x_ca and x_db about the b–a axis, radians. */
  theta: number;
}

/** Unsigned dihedral between u and v about the axis `ax`; 0 when either projection vanishes. */
export function dihedral(u: Vec3, v: Vec3, ax: Vec3): number {
  const a = len(ax);
  const e: Vec3 = [ax[0] / a, ax[1] / a, ax[2] / a];
  const pu = sub(u, [e[0] * dot(u, e), e[1] * dot(u, e), e[2] * dot(u, e)]);
  const pv = sub(v, [e[0] * dot(v, e), e[1] * dot(v, e), e[2] * dot(v, e)]);
  const nu = len(pu);
  const nv = len(pv);
  if (nu < 1e-12 || nv < 1e-12) return 0;
  return Math.acos(Math.min(1, Math.max(-1, dot(pu, pv) / (nu * nv))));
}

const angleBetween = (u: Vec3, v: Vec3) => Math.acos(Math.min(1, Math.max(-1, dot(u, v) / (len(u) * len(v)))));

/**
 * Quadruplets c → a ← b ← d of GemNet's Eq. (11): b ∈ N_a \ {c}, d ∈ N_b \ {a, c}, all four
 * atoms distinct, one cutoff for every edge.
 */
export function quadruplets(pos: readonly Vec3[], edges: Edge[]): Quad[] {
  const byPair = new Map<string, number>();
  edges.forEach((e, k) => byPair.set(`${e.j}>${e.i}`, k));
  const into = new Map<number, number[]>();
  edges.forEach((e, k) => {
    const list = into.get(e.i) ?? [];
    list.push(k);
    into.set(e.i, list);
  });
  const out: Quad[] = [];
  edges.forEach((e, ca) => {
    const c = e.j;
    const a = e.i;
    for (const ba of into.get(a) ?? []) {
      const b = edges[ba]!.j;
      if (b === c) continue;
      for (const db of into.get(b) ?? []) {
        const d = edges[db]!.j;
        if (d === a || d === c) continue;
        const xca = sub(pos[c]!, pos[a]!);
        const xba = sub(pos[b]!, pos[a]!);
        const xab = sub(pos[a]!, pos[b]!);
        const xdb = sub(pos[d]!, pos[b]!);
        out.push({
          ca,
          db,
          xca: len(xca),
          xba: len(xba),
          xdb: len(xdb),
          phiCab: angleBetween(xca, xba),
          phiAbd: angleBetween(xab, xdb),
          theta: dihedral(xca, xdb, xba),
        });
      }
    }
  });
  return out;
}

/** GemNet's 3D spherical Fourier–Bessel basis ẽ_SBF,lmn (Eq. 10), times the envelope, flattened over (l, m, n). */
export function sbf3Vector(x: number, polar: number, azimuth: number, c: number, s: BasisSizes, p = 6): Float64Array {
  const u = envelope(x / c, p);
  const out = new Float64Array(s.nSph * s.nSph * s.nSrbf);
  let k = 0;
  for (let l = 0; l < s.nSph; l++)
    for (let m = -l; m <= l; m++)
      for (let n = 1; n <= s.nSrbf; n++) {
        if (x < c) {
          const z = besselRoots(l, n)[n - 1]!;
          const norm = Math.sqrt(2 / (c ** 3 * sphericalBessel(l + 1, z) ** 2));
          out[k] = u * norm * sphericalBessel(l, (z * x) / c) * realY(l, m, polar, azimuth);
        }
        k++;
      }
  return out;
}

export interface DimeNetPass {
  edges: Edge[];
  triplets: Triplet[];
  /** messages[l][e]: m_e after l interaction blocks (l = 0 is the embedding block). */
  messages: Float64Array[][];
  /** atomOut[l][i] = t_i^(l), the output block's scalar for atom i after block l. */
  atomOut: number[][];
  /** atomFeat[l][i]: the summed, RBF-gated messages Σ_j (e_RBF W) ∘ m_ji before the dense layers. */
  atomFeat: Float64Array[][];
  /** Σ_i Σ_l t_i^(l). */
  energy: number;
  /** Quadruplets used (empty for DimeNet). */
  quads: Quad[];
}

export function forward(model: ToyDimeNet, Z: readonly number[], pos: readonly Vec3[]): DimeNetPass {
  const F = model.features;
  const { edges, triplets } = graph(pos, model.rc);
  const rbf = edges.map((e) => rbfVector(e.d, model.rc, model.basis.nRbf, model.p));
  const sbf = triplets.map((t) => sbfVector(edges[t.kj]!.d, t.alpha, model.rc, model.basis, model.p));
  const quads = model.quad ? quadruplets(pos, edges) : [];
  const qBasis = quads.map((q) => ({
    sbf: sbf3Vector(q.xca, q.phiCab, q.theta, model.rc, model.basis, model.p),
    cbf: sbfVector(q.xba, q.phiAbd, model.rc, model.basis, model.p),
    rbf: rbfVector(q.xdb, model.rc, model.basis.nRbf, model.p),
  }));
  const emb = (z: number) => {
    const e = model.embedding.get(z);
    if (!e) throw new Error(`no embedding for Z = ${z}`);
    return e;
  };
  let m = edges.map((e, k) => {
    const g = apply(model.embRbf, rbf[k]!, true);
    const cat = new Float64Array(3 * F);
    cat.set(emb(Z[e.j]!), 0);
    cat.set(emb(Z[e.i]!), F);
    cat.set(g, 2 * F);
    return apply(model.embDense, cat, true);
  });
  const messages = [m];
  const atomOut: number[][] = [];
  const atomFeat: Float64Array[][] = [];
  const readout = (o: Output, msg: Float64Array[]) => {
    const h = pos.map(() => new Float64Array(F));
    edges.forEach((e, k) => {
      const g = apply(o.rbf, rbf[k]!);
      const hi = h[e.i]!;
      const mk = msg[k]!;
      for (let f = 0; f < F; f++) hi[f]! += g[f]! * mk[f]!;
    });
    atomFeat.push(h.map((v) => Float64Array.from(v)));
    atomOut.push(h.map((v) => apply(o.final, o.layers.reduce((x, L) => apply(L, x, true), v as Float64Array))[0]!));
  };
  readout(model.outputs[0]!, m);
  model.blocks.forEach((b, t) => {
    const xji = m.map((v) => apply(b.ji, v, true));
    const xkj = m.map((v, k) => {
      const y = apply(b.kj, v, true);
      const g = apply(b.rbf, rbf[k]!);
      return y.map((a, f) => a * g[f]!);
    });
    const agg = m.map(() => new Float64Array(F));
    const nB = model.nBilinear;
    triplets.forEach((tr, q) => {
      const s = apply(b.sbf, sbf[q]!);
      const x = xkj[tr.kj]!;
      bilinearInto(agg[tr.ji]!, s, x, b.bilinear, F, nB);
    });
    const qw = model.quad?.[t];
    if (qw)
      quads.forEach((q, n) => {
        // Eq. (11): (W_SBF e_SBF)ᵀ W ((W_CBF e_CBF) ⊙ (W_RBF e_RBF) ⊙ m_db), summed into m̃_ca.
        const s = apply(qw.sbf, qBasis[n]!.sbf);
        const cg = apply(qw.cbf, qBasis[n]!.cbf);
        const rg = apply(qw.rbf, qBasis[n]!.rbf);
        const x = m[q.db]!.map((v, f) => v * cg[f]! * rg[f]!);
        bilinearInto(agg[q.ca]!, s, x, qw.bilinear, F, nB);
      });
    m = m.map((v, k) => {
      let x2: Float64Array = xji[k]!.map((a, f) => a + agg[k]![f]!);
      for (const r of b.before) x2 = residual(r, x2);
      x2 = apply(b.finalBefore, x2, true);
      let x: Float64Array = v.map((a, f) => a + x2[f]!);
      for (const r of b.after) x = residual(r, x);
      return x;
    });
    messages.push(m);
    readout(model.outputs[t + 1]!, m);
  });
  const energy = atomOut.reduce((s, row) => s + row.reduce((a, v) => a + v, 0), 0);
  return { edges, triplets, messages, atomOut, atomFeat, energy, quads };
}

/** einsum("wj,wl,ijl->wi", s, x, W) for one row w: out_i += Σ_j Σ_l s_j x_l W[i, j, l]. */
function bilinearInto(out: Float64Array, s: Float64Array, x: ArrayLike<number>, W: Float64Array, F: number, nB: number) {
  for (let i = 0; i < F; i++) {
    let acc = 0;
    for (let j = 0; j < nB; j++) {
      const base = (i * nB + j) * F;
      let inner = 0;
      for (let l = 0; l < F; l++) inner += x[l]! * W[base + l]!;
      acc += s[j]! * inner;
    }
    out[i]! += acc;
  }
}

// ---------------------------------------------------------------------------
// The paper's indistinguishable pair (App. A, Fig. 6): a hexagon vs two triangles
// ---------------------------------------------------------------------------

/** Regular planar n-gon of side d centred at `centre`, in the xy plane. */
export function ring(n: number, d: number, centre: Vec3 = [0, 0, 0], phase = 0): Vec3[] {
  const R = d / (2 * Math.sin(Math.PI / n));
  return Array.from({ length: n }, (_, k) => {
    const t = phase + (2 * Math.PI * k) / n;
    return [centre[0] + R * Math.cos(t), centre[1] + R * Math.sin(t), centre[2]] as Vec3;
  });
}

/** Six atoms: one hexagon of side d (left) and, far apart, two triangles of side d. */
export function hexagonAndTriangles(d: number, gap = 40): { hexagon: Vec3[]; triangles: Vec3[] } {
  return {
    hexagon: ring(6, d),
    triangles: [...ring(3, d, [0, 0, 0], Math.PI / 2), ...ring(3, d, [gap, 0, 0], Math.PI / 2)],
  };
}

/** Second-neighbour distance of the hexagon, d√3: the cutoff below which distance-only networks see two identical graphs. */
export const hexagonSecondNeighbour = (d: number) => d * Math.sqrt(3);

const norm = (v: ArrayLike<number>) => Math.sqrt(Array.from(v).reduce((s, a) => s + a * a, 0));

/** ‖a − b‖ / ‖b‖ for feature vectors. */
export function relDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const nb = norm(b);
  const diff = Array.from(a, (v, k) => v - b[k]!);
  return nb === 0 ? norm(diff) : norm(diff) / nb;
}

// ---------------------------------------------------------------------------
// A torsion: H–O–O–H with an adjustable dihedral
// ---------------------------------------------------------------------------

/** Illustrative peroxide geometry (Å, degrees); close to, not fitted to, the gas-phase molecule. */
export const PEROXIDE = { rOO: 1.45, rOH: 0.97, angle: 100 } as const;

/** Atoms [H₁, O₁, O₂, H₂]; O–O along x, H–O–O–H dihedral `dihedralDeg` (0 = cis). */
export function peroxide(dihedralDeg: number, g: { rOO: number; rOH: number; angle: number } = PEROXIDE): Vec3[] {
  const a = (g.angle * Math.PI) / 180;
  const t = (dihedralDeg * Math.PI) / 180;
  const along = -g.rOH * Math.cos(a); // H's projection onto the O–O axis, measured outward
  const perp = g.rOH * Math.sin(a);
  return [
    [-along, perp, 0],
    [0, 0, 0],
    [g.rOO, 0, 0],
    [g.rOO + along, perp * Math.cos(t), perp * Math.sin(t)],
  ];
}

const d3 = (a: Vec3, b: Vec3) => len(sub(a, b));

/** The pair distances of the peroxide chain: bonds, the two 1–3 distances, and the 1–4 (H···H) distance. */
export function peroxideDistances(dihedralDeg: number, g = PEROXIDE) {
  const p = peroxide(dihedralDeg, g);
  return { oh: g.rOH, oo: g.rOO, hOfar: d3(p[0]!, p[2]!), hh: d3(p[0]!, p[3]!) };
}

export interface TorsionScan {
  dihedrals: number[];
  /** Relative change of the energy-relevant features (all atoms' summed outputs per block) vs the reference dihedral. */
  dime: number[];
  gem: number[];
}

/** Features compared: every atom's RBF-gated message sum after the last block, concatenated. */
function torsionFeatures(pass: DimeNetPass): number[] {
  const last = pass.atomFeat[pass.atomFeat.length - 1]!;
  return last.flatMap((v) => Array.from(v));
}

/** The compared feature vector of the peroxide chain at one dihedral. */
export const peroxideFeatures = (model: ToyDimeNet, dihedralDeg: number) => torsionFeatures(forward(model, [1, 8, 8, 1], peroxide(dihedralDeg)));

export function torsionScan(dime: ToyDimeNet, gem: ToyDimeNet, dihedrals: number[], refDeg: number): TorsionScan {
  const Z = [1, 8, 8, 1];
  const refD = torsionFeatures(forward(dime, Z, peroxide(refDeg)));
  const refG = torsionFeatures(forward(gem, Z, peroxide(refDeg)));
  const out: TorsionScan = { dihedrals, dime: [], gem: [] };
  for (const t of dihedrals) {
    const p = peroxide(t);
    out.dime.push(relDiff(torsionFeatures(forward(dime, Z, p)), refD));
    out.gem.push(relDiff(torsionFeatures(forward(gem, Z, p)), refG));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Cost: edges, triplets and quadruplets per atom in a crystal
// ---------------------------------------------------------------------------

export interface InteractionCounts {
  /** Directed edges into the atom (embedding neighbours), k_emb. */
  edges: number;
  /** Triplets k → j → i updating a message into the atom: Σ_j (k_emb(j) − 1). */
  triplets: number;
  /** Quadruplets updating a message into the atom (GemNet Eq. 11, c ∈ N_a^emb, b ∈ N_a^int). */
  quadruplets: number;
}

/**
 * Exact counts for atom `i` of a periodic structure in which every atom has the same
 * number of embedding neighbours (true for fcc Cu and diamond Si by symmetry), so that
 * |N_b^emb| = k_emb for every b.
 */
export function interactionCounts(neighbours: (cutoff: number) => Vec3[], cEmb: number, cInt = cEmb): InteractionCounts {
  const emb = neighbours(cEmb);
  const int = neighbours(cInt);
  const k = emb.length;
  let quads = 0;
  for (const c of emb)
    for (const b of int) {
      if (len(sub(c, b)) < 1e-9) continue;
      const aInNb = len(b) < cEmb ? 1 : 0; // a ∈ N_b^emb
      const cInNb = len(sub(c, b)) < cEmb ? 1 : 0; // c ∈ N_b^emb
      quads += k - aInNb - cInNb;
    }
  return { edges: k, triplets: k * (k - 1), quadruplets: quads };
}

/** Upper bounds with GemNet-OC's neighbour limits (k_emb ≤ 30, k_qint ≤ 8; Table 3). */
export const GEMNET_OC_LIMITS = { emb: 30, qint: 8 } as const;
export const gemnetOcBounds = (kEmb = GEMNET_OC_LIMITS.emb, kQint = GEMNET_OC_LIMITS.qint): InteractionCounts => ({
  edges: kEmb,
  triplets: kEmb * (kEmb - 1),
  quadruplets: kEmb * kQint * (kEmb - 1),
});
