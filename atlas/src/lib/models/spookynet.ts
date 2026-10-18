// SpookyNet's building blocks (Unke et al., Nat. Commun. 12, 7273 (2021), Methods) and a small,
// faithful SpookyNet with fixed random weights (a TOY MODEL: the functional form follows the
// paper, the weights are not trained; no trained weights were released).
//
//   basis        ρ_k(r) = b_{k,K−1}(e^{−γr}) f_cut(r)                       (Eqs. 14–16)
//                Y_l^m without the √(4π/(2l+1)) normalisation                 (Eq. 17)
//   embeddings   x⁽⁰⁾ = e_Z + e_Q + e_S, e_Z = M d_Z + ẽ_Z                     (Eqs. 1, 9)
//                a_i = Ψ softplus(q_i·k/√F) / Σ_j softplus(q_j·k/√F)          (Eq. 10)
//   module       x̃ = residual₁(x), x = residual₂(x̃ + l + n), y = resmlp(x)    (Eq. 11)
//   local        s, p (l = 1), d (l = 2) sums over neighbours, ⟨P₁p, P₂p⟩, ⟨D₁d, D₂d⟩ (Eq. 12)
//   nonlocal     self-attention over all atoms, exact (Eq. 19) or FAVOR+ (Eq. 20)
//   outputs      E_i = w_E·f_i + Ẽ_Z, q_i with the uniform 1/N correction    (Eqs. 21, 24)
//   electrostatics with the switched (r² + 1)^{−1/2} kernel                    (Eqs. 23, 25)
// Simplifications (stated on the page): F = 12 features, K = 8 radial functions, one
// residual block per resmlp, silu with its initial α = 1, β = 1.702, no ZBL or D4 term.
// Lengths in Å; the paper's γ and r_cut are in bohr and converted here.

import { mulberry32 } from '../chem/rng';

export type Vec3 = [number, number, number];

export const BOHR = 0.529177210903;
/** r_cut = 10 a₀ (Methods, Training and hyperparameters). */
export const RC_PAPER = 10 * BOHR;
/** γ is initialised to ½ a₀⁻¹ (Methods). */
export const GAMMA_INIT = 0.5 / BOHR;
/** Coulomb constant in eV Å e⁻². */
export const K_E = 14.399645;

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.sqrt(dot(a, a));

// ---------------------------------------------------------------------------
// Basis functions (Eqs. 13–17)
// ---------------------------------------------------------------------------

/** f_cut(r) = exp(−r²/((r_c − r)(r_c + r))) for r < r_c, else 0 (Eq. 16). */
export function fcut(r: number, rc = RC_PAPER): number {
  if (r >= rc) return 0;
  return Math.exp((-r * r) / ((rc - r) * (rc + r)));
}

export function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let c = 1;
  for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i;
  return c;
}

const factorial = (n: number) => {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
};

/** Bernstein polynomial b_{k,n}(x) = C(n,k) x^k (1 − x)^{n−k} (Eq. 15, n = K − 1). */
export const bernstein = (k: number, n: number, x: number) => binom(n, k) * x ** k * (1 - x) ** (n - k);

/** ρ_k(r) = b_{k,K−1}(e^{−γ r}) f_cut(r), k = 0 … K − 1 (Eq. 14). */
export function radialBasis(r: number, K: number, gamma = GAMMA_INIT, rc = RC_PAPER): number[] {
  const x = Math.exp(-gamma * r);
  const c = fcut(r, rc);
  return Array.from({ length: K }, (_, k) => bernstein(k, K - 1, x) * c);
}

/** Π_l^m(z) of Eq. (17) for a unit vector (r = 1). */
function piLM(l: number, m: number, z: number): number {
  let s = 0;
  for (let p = 0; p <= Math.floor((l - m) / 2); p++) {
    const c = ((-1) ** p / 2 ** l) * binom(l, p) * binom(2 * l - 2 * p, l) * (factorial(l - 2 * p) / factorial(l - 2 * p - m));
    s += c * z ** (l - 2 * p - m);
  }
  return Math.sqrt(factorial(l - m) / factorial(l + m)) * s;
}

const aM = (m: number, x: number, y: number) => {
  let s = 0;
  for (let p = 0; p <= m; p++) s += binom(m, p) * x ** p * y ** (m - p) * Math.sin((Math.PI / 2) * (m - p));
  return s;
};
const bM = (m: number, x: number, y: number) => {
  let s = 0;
  for (let p = 0; p <= m; p++) s += binom(m, p) * x ** p * y ** (m - p) * Math.cos((Math.PI / 2) * (m - p));
  return s;
};

/**
 * Real spherical harmonic Y_l^m of the direction of v, exactly as Eq. (17): the standard real
 * harmonics without the factor √(4π/(2l + 1)), so Σ_m Y_l^m(u) Y_l^m(v) = P_l(û·v̂).
 */
export function realY(l: number, m: number, v: Vec3): number {
  const r = norm(v);
  const [x, y, z] = [v[0] / r, v[1] / r, v[2] / r];
  const am = Math.abs(m);
  if (m === 0) return piLM(l, 0, z);
  const pi = piLM(l, am, z);
  return Math.SQRT2 * pi * (m < 0 ? aM(am, x, y) : bM(am, x, y));
}

/** All 2l + 1 components, m = −l … l. */
export const shBlock = (l: number, v: Vec3) => Array.from({ length: 2 * l + 1 }, (_, i) => realY(l, i - l, v));

/** Angular power-spectrum invariant a_l = Σ_m (Σ_j w_j Y_l^m(r̂_j))² (Supplementary Eq. 2). */
export function powerSpectrum(vectors: Vec3[], l: number, weights?: number[]): number {
  let a = 0;
  for (let m = -l; m <= l; m++) {
    let s = 0;
    vectors.forEach((v, j) => (s += (weights ? weights[j]! : 1) * realY(l, m, v)));
    a += s * s;
  }
  return a;
}

/** Legendre polynomial P_l(x) by recursion. */
export function legendre(l: number, x: number): number {
  if (l === 0) return 1;
  let p0 = 1;
  let p1 = x;
  for (let n = 1; n < l; n++) [p0, p1] = [p1, ((2 * n + 1) * x * p1 - n * p0) / (n + 1)];
  return p1;
}

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

export interface Molecule {
  numbers: number[];
  positions: Vec3[];
}

/** Polar angle (degrees) at which the antiprism (τ = 60°) is a regular octahedron. */
export const MAGIC_THETA = (Math.acos(1 / Math.sqrt(3)) * 180) / Math.PI;

/**
 * ML₆ with all six ligands at distance `bond`: three at polar angle θ and azimuths τ, τ + 120°,
 * τ + 240°, three at π − θ and azimuths 0°, 120°, 240°. τ = 0 is a trigonal prism, τ = 60° a
 * trigonal antiprism (an octahedron when θ = MAGIC_THETA). Atom 0 is the metal.
 */
export function trigonalTwist(tauDeg: number, thetaDeg = MAGIC_THETA, bond = 2.0, metal = 26, ligand = 8): Molecule {
  const t = (thetaDeg * Math.PI) / 180;
  const tau = (tauDeg * Math.PI) / 180;
  const positions: Vec3[] = [[0, 0, 0]];
  for (const [polar, shift] of [
    [t, tau],
    [Math.PI - t, 0],
  ] as const) {
    for (let k = 0; k < 3; k++) {
      const phi = shift + (2 * Math.PI * k) / 3;
      positions.push([bond * Math.sin(polar) * Math.cos(phi), bond * Math.sin(polar) * Math.sin(phi), bond * Math.cos(polar)]);
    }
  }
  return { numbers: [metal, ...Array(6).fill(ligand)], positions };
}

/** Benzene in the xy plane, centred at c (C–C 1.397 Å, C–H 1.084 Å). */
export function benzene(c: Vec3 = [0, 0, 0]): Molecule {
  const numbers: number[] = [];
  const positions: Vec3[] = [];
  for (const [Z, r] of [
    [6, 1.397],
    [1, 1.397 + 1.084],
  ] as const) {
    for (let k = 0; k < 6; k++) {
      const a = (Math.PI / 3) * k;
      numbers.push(Z);
      positions.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]]);
    }
  }
  return { numbers, positions };
}

export type PairKind = 'LiF' | 'NaCl' | 'HF' | 'benzene-water' | 'benzene-benzene';

export const PAIRS: Record<PairKind, { label: string; nA: number; dMin: number }> = {
  LiF: { label: 'LiF', nA: 1, dMin: 1.2 },
  NaCl: { label: 'NaCl', nA: 1, dMin: 1.8 },
  HF: { label: 'HF', nA: 1, dMin: 0.7 },
  'benzene-water': { label: 'C₆H₆ + H₂O', nA: 12, dMin: 3.0 },
  'benzene-benzene': { label: 'C₆H₆ + C₆H₆', nA: 12, dMin: 3.4 },
};

/** Fragment A (atoms 0 … nA − 1) at the origin, fragment B at separation d (along x for diatomics, along the ring axis z otherwise). */
export function fragmentPair(kind: PairKind, d: number): Molecule {
  const di: Record<string, [number, number]> = { LiF: [3, 9], NaCl: [11, 17], HF: [1, 9] };
  if (kind in di) {
    const [a, b] = di[kind]!;
    return { numbers: [a, b], positions: [[0, 0, 0], [d, 0, 0]] };
  }
  const A = benzene();
  if (kind === 'benzene-benzene') {
    // face to face along the ring axis (z), d = ring–ring distance
    const B = benzene([0, 0, d]);
    return { numbers: [...A.numbers, ...B.numbers], positions: [...A.positions, ...B.positions] };
  }
  // water above the ring centre, O at height d, one O–H pointing straight down at the ring
  const al = (104.52 * Math.PI) / 180;
  const r = 0.9572;
  const B: Vec3[] = [
    [0, 0, d],
    [0, 0, d - r],
    [r * Math.sin(al), 0, d - r * Math.cos(al)],
  ];
  return { numbers: [...A.numbers, 8, 1, 1], positions: [...A.positions, ...B] };
}

export function subset(m: Molecule, idx: number[]): Molecule {
  return { numbers: idx.map((i) => m.numbers[i]!), positions: idx.map((i) => m.positions[i]!) };
}

// ---------------------------------------------------------------------------
// Element descriptors d_Z (Eq. 9; element descriptor table of the Supplementary Information): ground-state configuration,
// columns Z 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p vs vp vd vf, divided by the Rn row.
// ---------------------------------------------------------------------------

const CONFIG: Record<number, number[]> = {
  1: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  3: [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  6: [6, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
  7: [7, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0],
  8: [8, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
  9: [9, 2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],
  11: [11, 2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  16: [16, 2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
  17: [17, 2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],
  26: [26, 2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0],
};
export const RN_ROW = [86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14];
export const ELEMENTS = Object.keys(CONFIG).map(Number);
export const SYMBOL: Record<number, string> = { 1: 'H', 3: 'Li', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 16: 'S', 17: 'Cl', 26: 'Fe' };

export function elementDescriptor(Z: number): number[] {
  const row = CONFIG[Z];
  if (!row) throw new Error(`no electron configuration for Z = ${Z}`);
  return row.map((v, i) => v / RN_ROW[i]!);
}

// ---------------------------------------------------------------------------
// Dense layers (Eqs. 5–8)
// ---------------------------------------------------------------------------

interface Lin {
  W: number[][];
  b: number[] | null;
}

function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-300);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const randVec = (rand: () => number, n: number, s: number) => Array.from({ length: n }, () => s * gaussian(rand));

function lin(rand: () => number, nOut: number, nIn: number, bias = true, gain = 1): Lin {
  const s = gain / Math.sqrt(nIn);
  return { W: Array.from({ length: nOut }, () => randVec(rand, nIn, s)), b: bias ? randVec(rand, nOut, 0.1) : null };
}

function applyLin(L: Lin, x: number[]): number[] {
  return L.W.map((row, o) => {
    let s = L.b ? L.b[o]! : 0;
    for (let i = 0; i < row.length; i++) s += row[i]! * x[i]!;
    return s;
  });
}

/** Generalised SiLU αx/(1 + e^{−βx}) at its initial values α = 1, β = 1.702 (Eq. 6; Methods). */
export const silu = (x: number) => x / (1 + Math.exp(-1.702 * x));
const siluV = (x: number[]) => x.map(silu);
const add = (a: number[], b: number[]) => a.map((v, i) => v + b[i]!);

interface Residual {
  l1: Lin;
  l2: Lin;
}
interface ResMLP {
  res: Residual;
  out: Lin;
}

const residual = (rand: () => number, F: number, bias = true): Residual => ({ l1: lin(rand, F, F, bias), l2: lin(rand, F, F, bias, 0.5) });
const resmlp = (rand: () => number, F: number, bias = true, gain = 1): ResMLP => ({ res: residual(rand, F, bias), out: lin(rand, F, F, bias, gain) });

/** residual(x) = x + linear₂(silu(linear₁(silu(x)))) (Eq. 7). */
const applyResidual = (r: Residual, x: number[]) => add(x, applyLin(r.l2, siluV(applyLin(r.l1, siluV(x)))));
/** resmlp(x) = linear(silu(residual(x))) (Eq. 8). */
const applyResMLP = (m: ResMLP, x: number[]) => applyLin(m.out, siluV(applyResidual(m.res, x)));

// ---------------------------------------------------------------------------
// Attention (Eqs. 18–20)
// ---------------------------------------------------------------------------

export type Matrix = number[][];

/** Exact attention D⁻¹AV with A = exp(QKᵀ/√F) (Eq. 19). Also returns the row-normalised A. */
export function attentionExact(Q: Matrix, K: Matrix, V: Matrix): { out: Matrix; weights: Matrix } {
  const F = Q[0]!.length;
  const weights = Q.map((q) => {
    const logits = K.map((k) => dot3(q, k) / Math.sqrt(F));
    const mx = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
  return { out: weights.map((w) => V[0]!.map((_, c) => w.reduce((s, wj, j) => s + wj * V[j]![c]!, 0))), weights };
}

function dot3(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/**
 * m × d random projection with orthogonal blocks whose rows are rescaled to χ_d-distributed
 * norms, as in the reference implementation (Attention._omega) after Choromanski et al.
 */
export function drawOmega(seed: number, m: number, d: number, orthogonal = true): Matrix {
  const rand = mulberry32(seed);
  const rows: Matrix = [];
  while (rows.length < m) {
    const block = Array.from({ length: d }, () => randVec(rand, d, 1));
    if (orthogonal) {
      // Gram–Schmidt
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < i; j++) {
          const p = dot3(block[i]!, block[j]!);
          block[i] = block[i]!.map((v, c) => v - p * block[j]![c]!);
        }
        const n = Math.sqrt(dot3(block[i]!, block[i]!));
        block[i] = block[i]!.map((v) => v / n);
      }
      for (const row of block) {
        if (rows.length >= m) break;
        const n = Math.sqrt(randVec(rand, d, 1).reduce((s, v) => s + v * v, 0));
        rows.push(row.map((v) => v * n));
      }
    } else {
      for (const row of block) if (rows.length < m) rows.push(row);
    }
  }
  return rows;
}

export type FeatureKind = 'positive' | 'trig';

/**
 * Random-feature map φ with E[φ(q)·φ(k)] = exp(q·k/√d).
 * positive (FAVOR+): φ(x) = exp(ω·x′ − |x′|²/2)/√m with x′ = x/d^{1/4};
 * trig: φ(x) = exp(|x′|²/2)/√m · [cos ω·x′, sin ω·x′] (can make the normaliser negative).
 */
export function featureMap(x: number[], omega: Matrix, kind: FeatureKind): number[] {
  const d = x.length;
  const xs = x.map((v) => v / d ** 0.25);
  const h = dot3(xs, xs) / 2;
  const m = omega.length;
  if (kind === 'positive') return omega.map((w) => Math.exp(dot3(w, xs) - h) / Math.sqrt(m));
  const s = Math.exp(h) / Math.sqrt(m);
  return [...omega.map((w) => s * Math.cos(dot3(w, xs))), ...omega.map((w) => s * Math.sin(dot3(w, xs)))];
}

/** Linear-cost attention D̂⁻¹(Q̂(K̂ᵀV)) (Eq. 20); `weights` is the implied row-normalised matrix. */
export function attentionFavor(Q: Matrix, K: Matrix, V: Matrix, omega: Matrix, kind: FeatureKind = 'positive'): { out: Matrix; weights: Matrix; minDenominator: number } {
  const Qh = Q.map((q) => featureMap(q, omega, kind));
  const Kh = K.map((k) => featureMap(k, omega, kind));
  const f = Qh[0]!.length;
  const cols = V[0]!.length;
  // K̂ᵀV (f × cols) and K̂ᵀ1 (f): the O(N f F) route; the N × N matrix is only built for display
  const KtV = Array.from({ length: f }, (_, a) => Array.from({ length: cols }, (_, c) => Kh.reduce((s, k, j) => s + k[a]! * V[j]![c]!, 0)));
  const Kt1 = Array.from({ length: f }, (_, a) => Kh.reduce((s, k) => s + k[a]!, 0));
  let minDenominator = Infinity;
  const out = Qh.map((q) => {
    const den = dot3(q, Kt1);
    minDenominator = Math.min(minDenominator, den);
    return Array.from({ length: cols }, (_, c) => q.reduce((s, qa, a) => s + qa * KtV[a]![c]!, 0) / den);
  });
  const weights = Qh.map((q) => {
    const den = dot3(q, Kt1);
    return Kh.map((k) => dot3(q, k) / den);
  });
  return { out, weights, minDenominator };
}

// ---------------------------------------------------------------------------
// Toy SpookyNet
// ---------------------------------------------------------------------------

interface Module {
  res1: Residual;
  res2: Residual;
  post: ResMLP;
  c: ResMLP;
  s: ResMLP;
  p: ResMLP;
  d: ResMLP;
  l: ResMLP;
  Gs: Lin;
  Gp: Lin;
  Gd: Lin;
  P1: Lin;
  P2: Lin;
  D1: Lin;
  D2: Lin;
  nq: ResMLP;
  nk: ResMLP;
  nv: ResMLP;
}

interface ElectronicEmbedding {
  q: Lin;
  kPos: number[];
  kNeg: number[];
  vPos: number[];
  vNeg: number[];
  mlp: ResMLP;
}

export interface ToySpooky {
  F: number;
  K: number;
  gamma: number;
  rc: number;
  M: Lin;
  eTilde: Record<number, number[]>;
  charge: ElectronicEmbedding;
  spin: ElectronicEmbedding;
  modules: Module[];
  wE: number[];
  eBias: Record<number, number>;
  wq: number[];
  qBias: Record<number, number>;
}

function makeEmbedding(rand: () => number, F: number): ElectronicEmbedding {
  return { q: lin(rand, F, F), kPos: randVec(rand, F, 1), kNeg: randVec(rand, F, 1), vPos: randVec(rand, F, 1), vNeg: randVec(rand, F, 1), mlp: resmlp(rand, F, false) };
}

export function makeToySpooky(seed = 7, F = 12, K = 8, modules = 3): ToySpooky {
  const rand = mulberry32(seed);
  const eTilde: Record<number, number[]> = {};
  const eBias: Record<number, number> = {};
  const qBias: Record<number, number> = {};
  for (const Z of ELEMENTS) {
    eTilde[Z] = randVec(rand, F, 0.5);
    eBias[Z] = 0;
    qBias[Z] = 0.1 * gaussian(rand);
  }
  const mods: Module[] = Array.from({ length: modules }, () => ({
    res1: residual(rand, F),
    res2: residual(rand, F),
    post: resmlp(rand, F),
    c: resmlp(rand, F),
    s: resmlp(rand, F),
    p: resmlp(rand, F),
    d: resmlp(rand, F),
    l: resmlp(rand, F, true, 0.5),
    Gs: lin(rand, F, K, false, 1.5),
    Gp: lin(rand, F, K, false, 1.5),
    Gd: lin(rand, F, K, false, 1.5),
    P1: lin(rand, F, F, false),
    P2: lin(rand, F, F, false),
    D1: lin(rand, F, F, false),
    D2: lin(rand, F, F, false),
    nq: resmlp(rand, F, true, 0.6),
    nk: resmlp(rand, F, true, 0.6),
    nv: resmlp(rand, F),
  }));
  return {
    F,
    K,
    gamma: GAMMA_INIT,
    rc: RC_PAPER,
    M: lin(rand, F, 20, false, 2),
    eTilde,
    charge: makeEmbedding(rand, F),
    spin: makeEmbedding(rand, F),
    modules: mods,
    wE: randVec(rand, F, 0.3 / Math.sqrt(F)),
    eBias,
    wq: randVec(rand, F, 0.04 / Math.sqrt(F)),
    qBias,
  };
}

export const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)));

/** e_Z = M d_Z + ẽ_Z (Eq. 9). */
export const nuclearEmbedding = (model: ToySpooky, Z: number) => add(applyLin(model.M, elementDescriptor(Z)), model.eTilde[Z]!);

/**
 * Electronic embedding (Eq. 10). Returns the per-atom attention a_i (Σ a_i = Ψ) and e_Ψ.
 * Note: q_i depends only on e_Z, so a_i depends only on the elements present and on Ψ.
 */
export function electronicEmbedding(model: ToySpooky, emb: ElectronicEmbedding, eZ: number[][], psi: number): { a: number[]; e: number[][] } {
  const F = model.F;
  if (psi === 0) return { a: eZ.map(() => 0), e: eZ.map(() => Array(F).fill(0)) };
  const k = psi >= 0 ? emb.kPos : emb.kNeg;
  const v = psi >= 0 ? emb.vPos : emb.vNeg;
  const w = eZ.map((z) => softplus(dot3(applyLin(emb.q, z), k) / Math.sqrt(F)));
  const sum = w.reduce((s, x) => s + x, 0);
  const a = w.map((x) => (psi * x) / sum);
  return { a, e: a.map((ai) => applyResMLP(emb.mlp, v.map((vv) => ai * vv))) };
}

export type NonlocalMode = 'off' | 'exact' | 'favor';

export interface ToyOptions {
  Q?: number;
  S?: number;
  modules?: number;
  nonlocal?: NonlocalMode;
  omega?: Matrix;
  /** zero the local block's p and d terms (an l = 0-only update). */
  sOnly?: boolean;
}

export interface ToyRun {
  x0: number[][];
  /** x⁽ᵗ⁾ after each module. */
  xs: number[][][];
  f: number[][];
  Ei: number[];
  qTilde: number[];
  q: number[];
  /** Σ_i E_i: the network energy, without electrostatics. */
  energy: number;
  aQ: number[];
  aS: number[];
  /** Queries, keys and values of each module's nonlocal block (empty when it is off). */
  attention: Array<{ Q: Matrix; K: Matrix; V: Matrix }>;
}

interface Pair {
  i: number;
  j: number;
  rho: number[];
  y1: number[];
  y2: number[];
}

function neighbourPairs(m: Molecule, K: number, gamma: number, rc: number): Pair[] {
  const out: Pair[] = [];
  const n = m.positions.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const rij = sub(m.positions[j]!, m.positions[i]!);
      const r = norm(rij);
      if (r >= rc) continue;
      out.push({ i, j, rho: radialBasis(r, K, gamma, rc), y1: shBlock(1, rij), y2: shBlock(2, rij) });
    }
  }
  return out;
}

/** One forward pass of the toy SpookyNet. */
export function runToy(model: ToySpooky, mol: Molecule, opts: ToyOptions = {}): ToyRun {
  const { Q = 0, S = 0, nonlocal = 'exact', sOnly = false } = opts;
  const T = Math.min(opts.modules ?? model.modules.length, model.modules.length);
  const F = model.F;
  const N = mol.numbers.length;
  const eZ = mol.numbers.map((Z) => nuclearEmbedding(model, Z));
  const eQ = electronicEmbedding(model, model.charge, eZ, Q);
  const eS = electronicEmbedding(model, model.spin, eZ, Math.abs(S));
  const x0 = eZ.map((z, i) => add(add(z, eQ.e[i]!), eS.e[i]!));
  const pairs = neighbourPairs(mol, model.K, model.gamma, model.rc);
  let x = x0;
  const xs: number[][][] = [];
  const f = Array.from({ length: N }, () => Array(F).fill(0) as number[]);
  const attention: ToyRun['attention'] = [];
  for (let t = 0; t < T; t++) {
    const md = model.modules[t]!;
    const xt = x.map((xi) => applyResidual(md.res1, xi));
    // local interaction (Eq. 12)
    const c = xt.map((xi) => applyResMLP(md.c, xi));
    const xsj = xt.map((xi) => applyResMLP(md.s, xi));
    const xpj = xt.map((xi) => applyResMLP(md.p, xi));
    const xdj = xt.map((xi) => applyResMLP(md.d, xi));
    const s = Array.from({ length: N }, () => Array(F).fill(0) as number[]);
    const p = Array.from({ length: N }, () => Array.from({ length: 3 }, () => Array(F).fill(0) as number[]));
    const d = Array.from({ length: N }, () => Array.from({ length: 5 }, () => Array(F).fill(0) as number[]));
    for (const pr of pairs) {
      const gs = applyLin(md.Gs, pr.rho);
      const gp = applyLin(md.Gp, pr.rho);
      const gd = applyLin(md.Gd, pr.rho);
      for (let a = 0; a < F; a++) {
        s[pr.i]![a] = s[pr.i]![a]! + xsj[pr.j]![a]! * gs[a]!;
        for (let mm = 0; mm < 3; mm++) p[pr.i]![mm]![a] = p[pr.i]![mm]![a]! + xpj[pr.j]![a]! * gp[a]! * pr.y1[mm]!;
        for (let mm = 0; mm < 5; mm++) d[pr.i]![mm]![a] = d[pr.i]![mm]![a]! + xdj[pr.j]![a]! * gd[a]! * pr.y2[mm]!;
      }
    }
    const l = xt.map((_, i) => {
      let inner = add(c[i]!, s[i]!);
      if (!sOnly) {
        const pp = Array(F).fill(0) as number[];
        for (let mm = 0; mm < 3; mm++) {
          const a1 = applyLin(md.P1, p[i]![mm]!);
          const a2 = applyLin(md.P2, p[i]![mm]!);
          for (let a = 0; a < F; a++) pp[a] = pp[a]! + a1[a]! * a2[a]!;
        }
        for (let mm = 0; mm < 5; mm++) {
          const a1 = applyLin(md.D1, d[i]![mm]!);
          const a2 = applyLin(md.D2, d[i]![mm]!);
          for (let a = 0; a < F; a++) pp[a] = pp[a]! + a1[a]! * a2[a]!;
        }
        inner = add(inner, pp);
      }
      return applyResMLP(md.l, inner);
    });
    // nonlocal interaction (Eq. 18)
    let n: number[][];
    if (nonlocal === 'off') n = xt.map(() => Array(F).fill(0) as number[]);
    else {
      const Qm = xt.map((xi) => applyResMLP(md.nq, xi));
      const Km = xt.map((xi) => applyResMLP(md.nk, xi));
      const Vm = xt.map((xi) => applyResMLP(md.nv, xi));
      attention.push({ Q: Qm, K: Km, V: Vm });
      n = nonlocal === 'exact' ? attentionExact(Qm, Km, Vm).out : attentionFavor(Qm, Km, Vm, opts.omega ?? drawOmega(11 + t, F, F)).out;
    }
    x = xt.map((xi, i) => applyResidual(md.res2, add(add(xi, l[i]!), n[i]!)));
    xs.push(x);
    const y = x.map((xi) => applyResMLP(md.post, xi));
    y.forEach((yi, i) => yi.forEach((v, a) => (f[i]![a] = f[i]![a]! + v)));
  }
  const Ei = f.map((fi, i) => dot3(model.wE, fi) + model.eBias[mol.numbers[i]!]!);
  const qTilde = f.map((fi, i) => dot3(model.wq, fi) + model.qBias[mol.numbers[i]!]!);
  const q = chargeCorrection(qTilde, Q);
  return { x0, xs, f, Ei, qTilde, q, energy: Ei.reduce((s, v) => s + v, 0), aQ: eQ.a, aS: eS.a, attention };
}

/** q_i = q̃_i + (Q − Σ_j q̃_j)/N (Eq. 24). */
export function chargeCorrection(qTilde: number[], Q: number): number[] {
  const shift = (Q - qTilde.reduce((s, v) => s + v, 0)) / qTilde.length;
  return qTilde.map((v) => v + shift);
}

/** Smooth switch of Eq. (25): 1 below r_on, 0 above r_off. */
export function fswitch(r: number, rOn: number, rOff: number): number {
  const sig = (x: number) => (x > 0 ? Math.exp(-1 / x) : 0);
  const x = (r - rOn) / (rOff - rOn);
  const a = sig(1 - x);
  const b = sig(x);
  return a / (a + b);
}

/**
 * E_ele = k_e Σ_{i<j} q_i q_j [f_sw/√(r² + 1) + (1 − f_sw)/r] (Eq. 23), with r_on = r_c/4,
 * r_off = 3r_c/4 and r in bohr inside the damped term (the paper works in atomic units).
 */
export function electrostatics(q: number[], positions: Vec3[], rc = RC_PAPER): number {
  let e = 0;
  for (let i = 0; i < q.length; i++) {
    for (let j = i + 1; j < q.length; j++) {
      const r = norm(sub(positions[j]!, positions[i]!));
      const sw = fswitch(r, rc / 4, (3 * rc) / 4);
      const rb = r / BOHR;
      e += q[i]! * q[j]! * ((sw / Math.sqrt(rb * rb + 1)) / BOHR + (sw < 1 ? (1 - sw) / r : 0));
    }
  }
  return K_E * e;
}

/** Largest absolute difference between two feature lists. */
export function maxDiff(a: number[][], b: number[][]): number {
  let m = 0;
  a.forEach((row, i) => row.forEach((v, k) => (m = Math.max(m, Math.abs(v - b[i]![k]!)))));
  return m;
}
