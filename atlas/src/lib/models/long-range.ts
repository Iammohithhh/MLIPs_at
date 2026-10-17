// Long-range model families (4G-HDNNP, LES, CACE), the Atlas long-range page.
//
// Three self-contained pieces, all pure and tested in long-range.test.ts:
//
// 1. Who can see a change beyond the cutoff (Fig. 1). A C₁₀H₂ / C₁₀H₃⁺ chain, the first test system of
//    the 4G-HDNNP paper (Ko et al., Nat. Commun. 2021, Fig. 3). The same toy atomic network, fed by
//    Behler radial symmetry functions inside r_c = 4.23 Å (the paper's cutoff), predicts either
//      - a charge directly, q_i = q(G_i): 3G-HDNNP, and the latent charge of LES (Cheng 2025, Eq. 2);
//      - the same, shifted uniformly so Σq equals the total charge (3G "scaled" charges); or
//      - an electronegativity χ_i = χ(G_i), from which charge equilibration with Gaussian charges and a
//        fixed total charge gives the charges (4G-HDNNP, Eqs. 2–7).
//    Weights are random and untrained: whether a charge can change is a property of the construction,
//    its size is not meaningful.
//
// 2. What the latent Ewald sum computes (Fig. 2): E_lr = k_e (2π/V) Σ_{0<|k|<k_c} e^{−σ²k²/2}/k² |S(k)|²,
//    S(k) = Σ q_i e^{ik·r_i} (King et al. 2025, Eqs. 3–4; Cheng 2025, Eq. 4 up to the prefactor), with the
//    real-space Ewald pieces needed to show what it equals.
//
// 3. CACE's Cartesian angular basis (Fig. 3): A_l = Σ_j w_j L_l(r̂_j) with L_l(r̂) = x^lx y^ly z^lz,
//    B⁽²⁾_l = Σ_l C(l) A_l² and B⁽³⁾_{l1 l2} = Σ C(l1) C(l2) A_l1 A_(l1+l2) A_l2 (Cheng 2024, Eqs. 2, 4, 6–9),
//    checked against explicit sums of cos^l θ and decomposed into Legendre (spherical) invariants.

import { COULOMB_K } from '../chem/elements';
import { mulberry32 } from '../chem/rng';
import { erfc } from '../physics/madelung';
import { QEQ_TOY } from '../physics/qeq';
import { realSH } from '../physics/spherical-harmonics';
import { covalentRadius } from '../structure/elements';
import type { Mat3, Structure, Vec3 } from '../structure/types';

export type { Vec3 };

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const erf = (x: number) => 1 - erfc(x);

// ---------------------------------------------------------------------------
// 1. Non-local charge transfer on a carbon chain
// ---------------------------------------------------------------------------

/** 4G-HDNNP cutoff for C₁₀H₂/C₁₀H₃⁺ (Ko et al., Sec. III, "Rc = 4.23 Å"). */
export const RC_4G = 4.23;
/** Idealised geometry (not the paper's DFT geometries, which are in its SI): Å. */
export const CC = 1.28;
export const CH = 1.06;
export const CH_SP2 = 1.09;

export interface Molecule {
  symbols: string[];
  positions: Vec3[];
  /** Total charge in e. */
  charge: number;
}

/**
 * Linear C_nH₂ (atom 0 = left H, 1..n = C, n+1 = right H) or, protonated, C_nH₃⁺ whose right carbon
 * carries two H at 120° in the xz plane (atoms n+1, n+2). Atoms 0..n are at identical positions in
 * both: the toy, unlike DFT, does not relax the chain after protonation.
 */
export function carbonChain(n = 10, protonated = false): Molecule {
  const symbols = ['H'];
  const positions: Vec3[] = [[-CH, 0, 0]];
  for (let k = 0; k < n; k++) {
    symbols.push('C');
    positions.push([k * CC, 0, 0]);
  }
  const end = (n - 1) * CC;
  if (!protonated) {
    symbols.push('H');
    positions.push([end + CH, 0, 0]);
  } else {
    const a = (60 * Math.PI) / 180;
    symbols.push('H', 'H');
    positions.push([end + CH_SP2 * Math.cos(a), 0, CH_SP2 * Math.sin(a)], [end + CH_SP2 * Math.cos(a), 0, -CH_SP2 * Math.sin(a)]);
  }
  return { symbols, positions, charge: protonated ? 1 : 0 };
}

/** Behler's cosine cutoff function. */
export const fc = (r: number, rc: number) => (r < rc ? 0.5 * (Math.cos((Math.PI * r) / rc) + 1) : 0);

const ETAS = [0.05, 0.3, 1.0, 3.0]; // Å⁻², radial G2 widths (centred at r_s = 0)
const SPECIES = ['H', 'C'];
export const N_SF = ETAS.length * SPECIES.length;

/** Radial symmetry functions G²_i = Σ_j∈species e^{−η r²} f_c(r), one per (species, η). */
export function radialSF(m: Molecule, i: number, rc = RC_4G): number[] {
  const G = new Array<number>(N_SF).fill(0);
  m.positions.forEach((p, j) => {
    if (j === i) return;
    const r = norm(sub(p, m.positions[i]!));
    if (r >= rc) return;
    const s = SPECIES.indexOf(m.symbols[j]!);
    ETAS.forEach((eta, e) => (G[s * ETAS.length + e]! += Math.exp(-eta * r * r) * fc(r, rc)));
  });
  return G;
}

export interface AtomicNet {
  /** Per element: hidden weights W (H × N_SF), biases b (H), output weights w (H). */
  nets: Record<string, { W: number[][]; b: number[]; w: number[] }>;
}

export function makeAtomicNet(seed = 3, hidden = 6): AtomicNet {
  const rand = mulberry32(seed);
  const g = () => 2 * rand() - 1;
  const nets: AtomicNet['nets'] = {};
  for (const el of SPECIES)
    nets[el] = {
      W: Array.from({ length: hidden }, () => Array.from({ length: N_SF }, () => 1.2 * g())),
      b: Array.from({ length: hidden }, () => 0.3 * g()),
      w: Array.from({ length: hidden }, () => g() / Math.sqrt(hidden)),
    };
  return { nets };
}

/** The atomic network's scalar output for atom i, a function of its own G_i only. */
export function atomicOutput(net: AtomicNet, m: Molecule, i: number, rc = RC_4G): number {
  const G = radialSF(m, i, rc);
  const nn = net.nets[m.symbols[i]!]!;
  let y = 0;
  nn.W.forEach((row, h) => {
    let a = nn.b[h]!;
    row.forEach((v, k) => (a += v * G[k]!));
    y += nn.w[h]! * Math.tanh(a);
  });
  return y;
}

/** Toy reference charges and electronegativity scale (e, eV/e): signs follow C more electronegative than H. */
const Q0: Record<string, number> = { H: 0.08, C: -0.016 };
const Q_SCALE = 0.06;
const CHI_SCALE = 0.6;

/** 3G-HDNNP / LES: q_i = q(G_i), no constraint on Σq. */
export function localCharges(net: AtomicNet, m: Molecule, rc = RC_4G): number[] {
  return m.symbols.map((s, i) => Q0[s]! + Q_SCALE * atomicOutput(net, m, i, rc));
}

/** 3G-HDNNP "scaled" charges: the local charges shifted uniformly so that Σq = Q_tot. */
export function scaledCharges(net: AtomicNet, m: Molecule, rc = RC_4G): number[] {
  const q = localCharges(net, m, rc);
  const shift = (m.charge - q.reduce((a, b) => a + b, 0)) / q.length;
  return q.map((x) => x + shift);
}

export interface QEqSolution {
  q: number[];
  chi: number[];
  /** Lagrange multiplier λ of Eq. (7); −λ is the equalised chemical potential. */
  lambda: number;
  /** E_Qeq of Eq. (2) at the solution, eV. */
  energy: number;
}

/** Gaussian charge widths from covalent radii (Ko et al., Sec. II). */
export const sigmaOf = (sym: string) => covalentRadius(sym).r;

/** The matrix A of Ko et al. Eq. (6), in eV/e² (k_e = 14.40 eV Å). */
export function qeqMatrix(m: Molecule, J: number[]): number[][] {
  const n = m.positions.length;
  const sig = m.symbols.map(sigmaOf);
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return J[i]! + COULOMB_K / (sig[i]! * Math.sqrt(Math.PI));
      const r = norm(sub(m.positions[i]!, m.positions[j]!));
      const g = Math.sqrt(sig[i]! ** 2 + sig[j]! ** 2);
      return (COULOMB_K * erf(r / (Math.SQRT2 * g))) / r;
    }),
  );
}

export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    [M[c], M[p]] = [M[p]!, M[c]!];
    for (let r = c + 1; r < n; r++) {
      const f = M[r]![c]! / M[c]![c]!;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let k = r + 1; k < n; k++) s -= M[r]![k]! * x[k]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/** 4G-HDNNP: χ_i = χ⁰(Z) + network(G_i); charges from Eq. (7) with Σq = Q_tot. */
export function qeqCharges(net: AtomicNet, m: Molecule, rc = RC_4G): QEqSolution {
  const n = m.positions.length;
  const chi = m.symbols.map((s, i) => QEQ_TOY[s]!.chi + CHI_SCALE * atomicOutput(net, m, i, rc));
  const J = m.symbols.map((s) => QEQ_TOY[s]!.J);
  const A = qeqMatrix(m, J);
  const big = A.map((row) => [...row, 1]);
  big.push([...new Array<number>(n).fill(1), 0]);
  const x = solveLinear(big, [...chi.map((c) => -c), m.charge]);
  const q = x.slice(0, n);
  let energy = 0;
  for (let i = 0; i < n; i++) {
    energy += chi[i]! * q[i]! + 0.5 * A[i]![i]! * q[i]! ** 2;
    for (let j = 0; j < i; j++) energy += A[i]![j]! * q[i]! * q[j]!;
  }
  return { q, chi, lambda: x[n]!, energy };
}

export type ChargeModel = 'local' | 'scaled' | 'qeq';

export function chargesFor(model: ChargeModel, net: AtomicNet, m: Molecule): number[] {
  if (model === 'local') return localCharges(net, m);
  if (model === 'scaled') return scaledCharges(net, m);
  return qeqCharges(net, m).q;
}

/** Atoms (common to both molecules) whose descriptor G_i is identical in C_nH₂ and C_nH₃⁺. */
export function unchangedEnvironments(a: Molecule, b: Molecule, rc = RC_4G, tol = 1e-12): boolean[] {
  const common = Math.min(a.positions.length, b.positions.length) - 1; // the right-hand H differs
  return Array.from({ length: common }, (_, i) => {
    const ga = radialSF(a, i, rc);
    const gb = radialSF(b, i, rc);
    return ga.every((v, k) => Math.abs(v - gb[k]!) <= tol);
  });
}

export interface ChainComparison {
  neutral: number[];
  cation: number[];
  /** q(cation) − q(neutral) for the common atoms 0..n. */
  delta: number[];
  unchanged: boolean[];
}

export function compareChain(model: ChargeModel, net: AtomicNet, n = 10): ChainComparison {
  const a = carbonChain(n, false);
  const b = carbonChain(n, true);
  const neutral = chargesFor(model, net, a);
  const cation = chargesFor(model, net, b);
  const delta = Array.from({ length: n + 1 }, (_, i) => cation[i]! - neutral[i]!);
  return { neutral, cation, delta, unchanged: unchangedEnvironments(a, b) };
}

// ---------------------------------------------------------------------------
// 2. Latent Ewald summation
// ---------------------------------------------------------------------------

export const LES_SIGMA = 1.0; // Å, "typically set to 1 Å" (Cheng 2025, Eq. 4)
export const LES_KC = (2 * Math.PI) / 3; // Å⁻¹, dimers and molten NaCl (Cheng 2025, Secs. I B, I C)
export const DIMER_BOX = 30; // Å, cubic box of the dimer benchmark (Cheng 2025, Sec. I B)

export interface LesResult {
  /** eV, with q in e. */
  energy: number;
  /** Number of k vectors with 0 < |k| < k_c. */
  nk: number;
}

/** E_lr for an orthorhombic cell of edges L: k_e (2π/V) Σ_{0<|k|<k_c} e^{−σ²k²/2}/k² |S(k)|². */
export function lesEnergy(pos: Vec3[], q: number[], L: Vec3, sigma = LES_SIGMA, kc = LES_KC): LesResult {
  const V = L[0] * L[1] * L[2];
  const n = L.map((l) => Math.floor((kc * l) / (2 * Math.PI))) as Vec3;
  let sum = 0;
  let nk = 0;
  for (let a = -n[0]; a <= n[0]; a++)
    for (let b = -n[1]; b <= n[1]; b++)
      for (let c = -n[2]; c <= n[2]; c++) {
        if (a === 0 && b === 0 && c === 0) continue;
        const k: Vec3 = [(2 * Math.PI * a) / L[0], (2 * Math.PI * b) / L[1], (2 * Math.PI * c) / L[2]];
        const k2 = dot(k, k);
        if (k2 >= kc * kc) continue;
        let re = 0;
        let im = 0;
        pos.forEach((p, i) => {
          const ph = dot(k, p);
          re += q[i]! * Math.cos(ph);
          im += q[i]! * Math.sin(ph);
        });
        sum += (Math.exp((-sigma * sigma * k2) / 2) / k2) * (re * re + im * im);
        nk++;
      }
  return { energy: ((COULOMB_K * 2 * Math.PI) / V) * sum, nk };
}

/** Interaction of two Gaussian charges whose combined Ewald factor is e^{−σ²k²/2}: k_e q₁q₂ erf(r/√2σ)/r. */
export const gaussianPair = (r: number, q1: number, q2: number, sigma = LES_SIGMA) =>
  (COULOMB_K * q1 * q2 * erf(r / (Math.SQRT2 * sigma))) / r;

/** Constant self part of E_lr per atom: k_e q²/(√(2π) σ), present because Eq. (3) keeps i = j terms. */
export const selfTerm = (q: number, sigma = LES_SIGMA) => (COULOMB_K * q * q) / (Math.sqrt(2 * Math.PI) * sigma);

/** Tinfoil vs vacuum: dropping k = 0 in a cubic cell removes 2π|M|²/3V (k_e units). */
export const dipoleTerm = (M: Vec3, V: number) => (COULOMB_K * 2 * Math.PI * dot(M, M)) / (3 * V);

/** Real-space Ewald part: ½ Σ_i Σ_j Σ_n' k_e q_i q_j erfc(|r_ij + nL|/√2σ)/|r_ij + nL|. */
export function realSpaceEwald(pos: Vec3[], q: number[], L: Vec3, sigma: number, rcut: number): number {
  const n = L.map((l) => Math.ceil(rcut / l)) as Vec3;
  let e = 0;
  for (let i = 0; i < pos.length; i++)
    for (let j = 0; j < pos.length; j++)
      for (let a = -n[0]; a <= n[0]; a++)
        for (let b = -n[1]; b <= n[1]; b++)
          for (let c = -n[2]; c <= n[2]; c++) {
            if (i === j && a === 0 && b === 0 && c === 0) continue;
            const d: Vec3 = [pos[j]![0] - pos[i]![0] + a * L[0], pos[j]![1] - pos[i]![1] + b * L[1], pos[j]![2] - pos[i]![2] + c * L[2]];
            const r = norm(d);
            if (r < rcut) e += (0.5 * COULOMB_K * q[i]! * q[j]! * erfc(r / (Math.SQRT2 * sigma))) / r;
          }
  return e;
}

/** k beyond which e^{−σ²k²/2} < 1e−12: the sum is converged ("k_c → ∞"). */
export const kConverged = (sigma: number) => Math.sqrt(2 * 12 * Math.log(10)) / sigma;

/**
 * The i = j terms of Eq. (3) are a constant per atom, q_i² times this (eV/e²). With k_c → ∞ it is the
 * self term plus the interaction of a charge with its own images and background; truncating at k_c changes
 * it by a constant. We remove only the self term and the truncation change, so the curves below are the
 * interaction a neutral pair actually has in the periodic cell; for a local q a short-range network can
 * absorb any such constant.
 */
export function perAtomConstant(L: number, sigma = LES_SIGMA, kc = LES_KC): number {
  const box: Vec3 = [L, L, L];
  const one: Vec3[] = [[0, 0, 0]];
  return lesEnergy(one, [1], box, sigma, kc).energy - lesEnergy(one, [1], box, sigma, kConverged(sigma)).energy + selfTerm(1, sigma);
}

export interface DimerPoint {
  d: number;
  /** E_lr − (q₁² + q₂²) × perAtomConstant, eV. */
  les: number;
  /** The same with k_c → ∞. */
  lesFull: number;
  /** Isolated pair of Gaussian charges. */
  isolated: number;
  /** Bare point charges, k_e q₁q₂/d. */
  coulomb: number;
}

/** Two charges a distance d apart along x in a cubic box of edge L. */
export function dimerCurve(q1: number, q2: number, ds: number[], L = DIMER_BOX, sigma = LES_SIGMA, kc = LES_KC): DimerPoint[] {
  const kFull = kConverged(sigma);
  const q2sum = q1 * q1 + q2 * q2;
  const cKc = q2sum * perAtomConstant(L, sigma, kc);
  const cFull = q2sum * perAtomConstant(L, sigma, kFull);
  const box: Vec3 = [L, L, L];
  return ds.map((d) => {
    const pos: Vec3[] = [
      [L / 2 - d / 2, L / 2, L / 2],
      [L / 2 + d / 2, L / 2, L / 2],
    ];
    return {
      d,
      les: lesEnergy(pos, [q1, q2], box, sigma, kc).energy - cKc,
      lesFull: lesEnergy(pos, [q1, q2], box, sigma, kFull).energy - cFull,
      isolated: gaussianPair(d, q1, q2, sigma),
      coulomb: (COULOMB_K * q1 * q2) / d,
    };
  });
}

/**
 * The same curve, fast, for a cubic box: for two charges along x, |S(k)|² = q₁² + q₂² + 2q₁q₂ cos(k_x d),
 * so E_lr depends on d only through W(k_x) = k_e (2π/V) Σ_{k_y,k_z} e^{−σ²k²/2}/k² over |k| < k_c.
 */
export interface PairKernel {
  kx: number[];
  W: number[];
  nk: number;
  /** E_lr of one unit charge alone (the i = j part), eV. */
  single: number;
}

export function pairKernel(L: number, sigma = LES_SIGMA, kc = LES_KC): PairKernel {
  const n = Math.floor((kc * L) / (2 * Math.PI));
  const pre = (COULOMB_K * 2 * Math.PI) / L ** 3;
  const kx: number[] = [];
  const W: number[] = [];
  let nk = 0;
  for (let a = -n; a <= n; a++) {
    let w = 0;
    for (let b = -n; b <= n; b++)
      for (let c = -n; c <= n; c++) {
        if (a === 0 && b === 0 && c === 0) continue;
        const k2 = ((2 * Math.PI) / L) ** 2 * (a * a + b * b + c * c);
        if (k2 >= kc * kc) continue;
        w += Math.exp((-sigma * sigma * k2) / 2) / k2;
        nk++;
      }
    kx.push((2 * Math.PI * a) / L);
    W.push(pre * w);
  }
  return { kx, W, nk, single: W.reduce((t, v) => t + v, 0) };
}

/** E_lr − (q₁² + q₂²) × perAtomConstant from precomputed kernels (truncated and converged). */
export function pairEnergy(d: number, q1: number, q2: number, kcKernel: PairKernel, fullKernel: PairKernel, sigma = LES_SIGMA): number {
  let p = 0;
  kcKernel.kx.forEach((k, i) => (p += 2 * kcKernel.W[i]! * Math.cos(k * d)));
  return q1 * q2 * p + (q1 * q1 + q2 * q2) * (fullKernel.single - selfTerm(1, sigma));
}

// ---------------------------------------------------------------------------
// 3. CACE: the Cartesian angular basis
// ---------------------------------------------------------------------------

export type Mono = [number, number, number];

/** All (lx, ly, lz) with lx + ly + lz = l, (l+1)(l+2)/2 of them. */
export function monomials(l: number): Mono[] {
  const out: Mono[] = [];
  for (let lx = l; lx >= 0; lx--) for (let ly = l - lx; ly >= 0; ly--) out.push([lx, ly, l - lx - ly]);
  return out;
}

const fact = (n: number): number => (n <= 1 ? 1 : n * fact(n - 1));
/** C(l) = l!/(lx! ly! lz!), the multinomial weight of Eq. (4). */
export const multinomial = (m: Mono) => fact(m[0] + m[1] + m[2]) / (fact(m[0]) * fact(m[1]) * fact(m[2]));
export const monoValue = (m: Mono, v: Vec3) => v[0] ** m[0] * v[1] ** m[1] * v[2] ** m[2];

/** A neighbour as seen from the centre: unit vector and radial weight. */
export interface Neighbour {
  u: Vec3;
  w: number;
}

/** Neighbours of atom i within rc, with all periodic images, radial weight R(r) = f_c(r) (Behler cosine). */
export function environment(s: Structure, i: number, rc: number): { vectors: Vec3[]; neighbours: Neighbour[] } {
  const cell = s.cell as Mat3 | null;
  const vectors: Vec3[] = [];
  const reach = cell ? cell.map((v) => Math.ceil(rc / minHeight(cell, v)) + 1) : [0, 0, 0];
  const c = s.positions[i]!;
  for (let a = -reach[0]!; a <= reach[0]!; a++)
    for (let b = -reach[1]!; b <= reach[1]!; b++)
      for (let k = -reach[2]!; k <= reach[2]!; k++)
        s.positions.forEach((p) => {
          const t: Vec3 = cell
            ? [
                p[0] + a * cell[0][0] + b * cell[1][0] + k * cell[2][0],
                p[1] + a * cell[0][1] + b * cell[1][1] + k * cell[2][1],
                p[2] + a * cell[0][2] + b * cell[1][2] + k * cell[2][2],
              ]
            : p;
          if (!cell && (a || b || k)) return;
          const d = sub(t, c);
          const r = norm(d);
          if (r > 1e-8 && r < rc) vectors.push(d);
        });
  vectors.sort((x, y) => norm(x) - norm(y));
  return { vectors, neighbours: neighboursOf(vectors, rc) };
}

function minHeight(cell: Mat3, v: Vec3): number {
  // Height of the cell along the direction normal to the other two vectors.
  const others = cell.filter((x) => x !== v);
  const n: Vec3 = [
    others[0]![1] * others[1]![2] - others[0]![2] * others[1]![1],
    others[0]![2] * others[1]![0] - others[0]![0] * others[1]![2],
    others[0]![0] * others[1]![1] - others[0]![1] * others[1]![0],
  ];
  return Math.abs(dot(v, n)) / norm(n);
}

export function neighboursOf(vectors: Vec3[], rc: number): Neighbour[] {
  return vectors.map((d) => {
    const r = norm(d);
    return { u: [d[0] / r, d[1] / r, d[2] / r], w: fc(r, rc) };
  });
}

/** A_l for every monomial of order l (Eq. 6 with Eq. 2 on unit vectors). */
export function cartesianA(env: Neighbour[], l: number): number[] {
  return monomials(l).map((m) => env.reduce((s, nb) => s + nb.w * monoValue(m, nb.u), 0));
}

/** B⁽²⁾_l = Σ_l C(l) A_l² (Eq. 7). */
export function b2(env: Neighbour[], l: number): number {
  const A = cartesianA(env, l);
  return monomials(l).reduce((s, m, k) => s + multinomial(m) * A[k]! ** 2, 0);
}

/** The same invariant as an explicit double sum, Σ_jk w_j w_k cos^l θ_jk (Eq. 8). */
export function b2Direct(env: Neighbour[], l: number): number {
  let s = 0;
  for (const a of env) for (const b of env) s += a.w * b.w * dot(a.u, b.u) ** l;
  return s;
}

const addMono = (a: Mono, b: Mono): Mono => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** B⁽³⁾_{l1 l2} = Σ C(l1) C(l2) A_l1 A_(l1+l2) A_l2 (Eq. 9). */
export function b3(env: Neighbour[], l1: number, l2: number): number {
  const A = (m: Mono) => env.reduce((s, nb) => s + nb.w * monoValue(m, nb.u), 0);
  let s = 0;
  for (const m1 of monomials(l1)) for (const m2 of monomials(l2)) s += multinomial(m1) * multinomial(m2) * A(m1) * A(addMono(m1, m2)) * A(m2);
  return s;
}

/** Σ_jkh w_j w_k w_h cos^l1 θ_jk cos^l2 θ_kh: four-body terms with angles chained through k. */
export function b3Direct(env: Neighbour[], l1: number, l2: number): number {
  let s = 0;
  for (const a of env) for (const b of env) for (const c of env) s += a.w * b.w * c.w * dot(a.u, b.u) ** l1 * dot(b.u, c.u) ** l2;
  return s;
}

/** Legendre polynomial P_l(x) by recursion. */
export function legendreP(l: number, x: number): number {
  if (l === 0) return 1;
  let p0 = 1;
  let p1 = x;
  for (let k = 1; k < l; k++) [p0, p1] = [p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)];
  return p1;
}

/**
 * Coefficients c_{l,l'} of x^l = Σ_{l'} c_{l,l'} P_{l'}(x) (l' = l, l−2, ...):
 * c = (2l'+1) l! / (2^{(l−l')/2} ((l−l')/2)! (l+l'+1)!!).
 */
export function powerToLegendre(l: number): number[] {
  const out = new Array<number>(l + 1).fill(0);
  const dfact = (n: number): number => (n <= 1 ? 1 : n * dfact(n - 2));
  for (let lp = l; lp >= 0; lp -= 2) {
    const k = (l - lp) / 2;
    out[lp] = ((2 * lp + 1) * fact(l)) / (2 ** k * fact(k) * dfact(l + lp + 1));
  }
  return out;
}

/** Spherical power spectrum p_l = (4π/(2l+1)) Σ_m A_lm² = Σ_jk w_j w_k P_l(cos θ_jk) (addition theorem). */
export function sphericalPower(env: Neighbour[], l: number): number {
  let s = 0;
  for (let m = -l; m <= l; m++) {
    const a = env.reduce((t, nb) => t + nb.w * realSH(l, m, nb.u), 0);
    s += a * a;
  }
  return ((4 * Math.PI) / (2 * l + 1)) * s;
}

/** f_l(n̂) = Σ_j w_j (n̂ · r̂_j)^l = Σ_l C(l) A_l L_l(n̂): the order-l Cartesian moment as a function on the sphere. */
export const momentSurface = (env: Neighbour[], l: number, n: Vec3) => env.reduce((s, nb) => s + nb.w * dot(n, nb.u) ** l, 0);

export function rotateAll(vs: Vec3[], R: number[]): Vec3[] {
  return vs.map((v) => [
    R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2],
    R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2],
    R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2],
  ]);
}
