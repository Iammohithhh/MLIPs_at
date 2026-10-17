// Architecture X-ray for a long-range model: MACELES-OFF (MACE + Latent Ewald Summation).
// Types for the precomputed forward passes (pipeline/atlas_pipeline/xray_les.py) and the closed
// forms the page derives: the LES pair kernel, the latent energy and potential, the dipole of
// mean-removed charges, the Born-effective-charge sum rule and the dipole–dipole tail.

import type { Vec3 } from './xray';
import { matVec } from './xray';

export type { Vec3 };

export interface LesHyper {
  r_max: number;
  num_bessel: number;
  num_polynomial_cutoff: number;
  max_ell: number;
  num_interactions: number;
  hidden_irreps: string;
  MLP_irreps: string;
  correlation: number;
  avg_num_neighbors: number;
  radial_MLP: number[];
  atomic_inter_scale: number;
  atomic_inter_shift: number;
  atomic_numbers: number[];
  les: {
    sigma: number;
    dl: number;
    norm_factor: number;
    remove_self_interaction: boolean;
    use_atomwise: boolean;
    bec_remove_mean: boolean;
    epsilon_factor: number;
  };
}

export interface LesSystem {
  id: string;
  label: string;
  symbols: string[];
  positions: Vec3[];
  cell: null;
  n: number;
  smiles?: string;
  edges: { count: number; src: number[]; dst: number[]; vec: Vec3[] };
  features: { h1Scalar: number[]; h1Vector: number[]; h2Scalar: number[]; shapes: number[][] };
  /** Scaled readout contributions s·e_t per atom, t = 1, 2 (eV). */
  siteLayers: [number[], number[]];
  /** Latent-charge contributions of the two charge readouts. */
  qLayers: [number[], number[]];
  q: number[];
  /** Latent potential at each atom, φ_i (V, i.e. eV per unit latent charge). */
  phi: number[];
  eSr: number;
  eLr: number;
  /** E0 + scaled site energy per atom (the model's node_energy; excludes E_lr). */
  nodeEnergy: number[];
  fSr: Vec3[];
  fLr: Vec3[];
  P: Vec3;
  /** Z[i][α][β] = ∂P_α/∂r_iβ. */
  Z: Vec3[][];
  rotated?: { R: Vec3[]; positions: Vec3[]; eSr: number; eLr: number; q: number[]; fSr: Vec3[]; fLr: Vec3[]; P: Vec3 };
  dft?: { dipole: Vec3; Z: Vec3[][]; forces: Vec3[]; energy: number };
}

export interface DimerRow {
  dOO: number;
  minGap: number;
  eSr: number;
  eLr: number;
  q: number[];
  fLrO: Vec3;
  fSrO: Vec3;
}

export interface ValidationSubset {
  name: string;
  configs: number;
  atoms: number;
  energyRmsePerAtom: number;
  energyMeanPerAtom: number;
  forceRmse: number;
  forceRmseWithoutLr: number;
  becSlope: number;
  becCorr: number;
  dipoleSlope: number;
  dipoleCorr: number;
}

export interface LesDoc {
  format: 'atlas-xray-les';
  version: number;
  status: 'PRECOMPUTED';
  model: {
    family: string;
    checkpoint: string;
    file: string;
    url: string;
    md5: string;
    sha256: string;
    licence: string;
    hyperparameters: LesHyper;
    halfInvEps0: number;
    parameters: { mace: number; chargeReadouts: number };
  };
  provenance: Record<string, unknown>;
  systems: LesSystem[];
  dimer: {
    symbols: string[];
    positions: Vec3[];
    axis: Vec3;
    d0: number;
    monomers: { eSr: number; eLr: number; q: number[]; P: Vec3 }[];
    rows: DimerRow[];
  };
  validation: {
    subsets: ValidationSubset[];
    /** [subset, Z, Z_DFT,αα, Z_model,αα] */
    becPoints: [string, number, number, number][];
    /** [subset, μ_DFT, P_model] */
    dipolePoints: [string, Vec3, Vec3][];
  };
}

// ---------------------------------------------------------------------------
// Closed forms

const SQRT2 = Math.SQRT2;

/** Abramowitz–Stegun-free erf: series for small x, continued fraction (Lentz) for large x; |error| < 1e-14. */
export function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  if (a < 2.5) {
    // Maclaurin series: erf x = 2/√π Σ (−1)^n x^{2n+1} / (n! (2n+1))
    let term = a;
    let sum = a;
    for (let n = 1; n < 80; n++) {
      term *= (-a * a) / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return (s * 2 * sum) / Math.sqrt(Math.PI);
  }
  // erfc x = exp(−x²)/√π · 1/(x + 1/2/(x + 1/(x + 3/2/(x + …))))
  let f = a;
  let C = a;
  let D = 0;
  for (let n = 1; n < 200; n++) {
    const an = n / 2;
    D = a + an * D;
    D = D === 0 ? 1e-300 : 1 / D;
    C = a + an / C;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return s * (1 - Math.exp(-a * a) / (Math.sqrt(Math.PI) * f));
}

/** LES pair kernel erf(r/(√2 σ))/r; its r → 0 limit is √(2/π)/σ. */
export function lesKernel(r: number, sigma: number): number {
  if (r < 1e-8) return Math.sqrt(2 / Math.PI) / sigma;
  return erf(r / (SQRT2 * sigma)) / r;
}

/** The checkpoint's Coulomb constant n_f/2π (eV Å / e²). */
export const coulombConstant = (normFactor: number) => normFactor / (2 * Math.PI);

const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** φ_i = k Σ_{j≠i} q_j erf(r_ij/√2σ)/r_ij. */
export function lesPotential(q: readonly number[], pos: readonly Vec3[], sigma: number, normFactor: number): number[] {
  const k = coulombConstant(normFactor);
  return pos.map((pi, i) => k * pos.reduce((a, pj, j) => (j === i ? a : a + q[j]! * lesKernel(dist(pi, pj), sigma)), 0));
}

/** E_lr = ½ Σ_i q_i φ_i (no self term). */
export function lesEnergy(q: readonly number[], pos: readonly Vec3[], sigma: number, normFactor: number): number {
  const phi = lesPotential(q, pos, sigma, normFactor);
  return 0.5 * q.reduce((a, qi, i) => a + qi * phi[i]!, 0);
}

/** Net latent charge Σ q_i (not constrained to zero by the model). */
export const netCharge = (q: readonly number[]) => q.reduce((a, x) => a + x, 0);

/** P = Σ (q_i − q̄) r_i, LES's dipole with the mean removed. */
export function polarization(q: readonly number[], pos: readonly Vec3[], removeMean = true): Vec3 {
  const m = removeMean ? netCharge(q) / q.length : 0;
  return pos.reduce<Vec3>((P, r, i) => [P[0] + (q[i]! - m) * r[0], P[1] + (q[i]! - m) * r[1], P[2] + (q[i]! - m) * r[2]], [0, 0, 0]);
}

/** max |Σ_i Z_i,αβ|: zero when P does not change under a rigid translation. */
export function becSumRule(Z: readonly Vec3[][]): number {
  let worst = 0;
  for (let a = 0; a < 3; a++)
    for (let b = 0; b < 3; b++) worst = Math.max(worst, Math.abs(Z.reduce((s, Zi) => s + Zi[a]![b]!, 0)));
  return worst;
}

/** Point-dipole interaction k [μ₁·μ₂ − 3(μ₁·r̂)(μ₂·r̂)]/r³, r from dipole 1 to dipole 2. */
export function dipoleDipole(mu1: Vec3, mu2: Vec3, r: Vec3, k: number): number {
  const R = Math.hypot(...r);
  const u: Vec3 = [r[0] / R, r[1] / R, r[2] / R];
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return (k * (dot(mu1, mu2) - 3 * dot(mu1, u) * dot(mu2, u))) / R ** 3;
}

/** Least-squares slope through the origin and Pearson r of y against x. */
export function slopeThroughOrigin(x: readonly number[], y: readonly number[]): { slope: number; r: number } {
  const n = x.length;
  const sxy = x.reduce((a, xi, i) => a + xi * y[i]!, 0);
  const sxx = x.reduce((a, xi) => a + xi * xi, 0);
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let cxy = 0;
  let cxx = 0;
  let cyy = 0;
  for (let i = 0; i < n; i++) {
    cxy += (x[i]! - mx) * (y[i]! - my);
    cxx += (x[i]! - mx) ** 2;
    cyy += (y[i]! - my) ** 2;
  }
  return { slope: sxy / sxx, r: cxy / Math.sqrt(cxx * cyy) };
}

// ---------------------------------------------------------------------------
// Reading the stored passes

/** Receptive field of the local part: each message-passing layer reaches one cutoff further. */
export const receptiveField = (hp: LesHyper) => hp.num_interactions * hp.r_max;

export function energySplit(sys: LesSystem) {
  return { eSr: sys.eSr, eLr: sys.eLr, total: sys.eSr + sys.eLr, lrShare: sys.eLr / (sys.eSr + sys.eLr) };
}

/** Per-atom force magnitudes of the local and long-range parts. */
export function forceSplit(sys: LesSystem) {
  return sys.fSr.map((f, i) => ({ sr: Math.hypot(...f), lr: Math.hypot(...sys.fLr[i]!) }));
}

/** Rotation test on stored data: invariants unchanged, vectors co-rotated. */
export function rotationCheck(sys: LesSystem) {
  const r = sys.rotated;
  if (!r) return null;
  const maxAbs = (xs: number[]) => Math.max(...xs.map(Math.abs));
  const vecErr = (a: Vec3[], b: Vec3[]) =>
    Math.max(...a.map((v, i) => Math.max(...matVec(r.R, v).map((x, k) => Math.abs(x - b[i]![k]!)))));
  return {
    dE: Math.abs(r.eSr + r.eLr - sys.eSr - sys.eLr),
    dELr: Math.abs(r.eLr - sys.eLr),
    dq: maxAbs(r.q.map((x, i) => x - sys.q[i]!)),
    dFsr: vecErr(sys.fSr, r.fSr),
    dFlr: vecErr(sys.fLr, r.fLr),
    dP: vecErr([sys.P], [r.P]),
  };
}

/**
 * Once no atom of one water is within r_c of the other, the graph splits into two components and
 * no number of message-passing layers can couple them: local energies and charges are the
 * monomers’ exactly, while E_lr is still non-zero. Returns those rows with the deviations.
 */
export function separatedRows(doc: LesDoc) {
  const rf = doc.model.hyperparameters.r_max;
  const [A, B] = doc.dimer.monomers as [LesDoc['dimer']['monomers'][0], LesDoc['dimer']['monomers'][0]];
  const qMono = [...A.q, ...B.q];
  return doc.dimer.rows
    .filter((row) => row.minGap > rf)
    .map((row) => ({
      dOO: row.dOO,
      dEsr: Math.abs(row.eSr - A.eSr - B.eSr),
      dq: Math.max(...row.q.map((x, i) => Math.abs(x - qMono[i]!))),
      eLrInter: row.eLr - A.eLr - B.eLr,
    }));
}

/** Monomer positions of the scan at O–O distance d (monomer B shifted rigidly along the O–O axis). */
export function dimerPositions(doc: LesDoc, dOO: number): Vec3[] {
  const s = dOO - doc.dimer.d0;
  const ax = doc.dimer.axis;
  return doc.dimer.positions.map((p, i) => (i < 3 ? p : ([p[0] + s * ax[0], p[1] + s * ax[1], p[2] + s * ax[2]] as Vec3)));
}

/**
 * Multipole expansion of the latent interaction between the two waters about their O atoms, from
 * the isolated monomers' latent charges: charge–charge (1/R), charge–dipole (1/R²) and
 * dipole–dipole (1/R³) terms. Valid once the Gaussians no longer overlap (erf → 1).
 */
export function dimerMultipoles(doc: LesDoc, dOO: number) {
  const k = coulombConstant(doc.model.hyperparameters.les.norm_factor);
  const pos = dimerPositions(doc, dOO);
  const mono = doc.dimer.monomers;
  const moments = [0, 1].map((m) => {
    const idx = [3 * m, 3 * m + 1, 3 * m + 2];
    const q = mono[m]!.q;
    const O = pos[idx[0]!]!;
    const Q = netCharge(q);
    const p = idx.reduce<Vec3>((acc, i, kk) => [0, 1, 2].map((c) => acc[c]! + q[kk]! * (pos[i]![c]! - O[c]!)) as Vec3, [0, 0, 0]);
    return { Q, p, O };
  }) as [{ Q: number; p: Vec3; O: Vec3 }, { Q: number; p: Vec3; O: Vec3 }];
  const [A, B] = moments;
  const R: Vec3 = [B.O[0] - A.O[0], B.O[1] - A.O[1], B.O[2] - A.O[2]];
  const r = Math.hypot(...R);
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const qq = (k * A.Q * B.Q) / r;
  const qp = (k * (B.Q * dot(A.p, R) - A.Q * dot(B.p, R))) / r ** 3;
  const pp = dipoleDipole(A.p, B.p, R, k);
  return { qq, qp, pp, total: qq + qp + pp, QA: A.Q, QB: B.Q };
}

export type LesStepId = 'h1' | 'h2' | 'q' | 'phi' | 'energy';

/** Per-atom values for the 3D view of each step. */
export function stepValues(sys: LesSystem, id: LesStepId, part: 0 | 1 | 2): { values: number[]; signed: boolean; unit: string } {
  switch (id) {
    case 'h1':
      return { values: part === 1 ? sys.features.h1Vector : sys.features.h1Scalar, signed: false, unit: '‖h‖' };
    case 'h2':
      return { values: sys.features.h2Scalar, signed: false, unit: '‖h‖' };
    case 'q':
      return { values: part === 2 ? sys.q : sys.qLayers[part], signed: true, unit: 'latent e' };
    case 'phi':
      return { values: sys.phi, signed: true, unit: 'V' };
    case 'energy':
      return { values: sys.q.map((qi, i) => 0.5 * qi * sys.phi[i]!), signed: true, unit: 'eV' };
  }
}
