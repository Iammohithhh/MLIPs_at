// Model Observatory analysis: every number the page shows is computed here from the
// precomputed model outputs in src/data/observatory/observatory.json (written by
// pipeline/atlas_pipeline/observatory.py). Nothing in this module ranks models; it
// measures where they agree and where they do not.

export type Vec3 = [number, number, number];

export interface ObsFrame {
  symbols: string[];
  positions: Vec3[];
  cell: [Vec3, Vec3, Vec3];
  pbc: boolean[];
}

export interface ObsModel {
  key: string;
  name: string;
  family: string;
  checkpoint: string;
  url: string;
  licence: string;
  dtype: string;
  forces: 'conservative' | 'direct';
  sha256: string;
  versions: Record<string, string>;
  cutoff: number | null;
  /** Cap on neighbours per atom, if the graph has one. */
  neighbours?: number | null;
  seconds: number;
  results: {
    gallery: Record<string, { energy: number; forces: Vec3[]; stress?: number[] }>;
    'ev-cu': { energy: number[] };
    'ev-si': { energy: number[] };
    'dimer-si': { energy: number[]; force: number[] };
    'dimer-cu': { energy: number[]; force: number[] };
    'na-au': { energy: number[]; fz: number[] };
    rotation: { energy: [number, number]; forces: [Vec3[], Vec3[]] };
    loop: { energy: number[]; power: number[] };
  };
}

export interface ObsDoc {
  format: 'atlas-observatory';
  version: number;
  settings: {
    rattle: { sigma: number; seed: number };
    box: number;
    rotation: { axis: Vec3; angleDeg: number };
    loop: { points: number; amplitude: number; seed: number };
    units: Record<string, string>;
  };
  inputs: {
    gallery: (ObsFrame & { id: string })[];
    'ev-cu': { x: number[]; atoms: number; volume0: number };
    'ev-si': { x: number[]; atoms: number; volume0: number };
    'dimer-si': { x: number[] };
    'dimer-cu': { x: number[] };
    'na-au': { x: number[]; site: string; frame: ObsFrame };
    rotation: { R: [Vec3, Vec3, Vec3]; frames: [ObsFrame, ObsFrame] };
    loop: { theta: number[]; atoms: number };
  };
  models: ObsModel[];
}

/** eV/Å³ → GPa. */
export const EV_PER_A3_IN_GPA = 160.21766208;

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const matVec = (R: [Vec3, Vec3, Vec3], v: Vec3): Vec3 => [
  R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
  R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
  R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
];

// ---------------------------------------------------------------------------
// Forces: consensus, spread (the disagreement halo) and pairwise differences

export interface ForceAgreement {
  /** Mean force over models, per atom (eV/Å). */
  mean: Vec3[];
  /** Per-atom spread √(⟨|F_m − F̄|²⟩_m), eV/Å: the radius of the disagreement halo. */
  spread: number[];
  /** Per model: RMS over atoms of |F_m − F̄| (eV/Å). */
  deviation: number[];
  /** Pairwise mean over atoms of |F_a − F_b| (eV/Å), models × models. */
  pairwise: number[][];
  /** Mean |F̄| over atoms (eV/Å), for scale. */
  meanMagnitude: number;
}

export function forceAgreement(sets: Vec3[][]): ForceAgreement {
  const M = sets.length;
  if (M === 0) throw new Error('no models');
  const N = sets[0]!.length;
  if (sets.some((s) => s.length !== N)) throw new Error('models disagree on the number of atoms');
  const mean: Vec3[] = Array.from({ length: N }, (_, i) => {
    const s: Vec3 = [0, 0, 0];
    for (const f of sets) for (let k = 0; k < 3; k++) s[k] = s[k]! + f[i]![k]! / M;
    return s;
  });
  const spread = mean.map((m, i) => Math.sqrt(sets.reduce((acc, f) => acc + norm(sub(f[i]!, m)) ** 2, 0) / M));
  const deviation = sets.map((f) => Math.sqrt(f.reduce((acc, v, i) => acc + norm(sub(v, mean[i]!)) ** 2, 0) / N));
  const pairwise = sets.map((a) => sets.map((b) => a.reduce((acc, v, i) => acc + norm(sub(v, b[i]!)), 0) / N));
  const meanMagnitude = mean.reduce((acc, v) => acc + norm(v), 0) / N;
  return { mean, spread, deviation, pairwise, meanMagnitude };
}

/** Angle in degrees between two force vectors (0 when either is zero). */
export function angleDeg(a: Vec3, b: Vec3): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
}

// ---------------------------------------------------------------------------
// Curves

/** Energies relative to a reference point, optionally per atom. */
export function relative(energies: number[], ref: number, perAtom = 1): number[] {
  const e0 = energies[ref]!;
  return energies.map((e) => (e - e0) / perAtom);
}

/** Index and value of the minimum. */
export function argmin(y: number[]): { i: number; y: number } {
  let i = 0;
  for (let k = 1; k < y.length; k++) if (y[k]! < y[i]!) i = k;
  return { i, y: y[i]! };
}

/** Solve a small dense linear system (Gaussian elimination with partial pivoting). */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    [M[c], M[p]] = [M[p]!, M[c]!];
    const piv = M[c]![c]!;
    if (Math.abs(piv) < 1e-300) throw new Error('singular system');
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r]![c]! / piv;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  return M.map((r, i) => r[n]! / r[i]!);
}

export interface EosFit {
  /** Equilibrium volume per atom, Å³. */
  v0: number;
  /** Equilibrium lattice scale relative to the gallery cell. */
  scale0: number;
  /** Bulk modulus B0 = V d²E/dV² at V0, GPa. */
  b0: number;
  /** Energy per atom at V0, eV. */
  e0: number;
  /** RMS residual of the fit, meV/atom. */
  rmsMeV: number;
  /** Fitted coefficients of E(x) = Σ c_k x^k with x = V^(−2/3). */
  coef: number[];
}

/**
 * Third-order Birch–Murnaghan fit, done exactly as a cubic polynomial in x = V^(−2/3)
 * (the BM energy is a cubic in that variable). Volumes and energies per atom.
 */
export function birchMurnaghan(volumes: number[], energies: number[]): EosFit {
  const x = volumes.map((v) => v ** (-2 / 3));
  const P = 4;
  const AtA = Array.from({ length: P }, (_, i) => Array.from({ length: P }, (_, j) => x.reduce((s, xi) => s + xi ** (i + j), 0)));
  const Atb = Array.from({ length: P }, (_, i) => x.reduce((s, xi, k) => s + xi ** i * energies[k]!, 0));
  const c = solve(AtA, Atb);
  // dE/dx = c1 + 2 c2 x + 3 c3 x² = 0; take the root inside the sampled range.
  const [c0, c1, c2, c3] = c as [number, number, number, number];
  const disc = 4 * c2 * c2 - 12 * c3 * c1;
  const roots = disc >= 0 ? [(-2 * c2 + Math.sqrt(disc)) / (6 * c3), (-2 * c2 - Math.sqrt(disc)) / (6 * c3)] : [];
  const lo = Math.min(...x);
  const hi = Math.max(...x);
  const d2 = (xx: number) => 2 * c2 + 6 * c3 * xx;
  const xs = roots.filter((r) => Number.isFinite(r) && r >= lo * 0.97 && r <= hi * 1.03 && d2(r) > 0);
  if (!xs.length) throw new Error('no minimum inside the sampled volumes');
  const x0 = xs[0]!;
  const v0 = x0 ** -1.5;
  // d²E/dV² = E''(x) x'² + E'(x) x''; E'(x0) = 0 → only the first term.
  const dxdv = (-2 / 3) * v0 ** (-5 / 3);
  const b0 = v0 * d2(x0) * dxdv * dxdv * EV_PER_A3_IN_GPA;
  const E = (xx: number) => c0 + c1 * xx + c2 * xx * xx + c3 * xx * xx * xx;
  const rms = Math.sqrt(x.reduce((s, xi, k) => s + (E(xi) - energies[k]!) ** 2, 0) / x.length) * 1000;
  return { v0, scale0: NaN, b0, e0: E(x0), rmsMeV: rms, coef: c };
}

/** E(V) fit for an isotropic lattice-scale scan. */
export function eosFromScan(scales: number[], energies: number[], atoms: number, volume0: number): EosFit {
  const vols = scales.map((s) => (volume0 * s ** 3) / atoms);
  const fit = birchMurnaghan(
    vols,
    energies.map((e) => e / atoms),
  );
  return { ...fit, scale0: Math.cbrt((fit.v0 * atoms) / volume0) };
}

// ---------------------------------------------------------------------------
// Closed-loop test of conservativeness

export interface LoopTest {
  /** ∮ F · dx around the loop, eV (zero for a gradient field). */
  circulation: number;
  /** Energy reconstructed from the work −∫F·dx, relative to θ = 0, eV. */
  workEnergy: number[];
  /** Model energy relative to θ = 0, eV. */
  energy: number[];
  /** Largest |workEnergy − energy| along the loop, eV. */
  maxGap: number;
  /** Peak-to-peak energy variation along the loop, eV, for scale. */
  energyRange: number;
}

/**
 * Integrates P(θ) = F(θ) · dx/dθ around a closed loop sampled at n equally spaced θ.
 * The periodic trapezoid rule is spectrally accurate for a smooth periodic integrand.
 */
export function loopTest(theta: number[], power: number[], energy: number[]): LoopTest {
  const n = theta.length;
  if (n < 3 || power.length !== n || energy.length !== n) throw new Error('bad loop');
  const h = (2 * Math.PI) / n;
  const circulation = power.reduce((s, p) => s + p, 0) * h;
  const workEnergy = [0];
  for (let k = 1; k < n; k++) workEnergy.push(workEnergy[k - 1]! - 0.5 * h * (power[k - 1]! + power[k]!));
  const e = energy.map((v) => v - energy[0]!);
  const maxGap = Math.max(...e.map((v, k) => Math.abs(v - workEnergy[k]!)));
  return {
    circulation,
    workEnergy,
    energy: e,
    maxGap,
    energyRange: Math.max(...e) - Math.min(...e),
  };
}

// ---------------------------------------------------------------------------
// Rotation test

export interface RotationTest {
  /** |E(Rx) − E(x)|, eV. */
  dE: number;
  /** Max over atoms of |F(Rx) − R F(x)|, eV/Å. */
  maxForceError: number;
  /** RMS over atoms of |F(Rx) − R F(x)| divided by the RMS force, dimensionless. */
  relForceError: number;
  /** Per-atom |F(Rx) − R F(x)|, eV/Å. */
  perAtom: number[];
}

export function rotationTest(R: [Vec3, Vec3, Vec3], energy: [number, number], forces: [Vec3[], Vec3[]]): RotationTest {
  const [f0, f1] = forces;
  const perAtom = f0.map((f, i) => norm(sub(f1[i]!, matVec(R, f))));
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  return {
    dE: Math.abs(energy[1] - energy[0]),
    maxForceError: Math.max(...perAtom),
    relForceError: rms(perAtom) / rms(f0.map(norm)),
    perAtom,
  };
}

// ---------------------------------------------------------------------------
// Dimer and adsorption curves

export interface CurveSummary {
  /** Position of the minimum (Å). */
  xMin: number;
  /** Depth of the minimum relative to the last point (eV). */
  depth: number;
  /** Energy at the shortest distance relative to the last point (eV). */
  eShort: number;
  /** Number of local minima inside the curve (more than one is a spurious well). */
  minima: number;
  /** Positions where the force changes sign from repulsive to attractive inside the repulsive wall. */
  wrongWay: number[];
}

/**
 * Summary of a curve E(x) sampled on increasing x. `force` is the force on the moving
 * atom along +x (positive = pushed outward). Inside the repulsive wall (x below the
 * minimum) a physical curve pushes outward; a negative force there pulls atoms together.
 */
export function curveSummary(x: number[], energy: number[], force?: number[]): CurveSummary {
  const n = x.length;
  const ref = energy[n - 1]!;
  const rel = energy.map((e) => e - ref);
  const m = argmin(rel);
  let minima = 0;
  for (let k = 1; k < n - 1; k++) if (rel[k]! < rel[k - 1]! && rel[k]! <= rel[k + 1]!) minima++;
  const wrongWay = force ? x.filter((_, k) => k < m.i && force[k]! < 0) : [];
  return { xMin: x[m.i]!, depth: m.y, eShort: rel[0]!, minima, wrongWay };
}

// ---------------------------------------------------------------------------
// Stress

/** Pressure P = −tr(σ)/3 in GPa from an ASE Voigt stress (eV/Å³, positive = tensile). */
export function pressureGPa(stress: number[]): number {
  return (-(stress[0]! + stress[1]! + stress[2]!) / 3) * EV_PER_A3_IN_GPA;
}

/** Gallery results of every model for one structure, in model order. */
export function galleryForces(doc: ObsDoc, id: string): Vec3[][] {
  return doc.models.map((m) => {
    const r = m.results.gallery[id];
    if (!r) throw new Error(`${m.key} has no result for ${id}`);
    return r.forces;
  });
}

/**
 * Compare a stored force along a scan with the slope of the same model's energy:
 * |−dE/dx − F| at interior points, dE/dx by central differences. For F = −∇E this is
 * only the O(h²) finite-difference error; for a direct-force model it can be large.
 */
export function slopeMismatch(x: number[], energy: number[], force: number[], from = -Infinity): { x: number; diff: number }[] {
  const out: { x: number; diff: number }[] = [];
  for (let k = 1; k < x.length - 1; k++) {
    if (x[k]! < from) continue;
    const slope = (energy[k + 1]! - energy[k - 1]!) / (x[k + 1]! - x[k - 1]!);
    out.push({ x: x[k]!, diff: Math.abs(-slope - force[k]!) });
  }
  return out;
}
