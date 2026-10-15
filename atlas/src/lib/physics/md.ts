// Physics Lab P12: classical MD diagnostics and the LAMMPS inputs that drive the
// in-browser runs. Everything here is pure and tested (md.test.ts); the WASM
// LAMMPS engine only integrates the equations of motion.
//
// Units are LAMMPS `lj` reduced units throughout: length σ, energy ε, mass m,
// time τ = σ√(m/ε), temperature ε/k_B.

import { polynomialEnvelope } from '~/lib/cutoffs';

/** How the pair potential is switched off at r_c. */
export type CutoffTreatment = 'truncated' | 'shifted' | 'smooth';

export const TREATMENTS: { id: CutoffTreatment; label: string; short: string }[] = [
  { id: 'truncated', label: 'Truncated', short: 'E and F jump at r_c' },
  { id: 'shifted', label: 'Shifted', short: 'E continuous, F jumps' },
  { id: 'smooth', label: 'Smooth envelope', short: 'E, F, F′ continuous' },
];

/** The LAMMPS `bench/in.lj` state point, shrunk to a small box. */
export const LJ_STATE = {
  density: 0.8442,
  temperature: 1.44,
  seed: 87287,
  rc: 2.5,
  cells: 4,
  /** Envelope order (the DimeNet / NequIP default). */
  p: 6,
} as const;

/** Interval between energy samples and between rendered frames (τ). */
export const SAMPLE_DT = 0.02;
export const FRAME_DT = 0.1;
export const TIMESTEPS = [0.001, 0.002, 0.005, 0.01, 0.02] as const;

export interface PairValue {
  /** Energy (ε). */
  e: number;
  /** Radial force −dE/dr (ε/σ); positive is repulsive. */
  f: number;
}

/** Plain 12-6 Lennard-Jones, ε = σ = 1. */
export function lj(r: number): PairValue {
  const s6 = r ** -6;
  return { e: 4 * (s6 * s6 - s6), f: (24 * (2 * s6 * s6 - s6)) / r };
}

/** The pair potential each treatment actually integrates (zero beyond r_c). */
export function pairPotential(r: number, treatment: CutoffTreatment, rc: number = LJ_STATE.rc, p: number = LJ_STATE.p): PairValue {
  if (r >= rc) return { e: 0, f: 0 };
  const v = lj(r);
  if (treatment === 'truncated') return v;
  if (treatment === 'shifted') return { e: v.e - lj(rc).e, f: v.f };
  const u = polynomialEnvelope(r, rc, p);
  // E = V·u  ⇒  −dE/dr = −V′u − V u′ = f_V·u − V·u′
  return { e: v.e * u.f, f: v.f * u.f - v.e * u.df };
}

/** What a pair crossing r_c does to the energy and force of each treatment. */
export function cutoffJump(treatment: CutoffTreatment, rc: number = LJ_STATE.rc): PairValue {
  const inside = treatment === 'smooth' ? { e: 0, f: 0 } : pairPotential(rc * (1 - 1e-12), treatment, rc);
  return { e: inside.e, f: inside.f };
}

/**
 * A LAMMPS `pair_style table` file for the smooth envelope: N points evenly spaced
 * in r from rlo to rc (the R keyword), columns index, r, E, F with F = −dE/dr.
 */
export function smoothTableFile(keyword = 'LJ_SMOOTH', n = 4000, rlo = 0.5, rc: number = LJ_STATE.rc, p: number = LJ_STATE.p): string {
  const lines = [
    `# 12-6 Lennard-Jones times the polynomial envelope (p = ${p}) vanishing at r_c = ${rc}; lj units`,
    '',
    keyword,
    `N ${n} R ${rlo} ${rc}`,
    '',
  ];
  for (let i = 0; i < n; i++) {
    const r = rlo + ((rc - rlo) * i) / (n - 1);
    const v = pairPotential(Math.min(r, rc * (1 - 1e-12)), 'smooth', rc, p);
    lines.push(`${i + 1} ${r.toPrecision(10)} ${v.e.toExponential(12)} ${v.f.toExponential(12)}`);
  }
  return lines.join('\n') + '\n';
}

export const TABLE_PATH = '/work/lj_smooth.table';

export interface RunConfig {
  treatment: CutoffTreatment;
  /** Time step (τ). */
  dt: number;
  cells?: number;
  seed?: number;
}

/** LAMMPS input for one side: fcc lattice at ρ = 0.8442, velocities at T = 1.44, NVE. */
export function lammpsInput({ treatment, dt, cells = LJ_STATE.cells, seed = LJ_STATE.seed }: RunConfig): string {
  const { density, temperature, rc } = LJ_STATE;
  const pair =
    treatment === 'smooth'
      ? [`pair_style table spline 4000`, `pair_coeff 1 1 ${TABLE_PATH} LJ_SMOOTH ${rc}`]
      : [`pair_style lj/cut ${rc}`, `pair_coeff 1 1 1.0 1.0 ${rc}`, `pair_modify shift ${treatment === 'shifted' ? 'yes' : 'no'}`];
  return [
    'units lj',
    'atom_style atomic',
    'boundary p p p',
    `lattice fcc ${density}`,
    `region box block 0 ${cells} 0 ${cells} 0 ${cells}`,
    'create_box 1 box',
    'create_atoms 1 box',
    'mass 1 1.0',
    `velocity all create ${temperature} ${seed} loop geom`,
    ...pair,
    'neighbor 0.3 bin',
    'neigh_modify every 1 delay 0 check yes',
    `timestep ${dt}`,
    'fix 1 all nve',
    'thermo_style custom step temp pe ke etotal',
    'thermo_modify norm yes',
    'thermo 100000000',
    'run 0',
  ].join('\n');
}

/** Steps per energy sample for a time step that divides SAMPLE_DT. */
export function stepsPer(interval: number, dt: number): number {
  const n = Math.round(interval / dt);
  if (n < 1 || Math.abs(n * dt - interval) > 1e-9) throw new Error(`time step ${dt} does not divide ${interval}`);
  return n;
}

/** Cubic box edge for an fcc block of `cells`³ conventional cells at number density ρ. */
export function boxLength(cells: number = LJ_STATE.cells, density: number = LJ_STATE.density): number {
  return cells * Math.cbrt(4 / density);
}

/** fcc lattice sites of the starting structure (the poster before LAMMPS loads). */
export function fccSites(cells: number = LJ_STATE.cells, density: number = LJ_STATE.density): Float64Array {
  const a = Math.cbrt(4 / density);
  const basis = [
    [0, 0, 0],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
  ];
  const out = new Float64Array(cells ** 3 * 4 * 3);
  let k = 0;
  for (let z = 0; z < cells; z++)
    for (let y = 0; y < cells; y++)
      for (let x = 0; x < cells; x++)
        for (const b of basis) {
          out[k++] = (x + b[0]!) * a;
          out[k++] = (y + b[1]!) * a;
          out[k++] = (z + b[2]!) * a;
        }
  return out;
}

const mic = (d: number, L: number) => d - L * Math.round(d / L);

/**
 * Follows atoms across periodic boundaries: each new frame is unwrapped against the
 * previous one with the minimum-image convention, which is exact while no atom moves
 * more than L/2 between frames.
 */
export class Unwrapper {
  readonly L: number;
  private prevRaw: Float64Array;
  private current: Float64Array;
  constructor(L: number, first: Float64Array) {
    this.L = L;
    this.prevRaw = first.slice();
    this.current = first.slice();
  }
  push(raw: Float64Array): Float64Array {
    for (let i = 0; i < raw.length; i++) {
      this.current[i]! += mic(raw[i]! - this.prevRaw[i]!, this.L);
      this.prevRaw[i] = raw[i]!;
    }
    return this.current.slice();
  }
}

/** Mean squared displacement between two unwrapped frames (σ²). */
export function msd(ref: Float64Array, cur: Float64Array): number {
  let s = 0;
  for (let i = 0; i < ref.length; i++) s += (cur[i]! - ref[i]!) ** 2;
  return s / (ref.length / 3);
}

/** Per-atom displacement |r_i(t) − r_i(0)| (σ). */
export function displacements(ref: Float64Array, cur: Float64Array): Float64Array {
  const n = ref.length / 3;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++)
    out[i] = Math.hypot(cur[3 * i]! - ref[3 * i]!, cur[3 * i + 1]! - ref[3 * i + 1]!, cur[3 * i + 2]! - ref[3 * i + 2]!);
  return out;
}

/** Pair-distance histogram for g(r) in a cubic periodic box (minimum image, r < rmax ≤ L/2). */
export function pairHistogram(pos: Float64Array, L: number, rmax: number, nbins: number): Float64Array {
  if (rmax > L / 2 + 1e-12) throw new Error('rmax must not exceed L/2');
  const n = pos.length / 3;
  const h = new Float64Array(nbins);
  const dr = rmax / nbins;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = mic(pos[3 * j]! - pos[3 * i]!, L);
      const dy = mic(pos[3 * j + 1]! - pos[3 * i + 1]!, L);
      const dz = mic(pos[3 * j + 2]! - pos[3 * i + 2]!, L);
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r < rmax) h[Math.floor(r / dr)]! += 1;
    }
  return h;
}

/**
 * Radial distribution function from one or more pair histograms (summed over
 * `frames`): g(r) = counts / [frames · N(N−1)/2 · (4π/3)(r₂³ − r₁³) / V].
 */
export function rdfFromHistogram(h: Float64Array, frames: number, n: number, L: number, rmax: number): { r: number[]; g: number[] } {
  const dr = rmax / h.length;
  const pairs = (n * (n - 1)) / 2;
  const V = L ** 3;
  const r: number[] = [];
  const g: number[] = [];
  for (let k = 0; k < h.length; k++) {
    const r1 = k * dr;
    const r2 = r1 + dr;
    const shell = ((4 * Math.PI) / 3) * (r2 ** 3 - r1 ** 3);
    r.push(r1 + dr / 2);
    g.push(frames > 0 ? h[k]! / (frames * pairs * (shell / V)) : 0);
  }
  return { r, g };
}

/** Ordinary least squares y = a + b·x. */
export function linearFit(x: ArrayLike<number>, y: ArrayLike<number>): { slope: number; intercept: number } {
  const n = x.length;
  if (n < 2) return { slope: 0, intercept: n ? y[0]! : 0 };
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i]!;
    sy += y[i]!;
  }
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i]! - mx) * (y[i]! - my);
    sxx += (x[i]! - mx) ** 2;
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return { slope, intercept: my - slope * mx };
}

export interface EnergyDiagnostics {
  /** Linear drift of the total energy per atom (ε/τ). */
  drift: number;
  /** RMS deviation of the total energy per atom about the drift line (ε). */
  rms: number;
  /** Largest |E(t) − E(0)| per atom (ε). */
  maxDev: number;
}

export function energyDiagnostics(t: ArrayLike<number>, e: ArrayLike<number>): EnergyDiagnostics {
  const { slope, intercept } = linearFit(t, e);
  let ss = 0;
  let maxDev = 0;
  for (let i = 0; i < t.length; i++) {
    ss += (e[i]! - intercept - slope * t[i]!) ** 2;
    maxDev = Math.max(maxDev, Math.abs(e[i]! - e[0]!));
  }
  return { drift: slope, rms: t.length ? Math.sqrt(ss / t.length) : 0, maxDev };
}

/** Einstein relation D = MSD/(6t), fitted on the later part of the curve (σ²/τ). */
export function diffusionCoefficient(t: ArrayLike<number>, m: ArrayLike<number>, fromFraction = 0.5): number {
  const start = Math.floor(t.length * fromFraction);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = start; i < t.length; i++) {
    xs.push(t[i]!);
    ys.push(m[i]!);
  }
  return linearFit(xs, ys).slope / 6;
}

/** Least-squares slope of log y against log x (the power in y ∝ xᵏ). */
export function logLogSlope(x: number[], y: number[]): number {
  const keep = x.map((_, i) => i).filter((i) => x[i]! > 0 && y[i]! > 0);
  return linearFit(
    keep.map((i) => Math.log(x[i]!)),
    keep.map((i) => Math.log(y[i]!)),
  ).slope;
}

/**
 * Velocity Verlet on a unit harmonic oscillator, H = (p² + q²)/2, from (q, p) = (1, 0).
 * Returns the largest |H − H₀| over the run: the integrator's energy error.
 */
export function verletOscillatorError(dt: number, tEnd: number): number {
  let q = 1;
  let p = 0;
  let worst = 0;
  const steps = Math.round(tEnd / dt);
  for (let i = 0; i < steps; i++) {
    p -= 0.5 * dt * q;
    q += dt * p;
    p -= 0.5 * dt * q;
    worst = Math.max(worst, Math.abs(0.5 * (p * p + q * q) - 0.5));
  }
  return worst;
}

/** The LJ time unit τ = σ√(m/ε) in picoseconds, for σ in Å, ε/k_B in K, m in u. */
export function ljTimePs(sigmaA: number, epsKelvin: number, massAmu: number): number {
  const kB = 1.380649e-23;
  const amu = 1.66053906660e-27;
  return sigmaA * 1e-10 * Math.sqrt((massAmu * amu) / (epsKelvin * kB)) * 1e12;
}
