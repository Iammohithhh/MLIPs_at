// Experiment vs DFT reference (Failure Atlas): what a model inherits from its training functional,
// and what the UniFFBench density results say about it.
//
// THEORY: under isotropic strain the density scales as ρ ∝ 1/V ∝ a⁻³, so a relative lattice error
// ε gives a density error (1 + ε)⁻³ − 1 ≈ −3ε. Errors relative to experiment compose
// multiplicatively: a model that reproduces its functional exactly inherits the functional's error,
// and its own fit error to that functional comes on top.

/** Mean relative error of the equilibrium lattice constant against experiment corrected for thermal
 * and zero-point effects, 44 strongly bound solids (Tran, Stelzl & Blaha 2016, Table II, "MRE" for a₀;
 * same table, MRE of B₀). Values in %. */
export const FUNCTIONALS = [
  { id: 'lda', name: 'LDA', a0MRE: -1.5, b0MRE: 8.1 },
  { id: 'pbesol', name: 'PBEsol', a0MRE: -0.1, b0MRE: -1.4 },
  { id: 'scan', name: 'SCAN', a0MRE: 0.3, b0MRE: -0.4 },
  { id: 'pbe', name: 'PBE', a0MRE: 1.1, b0MRE: -9.8 },
] as const;

export type FunctionalId = (typeof FUNCTIONALS)[number]['id'];

/** Relative density error (fraction) from a relative lattice error ε (fraction), isotropic strain. */
export const densityFromLattice = (eps: number) => (1 + eps) ** -3 - 1;

/** Relative lattice error (fraction) that produces a relative density error d (fraction). */
export const latticeFromDensity = (d: number) => (1 + d) ** (-1 / 3) - 1;

/** Total lattice error relative to experiment when a model with fit error εfit (relative to its
 * reference) is trained on a reference whose own error is εref. */
export const composeLattice = (epsRef: number, epsFit: number) => (1 + epsRef) * (1 + epsFit) - 1;

/** The density error budget, all in % relative to experiment. */
export function densityBudget(refLatticePct: number, fitLatticePct: number) {
  const r = refLatticePct / 100;
  const f = fitLatticePct / 100;
  const total = composeLattice(r, f);
  return {
    reference: 100 * densityFromLattice(r),
    fit: 100 * (densityFromLattice(total) - densityFromLattice(r)),
    totalLattice: 100 * total,
    total: 100 * densityFromLattice(total),
  };
}

// ---------------------------------------------------------------------------
// UniFFBench per-run densities (src/data/benchmarks/uniffbench-density.json)

export interface UniffRun {
  mineral: string;
  T: number;
  p_field: number;
  rho_exp: number;
  rho_sim: number;
}

export interface UniffModel {
  key: string;
  name: string;
  file: string;
  loader: string;
  runs: UniffRun[];
}

/** Signed density error of one run, % ((ρ_sim − ρ_exp)/ρ_exp, the paper's Eq. 1 with ρ_i the experimental density). */
export const signedDensityError = (r: UniffRun) => (100 * (r.rho_sim - r.rho_exp)) / r.rho_exp;

export function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

export interface DensitySummary {
  runs: number;
  /** Runs kept after the paper's rule: an error above 100% counts as a failed simulation. */
  kept: number;
  median: number;
  /** Mean absolute percentage error over kept runs. */
  mape: number;
  /** Fraction of kept runs whose simulated density is below the experimental one. */
  fracUnder: number;
  /** Fraction of kept runs within ±threshold %. */
  fracWithin: number;
  threshold: number;
}

export function summarise(runs: UniffRun[], threshold = 2.5, failAbove = 100): DensitySummary {
  const e = runs.map(signedDensityError).filter((x) => Math.abs(x) <= failAbove);
  const n = e.length;
  return {
    runs: runs.length,
    kept: n,
    median: median(e),
    mape: e.reduce((s, x) => s + Math.abs(x), 0) / n,
    fracUnder: e.filter((x) => x < 0).length / n,
    fracWithin: e.filter((x) => Math.abs(x) <= threshold).length / n,
    threshold,
  };
}

/** Counts of errors in bins [lo + i·w, lo + (i+1)·w); values outside go to the end bins' `below`/`above`. */
export function histogram(xs: number[], lo: number, hi: number, width: number) {
  const nb = Math.round((hi - lo) / width);
  const counts = new Array<number>(nb).fill(0);
  let below = 0;
  let above = 0;
  for (const x of xs) {
    if (x < lo) below++;
    else if (x >= hi) above++;
    else counts[Math.min(nb - 1, Math.floor((x - lo) / width))]!++;
  }
  return { lo, hi, width, counts, below, above };
}
