// A linear ACE fitted in the browser to energies and forces of small periodic silicon cells whose
// "DFT" is a classical potential, so every reference value is exact and the question is only what
// the basis can represent: Stillinger–Weber (two- plus three-body) or Tersoff (bond order: all body
// orders). The site energy is E_i = c₀ + Σ_v c_v B_v(i) (./ace.ts); forces are its exact derivatives,
// so the fit is one linear least-squares problem with Tikhonov regularisation.

import { mulberry32 } from '../chem/rng';
import { phi2, phi3, SW_CUTOFF } from '../physics/stillinger-weber';
import { tersoffSite, TERSOFF_CUTOFF } from '../physics/tersoff';
import { atomicBase, buildBasis, evaluateBasisWithDerivative, type AceBasis, type AceSpec, type Radial, type Vec3 } from './ace';
import { diamondCell, rattled, strained, type Cell } from './gap';

export type Reference = 'sw' | 'tersoff';

export const REFERENCE_CUTOFF: Record<Reference, number> = { sw: SW_CUTOFF, tersoff: TERSOFF_CUTOFF };

/** Neighbours of every atom within rc over periodic images, with the index of the atom each vector points to. */
export function neighbourList(c: Cell, rc: number): Array<Array<{ j: number; v: Vec3 }>> {
  const n = Math.ceil(rc / c.L);
  return c.positions.map((pi, i) => {
    const out: Array<{ j: number; v: Vec3 }> = [];
    c.positions.forEach((pj, j) => {
      for (let a = -n; a <= n; a++)
        for (let b = -n; b <= n; b++)
          for (let k = -n; k <= n; k++) {
            if (i === j && a === 0 && b === 0 && k === 0) continue;
            const v: Vec3 = [pj[0] + a * c.L - pi[0], pj[1] + b * c.L - pi[1], pj[2] + k * c.L - pi[2]];
            if (Math.hypot(v[0], v[1], v[2]) < rc) out.push({ j, v });
          }
    });
    return out;
  });
}

/** Stillinger–Weber site energy: half of each pair plus the three-body terms centred on i. */
function swSite(env: Vec3[]): number {
  const r = env.map((v) => Math.hypot(v[0], v[1], v[2]));
  let e = 0;
  for (let j = 0; j < env.length; j++) {
    e += 0.5 * phi2(r[j]!);
    for (let k = j + 1; k < env.length; k++) {
      const u = env[j]!;
      const w = env[k]!;
      e += phi3(r[j]!, r[k]!, (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (r[j]! * r[k]!));
    }
  }
  return e;
}

/** Reference site energies of a cell (eV). */
export function referenceSites(c: Cell, ref: Reference): number[] {
  const nl = neighbourList(c, REFERENCE_CUTOFF[ref] + 1e-9);
  return nl.map((nb) => (ref === 'sw' ? swSite(nb.map((x) => x.v)) : tersoffSite(nb.map((x) => x.v))));
}

export const referenceEnergy = (c: Cell, ref: Reference) => referenceSites(c, ref).reduce((a, b) => a + b, 0);

/** Reference forces by central differences of the total energy (h = 10⁻⁴ Å; error ~10⁻⁷ eV/Å). */
export function referenceForces(c: Cell, ref: Reference, h = 1e-4): Vec3[] {
  return c.positions.map((_, i) => {
    const f: Vec3 = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const p = c.positions.map((x) => [...x] as Vec3);
      const m = c.positions.map((x) => [...x] as Vec3);
      p[i]![a]! += h;
      m[i]![a]! -= h;
      f[a] = -(referenceEnergy({ L: c.L, positions: p }, ref) - referenceEnergy({ L: c.L, positions: m }, ref)) / (2 * h);
    }
    return f;
  });
}

export interface Sample {
  cell: Cell;
  energy: number;
  forces: Vec3[];
  sites: number[];
}

export const sample = (cell: Cell, ref: Reference): Sample => ({ cell, energy: referenceEnergy(cell, ref), forces: referenceForces(cell, ref), sites: referenceSites(cell, ref) });

export interface DataSpec {
  count: number;
  maxStrain: number;
  maxRattle: number;
  seed: number;
}

/** Strained (uniform in ±maxStrain) and rattled (Gaussian, amplitude uniform in [0, maxRattle]) diamond cells. */
export function cells(spec: DataSpec): Cell[] {
  const rand = mulberry32(spec.seed);
  return Array.from({ length: spec.count }, (_, k) => {
    const eps = (2 * rand() - 1) * spec.maxStrain;
    const amp = rand() * spec.maxRattle;
    return rattled(strained(diamondCell(), eps), amp, spec.seed * 1000 + k + 1);
  });
}

// ---------------------------------------------------------------------------
// Features of a cell: Σ_i B(i) and −∂/∂r_j Σ_i B(i)
// ---------------------------------------------------------------------------

export interface CellFeatures {
  natoms: number;
  /** Per-site basis values [i][v]. */
  site: Float64Array[];
  /** Σ_i B_v(i). */
  total: Float64Array;
  /** g[3j + a][v] = −∂(Σ_i B_v(i))/∂r_{j,a}: force on atom j per unit coefficient. */
  force: Float64Array[];
}

export function cellFeatures(basis: AceBasis, R: Radial, c: Cell): CellFeatures {
  const nf = basis.functions.length;
  const nl = neighbourList(c, R.rc);
  const N = c.positions.length;
  const site: Float64Array[] = [];
  const total = new Float64Array(nf);
  const force = Array.from({ length: 3 * N }, () => new Float64Array(nf));
  nl.forEach((nb, i) => {
    const env = nb.map((x) => x.v);
    const { A, dA } = atomicBase(env, basis.spec.nmax, basis.spec.lmax, R, true);
    const { B, dBdA } = evaluateBasisWithDerivative(basis, A);
    site.push(B);
    for (let v = 0; v < nf; v++) total[v]! += B[v]!;
    nb.forEach(({ j }, q) => {
      const g = dA![q]!;
      for (let a = 0; a < 3; a++) {
        const d = g[a]!;
        if (!d.length) continue;
        for (let v = 0; v < nf; v++) {
          const row = dBdA[v]!;
          let s = 0;
          for (let k = 0; k < d.length; k++) s += row[k]! * d[k]!;
          // r_ij = r_j − r_i: ∂/∂r_j = +∂/∂r_ij, ∂/∂r_i = −∂/∂r_ij; force = −gradient
          force[3 * j + a]![v]! -= s;
          force[3 * i + a]![v]! += s;
        }
      }
    });
  });
  return { natoms: N, site, total, force };
}

// ---------------------------------------------------------------------------
// The fit: weighted least squares through the normal equations, so that every sub-basis (lower ν,
// lower l_max) is a column subset of one Gram matrix and costs only a Cholesky solve.
// ---------------------------------------------------------------------------

export interface FitOptions {
  /** Weight of a force component (eV/Å) relative to an energy per atom (eV/atom). */
  forceWeight: number;
  /** Tikhonov strength relative to the mean squared (scaled) column. */
  ridge: number;
}

export interface AceModel {
  c0: number;
  c: Float64Array;
}

export interface Normal {
  /** Columns: 0 = constant (per-atom energy), 1 + v = basis function v. */
  cols: number;
  G: Float64Array;
  g: Float64Array;
  rows: number;
}

/** XᵀX and Xᵀy of [energies per atom; forceWeight × forces]. */
export function normalEquations(train: Array<{ f: CellFeatures; s: Sample }>, o: Pick<FitOptions, 'forceWeight'>): Normal {
  const nf = train[0]!.f.total.length;
  const cols = nf + 1;
  const G = new Float64Array(cols * cols);
  const g = new Float64Array(cols);
  const row = new Float64Array(cols);
  let rows = 0;
  const add = (y: number) => {
    rows++;
    for (let a = 0; a < cols; a++) {
      const ra = row[a]!;
      if (ra === 0) continue;
      g[a]! += ra * y;
      for (let b = a; b < cols; b++) G[a * cols + b]! += ra * row[b]!;
    }
  };
  for (const { f, s } of train) {
    const N = f.natoms;
    row[0] = 1;
    for (let v = 0; v < nf; v++) row[v + 1] = f.total[v]! / N;
    add(s.energy / N);
    if (o.forceWeight > 0)
      for (let k = 0; k < 3 * N; k++) {
        row[0] = 0;
        for (let v = 0; v < nf; v++) row[v + 1] = o.forceWeight * f.force[k]![v]!;
        add(o.forceWeight * s.forces[Math.floor(k / 3)]![k % 3]!);
      }
  }
  for (let a = 0; a < cols; a++) for (let b = 0; b < a; b++) G[a * cols + b] = G[b * cols + a]!;
  return { cols, G, g, rows };
}

/** Solve the ridge problem on a subset of basis functions (the constant is always included). */
export function solveNormal(ne: Normal, keep: number[], ridge: number): AceModel {
  const idx = [0, ...keep.map((v) => v + 1)];
  const n = idx.length;
  const scale = idx.map((a) => Math.sqrt(ne.G[a * ne.cols + a]! / ne.rows) || 1);
  const M = new Float64Array(n * n);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    b[i] = ne.g[idx[i]!]! / scale[i]! / ne.rows;
    for (let j = 0; j < n; j++) M[i * n + j] = ne.G[idx[i]! * ne.cols + idx[j]!]! / (scale[i]! * scale[j]! * ne.rows);
    if (i > 0) M[i * n + i]! += ridge;
  }
  // Cholesky with a tiny jitter if the scaled Gram matrix is singular to rounding
  for (let jitter = 0; ; jitter = jitter ? jitter * 10 : 1e-12) {
    const A = M.slice();
    for (let i = 0; i < n; i++) A[i * n + i]! += jitter;
    const L = new Float64Array(n * n);
    let ok = true;
    for (let i = 0; i < n && ok; i++)
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j]!;
        for (let k = 0; k < j; k++) s -= L[i * n + k]! * L[j * n + k]!;
        if (i === j) {
          if (s <= 0) {
            ok = false;
            break;
          }
          L[i * n + i] = Math.sqrt(s);
        } else L[i * n + j] = s / L[j * n + j]!;
      }
    if (!ok) {
      if (jitter > 1e-3) throw new Error('normal equations are not positive definite');
      continue;
    }
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i]!;
      for (let k = 0; k < i; k++) s -= L[i * n + k]! * y[k]!;
      y[i] = s / L[i * n + i]!;
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i]!;
      for (let k = i + 1; k < n; k++) s -= L[k * n + i]! * x[k]!;
      x[i] = s / L[i * n + i]!;
    }
    const c = new Float64Array(ne.cols - 1);
    keep.forEach((v, q) => (c[v] = x[q + 1]! / scale[q + 1]!));
    return { c0: x[0]! / scale[0]!, c };
  }
}

/** Fit on every basis function. */
export function fitAce(train: Array<{ f: CellFeatures; s: Sample }>, o: FitOptions): AceModel {
  const ne = normalEquations(train, o);
  return solveNormal(ne, Array.from({ length: ne.cols - 1 }, (_, v) => v), o.ridge);
}

export interface Prediction {
  energy: number;
  forces: Vec3[];
  sites: number[];
}

export function predictAce(m: AceModel, f: CellFeatures): Prediction {
  const dot = (a: Float64Array) => a.reduce((s, x, k) => s + x * m.c[k]!, 0);
  const sites = f.site.map((B) => m.c0 + dot(B));
  const forces = Array.from({ length: f.natoms }, (_, j) => [dot(f.force[3 * j]!), dot(f.force[3 * j + 1]!), dot(f.force[3 * j + 2]!)] as Vec3);
  return { energy: sites.reduce((a, b) => a + b, 0), forces, sites };
}

export interface Errors {
  /** RMSE of energy per atom (eV/atom). */
  energy: number;
  /** RMSE per force component (eV/Å). */
  force: number;
}

export function errors(m: AceModel, set: Array<{ f: CellFeatures; s: Sample }>): Errors {
  let e2 = 0;
  let f2 = 0;
  let nf = 0;
  for (const { f, s } of set) {
    const p = predictAce(m, f);
    e2 += ((p.energy - s.energy) / f.natoms) ** 2;
    p.forces.forEach((v, j) => v.forEach((x, a) => ((f2 += (x - s.forces[j]![a]!) ** 2), nf++)));
  }
  return { energy: Math.sqrt(e2 / set.length), force: Math.sqrt(f2 / Math.max(1, nf)) };
}

/** Convenience: build the basis, features and fit for one setting. */
export function fitSetting(spec: AceSpec, R: Radial, train: Sample[], o: FitOptions) {
  const basis = buildBasis(spec);
  const feats = train.map((s) => ({ f: cellFeatures(basis, R, s.cell), s }));
  return { basis, model: fitAce(feats, o), feats };
}

// ---------------------------------------------------------------------------
// The lab: one feature computation, then every (ν, l_max) sub-basis as a column subset
// ---------------------------------------------------------------------------

/** The lab's basis and radial functions (one species, Si). */
export const LAB_SPEC: AceSpec = { nmax: 7, lmax: 3, numax: 4, degree: [7, 10, 10, 10], wL: 1 };
export const LAB_RADIAL: Radial = { rc: 3.8, lambda: 2 };

/** pacemaker's core repulsion per pair, pre·exp(−λr²)/r times the cosine envelope (ace_radial.cpp, radcore). */
export function coreRepulsion(r: number, pre: number, lambda: number, rc: number): number {
  if (r >= rc) return 0;
  return ((pre * Math.exp(-lambda * r * r)) / r) * 0.5 * (1 + Math.cos((Math.PI * r) / rc));
}
/** The pacemaker documentation's example setting, core-repulsion: [100.0, 5.0]. */
export const CORE = { pre: 100, lambda: 5 };

/** Reference energy of an isolated dimer (eV): both potentials reduce to their pair term. */
export function referenceDimer(r: number, ref: Reference): number {
  if (ref === 'sw') return phi2(r);
  return 2 * tersoffSite([[r, 0, 0]]);
}

export interface LabOptions {
  ref: Reference;
  count: number;
  ridge: number;
  seed?: number;
}

export interface LabFit {
  nu: number;
  lmax: number;
  functions: number;
  train: Errors;
  test: Errors;
  /** ACE dimer energies on DIMER_R (core repulsion not included). */
  dimer: number[];
  /** ACE energy per atom on SCAN_EPS (core not included). */
  scan: number[];
  /** Forces of the shown test cells [cell][atom]. */
  forces: Vec3[][];
}

export interface LabResult {
  ref: Reference;
  fits: LabFit[];
  dimerRef: number[];
  scanRef: number[];
  /** Core repulsion added to the dimer and to the scan (per atom), for the toggle. */
  dimerCore: number[];
  scanCore: number[];
  shown: Array<{ cell: Cell; forces: Vec3[] }>;
  /** Shortest interatomic distance in the training cells (Å). */
  rminTrain: number;
  strainRange: number;
}

export const DIMER_R = Array.from({ length: 57 }, (_, k) => 1.0 + 0.05 * k); // 1.0 … 3.8 Å
export const SCAN_EPS = Array.from({ length: 31 }, (_, k) => -0.2 + 0.01 * k); // −20 % … +10 %
export const LAB_MAX_STRAIN = 0.05;
export const LAB_MAX_RATTLE = 0.15;
export const SHOWN_TEST_CELLS = 4;

const diamondScan = () => SCAN_EPS.map((e) => strained(diamondCell(), e));

export function runLab(o: LabOptions, progress?: (msg: string) => void): LabResult {
  const seed = o.seed ?? 3;
  const basis = buildBasis(LAB_SPEC);
  const R = LAB_RADIAL;
  progress?.('reference energies and forces');
  const trainCells = cells({ count: o.count, maxStrain: LAB_MAX_STRAIN, maxRattle: LAB_MAX_RATTLE, seed });
  const testCells = cells({ count: 16, maxStrain: LAB_MAX_STRAIN, maxRattle: LAB_MAX_RATTLE, seed: 99 });
  const train = trainCells.map((c) => sample(c, o.ref));
  const test = testCells.map((c) => sample(c, o.ref));
  progress?.('ACE features with their derivatives');
  const tr = train.map((s) => ({ f: cellFeatures(basis, R, s.cell), s }));
  const te = test.map((s) => ({ f: cellFeatures(basis, R, s.cell), s }));
  const scanCells = diamondScan();
  const scanF = scanCells.map((c) => cellFeatures(basis, R, c));
  const dimerB = DIMER_R.map((r) => evaluateBasisWithDerivative(basis, atomicBase([[r, 0, 0]], LAB_SPEC.nmax, LAB_SPEC.lmax, R).A).B);
  progress?.('fitting every sub-basis');
  const ne = normalEquations(tr, { forceWeight: 1 });
  const fits: LabFit[] = [];
  for (let nu = 1; nu <= LAB_SPEC.numax; nu++)
    for (let L = 0; L <= LAB_SPEC.lmax; L++) {
      const keep = basis.functions.map((f, v) => (f.nls.length <= nu && Math.max(...f.nls.map((x) => x.l)) <= L ? v : -1)).filter((v) => v >= 0);
      const m = solveNormal(ne, keep, o.ridge);
      const dot = (B: Float64Array) => B.reduce((s, x, k) => s + x * m.c[k]!, 0);
      fits.push({
        nu,
        lmax: L,
        functions: keep.length,
        train: errors(m, tr),
        test: errors(m, te),
        dimer: dimerB.map((B) => 2 * dot(B)),
        scan: scanF.map((f) => predictAce(m, f).energy / f.natoms),
        forces: te.slice(0, SHOWN_TEST_CELLS).map(({ f }) => predictAce(m, f).forces),
      });
    }
  let rmin = Infinity;
  for (const c of trainCells) for (const nb of neighbourList(c, 3)) for (const { v } of nb) rmin = Math.min(rmin, Math.hypot(...v));
  const coreSite = (c: Cell) => neighbourList(c, R.rc).reduce((s, nb) => s + nb.reduce((t, { v }) => t + coreRepulsion(Math.hypot(...v), CORE.pre, CORE.lambda, R.rc), 0), 0);
  return {
    ref: o.ref,
    fits,
    dimerRef: DIMER_R.map((r) => referenceDimer(r, o.ref)),
    scanRef: scanCells.map((c) => referenceEnergy(c, o.ref) / c.positions.length),
    dimerCore: DIMER_R.map((r) => 2 * coreRepulsion(r, CORE.pre, CORE.lambda, R.rc)),
    scanCore: scanCells.map((c) => coreSite(c) / c.positions.length),
    shown: test.slice(0, SHOWN_TEST_CELLS).map((s) => ({ cell: s.cell, forces: s.forces })),
    rminTrain: rmin,
    strainRange: LAB_MAX_STRAIN,
  };
}
