// A linear MTP fitted to the energies of the Zuo et al. Si cells, and the extrapolation grade of every cell
// against a D-optimal active set. Input: configuration-level basis values precomputed on CPU by
// pipeline/atlas_pipeline/mtp.py (src/data/models/mtp-si.json).

import { grade, maxvol, ridgeFit, type ActiveSet, type Matrix } from './active-learning';

export interface SiConfig {
  split: 'train' | 'test';
  group: string;
  description: string;
  n: number;
  /** Total DFT (PBE) energy of the cell, eV. */
  energy: number;
  b: number[];
}

export interface SiData {
  basis: Array<{ level: number; key: string; mu: number[]; nu: number[]; adj: number[] }>;
  configs: SiConfig[];
  settings: { rmin: number; rcut: number; level_max: number; min_neighbour_distance: number };
  source: { commit: string; repository: string };
}

export interface FitOptions {
  levelMax: number;
  /** Group left out of training (its cells are still evaluated). */
  leaveOut: string | null;
  /** Fit on every training cell, or only on the m cells of the active set (as in Podryabinkin & Shapeev's Table 4). */
  fitOn: 'all' | 'active';
  /** ℓ2 regularisation γ of Shapeev 2016, Sec. 4.2.1, relative to diag(XᵀX). */
  lambda: number;
}

export interface CellResult {
  index: number;
  config: SiConfig;
  role: 'train' | 'test' | 'left-out';
  active: boolean;
  /** |E_MTP − E_DFT| per atom, meV. */
  error: number;
  gamma: number;
}

export interface FitResult {
  columns: number[];
  /** Number of fitted parameters: the basis functions plus one constant per atom. */
  m: number;
  active: ActiveSet | null;
  cells: CellResult[];
  rmse: { train: number; test: number; leftOut: number | null };
}

const rms = (v: number[]) => (v.length ? Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length) : NaN);

/** Columns: the atom count (mlip-2's per-species constant, summed over atoms) and every B_α of level ≤ levelMax. */
export function designRow(c: SiConfig, columns: number[]): number[] {
  return [c.n, ...columns.map((j) => c.b[j]!)];
}

export function fitSi(data: SiData, o: FitOptions): FitResult {
  const columns = data.basis.flatMap((b, j) => (b.level <= o.levelMax ? [j] : []));
  const role = (c: SiConfig): CellResult['role'] => (c.group === o.leaveOut ? 'left-out' : c.split);
  const rows = data.configs.map((c) => designRow(c, columns));
  const train = data.configs.flatMap((c, i) => (role(c) === 'train' ? [i] : []));
  // Columns are scaled to unit RMS over the training cells: γ is unchanged (A → AD, b → bD) and the
  // matrices become well conditioned.
  const m = columns.length + 1;
  const scale = Array.from({ length: m }, (_, j) => Math.sqrt(train.reduce((s, i) => s + rows[i]![j]! ** 2, 0) / train.length) || 1);
  const S = rows.map((r) => r.map((x, j) => x / scale[j]!));
  // Active set from the extensive rows b_j(cfg) = Σ_i B_j(n_i), as in Podryabinkin & Shapeev Eq. (8).
  const active = maxvol(train.map((i) => S[i]!));
  const activeRows = new Set(active ? active.rows.map((k) => train[k]!) : []);
  // Energies are fitted per atom (each cell weighs the same).
  const fitIdx = o.fitOn === 'active' && active ? [...activeRows] : train;
  const X: Matrix = fitIdx.map((i) => S[i]!.map((x) => x / data.configs[i]!.n));
  const y = fitIdx.map((i) => data.configs[i]!.energy / data.configs[i]!.n);
  const theta = ridgeFit(X, y, o.lambda);
  const cells = data.configs.map((c, i) => {
    const pred = theta ? S[i]!.reduce((s, x, j) => s + x * theta[j]!, 0) : NaN;
    return {
      index: i,
      config: c,
      role: role(c),
      active: activeRows.has(i),
      error: (Math.abs(pred - c.energy) / c.n) * 1000,
      gamma: active ? grade(S[i]!, active.Ainv) : NaN,
    };
  });
  const err = (r: CellResult['role']) => rms(cells.filter((c) => c.role === r).map((c) => c.error));
  return { columns, m, active, cells, rmse: { train: err('train'), test: err('test'), leftOut: o.leaveOut ? err('left-out') : null } };
}

/** Spearman rank correlation, ties given their mean rank. */
export function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]) => {
    const idx = v.map((x, i) => [x, i] as const).sort((p, q) => p[0] - q[0]);
    const r = new Array<number>(v.length);
    for (let s = 0; s < idx.length; ) {
      let e = s;
      while (e + 1 < idx.length && idx[e + 1]![0] === idx[s]![0]) e++;
      for (let k = s; k <= e; k++) r[idx[k]![1]] = (s + e) / 2;
      s = e + 1;
    }
    return r;
  };
  const ra = rank(a);
  const rb = rank(b);
  const ma = ra.reduce((s, x) => s + x, 0) / ra.length;
  const mb = rb.reduce((s, x) => s + x, 0) / rb.length;
  let num = 0;
  let da = 0;
  let db = 0;
  ra.forEach((x, i) => {
    num += (x - ma) * (rb[i]! - mb);
    da += (x - ma) ** 2;
    db += (rb[i]! - mb) ** 2;
  });
  return num / Math.sqrt(da * db);
}
