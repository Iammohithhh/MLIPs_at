// D-optimal active learning for linearly parametrised potentials (Podryabinkin & Shapeev 2017; MLIP
// package, Novikov et al. 2021). Rows of B are the basis values of candidate configurations; the active
// set is the m rows whose square submatrix A has the largest |det A| (found by maxvol), and the
// extrapolation grade of a configuration is γ = max_j |(b A⁻¹)_j|. γ ≤ 1 means its energy is an
// interpolation of the active set's energies with coefficients of modulus ≤ 1.

export type Matrix = number[][];

/** LU-based inverse with partial pivoting; null if (numerically) singular. */
export function invert(A: Matrix, tol = 1e-12): Matrix | null {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  const scale = Math.max(...A.flat().map(Math.abs)) || 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    if (Math.abs(M[p]![c]!) < tol * scale) return null;
    [M[c], M[p]] = [M[p]!, M[c]!];
    const piv = M[c]![c]!;
    for (let j = 0; j < 2 * n; j++) M[c]![j]! /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r]![c]!;
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r]![j]! -= f * M[c]![j]!;
    }
  }
  return M.map((r) => r.slice(n));
}

export function logAbsDet(A: Matrix): number {
  const n = A.length;
  const M = A.map((r) => [...r]);
  let s = 0;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    if (M[p]![c] === 0) return -Infinity;
    [M[c], M[p]] = [M[p]!, M[c]!];
    s += Math.log(Math.abs(M[c]![c]!));
    for (let r = c + 1; r < n; r++) {
      const f = M[r]![c]! / M[c]![c]!;
      for (let j = c; j < n; j++) M[r]![j]! -= f * M[c]![j]!;
    }
  }
  return s;
}

/** c = b A⁻¹ for a row vector b. */
export const coefficients = (b: number[], Ainv: Matrix) => Ainv[0]!.map((_, j) => b.reduce((s, x, k) => s + x * Ainv[k]![j]!, 0));

export const grade = (b: number[], Ainv: Matrix) => Math.max(...coefficients(b, Ainv).map(Math.abs));

export interface ActiveSet {
  /** Indices into the candidate rows, one per basis function. */
  rows: number[];
  Ainv: Matrix;
  logDet: number;
  swaps: number;
}

/**
 * Maxvol: pick m of the K candidate rows (K ≥ m) maximising |det A|. Greedy start (largest residual norm
 * after projecting out the rows already chosen), then swap a row for any candidate with |C_ij| > threshold
 * (C = B A⁻¹) until none remains. Returns null when the candidates do not span all m columns.
 */
export function maxvol(B: Matrix, threshold = 1 + 1e-9, maxSwaps = 10_000): ActiveSet | null {
  const K = B.length;
  const m = B[0]?.length ?? 0;
  if (K < m || m === 0) return null;
  const R = B.map((r) => [...r]);
  const norms0 = B.map((r) => Math.hypot(...r));
  const rows: number[] = [];
  for (let s = 0; s < m; s++) {
    let k = -1;
    let best = 0;
    R.forEach((r, i) => {
      if (rows.includes(i)) return;
      const n = Math.hypot(...r) / (norms0[i] || 1);
      if (n > best) [best, k] = [n, i];
    });
    if (k < 0 || best < 1e-10) return null;
    rows.push(k);
    const v = R[k]!.map((x) => x / Math.hypot(...R[k]!));
    for (const r of R) {
      const d = r.reduce((s2, x, j) => s2 + x * v[j]!, 0);
      for (let j = 0; j < m; j++) r[j]! -= d * v[j]!;
    }
  }
  let Ainv = invert(rows.map((i) => B[i]!));
  if (!Ainv) return null;
  let swaps = 0;
  for (; swaps < maxSwaps; swaps++) {
    let bi = -1;
    let bj = -1;
    let bv = threshold;
    for (let i = 0; i < K; i++) {
      const c = coefficients(B[i]!, Ainv);
      c.forEach((x, j) => {
        if (Math.abs(x) > bv) [bv, bi, bj] = [Math.abs(x), i, j];
      });
    }
    if (bi < 0) break;
    rows[bj] = bi;
    const next = invert(rows.map((i) => B[i]!));
    if (!next) return null;
    Ainv = next;
  }
  return { rows, Ainv, logDet: logAbsDet(rows.map((i) => B[i]!)), swaps };
}

/** Least squares min |Xθ − y|² + λ Σ_j (X^T X)_jj θ_j² (Shapeev 2016's ℓ2 form) via the normal equations. */
export function ridgeFit(X: Matrix, y: number[], lambda: number): number[] | null {
  const m = X[0]!.length;
  const G: Matrix = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  const r = new Array<number>(m).fill(0);
  X.forEach((row, k) => {
    for (let i = 0; i < m; i++) {
      r[i]! += row[i]! * y[k]!;
      for (let j = 0; j < m; j++) G[i]![j]! += row[i]! * row[j]!;
    }
  });
  for (let i = 0; i < m; i++) G[i]![i]! *= 1 + lambda;
  const Gi = invert(G, 1e-15);
  if (!Gi) return null;
  return Gi.map((row) => row.reduce((s, x, j) => s + x * r[j]!, 0));
}

// ---------------------------------------------------------------- the one-dimensional example

/** Podryabinkin & Shapeev 2017, Sec. 4.1: E_qm(x) = x² + x³ exp(−x²/2) on [−4, 4]; model θ₁x² + θ₂x³. */
export const eqm1d = (x: number) => x * x + x ** 3 * Math.exp((-x * x) / 2);
export const basis1d = (x: number) => [x * x, x ** 3];
export const model1d = (t: number[], x: number) => t[0]! * x * x + t[1]! * x ** 3;

const GRID = Array.from({ length: 8001 }, (_, i) => -4 + (8 * i) / 8000);

/** Boltzmann weights e^{−E_qm/kT} on a uniform grid over [−4, 4], normalised. */
function weights(kT: number): number[] {
  const w = GRID.map((x) => Math.exp(-eqm1d(x) / kT));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / s);
}

/** Minimise ∫(E − E_qm)² e^{−E_qm/kT} dx: the fit to infinitely long MD at temperature kT. */
export function boltzmannFit(kT = 1): number[] {
  const w = weights(kT);
  const G = [
    [0, 0],
    [0, 0],
  ];
  const r = [0, 0];
  GRID.forEach((x, i) => {
    const b = basis1d(x);
    for (let p = 0; p < 2; p++) {
      r[p]! += w[i]! * b[p]! * eqm1d(x);
      for (let q = 0; q < 2; q++) G[p]![q]! += w[i]! * b[p]! * b[q]!;
    }
  });
  const Gi = invert(G)!;
  return [Gi[0]![0]! * r[0]! + Gi[0]![1]! * r[1]!, Gi[1]![0]! * r[0]! + Gi[1]![1]! * r[1]!];
}

/** Fit through two configurations exactly (the active set is the training set). */
export function interpolate1d(x1: number, x2: number): number[] | null {
  const Ai = invert([basis1d(x1), basis1d(x2)]);
  if (!Ai) return null;
  const e = [eqm1d(x1), eqm1d(x2)];
  return [Ai[0]![0]! * e[0]! + Ai[0]![1]! * e[1]!, Ai[1]![0]! * e[0]! + Ai[1]![1]! * e[1]!];
}

/** RMS of E − E_qm under the Boltzmann distribution at kT. */
export function rms1d(t: number[], kT = 1): number {
  const w = weights(kT);
  return Math.sqrt(GRID.reduce((s, x, i) => s + w[i]! * (model1d(t, x) - eqm1d(x)) ** 2, 0));
}

export function grade1d(x: number, x1: number, x2: number): number {
  const Ai = invert([basis1d(x1), basis1d(x2)]);
  return Ai ? grade(basis1d(x), Ai) : Infinity;
}

/** Highest point of the model on [−4, a]: the spurious barrier the passive fit creates. */
export function barrier1d(t: number[], a = -1): { x: number; e: number } {
  let best = { x: -4, e: -Infinity };
  for (const x of GRID) if (x <= a && model1d(t, x) > best.e) best = { x, e: model1d(t, x) };
  return best;
}
