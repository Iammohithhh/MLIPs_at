// Charge equilibration (QEq-style), Physics Lab P8. TOY parameters.
//
// Energy of a set of partial charges q (total charge Q fixed):
//   E(q) = Σ_i (χ_i q_i + ½ J_i q_i²) + Σ_{i<j} γ_ij q_i q_j
// χ: electronegativity (eV/e), J: hardness (eV/e²), γ_ij: shielded Coulomb kernel
//   γ_ij = k / ∛(r_ij³ + a³)        (k = 14.3996 eV·Å, a = shielding length)
// Minimising with a Lagrange multiplier μ for Σq = Q gives the linear system
//   χ_i + J_i q_i + Σ_{j≠i} γ_ij q_j = μ  for every i,   Σ q_i = Q
// i.e. charge flows until every atom has the same effective electronegativity μ.

import { COULOMB_K } from '../chem/elements';
import type { Vec3 } from '../structure/types';

export interface QEqParams {
  chi: number;
  J: number;
}

/** Toy parameters (eV): ordered like real electronegativities, not fitted values. */
export const QEQ_TOY: Record<string, QEqParams> = {
  H: { chi: 4.5, J: 13.9 },
  C: { chi: 5.3, J: 10.1 },
  N: { chi: 6.9, J: 11.8 },
  O: { chi: 8.7, J: 13.4 },
  Na: { chi: 2.8, J: 7.0 },
  Cl: { chi: 8.6, J: 11.0 },
};

export const SHIELD = 1.6; // Å

export function gamma(r: number, a = SHIELD): number {
  return COULOMB_K / Math.cbrt(r * r * r + a * a * a);
}

export interface QEqResult {
  q: number[];
  /** Equalised chemical potential μ (eV/e). */
  mu: number;
  /** Effective electronegativity of each atom at the solution (all equal μ). */
  chiEff: number[];
  energy: number;
}

/** Solve the (n+1)×(n+1) linear system by Gaussian elimination with partial pivoting. */
function solve(A: number[][], b: number[]): number[] {
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

export function qeq(positions: Vec3[], params: QEqParams[], Q = 0, a = SHIELD): QEqResult {
  const n = positions.length;
  const G = (i: number, j: number) => {
    const p = positions[i]!;
    const r = positions[j]!;
    return gamma(Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2]), a);
  };
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n + 1).fill(0);
    for (let j = 0; j < n; j++) row[j] = i === j ? params[i]!.J : G(i, j);
    row[n] = -1; // −μ
    A.push(row);
    b.push(-params[i]!.chi);
  }
  A.push([...new Array<number>(n).fill(1), 0]);
  b.push(Q);
  const x = solve(A, b);
  const q = x.slice(0, n);
  const mu = x[n]!;
  const chiEff = q.map((qi, i) => params[i]!.chi + params[i]!.J * qi + q.reduce((s, qj, j) => (j === i ? s : s + G(i, j) * qj), 0));
  let energy = 0;
  for (let i = 0; i < n; i++) {
    energy += params[i]!.chi * q[i]! + 0.5 * params[i]!.J * q[i]! ** 2;
    for (let j = i + 1; j < n; j++) energy += G(i, j) * q[i]! * q[j]!;
  }
  return { q, mu, chiEff, energy };
}

/** Charge on the anion of a neutral A–B dimer as r → ∞ (γ → 0): (χ_B − χ_A)/(J_A + J_B), not zero. */
export function dimerLimit(a: QEqParams, b: QEqParams): number {
  return (b.chi - a.chi) / (a.J + b.J);
}
