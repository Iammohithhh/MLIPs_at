// Tersoff's bond-order potential for silicon, the Si(B) parameterisation of Tersoff, PRB 37, 6991
// (1988), as distributed in LAMMPS potentials/Si.tersoff and in the LAMMPS functional form:
//
//   E = ½ Σ_i Σ_{j≠i} f_C(r_ij) [ f_R(r_ij) + b_ij f_A(r_ij) ]
//   f_R = A exp(−λ₁ r),  f_A = −B exp(−λ₂ r)
//   b_ij = (1 + βⁿ ζ_ijⁿ)^(−1/2n),  ζ_ij = Σ_{k≠i,j} f_C(r_ik) g(θ_ijk) exp[λ₃ᵐ (r_ij − r_ik)ᵐ]
//   g(θ) = γ (1 + c²/d² − c² / [d² + (cos θ − cos θ₀)²])
//   f_C = 1 (r < R − D), ½ − ½ sin(π(r − R)/(2D)) (|r − R| ≤ D), 0 (r > R + D)
//
// The bond order b_ij depends on every neighbour of i at once through ζ, so the site energy is not a
// finite sum of two- and three-body terms: expanded in ζ it has terms of every body order.

import type { Vec3 } from '../structure/types';

export const TERSOFF_SI_B = {
  m: 3.0,
  gamma: 1.0,
  lambda3: 1.3258,
  c: 4.8381,
  d: 2.0417,
  cosTheta0: 0.0,
  n: 22.956,
  beta: 0.33675,
  lambda2: 1.3258,
  B: 95.373,
  R: 3.0,
  D: 0.2,
  lambda1: 3.2394,
  A: 3264.7,
} as const;

export type TersoffParams = typeof TERSOFF_SI_B;

export const TERSOFF_CUTOFF = TERSOFF_SI_B.R + TERSOFF_SI_B.D;

export function fC(r: number, P: TersoffParams = TERSOFF_SI_B): number {
  if (r < P.R - P.D) return 1;
  if (r > P.R + P.D) return 0;
  return 0.5 - 0.5 * Math.sin((Math.PI * (r - P.R)) / (2 * P.D));
}

const gAngle = (cos: number, P: TersoffParams) => P.gamma * (1 + (P.c * P.c) / (P.d * P.d) - (P.c * P.c) / (P.d * P.d + (cos - P.cosTheta0) ** 2));

/** Site energy of atom i from its neighbour vectors (½ of each bond is assigned to each end). */
export function tersoffSite(env: Vec3[], P: TersoffParams = TERSOFF_SI_B): number {
  const rs = env.map((v) => Math.hypot(v[0], v[1], v[2]));
  let e = 0;
  env.forEach((u, j) => {
    const rij = rs[j]!;
    if (rij > P.R + P.D) return;
    let zeta = 0;
    env.forEach((w, k) => {
      if (k === j) return;
      const rik = rs[k]!;
      if (rik > P.R + P.D) return;
      const cos = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (rij * rik);
      zeta += fC(rik, P) * gAngle(cos, P) * Math.exp((P.lambda3 * (rij - rik)) ** P.m);
    });
    const b = (1 + (P.beta * zeta) ** P.n) ** (-1 / (2 * P.n));
    e += 0.5 * fC(rij, P) * (P.A * Math.exp(-P.lambda1 * rij) - b * P.B * Math.exp(-P.lambda2 * rij));
  });
  return e;
}
