// Stillinger–Weber potential for silicon (Physics Lab P3): a real two-body +
// three-body potential with an infinitely smooth cutoff.
//
//   E = Σ_{i<j} φ₂(r_ij) + Σ_i Σ_{j≠i} Σ_{k>j} φ₃(r_ij, r_ik, θ_jik)
//   φ₂(r)  = A ε [B (σ/r)^p − (σ/r)^q] exp(σ / (r − aσ))
//   φ₃     = λ ε (cos θ − cos θ₀)² exp(γσ/(r_ij − aσ)) exp(γσ/(r_ik − aσ))
// for r < aσ, zero beyond. Every derivative of exp(σ/(r − aσ)) vanishes as r → aσ⁻,
// so the cutoff is C^∞, smoother than the cosine (C¹) or polynomial envelope (C²).
//
// Parameters: Stillinger & Weber, PRB 31, 5262 (1985), as distributed in LAMMPS
// potentials/Si.sw (metal units: ε in eV, σ in Å).

import type { Vec3 } from '../structure/types';

export const SW_SI = {
  epsilon: 2.1683,
  sigma: 2.0951,
  a: 1.8,
  lambda: 21.0,
  gamma: 1.2,
  cosTheta0: -1 / 3,
  A: 7.049556277,
  B: 0.6022245584,
  p: 4,
  q: 0,
} as const;

export const SW_CUTOFF = SW_SI.a * SW_SI.sigma; // 3.77118 Å

export function phi2(r: number, P = SW_SI): number {
  const rc = P.a * P.sigma;
  if (r >= rc) return 0;
  const s = P.sigma / r;
  return P.A * P.epsilon * (P.B * s ** P.p - s ** P.q) * Math.exp(P.sigma / (r - rc));
}

/** The radial screening factor exp(γσ/(r − aσ)) of the three-body term. */
export function screen3(r: number, P = SW_SI): number {
  const rc = P.a * P.sigma;
  return r >= rc ? 0 : Math.exp((P.gamma * P.sigma) / (r - rc));
}

export function phi3(rij: number, rik: number, cosTheta: number, P = SW_SI): number {
  return P.lambda * P.epsilon * (cosTheta - P.cosTheta0) ** 2 * screen3(rij, P) * screen3(rik, P);
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const cosAngle = (u: Vec3, v: Vec3) => (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (len(u) * len(v));

export interface SWEnergy {
  two: number;
  three: number;
  total: number;
  nPairs: number;
  nTriplets: number;
}

/** Energy of a finite cluster (open boundaries). */
export function swCluster(positions: Vec3[], P = SW_SI, threeBody = true): SWEnergy {
  let two = 0;
  let three = 0;
  let nPairs = 0;
  let nTriplets = 0;
  const n = positions.length;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const r = len(sub(positions[j]!, positions[i]!));
      if (r < SW_CUTOFF) {
        two += phi2(r, P);
        nPairs++;
      }
    }
  if (threeBody)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        for (let k = j + 1; k < n; k++) {
          if (k === i) continue;
          const u = sub(positions[j]!, positions[i]!);
          const v = sub(positions[k]!, positions[i]!);
          if (len(u) >= SW_CUTOFF || len(v) >= SW_CUTOFF) continue;
          three += phi3(len(u), len(v), cosAngle(u, v), P);
          nTriplets++;
        }
      }
  return { two, three, total: two + three, nPairs, nTriplets };
}

/** Energy per atom from per-atom neighbour vectors (for periodic crystals). */
export function swPerAtom(neighbourVectors: Vec3[][], P = SW_SI): { two: number; three: number } {
  let two = 0;
  let three = 0;
  for (const nb of neighbourVectors) {
    const inside = nb.filter((v) => len(v) < SW_CUTOFF);
    for (const v of inside) two += 0.5 * phi2(len(v), P); // each pair shared by two atoms
    for (let j = 0; j < inside.length; j++)
      for (let k = j + 1; k < inside.length; k++)
        three += phi3(len(inside[j]!), len(inside[k]!), cosAngle(inside[j]!, inside[k]!), P);
  }
  return { two: two / neighbourVectors.length, three: three / neighbourVectors.length };
}

/** Three-atom geometry: i at the origin, j along x at r1, k at r2 and angle θ in the xy plane. */
export function triplet(r1: number, r2: number, thetaDeg: number): Vec3[] {
  const t = (thetaDeg * Math.PI) / 180;
  return [
    [0, 0, 0],
    [r1, 0, 0],
    [r2 * Math.cos(t), r2 * Math.sin(t), 0],
  ];
}
