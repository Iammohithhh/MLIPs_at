// Ingredients of a MACE layer, computed for real on a structure (no trained weights):
//
//   radial basis   R_n(r) = √(2/r_c) sin(nπr/r_c)/r · u(r/r_c)    (Bessel × polynomial envelope, p = 5)
//   atomic basis   A_{i,nlm} = Σ_{j∈N(i)} R_n(r_ij) Y_lm(r̂_ij)     (Batatia et al. 2022, Eq. 9, with unit element weights)
//   2-body → 3-body invariants (ν = 2, L = 0):  P_{nl} = Σ_m A_{i,nlm}²     (the "power spectrum")
//
// The trained model multiplies these by learned weights and element embeddings; the
// symmetry properties shown here hold for any weights.

import { polynomialEnvelope } from '../cutoffs';
import { shBlock, type Vec3 } from '../physics/spherical-harmonics';

/** Normalised Bessel radial function (orthonormal on [0, r_c] with weight r²). */
export function bessel(n: number, r: number, rc: number): number {
  if (r <= 0 || r >= rc) return 0;
  return (Math.sqrt(2 / rc) * Math.sin((n * Math.PI * r) / rc)) / r;
}

/** MACE radial function: Bessel × polynomial envelope (p = 5 in the MACE paper). */
export function radial(n: number, r: number, rc: number, p = 5): number {
  return bessel(n, r, rc) * polynomialEnvelope(r, rc, p).f;
}

/** A_{nlm} for l = 0..lmax: blocks[l][m + l], from neighbour displacement vectors. */
export function atomicBasis(displacements: Vec3[], rc: number, n = 1, lmax = 3): number[][] {
  const blocks = Array.from({ length: lmax + 1 }, (_, l) => new Array<number>(2 * l + 1).fill(0));
  for (const d of displacements) {
    const r = Math.hypot(d[0], d[1], d[2]);
    if (r >= rc || r === 0) continue;
    const R = radial(n, r, rc);
    for (let l = 0; l <= lmax; l++) {
      const y = shBlock(l, d);
      for (let k = 0; k < y.length; k++) blocks[l]![k]! += R * y[k]!;
    }
  }
  return blocks;
}

/** Rotation-invariant power spectrum P_l = Σ_m A_lm². */
export function powerSpectrum(blocks: number[][]): number[] {
  return blocks.map((b) => b.reduce((s, v) => s + v * v, 0));
}

/** Number of distinct ν-fold products of D atomic-basis components (multiset count). */
export function productCount(D: number, nu: number): number {
  let c = 1;
  for (let k = 0; k < nu; k++) c = (c * (D + k)) / (k + 1);
  return Math.round(c);
}
