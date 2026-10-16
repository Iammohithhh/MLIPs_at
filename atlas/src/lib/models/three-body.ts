// Three-body invariant message passing (M3GNet, CHGNet): the bond angles a model sums
// over, counted exactly in real crystals, and M3GNet's radial × angular basis.
//
// M3GNet updates each bond e_ij with a sum over third atoms k within a three-body cutoff
// (4 Å in the paper; graph cutoff 5 Å), weighting each term by j_l(z_ln r_ik / r_c)
// Y_l^0(θ_jik) (Chen & Ong 2022, Eq. 2; Sec. 7.1.2). CHGNet builds a separate bond graph
// whose nodes are bonds shorter than 3 Å and whose edges are the angles between them
// (Deng et al. 2023, Sec. II.1, Methods V.2).

import type { Structure, Vec3 } from '../structure/types';
import { neighboursOf } from '../structure/geometry';

export interface GraphCounts {
  cutoff: number;
  /** Mean number of neighbours (bonds) per atom with r < cutoff. */
  bondsPerAtom: number;
  /** Mean number of bond angles centred on an atom: n(n − 1)/2 unordered pairs of its bonds. */
  anglesPerAtom: number;
}

export function graphCounts(s: Structure, cutoff: number): GraphCounts {
  let bonds = 0;
  let angles = 0;
  for (let i = 0; i < s.positions.length; i++) {
    const n = neighboursOf(s, i, cutoff).length;
    bonds += n;
    angles += (n * (n - 1)) / 2;
  }
  const N = s.positions.length;
  return { cutoff, bondsPerAtom: bonds / N, anglesPerAtom: angles / N };
}

export function countsVsCutoff(s: Structure, cutoffs: number[]): GraphCounts[] {
  return cutoffs.map((r) => graphCounts(s, r));
}

/** Centre atom at the origin, then neighbours within R (Cartesian, relative to the centre). */
export function shellAround(s: Structure, R: number, centre = 0): Array<{ p: Vec3; r: number; symbol: string }> {
  const c = s.positions[centre]!;
  return neighboursOf(s, centre, R).map((n) => ({
    p: [n.position[0] - c[0], n.position[1] - c[1], n.position[2] - c[2]] as Vec3,
    r: n.distance,
    symbol: s.symbols[n.j]!,
  }));
}

// ---------------------------------------------------------------------------
// M3GNet's three-body basis
// ---------------------------------------------------------------------------

/** Spherical Bessel function of the first kind j_l(x), by upward recurrence (l ≤ ~6 is accurate for x ≳ l). */
export function sphericalBessel(l: number, x: number): number {
  if (Math.abs(x) < 1e-6) return l === 0 ? 1 - (x * x) / 6 : 0;
  const j0 = Math.sin(x) / x;
  if (l === 0) return j0;
  let jm = j0;
  let j = Math.sin(x) / (x * x) - Math.cos(x) / x;
  for (let k = 1; k < l; k++) {
    const next = ((2 * k + 1) / x) * j - jm;
    jm = j;
    j = next;
  }
  return j;
}

/** The n-th positive root z_ln of j_l (n = 1, 2, …), by bracketing and bisection. */
export function besselRoot(l: number, n: number): number {
  let found = 0;
  const h = 0.05;
  let a = l === 0 ? h : l + 0.5;
  let fa = sphericalBessel(l, a);
  for (let b = a + h; b < 200; b += h) {
    const fb = sphericalBessel(l, b);
    if (fa * fb < 0) {
      found++;
      if (found === n) {
        let lo = b - h;
        let hi = b;
        for (let it = 0; it < 80; it++) {
          const mid = 0.5 * (lo + hi);
          if (sphericalBessel(l, lo) * sphericalBessel(l, mid) <= 0) hi = mid;
          else lo = mid;
        }
        return 0.5 * (lo + hi);
      }
    }
    a = b;
    fa = fb;
  }
  throw new Error(`root ${n} of j_${l} not found`);
}

/** Legendre polynomial P_l(x) by recurrence. */
export function legendre(l: number, x: number): number {
  if (l === 0) return 1;
  let pm = 1;
  let p = x;
  for (let k = 1; k < l; k++) {
    const next = ((2 * k + 1) * x * p - k * pm) / (k + 1);
    pm = p;
    p = next;
  }
  return p;
}

/** Real spherical harmonic with m = 0: Y_l^0(θ) = √((2l + 1)/4π) P_l(cos θ). */
export function ylm0(l: number, theta: number): number {
  return Math.sqrt((2 * l + 1) / (4 * Math.PI)) * legendre(l, Math.cos(theta));
}

/** M3GNet's polynomial cutoff f_c(r) = 1 − 6(r/r_c)⁵ + 15(r/r_c)⁴ − 10(r/r_c)³ (0 beyond r_c). */
export function m3gnetCutoff(r: number, rc: number): number {
  if (r >= rc) return 0;
  const x = r / rc;
  return 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3;
}

/** One three-body basis function j_l(z_ln r/r_c) Y_l^0(θ) f_c(r), with n counted from 1. */
export function threeBodyBasis(l: number, n: number, r: number, theta: number, rc: number, root = besselRoot(l, n)): number {
  return sphericalBessel(l, (root * r) / rc) * ylm0(l, theta) * m3gnetCutoff(r, rc);
}

/** Sample the basis on a grid: values[a][b] at r = rs[a], θ = thetas[b] (radians). */
export function basisGrid(l: number, n: number, rc: number, rs: number[], thetas: number[]): number[][] {
  const z = besselRoot(l, n);
  return rs.map((r) => thetas.map((t) => threeBodyBasis(l, n, r, t, rc, z)));
}
