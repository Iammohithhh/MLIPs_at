// Coulomb pair energy and a cutoff-limited toy pair term (Physics Lab P1).
//
// E_C(r)   = k q₁q₂ / r                      (k = e²/4πε₀ = 14.3996 eV·Å)
// E_sr(r)  = E_C(r) · u(r/r_c)               toy short-range term: the Coulomb
//                                            pair damped by the DimeNet polynomial
//                                            envelope u, so it is zero for r ≥ r_c
// E_C − E_sr = the tail a strictly local pair term cannot represent.
//
// This is a toy decomposition to make the cutoff visible. Real MLIPs do not
// learn a damped Coulomb term; their short-range part is a learned function of
// the whole neighbourhood.

import { COULOMB_K } from '../chem/elements';
import { polynomialEnvelope } from '../cutoffs';

export function coulomb(r: number, q1: number, q2: number): number {
  return (COULOMB_K * q1 * q2) / r;
}

export function shortRangeToy(r: number, q1: number, q2: number, rc: number, p = 6): number {
  return coulomb(r, q1, q2) * polynomialEnvelope(r, rc, p).f;
}

export function missingTail(r: number, q1: number, q2: number, rc: number, p = 6): number {
  return coulomb(r, q1, q2) - shortRangeToy(r, q1, q2, rc, p);
}
