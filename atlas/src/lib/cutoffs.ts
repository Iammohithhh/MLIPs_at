// Smooth cutoff functions used by MLIP descriptors and radial bases, with exact
// first and second derivatives. Both vanish for r ≥ rc.
//
// cosine:     f(r) = ½[cos(πr/rc) + 1]                       (Behler 2011)
// polynomial: u(d) = 1 − (p+1)(p+2)/2 dᵖ + p(p+2) dᵖ⁺¹ − p(p+1)/2 dᵖ⁺²,  d = r/rc
//                                                             (DimeNet envelope, 2020)
//
// At r = rc the cosine cutoff has f = f′ = 0 but f″ → π²/(2rc²) from below: the
// curvature jumps. The polynomial envelope has u = u′ = u″ = 0 at d = 1 (derived
// in cutoffs.test.ts), so the jump moves to the third derivative.

export interface CutoffValue {
  f: number;
  /** df/dr, in 1/length. */
  df: number;
  /** d²f/dr², in 1/length². */
  d2f: number;
}

const ZERO: CutoffValue = { f: 0, df: 0, d2f: 0 };

export function cosineCutoff(r: number, rc: number): CutoffValue {
  if (r >= rc) return ZERO;
  const k = Math.PI / rc;
  return {
    f: 0.5 * (Math.cos(k * r) + 1),
    df: -0.5 * k * Math.sin(k * r),
    d2f: -0.5 * k * k * Math.cos(k * r),
  };
}

export function polynomialEnvelope(r: number, rc: number, p: number): CutoffValue {
  if (r >= rc) return ZERO;
  const d = r / rc;
  const K = (p * (p + 1) * (p + 2)) / 2;
  const f = 1 - (((p + 1) * (p + 2)) / 2) * d ** p + p * (p + 2) * d ** (p + 1) - ((p * (p + 1)) / 2) * d ** (p + 2);
  // u′(d) = −K d^(p−1) (1−d)²;  u″(d) = −K d^(p−2) (1−d) [(p−1) − (p+1) d]
  const du = -K * d ** (p - 1) * (1 - d) ** 2;
  const d2u = -K * d ** (p - 2) * (1 - d) * (p - 1 - (p + 1) * d);
  return { f, df: du / rc, d2f: d2u / (rc * rc) };
}

/** Size of the jump in f″ at r = rc for the cosine cutoff (limit from below minus 0). */
export function cosineCurvatureJump(rc: number): number {
  return (Math.PI * Math.PI) / (2 * rc * rc);
}
