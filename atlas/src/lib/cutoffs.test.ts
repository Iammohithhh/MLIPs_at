import { describe, expect, it } from 'vitest';
import { cosineCurvatureJump, cosineCutoff, polynomialEnvelope } from './cutoffs';

const rc = 5;

/** Central finite differences for checking analytic derivatives. */
function fd(fn: (r: number) => number, r: number, h = 1e-5) {
  return {
    d1: (fn(r + h) - fn(r - h)) / (2 * h),
    d2: (fn(r + h) - 2 * fn(r) + fn(r - h)) / (h * h),
  };
}

describe('cosine cutoff', () => {
  it('is 1 at r = 0 and 0 at and beyond rc', () => {
    expect(cosineCutoff(0, rc).f).toBeCloseTo(1, 12);
    expect(cosineCutoff(rc, rc)).toEqual({ f: 0, df: 0, d2f: 0 });
    expect(cosineCutoff(rc + 1, rc).f).toBe(0);
  });

  it('analytic derivatives match finite differences', () => {
    for (const r of [0.3, 1.7, 2.5, 4.1, 4.9]) {
      const n = fd((x) => cosineCutoff(x, rc).f, r);
      expect(cosineCutoff(r, rc).df).toBeCloseTo(n.d1, 6);
      expect(cosineCutoff(r, rc).d2f).toBeCloseTo(n.d2, 3);
    }
  });

  it('f and f′ are continuous at rc but f″ jumps by π²/(2rc²)', () => {
    const below = cosineCutoff(rc - 1e-9, rc);
    expect(below.f).toBeCloseTo(0, 12);
    expect(below.df).toBeCloseTo(0, 8);
    expect(below.d2f).toBeCloseTo(cosineCurvatureJump(rc), 8);
    expect(cosineCurvatureJump(rc)).toBeGreaterThan(0);
  });
});

describe('polynomial envelope', () => {
  for (const p of [3, 5, 6, 8]) {
    it(`p = ${p}: u(0) = 1 and u = u′ = u″ = 0 at rc`, () => {
      expect(polynomialEnvelope(0, rc, p).f).toBeCloseTo(1, 12);
      const eps = 1e-9;
      const below = polynomialEnvelope(rc * (1 - eps), rc, p);
      expect(below.f).toBeCloseTo(0, 10);
      expect(below.df).toBeCloseTo(0, 10);
      // u″ vanishes linearly: u″(1 − ε) ≈ 2Kε with K = p(p+1)(p+2)/2.
      const K = (p * (p + 1) * (p + 2)) / 2;
      expect(below.d2f * rc * rc).toBeCloseTo(2 * K * eps, 12);
    });

    it(`p = ${p}: analytic derivatives match finite differences`, () => {
      for (const r of [0.8, 2.2, 3.3, 4.6]) {
        const n = fd((x) => polynomialEnvelope(x, rc, p).f, r);
        expect(polynomialEnvelope(r, rc, p).df).toBeCloseTo(n.d1, 6);
        expect(polynomialEnvelope(r, rc, p).d2f).toBeCloseTo(n.d2, 3);
      }
    });
  }

  it('third derivative is what jumps at rc (non-zero just below)', () => {
    const p = 6;
    const h = 1e-4;
    const r = rc - 2 * h;
    const d3 = (polynomialEnvelope(r + h, rc, p).d2f - polynomialEnvelope(r - h, rc, p).d2f) / (2 * h);
    // Analytic: u‴(1) = −2K with K = p(p+1)(p+2)/2, so f‴(rc⁻) = −p(p+1)(p+2)/rc³.
    expect(d3).toBeCloseTo((-p * (p + 1) * (p + 2)) / rc ** 3, 1);
  });
});
