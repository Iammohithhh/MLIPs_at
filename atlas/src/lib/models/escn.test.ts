import { describe, expect, it } from 'vitest';
import { mulberry32, randomRotation } from '../chem/rng';
import {
  ESEN_RCUT,
  applyBlocks,
  edgeFrame,
  edgeMessage,
  equivarianceError,
  esenEnvelope,
  idx,
  makeToyEscn,
  matMul,
  matVec,
  nCoef,
  randomFeature,
  rollDependence,
  shVector,
  signalAt,
  so2Terms,
  tensorProductTerms,
  wignerBlocks,
  type Vec3,
} from './escn';

const rand = mulberry32(5);
const R1 = randomRotation(rand);
const R2 = randomRotation(rand);
const rvec: Vec3 = [1.3, -0.8, 1.9];

describe('real Wigner matrices', () => {
  it('rotate spherical harmonics: Y(R v) = D(R) Y(v)', () => {
    for (const L of [1, 2, 4]) {
      const D = wignerBlocks(L, R1);
      const v: Vec3 = [0.3, -0.5, 0.81];
      const lhs = shVector(L, matVec(R1, v));
      const rhs = applyBlocks(D, shVector(L, v), L);
      for (let i = 0; i < nCoef(L); i++) expect(lhs[i]).toBeCloseTo(rhs[i]!, 10);
    }
  });
  it('are orthogonal and a representation: D(R1 R2) = D(R1) D(R2)', () => {
    const L = 3;
    const x = randomFeature(L, 1);
    const a = applyBlocks(wignerBlocks(L, matMul(R1, R2)), x, L);
    const b = applyBlocks(wignerBlocks(L, R1), applyBlocks(wignerBlocks(L, R2), x, L), L);
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i]!, 10);
    const back = applyBlocks(wignerBlocks(L, R1), applyBlocks(wignerBlocks(L, R1), x, L), L, true);
    for (let i = 0; i < x.length; i++) expect(back[i]).toBeCloseTo(x[i]!, 10);
  });
  it('rotate signals: (D x)(R u) = x(u)', () => {
    const L = 3;
    const x = randomFeature(L, 2);
    const u: Vec3 = [0.6, 0.0, 0.8];
    expect(signalAt(applyBlocks(wignerBlocks(L, R1), x, L), L, matVec(R1, u))).toBeCloseTo(signalAt(x, L, u), 10);
  });
});

describe('edge frame', () => {
  it('maps the bond onto the polar axis (+z here, +y in e3nn) for every roll, so Y_lm(edge) vanishes unless m = 0 (eSCN Eq. 9)', () => {
    for (const g of [0, 1.1, 4]) {
      const Q = edgeFrame(rvec, g);
      const y = matVec(Q, rvec);
      const r = Math.hypot(...rvec);
      expect(y[0]).toBeCloseTo(0, 12);
      expect(y[1]).toBeCloseTo(0, 12);
      expect(y[2]).toBeCloseTo(r, 12);
      const sh = shVector(4, y);
      for (let l = 0; l <= 4; l++) for (let m = -l; m <= l; m++) if (m !== 0) expect(Math.abs(sh[idx(l, m)]!)).toBeLessThan(1e-12);
    }
  });
});

describe('envelope', () => {
  it('is 1 at 0 and has zero value, slope and curvature at the cutoff', () => {
    expect(esenEnvelope(0)).toBe(1);
    const h = 1e-5;
    expect(esenEnvelope(1 - 1e-9)).toBeCloseTo(0, 6);
    expect((esenEnvelope(1) - esenEnvelope(1 - h)) / h).toBeCloseTo(0, 6);
    expect(Math.abs(esenEnvelope(1) - 2 * esenEnvelope(1 - h) + esenEnvelope(1 - 2 * h)) / h ** 2).toBeLessThan(1e-2);
  });
});

describe('SO(2) edge message', () => {
  const xs = randomFeature(3, 11);
  const xt = randomFeature(3, 12);
  const m = makeToyEscn({ L: 3 });

  it('does not depend on the random roll about the bond', () => {
    for (const mmax of [0, 1, 2, 3]) expect(rollDependence(m, xs, xt, rvec, { mmax, gamma: 0, mode: 'so2' })).toBeLessThan(1e-10);
  });
  it('is exactly equivariant for every M_max', () => {
    for (const mmax of [0, 2, 3]) {
      expect(equivarianceError(m, xs, xt, rvec, R1, { mmax, gamma: 0.4, mode: 'so2' })).toBeLessThan(1e-10);
      expect(equivarianceError(m, xs, xt, rvec, R2, { mmax, gamma: 2.0, mode: 'so2' })).toBeLessThan(1e-10);
    }
  });
  it('the dense control depends on the roll and is not equivariant', () => {
    const o = { mmax: 3, gamma: 0, mode: 'dense' as const };
    expect(rollDependence(m, xs, xt, rvec, o)).toBeGreaterThan(0.05);
    expect(equivarianceError(m, xs, xt, rvec, R1, o)).toBeGreaterThan(0.05);
  });
  it('dense with M_max = 0 keeps only m = 0, which the roll leaves alone, so it is equivariant again', () => {
    expect(rollDependence(m, xs, xt, rvec, { mmax: 0, gamma: 0, mode: 'dense' })).toBeLessThan(1e-10);
  });
  it('vanishes smoothly at the cutoff', () => {
    const u = Math.hypot(...rvec);
    const at = (r: number) => edgeMessage(m, xs, xt, rvec.map((c) => (c * r) / u) as Vec3, { mmax: 2, gamma: 0, mode: 'so2' }).message;
    expect(Math.max(...at(ESEN_RCUT + 0.1).map(Math.abs))).toBe(0);
    expect(Math.max(...at(ESEN_RCUT - 1e-3).map(Math.abs))).toBeLessThan(1e-8);
    expect(Math.max(...at(2.5).map(Math.abs))).toBeGreaterThan(1e-3);
  });
  it('M_max < L drops the high orders in the edge frame', () => {
    const r = edgeMessage(m, xs, xt, rvec, { mmax: 1, gamma: 0, mode: 'so2' });
    for (let l = 2; l <= 3; l++) for (const mm of [-l, l]) expect(r.edgeOut[idx(l, mm)]).toBe(0);
  });
});

describe('term counts', () => {
  it('grow as L⁶ for the tensor product and L³ for the SO(2) convolution', () => {
    expect(tensorProductTerms(0)).toBe(1);
    expect(tensorProductTerms(1)).toBe(1 + 9 + 9 + 9 * (1 + 3));
    expect(so2Terms(1).conv).toBe(4 + 4);
    const tp = (L: number) => tensorProductTerms(L);
    const s2 = (L: number) => so2Terms(L).total;
    expect(Math.log(tp(16) / tp(8)) / Math.log(2)).toBeGreaterThan(5.3);
    expect(Math.log(s2(16) / s2(8)) / Math.log(2)).toBeLessThan(3.2);
    expect(so2Terms(6, 2).conv).toBeLessThan(so2Terms(6).conv);
  });
});
