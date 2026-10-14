import { describe, expect, it } from 'vitest';
import { cubeSeries, cubeSum, erfc, MADELUNG_NACL, madelungEwald, sphereSeries } from './madelung';
import { coulomb, missingTail, shortRangeToy } from './pair';

describe('erfc', () => {
  it('matches reference values', () => {
    expect(erfc(0)).toBeCloseTo(1, 7);
    expect(erfc(0.5)).toBeCloseTo(0.4795001222, 7);
    expect(erfc(1)).toBeCloseTo(0.1572992071, 7);
    expect(erfc(2)).toBeCloseTo(0.004677734981, 8);
    expect(erfc(-1)).toBeCloseTo(1.8427007929, 7);
  });
});

describe('Madelung constant of rock salt', () => {
  it('Ewald summation reproduces M = 1.747565', () => {
    expect(madelungEwald()).toBeCloseTo(MADELUNG_NACL, 6);
  });

  it('Ewald result does not depend on the splitting parameter', () => {
    expect(madelungEwald(0.9)).toBeCloseTo(madelungEwald(1.6), 6);
  });

  it('Evjen-weighted cubes converge fast', () => {
    expect(cubeSum(6, true)).toBeCloseTo(MADELUNG_NACL, 3);
  });

  it('plain expanding cubes approach M', () => {
    const s = cubeSeries(20);
    expect(Math.abs(s[19]!.M - MADELUNG_NACL)).toBeLessThan(Math.abs(s[3]!.M - MADELUNG_NACL));
    expect(Math.abs(s[19]!.M - MADELUNG_NACL)).toBeLessThan(0.1);
  });

  it('expanding spheres keep oscillating instead of settling', () => {
    const s = sphereSeries(20).filter((p) => p.R > 10);
    const values = s.map((p) => p.M);
    const spread = Math.max(...values) - Math.min(...values);
    expect(spread).toBeGreaterThan(1);
  });

  it('first shell gives 6, the second 6 − 12/√2', () => {
    const s = sphereSeries(2);
    expect(s[0]!.M).toBeCloseTo(6, 12);
    expect(s[1]!.M).toBeCloseTo(6 - 12 / Math.SQRT2, 12);
    expect(s[0]!.count).toBe(6);
    expect(s[1]!.count).toBe(18);
  });
});

describe('pair toy', () => {
  it('Na⁺–Cl⁻ at 2.82 Å is about −5.1 eV', () => {
    expect(coulomb(2.82, 1, -1)).toBeCloseTo(-5.106, 3);
  });
  it('short-range toy + tail = Coulomb, and the toy is exactly zero beyond r_c', () => {
    for (const r of [1, 2.5, 4.9, 5, 7, 12]) {
      expect(shortRangeToy(r, 1, -1, 5) + missingTail(r, 1, -1, 5)).toBeCloseTo(coulomb(r, 1, -1), 12);
    }
    expect(Math.abs(shortRangeToy(5, 1, -1, 5))).toBe(0);
    expect(Math.abs(shortRangeToy(9, 1, -1, 5))).toBe(0);
  });
});
