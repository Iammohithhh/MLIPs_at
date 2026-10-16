import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import nacl from '~/data/structures/nacl-rocksalt.json';
import cu from '~/data/structures/cu-fcc.json';
import type { Structure } from '../structure/types';
import { basisGrid, besselRoot, graphCounts, legendre, m3gnetCutoff, shellAround, sphericalBessel, threeBodyBasis, ylm0 } from './three-body';

const SI = si as unknown as Structure;
const NACL = nacl as unknown as Structure;
const CU = cu as unknown as Structure;

describe('bonds and angles per atom in real crystals', () => {
  it('diamond Si: 4 bonds and 6 angles inside 3 Å (shells at 2.35, 3.84, 4.50 Å)', () => {
    expect(graphCounts(SI, 3)).toMatchObject({ bondsPerAtom: 4, anglesPerAtom: 6 });
    expect(graphCounts(SI, 4)).toMatchObject({ bondsPerAtom: 16, anglesPerAtom: 120 });
    expect(graphCounts(SI, 5)).toMatchObject({ bondsPerAtom: 28, anglesPerAtom: 378 });
  });

  it('rock-salt NaCl (a = 5.64 Å): 6, then 18, then 26 neighbours', () => {
    expect(graphCounts(NACL, 3).bondsPerAtom).toBe(6);
    expect(graphCounts(NACL, 4).bondsPerAtom).toBe(18);
    expect(graphCounts(NACL, 5).bondsPerAtom).toBe(26);
    expect(graphCounts(NACL, 5).anglesPerAtom).toBe(325);
  });

  it('fcc Cu: the 12 nearest neighbours give 66 angles', () => {
    expect(graphCounts(CU, 3)).toMatchObject({ bondsPerAtom: 12, anglesPerAtom: 66 });
  });

  it('the shell around an atom matches the per-atom count and is sorted by distance', () => {
    const sh = shellAround(SI, 5);
    expect(sh).toHaveLength(28);
    for (let k = 1; k < sh.length; k++) expect(sh[k]!.r).toBeGreaterThanOrEqual(sh[k - 1]!.r);
    expect(Math.hypot(...sh[0]!.p)).toBeCloseTo(sh[0]!.r, 10);
  });
});

describe('M3GNet three-body basis', () => {
  it('spherical Bessel functions match closed forms', () => {
    for (const x of [0.3, 1.7, 4.2, 9.1]) {
      expect(sphericalBessel(0, x)).toBeCloseTo(Math.sin(x) / x, 12);
      expect(sphericalBessel(1, x)).toBeCloseTo(Math.sin(x) / x ** 2 - Math.cos(x) / x, 12);
      expect(sphericalBessel(2, x)).toBeCloseTo((3 / x ** 2 - 1) * (Math.sin(x) / x) - (3 * Math.cos(x)) / x ** 2, 10);
    }
  });

  it('roots: j0 at nπ, j1 at 4.4934, j2 at 5.7635', () => {
    expect(besselRoot(0, 1)).toBeCloseTo(Math.PI, 9);
    expect(besselRoot(0, 3)).toBeCloseTo(3 * Math.PI, 9);
    expect(besselRoot(1, 1)).toBeCloseTo(4.4934, 4);
    expect(besselRoot(2, 1)).toBeCloseTo(5.7635, 4);
  });

  it('Y_l^0 is normalised on the sphere and P_l matches closed forms', () => {
    for (const l of [0, 1, 2, 3]) {
      let s = 0;
      const N = 4000;
      for (let k = 0; k < N; k++) {
        const t = ((k + 0.5) / N) * Math.PI;
        s += ylm0(l, t) ** 2 * Math.sin(t) * (Math.PI / N) * 2 * Math.PI;
      }
      expect(s).toBeCloseTo(1, 5);
    }
    expect(legendre(2, 0.3)).toBeCloseTo(0.5 * (3 * 0.09 - 1), 12);
  });

  it('the polynomial cutoff is 1 at 0 and reaches 0 with zero first and second derivative at r_c', () => {
    expect(m3gnetCutoff(0, 4)).toBe(1);
    expect(m3gnetCutoff(4, 4)).toBe(0);
    const h = 1e-4;
    const f = (r: number) => m3gnetCutoff(r, 4);
    expect(Math.abs((f(4) - f(4 - h)) / h)).toBeLessThan(1e-6);
    expect(Math.abs((f(4) - 2 * f(4 - h) + f(4 - 2 * h)) / h ** 2)).toBeLessThan(1e-3);
  });

  it('each basis function vanishes at the three-body cutoff; l = 0 does not depend on the angle', () => {
    expect(Math.abs(threeBodyBasis(1, 2, 4, 1.1, 4))).toBe(0);
    expect(threeBodyBasis(0, 1, 2, 0.4, 4)).toBeCloseTo(threeBodyBasis(0, 1, 2, 2.6, 4), 12);
    expect(threeBodyBasis(1, 1, 2, 0.4, 4)).not.toBeCloseTo(threeBodyBasis(1, 1, 2, 2.6, 4), 3);
    const g = basisGrid(2, 1, 4, [1, 2, 3], [0, 1, 2]);
    expect(g).toHaveLength(3);
    expect(g[0]).toHaveLength(3);
  });
});
