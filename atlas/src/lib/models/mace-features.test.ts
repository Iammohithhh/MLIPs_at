import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import zno from '~/data/structures/zno-wurtzite.json';
import type { Structure, Vec3 } from '../structure/types';
import { neighboursOf } from '../structure/geometry';
import { apply, rotation } from '../physics/rotation-toy';
import { atomicBasis, bessel, powerSpectrum, productCount, radial } from './mace-features';

const env = (s: Structure, i: number, rc: number): Vec3[] =>
  neighboursOf(s, i, rc).map((n) => [n.position[0] - s.positions[i]![0], n.position[1] - s.positions[i]![1], n.position[2] - s.positions[i]![2]] as Vec3);

describe('Bessel radial basis', () => {
  it('is orthonormal on [0, r_c] with weight r²', () => {
    const rc = 5;
    const N = 20000;
    for (let a = 1; a <= 4; a++)
      for (let b = 1; b <= 4; b++) {
        let s = 0;
        for (let k = 0; k < N; k++) {
          const r = ((k + 0.5) / N) * rc;
          s += bessel(a, r, rc) * bessel(b, r, rc) * r * r * (rc / N);
        }
        expect(s).toBeCloseTo(a === b ? 1 : 0, 4);
      }
  });
  it('the enveloped radial function vanishes smoothly at the cutoff', () => {
    expect(radial(3, 4.999999, 5)).toBeCloseTo(0, 8);
    expect(radial(3, 5.2, 5)).toBe(0);
  });
});

describe('atomic basis A and its power spectrum', () => {
  const S = si as unknown as Structure;
  const d = env(S, 0, 5);

  it('diamond: the l = 1 and l = 2 blocks vanish by tetrahedral symmetry; l = 3 does not', () => {
    const P = powerSpectrum(atomicBasis(d, 5));
    expect(P[1]!).toBeLessThan(1e-12);
    expect(P[2]!).toBeLessThan(1e-12);
    expect(P[3]!).toBeGreaterThan(1e-4);
  });

  it('wurtzite ZnO (no inversion centre at the atom): the l = 1 block is non-zero', () => {
    const Z = zno as unknown as Structure;
    const P = powerSpectrum(atomicBasis(env(Z, 0, 5), 5));
    expect(P[1]!).toBeGreaterThan(1e-6);
  });

  it('the power spectrum is invariant under rotation; the components are not', () => {
    const Q = rotation([0.3, -0.7, 0.4], 0.9);
    const Z = zno as unknown as Structure;
    const dz = env(Z, 0, 5);
    const A = atomicBasis(dz, 5);
    const B = atomicBasis(dz.map((v) => apply(Q, v)), 5);
    powerSpectrum(A).forEach((p, l) => expect(powerSpectrum(B)[l]!).toBeCloseTo(p, 10));
    expect(A[1]!.some((v, k) => Math.abs(v - B[1]![k]!) > 1e-4)).toBe(true);
  });
});

describe('product-basis counting', () => {
  it('counts multisets of the 16 components for l ≤ 3', () => {
    expect(productCount(16, 1)).toBe(16);
    expect(productCount(16, 2)).toBe(136);
    expect(productCount(16, 3)).toBe(816);
  });
});
