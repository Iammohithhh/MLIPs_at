import { describe, expect, it } from 'vitest';
import ethanol from '~/data/structures/ethanol.json';
import type { Vec3 } from '../structure/types';
import {
  apply,
  distort,
  energy,
  equivarianceError,
  forces,
  rotateAbout,
  rotation,
  springModel,
  type ModelId,
} from './rotation-toy';

const symbols = ethanol.symbols;
const ref = ethanol.positions as Vec3[];
const m = springModel(symbols, ref);
const R = distort(ref);

describe('rotation matrices', () => {
  it('are orthogonal with determinant 1', () => {
    const Q = rotation([0.3, -0.5, 0.8], 1.234);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const d = Q[i]![0] * Q[j]![0] + Q[i]![1] * Q[j]![1] + Q[i]![2] * Q[j]![2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 12);
      }
    const det =
      Q[0][0] * (Q[1][1] * Q[2][2] - Q[1][2] * Q[2][1]) -
      Q[0][1] * (Q[1][0] * Q[2][2] - Q[1][2] * Q[2][0]) +
      Q[0][2] * (Q[1][0] * Q[2][1] - Q[1][1] * Q[2][0]);
    expect(det).toBeCloseTo(1, 12);
  });
  it('rotate z by 90° maps x to y', () => {
    const v = apply(rotation([0, 0, 1], Math.PI / 2), [1, 0, 0]);
    expect(v[0]).toBeCloseTo(0, 12);
    expect(v[1]).toBeCloseTo(1, 12);
  });
});

describe('spring model on ethanol', () => {
  it('finds the 8 covalent bonds', () => {
    expect(m.bonds).toHaveLength(8);
  });
  it('has zero energy and forces at the reference geometry', () => {
    expect(energy('invariant', m, ref)).toBeCloseTo(0, 12);
    for (const f of forces('invariant', m, ref)) for (const c of f) expect(Math.abs(c)).toBeLessThan(1e-10);
  });
});

describe.each<ModelId>(['invariant', 'coordinate'])('%s model', (model) => {
  it('forces are minus the gradient (finite differences)', () => {
    const f = forces(model, m, R);
    const h = 1e-6;
    for (let i = 0; i < R.length; i++)
      for (let k = 0; k < 3; k++) {
        const plus = R.map((p) => [...p] as Vec3);
        const minus = R.map((p) => [...p] as Vec3);
        plus[i]![k as 0 | 1 | 2] += h;
        minus[i]![k as 0 | 1 | 2] -= h;
        const fd = -(energy(model, m, plus) - energy(model, m, minus)) / (2 * h);
        expect(f[i]![k]!).toBeCloseTo(fd, 5);
      }
  });
  it('forces sum to zero (translation invariance)', () => {
    const f = forces(model, m, R);
    for (let k = 0; k < 3; k++) expect(f.reduce((a, v) => a + v[k]!, 0)).toBeCloseTo(0, 10);
  });
});

describe('symmetry', () => {
  const Q = rotation([0.2, 0.9, -0.4], 2.1);
  it('the distance-only model is invariant and its forces equivariant', () => {
    expect(energy('invariant', m, rotateAbout(R, Q))).toBeCloseTo(energy('invariant', m, R), 10);
    expect(equivarianceError('invariant', m, R, Q)).toBeLessThan(1e-10);
  });
  it('the raw-coordinate model is neither', () => {
    expect(Math.abs(energy('coordinate', m, rotateAbout(R, Q)) - energy('coordinate', m, R))).toBeGreaterThan(0.05);
    expect(equivarianceError('coordinate', m, R, Q)).toBeGreaterThan(0.1);
  });
  it('rotation about the lab x axis leaves the raw-coordinate energy unchanged', () => {
    const Qx = rotation([1, 0, 0], 1.3);
    expect(energy('coordinate', m, rotateAbout(R, Qx))).toBeCloseTo(energy('coordinate', m, R), 10);
  });
});
