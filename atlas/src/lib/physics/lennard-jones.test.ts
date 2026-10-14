import { describe, expect, it } from 'vitest';
import { ljEnergy, ljForces } from './lennard-jones';
import type { Vec3 } from '../structure/types';

const p = { epsilon: 1, sigma: 1 };
const cluster: Vec3[] = [
  [0, 0, 0],
  [1.1, 0.05, 0],
  [0.5, 1.0, 0.1],
  [0.6, 0.4, 0.95],
  [-0.4, 0.8, 0.7],
];

describe('Lennard-Jones toy', () => {
  it('has its minimum at r = 2^(1/6) σ with depth −ε', () => {
    const rmin = 2 ** (1 / 6);
    expect(ljEnergy([[0, 0, 0], [rmin, 0, 0]], p)).toBeCloseTo(-1, 12);
    ljForces([[0, 0, 0], [rmin, 0, 0]], p).flat().forEach((v) => expect(v).toBeCloseTo(0, 12));
  });

  it('forces are the negative gradient of the energy (finite differences)', () => {
    const f = ljForces(cluster, p);
    const h = 1e-6;
    cluster.forEach((_, i) =>
      [0, 1, 2].forEach((c) => {
        const plus = cluster.map((q) => [...q] as Vec3);
        const minus = cluster.map((q) => [...q] as Vec3);
        plus[i]![c]! += h;
        minus[i]![c]! -= h;
        const g = (ljEnergy(plus, p) - ljEnergy(minus, p)) / (2 * h);
        expect(f[i]![c]!).toBeCloseTo(-g, 4);
      }),
    );
  });

  it('total force vanishes (Newton’s third law)', () => {
    const f = ljForces(cluster, p);
    [0, 1, 2].forEach((c) => expect(f.reduce((s, v) => s + v[c]!, 0)).toBeCloseTo(0, 10));
  });
});
