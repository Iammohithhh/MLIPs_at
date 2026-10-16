import { describe, expect, it } from 'vitest';
import data from '~/data/benchmarks/uniffbench-density.json';
import {
  FUNCTIONALS,
  composeLattice,
  densityBudget,
  densityFromLattice,
  histogram,
  latticeFromDensity,
  median,
  signedDensityError,
  summarise,
  type UniffModel,
} from './reference-gap';

describe('density from lattice errors (isotropic strain)', () => {
  it('a 1% larger lattice constant lowers the density by about 3%', () => {
    expect(densityFromLattice(0.01)).toBeCloseTo(1 / 1.01 ** 3 - 1, 12);
    expect(densityFromLattice(0.01)).toBeCloseTo(-0.0294, 4);
    expect(densityFromLattice(0)).toBe(0);
  });

  it('inverts exactly', () => {
    for (const d of [-0.06, -0.025, 0.025, 0.1]) expect(densityFromLattice(latticeFromDensity(d))).toBeCloseTo(d, 12);
  });

  it('the ±2.5% density threshold is a lattice tolerance of about 0.82–0.85%', () => {
    expect(latticeFromDensity(-0.025) * 100).toBeCloseTo(0.847, 3);
    expect(latticeFromDensity(0.025) * 100).toBeCloseTo(-0.820, 3);
  });

  it('PBE’s mean lattice error alone (+1.1%) puts density 3.2% low (−3.23%), outside ±2.5%', () => {
    const pbe = FUNCTIONALS.find((f) => f.id === 'pbe')!;
    const b = densityBudget(pbe.a0MRE, 0);
    expect(b.reference).toBeCloseTo(-3.23, 2);
    expect(b.fit).toBeCloseTo(0, 12);
    expect(Math.abs(b.total)).toBeGreaterThan(2.5);
  });

  it('budget parts add up to the total, and errors compose multiplicatively', () => {
    const b = densityBudget(1.1, 0.7);
    expect(b.reference + b.fit).toBeCloseTo(b.total, 12);
    expect(b.totalLattice).toBeCloseTo(100 * (1.011 * 1.007 - 1), 12);
    expect(composeLattice(0.011, -0.011)).toBeCloseTo(-0.000121, 9);
  });
});

describe('UniFFBench snapshot analysis', () => {
  const models = data.models as UniffModel[];

  it('signed error follows the paper’s definition', () => {
    expect(signedDensityError({ mineral: 'x', T: 298, p_field: 1, rho_exp: 2, rho_sim: 1.9 })).toBeCloseTo(-5, 12);
  });

  it('median, summary and histogram on a hand-made set', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    const runs = [-6, -3, -1, 2, 150].map((e) => ({ mineral: 'm', T: 298, p_field: 1, rho_exp: 1, rho_sim: 1 + e / 100 }));
    const s = summarise(runs);
    expect([s.runs, s.kept]).toEqual([5, 4]);
    expect(s.median).toBeCloseTo(-2, 9);
    expect(s.mape).toBeCloseTo(3, 9);
    expect(s.fracUnder).toBe(0.75);
    expect(s.fracWithin).toBe(0.5);
    const h = histogram([-30, -2.5, -0.1, 0, 0.9, 40], -20, 20, 1);
    expect(h.counts).toHaveLength(40);
    expect([h.below, h.above]).toEqual([1, 1]);
    expect(h.counts[17]).toBe(1);
    expect(h.counts[19]).toBe(1);
    expect(h.counts[20]).toBe(2);
  });

  it('the snapshot is complete and pinned to an upstream commit', () => {
    expect(data.source.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(models.map((m) => m.key)).toEqual(['orb', 'mattersim']);
    for (const m of models) {
      expect(m.runs.length).toBeGreaterThan(2000);
      expect(m.runs.every((r) => r.rho_exp > 0 && r.rho_sim > 0)).toBe(true);
    }
  });

  it('in both result sets most minerals come out less dense than measured', () => {
    for (const m of models) {
      const s = summarise(m.runs);
      expect(s.median).toBeLessThan(0);
      expect(s.fracUnder).toBeGreaterThan(0.5);
    }
  });
});
