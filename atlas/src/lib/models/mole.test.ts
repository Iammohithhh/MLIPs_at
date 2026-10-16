import { describe, expect, it } from 'vitest';
import {
  ALPHA_FLOOR,
  UMA_EXPERTS,
  atomCount,
  l1,
  makeToyMole,
  mergedApply,
  mergedWeights,
  mixtureApply,
  moleCost,
  relFrob,
  routingCoefficients,
  scaled,
  toyEnergy,
  type SystemSpec,
} from './mole';

const m = makeToyMole();
const nacl: SystemSpec = { composition: { Na: 4, Cl: 4 }, task: 'omat' };

describe('Mixture of Linear Experts', () => {
  it('coefficients are a softmax plus 0.005, one per expert', () => {
    const a = routingCoefficients(m, nacl);
    expect(a.length).toBe(UMA_EXPERTS);
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(1 + ALPHA_FLOOR * UMA_EXPERTS, 12);
    for (const v of a) expect(v).toBeGreaterThan(ALPHA_FLOOR);
  });
  it('merging the experts first gives exactly the expert-by-expert result (Eqs. 1–2)', () => {
    const a = routingCoefficients(m, nacl);
    const W = mergedWeights(m, a);
    const x = m.env.get('Na')!;
    const y1 = mixtureApply(m, a, x);
    const y2 = mergedApply(m, W, x);
    for (let i = 0; i < y1.length; i++) expect(y1[i]).toBeCloseTo(y2[i]!, 12);
  });
  it('depends on the reduced composition only: supercells route identically', () => {
    const a1 = routingCoefficients(m, nacl);
    const a8 = routingCoefficients(m, { ...nacl, composition: scaled(nacl.composition, 27) });
    expect(l1(a1, a8)).toBeLessThan(1e-12);
    const e1 = toyEnergy(m, nacl) / atomCount(nacl.composition);
    const e27 = toyEnergy(m, { ...nacl, composition: scaled(nacl.composition, 27) }) / (27 * 8);
    expect(e27).toBeCloseTo(e1, 12);
  });
  it('a vacancy in a compound changes the merged weights, by less in a larger cell', () => {
    const W0 = mergedWeights(m, routingCoefficients(m, nacl));
    const shift = (n: number) => {
      const c = scaled(nacl.composition, n);
      c.Na! -= 1;
      return relFrob(mergedWeights(m, routingCoefficients(m, { ...nacl, composition: c })), W0);
    };
    expect(shift(1)).toBeGreaterThan(1e-3);
    expect(shift(8)).toBeLessThan(shift(1));
    expect(shift(64)).toBeLessThan(shift(8));
  });
  it('task, charge and spin change the routing', () => {
    const water: SystemSpec = { composition: { H: 2, O: 1 }, task: 'omol', charge: 0, spin: 1 };
    const a = routingCoefficients(m, water);
    expect(l1(a, routingCoefficients(m, { ...water, task: 'omat' }))).toBeGreaterThan(1e-3);
    expect(l1(a, routingCoefficients(m, { ...water, charge: 1, spin: 2 }))).toBeGreaterThan(1e-3);
  });
  it('dividing by N + 1 (UMA-S-1.2) makes the energy per atom depend on the cell size', () => {
    const perAtom = (n: number, self: boolean) => toyEnergy(m, { ...nacl, composition: scaled(nacl.composition, n) }, self) / (8 * n);
    expect(Math.abs(perAtom(1, true) - perAtom(8, true))).toBeGreaterThan(1e-4);
    expect(Math.abs(perAtom(64, true) - perAtom(64, false))).toBeLessThan(Math.abs(perAtom(1, true) - perAtom(1, false)));
  });
  it('merged inference costs one dense layer per input plus a one-off merge', () => {
    const c = moleCost(128, 32, 10000);
    expect(c.mixture / c.merged).toBeGreaterThan(30);
  });
});
