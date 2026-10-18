import { describe, expect, it } from 'vitest';
import { buildBasis } from './ace';
import { cellFeatures, cells, CORE, coreRepulsion, LAB_RADIAL, LAB_SPEC, neighbourList, referenceDimer, referenceEnergy, referenceForces, runLab, sample } from './ace-fit';
import { diamondCell } from './gap';
import { phi2 } from '../physics/stillinger-weber';
import { tersoffSite } from '../physics/tersoff';

describe('reference potentials', () => {
  it('give the familiar diamond-Si energies and a Tersoff minimum near 5.43 Å', () => {
    expect(referenceEnergy(diamondCell(), 'sw') / 8).toBeCloseTo(-4.3366, 4);
    const e = (a: number) => referenceEnergy(diamondCell(a), 'tersoff') / 8;
    expect(e(5.43)).toBeCloseTo(-4.6304, 3);
    expect(e(5.43)).toBeLessThan(e(5.40));
    expect(e(5.43)).toBeLessThan(e(5.46));
  });

  it('have forces that sum to zero and vanish in the perfect crystal', () => {
    const c = cells({ count: 1, maxStrain: 0.05, maxRattle: 0.15, seed: 11 })[0]!;
    for (const ref of ['sw', 'tersoff'] as const) {
      const f = referenceForces(c, ref);
      for (let a = 0; a < 3; a++) expect(Math.abs(f.reduce((s, v) => s + v[a]!, 0))).toBeLessThan(1e-6);
      for (const v of referenceForces(diamondCell(), ref)) expect(Math.hypot(...v)).toBeLessThan(1e-6);
    }
  });

  it('reduce to the pair term for a dimer', () => {
    expect(referenceDimer(2.4, 'sw')).toBeCloseTo(phi2(2.4), 12);
    expect(referenceDimer(2.4, 'tersoff')).toBeCloseTo(2 * tersoffSite([[2.4, 0, 0]]), 12);
    expect(referenceDimer(2.4, 'tersoff')).toBeCloseTo(3264.7 * Math.exp(-3.2394 * 2.4) - 95.373 * Math.exp(-1.3258 * 2.4), 10);
  });
});

describe('ACE features of a cell', () => {
  it('have force rows equal to minus the finite-difference gradient of the summed basis', () => {
    const basis = buildBasis({ ...LAB_SPEC, numax: 3 });
    const c = cells({ count: 1, maxStrain: 0.05, maxRattle: 0.15, seed: 5 })[0]!;
    const f = cellFeatures(basis, LAB_RADIAL, c);
    const h = 1e-5;
    let worst = 0;
    let scale = 0;
    for (const [j, a] of [[0, 0], [5, 2]] as const) {
      const shift = (d: number) => ({ L: c.L, positions: c.positions.map((p, k) => (k === j ? (p.map((x, q) => (q === a ? x + d : x)) as typeof p) : p)) });
      const fp = cellFeatures(basis, LAB_RADIAL, shift(h)).total;
      const fm = cellFeatures(basis, LAB_RADIAL, shift(-h)).total;
      fp.forEach((x, v) => {
        worst = Math.max(worst, Math.abs(-(x - fm[v]!) / (2 * h) - f.force[3 * j + a]![v]!));
        scale = Math.max(scale, Math.abs(f.force[3 * j + a]![v]!));
      });
    }
    expect(worst / scale).toBeLessThan(1e-7);
  });

  it('see a neighbour list that is the same from every periodic image', () => {
    const nl = neighbourList(diamondCell(), 2.5);
    expect(nl.every((nb) => nb.length === 4)).toBe(true);
  });
});

describe('the lab', () => {
  const sw = runLab({ ref: 'sw', count: 30, ridge: 1e-6 });
  const at = (nu: number, l: number) => sw.fits.find((f) => f.nu === nu && f.lmax === l)!;

  it('reproduces Stillinger–Weber only from ν = 2 and l = 2, as Eq. (7) says', () => {
    expect(at(1, 3).train.force).toBeGreaterThan(0.2);
    expect(at(2, 1).train.force).toBeGreaterThan(3 * at(2, 2).train.force);
    expect(at(2, 2).train.force).toBeLessThan(0.03);
    // more orders cannot do much better than the radial basis allows
    expect(at(4, 3).train.force).toBeGreaterThan(0.3 * at(2, 2).train.force);
  });

  it('adds a core repulsion that is negligible on the training cells', () => {
    expect(coreRepulsion(2.0, CORE.pre, CORE.lambda, LAB_RADIAL.rc)).toBeLessThan(1e-7);
    const tr = cells({ count: 30, maxStrain: 0.05, maxRattle: 0.15, seed: 3 });
    for (const c of tr)
      for (const nb of neighbourList(c, LAB_RADIAL.rc))
        expect(nb.reduce((s, { v }) => s + coreRepulsion(Math.hypot(...v), CORE.pre, CORE.lambda, LAB_RADIAL.rc), 0)).toBeLessThan(1e-4);
    expect(sw.dimerCore[0]).toBeCloseTo(2 * 100 * Math.exp(-5) * 0.5 * (1 + Math.cos(Math.PI / 3.8)), 10);
  });

  it('fits every (ν, l_max) cell from fixed-seed data', () => {
    expect(sw.fits).toHaveLength(LAB_SPEC.numax * (LAB_SPEC.lmax + 1));
    const again = sample(cells({ count: 1, maxStrain: 0.05, maxRattle: 0.15, seed: 99 })[0]!, 'sw');
    expect(again.forces).toEqual(sw.shown[0]!.forces);
  });
});
