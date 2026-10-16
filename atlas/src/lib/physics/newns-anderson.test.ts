import { describe, expect, it } from 'vitest';
import { COULOMB_K } from '../chem/elements';
import { DEFAULT_DISCHARGE as P, discharge, levelAt, levelDos, occupation, shiftPerElectron } from './newns-anderson';

describe('Newns–Anderson ion discharge (toy)', () => {
  it('the level is the free-ion level shifted up by the image term', () => {
    expect(levelAt(5, P)).toBeCloseTo(-P.I + COULOMB_K / 20, 12);
    expect(levelAt(1e9, P)).toBeCloseTo(-P.I, 6);
  });

  it('occupation is ½ at resonance, bounded, and falls as the level rises', () => {
    expect(occupation(-5, 0.5, -5)).toBeCloseTo(0.5, 12);
    let prev = 1;
    for (let e = -8; e <= -2; e += 0.25) {
      const n = occupation(e, 0.4, -5);
      expect(n).toBeGreaterThan(0);
      expect(n).toBeLessThan(1);
      expect(n).toBeLessThanOrEqual(prev);
      prev = n;
    }
  });

  it('the Lorentzian DOS integrates to one', () => {
    let s = 0;
    for (let e = -200; e <= 200; e += 0.001) s += levelDos(e, 0.3, 0.5) * 0.001;
    expect(s).toBeCloseTo(1, 2);
  });

  it('constant charge solves the self-consistency and transfers less charge than constant potential', () => {
    for (const z of [2.5, 3.5, 5]) {
      const cp = discharge(z, P, 'potential');
      const cc = discharge(z, P, 'charge', 60);
      expect(cc.n).toBeCloseTo(occupation(cc.eps, cc.delta, cc.ef), 9);
      expect(cc.ef).toBeLessThanOrEqual(cp.ef);
      expect(cc.n).toBeLessThanOrEqual(cp.n + 1e-12);
    }
  });

  it('a large electrode recovers the constant-potential result', () => {
    const cp = discharge(3, P, 'potential');
    const big = discharge(3, P, 'charge', 1e9);
    expect(big.n).toBeCloseTo(cp.n, 6);
    expect(shiftPerElectron(100)).toBeGreaterThan(shiftPerElectron(400));
  });
});
