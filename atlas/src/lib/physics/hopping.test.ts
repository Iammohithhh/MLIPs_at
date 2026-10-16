import { describe, expect, it } from 'vitest';
import { KB, deltaForFactor, diffusivity, diffusivityRatio, hopRate, msdPerHop, walk } from './hopping';

describe('activated hopping (theory and toy)', () => {
  it('a 60 meV barrier error changes D by about an order of magnitude at 298 K', () => {
    expect(diffusivityRatio(0.06, 298)).toBeGreaterThan(10);
    expect(diffusivityRatio(0.06, 298)).toBeLessThan(11);
    expect(deltaForFactor(10, 298)).toBeCloseTo(KB * 298 * Math.log(10), 12);
  });

  it('D scales with the hop rate as a²ν exp(−E/kT)', () => {
    expect(diffusivity(0.3, 300) / diffusivity(0.5, 300)).toBeCloseTo(diffusivityRatio(0.2, 300), 8);
    expect(hopRate(0, 300)).toBe(1e13);
  });

  it('a lattice walk has mean squared displacement of one site spacing² per hop (D = Γa²/6 with Γ = 6 rates)', () => {
    expect(msdPerHop(400, 200, 7)).toBeCloseTo(1, 1);
  });

  it('walks are reproducible and take unit steps', () => {
    const a = walk(50, 3);
    expect(a).toEqual(walk(50, 3));
    for (let i = 1; i < a.length; i++) {
      const d = a[i]!.map((v, k) => Math.abs(v - a[i - 1]![k]!));
      expect(d.reduce((x, y) => x + y, 0)).toBe(1);
    }
  });
});
