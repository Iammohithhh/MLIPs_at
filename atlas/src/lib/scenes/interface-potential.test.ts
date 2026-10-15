import { describe, expect, it } from 'vitest';
import { EPS0 } from '../physics/slab-dipole';
import { FIRST_LAYER, interfaceProfile } from './interface-potential';

describe('interfacial water potential (toy)', () => {
  it('has a first water layer', () => {
    expect(FIRST_LAYER.length).toBeGreaterThan(5);
  });

  it('the potential step equals the total dipole per area over ε₀ (neutral film)', () => {
    const r = interfaceProfile(60);
    const dz = r.z[1]! - r.z[0]!;
    const total = r.rho.reduce((a, b) => a + b, 0) * dz;
    expect(Math.abs(total)).toBeLessThan(1e-6);
    const p = r.rho.reduce((a, q, i) => a + q * r.z[i]! * dz, 0);
    expect(r.drop).toBeCloseTo(p / EPS0, 4);
  });

  it('flipping the first layer from H-up to H-down changes the step by about twice its dipole over ε₀', () => {
    const up = interfaceProfile(0);
    const down = interfaceProfile(180);
    expect(up.firstLayerDipole).toBeGreaterThan(0);
    expect(down.firstLayerDipole).toBeCloseTo(-up.firstLayerDipole, 10);
    expect(up.drop - down.drop).toBeCloseTo((up.firstLayerDipole - down.firstLayerDipole) / EPS0, 3);
  });

  it('the step is the sum of the ion, first-layer and other-water contributions', () => {
    const r = interfaceProfile(45);
    expect(r.parts.ions + r.parts.firstLayer + r.parts.otherWater).toBeCloseTo(r.drop, 3);
  });
});
