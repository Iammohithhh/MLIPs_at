import { describe, expect, it } from 'vitest';
import { EPS0, potentialStep, slabEnergyPerArea, slabProfile, vacuumField } from './slab-dipole';

const params = { p: 0.005, Lz: 30, zSheet: 12 };
const prof = slabProfile(params);
const at = (arr: number[], z: number) => arr[Math.min(arr.length - 1, Math.floor((z / params.Lz) * arr.length))]!;

describe('ε₀ in Atlas units', () => {
  it('is 8.8541878e-12 F/m expressed in e/(V·Å)', () => {
    expect(EPS0).toBeCloseTo(8.8541878128e-12 / 1.602176634e-19 / 1e10, 9);
  });
});

describe('periodic boundary conditions', () => {
  it('have zero mean field and a periodic potential', () => {
    const meanE = prof.Eper.reduce((a, b) => a + b, 0) / prof.Eper.length;
    expect(Math.abs(meanE)).toBeLessThan(1e-10);
    expect(Math.abs(prof.phiPer.at(-1)!)).toBeLessThan(1e-6);
  });
  it('produce the artificial vacuum field p/(ε₀ L_z)', () => {
    expect(at(prof.Eper, 4)).toBeCloseTo(vacuumField(params.p, params.Lz), 6);
    expect(at(prof.Eper, 24)).toBeCloseTo(vacuumField(params.p, params.Lz), 6);
  });
  it('the artificial field decays as 1/L_z', () => {
    expect(vacuumField(0.005, 60)).toBeCloseTo(vacuumField(0.005, 30) / 2, 12);
  });
});

describe('dipole correction', () => {
  it('gives zero field in the vacuum on both sides', () => {
    expect(Math.abs(at(prof.Ecor, 4))).toBeLessThan(1e-9);
    expect(Math.abs(at(prof.Ecor, 24))).toBeLessThan(1e-9);
  });
  it('gives a potential step p/ε₀ across the dipole layer', () => {
    expect(at(prof.phiCor, 24) - at(prof.phiCor, 4)).toBeCloseTo(potentialStep(params.p), 6);
  });
  it('matches the Yeh–Berkowitz energy M_z²/(2ε₀V) per area', () => {
    const A = 100;
    const Mz = params.p * A;
    expect(slabEnergyPerArea(params.p, params.Lz) * A).toBeCloseTo((Mz * Mz) / (2 * EPS0 * A * params.Lz), 12);
  });
});
