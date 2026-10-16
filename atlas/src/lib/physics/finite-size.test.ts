import { describe, expect, it } from 'vitest';
import { atomCount, concentrationPerPair, potentialPerElectron, sideForConcentration, sideForPotential, sigmaPerElectron } from './finite-size';

describe('what a small interface cell can represent (theory)', () => {
  it('one ion pair in 10 × 10 × 20 Å³ is about 0.83 M', () => {
    expect(concentrationPerPair(10, 20)).toBeCloseTo(0.8302, 3);
  });

  it('one electron on 100 Å² is 16 μC/cm²', () => {
    expect(sigmaPerElectron(10)).toBeCloseTo(16.02, 2);
  });

  it('the inversions return the cell side that reaches the target exactly', () => {
    expect(concentrationPerPair(sideForConcentration(0.1, 20), 20)).toBeCloseTo(0.1, 10);
    expect(potentialPerElectron(sideForPotential(0.05))).toBeCloseTo(0.05, 10);
  });

  it('per-electron effects fall as 1/L² while atom count grows as L²', () => {
    expect(potentialPerElectron(20) / potentialPerElectron(10)).toBeCloseTo(0.25, 10);
    const a = atomCount(10, 20).total;
    const b = atomCount(20, 20).total;
    expect(b / a).toBeGreaterThan(3.9);
    expect(b / a).toBeLessThan(4.1);
  });
});
