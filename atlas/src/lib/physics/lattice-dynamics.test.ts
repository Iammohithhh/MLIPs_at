import { describe, expect, it } from 'vitest';
import { COULOMB_K } from '../chem/elements';
import { MASS, THZ, coulombT, dynamicalMatrix, jacobiEigen, modes, rockSalt, shortRange, type RockSaltModel } from './lattice-dynamics';

const M: RockSaltModel = { r0: 2.82, Z: 1, rho: 0.32, coulomb: 'ewald' };
const mu = (MASS.Na * MASS.Cl) / (MASS.Na + MASS.Cl);
const L = rockSalt(M.r0, M.Z);

describe('rigid-ion rock salt lattice dynamics (toy)', () => {
  it('Ewald lattice sums do not depend on the splitting parameter η', () => {
    const q: [number, number, number] = [0.3, 0.1, -0.2];
    const a = coulombT(L, q, [M.r0, 0, 0], q, 0.45, 13, 11);
    const b = coulombT(L, q, [M.r0, 0, 0], q, 0.7, 13, 13);
    for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) expect(a[x]![y]!).toBeCloseTo(b[x]![y]!, 6);
  });

  it('obeys the acoustic sum rule: three zero modes at Γ', () => {
    const { values } = jacobiEigen(dynamicalMatrix(M, [0, 0, 0], [1, 0, 0]));
    for (const v of values.slice(0, 3)) expect(Math.abs(v)).toBeLessThan(1e-6);
  });

  it('TO at Γ matches μω² = 2A + 4B − 4πkZ²/(3Ω) (Lorentz local field)', () => {
    const { A, B } = shortRange(M);
    const expected = Math.sqrt((2 * A + 4 * B - (4 * Math.PI * COULOMB_K) / (3 * L.omega)) / mu) * THZ;
    const opt = modes(M, [0, 0, 0], [1, 0, 0]).filter((m) => m.optical);
    const to = opt.filter((m) => m.longitudinal < 0.5);
    expect(to).toHaveLength(2);
    for (const m of to) expect(m.nu).toBeCloseTo(expected, 4);
  });

  it('LO–TO splitting at Γ: ω_LO² − ω_TO² = 4πkZ²/(Ωμ), the same in every direction', () => {
    const gap = ((4 * Math.PI * COULOMB_K) / (L.omega * mu)) * THZ * THZ;
    for (const dir of [[1, 0, 0], [1, 1, 1], [1, 2, 0]] as Array<[number, number, number]>) {
      const opt = modes(M, [0, 0, 0], dir).filter((m) => m.optical);
      const lo = opt.find((m) => m.longitudinal > 0.99)!;
      const to = opt.find((m) => m.longitudinal < 0.01)!;
      expect(lo.nu ** 2 - to.nu ** 2).toBeCloseTo(gap, 3);
    }
  });

  it('approaching Γ along q recovers the non-analytic limit', () => {
    const lim = modes(M, [0, 0, 0], [1, 0, 0]).at(-1)!.nu;
    const near = modes(M, [1e-4, 0, 0]).at(-1)!.nu;
    expect(near).toBeCloseTo(lim, 4);
  });

  it('the local model reproduces TO at Γ but has no LO–TO splitting, whatever its cutoff', () => {
    const to = modes(M, [0, 0, 0], [1, 0, 0]).filter((m) => m.optical && m.longitudinal < 0.5)[0]!.nu;
    for (const rc of [5, 9, 13]) {
      const nus = modes({ ...M, coulomb: 'cutoff', rc }, [0, 0, 0], [1, 0, 0]).filter((m) => m.optical).map((m) => m.nu);
      expect(nus).toHaveLength(3);
      for (const nu of nus) expect(nu).toBeCloseTo(to, 6);
    }
  });
});
