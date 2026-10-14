import { describe, expect, it } from 'vitest';
import { diamond, rocksalt, water } from '~/lib/structure/fixtures';
import { angle, dihedral } from '~/lib/structure/geometry';
import type { Structure } from '~/lib/structure/types';
import { DEFAULT_OPTIONS, buildModel, drawRadius, measure, pickAtom } from './model';

const nacl = rocksalt(5.64);

describe('display model', () => {
  it('draws a complete conventional cell with boundary images', () => {
    const m = buildModel(nacl, DEFAULT_OPTIONS);
    expect(m.atoms).toHaveLength(27);
    expect(m.atoms.filter((a) => a.boundary).length).toBe(19);
    expect(m.bonds.length).toBeGreaterThan(0);
  });

  it('cutoff shell marks the 6 nearest Cl around a Na and adds ghost images outside the cell', () => {
    const opts = { ...DEFAULT_OPTIONS, boundary: false, shell: 3.0 };
    const m = buildModel(nacl, opts, [0]);
    expect(m.shellCount).toBe(6);
    const inShell = m.atoms.filter((a) => a.inShell);
    expect(inShell).toHaveLength(6);
    expect(inShell.every((a) => a.symbol === 'Cl')).toBe(true);
    // Na at the cell corner: 3 of its 6 Cl neighbours are images outside the home cell.
    expect(m.atoms.filter((a) => a.ghost)).toHaveLength(3);
    inShell.forEach((a) => {
      const d = Math.hypot(...a.position.map((v, k) => v - m.atoms[0]!.position[k]!));
      expect(d).toBeCloseTo(2.82, 10);
    });
  });

  it('measures distance, angle and dihedral between drawn atoms', () => {
    const w = water();
    const m = buildModel(w, DEFAULT_OPTIONS);
    expect(measure(m.atoms, [0, 1])).toMatchObject({ kind: 'distance', unit: 'Å' });
    expect(measure(m.atoms, [0, 1])!.value).toBeCloseTo(0.9572, 10);
    expect(measure(m.atoms, [1, 0, 2])!.value).toBeCloseTo(angle(w, 1, 0, 2), 10);
    const chain: Structure = {
      id: 'c',
      title: 'c',
      symbols: ['C', 'C', 'C', 'C'],
      positions: [
        [0, 1, 0.2],
        [0, 0, 0],
        [1.5, 0, 0],
        [1.7, -0.6, 0.8],
      ],
      cell: null,
      pbc: [false, false, false],
    };
    const mc = buildModel(chain, DEFAULT_OPTIONS);
    expect(measure(mc.atoms, [0, 1, 2, 3])!.value).toBeCloseTo(dihedral(chain, 0, 1, 2, 3), 10);
  });

  it('picks the nearest sphere along a ray', () => {
    const m = buildModel(diamond(5.431), { ...DEFAULT_OPTIONS, boundary: false });
    const target = m.atoms[3]!.position;
    const origin = [target[0], target[1], target[2] + 50] as [number, number, number];
    expect(pickAtom(m.atoms, origin, [0, 0, -1])).toBe(3);
    expect(pickAtom(m.atoms, [100, 100, 100], [1, 0, 0])).toBeNull();
  });

  it('uses Cordero radii, scaled for ball-and-stick', () => {
    expect(drawRadius('Na', 'space-filling').r).toBeCloseTo(1.66, 10);
    expect(drawRadius('Na', 'ball-and-stick').r).toBeCloseTo(0.747, 10);
    expect(drawRadius('Og', 'space-filling').tabulated).toBe(false);
  });
});
