import { describe, expect, it } from 'vitest';
import { diamond, rocksalt, water } from './fixtures';
import {
  angle,
  cartToFrac,
  cellParameters,
  dihedral,
  displayAtoms,
  displayBondPairs,
  displayBonds,
  distance,
  fracToCart,
  inv3,
  minimumImage,
  neighboursOf,
  norm,
  pairList,
  planeSpacings,
  vecMat,
} from './geometry';
import type { Mat3, Structure, Vec3 } from './types';

const nacl = rocksalt(5.64);
const si = diamond(5.431);

describe('cell algebra', () => {
  const tri: Mat3 = [
    [4.1, 0, 0],
    [1.3, 3.9, 0],
    [0.7, -0.9, 5.2],
  ];
  it('inverts the cell and round-trips fractional coordinates', () => {
    const r: Vec3 = [1.234, -2.5, 3.75];
    const back = fracToCart(cartToFrac(r, tri), tri);
    back.forEach((v, k) => expect(v).toBeCloseTo(r[k]!, 12));
    const I = tri.map((row) => vecMat(row, inv3(tri)));
    I.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(i === j ? 1 : 0, 12)));
  });

  it('gives cubic lattice parameters and plane spacings', () => {
    const p = cellParameters(nacl.cell!);
    p.lengths.forEach((l) => expect(l).toBeCloseTo(5.64, 12));
    p.angles.forEach((a) => expect(a).toBeCloseTo(90, 12));
    planeSpacings(nacl.cell!).forEach((d) => expect(d).toBeCloseTo(5.64, 12));
  });
});

describe('minimum image', () => {
  it('matches a brute-force search over many images in a skewed triclinic cell', () => {
    const cell: Mat3 = [
      [5, 0, 0],
      [2.2, 4.6, 0],
      [-1.4, 1.1, 4.4],
    ];
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    for (let t = 0; t < 300; t++) {
      const d: Vec3 = [rnd() * 12, rnd() * 12, rnd() * 12];
      let best = Infinity;
      for (let a = -5; a <= 5; a++)
        for (let b = -5; b <= 5; b++)
          for (let c = -5; c <= 5; c++) best = Math.min(best, norm(vecMat([a, b, c], cell).map((v, k) => v + d[k]!) as Vec3));
      expect(norm(minimumImage(d, cell, [true, true, true]).d)).toBeCloseTo(best, 10);
    }
  });

  it('respects non-periodic directions', () => {
    const d: Vec3 = [0, 0, 4.5];
    expect(minimumImage(d, nacl.cell, [true, true, false]).d).toEqual(d);
    expect(norm(minimumImage(d, nacl.cell, [true, true, true]).d)).toBeCloseTo(5.64 - 4.5, 12);
  });
});

describe('crystal geometry from first principles of the lattice', () => {
  it('rock salt: nearest Na–Cl distance a/2 with octahedral coordination', () => {
    const nb = neighboursOf(nacl, 0, 0.5 * 5.64 + 0.01);
    expect(nb).toHaveLength(6);
    nb.forEach((n) => {
      expect(n.distance).toBeCloseTo(2.82, 10);
      expect(nacl.symbols[n.j]).toBe('Cl');
    });
    // Second shell: 12 Na at a/√2.
    const shell2 = neighboursOf(nacl, 0, 4.0).filter((n) => n.distance > 3);
    expect(shell2).toHaveLength(12);
    shell2.forEach((n) => expect(n.distance).toBeCloseTo(5.64 / Math.SQRT2, 10));
  });

  it('diamond: four neighbours at √3a/4 and the tetrahedral angle', () => {
    const nb = neighboursOf(si, 0, 2.5);
    expect(nb).toHaveLength(4);
    nb.forEach((n) => expect(n.distance).toBeCloseTo((Math.sqrt(3) * 5.431) / 4, 10));
    expect(angle(si, 1, 0, 3)).toBeCloseTo((Math.acos(-1 / 3) * 180) / Math.PI, 8);
  });

  it('finds neighbours that are periodic images of the atom itself', () => {
    const one: Structure = {
      id: 'sc',
      title: 'simple cubic',
      symbols: ['Po'],
      positions: [[0, 0, 0]],
      cell: [
        [3.35, 0, 0],
        [0, 3.35, 0],
        [0, 0, 3.35],
      ],
      pbc: [true, true, true],
    };
    const nb = neighboursOf(one, 0, 3.4);
    expect(nb).toHaveLength(6);
    nb.forEach((n) => expect(n.j).toBe(0));
  });
});

describe('pair list and bonds', () => {
  it('agrees with neighboursOf for every atom (cell list vs brute force)', () => {
    const cutoff = 4.2;
    const pairs = pairList(nacl, cutoff, () => cutoff);
    const count = new Array(nacl.positions.length).fill(0);
    for (const p of pairs) {
      count[p.i]++;
      count[p.j]++;
    }
    nacl.positions.forEach((_, i) => expect(count[i]).toBe(neighboursOf(nacl, i, cutoff).length));
  });

  it('is invariant to atoms stored outside the home cell', () => {
    const shifted: Structure = {
      ...si,
      positions: si.positions.map((p, k) => (k % 3 === 0 ? [p[0] + 5.431 * 2, p[1] - 5.431, p[2]] : p)),
    };
    expect(displayBonds(shifted)).toHaveLength(displayBonds(si).length);
    for (const b of displayBonds(shifted)) {
      const p = shifted.positions;
      const pj = vecMat(b.image, shifted.cell!).map((v, k) => v + p[b.j]![k]!) as Vec3;
      const d = norm(pj.map((v, k) => v - p[b.i]![k]!) as Vec3);
      expect(d).toBeCloseTo(b.distance, 10);
    }
  });

  it('bonds Si to 4 neighbours (16 bonds per 8-atom cell) and NaCl to 6 (24 bonds)', () => {
    expect(displayBonds(si)).toHaveLength(16);
    expect(displayBonds(nacl)).toHaveLength(24);
  });

  it('bonds a water molecule O–H but not H–H', () => {
    const w = water();
    const b = displayBonds(w);
    expect(b).toHaveLength(2);
    expect(distance(w, 0, 1)).toBeCloseTo(0.9572, 10);
    expect(angle(w, 1, 0, 2)).toBeCloseTo(104.52, 8);
  });
});

describe('display sets', () => {
  it('replicates cells', () => {
    expect(displayAtoms(nacl, [2, 2, 2])).toHaveLength(64);
  });

  it('adds boundary images to complete a conventional cell', () => {
    // FCC Na: 8 corners + 6 faces = 14 sites; Cl: 12 edges + 1 centre = 13.
    const atoms = displayAtoms(nacl, [1, 1, 1], true);
    expect(atoms.filter((a) => nacl.symbols[a.index] === 'Na')).toHaveLength(14);
    expect(atoms.filter((a) => nacl.symbols[a.index] === 'Cl')).toHaveLength(13);
  });

  it('draws only bonds whose both ends are displayed, each once', () => {
    const atoms = displayAtoms(si, [1, 1, 1], false);
    const pairs = displayBondPairs(atoms, displayBonds(si));
    const keys = new Set(pairs.map(([a, b]) => (a < b ? `${a}-${b}` : `${b}-${a}`)));
    expect(keys.size).toBe(pairs.length);
    // Inside one conventional cell of diamond, 4 of the 8 atoms have all 4 bonds internal.
    pairs.forEach(([a, b]) => {
      const d = norm(atoms[a]!.position.map((v, k) => v - atoms[b]!.position[k]!) as Vec3);
      expect(d).toBeCloseTo((Math.sqrt(3) * 5.431) / 4, 10);
    });
  });

  it('molecules display as-is', () => {
    expect(displayAtoms(water(), [3, 3, 3])).toHaveLength(3);
  });
});

describe('dihedral', () => {
  it('is 180° for a planar trans chain and ±90° for a perpendicular one', () => {
    const s: Structure = {
      id: 'c',
      title: 'chain',
      symbols: ['C', 'C', 'C', 'C'],
      positions: [
        [0, 1, 0],
        [0, 0, 0],
        [1.5, 0, 0],
        [1.5, -1, 0],
      ],
      cell: null,
      pbc: [false, false, false],
    };
    expect(Math.abs(dihedral(s, 0, 1, 2, 3))).toBeCloseTo(180, 8);
    s.positions[3] = [1.5, 0, 1];
    expect(Math.abs(dihedral(s, 0, 1, 2, 3))).toBeCloseTo(90, 8);
  });
});
