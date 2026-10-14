// Small analytic structures for tests and demos. Lattice constants are
// illustrative inputs, not measured values.
import type { Mat3, Structure, Vec3 } from './types';

const cubic = (a: number): Mat3 => [
  [a, 0, 0],
  [0, a, 0],
  [0, 0, a],
];

const FCC: Vec3[] = [
  [0, 0, 0],
  [0, 0.5, 0.5],
  [0.5, 0, 0.5],
  [0.5, 0.5, 0],
];

export function rocksalt(a: number, cation = 'Na', anion = 'Cl'): Structure {
  const positions: Vec3[] = [];
  const symbols: string[] = [];
  for (const f of FCC) {
    positions.push([f[0] * a, f[1] * a, f[2] * a]);
    symbols.push(cation);
  }
  for (const f of FCC) {
    positions.push([(f[0] + 0.5) * a, f[1] * a, f[2] * a]);
    symbols.push(anion);
  }
  return { id: 'rocksalt', title: `${cation}${anion} rock salt`, symbols, positions, cell: cubic(a), pbc: [true, true, true] };
}

export function diamond(a: number, sym = 'Si'): Structure {
  const positions: Vec3[] = [];
  for (const f of FCC) {
    positions.push([f[0] * a, f[1] * a, f[2] * a]);
    positions.push([(f[0] + 0.25) * a, (f[1] + 0.25) * a, (f[2] + 0.25) * a]);
  }
  return {
    id: 'diamond',
    title: `${sym} diamond`,
    symbols: positions.map(() => sym),
    positions,
    cell: cubic(a),
    pbc: [true, true, true],
  };
}

/** Water monomer with O–H = 0.9572 Å and H–O–H = 104.52° (TIP3P-style geometry). */
export function water(): Structure {
  const r = 0.9572;
  const half = ((104.52 / 2) * Math.PI) / 180;
  return {
    id: 'water',
    title: 'Water molecule',
    symbols: ['O', 'H', 'H'],
    positions: [
      [0, 0, 0],
      [r * Math.sin(half), r * Math.cos(half), 0],
      [-r * Math.sin(half), r * Math.cos(half), 0],
    ],
    cell: null,
    pbc: [false, false, false],
  };
}
