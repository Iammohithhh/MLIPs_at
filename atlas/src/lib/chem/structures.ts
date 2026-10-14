// Deterministic toy structures for the 3D scenes. Geometry follows stated, cited
// conventions, but these are illustrations, not equilibrated simulations:
// positions are placed by rule (lattice + random sequential addition), never
// relaxed or sampled from a real Hamiltonian. Every scene says so in view.
//
// Conventions
// - Au: fcc, a = 4.078 Å (room-temperature experimental lattice constant).
// - Water: SPC/E rigid geometry (O–H 1.0 Å, H–O–H 109.47°) and SPC/E partial
//   charges q_O = −0.8476 e, q_H = +0.4238 e (Berendsen et al., J. Phys. Chem. 91,
//   6269, 1987), so geometry and charges come from one consistent model.
// - Ions: Na⁺ (+1 e), Cl⁻ (−1 e). Au atoms carry no charge in the toy (a real
//   metal screens charge through its electrons; image charges are not modelled).
// - NaCl rock salt: nearest-neighbour distance 2.82 Å (a = 5.64 Å).

import type { El } from './elements';
import { mulberry32 } from './rng';

export type Vec3 = [number, number, number];

export interface Atom {
  el: El;
  p: Vec3;
  /** Toy partial charge in units of e. */
  q: number;
  /** Molecule index for waters, −1 otherwise. */
  mol: number;
}

export interface Structure {
  atoms: Atom[];
  /** Orthorhombic cell lengths (Å). */
  cell: Vec3;
  pbc: [boolean, boolean, boolean];
  /** Intramolecular bonds (atom index pairs). */
  bonds: Array<[number, number]>;
}

export const AU_LATTICE = 4.078;
export const SPCE = { rOH: 1.0, angleDeg: 109.47, qO: -0.8476, qH: 0.4238 } as const;
export const NACL_R0 = 2.82;

/** Displacement b − a under the minimum-image convention on periodic axes. */
export function minImage(a: Vec3, b: Vec3, cell: Vec3, pbc: [boolean, boolean, boolean]): Vec3 {
  let dx = b[0] - a[0];
  let dy = b[1] - a[1];
  let dz = b[2] - a[2];
  if (pbc[0]) dx -= cell[0] * Math.round(dx / cell[0]);
  if (pbc[1]) dy -= cell[1] * Math.round(dy / cell[1]);
  if (pbc[2]) dz -= cell[2] * Math.round(dz / cell[2]);
  return [dx, dy, dz];
}

export function dist(a: Vec3, b: Vec3, cell: Vec3, pbc: [boolean, boolean, boolean]): number {
  const d = minImage(a, b, cell, pbc);
  return Math.hypot(d[0], d[1], d[2]);
}

const norm = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** Hydrogen positions of an SPC/E water with oxygen at `o` and dipole along unit `u`, twisted by `twist`. */
export function waterHydrogens(o: Vec3, u: Vec3, twist: number): [Vec3, Vec3] {
  const e2 = norm(u);
  const helper: Vec3 = Math.abs(e2[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const a = norm(cross(e2, helper));
  const b = cross(e2, a);
  const e1: Vec3 = [
    Math.cos(twist) * a[0] + Math.sin(twist) * b[0],
    Math.cos(twist) * a[1] + Math.sin(twist) * b[1],
    Math.cos(twist) * a[2] + Math.sin(twist) * b[2],
  ];
  const half = ((SPCE.angleDeg / 2) * Math.PI) / 180;
  const s = Math.sin(half) * SPCE.rOH;
  const c = Math.cos(half) * SPCE.rOH;
  const h = (sign: number): Vec3 => [
    o[0] + sign * s * e1[0] + c * e2[0],
    o[1] + sign * s * e1[1] + c * e2[1],
    o[2] + sign * s * e1[2] + c * e2[2],
  ];
  return [h(1), h(-1)];
}

export interface InterfaceOptions {
  nx?: number;
  ny?: number;
  layers?: number;
  /** Thickness of the water film (Å). */
  waterHeight?: number;
  seed?: number;
}

export interface InterfaceInfo {
  structure: Structure;
  /** z of the top Au layer (Å). */
  zTop: number;
  naIndex: number;
  clIndex: number;
  waterCount: number;
  /** Indices of the first oxygen of each water, in molecule order. */
  oxygens: number[];
}

/**
 * Au(111) slab under a film of water with one Na⁺ and one Cl⁻ (net charge zero).
 * Periodic in x and y (rectangular fcc(111) supercell), open along z.
 */
export function auWaterInterface(opts: InterfaceOptions = {}): InterfaceInfo {
  const { nx = 8, ny = 5, layers = 4, waterHeight = 13, seed = 20261014 } = opts;
  const rand = mulberry32(seed);
  const as = AU_LATTICE / Math.SQRT2; // surface nearest-neighbour distance
  const rowH = (as * Math.sqrt(3)) / 2;
  const dz = AU_LATTICE / Math.sqrt(3); // (111) interlayer spacing
  const Lx = nx * as;
  const Ly = 2 * ny * rowH;
  const atoms: Atom[] = [];
  const bonds: Array<[number, number]> = [];

  // ABC stacking: B and C sites are shifted by (a1 + a2)/3 and 2(a1 + a2)/3.
  const shifts: Array<[number, number]> = [
    [0, 0],
    [as / 2, (as * Math.sqrt(3)) / 6],
    [as, (as * Math.sqrt(3)) / 3],
  ];
  for (let k = 0; k < layers; k++) {
    const [ox, oy] = shifts[k % 3]!;
    for (let j = 0; j < 2 * ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x = (((i * as + (j % 2) * (as / 2) + ox) % Lx) + Lx) % Lx;
        const y = (((j * rowH + oy) % Ly) + Ly) % Ly;
        atoms.push({ el: 'Au', p: [x, y, k * dz], q: 0, mol: -1 });
      }
    }
  }
  const zTop = (layers - 1) * dz;
  const zLo = zTop + 2.9;
  const zHi = zLo + waterHeight;
  const cell: Vec3 = [Lx, Ly, zHi + 12];
  const pbc: [boolean, boolean, boolean] = [true, true, false];

  const naIndex = atoms.length;
  const na: Vec3 = [Lx * 0.5, Ly * 0.5, zTop + 6.6];
  atoms.push({ el: 'Na', p: na, q: 1, mol: -1 });
  const clIndex = atoms.length;
  const cl: Vec3 = [Lx * 0.08, Ly * 0.18, zTop + 11.4];
  atoms.push({ el: 'Cl', p: cl, q: -1, mol: -1 });

  // Random sequential addition of oxygens at liquid-water number density.
  const target = Math.round(0.0334 * Lx * Ly * waterHeight);
  const oxy: Vec3[] = [];
  const tooClose = (p: Vec3) => {
    if (dist(p, na, cell, pbc) < 2.4 || dist(p, cl, cell, pbc) < 3.0) return true;
    for (const o of oxy) if (dist(p, o, cell, pbc) < 2.6) return true;
    return false;
  };
  for (let attempt = 0; attempt < 120000 && oxy.length < target; attempt++) {
    const p: Vec3 = [rand() * Lx, rand() * Ly, zLo + rand() * waterHeight];
    if (!tooClose(p)) oxy.push(p);
  }

  // Orient: dipole away from Na⁺, towards Cl⁻ (H points at the anion), nearly flat
  // in the first layer above the metal (illustrative), random elsewhere.
  const placedH: Vec3[] = [];
  const oxygens: number[] = [];
  let mol = 0;
  for (const o of oxy) {
    const dNa = dist(o, na, cell, pbc);
    const dCl = dist(o, cl, cell, pbc);
    let preferred: Vec3 | null = null;
    if (dNa < 3.4) preferred = norm(minImage(na, o, cell, pbc));
    else if (dCl < 3.9) preferred = norm(minImage(o, cl, cell, pbc));
    else if (o[2] - zTop < 3.6) {
      const phi = rand() * 2 * Math.PI;
      preferred = norm([Math.cos(phi), Math.sin(phi), (rand() - 0.5) * 0.5]);
    }
    let hs: [Vec3, Vec3] | null = null;
    for (let tries = 0; tries < 40 && !hs; tries++) {
      const u: Vec3 =
        preferred && tries < 25
          ? preferred
          : norm([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);
      const cand = waterHydrogens(o, u, rand() * 2 * Math.PI);
      const ok = cand.every((h) => {
        if (h[2] < zTop + 2.0) return false;
        if (dist(h, na, cell, pbc) < 2.2) return false;
        for (const p of oxy) if (p !== o && dist(h, p, cell, pbc) < 1.55) return false;
        for (const p of placedH) if (dist(h, p, cell, pbc) < 1.45) return false;
        return true;
      });
      if (ok) hs = cand;
    }
    if (!hs) continue; // drop a molecule that cannot be oriented without a clash
    const iO = atoms.length;
    atoms.push({ el: 'O', p: o, q: SPCE.qO, mol });
    atoms.push({ el: 'H', p: hs[0], q: SPCE.qH, mol });
    atoms.push({ el: 'H', p: hs[1], q: SPCE.qH, mol });
    bonds.push([iO, iO + 1], [iO, iO + 2]);
    placedH.push(hs[0], hs[1]);
    oxygens.push(iO);
    mol++;
  }

  return { structure: { atoms, cell, pbc, bonds }, zTop, naIndex, clIndex, waterCount: mol, oxygens };
}

/**
 * Rock-salt NaCl cluster with integer coordinates |nₓ|,|n_y|,|n_z| ≤ n (units of
 * the nearest-neighbour distance r0), Na⁺ at the origin. Charge (−1)^(nₓ+n_y+n_z).
 */
export function rockSaltCluster(n: number, r0 = NACL_R0): Structure {
  const atoms: Atom[] = [];
  for (let i = -n; i <= n; i++)
    for (let j = -n; j <= n; j++)
      for (let k = -n; k <= n; k++) {
        const q = (i + j + k) % 2 === 0 ? 1 : -1;
        atoms.push({ el: q > 0 ? 'Na' : 'Cl', p: [i * r0, j * r0, k * r0], q, mol: -1 });
      }
  const L = (2 * n + 1) * r0;
  return { atoms, cell: [L, L, L], pbc: [false, false, false], bonds: [] };
}
