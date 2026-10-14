// Periodic-cell geometry for the viewer and the Physics Lab: fractional ↔ Cartesian
// conversion, minimum-image vectors, neighbour lists with periodic images,
// display bonds, replicas, angles and dihedrals. Row-vector convention: a point
// with fractional coordinates f sits at r = f · M, where M's rows are a, b, c.

import { covalentRadius } from './elements';
import type { Mat3, Structure, Vec3 } from './types';

// ----------------------------------------------------------------------------
// Small vector helpers
// ----------------------------------------------------------------------------
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a: Vec3) => Math.sqrt(dot(a, a));

export function det3(m: Mat3): number {
  return dot(m[0], cross(m[1], m[2]));
}

export function inv3(m: Mat3): Mat3 {
  const d = det3(m);
  if (Math.abs(d) < 1e-12) throw new Error('Singular cell matrix');
  // Rows of M⁻¹ are the columns of the adjugate / det; for row-vector convention
  // f = r · M⁻¹, the columns of M⁻¹ are (b×c, c×a, a×b)/V.
  const c0 = scale(cross(m[1], m[2]), 1 / d);
  const c1 = scale(cross(m[2], m[0]), 1 / d);
  const c2 = scale(cross(m[0], m[1]), 1 / d);
  return [
    [c0[0], c1[0], c2[0]],
    [c0[1], c1[1], c2[1]],
    [c0[2], c1[2], c2[2]],
  ];
}

/** v · M (row vector times matrix). */
export function vecMat(v: Vec3, m: Mat3): Vec3 {
  return [
    v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
    v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
    v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
  ];
}

export const cartToFrac = (r: Vec3, cell: Mat3): Vec3 => vecMat(r, inv3(cell));
export const fracToCart = (f: Vec3, cell: Mat3): Vec3 => vecMat(f, cell);

export function cellVolume(cell: Mat3): number {
  return Math.abs(det3(cell));
}

/** Distance between opposite faces of the cell, for each lattice direction. */
export function planeSpacings(cell: Mat3): Vec3 {
  const v = cellVolume(cell);
  return [v / norm(cross(cell[1], cell[2])), v / norm(cross(cell[2], cell[0])), v / norm(cross(cell[0], cell[1]))];
}

/** Lattice lengths a, b, c (Å) and angles α, β, γ (degrees). */
export function cellParameters(cell: Mat3): { lengths: Vec3; angles: Vec3 } {
  const [a, b, c] = cell;
  const ang = (u: Vec3, w: Vec3) => (Math.acos(dot(u, w) / (norm(u) * norm(w))) * 180) / Math.PI;
  return { lengths: [norm(a), norm(b), norm(c)], angles: [ang(b, c), ang(a, c), ang(a, b)] };
}

// ----------------------------------------------------------------------------
// Minimum image
// ----------------------------------------------------------------------------

/**
 * Shortest periodic image of the displacement d. Wraps fractional components to
 * [-½, ½), then checks the 26 neighbouring images, which is exact unless the cell
 * is extremely skewed (then reduce it first, e.g. with ase.geometry.minkowski_reduce).
 */
export function minimumImage(d: Vec3, cell: Mat3 | null, pbc: [boolean, boolean, boolean]): { d: Vec3; image: Vec3 } {
  if (!cell || !pbc.some(Boolean)) return { d, image: [0, 0, 0] };
  const f = cartToFrac(d, cell);
  const base: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) if (pbc[k]) base[k] = -Math.round(f[k]!);
  let best = add(d, vecMat(base, cell));
  let bestImage = base;
  let bestLen = dot(best, best);
  const range = (k: number) => (pbc[k] ? [-1, 0, 1] : [0]);
  for (const i of range(0))
    for (const j of range(1))
      for (const k of range(2)) {
        if (i === 0 && j === 0 && k === 0) continue;
        const n: Vec3 = [base[0] + i, base[1] + j, base[2] + k];
        const cand = add(d, vecMat(n, cell));
        const l = dot(cand, cand);
        if (l < bestLen - 1e-12) {
          best = cand;
          bestLen = l;
          bestImage = n;
        }
      }
  return { d: best, image: bestImage };
}

export function displacement(s: Structure, i: number, j: number, mic = true): Vec3 {
  const d = sub(s.positions[j]!, s.positions[i]!);
  return mic ? minimumImage(d, s.cell, s.pbc).d : d;
}

export function distance(s: Structure, i: number, j: number, mic = true): number {
  return norm(displacement(s, i, j, mic));
}

/** Angle i–j–k at vertex j, in degrees (minimum-image vectors). */
export function angle(s: Structure, i: number, j: number, k: number): number {
  const u = displacement(s, j, i);
  const w = displacement(s, j, k);
  const c = dot(u, w) / (norm(u) * norm(w));
  return (Math.acos(Math.min(1, Math.max(-1, c))) * 180) / Math.PI;
}

/** Dihedral i–j–k–l in degrees, in (−180, 180] (IUPAC sign convention). */
export function dihedral(s: Structure, i: number, j: number, k: number, l: number): number {
  const b0 = scale(displacement(s, j, i), -1);
  const b1 = displacement(s, j, k);
  const b2 = displacement(s, k, l);
  const n1 = cross(b0, b1);
  const n2 = cross(b1, b2);
  const m1 = cross(n1, scale(b1, 1 / norm(b1)));
  const x = dot(n1, n2);
  const y = dot(m1, n2);
  return (Math.atan2(y, x) * 180) / Math.PI;
}

// ----------------------------------------------------------------------------
// Neighbours with periodic images
// ----------------------------------------------------------------------------

export interface Neighbour {
  j: number;
  /** Lattice translation applied to atom j's stored position. */
  image: Vec3;
  distance: number;
  /** Cartesian position of that image of j. */
  position: Vec3;
}

function imageRange(s: Structure, cutoff: number): Vec3 {
  if (!s.cell) return [0, 0, 0];
  const sp = planeSpacings(s.cell);
  return [0, 1, 2].map((k) => (s.pbc[k] ? Math.ceil(cutoff / sp[k]!) + 1 : 0)) as Vec3;
}

/** Every image of every atom within `cutoff` of atom i (excluding i itself). */
export function neighboursOf(s: Structure, i: number, cutoff: number): Neighbour[] {
  const out: Neighbour[] = [];
  const ri = s.positions[i]!;
  const [na, nb, nc] = imageRange(s, cutoff);
  for (let j = 0; j < s.positions.length; j++) {
    const rj = s.positions[j]!;
    for (let a = -na; a <= na; a++)
      for (let b = -nb; b <= nb; b++)
        for (let c = -nc; c <= nc; c++) {
          if (j === i && a === 0 && b === 0 && c === 0) continue;
          const image: Vec3 = [a, b, c];
          const p = s.cell ? add(rj, vecMat(image, s.cell)) : rj;
          const dist = norm(sub(p, ri));
          if (dist < cutoff) out.push({ j, image, distance: dist, position: p });
        }
  }
  return out.sort((x, y) => x.distance - y.distance);
}

export interface Pair {
  i: number;
  j: number;
  /** Lattice translation of j relative to its stored position. */
  image: Vec3;
  distance: number;
}

const imageKey = (n: Vec3) => `${n[0]},${n[1]},${n[2]}`;
const lexPositive = (n: Vec3) => n[0] > 0 || (n[0] === 0 && (n[1] > 0 || (n[1] === 0 && n[2] > 0)));

/**
 * All pairs (each once) closer than pairCutoff(i, j) ≤ maxCutoff, including pairs
 * across periodic boundaries and between an atom and its own images. Uses a cell
 * list over the wrapped structure padded with the periodic images it needs.
 */
export function pairList(s: Structure, maxCutoff: number, pairCutoff: (i: number, j: number) => number): Pair[] {
  const n = s.positions.length;
  const cell = s.cell;
  const periodic = !!cell && s.pbc.some(Boolean);
  // Wrap into the home cell; remember the integer shift so images map back to stored positions.
  const home: Vec3[] = [];
  const shift: Vec3[] = [];
  const pad: Vec3 = [0, 0, 0];
  let inv: Mat3 | null = null;
  if (periodic && cell) {
    inv = inv3(cell);
    const sp = planeSpacings(cell);
    for (let k = 0; k < 3; k++) pad[k] = s.pbc[k] ? maxCutoff / sp[k]! + 1e-9 : 0;
  }
  for (const r of s.positions) {
    if (!periodic || !cell || !inv) {
      home.push(r);
      shift.push([0, 0, 0]);
      continue;
    }
    const f = vecMat(r, inv);
    const w: Vec3 = [0, 1, 2].map((k) => (s.pbc[k] ? Math.floor(f[k]!) : 0)) as Vec3;
    home.push(sub(r, vecMat(w, cell)));
    shift.push(w);
  }

  // Padded point set: (atom, image n′ relative to the wrapped position).
  const pts: Array<{ j: number; img: Vec3; p: Vec3 }> = [];
  const range = periodic ? imageRange(s, maxCutoff) : ([0, 0, 0] as Vec3);
  for (let j = 0; j < n; j++) {
    for (let a = -range[0]; a <= range[0]; a++)
      for (let b = -range[1]; b <= range[1]; b++)
        for (let c = -range[2]; c <= range[2]; c++) {
          const img: Vec3 = [a, b, c];
          if (!periodic || !cell || !inv) {
            pts.push({ j, img, p: home[j]! });
            continue;
          }
          const p = add(home[j]!, vecMat(img, cell));
          const f = vecMat(p, inv);
          let inside = true;
          for (let k = 0; k < 3; k++) {
            if (s.pbc[k] && (f[k]! < -pad[k]! || f[k]! > 1 + pad[k]!)) inside = false;
          }
          if (inside) pts.push({ j, img, p });
        }
  }

  // Cell list (hash grid) with bin size = maxCutoff.
  const h = Math.max(maxCutoff, 1e-6);
  const bins = new Map<string, number[]>();
  const binOf = (p: Vec3) => [Math.floor(p[0] / h), Math.floor(p[1] / h), Math.floor(p[2] / h)];
  pts.forEach((pt, idx) => {
    const k = binOf(pt.p).join(',');
    const list = bins.get(k);
    if (list) list.push(idx);
    else bins.set(k, [idx]);
  });

  const out: Pair[] = [];
  for (let i = 0; i < n; i++) {
    const pi = home[i]!;
    const [bx, by, bz] = binOf(pi);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const list = bins.get(`${bx! + dx},${by! + dy},${bz! + dz}`);
          if (!list) continue;
          for (const idx of list) {
            const { j, img, p } = pts[idx]!;
            const zero = img[0] === 0 && img[1] === 0 && img[2] === 0;
            if (j === i && zero) continue;
            // Keep each pair once: i < j, or a self-image with a lexicographically positive offset.
            const nOrig = add(img, sub(shift[i]!, shift[j]!));
            if (j < i || (j === i && !lexPositive(nOrig))) continue;
            const d = norm(sub(p, pi));
            if (d < pairCutoff(i, j)) out.push({ i, j, image: nOrig, distance: d });
          }
        }
  }
  return out;
}

export const BOND_TOLERANCE = 1.15;

/**
 * Display bonds: pairs closer than 1.15 × (r_i + r_j) with Cordero covalent radii.
 * A drawing convention, not a statement about chemical bonding.
 */
export function displayBonds(s: Structure, tolerance = BOND_TOLERANCE): Pair[] {
  const radii = s.symbols.map((sym) => covalentRadius(sym).r);
  const maxR = Math.max(...radii, 0);
  return pairList(s, 2 * maxR * tolerance, (i, j) => (radii[i]! + radii[j]!) * tolerance).filter(
    (p) => p.distance > 0.1,
  );
}

// ----------------------------------------------------------------------------
// Display sets: replicas and boundary images
// ----------------------------------------------------------------------------

export interface DisplayAtom {
  /** Index into the source structure. */
  index: number;
  /** Lattice translation of the stored position. */
  image: Vec3;
  position: Vec3;
  /** True for extra images added only to complete the drawn cell (faces/edges/corners). */
  boundary: boolean;
}

/**
 * Atoms to draw for a `repeat` = [na, nb, nc] block of cells. With `boundary`,
 * atoms sitting on a cell face/edge/corner are also drawn on the opposite side,
 * so a conventional cell looks complete (as in crystallography figures).
 */
export function displayAtoms(s: Structure, repeat: Vec3 = [1, 1, 1], boundary = false, tol = 1e-3): DisplayAtom[] {
  const cell = s.cell;
  if (!cell || !s.pbc.some(Boolean)) {
    return s.positions.map((p, index) => ({ index, image: [0, 0, 0], position: p, boundary: false }));
  }
  const inv = inv3(cell);
  const reps = [0, 1, 2].map((k) => (s.pbc[k] ? Math.max(1, Math.round(repeat[k]!)) : 1)) as Vec3;
  const out: DisplayAtom[] = [];
  const seen = new Set<string>();
  s.positions.forEach((r, index) => {
    const f = vecMat(r, inv);
    const w: Vec3 = [0, 1, 2].map((k) => (s.pbc[k] ? Math.floor(f[k]! + tol) : 0)) as Vec3;
    const wrapped = sub(f, w);
    const extra = [0, 1, 2].map((k) => {
      const opts = [0];
      if (boundary && s.pbc[k] && Math.abs(wrapped[k]!) < tol) opts.push(reps[k]!);
      return opts;
    });
    for (let a = 0; a < reps[0]; a++)
      for (let b = 0; b < reps[1]; b++)
        for (let c = 0; c < reps[2]; c++)
          for (const ea of extra[0]!)
            for (const eb of extra[1]!)
              for (const ec of extra[2]!) {
                const cellIdx: Vec3 = [a + ea, b + eb, c + ec];
                if (cellIdx.some((v, k) => v > reps[k]!)) continue;
                const image = sub(cellIdx, w);
                const key = `${index}|${imageKey(image)}`;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({
                  index,
                  image,
                  position: add(r, vecMat(image, cell)),
                  boundary: ea !== 0 || eb !== 0 || ec !== 0,
                });
              }
  });
  return out;
}

/**
 * Bonds between drawn atoms, derived from the periodic bond list. Each stored pair
 * (i, j, n) is reached from its i side only, so every drawn bond appears once.
 */
export function displayBondPairs(atoms: DisplayAtom[], bonds: Pair[]): Array<[number, number]> {
  const lookup = new Map<string, number>();
  atoms.forEach((a, k) => lookup.set(`${a.index}|${imageKey(a.image)}`, k));
  const out: Array<[number, number]> = [];
  const byAtom = new Map<number, Pair[]>();
  for (const b of bonds) {
    (byAtom.get(b.i) ?? byAtom.set(b.i, []).get(b.i)!).push(b);
  }
  atoms.forEach((a, k) => {
    for (const b of byAtom.get(a.index) ?? []) {
      const partner = lookup.get(`${b.j}|${imageKey(add(a.image, b.image))}`);
      if (partner !== undefined) out.push([k, partner]);
    }
  });
  return out;
}
