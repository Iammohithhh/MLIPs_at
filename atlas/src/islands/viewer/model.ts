// Pure display model for the atomic viewer: from a Structure and the viewer's
// options, decide what to draw. No rendering code here, so it is unit-tested.

import { covalentRadius } from '~/lib/structure/elements';
import {
  add,
  displayAtoms,
  displayBondPairs,
  displayBonds,
  neighboursOf,
  norm,
  sub,
  dot,
  cross,
  type DisplayAtom,
  type Pair,
} from '~/lib/structure/geometry';
import type { Structure, Vec3 } from '~/lib/structure/types';

export type DisplayMode = 'ball-and-stick' | 'space-filling';

export interface ViewerOptions {
  mode: DisplayMode;
  repeat: Vec3;
  boundary: boolean;
  bonds: boolean;
  /** Cutoff-shell radius in Å around the first selected atom; null hides the shell. */
  shell: number | null;
}

export const DEFAULT_OPTIONS: ViewerOptions = {
  mode: 'ball-and-stick',
  repeat: [1, 1, 1],
  boundary: true,
  bonds: true,
  shell: null,
};

export interface DrawnAtom extends DisplayAtom {
  symbol: string;
  radius: number;
  /** Only drawn because it lies inside the cutoff shell (an image outside the drawn cells). */
  ghost: boolean;
  /** Inside the cutoff shell of the selected atom. */
  inShell: boolean;
  /** Radius is a fallback because the element has no tabulated covalent radius. */
  untabulated: boolean;
}

export interface DisplayModel {
  atoms: DrawnAtom[];
  bonds: Array<[number, number]>;
  /** Index (into atoms) of the shell centre, when a shell is shown. */
  shellCenter: number | null;
  shellCount: number;
  /** Bounding sphere of what is drawn, for camera framing. */
  center: Vec3;
  extent: number;
}

/** Ball-and-stick: 0.45 × covalent radius (min 0.22 Å); space-filling: covalent radius. */
export function drawRadius(symbol: string, mode: DisplayMode, override?: number): { r: number; tabulated: boolean } {
  const { r, tabulated } = covalentRadius(symbol);
  if (override !== undefined) return { r: mode === 'space-filling' ? override : override * 0.6, tabulated: true };
  return { r: mode === 'space-filling' ? r : Math.max(0.22, 0.45 * r), tabulated };
}

const key = (index: number, image: Vec3) => `${index}|${image[0]},${image[1]},${image[2]}`;

export function buildModel(
  s: Structure,
  opts: ViewerOptions,
  selected: number[] = [],
  bondList: Pair[] | null = null,
): DisplayModel {
  const base = displayAtoms(s, opts.repeat, opts.boundary);
  const radiusOverride = s.display?.radius;
  const atoms: DrawnAtom[] = base.map((a) => {
    const { r, tabulated } = drawRadius(s.symbols[a.index]!, opts.mode, radiusOverride);
    return { ...a, symbol: s.symbols[a.index]!, radius: r, ghost: false, inShell: false, untabulated: !tabulated };
  });

  let shellCenter: number | null = null;
  let shellCount = 0;
  const first = selected[0];
  if (opts.shell !== null && first !== undefined && atoms[first] && !atoms[first].ghost) {
    shellCenter = first;
    const c = atoms[first]!;
    const lookup = new Map(atoms.map((a, k) => [key(a.index, a.image), k]));
    for (const nb of neighboursOf(s, c.index, opts.shell)) {
      const image = add(c.image, nb.image);
      const k = lookup.get(key(nb.j, image));
      shellCount++;
      if (k !== undefined) {
        atoms[k]!.inShell = true;
      } else {
        const { r, tabulated } = drawRadius(s.symbols[nb.j]!, opts.mode, radiusOverride);
        atoms.push({
          index: nb.j,
          image,
          position: add(nb.position, sub(c.position, s.positions[c.index]!)),
          boundary: false,
          symbol: s.symbols[nb.j]!,
          radius: r,
          ghost: true,
          inShell: true,
          untabulated: !tabulated,
        });
      }
    }
  }

  const bonds = opts.bonds ? displayBondPairs(atoms, bondList ?? displayBonds(s)) : [];

  // Bounding sphere (centroid + max distance) of the drawn atoms.
  const n = Math.max(atoms.length, 1);
  const center = atoms.reduce<Vec3>((acc, a) => add(acc, a.position), [0, 0, 0]).map((v) => v / n) as Vec3;
  const extent = atoms.reduce((m, a) => Math.max(m, norm(sub(a.position, center)) + a.radius), 1);
  return { atoms, bonds, shellCenter, shellCount, center, extent };
}

// ----------------------------------------------------------------------------
// Measurements between drawn atoms (explicit images, so no minimum-image guessing)
// ----------------------------------------------------------------------------

export type Measurement =
  | { kind: 'distance'; atoms: [number, number]; value: number; unit: 'Å' }
  | { kind: 'angle'; atoms: [number, number, number]; value: number; unit: '°' }
  | { kind: 'dihedral'; atoms: [number, number, number, number]; value: number; unit: '°' };

const deg = (rad: number) => (rad * 180) / Math.PI;

export function measure(atoms: DrawnAtom[], picked: number[]): Measurement | null {
  const p = picked.map((k) => atoms[k]?.position).filter((x): x is Vec3 => !!x);
  if (p.length !== picked.length) return null;
  if (p.length === 2) {
    return { kind: 'distance', atoms: [picked[0]!, picked[1]!], value: norm(sub(p[1]!, p[0]!)), unit: 'Å' };
  }
  if (p.length === 3) {
    const u = sub(p[0]!, p[1]!);
    const w = sub(p[2]!, p[1]!);
    const c = Math.min(1, Math.max(-1, dot(u, w) / (norm(u) * norm(w))));
    return { kind: 'angle', atoms: [picked[0]!, picked[1]!, picked[2]!], value: deg(Math.acos(c)), unit: '°' };
  }
  if (p.length === 4) {
    // Same IUPAC convention as dihedral() in geometry.ts.
    const b0 = sub(p[1]!, p[0]!);
    const b1 = sub(p[2]!, p[1]!);
    const b2 = sub(p[3]!, p[2]!);
    const n1 = cross(b0, b1);
    const n2 = cross(b1, b2);
    const m1 = cross(n1, b1.map((v) => v / norm(b1)) as Vec3);
    return {
      kind: 'dihedral',
      atoms: [picked[0]!, picked[1]!, picked[2]!, picked[3]!],
      value: deg(Math.atan2(dot(m1, n2), dot(n1, n2))),
      unit: '°',
    };
  }
  return null;
}

/** Ray–sphere picking: index of the nearest drawn atom hit by the ray, or null. */
export function pickAtom(atoms: DrawnAtom[], origin: Vec3, dir: Vec3): number | null {
  let best: number | null = null;
  let bestT = Infinity;
  atoms.forEach((a, k) => {
    const oc = sub(origin, a.position);
    const b = dot(oc, dir);
    const c = dot(oc, oc) - a.radius * a.radius;
    const h = b * b - c;
    if (h < 0) return;
    const t = -b - Math.sqrt(h);
    if (t > 0 && t < bestT) {
      bestT = t;
      best = k;
    }
  });
  return best;
}
