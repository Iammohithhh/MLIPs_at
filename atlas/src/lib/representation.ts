// Representation Map data (precomputed by pipeline/atlas_pipeline/representation.py) and the small pieces of
// logic the island needs: colour classes, neighbour overlaps, bounds. Pure and tested in representation.test.ts.

export type Vec3 = [number, number, number];

export interface MapStructure {
  id: string;
  title: string;
  kind: 'crystal' | 'surface' | 'defect' | 'molecule';
  variant: string;
  symbols: string[];
  positions: Vec3[];
  cell: [Vec3, Vec3, Vec3] | null;
}
export interface MapEnv {
  /** Structure index, atom index in it, element, coordination number, number of identical copies. */
  s: number;
  i: number;
  el: string;
  cn: number;
  mult: number;
}
export interface MapProjection {
  xy: [number, number][];
  var: [number, number];
  procrustes?: number;
}
export interface MapModel {
  id: string;
  name: string;
  checkpoint: string | null;
  trained: boolean;
  raw: MapProjection;
  /** Ten nearest neighbours of each environment in the model's raw feature space. */
  nn: number[][];
  anchored: Record<string, MapProjection>;
}
export interface MapPair {
  a: string;
  b: string;
  cka: number;
  mknn_raw: number;
  mknn_anchored: Record<string, number>;
}
export interface RepresentationMap {
  source: Record<string, unknown> & { k_nn: number; checkpoints: Record<string, { name: string; sha256: string }> };
  structures: MapStructure[];
  environments: MapEnv[];
  anchors: Record<string, number[]>;
  models: MapModel[];
  pairs: MapPair[];
}

export type ElementClass = 'alkali' | 'alkaline-earth' | 'transition' | 'p-metal' | 'metalloid' | 'nonmetal' | 'halogen';

const CLASSES: Record<ElementClass, string[]> = {
  alkali: ['Li', 'Na', 'K', 'Rb', 'Cs'],
  'alkaline-earth': ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
  transition: ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
  'p-metal': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi'],
  metalloid: ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
  nonmetal: ['H', 'C', 'N', 'O', 'P', 'S', 'Se'],
  halogen: ['F', 'Cl', 'Br', 'I'],
};
export const CLASS_LABELS: Record<ElementClass, string> = {
  alkali: 'alkali',
  'alkaline-earth': 'alkaline earth',
  transition: 'transition metal',
  'p-metal': 'post-transition metal',
  metalloid: 'metalloid',
  nonmetal: 'non-metal',
  halogen: 'halogen',
};

export function elementClass(el: string): ElementClass | null {
  for (const [k, v] of Object.entries(CLASSES)) if (v.includes(el)) return k as ElementClass;
  return null;
}

/** Fraction of environment e's k nearest neighbours that models a and b share. */
export function nnOverlap(a: MapModel, b: MapModel, e: number): number {
  const A = new Set(a.nn[e]);
  const shared = b.nn[e]!.filter((x) => A.has(x)).length;
  return shared / a.nn[e]!.length;
}

export function projection(m: MapModel, space: string): MapProjection {
  return space === 'raw' ? m.raw : m.anchored[space]!;
}

export function bounds(xy: [number, number][]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [x, y] of xy) {
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  return { x0, x1, y0, y1 };
}

export function pairOf(map: RepresentationMap, a: string, b: string): MapPair | undefined {
  return map.pairs.find((p) => (p.a === a && p.b === b) || (p.a === b && p.b === a));
}

/** Total number of atoms the environments stand for (environments × multiplicity). */
export const atomCount = (map: RepresentationMap) => map.environments.reduce((t, e) => t + e.mult, 0);
