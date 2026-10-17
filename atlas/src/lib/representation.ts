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
export interface InvarianceSummary {
  n: number;
  median: number;
  p95: number;
  max: number;
}
/** Feature displacement of environments that should not move, in units of d_ref (the median distance from an
 * environment to its nearest distinct one in the same model's raw space). */
export interface MapInvariance {
  d_ref: number;
  equiv: InvarianceSummary;
  rot: InvarianceSummary;
  ref: InvarianceSummary;
  /** Per environment: largest displacement among its symmetry-equivalent atoms (0 when it has one atom). */
  equiv_env: number[];
  /** Per environment: displacement of its first atom after a rigid rotation of the whole structure. */
  rot_env: number[];
  /** Per environment: displacement of its first atom after a mirror reflection (z → −z) of the structure. */
  ref_env: number[];
}
export interface MapModel {
  id: string;
  name: string;
  family: 'mace' | 'chgnet' | 'sevennet' | 'orb';
  checkpoint: string | null;
  trained: boolean;
  data: string | null;
  feature: string;
  dim: number;
  invariance: MapInvariance;
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
  source: Record<string, unknown> & {
    k_nn: number;
    checkpoints: Record<string, { name: string | null; sha256: string | null; package: string }>;
    rotation: { axis: number[]; angle_deg: number };
    reflection: string;
  };
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

export type PairMetric = 'cka' | 'mknn_raw' | `mknn_${string}`;

/** Symmetric model × model matrix of one agreement metric (1 on the diagonal), in map.models order. */
export function pairMatrix(map: RepresentationMap, metric: PairMetric): number[][] {
  const ids = map.models.map((m) => m.id);
  return ids.map((a) =>
    ids.map((b) => {
      if (a === b) return 1;
      const p = pairOf(map, a, b)!;
      if (metric === 'cka') return p.cka;
      if (metric === 'mknn_raw') return p.mknn_raw;
      return p.mknn_anchored[metric.slice(5)]!;
    }),
  );
}

/** Position of a non-negative value on a log axis from 10^lo to 10^hi (0 and anything below 10^lo sit at 0). */
export function logPosition(v: number, lo: number, hi: number): number {
  if (!(v > 0)) return 0;
  return Math.min(1, Math.max(0, (Math.log10(v) - lo) / (hi - lo)));
}
