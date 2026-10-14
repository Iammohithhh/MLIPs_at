// The Atlas structure format, shared by the Python pipeline (which writes it) and
// the viewer (which reads it). Lengths in Å; positions are Cartesian.

export type Vec3 = [number, number, number];
/** Lattice vectors as rows: [a, b, c]. */
export type Mat3 = [Vec3, Vec3, Vec3];

export interface VectorArray {
  /** One vector per atom, in `unit`. */
  values: Vec3[];
  unit: string;
  label: string;
  /** How the values were produced, e.g. TOY_MODEL, PRECOMPUTED. */
  status: 'TOY_MODEL' | 'LIVE_CALCULATION' | 'PRECOMPUTED' | 'PUBLISHED_RESULT' | 'REPRODUCED';
}

export interface StructureProvenance {
  /** Human-readable origin, e.g. "ase.build.bulk('NaCl', 'rocksalt', a=5.64)". */
  source: string;
  generator?: string;
  created?: string;
  notes?: string;
}

export interface Structure {
  format?: 'atlas-structure';
  version?: 1;
  id: string;
  title: string;
  symbols: string[];
  positions: Vec3[];
  /** null for molecules and clusters. */
  cell: Mat3 | null;
  pbc: [boolean, boolean, boolean];
  vectors?: Record<string, VectorArray>;
  provenance?: StructureProvenance;
  /** Drawing hints, e.g. a uniform sphere radius for reduced-unit toy systems. */
  display?: { radius?: number };
}
