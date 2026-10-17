// Per-experiment display facts for the instrument, so captions, legends and walkthrough text stop
// assuming silicon. Anything that is a property of the evaluated run (does the baseline carry forces?
// do the layer-1 vector features vanish?) is *derived from the manifest*, never written by hand; only
// presentation choices (display colours, the bond-stick criterion, the structure's prose description
// and its source) are fixed here. Element colours follow the muted CPK/Jmol-derived palette of
// atlas/src/lib/chem/elements.ts (O copied from there; Si as already used by the lab; Zn added in the same
// style). Amber is reserved for changes and mint for the moved atom, so no element uses either.

import { baseline, norm, type Manifest } from './experiment';

export interface ElementDisplay {
  name: string;
  /** sRGB hex display colour (not a computed quantity). */
  color: string;
}

export const ELEMENT_DISPLAY: Record<string, ElementDisplay> = {
  Si: { name: 'Silicon', color: '#a9b4b8' },
  Zn: { name: 'Zinc', color: '#8f93bd' },
  O: { name: 'Oxygen', color: '#c96a5f' },
};

export interface ExperimentInfo {
  /** Prose description of the structure, e.g. for the provenance list and the exported figure. */
  structure: string;
  /** Where the geometry comes from and whether it was relaxed with the model. */
  geometrySource: string;
  /** Chemical-bond sticks are drawn between atoms closer than this (Å); a display criterion only. */
  bondCutoff: number;
  bondLabel: string;
  /** Why this structure is inside the checkpoint's training domain. */
  domain: string;
}

const INFO: Record<string, ExperimentInfo> = {
  'si-bond-stretch': {
    structure: 'Si diamond',
    geometrySource: "ase.build.bulk('Si', 'diamond', a=5.431, cubic=True) from atlas/src/data/structures/si-diamond.json; every atom sits on a site of tetrahedral (Td) symmetry, so every baseline force vanishes by symmetry (the lattice constant was not optimised with this model)",
    bondCutoff: 2.6, // Si–Si nearest neighbour 2.35 Å, second shell 3.84 Å
    bondLabel: 'Si–Si bonds < 2.6 Å',
    domain: 'MPtrj (Materials Project PBE/PBE+U relaxations) contains crystalline silicon; small displacements of a bulk Si atom are inside that domain.',
  },
  'zno-axial-bond': {
    structure: 'ZnO wurtzite',
    geometrySource: "ase.build.bulk('ZnO', 'wurtzite', a=3.25, c=5.21, u=0.382) from atlas/src/data/structures/zno-wurtzite.json; a, c and u are inputs and the cell was not relaxed with this model",
    bondCutoff: 2.3, // Zn–O 1.975 (basal) and 1.990 Å (axial); Zn–Zn and O–O ≥ 3.2 Å
    bondLabel: 'Zn–O bonds < 2.3 Å',
    domain: 'MPtrj is built from relaxation trajectories of Materials Project structures, which include wurtzite ZnO (mp-2133); small displacements of a bulk Zn atom stay close to that domain. The run is not compared with DFT, so this is a domain statement, not an accuracy claim.',
  },
};

export function experimentInfo(id: string): ExperimentInfo | null {
  return INFO[id] ?? null;
}

/** Below this, a stored force component is indistinguishable from zero (storage precision 1e-7 eV/Å). */
export const FORCE_ZERO = 1e-6;

export interface BaselineFacts {
  /** Largest baseline force magnitude over all atoms (eV/Å). */
  maxForce: number;
  /** True when every baseline force is zero to storage precision (symmetry or a relaxed minimum). */
  forceFree: boolean;
  /** Largest layer-1 l = 1 block norm at the baseline over all atoms, or null if the layer has no l = 1 block. */
  maxVectorNorm: number | null;
  /** True when every layer-1 l = 1 feature is zero at the baseline (written as 0 below 1e-12). */
  vectorsVanish: boolean | null;
}

/** Facts about the baseline that captions may state, computed from the stored run. */
export function baselineFacts(m: Manifest): BaselineFacts {
  const b = baseline(m);
  const maxForce = Math.max(...b.forces.map((F) => norm(F)));
  const h1 = m.baseline.norms[0]?.h1 ?? null;
  const maxVectorNorm = h1 ? Math.max(...h1) : null;
  return { maxForce, forceFree: maxForce < FORCE_ZERO, maxVectorNorm, vectorsVanish: maxVectorNorm === null ? null : maxVectorNorm === 0 };
}

/** Species present, with counts, in first-appearance order (e.g. "Zn 150 · O 150"). */
export function composition(m: Manifest): { symbol: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const s of m.structure.symbols) counts.set(s, (counts.get(s) ?? 0) + 1);
  return [...counts].map(([symbol, count]) => ({ symbol, count }));
}
