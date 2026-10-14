// Data for the homepage 3D figure "Which atoms can influence atom i?" on an
// illustrative Au(111)/water/NaCl interface. Pure functions (no WebGL), so the
// counts the figure reports are unit-tested.

import { ELEMENTS, type El } from '../chem/elements';
import { auWaterInterface, minImage, type Structure, type Vec3 } from '../chem/structures';
import { countWithin, cutoffGraph, hopDistances, reachCounts, type Adjacency } from '../chem/graph';

export type IXMode = 'local' | 'mp' | 'long';
export const IX_MAX_T = 4;

export interface IXModel {
  structure: Structure;
  /** World-space offset subtracted from every position (puts the scene centre at the origin). */
  origin: Vec3;
  zTop: number;
  naIndex: number;
  clIndex: number;
  /** Notable atoms offered in the "atom i" picker. */
  presets: Array<{ id: string; label: string; index: number }>;
  /** Ambient-occlusion factor per atom (0–1) from local coordination. */
  ao: Float32Array;
}

let cached: IXModel | null = null;

export function ixModel(): IXModel {
  if (cached) return cached;
  const info = auWaterInterface();
  const s = info.structure;
  const origin: Vec3 = [s.cell[0] / 2, s.cell[1] / 2, info.zTop + 3];

  // Coordination-based ambient occlusion: buried atoms are darker.
  const g = cutoffGraph(s, 3.4);
  const ao = new Float32Array(s.atoms.length);
  for (let i = 0; i < s.atoms.length; i++) {
    const deg = g.offsets[i + 1]! - g.offsets[i]!;
    const t = Math.min(1, Math.max(0, (deg - 2) / 10));
    ao[i] = 1 - 0.45 * t * t * (3 - 2 * t);
  }

  // Presets: the ions, a surface Au under the ion, an interfacial and a bulk-like water oxygen.
  const na = s.atoms[info.naIndex]!.p;
  const auTop = s.atoms
    .map((a, i) => ({ a, i }))
    .filter(({ a }) => a.el === 'Au' && Math.abs(a.p[2] - info.zTop) < 0.1);
  const nearestTo = (list: Array<{ a: { p: Vec3 }; i: number }>, p: Vec3) =>
    list.reduce((best, cur) => {
      const d = (x: Vec3) => {
        const v = minImage(p, x, s.cell, s.pbc);
        return v[0] * v[0] + v[1] * v[1];
      };
      return d(cur.a.p) < d(best.a.p) ? cur : best;
    }).i;
  const surfaceAu = nearestTo(auTop, na);
  const waters = info.oxygens.map((i) => ({ a: s.atoms[i]!, i }));
  const interfacial = waters
    .filter(({ a }) => a.p[2] - info.zTop < 3.6)
    .reduce((best, cur) => (Math.abs(cur.a.p[0] - origin[0]) + Math.abs(cur.a.p[1] - origin[1]) < Math.abs(best.a.p[0] - origin[0]) + Math.abs(best.a.p[1] - origin[1]) ? cur : best)).i;
  const bulk = waters
    .filter(({ a }) => a.p[2] - info.zTop > 9)
    .reduce((best, cur) => (Math.abs(cur.a.p[0] - origin[0] * 1.4) + Math.abs(cur.a.p[1] - origin[1] * 0.6) < Math.abs(best.a.p[0] - origin[0] * 1.4) + Math.abs(best.a.p[1] - origin[1] * 0.6) ? cur : best)).i;

  cached = {
    structure: s,
    origin,
    zTop: info.zTop,
    naIndex: info.naIndex,
    clIndex: info.clIndex,
    presets: [
      { id: 'na', label: 'Na⁺ ion', index: info.naIndex },
      { id: 'cl', label: 'Cl⁻ ion', index: info.clIndex },
      { id: 'au', label: 'Surface Au atom', index: surfaceAu },
      { id: 'o-int', label: 'Interfacial water O', index: interfacial },
      { id: 'o-bulk', label: 'Water O far from the metal', index: bulk },
    ],
    ao,
  };
  return cached;
}

const graphCache = new Map<number, Adjacency>();
export function ixGraph(rc: number): Adjacency {
  const key = Math.round(rc * 100);
  let g = graphCache.get(key);
  if (!g) {
    g = cutoffGraph(ixModel().structure, rc);
    graphCache.set(key, g);
  }
  return g;
}

export interface IXState {
  mode: IXMode;
  T: number;
  rc: number;
  center: number;
}

export interface IXDerived {
  hops: Int16Array;
  /** Atoms reachable after t = 1..4 layers. */
  reach: number[];
  /** Atoms inside the ball of radius t·r_c, t = 1..4. */
  ball: number[];
  /** Effective layers for this mode (local = 1). */
  layers: number;
  /** Atoms not reachable at the current layer count. */
  beyond: number;
}

export function ixDerive(state: IXState): IXDerived {
  const m = ixModel();
  const hops = hopDistances(ixGraph(state.rc), state.center, IX_MAX_T);
  const reach = reachCounts(hops, IX_MAX_T);
  const ball = [1, 2, 3, 4].map((t) => countWithin(m.structure, state.center, t * state.rc));
  const layers = state.mode === 'local' ? 1 : state.T;
  const beyond = m.structure.atoms.length - 1 - reach[layers - 1]!;
  return { hops, reach, ball, layers, beyond };
}

export interface Coupling {
  j: number;
  /** World-space end point (minimum image relative to atom i). */
  to: Vec3;
  /** |q_i q_j| / r, e²/Å. */
  weight: number;
  attractive: boolean;
}

/**
 * Toy Coulomb couplings between atom i and charged atoms it cannot reach after
 * `layers` hops, strongest first. Minimum-image partners only: this is a picture
 * of which pairs interact, not an energy (that needs an Ewald sum).
 */
export function ixCouplings(center: number, hops: Int16Array, layers: number, limit = 220): Coupling[] {
  const { structure: s } = ixModel();
  const ai = s.atoms[center]!;
  if (ai.q === 0) return [];
  const out: Coupling[] = [];
  s.atoms.forEach((a, j) => {
    if (j === center || a.q === 0) return;
    const h = hops[j]!;
    if (h !== -1 && h <= layers) return;
    const d = minImage(ai.p, a.p, s.cell, s.pbc);
    const r = Math.hypot(d[0], d[1], d[2]);
    out.push({
      j,
      to: [ai.p[0] + d[0], ai.p[1] + d[1], ai.p[2] + d[2]],
      weight: Math.abs(ai.q * a.q) / r,
      attractive: ai.q * a.q < 0,
    });
  });
  out.sort((x, y) => y.weight - x.weight);
  return out.slice(0, limit);
}

export function describeAtom(i: number): { name: string; charge: string } {
  const m = ixModel();
  const a = m.structure.atoms[i]!;
  const q = a.q === 0 ? '0 (no charge in this toy)' : `${a.q > 0 ? '+' : '−'}${Math.abs(a.q).toFixed(a.q % 1 === 0 ? 0 : 4)} e`;
  let name: string = ELEMENTS[a.el as El].name;
  if (a.el === 'Na') name = 'Na⁺ ion';
  else if (a.el === 'Cl') name = 'Cl⁻ ion';
  else if (a.el === 'O' || a.el === 'H') name = `${a.el === 'O' ? 'Oxygen' : 'Hydrogen'} of water ${a.mol + 1}`;
  else if (a.el === 'Au') {
    const layer = Math.round(a.p[2] / (m.zTop / 3)) + 1;
    name = layer === 4 ? 'Au, surface layer' : `Au, layer ${layer} of 4`;
  }
  return { name, charge: q };
}
