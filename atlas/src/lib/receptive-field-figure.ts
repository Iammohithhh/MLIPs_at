// Model behind the homepage figure "Which atoms can influence atom i?" (TOY MODEL).
// Shared by the server render and the client script so both show identical state.

import {
  cutoffGraph,
  distance,
  hopDistances,
  makeIonicLayer,
  receptiveField,
  type Atom2D,
} from './toy-lattice';
import { typesetInline } from './typeset';

export const RF = {
  nx: 20,
  ny: 12,
  a: 44,
  jitter: 0.12,
  seed: 7,
  origin: { x: 22, y: 22 },
  width: 880,
  height: 528,
  /** Cutoff in units of the lattice spacing: includes 1st and 2nd neighbours. */
  rcFactor: 1.5,
  atomRadius: 9,
  maxLayers: 4,
} as const;

export type RFMode = 'local' | 'mp' | 'long';

export interface RFModel {
  atoms: Atom2D[];
  rc: number;
  adj: number[][];
}

export function rfModel(): RFModel {
  const atoms = makeIonicLayer({
    nx: RF.nx,
    ny: RF.ny,
    a: RF.a,
    jitter: RF.jitter,
    seed: RF.seed,
    origin: RF.origin,
  });
  const rc = RF.a * RF.rcFactor;
  return { atoms, rc, adj: cutoffGraph(atoms, rc) };
}

export interface LongRangeLine {
  j: number;
  /** Opposite charges attract (q_i q_j < 0). */
  attractive: boolean;
  /** Drawn opacity, proportional to 1/r_ij (clamped for legibility). */
  opacity: number;
}

export interface RFState {
  hops: number[];
  /** Layers whose receptive field is shown (1 for the local and long views). */
  layers: number;
  inField: number;
  inBall: number;
  flowEdges: Array<[number, number]>;
  longRange: LongRangeLine[];
}

export function rfState(model: RFModel, center: number, mode: RFMode, T: number): RFState {
  const layers = mode === 'mp' ? T : 1;
  const hops = hopDistances(model.adj, center);
  const rf = receptiveField(model.atoms, model.adj, hops, center, model.rc, layers);
  const longRange: LongRangeLine[] = [];
  if (mode === 'long') {
    const c = model.atoms[center]!;
    model.atoms.forEach((a, j) => {
      if (j === center) return;
      const r = distance(a, c);
      if (r < model.rc) return;
      longRange.push({
        j,
        attractive: a.q !== c.q,
        opacity: Math.min(0.85, Math.max(0.06, (0.85 * model.rc) / r)),
      });
    });
  }
  return { hops, layers, ...rf, longRange };
}

export function rfReadout(state: RFState, mode: RFMode, total: number): string {
  if (mode === 'local') {
    return `${state.inField} atoms lie within r_c of atom i. A strictly local model builds atom i's energy from these alone.`;
  }
  if (mode === 'mp') {
    const T = state.layers;
    const extra = state.inBall - state.inField;
    const ball =
      extra > 0
        ? `The disc of radius ${T}·r_c holds ${state.inBall}: distance alone over-counts by ${extra}.`
        : `Here that equals every atom in the disc of radius ${T}·r_c.`;
    return `After T = ${T} layer${T > 1 ? 's' : ''}, ${state.inField} atoms can influence atom i (≤ ${T} graph hop${T > 1 ? 's' : ''}). ${ball}`;
  }
  return `Atom i also couples to all ${total - 1 - state.inField} ions beyond r_c through 1/r Coulomb terms (drawn schematically). A cutoff model sees none of them.`;
}

/** Pick a label position on a circle that stays inside the figure. */
export function labelOnCircle(cx: number, cy: number, r: number): { x: number; y: number } {
  const angles = [-90, -60, -120, -30, -150, 0, 180, 90];
  for (const deg of angles) {
    const t = (deg * Math.PI) / 180;
    const x = cx + r * Math.cos(t);
    const y = cy + r * Math.sin(t) - 6;
    if (x > 30 && x < RF.width - 30 && y > 16 && y < RF.height - 8) return { x, y };
  }
  return { x: cx, y: Math.max(16, cy - r - 6) };
}

/** Readout as HTML with r_c typeset. */
export function rfReadoutHtml(text: string): string {
  return typesetInline(text);
}
