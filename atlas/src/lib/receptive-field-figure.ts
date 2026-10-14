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
  /**
   * Cutoff in units of the lattice spacing. Nominally includes 1st and 2nd neighbours
   * of the ideal lattice; with jitter a few 2nd-neighbour pairs fall outside r_c.
   */
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
  /** Long-range view: atoms beyond r_c that T message-passing layers could still reach. */
  beyondReachable: number;
  /** Layer count T selected for message passing (kept while other views are shown). */
  T: number;
}

export function rfState(model: RFModel, center: number, mode: RFMode, T: number): RFState {
  const layers = mode === 'mp' ? T : 1;
  const hops = hopDistances(model.adj, center);
  const rf = receptiveField(model.atoms, model.adj, hops, center, model.rc, layers);
  const longRange: LongRangeLine[] = [];
  let beyondReachable = 0;
  if (mode === 'long') {
    const c = model.atoms[center]!;
    model.atoms.forEach((a, j) => {
      if (j !== center && hops[j]! <= T && distance(a, c) >= model.rc) beyondReachable++;
    });
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
  return { hops, layers, ...rf, longRange, beyondReachable, T };
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
  const beyond = total - 1 - state.inField;
  const T = state.T;
  return (
    `In this finite toy cluster, atom i also couples to all ${beyond} ions beyond r_c through 1/r Coulomb terms (drawn schematically). ` +
    `A strictly local model sees none of them; with T = ${T} message-passing layer${T > 1 ? 's' : ''}, ${state.beyondReachable} are reachable — through learned messages, not a 1/r law.`
  );
}

/** Horizontal window of the figure that is currently visible (narrow screens crop it). */
export interface VisibleWindow {
  x0: number;
  x1: number;
}

export const FULL_WINDOW: VisibleWindow = { x0: 0, x1: RF.width };

/** Square crop of the figure centred on x, clamped to the figure. */
export function cropAround(x: number): VisibleWindow {
  const w = RF.height;
  const x0 = Math.min(RF.width - w, Math.max(0, x - w / 2));
  return { x0, x1: x0 + w };
}

/** Pick a label position on a circle that stays inside the visible part of the figure. */
export function labelOnCircle(
  cx: number,
  cy: number,
  r: number,
  view: VisibleWindow = FULL_WINDOW,
): { x: number; y: number } {
  const angles = [-90, -60, -120, -30, -150, 0, 180, 90];
  for (const deg of angles) {
    const t = (deg * Math.PI) / 180;
    const x = cx + r * Math.cos(t);
    const y = cy + r * Math.sin(t) - 6;
    if (x > view.x0 + 30 && x < view.x1 - 30 && y > 16 && y < RF.height - 8) return { x, y };
  }
  return { x: Math.min(view.x1 - 30, Math.max(view.x0 + 30, cx)), y: Math.max(16, cy - r - 6) };
}

/** Grid position of an atom, for the screen-reader readout. */
export function atomPosition(index: number, q: number): string {
  const row = Math.floor(index / RF.nx) + 1;
  const col = (index % RF.nx) + 1;
  return `Atom i: row ${row} of ${RF.ny}, column ${col} of ${RF.nx}, ${q > 0 ? 'positive' : 'negative'} ion.`;
}

/** Readout as HTML with r_c typeset. */
export function rfReadoutHtml(text: string): string {
  return typesetInline(text);
}
