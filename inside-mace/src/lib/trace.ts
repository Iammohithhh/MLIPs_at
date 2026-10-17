// "Trace the change" (brief §9): which atom-level representations differ from the baseline, stage by
// stage, for one evaluated frame. Pure module, covered by trace.test.ts.
//
// Each stage reports one non-negative number per atom, keyed to the atom's baseline distance from the
// moved atom (minimum image). The sequence is an order of computational dependency
// (geometry → layer 1 → layer 2 → site energies → forces), not a timing or propagation measurement.
// Exact zeros are kept as exact zeros (the stored float64 difference is bitwise 0), so the view can
// distinguish "did not change at all" from "changed by a small amount". Missing data is null, never 0.

import { featureChange, forceChange, mic, norm, okFrameAt, siteEnergyChange, type Manifest } from './experiment';

export type TraceStageKey = 'edges' | 'layer1' | 'layer2' | 'site' | 'force';

export interface TraceStage {
  key: TraceStageKey;
  title: string;
  quantity: string;
  unit: string;
  sentence: string;
}

export const TRACE_STAGES: TraceStage[] = [
  {
    key: 'edges',
    title: 'Graph edges',
    quantity: '|Δr| of the graph edge between atom i and the moved atom',
    unit: 'Å',
    sentence: 'Only edges that touch the moved atom change length; the model sees them only inside the cutoff r_max.',
  },
  {
    key: 'layer1',
    title: 'Layer 1 features',
    quantity: '‖Δh⁽¹⁾ᵢ‖, 128 scalar channels',
    unit: 'model units',
    sentence: 'Layer 1 builds each atom’s features from its own edges, so only atoms within r_max of the moved atom can respond.',
  },
  {
    key: 'layer2',
    title: 'Layer 2 features',
    quantity: '‖Δh⁽²⁾ᵢ‖, 128 scalar channels',
    unit: 'model units',
    sentence: 'Layer 2 gathers layer-1 features from neighbours within r_max, so the change can reach atoms up to 2 r_max away.',
  },
  {
    key: 'site',
    title: 'Site energies',
    quantity: '|ΔEᵢ|, change of atom i’s energy contribution',
    unit: 'eV',
    sentence: 'Each site energy is read out from that atom’s features, so it changes exactly where layer 1 or layer 2 features changed.',
  },
  {
    key: 'force',
    title: 'Forces',
    quantity: '‖ΔFᵢ‖, change of the force on atom i',
    unit: 'eV/Å',
    sentence: 'Fᵢ = −∂E/∂rᵢ uses the total energy, so a force can change even where that atom’s own site energy did not.',
  },
];

export interface TracePoint {
  atom: number;
  /** Baseline distance from the moved atom (Å, minimum image). */
  r0: number;
  /** Non-negative change, exact 0 kept; null when the frame or value is missing. */
  value: number | null;
}

/** Baseline minimum-image distance of every atom from the moved atom (Å). */
export function baselineDistances(m: Manifest): number[] {
  const c = m.structure.positions[m.displacement.atom]!;
  return m.structure.positions.map((p) => norm(mic(m, [c[0], c[1], c[2]], [p[0], p[1], p[2]])));
}

/** Length change of the edge (i, moved) when it is a graph edge in the baseline or the frame; 0 if neither. */
function edgeChange(m: Manifest, k: number, i: number, r0: number): number | null {
  const f = okFrameAt(m, k);
  if (!f) return null;
  const mv = m.displacement.atom;
  const p = m.structure.positions[i]!;
  if (i === mv) {
    // The moved atom: the largest length change among all its edges.
    let best = 0;
    m.structure.positions.forEach((q, j) => {
      if (j === mv) return;
      const a = norm(mic(m, [q[0], q[1], q[2]], [p[0], p[1], p[2]]));
      const b = norm(mic(m, [q[0], q[1], q[2]], f.movedPosition));
      if (Math.min(a, b) < m.rMax) best = Math.max(best, Math.abs(b - a));
    });
    return best;
  }
  const r = norm(mic(m, [p[0], p[1], p[2]], f.movedPosition));
  return Math.min(r, r0) < m.rMax ? Math.abs(r - r0) : 0;
}

export function tracePoints(m: Manifest, k: number, stage: TraceStageKey, r0s: number[] = baselineDistances(m)): TracePoint[] {
  return r0s.map((r0, atom) => {
    let value: number | null;
    if (stage === 'edges') value = edgeChange(m, k, atom, r0);
    else if (stage === 'layer1') value = featureChange(m, k, 0, 'h0', atom);
    else if (stage === 'layer2') value = featureChange(m, k, 1, 'h0', atom);
    else if (stage === 'site') {
      const v = siteEnergyChange(m, k, atom);
      value = v === null ? null : Math.abs(v);
    } else {
      const v = forceChange(m, k, atom);
      value = v === null ? null : norm(v);
    }
    return { atom, r0, value };
  });
}

export interface TraceSummary {
  changed: number;
  exactZero: number;
  missing: number;
  /** Largest baseline distance among atoms that changed (Å), or null if none changed. */
  reach: number | null;
  max: number | null;
}

export function summarise(points: TracePoint[]): TraceSummary {
  let changed = 0;
  let exactZero = 0;
  let missing = 0;
  let reach: number | null = null;
  let max: number | null = null;
  for (const p of points) {
    if (p.value === null) missing++;
    else if (p.value === 0) exactZero++;
    else {
      changed++;
      reach = Math.max(reach ?? 0, p.r0);
      max = Math.max(max ?? 0, p.value);
    }
  }
  return { changed, exactZero, missing, reach, max };
}

/** Smallest positive and largest value of a stage over every evaluated frame: a fixed log axis. */
export function stageRange(m: Manifest, stage: TraceStageKey, r0s: number[] = baselineDistances(m)): [number, number] | null {
  let lo = Infinity;
  let hi = 0;
  m.frames.forEach((f, k) => {
    if (f.status !== 'ok' || k === m.displacement.baselineIndex) return;
    for (const p of tracePoints(m, k, stage, r0s)) {
      if (p.value === null || p.value <= 0) continue;
      lo = Math.min(lo, p.value);
      hi = Math.max(hi, p.value);
    }
  });
  return hi > 0 ? [lo, hi] : null;
}
