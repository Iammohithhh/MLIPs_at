// Channel inspector: the 128 scalar channels (l = 0) and the l = 1 vector channels of the selected
// atom's layer features, baseline vs the selected evaluated frame vs their difference. Frame data is
// lazy-loaded per frame. Colour scales are fixed per layer and block over all frames (scales.json).

import { useEffect, useState } from 'react';
import type { Manifest } from '../lib/experiment';

// Every experiment's frame chunks are separate lazy modules; only the selected experiment's frames are fetched.
const loaders = import.meta.glob<{ default: FrameChunk }>('../data/experiments/*/frames/*.json');
// Fixed colour scales per experiment (small; bundled).
const scaleFiles = import.meta.glob<{ default: Scales }>('../data/experiments/*/scales.json', { eager: true });

interface BlockScale {
  value: number;
  change: number;
}
interface Scales {
  experiment: string;
  layers: { h0: BlockScale; h1: BlockScale | null; Anorm: BlockScale }[];
}

export function scalesFor(id: string): Scales | null {
  return scaleFiles[`../data/experiments/${id}/scales.json`]?.default ?? null;
}

export interface FrameChunk {
  experiment: string;
  k: number;
  d: number;
  local: number[];
  layers: { h0: number[][]; h1: number[][][] | null; Anorm: number[][][] }[];
}

const cache = new Map<string, Promise<FrameChunk>>();
/** The lazy per-frame chunk of experiment `id`; rejects (never resolves to placeholder data) when it is missing or belongs to another experiment or frame. */
export function loadChunk(id: string, k: number): Promise<FrameChunk> {
  const key = `../data/experiments/${id}/frames/${String(k).padStart(2, '0')}.json`;
  let p = cache.get(key);
  if (!p) {
    const loader = loaders[key];
    p = loader
      ? loader().then((x) => {
          if (x.default.experiment !== id || x.default.k !== k) throw new Error(`frame chunk ${key} does not belong to ${id} frame ${k}`);
          return x.default;
        })
      : Promise.reject(new Error(`no data for ${id} frame ${k}`));
    cache.set(key, p);
  }
  return p;
}

function useChunk(id: string, k: number): { data: FrameChunk | null; error: string | null } {
  const [state, setState] = useState<{ data: FrameChunk | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let live = true;
    setState({ data: null, error: null });
    loadChunk(id, k).then(
      (data) => live && setState({ data, error: null }),
      (e: unknown) => live && setState({ data: null, error: String(e) }),
    );
    return () => {
      live = false;
    };
  }, [id, k]);
  return state;
}

/** Diverging colour for v in [−max, max]: teal (negative) – stage grey (0) – amber (positive). */
function diverging(v: number, max: number): string {
  const t = Math.max(-1, Math.min(1, v / (max || 1)));
  const a = Math.abs(t);
  const [r, g, b] = t >= 0 ? [240, 169, 69] : [111, 211, 189];
  const base = [36, 44, 42];
  const mix = (x: number, y: number) => Math.round(x + (y - x) * Math.sqrt(a));
  return `rgb(${mix(base[0]!, r)} ${mix(base[1]!, g)} ${mix(base[2]!, b)})`;
}

function Heat({ values, max, label, channel, onChannel }: { values: number[]; max: number; label: string; channel: number; onChannel: (c: number) => void }) {
  return (
    <div className="fi__row">
      <span className="fi__rowlabel">{label}</span>
      <div className="fi__cells" role="group" aria-label={`${label}: 128 channels`}>
        {values.map((v, c) => (
          <button
            key={c}
            className={c === channel ? 'fi__cell is-on' : 'fi__cell'}
            style={{ background: diverging(v, max) }}
            aria-label={`channel ${c}: ${v.toPrecision(4)}`}
            aria-pressed={c === channel}
            onClick={() => onChannel(c)}
          />
        ))}
      </div>
    </div>
  );
}

export function FeatureInspector({ m, frame, atom, layer, channel, onChannel }: { m: Manifest; frame: number; atom: number; layer: number; channel: number; onChannel: (c: number) => void }) {
  const B = m.displacement.baselineIndex;
  const base = useChunk(m.id, B);
  const cur = useChunk(m.id, frame);
  const row = m.local.indexOf(atom);
  const sc = scalesFor(m.id)?.layers[layer];

  if (row < 0)
    return (
      <p className="fi__note">
        Full channel values are stored for the moved atom and its first two neighbour shells ({m.local.length} atoms). Atom {atom} is outside
        that set; select one of them to inspect channels.
      </p>
    );
  if (base.error || cur.error) return <p className="fi__note" role="alert">Could not load the frame data: {base.error ?? cur.error}</p>;
  if (!sc) return <p className="fi__note" role="alert">No fixed colour scales are stored for this experiment, so no heatmap is drawn.</p>;
  if (!base.data || !cur.data) return <p className="fi__note" role="status">Loading channel values for this frame…</p>;

  const hb = base.data.layers[layer]!.h0[row]!;
  const hf = cur.data.layers[layer]!.h0[row]!;
  const diff = hf.map((v, c) => v - hb[c]!);
  const v1b = base.data.layers[layer]!.h1?.[row]?.[channel] ?? null;
  const v1f = cur.data.layers[layer]!.h1?.[row]?.[channel] ?? null;

  return (
    <div className="fi">
      <p className="fi__what">
        Layer {layer + 1} features h of atom {atom}: 128 scalar (l = 0) channels. Internal model features in arbitrary units, not physical
        quantities. Colour scales are fixed over all frames: values ±{sc.h0.value.toPrecision(3)}, difference ±{sc.h0.change.toPrecision(3)}.
      </p>
      <Heat values={hb} max={sc.h0.value} label="Baseline" channel={channel} onChannel={onChannel} />
      <Heat values={hf} max={sc.h0.value} label={`d = ${cur.data.d.toFixed(3)} Å`} channel={channel} onChannel={onChannel} />
      <Heat values={diff} max={sc.h0.change} label="Difference" channel={channel} onChannel={onChannel} />
      <dl className="fi__readout mono">
        <dt>channel {channel}, l = 0</dt>
        <dd>
          {hb[channel]!.toPrecision(4)} → {hf[channel]!.toPrecision(4)} (Δ {diff[channel]!.toPrecision(3)})
        </dd>
        <dt>channel {channel}, l = 1 (x, y, z)</dt>
        <dd>
          {v1b && v1f
            ? `(${v1b.map((x) => x.toPrecision(2)).join(', ')}) → (${v1f.map((x) => x.toPrecision(2)).join(', ')})`
            : 'none: the last layer of this checkpoint carries scalar features only'}
        </dd>
      </dl>
      {v1b && v1b.every((x) => x === 0) && (
        <p className="fi__note">
          At the baseline every l = 1 channel of every atom is zero (|h| &lt; 10⁻¹² in float64): each Si site in a perfect diamond lattice has tetrahedral
          symmetry, so no vector feature can point anywhere. Moving the atom breaks that symmetry and the vectors appear.
        </p>
      )}
    </div>
  );
}
