import { useState } from 'react';
import './PhysicsLab.css';
import './DataUniverse.css';

// The training-data universe: every Atlas dataset placed by size (log scale) and
// domain, coloured by the rung of its level of theory. Sizes are the numbers
// recorded with their sources; "≈" marks values given as approximate.

export type Rung = 'GGA' | 'meta-GGA' | 'hybrid';

export interface DatasetPoint {
  id: string;
  name: string;
  domain: string;
  size: number;
  approximate: boolean;
  sizeSource: string;
  rung: Rung;
  levels: string[];
  configs: string[];
  summary: string;
  year: number;
}

const LANES = [
  { id: 'inorganic-materials', label: 'Inorganic materials' },
  { id: 'catalysis', label: 'Catalysis (surfaces)' },
  { id: 'molecules', label: 'Molecules' },
];
const RUNGS: Rung[] = ['GGA', 'meta-GGA', 'hybrid'];

export default function DataUniverse({ points }: { points: DatasetPoint[] }) {
  const [rungs, setRungs] = useState<Set<Rung>>(new Set(RUNGS));
  const [active, setActive] = useState<string>(points[0]?.id ?? '');
  const W = 760;
  const M = { l: 168, r: 30, t: 20, b: 46 };
  const laneIdx = (p: DatasetPoint) => Math.max(0, LANES.findIndex((l) => l.id === p.domain));
  const laneH = LANES.map((_, k) => Math.max(84, 34 + 38 * points.filter((p) => laneIdx(p) === k).length));
  const laneTop = laneH.map((_, k) => M.t + laneH.slice(0, k).reduce((a, b) => a + b, 0));
  const H = M.t + laneH.reduce((a, b) => a + b, 0) + M.b;
  const lo = 5; // 10^5
  const hi = 8.4; // ~2.5 × 10^8
  const x = (n: number) => M.l + ((Math.log10(n) - lo) / (hi - lo)) * (W - M.l - M.r);
  const laneOf = laneIdx;
  const sel = points.find((p) => p.id === active) ?? points[0]!;
  const fmt = (n: number) => (n >= 1e6 ? `${(n / 1e6).toLocaleString('en-GB', { maximumFractionDigits: 2 })} M` : `${(n / 1e3).toLocaleString('en-GB', { maximumFractionDigits: 0 })} k`);

  return (
    <div className="lab du">
      <div className="du-top">
        <div className="seg" role="group" aria-label="Level-of-theory rungs to show">
          {RUNGS.map((r) => (
            <button
              key={r}
              type="button"
              aria-pressed={rungs.has(r)}
              className={`du-r du-r-${r}`}
              onClick={() =>
                setRungs((s) => {
                  const n = new Set(s);
                  if (n.has(r)) n.delete(r);
                  else n.add(r);
                  return n.size ? n : new Set(RUNGS);
                })
              }
            >
              <span className="du-sw" /> {r}
            </button>
          ))}
        </div>
      </div>
      <figure className="lab-plot du-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Training datasets by size and domain: ${points.map((p) => `${p.name} ${fmt(p.size)} (${p.rung})`).join('; ')}.`}>
          {LANES.map((lane, k) => (
            <g key={lane.id}>
              <rect x={M.l} y={laneTop[k]} width={W - M.l - M.r} height={laneH[k]} className={k % 2 ? 'du-lane alt' : 'du-lane'} />
              <text x={M.l - 12} y={laneTop[k]! + laneH[k]! / 2 + 4} textAnchor="end" className="du-lane-label">
                {lane.label}
              </text>
            </g>
          ))}
          {[5, 6, 7, 8].map((e) => (
            <g key={e}>
              <line x1={x(10 ** e)} x2={x(10 ** e)} y1={M.t} y2={H - M.b} className="grid" />
              <text x={x(10 ** e)} y={H - M.b + 16} textAnchor="middle" className="tick">
                10{['⁵', '⁶', '⁷', '⁸'][e - 5]}
              </text>
            </g>
          ))}
          <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">
            structures / calculations (log scale)
          </text>
          {points.map((p) => {
            const on = rungs.has(p.rung);
            const cx = x(p.size);
            const lane = laneOf(p);
            const sameLane = points.filter((q) => laneOf(q) === lane);
            const slot = sameLane.indexOf(p);
            const cy = laneTop[lane]! + laneH[lane]! / 2 + (sameLane.length > 1 ? (slot - (sameLane.length - 1) / 2) * 38 : 0);
            const r = 7 + 2.2 * (Math.log10(p.size) - 5);
            return (
              <g
                key={p.id}
                className={`du-pt du-r-${p.rung}${on ? '' : ' is-off'}${p.id === active ? ' is-active' : ''}`}
                tabIndex={on ? 0 : -1}
                role="button"
                aria-label={`${p.name}: ${fmt(p.size)}, ${p.rung}`}
                onPointerEnter={() => setActive(p.id)}
                onFocus={() => setActive(p.id)}
                onClick={() => setActive(p.id)}
              >
                <circle cx={cx} cy={cy} r={r} />
                <text x={cx + r + 6} y={cy + 4} className="du-name">
                  {p.name}
                </text>
                <text x={cx + r + 6} y={cy + 17} className="du-size">
                  {p.approximate ? '≈ ' : ''}
                  {fmt(p.size)}
                </text>
              </g>
            );
          })}
        </svg>
      </figure>
      <div className="du-card" aria-live="polite">
        <p className="du-card-head">
          <strong>{sel.name}</strong> <span className={`du-chip du-r-${sel.rung}`}>{sel.rung}</span> <span className="du-year">{sel.year}</span>
        </p>
        <p className="du-summary">{sel.summary}</p>
        <dl className="du-facts">
          <div>
            <dt>Size</dt>
            <dd>
              {sel.approximate ? '≈ ' : ''}
              {sel.size.toLocaleString('en-GB')} <small>({sel.sizeSource})</small>
            </dd>
          </div>
          <div>
            <dt>Level of theory</dt>
            <dd>{sel.levels.join(' · ')}</dd>
          </div>
          <div>
            <dt>Configurations</dt>
            <dd>{sel.configs.join(', ') || '—'}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
