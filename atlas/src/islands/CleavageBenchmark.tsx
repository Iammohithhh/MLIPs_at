import { useMemo, useState } from 'react';
import './PhysicsLab.css';
import data from '~/data/benchmarks/cleavage-mehdizadeh2025.json';
import { useSceneTheme, type SceneTheme } from '~/three/theme';

// PUBLISHED RESULT: Tables 1–2 of Mehdizadeh & Schindler (arXiv:2508.21663v1), snapshotted by
// pipeline/atlas_pipeline/cleavage.py. Numbers are shown as published; nothing is ranked or combined.

type Row = (typeof data.models)[number];
type MetricKey = 'ape_median' | 'mae' | 'term';
const METRICS: Array<{ key: MetricKey; label: string; unit: string; log: boolean; better: 'low' | 'high' }> = [
  { key: 'ape_median', label: 'Median absolute % error', unit: '%', log: true, better: 'low' },
  { key: 'mae', label: 'Mean absolute error', unit: 'meV/Å²', log: true, better: 'low' },
  { key: 'term', label: 'Most stable termination found', unit: '%', log: false, better: 'high' },
];
const CLASSES: Array<{ key: string; label: string }> = [
  { key: 'omat', label: 'OMat24 only' },
  { key: 'omat-combined', label: 'OMat24 + MPtrj + sAlex' },
  { key: 'mp-alexandria', label: 'MP + Alexandria' },
  { key: 'mp', label: 'MPtrj / Materials Project' },
  { key: 'oc20', label: 'OC20 (surface–adsorbate)' },
  { key: 'other', label: 'Proprietary' },
];

function classColor(t: SceneTheme, c: string) {
  return ({ omat: t.phys, 'omat-combined': t.arch, 'mp-alexandria': t.lit, mp: t.caveat, oc20: t.fail } as Record<string, string>)[c] ?? t.inkMuted;
}

export default function CleavageBenchmark() {
  const theme = useSceneTheme();
  const [metric, setMetric] = useState<MetricKey>('ape_median');
  const [hover, setHover] = useState<string | null>(null);
  const m = METRICS.find((x) => x.key === metric)!;
  const families = useMemo(() => {
    const out = new Map<string, Row[]>();
    for (const r of data.models) out.set(r.family, [...(out.get(r.family) ?? []), r]);
    return [...out.entries()].sort((a, b) => b[1].length - a[1].length);
  }, []);
  const rows = families.flatMap(([f, rs]) => [{ family: f, header: true as const }, ...rs.map((r) => ({ family: f, header: false as const, r }))]);

  const W = 760;
  const RH = 22;
  const M = { l: 200, r: 24, t: 40, b: 44 };
  const H = M.t + rows.length * RH + M.b;
  const lo = m.log ? 1 : 0;
  const hi = m.log ? 100 : 100;
  const X = (v: number) => {
    const t = m.log ? (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return M.l + t * (W - M.l - M.r);
  };
  const ticks = m.log ? [1, 2, 5, 10, 20, 50, 100] : [0, 20, 40, 60, 80, 100];
  const base = data.baseline?.[metric] as number | null | undefined;
  const Y = (i: number) => M.t + i * RH + RH / 2;
  const hovered = data.models.find((r) => r.name === hover);

  return (
    <div className="lab">
      <div className="lab-toolbar" style={{ position: 'static', marginBottom: '0.6rem' }}>
        <div className="seg" role="group" aria-label="Metric">
          {METRICS.map((x) => (
            <button key={x.key} type="button" aria-pressed={metric === x.key} onClick={() => setMetric(x.key)}>
              {x.label}
            </button>
          ))}
        </div>
      </div>
      <figure className="lab-plot" style={{ margin: 0 }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${m.label} for 19 universal potentials on cleavage energies, grouped by architecture and coloured by training data (Mehdizadeh and Schindler, Table 2).`}>
          <text x={M.l} y={M.t - 22} className="axis-label">
            {m.label} [{m.unit}] — {m.better === 'low' ? 'lower is closer to DFT' : 'higher agrees more often with DFT'}{m.log ? ', log scale' : ''}
          </text>
          {ticks.map((v) => (
            <g key={v}>
              <line x1={X(v)} x2={X(v)} y1={M.t - 8} y2={H - M.b} className="grid" />
              <text x={X(v)} y={H - M.b + 16} textAnchor="middle" className="tick">{v}</text>
            </g>
          ))}
          {base != null && (
            <g>
              <line x1={X(base)} x2={X(base)} y1={M.t - 8} y2={H - M.b} className="rc-line" />
              <text x={X(base) + 4} y={M.t - 2} className="tick">predict the mean: {base}</text>
            </g>
          )}
          {rows.map((row, i) => {
            if (row.header) {
              const rs = families.find(([f]) => f === row.family)![1];
              const v = rs.map((r) => r[metric] as number);
              return (
                <g key={`h-${row.family}`}>
                  <text x={8} y={Y(i) + 4} className="axis-label" style={{ fontWeight: 600 }}>{row.family}</text>
                  {rs.length > 1 && <line x1={X(Math.min(...v))} x2={X(Math.max(...v))} y1={Y(i)} y2={Y(i)} style={{ stroke: theme.ink, strokeWidth: 2, opacity: 0.35 }} />}
                  {rs.length > 1 && (
                    <text x={X(Math.max(...v)) + 6} y={Y(i) + 4} className="tick">
                      ×{(Math.max(...v) / Math.min(...v)).toFixed(1)} spread
                    </text>
                  )}
                </g>
              );
            }
            const r = row.r;
            const v = r[metric] as number;
            const c = classColor(theme, r.data_class);
            const on = hover === r.name;
            return (
              <g key={r.name} onMouseEnter={() => setHover(r.name)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(r.name)} onBlur={() => setHover(null)} tabIndex={0} aria-label={`${r.name}: ${v} ${m.unit}; trained on ${r.training}`}>
                <rect x={0} y={Y(i) - RH / 2} width={W} height={RH} style={{ fill: on ? theme.rule : 'transparent', opacity: 0.5 }} />
                <text x={20} y={Y(i) + 4} className="tick" style={{ fill: theme.inkMuted }}>{r.name}</text>
                <line x1={X(m.log ? lo : 0)} x2={X(v)} y1={Y(i)} y2={Y(i)} style={{ stroke: c, strokeWidth: 1, opacity: 0.35 }} />
                <circle cx={X(v)} cy={Y(i)} r={on ? 7 : 5.5} style={{ fill: c, stroke: theme.canvas, strokeWidth: 1.5 }} />
                <text x={X(v) + 10} y={Y(i) + 4} className="tick">{v}</text>
              </g>
            );
          })}
        </svg>
        <figcaption className="lab-legend">
          {CLASSES.filter((c) => data.models.some((r) => r.data_class === c.key)).map((c) => (
            <span key={c.key} className="item">
              <span className="swatch" style={{ background: classColor(theme, c.key) }} /> {c.label}
            </span>
          ))}
        </figcaption>
      </figure>
      <p className="lab-note" aria-live="polite">
        {hovered
          ? `${hovered.name} (${hovered.params} parameters) — training data as described by the authors: ${hovered.training}. Median APE ${hovered.ape_median}%, MAE ${hovered.mae} meV/Å², most stable termination identified for ${hovered.term}% of 3,699 materials.`
          : 'Each row is one published checkpoint; the grey bar over a family spans its checkpoints. The authors describe the eSEN and GRACE checkpoints as architecturally identical apart from training data; in other families sizes or versions differ too (compare parameter counts). Hover or focus a row for its training set.'}
      </p>
    </div>
  );
}
