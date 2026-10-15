import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './BenchmarkExplorer.css';

// Matbench Discovery results as published upstream (one JSON snapshot per commit).
// Any metric against any other; no combined ranking is computed here.

export interface MBModel {
  key: string;
  name: string;
  version: string | null;
  added: string | null;
  openness: string | null;
  training_sets: string[];
  params: number | null;
  licence_code: string | null;
  licence_checkpoint: string | null;
  paper: string | null;
  repo: string | null;
  checkpoint_url: string | null;
  metrics: {
    discovery_unique: Record<string, number>;
    discovery_full: Record<string, number>;
    rmsd: number | null;
    kappa_srme: number | null;
  };
}

interface MetricDef {
  id: string;
  label: string;
  better: 'higher' | 'lower' | 'none';
  get: (m: MBModel) => number | null;
  log?: boolean;
  fmt: (v: number) => string;
}

const METRICS: MetricDef[] = [
  { id: 'added', label: 'Date added to the benchmark', better: 'none', get: (m) => (m.added ? Date.parse(m.added) : null), fmt: (v) => String(new Date(v).getUTCFullYear()) },
  { id: 'f1', label: 'F1 · discovery (unique prototypes)', better: 'higher', get: (m) => m.metrics.discovery_unique['F1'] ?? null, fmt: (v) => v.toFixed(3) },
  { id: 'daf', label: 'DAF · discovery acceleration (unique)', better: 'higher', get: (m) => m.metrics.discovery_unique['DAF'] ?? null, fmt: (v) => v.toFixed(2) },
  { id: 'mae', label: 'MAE of E above hull (unique) [eV/atom]', better: 'lower', get: (m) => m.metrics.discovery_unique['MAE'] ?? null, fmt: (v) => v.toFixed(3) },
  { id: 'r2', label: 'R² of E above hull (unique)', better: 'higher', get: (m) => m.metrics.discovery_unique['R2'] ?? null, fmt: (v) => v.toFixed(3) },
  { id: 'kappa', label: 'κ_SRME · thermal conductivity error', better: 'lower', get: (m) => m.metrics.kappa_srme, fmt: (v) => v.toFixed(3) },
  { id: 'rmsd', label: 'RMSD of relaxed structures [Å]', better: 'lower', get: (m) => m.metrics.rmsd, fmt: (v) => v.toFixed(4) },
  { id: 'params', label: 'Model parameters (log scale)', better: 'none', get: (m) => m.params, log: true, fmt: (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : `${(v / 1e3).toFixed(0)} k`) },
];

type Group = 'mptrj' | 'oam' | 'other';
const GROUPS: Record<Group, string> = {
  mptrj: 'MPtrj only',
  oam: 'MPtrj + OMat24 / Alexandria',
  other: 'other training data',
};
function groupOf(m: MBModel): Group {
  const t = m.training_sets.map((s) => s.toLowerCase());
  if (t.length === 1 && t[0] === 'mptrj') return 'mptrj';
  if (t.some((s) => s.includes('omat') || s.includes('alex'))) return 'oam';
  return 'other';
}

export default function BenchmarkExplorer({
  models,
  source,
}: {
  models: MBModel[];
  source: { repo: string; commit: string; commit_date: string; fetched: string };
}) {
  const ids = useId();
  const [xId, setX] = useState('added');
  const [yId, setY] = useState('f1');
  const [groups, setGroups] = useState<Set<Group>>(new Set(['mptrj', 'oam', 'other']));
  const [openOnly, setOpenOnly] = useState(false);
  const [sel, setSel] = useState<string>(models.at(-1)?.key ?? '');

  const X = METRICS.find((m) => m.id === xId)!;
  const Y = METRICS.find((m) => m.id === yId)!;
  const pts = useMemo(
    () =>
      models
        .map((m) => ({ m, x: X.get(m), y: Y.get(m), g: groupOf(m) }))
        .filter((p): p is { m: MBModel; x: number; y: number; g: Group } => p.x !== null && p.y !== null)
        .filter((p) => groups.has(p.g) && (!openOnly || p.m.openness === 'OSOD')),
    [models, X, Y, groups, openOnly],
  );
  const missing = models.length - pts.length;

  const W = 720;
  const H = 440;
  const M = { l: 62, r: 18, t: 18, b: 52 };
  const tx = (v: number, d: MetricDef) => (d.log ? Math.log10(v) : v);
  const ext = (vals: number[]) => {
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || 1;
    return [lo - pad, hi + pad] as const;
  };
  const [x0, x1] = pts.length ? ext(pts.map((p) => tx(p.x, X))) : [0, 1];
  const [y0, y1] = pts.length ? ext(pts.map((p) => tx(p.y, Y))) : [0, 1];
  const sx = (v: number) => M.l + ((tx(v, X) - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (v: number) => M.t + (1 - (tx(v, Y) - y0) / (y1 - y0)) * (H - M.t - M.b);
  // Round-number ticks: years for dates, powers of ten for log axes, 1/2/5 steps otherwise.
  const ticks = (a: number, b: number, d: MetricDef): number[] => {
    if (d.id === 'added') {
      const y0 = new Date(a).getUTCFullYear();
      const y1 = new Date(b).getUTCFullYear();
      return Array.from({ length: y1 - y0 + 1 }, (_, k) => Date.UTC(y0 + k, 0, 1)).filter((t) => t >= a && t <= b);
    }
    if (d.log) {
      const out: number[] = [];
      for (let e = Math.ceil(a); e <= Math.floor(b); e++) out.push(10 ** e);
      return out;
    }
    const raw = (b - a) / 5;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((k) => k * mag).find((st) => st >= raw) ?? raw;
    const out: number[] = [];
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-12; t += step) out.push(Number(t.toFixed(10)));
    return out;
  };
  const chosen = models.find((m) => m.key === sel) ?? models[0]!;
  const maxP = Math.max(...models.map((m) => m.params ?? 0));
  const radius = (m: MBModel) => (m.params ? 3.5 + 7 * Math.sqrt(m.params / maxP) : 4);

  const options = METRICS.map((m) => (
    <option key={m.id} value={m.id}>
      {m.label}
    </option>
  ));

  return (
    <div className="lab bx">
      <div className="bx-controls">
        <label htmlFor={`${ids}-y`}>
          <span>Vertical</span>
          <select id={`${ids}-y`} value={yId} onChange={(e) => setY(e.target.value)}>
            {options}
          </select>
        </label>
        <label htmlFor={`${ids}-x`}>
          <span>Horizontal</span>
          <select id={`${ids}-x`} value={xId} onChange={(e) => setX(e.target.value)}>
            {options}
          </select>
        </label>
        <div className="bx-groups" role="group" aria-label="Training data">
          {(Object.keys(GROUPS) as Group[]).map((g) => (
            <label key={g} className={`bx-g bx-g-${g}`}>
              <input
                type="checkbox"
                checked={groups.has(g)}
                onChange={() =>
                  setGroups((s) => {
                    const n = new Set(s);
                    if (n.has(g)) n.delete(g);
                    else n.add(g);
                    return n.size ? n : s;
                  })
                }
              />
              <span className="bx-sw" /> {GROUPS[g]}
            </label>
          ))}
          <label className="bx-g">
            <input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} /> open code and data only (OSOD)
          </label>
        </div>
      </div>

      <div className="bx-body">
        <figure className="lab-plot bx-plot">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${Y.label} against ${X.label} for ${pts.length} Matbench Discovery models.`}>
            <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
            {ticks(x0, x1, X).map((t, k) => (
              <g key={`x${k}`}>
                <line x1={sx(t)} x2={sx(t)} y1={M.t} y2={H - M.b} className="grid" />
                <text x={sx(t)} y={H - M.b + 16} textAnchor="middle" className="tick">{X.fmt(t)}</text>
              </g>
            ))}
            {ticks(y0, y1, Y).map((t, k) => (
              <g key={`y${k}`}>
                <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} className="grid" />
                <text x={M.l - 6} y={sy(t) + 4} textAnchor="end" className="tick">{Y.fmt(t)}</text>
              </g>
            ))}
            <text x={W - M.r} y={H - 10} textAnchor="end" className="axis-label">
              {X.label}
              {X.better !== 'none' ? (X.better === 'higher' ? ' → better' : ' ← better') : ''}
            </text>
            <text x={M.l + 6} y={M.t + 14} className="axis-label">
              {Y.label}
              {Y.better !== 'none' ? (Y.better === 'higher' ? ' ↑ better' : ' ↓ better') : ''}
            </text>
            {pts.map((p) => (
              <circle
                key={p.m.key}
                cx={sx(p.x)}
                cy={sy(p.y)}
                r={radius(p.m)}
                className={`bx-pt bx-g-${p.g}${p.m.key === sel ? ' is-sel' : ''}`}
                tabIndex={0}
                role="button"
                aria-label={`${p.m.name}: ${Y.fmt(p.y)}, ${X.fmt(p.x)}`}
                onPointerEnter={() => setSel(p.m.key)}
                onFocus={() => setSel(p.m.key)}
                onClick={() => setSel(p.m.key)}
              />
            ))}
          </svg>
          <figcaption className="bx-cap">
            {pts.length} models shown{missing ? ` · ${missing} without this pair of metrics or filtered out` : ''} · dot area ∝ parameter count
          </figcaption>
        </figure>

        <aside className="bx-card" aria-live="polite">
          <p className="bx-name">{chosen.name}</p>
          <p className="bx-meta">
            {chosen.version && <span>{chosen.version}</span>}
            {chosen.added && <span>added {chosen.added}</span>}
            {chosen.openness && <span>{chosen.openness}</span>}
          </p>
          <dl>
            <div><dt>Training data</dt><dd>{chosen.training_sets.join(', ') || '—'}</dd></div>
            <div><dt>Parameters</dt><dd>{chosen.params ? chosen.params.toLocaleString('en-GB') : '—'}</dd></div>
            <div><dt>F1 (unique / full)</dt><dd>{fmtOr(chosen.metrics.discovery_unique['F1'], 3)} / {fmtOr(chosen.metrics.discovery_full['F1'], 3)}</dd></div>
            <div><dt>MAE (unique)</dt><dd>{fmtOr(chosen.metrics.discovery_unique['MAE'], 3)} eV/atom</dd></div>
            <div><dt>κ_SRME</dt><dd>{fmtOr(chosen.metrics.kappa_srme, 3)}</dd></div>
            <div><dt>RMSD</dt><dd>{fmtOr(chosen.metrics.rmsd, 4)} Å</dd></div>
            <div><dt>Licences</dt><dd>code {chosen.licence_code ?? '—'} · weights {chosen.licence_checkpoint ?? '—'}</dd></div>
          </dl>
          <p className="bx-links">
            {chosen.paper && <a href={chosen.paper}>paper</a>}
            {chosen.repo && <a href={chosen.repo}>code</a>}
            {chosen.checkpoint_url && <a href={chosen.checkpoint_url}>checkpoint</a>}
          </p>
          <p className="bx-src">
            Upstream <a href={`${source.repo}/tree/${source.commit}/models`}>{source.commit.slice(0, 10)}</a> · {source.commit_date} · fetched {source.fetched}
          </p>
        </aside>
      </div>
    </div>
  );
}

function fmtOr(v: number | null | undefined, d: number): string {
  return typeof v === 'number' ? v.toFixed(d) : '—';
}
