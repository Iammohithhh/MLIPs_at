import { useMemo, useState } from 'react';
import './PhysicsLab.css';
import data from '~/data/benchmarks/migration-bheemaguli2025.json';
import { useSceneTheme } from '~/three/theme';
import HopViewer, { useHopPath } from './HopViewer';

// REPRODUCED: per-system DFT-NEB barriers (Zenodo Dataset-2) joined with each foundation model's NEB
// barrier from the authors' repository, snapshotted by pipeline/atlas_pipeline/migration_barriers.py.
// Summary numbers are recomputed here from that snapshot, not copied from the paper.

type Sys = (typeof data.systems)[number];
const MODELS = data.models;
const MAX = 2.6;

export default function BarrierParity() {
  const theme = useSceneTheme();
  const [model, setModel] = useState(MODELS[3]!);
  const [thr, setThr] = useState(0.5);
  const [hover, setHover] = useState<Sys | null>(null);
  const [selected, setSelected] = useState<string>(() => {
    // Start on the system with the largest CHGNet error, the paper's most underestimating model family.
    let best = data.systems[0]!;
    for (const s of data.systems) {
      const m = s.models.CHGNet;
      const b = best.models.CHGNet;
      if (m != null && b != null && Math.abs(m - s.dft) > Math.abs(b - best.dft) && m >= 0 && m <= MAX) best = s;
    }
    return best.id;
  });
  const { path, error } = useHopPath(selected);
  const sel = data.systems.find((s) => s.id === selected)!;
  const pts = useMemo(
    () => data.systems.map((s) => ({ s, y: (s.models as Record<string, number | null>)[model] })).filter((p): p is { s: Sys; y: number } => p.y != null),
    [model],
  );
  const stats = useMemo(() => {
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    let off = 0;
    for (const p of pts) {
      const good = p.s.dft < thr;
      const pred = p.y < thr;
      if (good && pred) tp++;
      else if (!good && !pred) tn++;
      else if (pred) fp++;
      else fn++;
      if (p.y > MAX || p.y < 0) off++;
    }
    return { tp, tn, fp, fn, acc: (tp + tn) / pts.length, off };
  }, [pts, thr]);
  const sm = (data.summary as Record<string, { n: number; mae: number; under: number; over: number }>)[model]!;

  const W = 520;
  const H = 470;
  const M = { l: 56, r: 16, t: 24, b: 44 };
  const X = (v: number) => M.l + (Math.min(Math.max(v, 0), MAX) / MAX) * (W - M.l - M.r);
  const Y = (v: number) => H - M.b - (Math.min(Math.max(v, 0), MAX) / MAX) * (H - M.t - M.b);
  const ticks = [0, 0.5, 1, 1.5, 2, 2.5];

  return (
    <div className="lab">
      <div className="lab-toolbar" style={{ position: 'static', marginBottom: '0.6rem' }}>
        <div className="seg" role="group" aria-label="Model">
          {MODELS.map((m) => (
            <button key={m} type="button" aria-pressed={model === m} onClick={() => setModel(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="lab-split lab-split--wide">
        <figure className="lab-plot" style={{ margin: 0 }}>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Parity plot of ${model} migration barriers against DFT-NEB for ${pts.length} systems.`}>
            <text x={M.l} y={M.t - 9} className="axis-label">{model} barrier [eV]</text>
            <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
            <rect x={M.l} y={Y(thr)} width={X(thr) - M.l} height={H - M.b - Y(thr)} style={{ fill: theme.phys, opacity: 0.08 }} />
            <rect x={X(thr)} y={M.t} width={W - M.r - X(thr)} height={Y(thr) - M.t} style={{ fill: theme.phys, opacity: 0.08 }} />
            {ticks.map((v) => (
              <g key={v}>
                <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className="grid" />
                <line x1={X(v)} x2={X(v)} y1={M.t} y2={H - M.b} className="grid" />
                <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
                <text x={X(v)} y={H - M.b + 16} textAnchor="middle" className="tick">{v}</text>
              </g>
            ))}
            <line x1={X(0)} y1={Y(0)} x2={X(MAX)} y2={Y(MAX)} className="axis" />
            <line x1={X(thr)} x2={X(thr)} y1={M.t} y2={H - M.b} className="rc-line" />
            <line x1={M.l} x2={W - M.r} y1={Y(thr)} y2={Y(thr)} className="rc-line" />
            {pts.map((p) => {
              const under = p.y < p.s.dft;
              return (
                <circle
                  key={p.s.id}
                  cx={X(p.s.dft)}
                  cy={Y(p.y)}
                  r={hover?.id === p.s.id || selected === p.s.id ? 5.5 : 3}
                  stroke={selected === p.s.id ? theme.ink : 'none'}
                  strokeWidth={2}
                  onMouseEnter={() => setHover(p.s)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => setSelected(p.s.id)}
                  style={{ cursor: 'pointer', fill: under ? theme.caveat : theme.arch, opacity: hover && hover.id !== p.s.id ? 0.35 : 0.7 }}
                />
              );
            })}
            <text x={X(0.05)} y={Y(MAX) + 14} className="tick">over-predicted</text>
            <text x={X(MAX) - 4} y={Y(0.1)} textAnchor="end" className="tick">under-predicted</text>
            <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">DFT-NEB barrier [eV]</text>
          </svg>
          <figcaption className="lab-legend">
            <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> model below DFT ({sm.under})</span>
            <span className="item"><span className="swatch" style={{ background: theme.arch }} /> model above DFT ({sm.over})</span>
            <span className="item"><span className="swatch" style={{ background: theme.phys, opacity: 0.3 }} /> classified the same as DFT</span>
          </figcaption>
        </figure>
        <div className="lab-stage">
          {path ? (
            <HopViewer path={path} label={`${sel.formula}: the migrating ion and its hop of ${path.hop.toFixed(2)} Å between the NEB end points.`} />
          ) : (
            <p className="lab-note" style={{ padding: '1rem' }}>{error ? 'The structure file could not be loaded.' : 'Loading the structure…'}</p>
          )}
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: theme.caveat }} /> migrating {path ? path.elements[path.mover] : ''} ion and its end point</li>
            <li>{sel.formula} · {sel.class ?? '—'}</li>
          </ul>
        </div>
      </div>
      <div className="lab-panel">
        <div style={{ display: 'contents' }}>
          <div className="lab-controls">
            <label>
              <span>“good conductor” threshold</span>
              <input type="range" min={0.2} max={1.2} step={0.05} value={thr} onChange={(e) => setThr(Number(e.target.value))} />
              <output>{thr.toFixed(2)} eV</output>
            </label>
          </div>
          <dl className="lab-readout" aria-live="polite">
            <div>
              <dt>Systems with a released barrier</dt>
              <dd>{sm.n} of {data.systems.length}</dd>
            </div>
            <div>
              <dt>MAE recomputed from the data</dt>
              <dd>{sm.mae.toFixed(3)} eV</dd>
            </div>
            <div>
              <dt>Same good/bad call as DFT</dt>
              <dd>{(100 * stats.acc).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Good conductors missed / false alarms</dt>
              <dd>{stats.fn} / {stats.fp}</dd>
            </div>
          </dl>
          <p className="lab-note" aria-live="polite">
            <b>{sel.formula}</b> ({sel.class ?? 'unclassified'}): DFT {sel.dft.toFixed(2)} eV;{' '}
            {MODELS.map((m) => `${m} ${(sel.models as Record<string, number | null>)[m]?.toFixed(2) ?? '—'}`).join(' · ')} eV. Click a point to show another path.
            <br />
            {hover
              ? `${hover.formula} (${hover.class ?? 'unclassified'}) — DFT ${hover.dft.toFixed(2)} eV, ${model} ${((hover.models as Record<string, number | null>)[model] ?? NaN).toFixed(2)} eV.`
              : `Each point is one literature migration path. ${stats.off > 0 ? `${stats.off} predictions outside 0–${MAX} eV are pinned to the edge. ` : ''}Hover a point for the material.`}
          </p>
        </div>
      </div>
    </div>
  );
}
