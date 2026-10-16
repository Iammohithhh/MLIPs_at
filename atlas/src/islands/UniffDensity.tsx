import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { useSceneTheme } from '~/three/theme';
import { histogram, median } from '~/lib/physics/reference-gap';

// PUBLISHED RESULT, re-plotted: signed density errors (simulated − experimental) of every run in the
// UniFFBench result files. Each model has its own runs; the two are never merged into one number.

export interface UniffSeries {
  name: string;
  loader: string;
  runs: number;
  /** Signed density errors (%), runs with |error| ≤ 100% only. */
  errors: number[];
}

const LO = -30;
const HI = 15;
const BIN = 1;

export default function UniffDensity({ series, floor }: { series: UniffSeries[]; floor: number }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [tol, setTol] = useState(2.5);
  const colors = [theme.arch, theme.caveat];
  const hists = useMemo(() => series.map((s) => histogram(s.errors, LO, HI, BIN)), [series]);
  const stats = useMemo(
    () =>
      series.map((s) => ({
        median: median(s.errors),
        mape: s.errors.reduce((a, x) => a + Math.abs(x), 0) / s.errors.length,
        under: s.errors.filter((x) => x < 0).length / s.errors.length,
      })),
    [series],
  );
  const within = series.map((s) => s.errors.filter((x) => Math.abs(x) <= tol).length / s.errors.length);
  const belowFloor = series.map((s) => s.errors.filter((x) => x <= floor).length / s.errors.length);

  const W = 760;
  const H = 360;
  const M = { l: 56, r: 18, t: 30, b: 46 };
  const ymax = Math.max(...hists.flatMap((h) => h.counts));
  const yTop = Math.ceil((ymax * 1.08) / 50) * 50;
  const X = (v: number) => M.l + ((v - LO) / (HI - LO)) * (W - M.l - M.r);
  const Y = (c: number) => H - M.b - (c / yTop) * (H - M.t - M.b);
  const step = (counts: number[]) => {
    let d = `M${X(LO)},${Y(0)}`;
    counts.forEach((c, i) => {
      d += `L${X(LO + i * BIN)},${Y(c)}L${X(LO + (i + 1) * BIN)},${Y(c)}`;
    });
    return d + `L${X(HI)},${Y(0)}`;
  };
  const xt = Array.from({ length: (HI - LO) / 5 + 1 }, (_, i) => LO + i * 5);
  const yt = Array.from({ length: yTop / 100 + 1 }, (_, i) => i * 100);
  const pct = (x: number) => `${(100 * x).toFixed(0)}%`;
  const s2 = (x: number) => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(1)}`;

  return (
    <div className="lab">
      <figure className="lab-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histograms of signed density errors against experiment: ${series.map((s, i) => `${s.name} median ${stats[i]!.median.toFixed(1)}%`).join(', ')}.`}>
          <text x={M.l} y={M.t - 12} className="axis-label">runs per 1% bin</text>
          <rect x={X(-tol)} y={M.t} width={X(tol) - X(-tol)} height={H - M.t - M.b} style={{ fill: theme.verified, opacity: 0.12 }} />
          <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
          {yt.map((v) => (
            <g key={v}>
              <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className={v === 0 ? 'axis' : 'grid'} />
              <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
            </g>
          ))}
          {xt.map((v) => (
            <g key={v}>
              <line x1={X(v)} x2={X(v)} y1={H - M.b} y2={H - M.b + 4} className="axis" />
              <text x={X(v)} y={H - M.b + 17} textAnchor="middle" className="tick">{v > 0 ? `+${v}` : v}</text>
            </g>
          ))}
          <line x1={X(0)} x2={X(0)} y1={M.t} y2={H - M.b} className="axis" />
          <line x1={X(floor)} x2={X(floor)} y1={M.t} y2={H - M.b} style={{ stroke: theme.lit, strokeWidth: 2, strokeDasharray: '5 4' }} />
          <text x={X(floor) + 6} y={M.t + 14} textAnchor="start" className="tick" style={{ fill: theme.lit, fontWeight: 600 }}>
            PBE copied exactly: {s2(floor)}%
          </text>
          <text x={X(tol) + 5} y={M.t + 32} className="tick" style={{ fill: theme.verified }}>±{tol}%</text>
          {hists.map((h, i) => (
            <path key={series[i]!.name} d={step(h.counts)} style={{ fill: colors[i], fillOpacity: 0.16, stroke: colors[i], strokeWidth: 2 }} />
          ))}
          {stats.map((s, i) => (
            <g key={series[i]!.name}>
              <line x1={X(s.median)} x2={X(s.median)} y1={M.t} y2={H - M.b} style={{ stroke: colors[i], strokeWidth: 1.5 }} />
              <text x={X(s.median) - 6} y={M.t + 34 + 16 * i} textAnchor="end" className="tick" style={{ fill: colors[i], fontWeight: 600 }}>
                median {series[i]!.name} {s2(s.median)}%
              </text>
            </g>
          ))}
          <text x={X(LO) + 6} y={M.t + 16} className="tick">← less dense than measured</text>
          <text x={X(HI) - 6} y={M.t + 52} textAnchor="end" className="tick">denser →</text>
          <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">(ρ_sim − ρ_exp) / ρ_exp [%]</text>
        </svg>
        <figcaption className="lab-legend">
          {series.map((s, i) => (
            <span className="item" key={s.name}>
              <span className="swatch" style={{ background: colors[i] }} /> {s.name}: {s.errors.length.toLocaleString('en-US')} runs
              {hists[i]!.below + hists[i]!.above > 0 ? ` (${hists[i]!.below} below ${LO}%, ${hists[i]!.above} above +${HI}% not drawn)` : ''}
            </span>
          ))}
        </figcaption>
      </figure>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-tol`}>
            <span>tolerance band</span>
            <input id={`${ids}-tol`} type="range" min={0.5} max={10} step={0.5} value={tol} onChange={(e) => setTol(Number(e.target.value))} />
            <output>±{tol.toFixed(1)}%</output>
          </label>
        </div>
        <div style={{ overflowX: 'auto', gridColumn: '1 / -1' }}>
          <table className="lab-table" aria-live="polite">
            <thead>
              <tr>
                <th scope="col">Result file</th>
                <th scope="col">median</th>
                <th scope="col">MAPE</th>
                <th scope="col">less dense than measured</th>
                <th scope="col">within ±{tol}%</th>
                <th scope="col">at or below the PBE line</th>
              </tr>
            </thead>
            <tbody>
              {series.map((s, i) => (
                <tr key={s.name}>
                  <th scope="row">
                    <span className="swatch" style={{ background: colors[i] }} aria-hidden="true" /> {s.name}
                  </th>
                  <td>{s2(stats[i]!.median)}%</td>
                  <td>{stats[i]!.mape.toFixed(1)}%</td>
                  <td>{pct(stats[i]!.under)}</td>
                  <td>{pct(within[i]!)}</td>
                  <td>{pct(belowFloor[i]!)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Errors are not centred on zero: in both files most minerals come out less dense than measured, the sign a
          PBE-like reference would give. The dashed line is where a perfect copy of PBE would sit on average; it is a mean
          over simple solids, not a prediction for each mineral. These are the authors’ result files at one commit, not the
          paper’s figures, and the two models are shown side by side, not ranked.
        </p>
      </div>
    </div>
  );
}
