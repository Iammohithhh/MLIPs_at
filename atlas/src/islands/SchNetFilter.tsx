import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { useSceneTheme } from '~/three/theme';
import { cosineCutoff, filterAt, makeToySchNet, rbfCentres, rbfExpand } from '~/lib/models/schnet';

// The filter-generating network: a distance is expanded in Gaussians on a 0.1 Å grid
// (γ = 10 Å⁻²) and two dense layers turn that expansion into one filter value per
// feature channel. Weights are random (untrained); the shapes show what the network
// can represent: smooth, continuous functions of distance that vanish at the cutoff.

const RC = 5;
const SHOWN = [0, 3, 7, 12];
const NS = Array.from({ length: 251 }, (_, k) => (k / 250) * RC);

export default function SchNetFilter() {
  const theme = useSceneTheme();
  const ids = useId();
  const [r, setR] = useState(2.35);
  const model = useMemo(() => makeToySchNet({ rc: RC, blocks: 1, features: 16 }), []);
  const mu = useMemo(() => rbfCentres(model.rbf), [model]);
  const e = useMemo(() => rbfExpand(r, model.rbf), [r, model]);
  const curves = useMemo(() => NS.map((d) => filterAt(model, 0, d)), [model]);
  const now = useMemo(() => filterAt(model, 0, r), [model, r]);
  const W = 900;
  const M = { l: 44, r: 14 };
  const x = (d: number) => M.l + (d / RC) * (W - M.l - M.r);
  const H1 = 170;
  const H2 = 240;
  const yMax = Math.max(...curves.flatMap((c) => SHOWN.map((f) => Math.abs(c[f]!)))) * 1.1;
  const y2 = (v: number) => 20 + (1 - (v + yMax) / (2 * yMax)) * (H2 - 50);
  const palette = [theme.arch, theme.phys, theme.lit, theme.caveat];
  const cmax = Math.max(...Array.from(now, Math.abs), 1e-9);
  const sigma = 1 / Math.sqrt(2 * model.rbf.gamma);

  return (
    <div className="lab">
      <div className="lab-plot">
        <svg viewBox={`0 0 ${W} ${H1}`} role="img" aria-label={`${mu.length} Gaussians centred every 0.1 Å from 0 to ${RC} Å; at r = ${r.toFixed(2)} Å the largest activations are on the nearest centres.`}>
          <text x={M.l} y={14} className="axis-label">step 1 · Gaussian expansion e_k(r) = exp(−γ (r − μ_k)²)</text>
          {mu.map((m, k) => (
            <path
              key={k}
              d={NS.map((d, n) => `${n ? 'L' : 'M'}${x(d)},${H1 - 30 - Math.exp(-model.rbf.gamma * (d - m) ** 2) * 80}`).join('')}
              style={{ fill: 'none', stroke: theme.ruleStrong, strokeWidth: 0.8 }}
            />
          ))}
          {mu.map((m, k) =>
            e[k]! > 0.01 ? <line key={k} x1={x(m)} x2={x(m)} y1={H1 - 30} y2={H1 - 30 - e[k]! * 80} style={{ stroke: theme.arch, strokeWidth: 3 }} /> : null,
          )}
          <line x1={x(r)} x2={x(r)} y1={22} y2={H1 - 26} style={{ stroke: theme.ink, strokeWidth: 1 }} strokeDasharray="3 3" />
          <line x1={M.l} x2={W - M.r} y1={H1 - 30} y2={H1 - 30} className="axis" />
          {[0, 1, 2, 3, 4, 5].map((d) => (
            <text key={d} x={x(d)} y={H1 - 14} textAnchor="middle" className="tick">
              {d} Å
            </text>
          ))}
        </svg>
        <svg viewBox={`0 0 ${W} ${H2}`} role="img" aria-label="Four of the sixteen filter channels of a random SchNet filter network as functions of distance, multiplied by the cosine cutoff.">
          <text x={M.l} y={12} className="axis-label">step 2 · filter W_f(r) · f_c(r) from two dense layers (4 of 16 channels)</text>
          <line x1={M.l} x2={W - M.r} y1={y2(0)} y2={y2(0)} className="axis" />
          {SHOWN.map((f, n) => (
            <path key={f} d={curves.map((c, k) => `${k ? 'L' : 'M'}${x(NS[k]!)},${y2(c[f]!)}`).join('')} className="curve" style={{ stroke: palette[n] }} />
          ))}
          {SHOWN.map((f, n) => (
            <circle key={f} cx={x(r)} cy={y2(now[f]!)} r={4.5} style={{ fill: palette[n] }} />
          ))}
          <line x1={x(r)} x2={x(r)} y1={20} y2={H2 - 30} style={{ stroke: theme.ink, strokeWidth: 1 }} strokeDasharray="3 3" />
          <text x={x(RC) - 4} y={y2(0) - 6} textAnchor="end" className="tick">r_c</text>
          {[0, 1, 2, 3, 4, 5].map((d) => (
            <text key={d} x={x(d)} y={H2 - 12} textAnchor="middle" className="tick">
              {d} Å
            </text>
          ))}
        </svg>
        <div className="filter-strip" aria-label={`All 16 filter values at r = ${r.toFixed(2)} Å`} role="img">
          {Array.from(now, (v, f) => (
            <span
              key={f}
              title={`channel ${f + 1}: ${v.toFixed(3)}`}
              style={{ background: v >= 0 ? theme.arch : theme.fail, opacity: 0.15 + 0.85 * (Math.abs(v) / cmax) }}
            />
          ))}
        </div>
        <figcaption className="lab-legend">
          <span className="item">step 3 · all 16 channels at this r multiply neighbour j’s features element-wise (blue +, red −)</span>
        </figcaption>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-r`}>
            <span>distance r<sub>ij</sub></span>
            <input id={`${ids}-r`} type="range" min={0} max={RC} step={0.01} value={r} onChange={(ev) => setR(Number(ev.target.value))} />
            <output>{r.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Gaussians up to r<sub>c</sub></dt>
            <dd>{mu.length}</dd>
          </div>
          <div>
            <dt>width σ = 1/√(2γ)</dt>
            <dd>{sigma.toFixed(2)} Å</dd>
          </div>
          <div>
            <dt>cutoff weight f<sub>c</sub>(r)</dt>
            <dd>{cosineCutoff(r, RC).toFixed(3)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Each Gaussian is about twice as wide as the 0.1 Å grid, so neighbouring ones overlap and the expansion changes
          smoothly with <i>r</i>. That smoothness is the point of a continuous filter: a filter on a discrete grid would
          jump when an atom moved between grid cells, making the energy discontinuous.
        </p>
      </div>
      <style>{`
        .filter-strip { display: grid; grid-template-columns: repeat(16, 1fr); gap: 3px; margin: 0.2rem 0 0 4.9%; width: 93.6%; }
        .filter-strip span { height: 18px; border-radius: 3px; }
      `}</style>
    </div>
  );
}
