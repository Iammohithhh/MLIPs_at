import { useId, useMemo, useState } from 'react';
import './CutoffExplorer.css';
import { cosineCurvatureJump, cosineCutoff, polynomialEnvelope, type CutoffValue } from '~/lib/cutoffs';
import { useLinkedSymbol } from '~/lib/useLinkedSymbol';

// Value / slope / curvature of two smooth cutoffs, to show which derivatives are
// continuous at r = rc. Linked symbols: rc (the cutoff line), fc (cosine curve),
// u (polynomial envelope curve), p (envelope order).

const W = 680;
const PANEL_H = 112;
const GAP = 26;
const M = { l: 64, r: 18, t: 14, b: 30 };
const R_MAX = 8;
const N = 480;

type Key = 'f' | 'df' | 'd2f';
const PANELS: Array<{ key: Key; label: string; units: string }> = [
  { key: 'f', label: 'f', units: '1' },
  { key: 'df', label: 'df/dr', units: 'Å⁻¹' },
  { key: 'd2f', label: 'd²f/dr²', units: 'Å⁻²' },
];

function niceRange(values: number[]): [number, number] {
  let lo = Math.min(0, ...values);
  let hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.08 || 0.1;
  lo -= lo < 0 ? pad : 0;
  hi += pad;
  return [lo, hi];
}

export default function CutoffExplorer() {
  const [rc, setRc] = useState(5);
  const [p, setP] = useState(6);
  const [active, setSym] = useLinkedSymbol('cutoff-explorer');
  const ids = useId();

  const x = (r: number) => M.l + (r / R_MAX) * (W - M.l - M.r);
  const rs = useMemo(() => Array.from({ length: N + 1 }, (_, i) => (i / N) * R_MAX), []);

  const data = useMemo(() => {
    const cos = rs.map((r) => cosineCutoff(r, rc));
    const poly = rs.map((r) => polynomialEnvelope(r, rc, p));
    return { cos, poly };
  }, [rs, rc, p]);

  const jump = cosineCurvatureJump(rc);
  const height = M.t + PANELS.length * PANEL_H + (PANELS.length - 1) * GAP + M.b;

  function path(vals: CutoffValue[], key: Key, y: (v: number) => number, split: boolean) {
    // Split the polyline at rc so a discontinuity is drawn as a gap, not a cliff.
    let d = '';
    let pen = false;
    rs.forEach((r, i) => {
      const v = vals[i]![key];
      const startNew = !pen || (split && r >= rc && rs[i - 1]! < rc);
      d += `${startNew ? 'M' : 'L'}${x(r).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  }

  return (
    <div className="cutoff-explorer">
      <div className="ce-controls">
        <label htmlFor={`${ids}-rc`} data-sym="rc" onPointerEnter={() => setSym('rc')} onPointerLeave={() => setSym(null)}>
          <span>
            cutoff <i>r</i>
            <sub>c</sub>
          </span>
          <input
            id={`${ids}-rc`}
            type="range"
            min={2.5}
            max={7.5}
            step={0.1}
            value={rc}
            onChange={(e) => setRc(Number(e.target.value))}
          />
          <output>{rc.toFixed(1)} Å</output>
        </label>
        <label htmlFor={`${ids}-p`} onPointerEnter={() => setSym('p')} onPointerLeave={() => setSym(null)}>
          <span>
            envelope order <i>p</i>
          </span>
          <input
            id={`${ids}-p`}
            type="range"
            min={2}
            max={10}
            step={1}
            value={p}
            onChange={(e) => setP(Number(e.target.value))}
          />
          <output>{p}</output>
        </label>
      </div>

      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-labelledby={`${ids}-desc`} className="ce-plot">
        <desc id={`${ids}-desc`}>
          Three stacked panels plot the cosine cutoff and the polynomial envelope against distance r from 0 to {R_MAX} Å
          with cutoff r_c = {rc.toFixed(1)} Å: their values, first derivatives and second derivatives. The cosine
          cutoff's second derivative jumps by {jump.toFixed(3)} Å⁻² at r_c; the envelope's does not.
        </desc>
        {PANELS.map((panel, k) => {
          const top = M.t + k * (PANEL_H + GAP);
          const [lo, hi] =
            panel.key === 'f'
              ? [0, 1.08]
              : niceRange([...data.cos.map((v) => v[panel.key]), ...data.poly.map((v) => v[panel.key])]);
          const y = (v: number) => top + PANEL_H - ((v - lo) / (hi - lo)) * PANEL_H;
          const cosBelow = cosineCutoff(rc - 1e-9, rc)[panel.key];
          return (
            <g key={panel.key}>
              <rect x={M.l} y={top} width={W - M.l - M.r} height={PANEL_H} className="ce-frame" />
              <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="ce-zero" />
              <text x={M.l - 8} y={top + 12} className="ce-axis" textAnchor="end">
                {panel.label}
              </text>
              <text x={M.l - 8} y={top + 27} className="ce-units" textAnchor="end">
                [{panel.units}]
              </text>
              <text x={W - M.r - 6} y={y(0) - top < 16 ? y(0) + 14 : y(0) - 5} className="ce-tick" textAnchor="end">
                0
              </text>
              {panel.key === 'f' && (
                <line x1={M.l} x2={W - M.r} y1={y(1)} y2={y(1)} className="ce-zero ce-one" />
              )}
              <path
                d={path(data.poly, panel.key, y, false)}
                className={`ce-curve ce-poly${active === 'u' || active === 'p' ? ' is-active' : ''}`}
                onPointerEnter={() => setSym('u')}
                onPointerLeave={() => setSym(null)}
              />
              <path
                d={path(data.cos, panel.key, y, true)}
                className={`ce-curve ce-cos${active === 'fc' ? ' is-active' : ''}`}
                onPointerEnter={() => setSym('fc')}
                onPointerLeave={() => setSym(null)}
              />
              {panel.key === 'd2f' && (
                <g className="ce-jump">
                  <circle cx={x(rc)} cy={y(cosBelow)} r={4} className="open" />
                  <circle cx={x(rc)} cy={y(0)} r={4} className="closed" />
                  <text x={x(rc) + 10} y={y(cosBelow / 2) + 4}>
                    jump π²/(2r<tspan dy="3" fontSize="0.8em">c</tspan>
                    <tspan dy="-3">²) = {jump.toFixed(3)} Å⁻²</tspan>
                  </text>
                </g>
              )}
              {k === PANELS.length - 1 &&
                [0, 2, 4, 6].map((r) => (
                  <text key={r} x={x(r)} y={top + PANEL_H + 18} className="ce-tick" textAnchor="middle">
                    {r}
                  </text>
                ))}
            </g>
          );
        })}
        <line
          x1={x(rc)}
          x2={x(rc)}
          y1={M.t - 6}
          y2={height - M.b + 4}
          className={`ce-rc${active === 'rc' ? ' is-active' : ''}`}
          onPointerEnter={() => setSym('rc')}
          onPointerLeave={() => setSym(null)}
        />
        <text x={x(rc) + 6} y={M.t + 4} className={`ce-rc-label${active === 'rc' ? ' is-active' : ''}`}>
          r<tspan dy="4" fontSize="0.75em">c</tspan>
        </text>
        <text x={x(7)} y={height - M.b + 18} className="ce-tick" textAnchor="middle">
          r [Å]
        </text>
      </svg>

      <ul className="ce-legend">
        <li className="cos" onPointerEnter={() => setSym('fc')} onPointerLeave={() => setSym(null)}>
          cosine cutoff — <em>f</em> and <em>f′</em> continuous, <em>f″</em> jumps
        </li>
        <li className="poly" onPointerEnter={() => setSym('u')} onPointerLeave={() => setSym(null)}>
          polynomial envelope (p = {p}) — continuous through <em>f″</em>
        </li>
      </ul>
    </div>
  );
}
