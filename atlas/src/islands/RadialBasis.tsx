import { useId, useState } from 'react';
import './PhysicsLab.css';
import { bessel, radial } from '~/lib/models/mace-features';
import { useSceneTheme } from '~/three/theme';

// MACE radial basis: 8 Bessel functions × polynomial envelope (p = 5), plotted as
// r·R_n(r) so the oscillation is visible near r = 0.

export default function RadialBasis() {
  const theme = useSceneTheme();
  const ids = useId();
  const [rc, setRc] = useState(5);
  const [focus, setFocus] = useState(3);
  const [envelope, setEnvelope] = useState(true);
  const W = 640;
  const H = 260;
  const M = { l: 46, r: 14, t: 16, b: 34 };
  const rs = Array.from({ length: 300 }, (_, k) => 0.01 + (k * 6.5) / 299);
  const f = (n: number, r: number) => r * (envelope ? radial(n, r, rc) : bessel(n, r, rc));
  const lim = Math.sqrt(2 / rc) * 1.08;
  const x = (r: number) => M.l + (r / 6.5) * (W - M.l - M.r);
  const y = (v: number) => M.t + ((lim - v) / (2 * lim)) * (H - M.t - M.b);
  return (
    <div className="lab rb">
      <figure className="lab-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Eight Bessel radial functions of MACE times the polynomial envelope, cutoff ${rc} Å.`}>
          <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
          <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="axis" />
          <line x1={x(rc)} x2={x(rc)} y1={M.t} y2={H - M.b} className="rc-line" />
          <text x={x(rc) + 4} y={M.t + 12} className="rc-label">r<tspan dy="4" fontSize="0.75em">c</tspan></text>
          {Array.from({ length: 8 }, (_, k) => k + 1).map((n) => (
            <path
              key={n}
              d={rs.map((r, k) => `${k ? 'L' : 'M'}${x(r).toFixed(1)},${y(f(n, r)).toFixed(1)}`).join('')}
              className="curve"
              style={{ stroke: n === focus ? theme.arch : theme.ruleStrong, strokeWidth: n === focus ? 2.8 : 1.2 }}
              onPointerEnter={() => setFocus(n)}
            />
          ))}
          {[0, 1, 2, 3, 4, 5, 6].map((r) => (
            <text key={r} x={x(r)} y={H - M.b + 16} textAnchor="middle" className="tick">{r}</text>
          ))}
          <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">r [Å]</text>
          <text x={8} y={M.t + 4} className="axis-label">r·Rₙ(r)</text>
        </svg>
      </figure>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-n`}>
            <span>highlight n</span>
            <input id={`${ids}-n`} type="range" min={1} max={8} step={1} value={focus} onChange={(e) => setFocus(Number(e.target.value))} />
            <output>{focus}</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>cutoff r_c</span>
            <input id={`${ids}-rc`} type="range" min={3} max={6} step={0.1} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
          <label className="rb-check">
            <input type="checkbox" checked={envelope} onChange={(e) => setEnvelope(e.target.checked)} /> apply the polynomial envelope (p = 5)
          </label>
        </div>
        <p className="lab-note">
          The n-th function has n − 1 nodes inside the cutoff: low n describes the neighbour shell coarsely, high n finely.
          Without the envelope every function reaches zero at r<sub>c</sub> but with a non-zero slope (±nπ/r<sub>c</sub>²
          for sin(nπr/r<sub>c</sub>)/r); the envelope brings value, slope and curvature to zero, so forces stay continuous
          as atoms cross the cutoff.
        </p>
      </div>
    </div>
  );
}
