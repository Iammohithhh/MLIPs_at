import { useEffect, useId, useMemo, useRef, useState } from 'react';
import './PhysicsLab.css';
import { useSceneTheme } from '~/three/theme';
import { basisGrid, besselRoot } from '~/lib/models/three-body';
import { hexToLinearRgb } from '~/lib/structure/elements';

// M3GNet's three-body basis j_l(z_ln r_ik / r_c) Y_l^0(θ_jik) f_c(r_ik) on the (r_ik, θ)
// plane. l = 0 ignores the angle; l ≥ 1 is what lets a single block see bond angles.

const RC = 4;
const NR = 120;
const NT = 120;
const RS = Array.from({ length: NR }, (_, k) => ((k + 0.5) / NR) * RC);
const TS = Array.from({ length: NT }, (_, k) => ((k + 0.5) / NT) * Math.PI);

function toSrgb(c: number) {
  return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
}

export default function ThreeBodyBasis() {
  const theme = useSceneTheme();
  const ids = useId();
  const [l, setL] = useState(1);
  const [n, setN] = useState(1);
  const grid = useMemo(() => basisGrid(l, n, RC, RS, TS), [l, n]);
  const vmax = useMemo(() => Math.max(...grid.flat().map(Math.abs), 1e-12), [grid]);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvas.current;
    const ctx = cv?.getContext('2d');
    if (!cv || !ctx) return;
    const img = ctx.createImageData(NT, NR);
    const pos = hexToLinearRgb(theme.arch);
    const neg = hexToLinearRgb(theme.fail);
    const bg = hexToLinearRgb(theme.surface);
    for (let a = 0; a < NR; a++)
      for (let b = 0; b < NT; b++) {
        const v = grid[a]![b]! / vmax;
        const c = v >= 0 ? pos : neg;
        const t = Math.min(1, Math.abs(v)) ** 0.8;
        // Row 0 of the image is the largest r (top of the plot).
        const o = 4 * ((NR - 1 - a) * NT + b);
        img.data[o] = toSrgb(bg[0] + (c[0] - bg[0]) * t);
        img.data[o + 1] = toSrgb(bg[1] + (c[1] - bg[1]) * t);
        img.data[o + 2] = toSrgb(bg[2] + (c[2] - bg[2]) * t);
        img.data[o + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  }, [grid, vmax, theme]);

  const z = besselRoot(l, n);
  const W = 520;
  const H = 360;
  const M = { l: 52, r: 16, t: 20, b: 44 };
  const pw = W - M.l - M.r;
  const ph = H - M.t - M.b;

  return (
    <div className="lab">
      <div className="lab-plot">
        <div style={{ position: 'relative', width: '100%', maxWidth: 640, margin: '0 auto' }}>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Basis function with l = ${l}, n = ${n} over bond length r_ik from 0 to ${RC} Å and angle θ from 0 to 180°. ${l === 0 ? 'It does not change with the angle.' : `It changes sign ${l} time${l > 1 ? 's' : ''} across the angle range.`}`}>
            <foreignObject x={M.l} y={M.t} width={pw} height={ph}>
              <canvas ref={canvas} width={NT} height={NR} style={{ width: '100%', height: '100%', imageRendering: 'auto', display: 'block' }} />
            </foreignObject>
            <rect x={M.l} y={M.t} width={pw} height={ph} style={{ fill: 'none', stroke: theme.ruleStrong }} />
            {[0, 45, 90, 109.47, 135, 180].map((d) => (
              <text key={d} x={M.l + (d / 180) * pw} y={H - M.b + 16} textAnchor="middle" className="tick">
                {d === 109.47 ? '109.5' : d}°
              </text>
            ))}
            {[0, 1, 2, 3, 4].map((r) => (
              <text key={r} x={M.l - 6} y={M.t + (1 - r / RC) * ph + 4} textAnchor="end" className="tick">
                {r} Å
              </text>
            ))}
            <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">bond angle θ_jik</text>
            <text x={M.l} y={M.t - 6} className="axis-label">r_ik (three-body cutoff {RC} Å at the top edge)</text>
          </svg>
        </div>
        <figcaption className="lab-legend">
          <span className="item"><span className="swatch" style={{ background: theme.arch }} /> positive</span>
          <span className="item"><span className="swatch" style={{ background: theme.fail }} /> negative</span>
          <span className="item">colour scaled to this function’s largest |value|</span>
        </figcaption>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-l`}>
            <span>angular order l</span>
            <input id={`${ids}-l`} type="range" min={0} max={3} step={1} value={l} onChange={(e) => setL(Number(e.target.value))} />
            <output>{l}</output>
          </label>
          <label htmlFor={`${ids}-n`}>
            <span>radial index n</span>
            <input id={`${ids}-n`} type="range" min={1} max={3} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} />
            <output>{n}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>root z<sub>ln</sub> of j<sub>l</sub></dt>
            <dd>{z.toFixed(4)}</dd>
          </div>
          <div className={l === 0 ? 'is-alert' : 'is-good'}>
            <dt>depends on the angle</dt>
            <dd>{l === 0 ? 'no' : 'yes'}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Each term of M3GNet’s three-body sum is weighted by one of these functions. The <i>l</i> = 0 row behaves like a
          SchNet filter — distance only. From <i>l</i> = 1 the weight changes with the angle between bonds <i>ij</i> and
          <i> ik</i>, so one block already separates the angles that a distance-only block cannot. Using the root
          <i> z</i><sub>ln</sub> makes every radial factor vanish at the three-body cutoff; the polynomial cutoff
          then smooths its derivatives too.
        </p>
      </div>
    </div>
  );
}
