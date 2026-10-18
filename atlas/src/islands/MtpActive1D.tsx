import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Mtp.css';
import { useSceneTheme } from '~/three/theme';
import { barrier1d, boltzmannFit, eqm1d, grade1d, interpolate1d, model1d, rms1d } from '~/lib/models/active-learning';

// Podryabinkin & Shapeev (2017), Sec. 4.1, recomputed (src/lib/models/active-learning.ts): a two-parameter
// model fitted to Boltzmann-weighted samples (passive) or through two chosen configurations (active), and the
// extrapolation grade γ(x) of the active pair.

const XS = Array.from({ length: 401 }, (_, i) => -5 + i / 40);

export default function MtpActive1D() {
  const theme = useSceneTheme();
  const ids = useId();
  const [x1, setX1] = useState(-1);
  const [x2, setX2] = useState(1.5);
  const [kT, setKT] = useState(1);
  const passive = useMemo(() => boltzmannFit(kT), [kT]);
  const active = useMemo(() => interpolate1d(x1, x2), [x1, x2]);
  const rmsP = useMemo(() => rms1d(passive, kT), [passive, kT]);
  const rmsA = useMemo(() => (active ? rms1d(active, kT) : NaN), [active, kT]);
  const barP = useMemo(() => barrier1d(passive), [passive]);
  const barA = useMemo(() => (active ? barrier1d(active) : null), [active]);
  const escapes = (t: number[], b: { x: number; e: number }) => b.x > -3.99 && model1d(t, -4) < b.e;

  const W = 560;
  const x0 = 50;
  const w = W - x0 - 14;
  const top = 26;
  const h1 = 210;
  const gap = 46;
  const h2 = 120;
  const px = (x: number) => x0 + ((x + 5) / 10) * w;
  const eLo = -8;
  const eHi = 22;
  const pe = (e: number) => top + ((eHi - Math.max(eLo, Math.min(eHi, e))) / (eHi - eLo)) * h1;
  const g0 = top + h1 + gap;
  const gHi = 3;
  const gLo = -1;
  const pg = (g: number) => g0 + ((gHi - Math.max(gLo, Math.min(gHi, Math.log10(Math.max(g, 1e-3))))) / (gHi - gLo)) * h2;
  const path = (f: (x: number) => number, p: (v: number) => number) => XS.map((x, i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${p(f(x)).toFixed(1)}`).join('');

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <figure className="lab-plot">
          <svg viewBox={`0 0 ${W} ${g0 + h2 + 40}`} role="img" aria-label="Top: the reference energy, the passive fit and the active fit through two chosen points. Bottom: the extrapolation grade of the active pair on a log scale, below 1 between the points and above 1 outside them.">
            <text x={x0} y={top - 10} className="axis-label">energy E(x), in units of kT</text>
            <rect x={px(-4)} width={px(4) - px(-4)} y={top} height={h1} style={{ fill: theme.ruleStrong, opacity: 0.16 }} />
            {[-5, 0, 5, 10, 15, 20].map((e) => (
              <g key={e}>
                <line x1={x0} x2={x0 + w} y1={pe(e)} y2={pe(e)} className="grid" />
                <text x={x0 - 6} y={pe(e) + 4} textAnchor="end" className="tick">{e}</text>
              </g>
            ))}
            <path d={path(eqm1d, pe)} style={{ fill: 'none', stroke: theme.ink, strokeWidth: 1.6, strokeDasharray: '2 3' }} />
            <path d={path((x) => model1d(passive, x), pe)} style={{ fill: 'none', stroke: theme.fail, strokeWidth: 2, strokeDasharray: '7 4' }} />
            {active ? <path d={path((x) => model1d(active, x), pe)} style={{ fill: 'none', stroke: theme.arch, strokeWidth: 2.2 }} /> : null}
            {escapes(passive, barP) ? (
              <g>
                <circle cx={px(barP.x)} cy={pe(barP.e)} r={5} style={{ fill: 'none', stroke: theme.fail, strokeWidth: 1.6 }} />
                <text x={px(barP.x) + 8} y={pe(barP.e) - 8} className="tick" style={{ fill: theme.fail }}>spurious barrier</text>
              </g>
            ) : null}
            {[x1, x2].map((x, k) => (
              <circle key={k} cx={px(x)} cy={pe(eqm1d(x))} r={5.5} style={{ fill: theme.arch, stroke: theme.canvas, strokeWidth: 1.5 }} />
            ))}
            <text x={px(0)} y={top + h1 - 8} textAnchor="middle" className="tick">region of interest [−4, 4]</text>

            <text x={x0} y={g0 - 10} className="axis-label">extrapolation grade γ(x) of the active pair (log)</text>
            {[-1, 0, 1, 2, 3].map((e) => (
              <g key={e}>
                <line x1={x0} x2={x0 + w} y1={pg(10 ** e)} y2={pg(10 ** e)} className="grid" />
                <text x={x0 - 6} y={pg(10 ** e) + 4} textAnchor="end" className="tick">{e === -1 ? '0.1' : (10 ** e).toLocaleString('en')}</text>
              </g>
            ))}
            {[
              [2, 'select ≈ 2'],
              [10, 'break ≈ 10'],
            ].map(([g, t]) => (
              <g key={t as string}>
                <line x1={x0} x2={x0 + w} y1={pg(g as number)} y2={pg(g as number)} style={{ stroke: theme.caveat, strokeDasharray: '4 3', strokeWidth: 1 }} />
                <text x={x0 + w - 4} y={pg(g as number) - 4} textAnchor="end" className="tick" style={{ fill: theme.caveat }}>{t}</text>
              </g>
            ))}
            <line x1={x0} x2={x0 + w} y1={pg(1)} y2={pg(1)} style={{ stroke: theme.verified, strokeWidth: 1.4 }} />
            <path d={path((x) => grade1d(x, x1, x2), pg)} style={{ fill: 'none', stroke: theme.arch, strokeWidth: 2 }} />
            {[-5, -4, -2, 0, 2, 4, 5].map((x) => (
              <text key={x} x={px(x)} y={g0 + h2 + 16} textAnchor="middle" className="tick">{x}</text>
            ))}
            <text x={x0 + w} y={g0 + h2 + 34} textAnchor="end" className="tick">x</text>
          </svg>
          <figcaption className="lab-legend">
            <span className="item"><span className="swatch" style={{ background: theme.ink }} /> reference E_qm(x)</span>
            <span className="item"><span className="swatch" style={{ background: theme.fail }} /> passive: fit to MD at kT</span>
            <span className="item"><span className="swatch" style={{ background: theme.arch }} /> active: through the two points</span>
          </figcaption>
        </figure>
        <div className="mtp-side">
          <div className="lab-controls">
            <label htmlFor={`${ids}-1`}>
              <span>first active point x₁</span>
              <input id={`${ids}-1`} type="range" min={-4} max={4} step={0.1} value={x1} onChange={(e) => setX1(Number(e.target.value))} />
              <output>{x1.toFixed(1)}</output>
            </label>
            <label htmlFor={`${ids}-2`}>
              <span>second active point x₂</span>
              <input id={`${ids}-2`} type="range" min={-4} max={4} step={0.1} value={x2} onChange={(e) => setX2(Number(e.target.value))} />
              <output>{x2.toFixed(1)}</output>
            </label>
            <label htmlFor={`${ids}-t`}>
              <span>MD temperature for the passive fit, k<sub>B</sub>T</span>
              <input id={`${ids}-t`} type="range" min={0.5} max={3} step={0.1} value={kT} onChange={(e) => setKT(Number(e.target.value))} />
              <output>{kT.toFixed(1)}</output>
            </label>
            <div className="seg" role="group" aria-label="Presets">
              <button type="button" aria-pressed={x1 === -4 && x2 === 4} onClick={() => (setX1(-4), setX2(4))}>
                D-optimal pair (±4)
              </button>
              <button type="button" aria-pressed={kT === 1} onClick={() => setKT(1)}>
                paper’s k<sub>B</sub>T = 1
              </button>
            </div>
          </div>
          <dl className="lab-readout" aria-live="polite">
            <div>
              <dt>passive fit</dt>
              <dd>
                {passive[0]!.toFixed(2)}x² + {passive[1]!.toFixed(2)}x³ <small>RMS {rmsP.toFixed(2)}</small>
              </dd>
            </div>
            <div className={escapes(passive, barP) ? 'is-alert' : 'is-good'}>
              <dt>passive: barrier inside [−4, 4]?</dt>
              <dd>{escapes(passive, barP) ? `yes, at x = ${barP.x.toFixed(2)}: MD can escape to −∞` : 'no'}</dd>
            </div>
            <div>
              <dt>active fit</dt>
              <dd>
                {active ? (
                  <>
                    {active[0]!.toFixed(2)}x² + {Math.abs(active[1]!) < 0.005 ? active[1]!.toExponential(1) : active[1]!.toFixed(2)}x³ <small>RMS {rmsA.toFixed(2)}</small>
                  </>
                ) : (
                  'points coincide: no fit'
                )}
              </dd>
            </div>
            <div className={active && barA && escapes(active, barA) ? 'is-alert' : 'is-good'}>
              <dt>active: barrier inside [−4, 4]?</dt>
              <dd>{active && barA && escapes(active, barA) ? `yes, at x = ${barA.x.toFixed(2)}` : 'no'}</dd>
            </div>
            <div>
              <dt>largest γ on [−4, 4]</dt>
              <dd>{Math.max(...XS.filter((x) => Math.abs(x) <= 4).map((x) => grade1d(x, x1, x2))).toFixed(2)}</dd>
            </div>
          </dl>
        </div>
      </div>
      <p className="lab-note">
        The passive fit is the least-squares fit an infinitely long MD at k<sub>B</sub>T would give. It is most accurate where the system spends its time, but
        bends down beyond x ≈ −3, so a trajectory that crosses that barrier never comes back. Two training points are enough for two
        parameters; the pair that maximises |det A| is x = ±4, and then γ ≤ 1 over the whole region: every energy inside is an
        interpolation of the two. Move the points inward and γ climbs above 1 near the edges, exactly where the fit goes wrong.
      </p>
    </div>
  );
}
