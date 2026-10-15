import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, CellBox, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { NACL_R0 } from '~/lib/chem/structures';
import { MADELUNG_NACL, erfc, ewaldParts } from '~/lib/physics/madelung';

// Physics Lab P6: Ewald summation on rock salt. Each point charge is split into a
// screened charge (point + opposite Gaussian; short-ranged, summed in real space) and
// a smooth Gaussian (summed in reciprocal space). α sets the Gaussian width.

type View = 'points' | 'real' | 'recip';
const VIEWS: Array<{ id: View; label: string }> = [
  { id: 'points', label: 'Point charges' },
  { id: 'real', label: 'Real-space part' },
  { id: 'recip', label: 'Reciprocal part' },
];

// Conventional cell, units of r0: ions at integer corners 0..1 of a 2-r0 cube.
const CELL: Array<{ p: [number, number, number]; q: number }> = [];
for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) CELL.push({ p: [i, j, k], q: (i + j + k) % 2 === 0 ? 1 : -1 });
const SHIFT = 0.5; // centre the drawn cell on the origin

export default function EwaldLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [alpha, setAlpha] = useState(1.2);
  const [rCut, setRCut] = useState(3);
  const [kCut, setKCut] = useState(4);
  const [view, setView] = useState<View>('real');

  const parts = useMemo(() => ewaldParts(alpha, rCut, kCut), [alpha, rCut, kCut]);
  const M = -parts.total;
  const err = M - MADELUNG_NACL;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [16, -24, 13], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Rock-salt conventional cell with Gaussian charge clouds of width 1/α, α = ${alpha.toFixed(2)} per r0, showing the ${view === 'points' ? 'bare point charges' : view === 'real' ? 'screened charges summed in real space' : 'smooth Gaussians summed in reciprocal space'}.`}
            minDistance={10}
            maxDistance={80}
          >
            <EwaldScene theme={theme} alpha={alpha} rCut={rCut} view={view} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="What to show">
              {VIEWS.map((v) => (
                <button key={v.id} type="button" aria-pressed={view === v.id} onClick={() => setView(v.id)}>
                  {v.label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: ELEMENTS.Na.color }} /> +1 e
            </li>
            <li>
              <span className="dot" style={{ background: ELEMENTS.Cl.color }} /> −1 e
            </li>
            <li>
              <span className="ring" /> real-space cutoff around the reference Na⁺
            </li>
          </ul>
        </div>
        <EwaldPlots alpha={alpha} rCut={rCut} kCut={kCut} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-a`} data-sym="ewa">
            <span>
              splitting <i>α</i>
            </span>
            <input id={`${ids}-a`} type="range" min={0.5} max={3} step={0.02} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
            <output>
              {alpha.toFixed(2)} / r<sub>0</sub>
            </output>
          </label>
          <label htmlFor={`${ids}-r`}>
            <span>real cutoff</span>
            <input id={`${ids}-r`} type="range" min={1} max={6} step={0.1} value={rCut} onChange={(e) => setRCut(Number(e.target.value))} />
            <output>
              {rCut.toFixed(1)} r<sub>0</sub>
            </output>
          </label>
          <label htmlFor={`${ids}-k`}>
            <span>k-space cutoff |m|</span>
            <input id={`${ids}-k`} type="range" min={1} max={8} step={1} value={kCut} onChange={(e) => setKCut(Number(e.target.value))} />
            <output>{kCut}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Real space</dt>
            <dd>
              {parts.real.toFixed(5)} <small>({parts.nReal} terms)</small>
            </dd>
          </div>
          <div>
            <dt>Reciprocal</dt>
            <dd>
              {parts.recip.toFixed(5)} <small>({parts.nK} k)</small>
            </dd>
          </div>
          <div>
            <dt>Self</dt>
            <dd>{parts.self.toFixed(5)}</dd>
          </div>
          <div className={Math.abs(err) < 1e-5 ? 'is-good' : 'is-alert'}>
            <dt>M = −(sum)</dt>
            <dd>
              {M.toFixed(6)} <small>({err >= 0 ? '+' : '−'}{Math.abs(err).toExponential(1)})</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Slide <i>α</i>: weight moves between the real-space and reciprocal sums, but their total stays at
          {' '}<strong>M = {MADELUNG_NACL.toFixed(6)}</strong> as long as both cutoffs are generous. Small <i>α</i> (wide
          Gaussians) needs a long real-space cutoff; large <i>α</i> (narrow Gaussians) needs many <b>k</b> vectors.
        </p>
      </div>
    </div>
  );
}

function EwaldScene({ theme, alpha, rCut, view }: { theme: SceneTheme; alpha: number; rCut: number; view: View }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 60);
  const s = NACL_R0;
  const buffers = useMemo<AtomBuffers>(() => {
    const na = linear(ELEMENTS.Na.color);
    const cl = linear(ELEMENTS.Cl.color);
    const stone = linear(theme.dark ? theme.canvas : '#b7bec9');
    const list: Array<{ p: [number, number, number]; q: number; central: boolean }> = [];
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++)
        for (let c = -1; c <= 1; c++)
          for (const ion of CELL)
            list.push({
              p: [(ion.p[0] + 2 * a - SHIFT) * s, (ion.p[1] + 2 * b - SHIFT) * s, (ion.p[2] + 2 * c - SHIFT) * s],
              q: ion.q,
              central: a === 0 && b === 0 && c === 0,
            });
    const n = list.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    list.forEach((ion, k) => {
      positions.set(ion.p, 3 * k);
      const base = ion.q > 0 ? na : cl;
      const r = (ion.q > 0 ? ELEMENTS.Na.ionicRadius : ELEMENTS.Cl.ionicRadius) * 0.6;
      const visible = view !== 'recip';
      radii[k] = visible ? (ion.central ? r : r * 0.38) : ion.central ? r * 0.25 : 0;
      colors.set(ion.central ? base : mixRGB(base, stone, 0.8), 3 * k);
    });
    return { positions, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [theme, view, s]);

  // Gaussian "radius" ≈ 1.5/α (encloses most of the cloud), in Å.
  const cloud = (1.5 / alpha) * s;
  const ref: [number, number, number] = [-SHIFT * s, -SHIFT * s, -SHIFT * s];
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {view !== 'points' &&
        CELL.map((ion, k) => {
          const sign = view === 'real' ? -ion.q : ion.q; // real part: opposite screening cloud
          return (
            <Shell
              key={k}
              center={[(ion.p[0] - SHIFT) * s, (ion.p[1] - SHIFT) * s, (ion.p[2] - SHIFT) * s]}
              radius={cloud}
              color={sign > 0 ? ELEMENTS.Na.color : ELEMENTS.Cl.color}
              opacity={0.28}
            />
          );
        })}
      <Shell center={ref} radius={rCut * s} color={theme.phys} opacity={0.2} grid />
      <CellBox origin={[-SHIFT * s - 0.5 * s, -SHIFT * s - 0.5 * s, -SHIFT * s - 0.5 * s]} size={[2 * s, 2 * s, 2 * s]} color={theme.ruleStrong} opacity={0.7} />
    </>
  );
}

function EwaldPlots({ alpha, rCut, kCut }: { alpha: number; rCut: number; kCut: number }) {
  const sweep = useMemo(() => {
    const as = Array.from({ length: 51 }, (_, k) => 0.5 + (k * 2.5) / 50);
    return as.map((a) => ({ a, ...ewaldParts(a, 7, 8) }));
  }, []);
  const conv = useMemo(() => {
    const r = Array.from({ length: 26 }, (_, k) => 1 + k * 0.2).map((rc) => ({ x: rc, e: Math.abs(-ewaldParts(alpha, rc, 8).total - MADELUNG_NACL) }));
    const k = Array.from({ length: 8 }, (_, i) => i + 1).map((kc) => ({ x: kc, e: Math.abs(-ewaldParts(alpha, 7, kc).total - MADELUNG_NACL) }));
    return { r, k };
  }, [alpha]);

  const W = 520;
  const L = 44;
  const GX = 40;
  const PW = (W - L - GX - 12) / 2;
  const PH = 132;
  const GY = 54;
  const T = 22;
  const H = T + 2 * PH + GY + 34;
  const panel = (col: number, row: number) => ({ x0: L + col * (PW + GX), y0: T + row * (PH + GY) });

  // Panel A: radial functions.
  const A = panel(0, 0);
  const ra = (r: number) => A.x0 + (r / 4) * PW;
  const ya = (v: number) => A.y0 + PH - (Math.min(v, 2.6) / 2.6) * PH;
  const rs = Array.from({ length: 120 }, (_, k) => 0.12 + (k * 3.88) / 119);
  const path = (f: (r: number) => number) => rs.map((r, k) => `${k ? 'L' : 'M'}${ra(r).toFixed(1)},${ya(f(r)).toFixed(1)}`).join('');

  // Panel B: parts vs α.
  const B = panel(1, 0);
  const vals = sweep.flatMap((p) => [p.real, p.recip, p.self, p.total]);
  const lo = Math.min(...vals) - 0.2;
  const hi = Math.max(...vals) + 0.2;
  const xb = (a: number) => B.x0 + ((a - 0.5) / 2.5) * PW;
  const yb = (v: number) => B.y0 + PH - ((v - lo) / (hi - lo)) * PH;
  const series = (key: 'real' | 'recip' | 'self' | 'total') => sweep.map((p, k) => `${k ? 'L' : 'M'}${xb(p.a).toFixed(1)},${yb(p[key]).toFixed(1)}`).join('');

  // Panels C, D: log10 error.
  const C = panel(0, 1);
  const D = panel(1, 1);
  const ly = (y0: number) => (e: number) => y0 + PH - ((Math.log10(Math.max(e, 1e-10)) + 10) / 10) * PH;
  const yc = ly(C.y0);
  const yd = ly(D.y0);
  const xc = (r: number) => C.x0 + ((r - 1) / 5) * PW;
  const xd = (k: number) => D.x0 + ((k - 1) / 7) * PW;

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Four panels: the three radial functions; the real, reciprocal and self contributions against alpha with a flat total; and the error against the real-space and reciprocal-space cutoffs on a log scale.">
        {/* A */}
        <text x={A.x0} y={A.y0 - 8} className="axis-label">1/r split at α = {alpha.toFixed(2)}</text>
        <rect x={A.x0} y={A.y0} width={PW} height={PH} className="frame" />
        <path d={path((r) => 1 / r)} className="curve ew-bare" />
        <path d={path((r) => erfc(alpha * r) / r)} className="curve ew-real" />
        <path d={path((r) => (1 - erfc(alpha * r)) / r)} className="curve ew-recip" />
        {[0, 1, 2, 3, 4].map((r) => (
          <text key={r} x={ra(r)} y={A.y0 + PH + 14} textAnchor="middle" className="tick">{r}</text>
        ))}
        <text x={A.x0 + PW} y={A.y0 + PH + 28} textAnchor="end" className="tick">r [r₀]</text>

        {/* B */}
        <text x={B.x0} y={B.y0 - 8} className="axis-label">contributions vs α</text>
        <rect x={B.x0} y={B.y0} width={PW} height={PH} className="frame" />
        <path d={series('real')} className="curve ew-real" />
        <path d={series('recip')} className="curve ew-recip" />
        <path d={series('self')} className="curve ew-self" />
        <path d={series('total')} className="curve ew-total" />
        <line x1={xb(alpha)} x2={xb(alpha)} y1={B.y0} y2={B.y0 + PH} className="r-line" />
        {[0.5, 1.5, 2.5].map((a) => (
          <text key={a} x={xb(a)} y={B.y0 + PH + 14} textAnchor="middle" className="tick">{a}</text>
        ))}
        <text x={B.x0 + PW} y={B.y0 + PH + 28} textAnchor="end" className="tick">α [1/r₀]</text>
        <text x={B.x0 - 6} y={yb(-MADELUNG_NACL) + 4} textAnchor="end" className="tick">−M</text>

        {/* C */}
        <text x={C.x0} y={C.y0 - 8} className="axis-label">error vs real cutoff</text>
        <rect x={C.x0} y={C.y0} width={PW} height={PH} className="frame" />
        <path d={conv.r.map((p, k) => `${k ? 'L' : 'M'}${xc(p.x).toFixed(1)},${yc(p.e).toFixed(1)}`).join('')} className="curve ew-real" />
        <line x1={xc(rCut)} x2={xc(rCut)} y1={C.y0} y2={C.y0 + PH} className="r-line" />
        {[1, 2, 3, 4, 5, 6].map((r) => (
          <text key={r} x={xc(r)} y={C.y0 + PH + 14} textAnchor="middle" className="tick">{r}</text>
        ))}
        {[-2, -6, -10].map((e) => (
          <text key={e} x={C.x0 - 6} y={yc(10 ** e) + 4} textAnchor="end" className="tick">10{sup(e)}</text>
        ))}
        <text x={C.x0 + PW} y={C.y0 + PH + 28} textAnchor="end" className="tick">r_cut [r₀]</text>

        {/* D */}
        <text x={D.x0} y={D.y0 - 8} className="axis-label">error vs k cutoff</text>
        <rect x={D.x0} y={D.y0} width={PW} height={PH} className="frame" />
        <path d={conv.k.map((p, k) => `${k ? 'L' : 'M'}${xd(p.x).toFixed(1)},${yd(p.e).toFixed(1)}`).join('')} className="curve ew-recip" />
        {conv.k.map((p) => (
          <circle key={p.x} cx={xd(p.x)} cy={yd(p.e)} r={2.6} className="pt-recip" />
        ))}
        <line x1={xd(kCut)} x2={xd(kCut)} y1={D.y0} y2={D.y0 + PH} className="r-line" />
        {[1, 2, 4, 6, 8].map((k) => (
          <text key={k} x={xd(k)} y={D.y0 + PH + 14} textAnchor="middle" className="tick">{k}</text>
        ))}
        <text x={D.x0 + PW} y={D.y0 + PH + 28} textAnchor="end" className="tick">|m| max</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch ew-bare" /> 1/r</span>
        <span className="item"><span className="swatch ew-real" /> real space: erfc(αr)/r</span>
        <span className="item"><span className="swatch ew-recip" /> reciprocal: erf(αr)/r</span>
        <span className="item"><span className="swatch ew-self" /> self</span>
        <span className="item"><span className="swatch ew-total" /> total (= −M)</span>
      </figcaption>
    </figure>
  );
}

function sup(e: number): string {
  const map: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  return String(e).split('').map((c) => map[c] ?? c).join('');
}
