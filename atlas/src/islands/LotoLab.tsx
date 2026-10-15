import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, linear, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { NACL_R0, rockSaltCluster } from '~/lib/chem/structures';
import { MASS, modes, zonePoints, type Mode, type RockSaltModel } from '~/lib/physics/lattice-dynamics';

// TOY MODEL: rigid-ion rock salt. Exact Coulomb (Ewald) against a local model (damped shifted
// force Coulomb, zero beyond r_c); the LO–TO splitting at Γ needs the long-range part.

type V3 = [number, number, number];
type Branch = 'TO' | 'LO' | 'TA' | 'LA';
const R0 = NACL_R0;
const ZP = zonePoints(R0);
const N = 40;
const QMIN = 0.02;

/** Path X ← Γ → L parametrised by s ∈ [−1, 1]. */
function qAt(s: number): { q: V3; hat: V3 } {
  const end = s < 0 ? ZP.X : ZP.L;
  const t = Math.max(Math.abs(s), QMIN / Math.hypot(...end));
  return { q: [end[0] * t, end[1] * t, end[2] * t], hat: end };
}
const xOf = (s: number) => (s < 0 ? s * Math.hypot(...ZP.X) : s * Math.hypot(...ZP.L));

function pick(ms: Mode[], b: Branch): Mode {
  const opt = b === 'TO' || b === 'LO';
  const lon = b === 'LO' || b === 'LA';
  const pool = ms.filter((m) => m.optical === opt);
  const list = pool.length ? pool : ms;
  return list.reduce((best, m) => (lon ? (m.longitudinal > best.longitudinal ? m : best) : m.longitudinal < best.longitudinal ? m : best));
}

function bands(model: RockSaltModel) {
  const ss = [...Array.from({ length: N + 1 }, (_, i) => -1 + i / N), ...Array.from({ length: N }, (_, i) => (i + 1) / N)];
  return ss.map((s) => {
    const { q, hat } = qAt(s);
    const exactGamma = Math.abs(s) < 1e-9;
    return { s, nu: modes(model, exactGamma ? [0, 0, 0] : q, hat).map((m) => m.nu) };
  });
}

export default function LotoLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [rho, setRho] = useState(0.32);
  const [rc, setRc] = useState(9);
  const [which, setWhich] = useState<'ewald' | 'cutoff'>('ewald');
  const [branch, setBranch] = useState<Branch>('LO');
  const [s, setS] = useState(0.1);
  const [playing, setPlaying] = useState(false);
  const [phase, setPhase] = useState(0);

  const exact = useMemo<RockSaltModel>(() => ({ r0: R0, Z: 1, rho, coulomb: 'ewald' }), [rho]);
  const local = useMemo<RockSaltModel>(() => ({ r0: R0, Z: 1, rho, coulomb: 'cutoff', rc }), [rho, rc]);
  const bExact = useMemo(() => bands(exact), [exact]);
  const bLocal = useMemo(() => bands(local), [local]);
  const gammaExact = useMemo(() => modes(exact, [0, 0, 0], [1, 0, 0]).filter((m) => m.optical), [exact]);
  const gammaLocal = useMemo(() => modes(local, [0, 0, 0], [1, 0, 0]).filter((m) => m.optical), [local]);
  const nuTO = Math.min(...gammaExact.map((m) => m.nu));
  const nuLO = Math.max(...gammaExact.map((m) => m.nu));
  const locTO = Math.min(...gammaLocal.map((m) => m.nu));
  const locLO = Math.max(...gammaLocal.map((m) => m.nu));
  const { q, hat } = qAt(s);
  const mode = useMemo(() => pick(modes(which === 'ewald' ? exact : local, q, hat), branch), [which, exact, local, q[0], q[1], q[2], branch]);
  const other = useMemo(() => pick(modes(which === 'ewald' ? local : exact, q, hat), branch), [which, exact, local, q[0], q[1], q[2], branch]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      if (t - last > 33) {
        setPhase((p) => (p + ((t - last) / 1000) * 2 * Math.PI * 0.6) % (2 * Math.PI));
        last = t;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const dirLabel = s < 0 ? 'Γ→X [100]' : 'Γ→L [111]';
  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [26, -30, 18], target: [0, 0, 0] }} background={theme.canvas} label={`Rock-salt crystal vibrating in its ${branch} mode along ${dirLabel}, ${mode.nu.toFixed(2)} THz.`} minDistance={12} maxDistance={90}>
            <ModeScene theme={theme} mode={mode} q={q} phase={phase} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Branch">
              {(['TO', 'LO', 'TA', 'LA'] as Branch[]).map((b) => (
                <button key={b} type="button" aria-pressed={branch === b} onClick={() => setBranch(b)}>
                  {b}
                </button>
              ))}
            </div>
            <button type="button" className="lab-btn" aria-pressed={playing} onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: ELEMENTS.Na.color }} /> Na⁺ <span className="dot" style={{ background: ELEMENTS.Cl.color }} /> Cl⁻</li>
            <li><span className="bar" style={{ background: theme.ink }} /> wavevector q</li>
          </ul>
        </div>
        <DispersionPlot theme={theme} exact={bExact} local={bLocal} s={s} nu={mode.nu} which={which} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Coulomb treatment for the 3D mode">
            <button type="button" aria-pressed={which === 'ewald'} onClick={() => setWhich('ewald')}>Exact Coulomb (Ewald)</button>
            <button type="button" aria-pressed={which === 'cutoff'} onClick={() => setWhich('cutoff')}>Local model, cutoff r_c</button>
          </div>
          <label htmlFor={`${ids}-s`}>
            <span>wavevector along X ← Γ → L</span>
            <input id={`${ids}-s`} type="range" min={-1} max={1} step={0.025} value={s} onChange={(e) => setS(Number(e.target.value))} />
            <output>{Math.abs(s) < 1e-9 ? 'Γ' : `${Math.abs(s).toFixed(2)} ${s < 0 ? 'X' : 'L'}`}</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>local cutoff r_c</span>
            <input id={`${ids}-rc`} type="range" min={4} max={14} step={0.5} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-rho`}>
            <span>repulsion range ρ</span>
            <input id={`${ids}-rho`} type="range" min={0.26} max={0.42} step={0.01} value={rho} onChange={(e) => setRho(Number(e.target.value))} />
            <output>{rho.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>{branch} here, {which === 'ewald' ? 'exact' : 'local'}</dt>
            <dd>{mode.nu.toFixed(2)} THz</dd>
          </div>
          <div>
            <dt>{branch} here, {which === 'ewald' ? 'local' : 'exact'}</dt>
            <dd>{other.nu.toFixed(2)} THz</dd>
          </div>
          <div>
            <dt>Γ: TO / LO, exact</dt>
            <dd>{nuTO.toFixed(2)} / {nuLO.toFixed(2)}</dd>
          </div>
          <div className="is-alert">
            <dt>Γ: TO / LO, local</dt>
            <dd>{locTO.toFixed(2)} / {locLO.toFixed(2)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          In a longitudinal optical wave the Na⁺ and Cl⁻ planes move against each other along q, building sheets of
          charge whose field stiffens the vibration; in a transverse wave they slide sideways and no such field forms.
          That field reaches across the whole crystal, so a model that only sees r_c has LO and TO meet at Γ however
          large r_c is.
        </p>
      </div>
    </div>
  );
}

const CLUSTER = rockSaltCluster(2, R0);

function ModeScene({ theme, mode, q, phase }: { theme: SceneTheme; mode: Mode; q: V3; phase: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 70);
  const disp = useMemo(() => {
    const u = CLUSTER.atoms.map((a) => {
      const k = a.el === 'Na' ? 0 : 3;
      const m = a.el === 'Na' ? MASS.Na : MASS.Cl;
      const c = Math.cos(q[0] * a.p[0] + q[1] * a.p[1] + q[2] * a.p[2]);
      return [(mode.e[k]! / Math.sqrt(m)) * c, (mode.e[k + 1]! / Math.sqrt(m)) * c, (mode.e[k + 2]! / Math.sqrt(m)) * c] as V3;
    });
    const max = Math.max(...u.map((v) => Math.hypot(...v))) || 1;
    return u.map((v) => [(v[0] / max) * 0.55, (v[1] / max) * 0.55, (v[2] / max) * 0.55] as V3);
  }, [mode, q]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = CLUSTER.atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const c = Math.cos(phase);
    CLUSTER.atoms.forEach((a, i) => {
      const d = disp[i]!;
      pos.set([a.p[0] + d[0] * c, a.p[1] + d[1] * c, a.p[2] + d[2] * c], 3 * i);
      radii[i] = ELEMENTS[a.el].ionicRadius * 0.48;
      colors.set(linear(ELEMENTS[a.el].color), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [disp, phase]);
  const arrows = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    const ink = linear(theme.inkMuted);
    CLUSTER.atoms.forEach((a, i) => {
      const d = disp[i]!;
      const l = Math.hypot(...d);
      if (l < 0.08) return;
      const s = 4.2;
      out.push({ a: a.p, b: [a.p[0] + d[0] * s, a.p[1] + d[1] * s, a.p[2] + d[2] * s], color: ink, radius: 0.08 });
    });
    const ql = Math.hypot(...q) || 1;
    const top: V3 = [-3, -8.5, -8];
    out.push({ a: top, b: [top[0] + (q[0] / ql) * 5, top[1] + (q[1] / ql) * 5, top[2] + (q[2] / ql) * 5], color: linear(theme.ink), radius: 0.12 });
    return out;
  }, [disp, q, theme.inkMuted, theme.ink]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Arrows arrows={arrows} uniforms={uniforms} head={0.3} />
    </>
  );
}

function DispersionPlot({ theme, exact, local, s, nu, which }: { theme: SceneTheme; exact: ReturnType<typeof bands>; local: ReturnType<typeof bands>; s: number; nu: number; which: 'ewald' | 'cutoff' }) {
  const W = 520;
  const H = 380;
  const M = { l: 50, r: 16, t: 24, b: 40 };
  const x0 = xOf(-1);
  const x1 = xOf(1);
  const top = Math.ceil(Math.max(...exact.flatMap((p) => p.nu), ...local.flatMap((p) => p.nu)) + 0.5);
  const X = (s2: number) => M.l + ((xOf(s2) - x0) / (x1 - x0)) * (W - M.l - M.r);
  const Y = (v: number) => H - M.b - (Math.max(0, v) / top) * (H - M.t - M.b);
  const path = (pts: ReturnType<typeof bands>, k: number, side: -1 | 1) =>
    pts
      .filter((p) => (side < 0 ? p.s <= 0 : p.s >= 0))
      .map((p, i) => `${i ? 'L' : 'M'}${X(p.s).toFixed(1)},${Y(p.nu[k]!).toFixed(1)}`)
      .join('');
  const ticks = Array.from({ length: top + 1 }, (_, i) => i).filter((v) => v % 2 === 0);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Phonon dispersion of the rigid-ion rock-salt toy along X–Γ–L, exact Coulomb against the local model.">
        <text x={M.l} y={M.t - 9} className="axis-label">frequency ν [THz]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className={v === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <line x1={X(0)} x2={X(0)} y1={M.t} y2={H - M.b} className="grid" />
        {[0, 1, 2, 3, 4, 5].flatMap((k) =>
          ([-1, 1] as const).map((side) => (
            <path key={`l${k}${side}`} d={path(local, k, side)} className="curve" style={{ stroke: theme.caveat, strokeDasharray: '5 3', strokeWidth: 1.6, opacity: which === 'cutoff' ? 1 : 0.75 }} />
          )),
        )}
        {[0, 1, 2, 3, 4, 5].flatMap((k) =>
          ([-1, 1] as const).map((side) => (
            <path key={`e${k}${side}`} d={path(exact, k, side)} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.2, opacity: which === 'ewald' ? 1 : 0.75 }} />
          )),
        )}
        <line x1={X(s)} x2={X(s)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={X(s)} cy={Y(nu)} r={5} style={{ fill: which === 'ewald' ? theme.phys : theme.caveat, stroke: theme.canvas, strokeWidth: 1.5 }} />
        {[
          { s2: -1, t: 'X' },
          { s2: 0, t: 'Γ' },
          { s2: 1, t: 'L' },
        ].map((p) => (
          <text key={p.t} x={X(p.s2)} y={H - M.b + 18} textAnchor="middle" className="tick">{p.t}</text>
        ))}
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> exact Coulomb (Ewald)</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> local model (damped Coulomb, zero beyond r_c)</span>
      </figcaption>
    </figure>
  );
}
