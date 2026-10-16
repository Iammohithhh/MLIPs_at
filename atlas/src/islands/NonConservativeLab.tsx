import { useEffect, useId, useMemo, useRef, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, ShadedSurface, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, usePrefersReducedMotion, type SceneTheme } from '~/three/theme';
import {
  MODELS,
  amplitudeFor,
  circulation,
  deltaF1Curl,
  deltaF1Gradient,
  launch,
  potential,
  run,
  wellMesh,
  type ForceModel,
  type Trajectory,
} from '~/lib/physics/nonconservative';

// TOY MODEL: one particle in a 2D anharmonic well, integrated with velocity Verlet under
// the exact force and two erroneous "models" with the same RMS force error — one whose
// error is a gradient, one whose error has a curl. All numbers come from
// src/lib/physics/nonconservative.ts (tested).

const T_END = 150;
const STORED = 1500;
const Z_SCALE = 0.35;
const R_BOWL = 3.6;
const LOOP_R = 1.5;
const DTS = [0.01, 0.05, 0.1, 0.2] as const;

const colorOf = (t: SceneTheme, m: ForceModel) => (m === 'exact' ? t.verified : m === 'gradient' ? t.arch : t.fail);

export default function NonConservativeLab() {
  const theme = useSceneTheme();
  const reduced = usePrefersReducedMotion();
  const ids = useId();
  const [sigma, setSigma] = useState(0.03);
  const [dt, setDt] = useState<(typeof DTS)[number]>(0.05);
  const [dir, setDir] = useState<1 | -1>(1);
  const [frac, setFrac] = useState(1);
  const [playing, setPlaying] = useState(false);

  const runs = useMemo(() => {
    const steps = Math.round(T_END / dt);
    const every = Math.max(1, Math.round(steps / STORED));
    return Object.fromEntries(MODELS.map((m) => [m.id, run({ model: m.id, sigma, dt, tEnd: T_END, ...launch(dir), every })])) as Record<ForceModel, Trajectory>;
  }, [sigma, dt, dir]);
  const n = runs.exact.t.length;
  const k = Math.min(n - 1, Math.round(frac * (n - 1)));

  const raf = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      const d = (now - last) / 1000;
      last = now;
      setFrac((f) => {
        const next = f + d / 12;
        if (next >= 1) {
          setPlaying(false);
          return 1;
        }
        return next;
      });
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing]);

  const play = () => {
    if (reduced) {
      setFrac(1);
      return;
    }
    if (frac >= 1) setFrac(0);
    setPlaying((p) => !p);
  };

  const dE = (m: ForceModel) => runs[m].eRef[k]! - runs[m].eRef[0]!;
  const circ = (m: ForceModel) => circulation(m, amplitudeFor(m, sigma), LOOP_R);
  const t = runs.exact.t[k]!;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [7, -13, 17], target: [0, 0, 1] }}
            background={theme.canvas}
            label={`A particle in a bowl-shaped potential well followed for ${t.toFixed(0)} time units under three force models. The model whose force error has a curl climbs the bowl: its energy has risen by ${dE('direct').toFixed(2)}.`}
            minDistance={8}
            maxDistance={60}
          >
            <WellScene theme={theme} runs={runs} k={k} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            {MODELS.map((m) => (
              <li key={m.id}>
                <span className="bar" style={{ background: colorOf(theme, m.id) }} /> {m.label} <small>{m.short}</small>
              </li>
            ))}
          </ul>
        </div>
        <EnergyPlot theme={theme} runs={runs} k={k} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-s`}>
            <span>RMS force error σ (both erroneous models)</span>
            <input id={`${ids}-s`} type="range" min={0} max={0.05} step={0.001} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
            <output>{sigma.toFixed(3)}</output>
          </label>
          <label htmlFor={`${ids}-t`}>
            <span>time</span>
            <input id={`${ids}-t`} type="range" min={0} max={1} step={0.001} value={frac} onChange={(e) => { setPlaying(false); setFrac(Number(e.target.value)); }} />
            <output>{t.toFixed(0)}</output>
          </label>
          <div className="seg" role="group" aria-label="Time step">
            {DTS.map((h) => (
              <button key={h} type="button" aria-pressed={dt === h} onClick={() => setDt(h)}>h = {h}</button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Launch direction">
            <button type="button" aria-pressed={dir === 1} onClick={() => setDir(1)}>launch anticlockwise</button>
            <button type="button" aria-pressed={dir === -1} onClick={() => setDir(-1)}>launch clockwise</button>
          </div>
          <div className="seg" role="group" aria-label="Playback">
            <button type="button" aria-pressed={playing} onClick={play}>{playing ? 'pause' : frac >= 1 ? 'replay' : 'play'}</button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          {MODELS.map((m) => {
            const v = dE(m.id);
            return (
              <div key={m.id} className={m.id === 'direct' && Math.abs(v) > 0.05 ? 'is-alert' : m.id === 'exact' ? 'is-good' : ''}>
                <dt>ΔE at t = {t.toFixed(0)}, {m.label.toLowerCase()}</dt>
                <dd>{Math.abs(v) < 5e-4 ? '0.000' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`}</dd>
              </div>
            );
          })}
        </dl>
        <p className="lab-note">
          Both erroneous models miss the exact force by the same RMS amount, σ = {sigma.toFixed(3)}, on the disc r ≤ 2, so a
          force-error table scores them equally. The error that is a gradient conserves the model’s own energy V + δV (its
          reference energy only wobbles by at most the size of δV); the error with a curl does net work on every orbit, and
          the energy runs away. Change the time step: the exact model’s small wobble shrinks as h², the runaway does not —
          it is not an integrator error. Well V = ½(x² + y²) + 0.05(x² + y²)², energies E = ½|v|² + V, reduced units (mass 1); the start energy is{' '}
          {runs.exact.eRef[0]!.toFixed(3)}.
        </p>
      </div>
      <div className="lab-split">
        <LoopField theme={theme} model="gradient" sigma={sigma} circ={circ('gradient')} />
        <LoopField theme={theme} model="direct" sigma={sigma} circ={circ('direct')} />
      </div>
    </div>
  );
}

function WellScene({ theme, runs, k }: { theme: SceneTheme; runs: Record<ForceModel, Trajectory>; k: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const surface = useMemo<SurfaceData>(() => {
    const m = wellMesh(R_BOWL, 48, 128);
    const zMax = m.heights[m.heights.length - 1]!;
    const positions = new Float32Array(m.positions);
    for (let i = 2; i < positions.length; i += 3) positions[i] = positions[i]! * Z_SCALE;
    const lo = linear(theme.dark ? '#2a3340' : '#e7e3da');
    const hi = linear(theme.phys);
    const colors = new Float32Array(m.heights.length * 3);
    m.heights.forEach((h, i) => colors.set(mixRGB(lo, hi, 0.15 + 0.5 * (h / zMax)), 3 * i));
    return { positions, colors, indices: m.indices };
  }, [theme.dark, theme.phys]);

  const lines = useMemo<ColoredLine[]>(() => {
    const out: ColoredLine[] = [];
    for (const m of MODELS) {
      const tr = runs[m.id];
      const c = linear(colorOf(theme, m.id));
      const lift = m.id === 'exact' ? 0.04 : m.id === 'gradient' ? 0.07 : 0.1;
      const at = (i: number): [number, number, number] => [tr.x[i]!, tr.y[i]!, potential([tr.x[i]!, tr.y[i]!]) * Z_SCALE + lift];
      const start = Math.max(1, k - 900);
      for (let i = start; i <= k; i++) {
        const a = 0.25 + 0.75 * ((i - start) / Math.max(1, k - start));
        out.push({ a: at(i - 1), b: at(i), rgba: [c[0], c[1], c[2], a] });
      }
    }
    return out;
  }, [runs, k, theme]);

  const balls = useMemo<AtomBuffers>(() => {
    const count = MODELS.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    MODELS.forEach((m, j) => {
      const tr = runs[m.id];
      positions.set([tr.x[k]!, tr.y[k]!, potential([tr.x[k]!, tr.y[k]!]) * Z_SCALE + 0.18], 3 * j);
      colors.set(linear(colorOf(theme, m.id)), 3 * j);
    });
    return { positions, radii: new Float32Array(count).fill(0.17), colors, metal: new Float32Array(count), ao: new Float32Array(count).fill(1), count };
  }, [runs, k, theme]);

  return (
    <>
      <ShadedSurface data={surface} uniforms={uniforms} />
      <Lines lines={lines} />
      <ImpostorAtoms buffers={balls} version={k} uniforms={uniforms} />
    </>
  );
}

function EnergyPlot({ theme, runs, k }: { theme: SceneTheme; runs: Record<ForceModel, Trajectory>; k: number }) {
  const W = 520;
  const H = 330;
  const M = { l: 56, r: 16, t: 24, b: 40 };
  const tr = runs.exact;
  const n = tr.t.length;
  const series = MODELS.map((m) => ({ m: m.id, d: Array.from(runs[m.id].eRef, (v) => v - runs[m.id].eRef[0]!) }));
  let lo = Math.min(-0.05, ...series.flatMap((s) => s.d));
  let hi = Math.max(0.05, ...series.flatMap((s) => s.d));
  const pad = 0.06 * (hi - lo);
  lo -= pad;
  hi += pad;
  const X = (t: number) => M.l + (t / T_END) * (W - M.l - M.r);
  const Y = (e: number) => M.t + ((hi - e) / (hi - lo)) * (H - M.t - M.b);
  const ticks = niceTicks(lo, hi, 5);
  const path = (d: number[]) => {
    const stride = Math.max(1, Math.floor(n / 600));
    let s = '';
    for (let i = 0; i < n; i += stride) s += `${i ? 'L' : 'M'}${X(tr.t[i]!).toFixed(1)},${Y(d[i]!).toFixed(1)}`;
    return s + `L${X(tr.t[n - 1]!).toFixed(1)},${Y(d[n - 1]!).toFixed(1)}`;
  };
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Energy change against time for the three force models.">
        <text x={M.l} y={M.t - 9} className="axis-label">energy change E(t) − E(0), reference energy ½|v|² + V</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {ticks.map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={Y(e)} y2={Y(e)} className="grid" />
            <text x={M.l - 6} y={Y(e) + 4} textAnchor="end" className="tick">{fmt(e)}</text>
          </g>
        ))}
        {series.map((s) => (
          <path key={s.m} d={path(s.d)} className="curve" style={{ stroke: colorOf(theme, s.m), strokeDasharray: s.m === 'gradient' ? '6 3' : undefined }} />
        ))}
        <line x1={X(tr.t[k]!)} x2={X(tr.t[k]!)} y1={M.t} y2={H - M.b} className="r-line" />
        {series.map((s) => (
          <circle key={s.m} cx={X(tr.t[k]!)} cy={Y(s.d[k]!)} r={4} style={{ fill: colorOf(theme, s.m) }} />
        ))}
        {[0, 50, 100, 150].map((t) => (
          <text key={t} x={X(t)} y={H - M.b + 16} textAnchor="middle" className="tick">{t}</text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">time (reduced units)</text>
      </svg>
      <figcaption className="lab-legend">
        {MODELS.map((m) => (
          <span key={m.id} className="item"><span className="swatch" style={{ background: colorOf(theme, m.id) }} /> {m.label}</span>
        ))}
      </figcaption>
    </figure>
  );
}

function LoopField({ theme, model, sigma, circ }: { theme: SceneTheme; model: 'gradient' | 'direct'; sigma: number; circ: number }) {
  const W = 360;
  const H = 300;
  const R = 2.4;
  const S = (H - 40) / (2 * R);
  const cx = W / 2;
  const cy = (H - 20) / 2 + 4;
  const field = model === 'gradient' ? deltaF1Gradient : deltaF1Curl;
  const color = colorOf(theme, model);
  const arrows: { x: number; y: number; dx: number; dy: number }[] = [];
  const g = 9;
  let maxLen = 0;
  for (let i = 0; i <= g; i++)
    for (let j = 0; j <= g; j++) {
      const x = -2 + (4 * i) / g;
      const y = -2 + (4 * j) / g;
      if (x * x + y * y > 4.4) continue;
      const f = field([x, y]);
      maxLen = Math.max(maxLen, Math.hypot(f[0], f[1]));
      arrows.push({ x, y, dx: f[0], dy: f[1] });
    }
  const scale = (0.42 * (4 / g)) / maxLen;
  const title = model === 'gradient' ? 'error δF = −∇δV (no curl)' : 'error δF ∝ (−y, x) (curl 2)';
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Force-error field of the ${model === 'gradient' ? 'gradient-error' : 'curl-error'} model and its circulation round a circle: ${circ.toExponential(2)}.`}>
        <defs>
          <marker id={`nc-head-${model}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="3.5" markerHeight="3.5" orient="auto-start-reverse">
            <path d="M0,0L10,5L0,10z" style={{ fill: color }} />
          </marker>
        </defs>
        <text x={12} y={16} className="axis-label">{title}</text>
        <circle cx={cx} cy={cy} r={2 * S} className="frame" style={{ strokeDasharray: '3 4' }} />
        <circle cx={cx} cy={cy} r={LOOP_R * S} fill="none" style={{ stroke: theme.ink, strokeWidth: 1.6 }} />
        {arrows.map((a, i) => (
          <line
            key={i}
            x1={cx + a.x * S}
            y1={cy - a.y * S}
            x2={cx + (a.x + a.dx * scale) * S}
            y2={cy - (a.y + a.dy * scale) * S}
            style={{ stroke: color, strokeWidth: 1.4 }}
            markerEnd={`url(#nc-head-${model})`}
          />
        ))}
        <text x={W - 12} y={H - 8} textAnchor="end" className="tick">
          ∮ F·dl round r = {LOOP_R}: {sigma === 0 ? '0' : fmtSci(circ)}
          {model === 'direct' && sigma > 0 ? ` = 2πR²a, a = ${amplitudeFor('direct', sigma).toFixed(4)}` : ''}
        </text>
      </svg>
    </figure>
  );
}

function fmtSci(v: number): string {
  return Math.abs(v) < 1e-9 ? '0 to round-off' : v.toFixed(4);
}

function fmt(v: number): string {
  const a = Math.abs(v);
  const s = a >= 10 ? a.toFixed(0) : a >= 1 ? a.toFixed(1) : a.toFixed(2);
  return v < 0 ? `−${s}` : s;
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const raw = (hi - lo) / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
