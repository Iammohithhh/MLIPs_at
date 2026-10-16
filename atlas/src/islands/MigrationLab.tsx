import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, linear, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { KB, diffusivity, diffusivityRatio, walk } from '~/lib/physics/hopping';

// THEORY + TOY MODEL: two ions hop on the same lattice for the same time; one sees the reference
// barrier, the other a barrier lowered by the model's error δ. Rates follow ν·exp(−E/k_BT).

const N_REF = 80;
const MAX_DRAWN = 2500;
const SITE = 1.6; // drawing scale per lattice spacing

// Published error sizes, used only as example magnitudes of δ (claims ion-migration-*).
const PRESETS = [
  { delta: 0.31, label: '0.31 eV · MACE-MP-0 MAE, 574 barriers' },
  { delta: 0.089, label: '0.089 eV · MACE + sAlex + OMat24 MAE' },
  { delta: 0.06, label: '0.06 eV · typical DFT-NEB error' },
];

export default function MigrationLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [em, setEm] = useState(0.5);
  const [delta, setDelta] = useState(0.31);
  const [T, setT] = useState(300);
  const [progress, setProgress] = useState(1);
  const [playing, setPlaying] = useState(false);
  const ratio = diffusivityRatio(delta, T);
  const nModel = Math.max(1, Math.round(N_REF * ratio));
  const refWalk = useMemo(() => walk(N_REF, 11), []);
  const modelWalk = useMemo(() => walk(Math.min(nModel, MAX_DRAWN), 29), [nModel]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const start = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / 6000);
      setProgress(p);
      if (p < 1) raf = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    setProgress(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const shownRef = Math.round(progress * N_REF);
  const shownModel = Math.round(progress * Math.min(nModel, MAX_DRAWN));
  const fmt = (x: number) => {
    if (x < 1e4 && x >= 1e-2) return x.toFixed(x < 10 ? 1 : 0);
    const [m, e] = x.toExponential(1).split('e');
    const sup = Number(e).toString().replace(/[-0-9]/g, (c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]!);
    return `${m} × 10${sup}`;
  };

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [34, -60, 40], target: [0, 0, 0] }} background={theme.canvas} label={`Two ions hopping for the same time: the model's ion makes ${fmt(ratio)} times as many hops.`} minDistance={10} maxDistance={220}>
            <WalkScene theme={theme} refWalk={refWalk} modelWalk={modelWalk} nRef={shownRef} nModel={shownModel} />
          </Stage>
          <div className="lab-toolbar">
            <button type="button" className="lab-btn" aria-pressed={playing} onClick={() => setPlaying(true)}>
              Replay the same time window
            </button>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="bar" style={{ background: theme.phys }} /> reference barrier: {N_REF} hops</li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> model barrier: {nModel.toLocaleString('en-US')} hops
              {nModel > MAX_DRAWN ? ` (first ${MAX_DRAWN.toLocaleString('en-US')} drawn)` : ''}
            </li>
          </ul>
        </div>
        <ArrheniusPlot theme={theme} em={em} delta={delta} T={T} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-em`}>
            <span>reference barrier E_m</span>
            <input id={`${ids}-em`} type="range" min={0.15} max={1.2} step={0.01} value={em} onChange={(e) => setEm(Number(e.target.value))} />
            <output>{em.toFixed(2)} eV</output>
          </label>
          <label htmlFor={`${ids}-d`}>
            <span>model underestimates it by δ</span>
            <input id={`${ids}-d`} type="range" min={-0.4} max={0.4} step={0.005} value={delta} onChange={(e) => setDelta(Number(e.target.value))} />
            <output>{delta.toFixed(3)} eV</output>
          </label>
          <label htmlFor={`${ids}-T`}>
            <span>temperature</span>
            <input id={`${ids}-T`} type="range" min={250} max={900} step={5} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <output>{T} K</output>
          </label>
          <div className="presets" role="group" aria-label="Published error sizes">
            <span className="seg-label">Set δ to a published error size:</span>
            {PRESETS.map((p) => (
              <button key={p.delta} type="button" className="lab-btn" aria-pressed={Math.abs(delta - p.delta) < 1e-9} onClick={() => setDelta(p.delta)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={ratio > 10 || ratio < 0.1 ? 'is-alert' : undefined}>
            <dt>D(model) / D(reference)</dt>
            <dd>{fmt(ratio)}</dd>
          </div>
          <div>
            <dt>k_BT</dt>
            <dd>{(KB * T * 1000).toFixed(1)} meV</dd>
          </div>
          <div>
            <dt>D, reference (toy)</dt>
            <dd>{diffusivity(em, T).toExponential(1)} cm²/s</dd>
          </div>
          <div>
            <dt>D, model (toy)</dt>
            <dd>{diffusivity(em - delta, T).toExponential(1)} cm²/s</dd>
          </div>
        </dl>
        <p className="lab-note">
          Every k_BT·ln 10 = {(KB * T * Math.log(10) * 1000).toFixed(0)} meV of barrier error is one order of magnitude in the
          diffusion coefficient at {T} K. A mean error is not a signed error: the presets show how large δ can be, not that
          every model errs in this direction.
        </p>
      </div>
    </div>
  );
}

function WalkScene({ theme, refWalk, modelWalk, nRef, nModel }: { theme: SceneTheme; refWalk: Array<[number, number, number]>; modelWalk: Array<[number, number, number]>; nRef: number; nModel: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 40, 200);
  const sp = (p: [number, number, number]): [number, number, number] => [p[0] * SITE, p[1] * SITE, p[2] * SITE];
  const lines = useMemo<ColoredLine[]>(() => {
    const out: ColoredLine[] = [];
    const a = linear(theme.caveat);
    const b = linear(theme.phys);
    for (let i = 1; i <= nModel; i++) out.push({ a: sp(modelWalk[i - 1]!), b: sp(modelWalk[i]!), rgba: [a[0], a[1], a[2], 0.55] });
    for (let i = 1; i <= nRef; i++) out.push({ a: sp(refWalk[i - 1]!), b: sp(refWalk[i]!), rgba: [b[0], b[1], b[2], 1] });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refWalk, modelWalk, nRef, nModel, theme]);
  const buffers = useMemo<AtomBuffers>(() => {
    const R = 4;
    const sites: Array<[number, number, number]> = [];
    for (let i = -R; i <= R; i++) for (let j = -R; j <= R; j++) for (let k = -R; k <= R; k++) sites.push([i, j, k]);
    const n = sites.length + 3;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const host = linear(theme.dark ? '#5a6370' : '#c3c9d2');
    sites.forEach((s, i) => {
      pos.set(sp(s), 3 * i);
      radii[i] = 0.12;
      colors.set(host, 3 * i);
    });
    const put = (k: number, p: [number, number, number], c: string, r: number) => {
      pos.set(sp(p), 3 * k);
      radii[k] = r;
      colors.set(linear(c), 3 * k);
    };
    put(sites.length, [0, 0, 0], theme.ink, 0.3);
    put(sites.length + 1, refWalk[nRef]!, theme.phys, 0.55);
    put(sites.length + 2, modelWalk[nModel]!, theme.caveat, 0.55);
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refWalk, modelWalk, nRef, nModel, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
    </>
  );
}

function ArrheniusPlot({ theme, em, delta, T }: { theme: SceneTheme; em: number; delta: number; T: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 58, r: 16, t: 24, b: 44 };
  const x0 = 1.0;
  const x1 = 4.0;
  const lg = (v: number) => Math.log10(v);
  const ys = [em, em - delta].flatMap((e) => [lg(diffusivity(e, 1000 / x0)), lg(diffusivity(e, 1000 / x1))]);
  const ymax = Math.ceil(Math.max(...ys));
  const ymin = Math.max(Math.floor(Math.min(...ys)), ymax - 24);
  const X = (x: number) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const Y = (y: number) => M.t + ((ymax - Math.max(ymin, y)) / (ymax - ymin)) * (H - M.t - M.b);
  const line = (e: number) => `M${X(x0)},${Y(lg(diffusivity(e, 1000 / x0)))}L${X(x1)},${Y(lg(diffusivity(e, 1000 / x1)))}`;
  const step = Math.max(1, Math.ceil((ymax - ymin) / 8));
  const ticks = Array.from({ length: Math.floor((ymax - ymin) / step) + 1 }, (_, i) => ymax - i * step);
  const xt = 1000 / T;
  const yr = lg(diffusivity(em, T));
  const ym = lg(diffusivity(em - delta, T));
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Arrhenius plot of the toy diffusion coefficient for the reference and model barriers; at ${T} K they differ by a factor ${diffusivityRatio(delta, T).toPrecision(2)}.`}>
        <text x={M.l} y={M.t - 9} className="axis-label">log₁₀ D [cm²/s] (toy: a = 3 Å, ν = 10¹³ s⁻¹)</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className="grid" />
            <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <path d={line(em)} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.4 }} />
        <path d={line(em - delta)} className="curve" style={{ stroke: theme.caveat, strokeWidth: 2.4, strokeDasharray: '6 3' }} />
        <line x1={X(xt)} x2={X(xt)} y1={M.t} y2={H - M.b} className="r-line" />
        <line x1={X(xt)} x2={X(xt)} y1={Y(yr)} y2={Y(ym)} style={{ stroke: theme.caveat, strokeWidth: 3, opacity: 0.6 }} />
        <text x={X(xt) + 8} y={(Y(yr) + Y(ym)) / 2 + 4} className="tick" style={{ fill: theme.caveat, fontWeight: 600 }}>
          {Math.abs(ym - yr).toFixed(1)} decades
        </text>
        <circle cx={X(xt)} cy={Y(yr)} r={5} style={{ fill: theme.phys }} />
        <circle cx={X(xt)} cy={Y(ym)} r={5} style={{ fill: theme.caveat }} />
        {[1, 1.5, 2, 2.5, 3, 3.5, 4].map((v) => (
          <text key={v} x={X(v)} y={H - M.b + 16} textAnchor="middle" className="tick">{v}</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">1000 / T [K⁻¹]</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> reference barrier E_m</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> model barrier E_m − δ</span>
      </figcaption>
    </figure>
  );
}
