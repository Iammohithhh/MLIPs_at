// Atom Intervention Lab (first version): the moved atom's neighbourhood in 3D, a displacement control
// that snaps to evaluated frames, baseline / perturbed / difference views, and linked energy and force
// plots. All numbers come from the precomputed MACE-MP-0a medium evaluations in the manifest.

import { useEffect, useMemo, useState } from 'react';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import {
  axialForce,
  blockScaleMax,
  energyChange,
  featureChange,
  force,
  forceChange,
  norm,
  parseManifest,
  siteEnergyChange,
  type Vec3,
} from '../lib/experiment';
import { neighbourhood, bonds, graphEdges } from '../lib/scene';
import { decodeState, encodeState, type View } from '../lib/urlState';
import { useExperiment } from '../lib/store';
import { Stage } from '../three/Stage';
import { Arrows, AtomPicker, ImpostorAtoms, Lines, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '../three/primitives';
import { usePrefersReducedMotion } from '../three/theme';
import { FeatureInspector } from './FeatureInspector';
import { SceneGuard } from './SceneGuard';
import { ResponsePlots } from './ResponsePlots';
import { StageDetail, StageNav } from './StagePanel';
import './InterventionLab.css';

const m = parseManifest(raw);
const CHECKPOINT = String(m.provenance.checkpoint);
const DISPLAY_RADIUS = 6.0; // Å: moved atom, its graph neighbours within r_max
const FORCE_SCALE = 0.5; // Å of arrow per eV/Å: the largest stored force (|F| = 4.7 eV/Å, moved atom at d = +0.3 Å) draws as a 2.3 Å arrow
const B = m.displacement.baselineIndex;
const moved = m.displacement.atom;
// Atoms with full channel data (moved atom + first two shells), labelled by baseline distance.
const LOCAL_OPTIONS = (neighbourhood(m, B, 4.0) ?? []).map((a) => ({
  index: a.index,
  label: a.index === moved ? `${a.index}: moved atom` : `${a.index}: ${a.r0.toFixed(2)} Å${a.index === m.displacement.partner ? ' (bond partner)' : ''}`,
}));

const COLORS = {
  stage: '#141a19',
  atom: '#a9b4b8',
  mint: '#6fd3bd',
  teal: '#3a9f8c',
  amber: '#f0a945',
  edge: '#5f7f78',
};

function fmt(x: number | null, digits = 3, unit = ''): string {
  if (x === null || !Number.isFinite(x)) return '—';
  const s = Math.abs(x) < 10 ** -digits && x !== 0 ? x.toExponential(2) : x.toFixed(digits);
  return `${x > 0 ? '+' : x < 0 ? '−' : ''}${s.replace('-', '')}${unit ? ` ${unit}` : ''}`;
}

export default function InterventionLab() {
  const { s, init, set } = useExperiment();
  const [issues, setIssues] = useState<string[]>([]);
  const [showEdges, setShowEdges] = useState(true);
  const [showShell, setShowShell] = useState(false);
  const [colourMode, setColourMode] = useState<'element' | 'change'>('element');
  const [resetKey, setResetKey] = useState(0);
  const [camera, setCamera] = useState<CameraPreset>('overview');
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const r = decodeState(window.location.search.slice(1), m, CHECKPOINT);
    init(r.state);
    setIssues(r.issues);
  }, [init]);

  useEffect(() => {
    if (!s) return;
    const q = encodeState(s);
    window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
  }, [s]);

  if (!s) return <div className="lab lab--loading" role="status">Loading the precomputed evaluations…</div>;

  const frame = s.frame;
  const view: View = s.view;
  const shown = view === 'baseline' ? B : frame;
  const d = m.displacement.values[frame] ?? 0;

  return (
    <section className="lab" aria-labelledby="lab-title">
      <header className="lab__head">
        <h2 id="lab-title">What changes when this atom moves?</h2>
        <div className="lab__status">
          <span className="label status-pre">Precomputed model evaluations; interactive inspection</span>
          <span className="label status-ck mono" title={`SHA-256 ${String(m.provenance.sha256)}`}>
            {CHECKPOINT} · sha256 {String(m.provenance.sha256).slice(0, 8)}…
          </span>
        </div>
      </header>
      {issues.length > 0 && (
        <p className="lab__issues" role="alert">
          This link could not be restored exactly: {issues.join('; ')}.
        </p>
      )}
      <div className="lab__modes" role="tablist" aria-label="Mode">
        <button role="tab" aria-selected={s.mode === 'lab'} className={s.mode === 'lab' ? 'is-on' : ''} onClick={() => set({ mode: 'lab' })}>
          Move an atom
        </button>
        <button role="tab" aria-selected={s.mode === 'walk'} className={s.mode === 'walk' ? 'is-on' : ''} onClick={() => set({ mode: 'walk' })}>
          Walk me through it
        </button>
      </div>
      {s.mode === 'walk' && <StageNav stage={s.stage} onStage={(st) => set({ stage: st })} />}
      <div className="lab__grid">
        <div className="lab__stage">
          <SceneGuard>
            <Scene shown={shown} view={view} selected={s.atom} showEdges={s.mode === 'walk' ? ['neighbours', 'radial', 'messages'].includes(s.stage) : showEdges} showShell={s.mode === 'walk' ? s.stage === 'neighbours' : showShell} showForces={s.mode !== 'walk' || s.stage === 'forces'} colourMode={colourMode} layer={s.layer} resetKey={resetKey} camera={camera} reduced={reduced} onPick={(i) => set({ atom: i })} />
          </SceneGuard>
          <ForceScale />
          <ColourLegend mode={colourMode} layer={s.layer} />
          <div className="lab__views" role="radiogroup" aria-label="Comparison view">
            {(['baseline', 'perturbed', 'difference'] as const).map((v) => (
              <button key={v} role="radio" aria-checked={view === v} className={view === v ? 'is-on' : ''} onClick={() => set({ view: v })}>
                {v === 'baseline' ? 'Baseline' : v === 'perturbed' ? 'Perturbed' : 'Difference'}
              </button>
            ))}
          </div>
        </div>
        <aside className="lab__panel">
          {s.mode === 'walk' && <StageDetail stage={s.stage} onStage={(st) => set({ stage: st })} />}
          <Displacement frame={frame} onFrame={(k) => set({ frame: k })} />
          <Readouts frame={frame} atom={s.atom} />
          <fieldset className="lab__toggles">
            <legend>Scene</legend>
            <label>
              <input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} /> Graph edges into the selected atom (r &lt; {m.rMax} Å)
            </label>
            <label>
              <input type="checkbox" checked={showShell} onChange={(e) => setShowShell(e.target.checked)} /> Cutoff sphere
            </label>
            <label>
              Colour{' '}
              <select value={colourMode} onChange={(e) => setColourMode(e.target.value as 'element' | 'change')}>
                <option value="element">by element (Si)</option>
                <option value="change">by feature change, layer {s.layer + 1} scalars</option>
              </select>
            </label>
            <label>
              Atom{' '}
              <select value={s.atom} onChange={(e) => set({ atom: Number(e.target.value) })} aria-label="Selected atom">
                {LOCAL_OPTIONS.map((o) => (
                  <option key={o.index} value={o.index}>
                    {o.label}
                  </option>
                ))}
                {!LOCAL_OPTIONS.some((o) => o.index === s.atom) && <option value={s.atom}>atom {s.atom} (picked in scene)</option>}
              </select>
            </label>
            <label>
              Layer{' '}
              <select value={s.layer} onChange={(e) => set({ layer: Number(e.target.value) })}>
                {m.baseline.norms.map((_, t) => (
                  <option key={t} value={t}>
                    {t + 1}
                  </option>
                ))}
              </select>
            </label>
          </fieldset>
          <div className="lab__actions">
            <button onClick={() => set({ frame: B, view: 'perturbed', atom: moved })}>Reset experiment</button>
            <button onClick={() => { setCamera('overview'); setResetKey((x) => x + 1); }}>Reset camera</button>
          </div>
          <div className="lab__actions" role="group" aria-label="Camera presets">
            {(Object.keys(PRESET_LABEL) as CameraPreset[]).map((c) => (
              <button key={c} aria-pressed={camera === c} onClick={() => { setCamera(c); setResetKey((x) => x + 1); }}>
                {PRESET_LABEL[c]}
              </button>
            ))}
          </div>
        </aside>
      </div>
      <section className="lab__inspect" aria-labelledby="inspect-title">
        <h3 id="inspect-title">Which parts of the model respond?</h3>
        <ResponsePlots m={m} atom={s.atom} frame={frame} onFrame={(k) => set({ frame: k })} />
        <FeatureInspector m={m} frame={frame} atom={s.atom} layer={s.layer} channel={s.channel} onChannel={(c) => set({ channel: c })} />
      </section>
      <Plots frame={frame} onFrame={(k) => set({ frame: k })} d={d} />
      <p className="lab__scope">
        These are observed changes in a fixed model under a specified input perturbation. They are not explanations of chemical reasoning,
        confidence estimates, prediction errors, or calibrated importance.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------

function Scene(props: {
  shown: number;
  view: View;
  selected: number;
  showEdges: boolean;
  showShell: boolean;
  showForces: boolean;
  colourMode: 'element' | 'change';
  layer: number;
  resetKey: number;
  camera: CameraPreset;
  reduced: boolean;
  onPick: (i: number) => void;
}) {
  const { shown, view, selected, showEdges, showShell, colourMode, layer } = props;
  const atoms = useMemo(() => neighbourhood(m, shown, DISPLAY_RADIUS) ?? [], [shown]);
  const bondPairs = useMemo(() => bonds(atoms), [atoms]);
  const scaleMax = useMemo(() => blockScaleMax(m, layer, 'h0') ?? 1, [layer]);

  const buffers = useMemo<AtomBuffers>(() => {
    const n = atoms.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const ao = new Float32Array(n);
    const base = linear(COLORS.atom);
    const amber = linear(COLORS.amber);
    const mint = linear(COLORS.mint);
    atoms.forEach((a, i) => {
      positions.set(a.pos, i * 3);
      radii[i] = a.index === moved ? 0.62 : 0.5;
      let c = base;
      if (colourMode === 'change') {
        const v = featureChange(m, shown, layer, 'h0', a.index);
        c = v === null ? [0.2, 0.2, 0.2] : mixRGB(base, amber, Math.min(1, v / scaleMax));
      }
      if (a.index === moved) c = mint;
      colors.set(c, i * 3);
      ao[i] = 0.85 + 0.15 * Math.max(0, 1 - a.r0 / DISPLAY_RADIUS);
    });
    return { positions, radii, colors, metal, ao, count: n };
  }, [atoms, colourMode, layer, shown, scaleMax]);

  const sticks = useMemo<Segment[]>(() => {
    const c = linear('#59656a');
    return bondPairs.map(([a, b]) => ({ a: atoms[a]!.pos, b: atoms[b]!.pos, color: c, radius: 0.09 }));
  }, [bondPairs, atoms]);

  const arrows = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    if (!props.showForces) return out;
    const teal = linear(COLORS.mint);
    const amber = linear(COLORS.amber);
    for (const a of atoms) {
      const f = view === 'difference' ? forceChange(m, props.shown, a.index) : force(m, props.shown, a.index);
      if (!f) continue;
      const mag = norm(f);
      if (mag < 0.12) continue;
      out.push({ a: a.pos, b: [a.pos[0] + f[0] * FORCE_SCALE, a.pos[1] + f[1] * FORCE_SCALE, a.pos[2] + f[2] * FORCE_SCALE], color: view === 'difference' ? amber : teal, radius: 0.075 });
    }
    return out;
  }, [atoms, view, props.shown, props.showForces]);

  const lines = useMemo(() => {
    const out: ColoredLine[] = [];
    // Evaluated displacement path, with a tick per frame.
    const u = m.displacement.axis;
    const vals = m.displacement.values;
    const p0: Vec3 = [u[0] * vals[0]!, u[1] * vals[0]!, u[2] * vals[0]!];
    const p1: Vec3 = [u[0] * vals.at(-1)!, u[1] * vals.at(-1)!, u[2] * vals.at(-1)!];
    const amber = linear(COLORS.amber);
    out.push({ a: p0, b: p1, rgba: [...amber, 0.9] });
    if (showEdges) {
      const sel = atoms.find((a) => a.index === selected);
      const edge = linear(COLORS.edge);
      if (sel)
        for (const e of graphEdges(m, props.shown, atoms, selected)) {
          const t = atoms.find((a) => a.index === e.j)!;
          out.push({ a: sel.pos, b: t.pos, rgba: [...edge, 0.55] });
        }
    }
    return out;
  }, [atoms, selected, showEdges, props.shown]);

  const sel = atoms.find((a) => a.index === selected);

  return (
    <div className="lab__canvas">
      <Stage view={cameraView(props.camera, sel?.pos ?? [0, 0, 0])} resetKey={props.resetKey} background={COLORS.stage} label="Silicon neighbourhood of the moved atom with force arrows" minDistance={6} maxDistance={60}>
        <SceneContents onPick={(i) => i !== null && atoms[i] && props.onPick(atoms[i]!.index)} buffers={buffers} version={shown * 7 + layer + (colourMode === 'change' ? 1000 : 0)} sticks={sticks} arrows={arrows} lines={lines} sel={sel?.pos ?? null} showShell={showShell} />
      </Stage>
    </div>
  );
}

type CameraPreset = 'overview' | 'environment' | 'axis';
const PRESET_LABEL: Record<CameraPreset, string> = { overview: 'Overview', environment: 'Selected environment', axis: 'Side-on to the path' };

/** Camera presets keep z up; 'axis' looks perpendicular to the displacement axis so the motion is in the image plane. */
function cameraView(preset: CameraPreset, sel: Vec3): { position: [number, number, number]; target: [number, number, number] } {
  if (preset === 'environment') return { position: [sel[0] + 7, sel[1] - 9, sel[2] + 5], target: sel };
  if (preset === 'axis') {
    const u = m.displacement.axis;
    // perpendicular to û, as horizontal as possible
    const p: Vec3 = [-u[1], u[0], 0];
    const n = Math.hypot(p[0], p[1]) || 1;
    return { position: [(p[0] / n) * 24, (p[1] / n) * 24, 4], target: [0, 0, 0] };
  }
  return { position: [16, -20, 11], target: [0, 0, 0] };
}

function SceneContents(p: {
  buffers: AtomBuffers;
  version: number;
  sticks: Segment[];
  arrows: Segment[];
  lines: ColoredLine[];
  sel: Vec3 | null;
  showShell: boolean;
  onPick: (i: number | null) => void;
}) {
  // Hooks that need the R3F context live inside <Stage>.
  const uniforms = useShadingUniforms(COLORS.stage, true, 14, 40);
  return (
    <>
      <ImpostorAtoms buffers={p.buffers} version={p.version} uniforms={uniforms} />
      <AtomPicker positions={p.buffers.positions} radii={p.buffers.radii} count={p.buffers.count} onPick={p.onPick} />
      <Sticks segments={p.sticks} uniforms={uniforms} />
      <Arrows arrows={p.arrows} uniforms={uniforms} head={0.3} />
      <Lines lines={p.lines} />
      {p.sel && <Shell center={p.sel} radius={0.85} color={COLORS.mint} opacity={0.35} overlay />}
      {p.showShell && p.sel && <Shell center={p.sel} radius={m.rMax} color={COLORS.teal} opacity={0.12} grid />}
    </>
  );
}

function ForceScale() {
  return (
    <div className="lab__scale" aria-label={`Force scale: ${FORCE_SCALE} Å of arrow per 1 eV/Å`}>
      <span className="lab__scale-bar" style={{ width: '24px' }} aria-hidden="true" />
      <span className="mono">force arrows: {FORCE_SCALE} Å of length per eV/Å (in scene units, not screen pixels); arrows under 0.12 eV/Å hidden</span>
    </div>
  );
}

function ColourLegend({ mode, layer }: { mode: 'element' | 'change'; layer: number }) {
  if (mode === 'element')
    return (
      <div className="lab__legend">
        <span className="swatch" style={{ background: COLORS.atom }} /> Si <span className="swatch" style={{ background: COLORS.mint }} /> moved atom
        <span className="lab__legend-note">edges: model graph (r &lt; {m.rMax} Å), sticks: Si–Si bonds &lt; 2.6 Å</span>
      </div>
    );
  const max = blockScaleMax(m, layer, 'h0');
  return (
    <div className="lab__legend">
      <span className="ramp" /> ‖Δh‖ layer {layer + 1}, 128 scalar channels: 0 – {max?.toPrecision(3)} (fixed over all frames)
    </div>
  );
}

function Displacement({ frame, onFrame }: { frame: number; onFrame: (k: number) => void }) {
  const vals = m.displacement.values;
  const d = vals[frame] ?? 0;
  return (
    <div className="lab__disp">
      <label htmlFor="disp" className="lab__disp-label">
        Move atom {moved} along {m.displacement.axisLabel}
      </label>
      <div className="lab__disp-row">
        <button aria-label="Previous evaluated frame" onClick={() => onFrame(Math.max(0, frame - 1))} disabled={frame === 0}>
          −
        </button>
        <input
          id="disp"
          type="range"
          min={0}
          max={vals.length - 1}
          step={1}
          value={frame}
          aria-valuetext={`${d.toFixed(3)} ångström`}
          onChange={(e) => onFrame(Number(e.target.value))}
        />
        <button aria-label="Next evaluated frame" onClick={() => onFrame(Math.min(vals.length - 1, frame + 1))} disabled={frame === vals.length - 1}>
          +
        </button>
      </div>
      <output htmlFor="disp" className="lab__disp-value mono">
        d = {d >= 0 ? '+' : '−'}
        {Math.abs(d).toFixed(3)} Å <span className="lab__disp-frame">frame {frame + 1}/{vals.length}</span>
      </output>
      <p className="lab__hint">Snaps to the {vals.length} evaluated frames, {Math.abs(vals[1]! - vals[0]!).toFixed(3)} Å apart. d &gt; 0 shortens the bond to atom {m.displacement.partner}.</p>
    </div>
  );
}

function Readouts({ frame, atom }: { frame: number; atom: number }) {
  const dE = energyChange(m, frame);
  const F = force(m, frame, atom);
  const dEi = siteEnergyChange(m, frame, atom);
  const fa = axialForce(m, frame);
  return (
    <dl className="lab__readouts">
      <div>
        <dt>ΔE total</dt>
        <dd className="mono delta">{fmt(dE, 4, 'eV')}</dd>
      </div>
      <div>
        <dt>F on moved atom · û</dt>
        <dd className="mono">{fmt(fa, 3, 'eV/Å')}</dd>
      </div>
      <div>
        <dt>|F| on atom {atom}</dt>
        <dd className="mono">{F ? `${norm(F).toFixed(3)} eV/Å` : '—'}</dd>
      </div>
      <div>
        <dt>ΔEᵢ, atom {atom} site energy</dt>
        <dd className="mono delta">{fmt(dEi, 4, 'eV')}</dd>
      </div>
    </dl>
  );
}

function Plots({ frame, onFrame, d }: { frame: number; onFrame: (k: number) => void; d: number }) {
  const vals = m.displacement.values;
  const W = 300;
  const H = 150;
  const pad = { l: 46, r: 10, t: 12, b: 30 };
  const x = (v: number) => pad.l + ((v - vals[0]!) / (vals.at(-1)! - vals[0]!)) * (W - pad.l - pad.r);
  const series = [
    { key: 'E', title: 'ΔE total (eV)', get: (k: number) => energyChange(m, k) },
    { key: 'F', title: 'F·û on moved atom (eV/Å)', get: (k: number) => axialForce(m, k) },
  ];
  return (
    <div className="lab__plots">
      {series.map((sr) => {
        const ys = vals.map((_, k) => sr.get(k));
        const finite = ys.filter((v): v is number => v !== null);
        const lo = Math.min(0, ...finite);
        const hi = Math.max(0, ...finite);
        const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
        return (
          <figure key={sr.key} className="lab__plot">
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${sr.title} against displacement; selected d = ${d.toFixed(3)} Å`}>
              <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} className="axis" />
              <line x1={x(0)} x2={x(0)} y1={pad.t} y2={H - pad.b} className="axis" />
              <line x1={x(d)} x2={x(d)} y1={pad.t} y2={H - pad.b} className="cursor" />
              {ys.map((v, k) =>
                v === null ? (
                  <text key={k} x={x(vals[k]!)} y={H - pad.b - 2} className="missing">
                    ×
                  </text>
                ) : (
                  <circle key={k} cx={x(vals[k]!)} cy={y(v)} r={k === frame ? 4.5 : 2.6} className={k === frame ? 'pt is-on' : 'pt'} onClick={() => onFrame(k)}>
                    <title>
                      d = {vals[k]!.toFixed(3)} Å: {v.toPrecision(4)}
                    </title>
                  </circle>
                ),
              )}
              <text x={pad.l - 6} y={y(hi) + 4} className="tick" textAnchor="end">
                {hi.toPrecision(2)}
              </text>
              <text x={pad.l - 6} y={y(lo) + 4} className="tick" textAnchor="end">
                {lo.toPrecision(2)}
              </text>
              <text x={x(vals[0]!)} y={H - 10} className="tick" textAnchor="start">
                {vals[0]!.toFixed(2)}
              </text>
              <text x={x(vals.at(-1)!)} y={H - 10} className="tick" textAnchor="end">
                +{vals.at(-1)!.toFixed(2)} Å
              </text>
            </svg>
            <figcaption>
              {sr.title}. Points are evaluated frames; there are no values between them.
              {sr.key === 'F' && ' F = −∂E/∂r for the total energy E.'}
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}
