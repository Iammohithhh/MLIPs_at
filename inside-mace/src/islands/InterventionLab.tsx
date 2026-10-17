// Atom Intervention Lab (first version): the moved atom's neighbourhood in 3D, a displacement control
// that snaps to evaluated frames, baseline / perturbed / difference views, and linked energy and force
// plots. All numbers come from the precomputed MACE-MP-0a medium evaluations in the manifest.

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
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
  type Manifest,
  type Vec3,
} from '../lib/experiment';
import { neighbourhood, bonds, graphEdges, BOND_CUTOFF } from '../lib/scene';
import { ELEMENT_DISPLAY, composition, experimentInfo } from '../lib/experimentInfo';
import { decodeState, encodeState, type View } from '../lib/urlState';
import { useExperiment } from '../lib/store';
import { demoSteps } from '../lib/demo';
import { figureText, type FigureText } from '../lib/figureText';
import { Stage } from '../three/Stage';
import { CameraSync } from '../three/cameraSync';
import { Arrows, AtomPicker, ImpostorAtoms, Lines, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '../three/primitives';
import { usePrefersReducedMotion } from '../three/theme';
import { FeatureInspector } from './FeatureInspector';
import { SceneGuard } from './SceneGuard';
import { ResponsePlots } from './ResponsePlots';
import { RotationDemo } from './RotationDemo';
import { StageEvidence } from './StageEvidence';
import { TraceChange } from './TraceChange';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { StageDetail, StageNav } from './StagePanel';
import './InterventionLab.css';

const DISPLAY_RADIUS = 6.0; // Å: moved atom, its graph neighbours within r_max
const FORCE_SCALE = 0.5; // Å of arrow per eV/Å: the largest stored force (|F| = 4.7 eV/Å, moved atom at d = +0.3 Å) draws as a 2.3 Å arrow

// Everything the instrument derives from one experiment's manifest, shared through context so the same
// components serve any committed experiment (the experiment is fixed for the lifetime of the instrument).
interface Lab {
  m: Manifest;
  demo: ReturnType<typeof demoSteps>;
  checkpoint: string;
  B: number;
  moved: number;
  /** Atoms with full channel data (moved atom + first two shells), labelled by baseline distance. */
  localOptions: { index: number; label: string }[];
}

function makeLab(m: Manifest): Lab {
  const B = m.displacement.baselineIndex;
  const moved = m.displacement.atom;
  return {
    m,
    demo: demoSteps(m),
    checkpoint: String(m.provenance.checkpoint),
    B,
    moved,
    localOptions: (neighbourhood(m, B, 4.0) ?? []).map((a) => ({
      index: a.index,
      label: a.index === moved ? `${a.index}: moved atom` : `${a.index}: ${a.r0.toFixed(2)} Å${a.index === m.displacement.partner ? ' (bond partner)' : ''}`,
    })),
  };
}

const LabContext = createContext<Lab | null>(null);
function useLab(): Lab {
  const lab = useContext(LabContext);
  if (!lab) throw new Error('InterventionLab components need an experiment');
  return lab;
}

const DEFAULT_MANIFEST = parseManifest(raw);

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

export default function InterventionLab({ manifest = DEFAULT_MANIFEST }: { manifest?: Manifest }) {
  const lab = useMemo(() => makeLab(manifest), [manifest]);
  return (
    <LabContext.Provider value={lab}>
      <LabBody />
    </LabContext.Provider>
  );
}

function LabBody() {
  const { m, demo: DEMO, checkpoint: CHECKPOINT, B, moved, localOptions: LOCAL_OPTIONS } = useLab();
  const { s, init, set } = useExperiment();
  const [issues, setIssues] = useState<string[]>([]);
  const [showEdges, setShowEdges] = useState(true);
  const [showShell, setShowShell] = useState(false);
  const [colourMode, setColourMode] = useState<'element' | 'change'>('element');
  const [resetKey, setResetKey] = useState(0);
  const [camera, setCamera] = useState<CameraPreset>('overview');
  const reduced = usePrefersReducedMotion();
  const wide = useWide();
  const sync = useMemo(() => new CameraSync(), []);

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

  const demo = s?.demo ?? 0;
  useEffect(() => {
    const step = DEMO[demo - 1];
    if (!step) return;
    set(step.state);
    setColourMode(step.colour);
    setCamera('overview');
    setResetKey((x) => x + 1);
  }, [demo, set]);

  const present = s?.present ?? false;
  useEffect(() => {
    document.body.classList.toggle('is-present', present);
  }, [present]);

  if (!s) return <div className="lab lab--loading" role="status">Loading the precomputed evaluations…</div>;

  const frame = s.frame;
  const split = s.view === 'split' && wide;
  // On narrow screens the side-by-side view shows one scene (the perturbed state); the toggle stays available.
  const view: View = s.view === 'split' && !wide ? 'perturbed' : s.view;
  const shown = view === 'baseline' ? B : frame;
  const d = m.displacement.values[frame] ?? 0;
  const sceneEdges = s.mode === 'walk' ? ['neighbours', 'radial', 'messages'].includes(s.stage) : showEdges;
  const sceneShell = s.mode === 'walk' ? s.stage === 'neighbours' : showShell;
  const sceneForces = s.mode !== 'walk' || s.stage === 'forces';
  const scene = (
    <SceneGuard>
      {split ? (
        <div className="lab__split">
          {([B, frame] as const).map((k, side) => (
            <div key={side} className="lab__pane">
              <Scene shown={k} view={side === 0 ? 'baseline' : 'perturbed'} selected={s.atom} showEdges={sceneEdges} showShell={sceneShell} showForces={sceneForces} colourMode={colourMode} layer={s.layer} resetKey={resetKey} camera={camera} reduced={reduced} onPick={(i) => set({ atom: i })} sync={sync} />
              <span className={`lab__pane-label mono${side === 1 ? ' is-pert' : ''}`}>
                {side === 0 ? 'Baseline · d = 0' : `Perturbed · d = ${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(3)} Å`}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <Scene shown={shown} view={view} selected={s.atom} showEdges={sceneEdges} showShell={sceneShell} showForces={sceneForces} colourMode={colourMode} layer={s.layer} resetKey={resetKey} camera={camera} reduced={reduced} onPick={(i) => set({ atom: i })} />
      )}
    </SceneGuard>
  );

  const views = (
    <div className="lab__views" role="radiogroup" aria-label="Comparison view">
      {(wide ? (['baseline', 'perturbed', 'difference', 'split'] as const) : (['baseline', 'perturbed', 'difference'] as const)).map((v) => (
        <button key={v} role="radio" aria-checked={view === v || (split && v === 'split')} className={`${view === v || (split && v === 'split') ? 'is-on' : ''} v-${v}`} onClick={() => set({ view: v })}>
          {v === 'baseline' ? 'Baseline' : v === 'perturbed' ? 'Perturbed' : v === 'difference' ? 'Difference' : 'Side by side'}
        </button>
      ))}
    </div>
  );

  if (s.present) {
    const step = DEMO[s.demo - 1];
    const fullHref = `?${encodeState({ ...s, present: false, demo: 0 })}`;
    return (
      <section className="lab lab--present" aria-labelledby="lab-title">
        <header className="lab__head">
          <h2 id="lab-title">Move one atom. Trace what changes inside the model.</h2>
          <div className="lab__status">
            <span className="label status-pre">Precomputed model evaluations; interactive inspection</span>
            <span className="label status-ck mono">{CHECKPOINT} · sha256 {String(m.provenance.sha256).slice(0, 8)}…</span>
          </div>
        </header>
        <div className="lab__stage lab__stage--present">
          {scene}
          <ForceScale />
          <ColourLegend mode={colourMode} layer={s.layer} />
          {views}
        </div>
        <div className="present__bar">
          <p className="present__caption" aria-live="polite">
            {step ? (
              <>
                <span className="present__n mono">
                  {step.n}/{DEMO.length}
                </span>{' '}
                <strong>{step.title}.</strong> {!wide && step.captionNarrow ? step.captionNarrow : step.caption}
              </>
            ) : (
              <>
                d = {d >= 0 ? '+' : '−'}
                {Math.abs(d).toFixed(3)} Å, evaluated frame {frame + 1} of {m.frames.length}. ΔE = {fmt(energyChange(m, frame), 3, 'eV')}.
              </>
            )}
          </p>
          <div className="present__nav" role="group" aria-label="Demonstration steps">
            <button onClick={() => set({ demo: Math.max(1, s.demo - 1) })} disabled={s.demo <= 1}>
              ← Previous
            </button>
            <button onClick={() => set({ demo: Math.min(DEMO.length, s.demo + 1) })} disabled={s.demo >= DEMO.length}>
              Next →
            </button>
            <a href={fullHref} onClick={(e) => { e.preventDefault(); set({ present: false, demo: 0 }); }}>
              Open the full instrument
            </a>
          </div>
        </div>
      </section>
    );
  }

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
          {scene}
          <ForceScale />
          <ColourLegend mode={colourMode} layer={s.layer} />
          {views}
        </div>
        <aside className="lab__panel">
          {s.mode === 'walk' && <StageDetail stage={s.stage} onStage={(st) => set({ stage: st })} />}
          {s.mode === 'walk' && <StageEvidence m={m} stage={s.stage} frame={frame} atom={s.atom} />}
          <Displacement frame={frame} onFrame={(k) => set({ frame: k })} />
          <Readouts frame={frame} atom={s.atom} />
          <fieldset className="lab__toggles">
            <legend>Scene</legend>
            {s.mode === 'walk' && <p className="lab__hint">Edges and cutoff sphere follow the walkthrough stage.</p>}
            <label>
              <input type="checkbox" checked={sceneEdges} disabled={s.mode === 'walk'} onChange={(e) => setShowEdges(e.target.checked)} /> Graph edges into the selected atom (r &lt; {m.rMax} Å)
            </label>
            <label>
              <input type="checkbox" checked={sceneShell} disabled={s.mode === 'walk'} onChange={(e) => setShowShell(e.target.checked)} /> Cutoff sphere
            </label>
            <label>
              Colour{' '}
              <select value={colourMode} onChange={(e) => setColourMode(e.target.value as 'element' | 'change')}>
                <option value="element">by element ({composition(m).map((c) => c.symbol).join(', ')})</option>
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
            <button onClick={() => set({ present: true, demo: 1 })}>Presentation mode</button>
            <button onClick={() => exportPng(figureText(m, s, colourMode, FORCE_SCALE, window.location.href))}>Save image (PNG)</button>
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
      <ArchitectureDiagram stage={s.mode === 'walk' ? s.stage : null} onStage={(st) => set({ mode: 'walk', stage: st })} />
      <section className="lab__inspect" aria-labelledby="inspect-title">
        <h3 id="inspect-title">Which parts of the model respond?</h3>
        <ResponsePlots m={m} atom={s.atom} frame={frame} onFrame={(k) => set({ frame: k })} />
        <FeatureInspector m={m} frame={frame} atom={s.atom} layer={s.layer} channel={s.channel} onChannel={(c) => set({ channel: c })} />
      </section>
      <TraceChange m={m} frame={frame} atom={s.atom} onFrame={(k) => set({ frame: k, view: 'perturbed' })} onAtom={(i) => set({ atom: i })} />
      <Plots frame={frame} onFrame={(k) => set({ frame: k })} d={d} />
      <RotationDemo m={m} channel={s.channel} />
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
  sync?: CameraSync;
}) {
  const { shown, view, selected, showEdges, showShell, colourMode, layer } = props;
  const { m, moved } = useLab();
  const atoms = useMemo(() => neighbourhood(m, shown, DISPLAY_RADIUS) ?? [], [m, shown]);
  const bondPairs = useMemo(() => bonds(atoms, experimentInfo(m.id)?.bondCutoff ?? BOND_CUTOFF), [m, atoms]);
  const scaleMax = useMemo(() => blockScaleMax(m, layer, 'h0') ?? 1, [m, layer]);

  const buffers = useMemo<AtomBuffers>(() => {
    const n = atoms.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const ao = new Float32Array(n);
    const elementColour = (i: number) => linear(ELEMENT_DISPLAY[m.structure.symbols[i] ?? '']?.color ?? COLORS.atom);
    const amber = linear(COLORS.amber);
    const mint = linear(COLORS.mint);
    atoms.forEach((a, i) => {
      positions.set(a.pos, i * 3);
      radii[i] = a.index === moved ? 0.62 : 0.5;
      const base = elementColour(a.index);
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
  }, [m, atoms, colourMode, layer, shown, scaleMax]);

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
      <Stage view={cameraView(props.camera, sel?.pos ?? [0, 0, 0], m.displacement.axis)} resetKey={props.resetKey} background={COLORS.stage} label="Silicon neighbourhood of the moved atom with force arrows" minDistance={6} maxDistance={60} sync={props.sync} preserveDrawingBuffer>
        <SceneContents onPick={(i) => i !== null && atoms[i] && props.onPick(atoms[i]!.index)} buffers={buffers} version={shown * 7 + layer + (colourMode === 'change' ? 1000 : 0)} sticks={sticks} arrows={arrows} lines={lines} sel={sel?.pos ?? null} showShell={showShell} />
      </Stage>
    </div>
  );
}

/** Desktop-width check for the side-by-side view (brief §12: one main scene at a time on mobile). */
function useWide(): boolean {
  const q = '(min-width: 900px)';
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return wide;
}

type CameraPreset = 'overview' | 'environment' | 'axis';
const PRESET_LABEL: Record<CameraPreset, string> = { overview: 'Overview', environment: 'Selected environment', axis: 'Side-on to the path' };

/** Camera presets keep z up; 'axis' looks perpendicular to the displacement axis so the motion is in the image plane. */
function cameraView(preset: CameraPreset, sel: Vec3, u: Vec3): { position: [number, number, number]; target: [number, number, number] } {
  if (preset === 'environment') return { position: [sel[0] + 7, sel[1] - 9, sel[2] + 5], target: sel };
  if (preset === 'axis') {
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
  const { m } = useLab();
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
  const { m } = useLab();
  if (mode === 'element')
    return (
      <div className="lab__legend">
        {composition(m).map(({ symbol }) => (
          <span key={symbol}>
            <span className="swatch" style={{ background: ELEMENT_DISPLAY[symbol]?.color ?? COLORS.atom }} /> {symbol}{' '}
          </span>
        ))}
        <span className="swatch" style={{ background: COLORS.mint }} /> moved atom
        <span className="lab__legend-note">
          edges: model graph (r &lt; {m.rMax} Å), sticks: {experimentInfo(m.id)?.bondLabel ?? `bonds < ${BOND_CUTOFF} Å`}
        </span>
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
  const { m, moved } = useLab();
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
  const { m } = useLab();
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
  const { m } = useLab();
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

// ---------------------------------------------------------------------------
// Still-image export: the visible scene canvas(es) plus title, settings, units, legends, status and source.

function wrap(ctx: CanvasRenderingContext2D, text: string, width: number): string[] {
  const out: string[] = [];
  let line = '';
  for (const w of text.split(' ')) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > width && line) {
      out.push(line);
      line = w;
    } else line = t;
  }
  if (line) out.push(line);
  return out;
}

export function exportPng(t: FigureText): void {
  const canvases = [...document.querySelectorAll<HTMLCanvasElement>('.lab__stage canvas')];
  if (!canvases.length) return;
  const W = 1800;
  const M = 48;
  const gap = 6;
  const sceneW = W - 2 * M;
  const cw = (sceneW - gap * (canvases.length - 1)) / canvases.length;
  const ch = Math.round(cw * (canvases[0]!.height / canvases[0]!.width));
  const out = document.createElement('canvas');
  const ctx = out.getContext('2d');
  if (!ctx) return;
  const mono = '"JetBrains Mono Variable", ui-monospace, monospace';
  const sans = '"Inter Variable", system-ui, sans-serif';
  ctx.font = `600 40px ${sans}`;
  const titleLines = wrap(ctx, t.title, sceneW);
  ctx.font = `22px ${mono}`;
  const textLines = [...t.settings, ...t.legend].flatMap((l) => wrap(ctx, l, sceneW));
  ctx.font = `20px ${sans}`;
  const footLines = [...wrap(ctx, t.attribution, sceneW), ...wrap(ctx, `Source: ${t.source}`, sceneW)];
  const H = M + titleLines.length * 50 + 46 + ch + 24 + textLines.length * 32 + 16 + footLines.length * 28 + M;
  out.width = W;
  out.height = H;
  ctx.fillStyle = COLORS.stage;
  ctx.fillRect(0, 0, W, H);
  let y = M + 38;
  ctx.fillStyle = '#dde5e2';
  ctx.font = `600 40px ${sans}`;
  for (const l of titleLines) {
    ctx.fillText(l, M, y);
    y += 50;
  }
  ctx.font = `600 20px ${mono}`;
  ctx.fillStyle = COLORS.mint;
  ctx.fillText(t.status.toUpperCase(), M, y);
  y += 26;
  canvases.forEach((c, i) => {
    ctx.drawImage(c, M + i * (cw + gap), y, cw, ch);
    const label = canvases.length > 1 ? t.panes[i] : undefined;
    if (label) {
      ctx.font = `600 22px ${mono}`;
      ctx.fillStyle = i === 0 ? COLORS.mint : COLORS.amber;
      ctx.fillText(label, M + i * (cw + gap) + 16, y + ch - 18);
    }
  });
  y += ch + 40;
  ctx.font = `22px ${mono}`;
  ctx.fillStyle = '#dde5e2';
  for (const l of textLines) {
    ctx.fillText(l, M, y);
    y += 32;
  }
  y += 8;
  ctx.font = `20px ${sans}`;
  ctx.fillStyle = '#9fb0ab';
  for (const l of footLines) {
    ctx.fillText(l, M, y);
    y += 28;
  }
  const a = document.createElement('a');
  a.download = 'inside-mace.png';
  a.href = out.toDataURL('image/png');
  a.click();
}
