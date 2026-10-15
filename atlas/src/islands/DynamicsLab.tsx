import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { scaleLinear, scaleLog } from 'd3';
import './PhysicsLab.css';
import './DynamicsLab.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import {
  FRAME_DT,
  LJ_STATE,
  SAMPLE_DT,
  TIMESTEPS,
  TREATMENTS,
  Unwrapper,
  boxLength,
  diffusionCoefficient,
  displacements,
  energyDiagnostics,
  fccSites,
  logLogSlope,
  msd,
  pairHistogram,
  rdfFromHistogram,
  type CutoffTreatment,
  type RunConfig,
} from '~/lib/physics/md';
import type { Sample, WorkerRequest, WorkerResponse } from './md/lammps.worker';

// Physics Lab P12: two NVE trajectories of the same Lennard-Jones liquid, run by
// LAMMPS compiled to WebAssembly (lammps.js) in two Web Workers and advanced in
// lock-step. The only differences between the sides are how the pair potential
// is switched off at r_c and the time step.

const L = boxLength();
const N = 4 * LJ_STATE.cells ** 3;
const T_END = 40; // τ
const RDF_MAX = L / 2;
const RDF_BINS = 60;
const RDF_WINDOW = 30; // frames averaged into g(r)
const RDF_FROM = 5; // τ: skip the melting transient
const SWEEP_DTS = [0.001, 0.002, 0.005, 0.01];
const SWEEP_T = 4; // τ per sweep run
const SITES = fccSites();

interface Side {
  treatment: CutoffTreatment;
  dt: number;
}

interface Trajectory {
  samples: Sample[];
  ref: Float64Array;
  unwrapper: Unwrapper;
  current: Float64Array;
  msd: { t: number; v: number }[];
  hist: Float64Array[];
}

type Status = 'idle' | 'loading' | 'ready' | 'running' | 'sweeping' | 'done' | 'error';

function newTrajectory(first: Float64Array, s0: Sample): Trajectory {
  return { samples: [s0], ref: first.slice(), unwrapper: new Unwrapper(L, first), current: first.slice(), msd: [{ t: 0, v: 0 }], hist: [] };
}

// --- Worker plumbing --------------------------------------------------------

class LammpsWorker {
  private w: Worker;
  private queue: ((r: WorkerResponse) => void)[] = [];
  onSweep: ((r: WorkerResponse) => void) | null = null;
  constructor() {
    this.w = new Worker(new URL('./md/lammps.worker.ts', import.meta.url), { type: 'module' });
    this.w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const r = ev.data;
      if ((r.type === 'sweep-point' || r.type === 'sweep-done') && this.onSweep) return this.onSweep(r);
      if (r.type === 'error' && this.onSweep) return this.onSweep(r);
      this.queue.shift()?.(r);
    };
    this.w.onerror = (ev) => {
      const r: WorkerResponse = { type: 'error', message: ev.message || 'worker failed to start' };
      if (this.onSweep) this.onSweep(r);
      else this.queue.shift()?.(r);
    };
  }
  request(req: WorkerRequest): Promise<Extract<WorkerResponse, { type: 'frame' }>> {
    return new Promise((resolve, reject) => {
      this.queue.push((r) => (r.type === 'frame' ? resolve(r) : reject(new Error(r.type === 'error' ? r.message : 'unexpected reply'))));
      this.w.postMessage(req);
    });
  }
  sweep(treatment: CutoffTreatment, onPoint: (dt: number, rms: number, drift: number) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      this.onSweep = (r) => {
        if (r.type === 'sweep-point') onPoint(r.dt, r.rms, r.drift);
        else {
          this.onSweep = null;
          if (r.type === 'sweep-done') resolve();
          else reject(new Error(r.type === 'error' ? r.message : 'unexpected reply'));
        }
      };
      this.w.postMessage({ type: 'sweep', treatment, dts: SWEEP_DTS, duration: SWEEP_T } satisfies WorkerRequest);
    });
  }
  terminate() {
    this.w.terminate();
  }
}

// --- Component --------------------------------------------------------------

export default function DynamicsLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [sides, setSides] = useState<[Side, Side]>([
    { treatment: 'truncated', dt: 0.005 },
    { treatment: 'smooth', dt: 0.005 },
  ]);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState('');
  const [version, setVersion] = useState(0);
  const [sweep, setSweep] = useState<Record<CutoffTreatment, { dt: number; rms: number }[]>>({ truncated: [], shifted: [], smooth: [] });
  const workers = useRef<LammpsWorker[] | null>(null);
  const traj = useRef<(Trajectory | null)[]>([null, null]);
  const playing = useRef(false);
  const generation = useRef(0);

  useEffect(
    () => () => {
      playing.current = false;
      workers.current?.forEach((w) => w.terminate());
    },
    [],
  );

  const ensureWorkers = () => {
    if (!workers.current) workers.current = [new LammpsWorker(), new LammpsWorker()];
    return workers.current;
  };

  const setup = useCallback(async (cfg: [Side, Side]) => {
    const gen = ++generation.current;
    playing.current = false;
    setStatus('loading');
    setError('');
    try {
      const ws = ensureWorkers();
      const frames = await Promise.all(ws.map((w, k) => w.request({ type: 'setup', config: cfg[k]! satisfies RunConfig })));
      if (gen !== generation.current) return;
      traj.current = frames.map((f) => newTrajectory(f.positions, f.samples[0]!));
      setVersion((v) => v + 1);
      setStatus('ready');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }, []);

  const play = useCallback(async () => {
    const ws = workers.current;
    if (!ws || !traj.current[0]) return;
    const gen = generation.current;
    playing.current = true;
    setStatus('running');
    const per = Math.round(FRAME_DT / SAMPLE_DT);
    try {
      while (playing.current && gen === generation.current && traj.current[0]!.samples.at(-1)!.t < T_END - 1e-9) {
        const frames = await Promise.all(ws.map((w) => w.request({ type: 'advance', samples: per })));
        if (gen !== generation.current) return;
        frames.forEach((f, k) => {
          const tr = traj.current[k]!;
          tr.samples.push(...f.samples);
          tr.current = tr.unwrapper.push(f.positions);
          const t = f.samples.at(-1)!.t;
          tr.msd.push({ t, v: msd(tr.ref, tr.current) });
          if (t >= RDF_FROM - 1e-9) {
            tr.hist.push(pairHistogram(f.positions, L, RDF_MAX, RDF_BINS));
            if (tr.hist.length > RDF_WINDOW) tr.hist.shift();
          }
        });
        setVersion((v) => v + 1);
      }
      if (gen === generation.current) setStatus(traj.current[0]!.samples.at(-1)!.t >= T_END - 1e-9 ? 'done' : 'ready');
    } catch (e) {
      playing.current = false;
      setError(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }, []);

  const pause = () => {
    playing.current = false;
  };

  const changeSide = (k: 0 | 1, patch: Partial<Side>) => {
    const next = [...sides] as [Side, Side];
    next[k] = { ...next[k], ...patch };
    setSides(next);
    if (workers.current) void setup(next);
  };

  const runSweep = async () => {
    const ws = ensureWorkers();
    generation.current++;
    playing.current = false;
    setStatus('sweeping');
    setError('');
    setSweep({ truncated: [], shifted: [], smooth: [] });
    const add = (t: CutoffTreatment) => (dt: number, rms: number) =>
      setSweep((s) => ({ ...s, [t]: [...s[t], { dt, rms }].sort((a, b) => a.dt - b.dt) }));
    try {
      await Promise.all([
        ws[0]!.sweep('truncated', add('truncated')).then(() => ws[0]!.sweep('shifted', add('shifted'))),
        ws[1]!.sweep('smooth', add('smooth')),
      ]);
      await setup(sides);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  };

  // --- Derived diagnostics (all from ~/lib/physics/md) -----------------------

  const diag = useMemo(
    () =>
      traj.current.map((tr) => {
        if (!tr) return null;
        const t = tr.samples.map((s) => s.t);
        const e = tr.samples.map((s) => s.etotal);
        const energy = energyDiagnostics(t, e);
        const sum = new Float64Array(RDF_BINS);
        tr.hist.forEach((h) => h.forEach((v, i) => (sum[i]! += v)));
        const rdf = rdfFromHistogram(sum, tr.hist.length, N, L, RDF_MAX);
        const late = tr.msd.filter((m) => m.t >= RDF_FROM);
        const D =
          late.length > 10
            ? diffusionCoefficient(
                late.map((m) => m.t),
                late.map((m) => m.v),
                0.3,
              )
            : NaN;
        const recent = tr.samples.filter((s) => s.t >= RDF_FROM);
        const Tmean = recent.length ? recent.reduce((a, s) => a + s.temp, 0) / recent.length : NaN;
        return { energy, rdf, D, Tmean, t: t.at(-1)! };
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version],
  );

  const loaded = !!traj.current[0];
  const tNow = diag[0]?.t ?? 0;
  const busy = status === 'loading' || status === 'sweeping';
  const colors = [theme.arch, theme.lit];

  const statusText = {
    idle: 'LAMMPS not loaded. The scenes show the starting fcc lattice.',
    loading: 'Loading LAMMPS (WebAssembly) and building both systems…',
    ready: `Ready at t = ${tNow.toFixed(1)} τ.`,
    running: `Running: t = ${tNow.toFixed(1)} τ of ${T_END} τ.`,
    sweeping: 'Time-step sweep running in both workers…',
    done: `Finished ${T_END} τ.`,
    error: `LAMMPS stopped: ${error}`,
  }[status];

  return (
    <div className="lab md-lab">
      <div className="md-transport">
        {!loaded ? (
          <button type="button" className="md-primary" onClick={() => void setup(sides)} disabled={busy}>
            {status === 'loading' ? 'Loading LAMMPS…' : 'Start LAMMPS'}
            <small>downloads ≈ 11 MB of WebAssembly once</small>
          </button>
        ) : status === 'running' ? (
          <button type="button" className="md-primary" onClick={pause}>
            Pause
          </button>
        ) : (
          <button type="button" className="md-primary" onClick={() => void play()} disabled={busy || status === 'done' || status === 'error'}>
            {tNow > 0 ? 'Resume' : 'Run both'}
          </button>
        )}
        <button type="button" className="md-secondary" onClick={() => void setup(sides)} disabled={!loaded || busy}>
          Restart
        </button>
        <div className="md-clock" aria-hidden="true">
          <span style={{ width: `${(100 * tNow) / T_END}%` }} />
        </div>
        <p className="md-status" aria-live="polite">
          {statusText}
        </p>
      </div>

      <div className="md-sides">
        {([0, 1] as const).map((k) => (
          <SidePanel
            key={k}
            k={k}
            idPrefix={ids}
            side={sides[k]}
            color={colors[k]!}
            theme={theme}
            tr={traj.current[k] ?? null}
            version={version}
            disabled={busy}
            onChange={(p) => changeSide(k, p)}
            readout={diag[k] ?? null}
          />
        ))}
      </div>

      <div className="md-plots">
        <SeriesPlot
          title="total energy drift E(t) − E(0) [ε per atom]"
          xLabel="t [τ]"
          xMax={T_END}
          series={traj.current.map((tr, k) => ({
            color: colors[k]!,
            pts: tr ? tr.samples.map((s) => [s.t, s.etotal - tr.samples[0]!.etotal] as [number, number]) : [],
          }))}
          symmetric
          empty={!loaded}
        />
        <SeriesPlot
          title="temperature T(t) [ε/k_B]"
          xLabel="t [τ]"
          xMax={T_END}
          series={traj.current.map((tr, k) => ({ color: colors[k]!, pts: tr ? tr.samples.map((s) => [s.t, s.temp] as [number, number]) : [] }))}
          empty={!loaded}
        />
        <SeriesPlot
          title={`radial distribution g(r), last ${RDF_WINDOW} frames after ${RDF_FROM} τ`}
          xLabel="r [σ]"
          xMax={RDF_MAX}
          series={diag.map((d, k) => ({ color: colors[k]!, pts: d && traj.current[k]!.hist.length ? d.rdf.r.map((r, i) => [r, d.rdf.g[i]!] as [number, number]) : [] }))}
          refY={1}
          vLine={{ x: LJ_STATE.rc, label: 'r_c' }}
          empty={!loaded}
        />
        <SeriesPlot
          title="mean squared displacement [σ²]"
          xLabel="t [τ]"
          xMax={T_END}
          series={traj.current.map((tr, k) => ({ color: colors[k]!, pts: tr ? tr.msd.map((m) => [m.t, m.v] as [number, number]) : [] }))}
          empty={!loaded}
        />
      </div>

      <SweepPanel theme={theme} sweep={sweep} onRun={() => void runSweep()} busy={busy} />
    </div>
  );
}

// --- One side: controls, 3D, readout ----------------------------------------

function SidePanel({
  k,
  idPrefix,
  side,
  color,
  theme,
  tr,
  version,
  disabled,
  onChange,
  readout,
}: {
  k: 0 | 1;
  idPrefix: string;
  side: Side;
  color: string;
  theme: SceneTheme;
  tr: Trajectory | null;
  version: number;
  disabled: boolean;
  onChange: (p: Partial<Side>) => void;
  readout: { energy: ReturnType<typeof energyDiagnostics>; D: number; Tmean: number } | null;
}) {
  const name = k === 0 ? 'A' : 'B';
  const tdesc = TREATMENTS.find((t) => t.id === side.treatment)!;
  return (
    <section className="md-side" aria-labelledby={`${idPrefix}-side${k}`}>
      <header className="md-side-head">
        <h3 id={`${idPrefix}-side${k}`}>
          <span className="md-key" style={{ background: color }} aria-hidden="true" /> Run {name}
        </h3>
        <span className="md-sub">
          {tdesc.label}, Δt = {side.dt} τ
        </span>
      </header>
      <div className="lab-stage md-stage">
        <Stage
          view={{ position: [11, -15, 8], target: [0, 0, 0] }}
          background={theme.canvas}
          label={`Run ${name}: ${N} Lennard-Jones atoms in a periodic box, ${tdesc.label.toLowerCase()} cutoff, time step ${side.dt} τ. Atoms are coloured by how far they have moved from their starting site.`}
          minDistance={8}
          maxDistance={60}
        >
          <Box theme={theme} tr={tr} version={version} />
        </Stage>
      </div>
      <div className="md-side-controls">
        <div className="seg" role="group" aria-label={`Run ${name}: cutoff treatment`}>
          {TREATMENTS.map((t) => (
            <button key={t.id} type="button" aria-pressed={side.treatment === t.id} onClick={() => onChange({ treatment: t.id })} disabled={disabled} title={t.short}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label={`Run ${name}: time step`}>
          {TIMESTEPS.map((dt) => (
            <button key={dt} type="button" aria-pressed={side.dt === dt} onClick={() => onChange({ dt })} disabled={disabled}>
              {dt}
            </button>
          ))}
        </div>
      </div>
      <dl className="lab-readout md-readout" aria-live="polite">
        <div className={readout && Math.abs(readout.energy.drift) > 1e-4 ? 'is-alert' : undefined}>
          <dt>Energy drift</dt>
          <dd>{readout ? `${sci(readout.energy.drift)} ε/τ` : '–'}</dd>
        </div>
        <div>
          <dt>Fluctuation (rms)</dt>
          <dd>{readout ? `${sci(readout.energy.rms)} ε` : '–'}</dd>
        </div>
        <div>
          <dt>⟨T⟩ after {RDF_FROM} τ</dt>
          <dd>{readout && Number.isFinite(readout.Tmean) ? readout.Tmean.toFixed(3) : '–'}</dd>
        </div>
        <div>
          <dt>D = MSD/6t</dt>
          <dd>{readout && Number.isFinite(readout.D) ? `${readout.D.toFixed(4)} σ²/τ` : '–'}</dd>
        </div>
      </dl>
    </section>
  );
}

function sci(v: number): string {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  return `${m.toFixed(2)}×10${superscript(e)}`;
}

function superscript(n: number): string {
  const map: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  return String(n)
    .split('')
    .map((c) => map[c] ?? c)
    .join('');
}

function Box({ theme, tr, version }: { theme: SceneTheme; tr: Trajectory | null; version: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 34);
  const buffers = useMemo<AtomBuffers>(() => {
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);
    const cur = tr?.current ?? SITES;
    const disp = tr ? displacements(tr.ref, tr.current) : new Float64Array(N);
    const lo = linear(theme.dark ? '#5d6b82' : '#9aa6b8');
    const hi = linear(theme.caveat);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        const w = cur[3 * i + c]! - L * Math.floor(cur[3 * i + c]! / L);
        positions[3 * i + c] = w - L / 2;
      }
      // Colour saturates at one nearest-neighbour distance (~1.1 σ).
      colors.set(mixRGB(lo, hi, Math.min(1, disp[i]! / 1.1)), 3 * i);
    }
    return { positions, radii: new Float32Array(N).fill(0.42), colors, metal: new Float32Array(N), ao: new Float32Array(N).fill(1), count: N };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tr, version, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={version} uniforms={uniforms} />
      <CellBox origin={[-L / 2, -L / 2, -L / 2]} size={[L, L, L]} color={theme.ruleStrong} />
    </>
  );
}

// --- Plots ------------------------------------------------------------------

function SeriesPlot({
  title,
  xLabel,
  xMax,
  series,
  symmetric = false,
  refY,
  vLine,
  empty,
}: {
  title: string;
  xLabel: string;
  xMax: number;
  series: { color: string; pts: [number, number][] }[];
  symmetric?: boolean;
  refY?: number;
  vLine?: { x: number; label: string };
  empty: boolean;
}) {
  const W = 360;
  const H = 200;
  const M = { l: 56, r: 12, t: 26, b: 32 };
  const all = series.flatMap((s) => s.pts.map((p) => p[1]));
  let lo = all.length ? Math.min(...all) : 0;
  let hi = all.length ? Math.max(...all) : 1;
  if (refY !== undefined) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, refY * 1.2);
  }
  if (symmetric) {
    const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6);
    lo = -m;
    hi = m;
  }
  if (hi - lo < 1e-12) hi = lo + 1;
  const x = scaleLinear().domain([0, xMax]).range([M.l, W - M.r]);
  const y = scaleLinear().domain([lo, hi]).nice(4).range([H - M.b, M.t]);
  const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('');
  const fmt = y.tickFormat(4, Math.max(Math.abs(lo), Math.abs(hi)) < 0.01 ? '.0e' : undefined);
  return (
    <figure className="lab-plot md-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. ${empty ? 'No data yet.' : ''}`}>
        <text x={M.l} y={M.t - 10} className="axis-label">
          {title}
        </text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {y.ticks(4).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">
              {fmt(v)}
            </text>
          </g>
        ))}
        {x.ticks(5).map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 15} textAnchor="middle" className="tick">
            {v}
          </text>
        ))}
        <text x={W - M.r} y={H - 4} textAnchor="end" className="tick">
          {xLabel}
        </text>
        {refY !== undefined && <line x1={M.l} x2={W - M.r} y1={y(refY)} y2={y(refY)} className="md-ref" />}
        {vLine && (
          <g>
            <line x1={x(vLine.x)} x2={x(vLine.x)} y1={M.t} y2={H - M.b} className="md-ref" />
            <text x={x(vLine.x) + 4} y={M.t + 12} className="tick">
              {vLine.label}
            </text>
          </g>
        )}
        {/* Run B dashed so it stays visible where the two runs agree. */}
        {series.map(
          (s, k) =>
            s.pts.length > 1 && <path key={k} d={path(s.pts)} className="curve is-active" style={{ stroke: s.color, strokeDasharray: k === 1 ? '5 3' : undefined }} />,
        )}
        {empty && (
          <text x={(M.l + W - M.r) / 2} y={(M.t + H - M.b) / 2} textAnchor="middle" className="tick">
            start LAMMPS to fill this plot
          </text>
        )}
      </svg>
    </figure>
  );
}

function SweepPanel({
  theme,
  sweep,
  onRun,
  busy,
}: {
  theme: SceneTheme;
  sweep: Record<CutoffTreatment, { dt: number; rms: number }[]>;
  onRun: () => void;
  busy: boolean;
}) {
  const W = 520;
  const H = 260;
  const M = { l: 62, r: 16, t: 26, b: 38 };
  const col: Record<CutoffTreatment, string> = { truncated: theme.fail, shifted: theme.caveat, smooth: theme.phys };
  const all = Object.values(sweep).flat();
  const yLo = all.length ? Math.min(...all.map((p) => p.rms)) / 3 : 1e-7;
  const yHi = all.length ? Math.max(...all.map((p) => p.rms)) * 3 : 1e-2;
  const x = scaleLog().domain([SWEEP_DTS[0]! / 1.3, SWEEP_DTS.at(-1)! * 1.3]).range([M.l, W - M.r]);
  const y = scaleLog().domain([yLo, yHi]).range([H - M.b, M.t]);
  const slopes = (Object.keys(sweep) as CutoffTreatment[]).map((t) => ({
    t,
    k: sweep[t].length >= 2 ? logLogSlope(sweep[t].map((p) => p.dt), sweep[t].map((p) => p.rms)) : NaN,
  }));
  // Slope-2 guide through the smooth series' largest time step.
  const anchor = sweep.smooth.at(-1);
  const guide = anchor
    ? [SWEEP_DTS[0]!, SWEEP_DTS.at(-1)!].map((dt) => [dt, anchor.rms * (dt / anchor.dt) ** 2] as [number, number])
    : null;
  return (
    <div className="md-sweep">
      <div className="md-sweep-text">
        <h3>Time-step sweep</h3>
        <p>
          Fresh {SWEEP_T} τ runs from the same start at Δt = {SWEEP_DTS.join(', ')} τ for each cutoff treatment. The y-axis
          is the rms fluctuation of the total energy per atom about its drift line. For a smooth Hamiltonian, velocity
          Verlet keeps this error at order Δt², a slope of 2 on these axes.
        </p>
        <button type="button" className="md-primary" onClick={onRun} disabled={busy}>
          Run the sweep <small>{SWEEP_DTS.length * 3} short LAMMPS runs</small>
        </button>
        <ul className="md-slopes">
          {slopes.map(({ t, k }) => (
            <li key={t}>
              <span className="md-key" style={{ background: col[t] }} aria-hidden="true" />
              {TREATMENTS.find((x) => x.id === t)!.label}: fitted slope {Number.isFinite(k) ? k.toFixed(2) : '–'}
            </li>
          ))}
        </ul>
      </div>
      <figure className="lab-plot md-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy fluctuation against time step, log-log. ${slopes.map((s) => `${s.t}: slope ${Number.isFinite(s.k) ? s.k.toFixed(2) : 'not run'}`).join('; ')}.`}>
          <text x={M.l} y={M.t - 10} className="axis-label">
            rms energy fluctuation [ε per atom] vs Δt [τ]
          </text>
          <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
          {SWEEP_DTS.map((dt) => (
            <text key={dt} x={x(dt)} y={H - M.b + 15} textAnchor="middle" className="tick">
              {dt}
            </text>
          ))}
          {y.ticks(4).filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9).map((v) => (
            <g key={v}>
              <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
              <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">
                {`10${superscript(Math.round(Math.log10(v)))}`}
              </text>
            </g>
          ))}
          {guide && (
            <g>
              <line x1={x(guide[0]![0])} y1={y(guide[0]![1])} x2={x(guide[1]![0])} y2={y(guide[1]![1])} className="md-ref" />
              <text x={x(guide[0]![0]) + 4} y={y(guide[0]![1]) - 6} className="tick">
                ∝ Δt²
              </text>
            </g>
          )}
          {(Object.keys(sweep) as CutoffTreatment[]).map((t) => (
            <g key={t}>
              {sweep[t].length > 1 && (
                <path
                  d={sweep[t].map((p, i) => `${i ? 'L' : 'M'}${x(p.dt).toFixed(1)},${y(p.rms).toFixed(1)}`).join('')}
                  className="curve is-active"
                  style={{ stroke: col[t] }}
                />
              )}
              {sweep[t].map((p) => (
                <circle key={p.dt} cx={x(p.dt)} cy={y(p.rms)} r={4} style={{ fill: col[t] }} />
              ))}
            </g>
          ))}
          {all.length === 0 && (
            <text x={(M.l + W - M.r) / 2} y={(M.t + H - M.b) / 2} textAnchor="middle" className="tick">
              run the sweep to fill this plot
            </text>
          )}
        </svg>
      </figure>
    </div>
  );
}
