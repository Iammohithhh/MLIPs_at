import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Gap.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { DEFAULT_GAP, cellDescriptors, diamondCell, fitGap, predict, rattled, selectSparse, strained, swEnergy, trainingCells, type Cell } from '~/lib/models/gap';
import { DEFAULT_SOAP, type SoapOptions } from '~/lib/models/soap';

// A sparse GAP trained in the browser on Stillinger–Weber "reference" energies of small diamond-Si
// cells (src/lib/models/gap.ts), then asked about cells it has and has not seen. The band is the Si
// GAP's simplified per-atom variance: it measures distance from the representative environments.

const SOAP: SoapOptions = { ...DEFAULT_SOAP, lmax: 4, nr: 14 };
const NPTS = 41;
type Scan = 'strain' | 'rattle';
const SCANS: Record<Scan, { label: string; xs: number[]; cell: (x: number) => Cell; axis: string; fmt: (x: number) => string }> = {
  strain: {
    label: 'isotropic strain',
    xs: Array.from({ length: NPTS }, (_, k) => -0.1 + (0.2 * k) / (NPTS - 1)),
    cell: (x) => strained(diamondCell(), x),
    axis: 'strain ε (perfect diamond)',
    fmt: (x) => `${(100 * x).toFixed(1)} %`,
  },
  rattle: {
    label: 'random displacements',
    xs: Array.from({ length: NPTS }, (_, k) => (0.4 * k) / (NPTS - 1)),
    cell: (x) => rattled(diamondCell(), x, 2024),
    axis: 'rattle amplitude (Å per component, at a₀)',
    fmt: (x) => `${x.toFixed(2)} Å`,
  },
};

export default function GapLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [nTrain, setNTrain] = useState(24);
  const [M, setM] = useState(30);
  const [maxStrain, setMaxStrain] = useState(0.03);
  const [maxRattle, setMaxRattle] = useState(0.1);
  const [zeta, setZeta] = useState(4);
  const [scan, setScan] = useState<Scan>('strain');
  const [k, setK] = useState(30);

  const train = useMemo(() => {
    const cells = trainingCells({ count: nTrain, maxStrain, maxRattle, seed: 7 });
    return cells.map((c) => ({ envs: cellDescriptors(c, SOAP), energy: swEnergy(c) }));
  }, [nTrain, maxStrain, maxRattle]);
  const settings = useMemo(() => ({ ...DEFAULT_GAP, zeta }), [zeta]);
  const model = useMemo(() => {
    const all = train.flatMap((c) => c.envs);
    const idx = selectSparse(all, M, zeta);
    return fitGap(train, idx.map((i) => all[i]!), settings);
  }, [train, M, zeta, settings]);
  const rmse = useMemo(() => Math.sqrt(train.reduce((s, c) => s + ((predict(model, c.envs).energy - c.energy) / c.envs.length) ** 2, 0) / train.length), [train, model]);

  const sc = SCANS[scan];
  const scanData = useMemo(() => sc.xs.map((x) => {
    const cell = sc.cell(x);
    return { x, cell, envs: cellDescriptors(cell, SOAP), ref: swEnergy(cell) / 8 };
  }), [sc]);
  const preds = useMemo(() => scanData.map((p) => predict(model, p.envs)), [scanData, model]);
  const rows = scanData.map((p, i) => {
    const pr = preds[i]!;
    const sbar = pr.sigma.reduce((a, b) => a + b, 0) / pr.sigma.length;
    return { x: p.x, ref: p.ref, pred: pr.energy / 8, sbar, sig: pr.sigma };
  });
  const sel = rows[Math.min(k, rows.length - 1)]!;
  const selCell = scanData[Math.min(k, rows.length - 1)]!.cell;
  const sigMax = Math.max(1e-6, ...rows.flatMap((r) => r.sig));
  const trained = (x: number) => (scan === 'strain' ? Math.abs(x) <= maxStrain : x <= maxRattle);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [17, -21, 13], target: [2.7, 2.7, 2.7] }}
            resetKey={scan === 'strain' ? 0 : 1}
            background={theme.canvas}
            label={`An 8-atom diamond silicon cell at ${sc.label} ${sc.fmt(sel.x)}. Atoms are coloured by the GAP's per-atom uncertainty, from grey (small) to red (the largest in this scan).`}
            minDistance={8}
            maxDistance={80}
          >
            <Scene theme={theme} cell={selCell} sig={sel.sig} sigMax={sigMax} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Test scan">
              {(Object.keys(SCANS) as Scan[]).map((s) => (
                <button key={s} type="button" aria-pressed={scan === s} onClick={() => setScan(s)}>
                  {SCANS[s].label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.dark ? '#3c4552' : '#c9ced6' }} /> small σ
            </li>
            <li>
              <span className="dot" style={{ background: theme.fail }} /> σ = {(1000 * sigMax).toFixed(0)} meV
            </li>
          </ul>
        </div>
        <div className="gap-side">
          <h4>Energy per atom along the scan</h4>
          <EnergyPlot theme={theme} rows={rows} sel={k} trained={trained} axis={sc.axis} onPick={setK} />
          <ErrorPlot theme={theme} rows={rows} sel={k} trained={trained} />
          <label htmlFor={`${ids}-k`} className="gap-meter">
            <span>test point</span>
            <input id={`${ids}-k`} type="range" min={0} max={NPTS - 1} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            <output>{sc.fmt(sel.x)}</output>
          </label>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-n`}>
            <span>training cells</span>
            <input id={`${ids}-n`} type="range" min={2} max={60} step={1} value={nTrain} onChange={(e) => setNTrain(Number(e.target.value))} />
            <output>{nTrain}</output>
          </label>
          <label htmlFor={`${ids}-m`}>
            <span>representative environments M</span>
            <input id={`${ids}-m`} type="range" min={2} max={80} step={1} value={M} onChange={(e) => setM(Number(e.target.value))} />
            <output>{model.sparse.length}</output>
          </label>
          <label htmlFor={`${ids}-e`}>
            <span>training strains up to</span>
            <input id={`${ids}-e`} type="range" min={0} max={0.1} step={0.005} value={maxStrain} onChange={(e) => setMaxStrain(Number(e.target.value))} />
            <output>±{(100 * maxStrain).toFixed(1)} %</output>
          </label>
          <label htmlFor={`${ids}-r`}>
            <span>training rattles up to</span>
            <input id={`${ids}-r`} type="range" min={0} max={0.3} step={0.01} value={maxRattle} onChange={(e) => setMaxRattle(Number(e.target.value))} />
            <output>{maxRattle.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-z`}>
            <span>kernel exponent ζ</span>
            <input id={`${ids}-z`} type="range" min={1} max={6} step={1} value={zeta} onChange={(e) => setZeta(Number(e.target.value))} />
            <output>{zeta}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={rmse < 0.002 ? 'is-good' : undefined}>
            <dt>training RMSE</dt>
            <dd>{(1000 * rmse).toFixed(2)} meV/atom</dd>
          </div>
          <div>
            <dt>reference (SW) at test point</dt>
            <dd>{sel.ref.toFixed(4)} eV/atom</dd>
          </div>
          <div className={Math.abs(sel.pred - sel.ref) > 0.01 ? 'is-alert' : 'is-good'}>
            <dt>GAP error</dt>
            <dd>{(1000 * (sel.pred - sel.ref)).toFixed(1)} meV/atom</dd>
          </div>
          <div className={trained(sel.x) ? undefined : 'is-alert'}>
            <dt>mean per-atom σ</dt>
            <dd>
              {(1000 * sel.sbar).toFixed(1)} meV <small>{trained(sel.x) ? 'inside training range' : 'outside training range'}</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Inside the shaded training range the fit is within a few meV per atom. Outside it the error grows quickly, and so
          does σ, because those environments are far from every representative one; σ is a distance in kernel space, not an
          error estimate, so its size need not match the error. Train on wider strains and the band shrinks over them. Try two
          training cells: the fit is still exact on them, but with so few energies the local energies are barely constrained.
        </p>
      </div>
    </div>
  );
}

interface Row {
  x: number;
  ref: number;
  pred: number;
  sbar: number;
  sig: number[];
}

function frame(rows: Row[], W: number, M: { l: number; r: number; t: number; b: number }) {
  const x0 = rows[0]!.x;
  const x1 = rows[rows.length - 1]!.x;
  return (x: number) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
}

function TrainBand({ rows, x, trained, top, bottom, theme }: { rows: Row[]; x: (v: number) => number; trained: (v: number) => boolean; top: number; bottom: number; theme: SceneTheme }) {
  const inside = rows.filter((r) => trained(r.x));
  if (!inside.length) return null;
  const a = x(inside[0]!.x);
  const b = x(inside[inside.length - 1]!.x);
  return <rect x={a - 2} y={top} width={Math.max(4, b - a + 4)} height={bottom - top} style={{ fill: theme.verified, opacity: 0.12 }} />;
}

function EnergyPlot({ theme, rows, sel, trained, axis, onPick }: { theme: SceneTheme; rows: Row[]; sel: number; trained: (x: number) => boolean; axis: string; onPick: (k: number) => void }) {
  const W = 440;
  const H = 210;
  const M = { l: 52, r: 10, t: 12, b: 34 };
  const x = frame(rows, W, M);
  const vals = rows.flatMap((r) => [r.ref, r.pred]);
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = 0.08 * (hi - lo || 1);
  lo -= pad;
  hi += pad;
  const clamp = (v: number) => Math.max(lo, Math.min(hi, v));
  const y = (v: number) => M.t + (1 - (clamp(v) - lo) / (hi - lo)) * (H - M.t - M.b);
  const path = (f: (r: Row) => number) => rows.map((r, k) => `${k ? 'L' : 'M'}${x(r.x).toFixed(1)},${y(f(r)).toFixed(1)}`).join('');
  const band = `${rows.map((r, k) => `${k ? 'L' : 'M'}${x(r.x).toFixed(1)},${y(r.pred + 2 * r.sbar).toFixed(1)}`).join('')}${[...rows].reverse().map((r) => `L${x(r.x).toFixed(1)},${y(r.pred - 2 * r.sbar).toFixed(1)}`).join('')}Z`;
  const ticks = Array.from({ length: 5 }, (_, k) => lo + ((k + 0.5) * (hi - lo)) / 5);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy per atom against ${axis}: Stillinger–Weber reference, GAP prediction and a band of two per-atom σ; the training range is shaded.`}>
        <TrainBand rows={rows} x={x} trained={trained} top={M.t} bottom={H - M.b} theme={theme} />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <path d={band} style={{ fill: theme.arch, opacity: 0.16 }} />
        <path d={path((r) => r.ref)} className="curve" style={{ stroke: theme.ink, strokeDasharray: '5 3' }} />
        <path d={path((r) => r.pred)} className="curve" style={{ stroke: theme.arch }} />
        {rows.map((r, k) => (
          <rect key={k} x={x(r.x) - 5} y={M.t} width={10} height={H - M.t - M.b} style={{ fill: 'transparent', cursor: 'pointer' }} onClick={() => onPick(k)} aria-hidden="true" />
        ))}
        <line x1={x(rows[sel]!.x)} x2={x(rows[sel]!.x)} y1={M.t} y2={H - M.b} style={{ stroke: theme.fail, strokeDasharray: '2 3' }} />
        <text x={M.l + 4} y={M.t + 12} className="tick" style={{ fill: theme.arch }}>
          GAP ± 2σ̄
        </text>
        <text x={M.l + 80} y={M.t + 12} className="tick" style={{ fill: theme.ink }}>
          - - SW reference
        </text>
        <text x={M.l + 180} y={M.t + 12} className="tick" style={{ fill: theme.verified }}>
          ▮ trained
        </text>
        <text x={14} y={H / 2} transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle" className="axis-label">
          eV / atom
        </text>
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">
          {axis}
        </text>
      </svg>
    </figure>
  );
}

function ErrorPlot({ theme, rows, sel, trained }: { theme: SceneTheme; rows: Row[]; sel: number; trained: (x: number) => boolean }) {
  const W = 440;
  const H = 156;
  const M = { l: 52, r: 10, t: 10, b: 34 };
  const x = frame(rows, W, M);
  const y0 = 1e-5;
  const y1 = 1;
  const y = (v: number) => M.t + (1 - (Math.log10(Math.min(Math.max(v, y0), y1)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - M.t - M.b);
  const path = (f: (r: Row) => number) => rows.map((r, k) => `${k ? 'L' : 'M'}${x(r.x).toFixed(1)},${y(f(r)).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Absolute GAP error and mean per-atom σ along the scan, logarithmic scale.">
        <TrainBand rows={rows} x={x} trained={trained} top={M.t} bottom={H - M.b} theme={theme} />
        {[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1].map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {String(Number((1000 * t).toPrecision(1)))}
            </text>
          </g>
        ))}
        <path d={path((r) => Math.abs(r.pred - r.ref))} className="curve" style={{ stroke: theme.fail }} />
        <path d={path((r) => r.sbar)} className="curve" style={{ stroke: theme.arch, strokeDasharray: '4 3' }} />
        <line x1={x(rows[sel]!.x)} x2={x(rows[sel]!.x)} y1={M.t} y2={H - M.b} style={{ stroke: theme.fail, strokeDasharray: '2 3' }} />
        <text x={14} y={H / 2} transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle" className="axis-label">
          meV / atom
        </text>
        <text x={M.l} y={H - 8} className="tick" style={{ fill: theme.fail }}>
          — |GAP − SW| per atom
        </text>
        <text x={M.l + 150} y={H - 8} className="tick" style={{ fill: theme.arch }}>
          - - mean σ per atom
        </text>
      </svg>
    </figure>
  );
}

function Scene({ theme, cell, sig, sigMax }: { theme: SceneTheme; cell: Cell; sig: number[]; sigMax: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 90);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = cell.positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n).fill(0.6);
    const colors = new Float32Array(n * 3);
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    const hot = linear(theme.fail);
    cell.positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      colors.set(mixRGB(stone, hot, Math.min(1, sig[i]! / sigMax)), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.35), ao: new Float32Array(n).fill(1), count: n };
  }, [cell, sig, sigMax, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <CellBox origin={[0, 0, 0]} size={[cell.L, cell.L, cell.L]} color={theme.inkMuted} opacity={0.6} />
    </>
  );
}
