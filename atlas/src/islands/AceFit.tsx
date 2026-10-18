import { useEffect, useMemo, useRef, useState } from 'react';
import './PhysicsLab.css';
import './Ace.css';
import { Stage } from '~/three/Stage';
import { Arrows, CellBox, ImpostorAtoms, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { DIMER_R, LAB_SPEC, SCAN_EPS, type LabFit, type LabResult, type Reference } from '~/lib/models/ace-fit';
import type { AceWorkerRequest, AceWorkerResponse } from './ace/ace.worker';
import type { Vec3 } from '~/lib/models/ace';

// A linear ACE fitted in your browser (in a Web Worker) to energies and forces of 8-atom silicon
// cells whose reference is Stillinger–Weber or Tersoff, for every body order ν ≤ 4 and l_max ≤ 3.

const REFS: Array<{ id: Reference; label: string }> = [
  { id: 'sw', label: 'Stillinger–Weber (2 + 3-body)' },
  { id: 'tersoff', label: 'Tersoff (bond order)' },
];
const COUNTS = [20, 40, 60, 100];
const RIDGES = [1e-9, 1e-6, 1e-3];

export default function AceFit() {
  const theme = useSceneTheme();
  const [ref, setRef] = useState<Reference>('sw');
  const [count, setCount] = useState(60);
  const [ridge, setRidge] = useState(1e-6);
  const [nu, setNu] = useState(2);
  const [lmax, setLmax] = useState(2);
  const [cellIdx, setCellIdx] = useState(0);
  const [core, setCore] = useState(false);
  const [res, setRes] = useState<LabResult | null>(null);
  const [busy, setBusy] = useState<string | null>('starting');
  const [error, setError] = useState<string | null>(null);
  const worker = useRef<Worker | null>(null);
  const reqId = useRef(0);

  useEffect(() => {
    const w = new Worker(new URL('./ace/ace.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (e: MessageEvent<AceWorkerResponse>) => {
      const m = e.data;
      if (m.id !== reqId.current) return;
      if (m.type === 'progress') setBusy(m.message);
      else if (m.type === 'done') {
        setRes(m.result);
        setBusy(null);
      } else {
        setError(m.message);
        setBusy(null);
      }
    };
    return () => w.terminate();
  }, []);

  useEffect(() => {
    const w = worker.current;
    if (!w) return;
    reqId.current++;
    setBusy('starting');
    setError(null);
    const msg: AceWorkerRequest = { id: reqId.current, options: { ref, count, ridge } };
    w.postMessage(msg);
  }, [ref, count, ridge]);

  const fit = res?.fits.find((f) => f.nu === nu && f.lmax === lmax) ?? null;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          {res && fit ? (
            <Stage
              view={{ position: [16, -20, 12], target: [2.7, 2.7, 2.7] }}
              background={theme.canvas}
              label={`Test cell ${cellIdx + 1}: eight silicon atoms with the reference forces (dark arrows) and the ACE forces (blue arrows) for body order ${nu + 1} and l up to ${lmax}. Atoms are coloured by the force error.`}
              minDistance={8}
              maxDistance={80}
            >
              <CellScene theme={theme} cell={res.shown[cellIdx]!.cell} refF={res.shown[cellIdx]!.forces} aceF={fit.forces[cellIdx]!} />
            </Stage>
          ) : (
            <p className="lab-note ace-wait">{error ? `The fit failed: ${error}` : `Computing ${busy ?? ''}…`}</p>
          )}
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Test cell shown">
              <span className="seg-label">test cell</span>
              {[0, 1, 2, 3].map((k) => (
                <button key={k} type="button" aria-pressed={cellIdx === k} onClick={() => setCellIdx(k)}>
                  {k + 1}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.ink }} /> reference force
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> ACE force
            </li>
            <li>
              <span className="dot" style={{ background: theme.fail }} /> |ΔF| ≥ 0.5 eV/Å
            </li>
          </ul>
        </div>
        <div className="ace-side">
          <Grid theme={theme} res={res} nu={nu} lmax={lmax} onPick={(a, b) => (setNu(a), setLmax(b))} stale={busy !== null} />
          {fit ? <Readout fit={fit} /> : null}
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Reference potential">
            <span className="seg-label">reference</span>
            {REFS.map((r) => (
              <button key={r.id} type="button" aria-pressed={ref === r.id} onClick={() => setRef(r.id)}>
                {r.label}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Training cells">
            <span className="seg-label">training cells</span>
            {COUNTS.map((c) => (
              <button key={c} type="button" aria-pressed={count === c} onClick={() => setCount(c)}>
                {c}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Ridge regularisation">
            <span className="seg-label">ridge λ</span>
            {RIDGES.map((r) => (
              <button key={r} type="button" aria-pressed={ridge === r} onClick={() => setRidge(r)}>
                10<sup>{Math.round(Math.log10(r))}</sup>
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Core repulsion added to the curves below">
            <span className="seg-label">core repulsion (100 eV·Å, λ = 5 Å⁻²)</span>
            <button type="button" aria-pressed={!core} onClick={() => setCore(false)}>
              off
            </button>
            <button type="button" aria-pressed={core} onClick={() => setCore(true)}>
              added
            </button>
          </div>
        </div>
        {res && fit ? (
          <div className="ace-plots">
            <Curve
              theme={theme}
              title="Isolated Si₂ dimer (no dimer in training)"
              xs={DIMER_R}
              refY={res.dimerRef}
              aceY={fit.dimer.map((y, k) => y + (core ? res.dimerCore[k]! : 0))}
              trained={(r) => r >= res.rminTrain}
              xLabel="r (Å)"
              yLabel="E − 2E_atom (eV)"
              lo={-12}
              hi={20}
              fmtX={(x) => x.toFixed(1)}
            />
            <Curve
              theme={theme}
              title="Diamond Si under isotropic strain"
              xs={SCAN_EPS}
              refY={res.scanRef}
              aceY={fit.scan.map((y, k) => y + (core ? res.scanCore[k]! : 0))}
              trained={(e) => Math.abs(e) <= res.strainRange + 1e-9}
              xLabel="strain ε"
              yLabel="eV / atom"
              lo={-5}
              hi={1}
              fmtX={(x) => `${Math.round(100 * x)} %`}
            />
          </div>
        ) : null}
        <p className="lab-note">
          Stillinger–Weber needs ν = 2 and l ≤ 2 (derived above): below that the force error stays above about 50 meV/Å,
          at it the error drops several-fold, and higher orders add little. Tersoff’s bond order is a steep function of
          all neighbours at once, so its errors fall slowly with ν. Neither potential’s dimer is in the training data
          (shortest distance in training: {res ? `${res.rminTrain.toFixed(2)} Å` : '…'}); outside the shaded ranges the
          polynomial is unconstrained, and larger bases can dive far below the reference. The core-repulsion term of the
          pacemaker documentation example adds only about 1 eV at 1 Å, so it cannot fill a hole that deep.
        </p>
      </div>
    </div>
  );
}

const cls = (v: number) => (v < 0.03 ? 'good' : v < 0.1 ? 'mid' : 'bad');

function Grid({ theme, res, nu, lmax, onPick, stale }: { theme: SceneTheme; res: LabResult | null; nu: number; lmax: number; onPick: (nu: number, l: number) => void; stale: boolean }) {
  const ls = Array.from({ length: LAB_SPEC.lmax + 1 }, (_, l) => l);
  const nus = Array.from({ length: LAB_SPEC.numax }, (_, k) => k + 1);
  const get = (a: number, b: number): LabFit | undefined => res?.fits.find((f) => f.nu === a && f.lmax === b);
  return (
    <table className={`ace-grid${stale ? ' is-stale' : ''}`}>
      <caption>Test force RMSE (meV/Å) by body order and l<sub>max</sub>; train in small type. Click a cell to show its fit.</caption>
      <thead>
        <tr>
          <th scope="col">ν (body order)</th>
          {ls.map((l) => (
            <th scope="col" key={l}>
              l ≤ {l}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {nus.map((a) => (
          <tr key={a}>
            <th scope="row">
              {a} ({a + 1}-body)
            </th>
            {ls.map((l) => {
              const f = get(a, l);
              const sel = a === nu && l === lmax;
              return (
                <td key={l} className={f ? `is-${cls(f.test.force)}` : undefined} aria-current={sel ? 'true' : undefined}>
                  <button type="button" onClick={() => onPick(a, l)} aria-pressed={sel} style={sel ? { outlineColor: theme.arch } : undefined}>
                    {f ? Math.round(1000 * f.test.force) : '…'}
                    <small>{f ? Math.round(1000 * f.train.force) : ''}</small>
                  </button>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Readout({ fit }: { fit: LabFit }) {
  return (
    <dl className="lab-readout" aria-live="polite">
      <div>
        <dt>basis functions</dt>
        <dd>
          {fit.functions} <small>+ 1 constant</small>
        </dd>
      </div>
      <div>
        <dt>energy RMSE train / test</dt>
        <dd>
          {(1000 * fit.train.energy).toFixed(1)} / {(1000 * fit.test.energy).toFixed(1)} <small>meV/atom</small>
        </dd>
      </div>
      <div className={fit.test.force < 0.03 ? 'is-good' : fit.test.force > 0.1 ? 'is-alert' : undefined}>
        <dt>force RMSE train / test</dt>
        <dd>
          {(1000 * fit.train.force).toFixed(0)} / {(1000 * fit.test.force).toFixed(0)} <small>meV/Å</small>
        </dd>
      </div>
    </dl>
  );
}

function Curve({
  theme,
  title,
  xs,
  refY,
  aceY,
  trained,
  xLabel,
  yLabel,
  lo,
  hi,
  fmtX,
}: {
  theme: SceneTheme;
  title: string;
  xs: number[];
  refY: number[];
  aceY: number[];
  trained: (x: number) => boolean;
  xLabel: string;
  yLabel: string;
  lo: number;
  hi: number;
  fmtX: (x: number) => string;
}) {
  const W = 420;
  const H = 230;
  const M = { l: 46, r: 10, t: 40, b: 34 };
  const x0 = xs[0]!;
  const x1 = xs[xs.length - 1]!;
  const x = (v: number) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - M.t - M.b);
  const path = (ys: number[]) => ys.map((v, k) => `${k ? 'L' : 'M'}${x(xs[k]!).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const inside = xs.filter(trained);
  const ticks = Array.from({ length: 5 }, (_, k) => lo + (k * (hi - lo)) / 4);
  const xt = [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  const off = aceY.filter((v) => v < lo || v > hi).length;
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: reference and ACE curves against ${xLabel}; the range covered by training is shaded.`}>
        {inside.length ? <rect x={x(inside[0]!)} y={M.t} width={x(inside[inside.length - 1]!) - x(inside[0]!)} height={H - M.t - M.b} style={{ fill: theme.verified, opacity: 0.12 }} /> : null}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {Number(t.toFixed(1))}
            </text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 14} textAnchor="middle" className="tick">
            {fmtX(t)}
          </text>
        ))}
        <path d={path(refY)} className="curve" style={{ stroke: theme.ink, strokeDasharray: '5 3' }} />
        <path d={path(aceY)} className="curve" style={{ stroke: theme.arch }} />
        <text x={M.l} y={14} className="tick ace-plot-title">
          {title}
        </text>
        <text x={M.l} y={30} className="tick">
          <tspan style={{ fill: theme.arch }}>— ACE</tspan> <tspan style={{ fill: theme.ink }}>- - reference</tspan>
          {off ? <tspan style={{ fill: theme.fail }}> · {off} off scale</tspan> : null}
        </text>
        <text x={12} y={(H - M.b + M.t) / 2} transform={`rotate(-90 12 ${(H - M.b + M.t) / 2})`} textAnchor="middle" className="axis-label">
          {yLabel}
        </text>
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          {xLabel}
        </text>
      </svg>
    </figure>
  );
}

function CellScene({ theme, cell, refF, aceF }: { theme: SceneTheme; cell: { L: number; positions: Vec3[] }; refF: Vec3[]; aceF: Vec3[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 90);
  const fmax = Math.max(1e-6, ...refF.map((f) => Math.hypot(...f)), ...aceF.map((f) => Math.hypot(...f)));
  const s = 1.7 / fmax;
  const buffers = useMemo<AtomBuffers>(() => {
    const n = cell.positions.length;
    const pos = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    const hot = linear(theme.fail);
    cell.positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      const d = Math.hypot(aceF[i]![0] - refF[i]![0], aceF[i]![1] - refF[i]![1], aceF[i]![2] - refF[i]![2]);
      colors.set(mixRGB(stone, hot, Math.min(1, d / 0.5)), 3 * i);
    });
    return { positions: pos, radii: new Float32Array(n).fill(0.45), colors, metal: new Float32Array(n).fill(0.3), ao: new Float32Array(n).fill(1), count: n };
  }, [cell, refF, aceF, theme]);
  const arrows = useMemo<Segment[]>(() => {
    const ink = linear(theme.ink);
    const arch = linear(theme.arch);
    const out: Segment[] = [];
    cell.positions.forEach((p, i) => {
      for (const [f, color] of [[refF[i]!, ink], [aceF[i]!, arch]] as const) {
        if (Math.hypot(...f) * s < 0.15) continue;
        out.push({ a: p, b: [p[0] + s * f[0], p[1] + s * f[1], p[2] + s * f[2]], radius: 0.06, color });
      }
    });
    return out;
  }, [cell, refF, aceF, s, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Arrows arrows={arrows} uniforms={uniforms} head={0.28} />
      <CellBox origin={[0, 0, 0]} size={[cell.L, cell.L, cell.L]} color={theme.inkMuted} opacity={0.6} />
    </>
  );
}
