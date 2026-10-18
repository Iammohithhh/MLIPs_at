import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Aimnet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { DATA, HP, dimerPositions, integerLimits, type Dimer, type Vec3 } from '~/lib/models/aimnet2';

// Fig. 2: charged and neutral dimers pulled apart, run through the released AIMNet2 checkpoint on CPU
// (pipeline/atlas_pipeline/aimnet2.py). Atoms coloured by the model's charges; fragment charges and energy
// terms against separation, with the model's own integer-charge dissociation limits.

const NAMES: Record<string, { a: string; b: string; short: string }> = {
  bz2: { a: 'benzene 1', b: 'benzene 2', short: '(C₆H₆)₂⁺' },
  bzw: { a: 'benzene', b: 'water', short: 'C₆H₆·H₂O⁺' },
  bzw0: { a: 'benzene', b: 'water', short: 'C₆H₆·H₂O' },
};

function minGap(dm: Dimer, d: number): number {
  const p = dimerPositions(dm, d);
  let m = Infinity;
  dm.fragment.forEach((fi, i) =>
    dm.fragment.forEach((fj, j) => {
      if (fi === 0 && fj === 1) m = Math.min(m, Math.hypot(p[i]![0] - p[j]![0], p[i]![1] - p[j]![1], p[i]![2] - p[j]![2]));
    }),
  );
  return m;
}

export default function AimnetDimer() {
  const theme = useSceneTheme();
  const id = useId();
  const [sys, setSys] = useState('bz2');
  const [k, setK] = useState(13);
  const dm = DATA.dimers.find((x) => x.id === sys)!;
  const row = dm.rows[k]!;
  const names = NAMES[sys]!;
  const lim = integerLimits(dm);
  const ref = Math.min(lim.plusA, lim.plusB);
  const positions = useMemo(() => dimerPositions(dm, row.d), [dm, row.d]);
  const qMax = useMemo(() => Math.max(...DATA.dimers.flatMap((x) => x.rows.flatMap((r) => r.q.map(Math.abs)))), []);
  const split = useMemo(() => dm.rows.find((r) => minGap(dm, r.d) > HP.rc)?.d ?? null, [dm]);
  const last = row.passes[row.passes.length - 1]!;
  const view = useMemo(() => ({ position: [34, -26, 12] as [number, number, number], target: [0, 0, 9] as [number, number, number] }), []);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={view} resetKey={0} background={theme.canvas} label={`${names.short} with the fragments ${row.d.toFixed(1)} Å apart along the vertical axis, atoms coloured by the released model's charges: red positive, blue negative.`} minDistance={6} maxDistance={90}>
            <DimerScene theme={theme} dm={dm} positions={positions} q={row.q} qMax={qMax} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="System">
              {DATA.dimers.map((x) => (
                <button key={x.id} type="button" aria-pressed={sys === x.id} onClick={() => setSys(x.id)}>
                  {NAMES[x.id]!.short}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: 'var(--atlas-fail)' }} /> q &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: 'var(--atlas-arch)' }} /> q &lt; 0
            </li>
            <li>d = {row.d.toFixed(1)} Å</li>
          </ul>
        </div>
        <div className="am-side">
          <p className="am-h">Charge on the {names.a}</p>
          <ChargePlot dm={dm} k={k} split={split} />
          <ul className="am-legend" aria-hidden="true">
            <li style={{ color: 'var(--atlas-fail)' }}>
              <span className="swatch" /> final (after pass 2’s NQE)
            </li>
            <li style={{ color: 'var(--atlas-caveat)' }}>
              <span className="swatch is-dashed" /> after pass 1’s NQE
            </li>
          </ul>
          <p className="am-h">Energy against the model’s own integer limit</p>
          <EnergyPlot dm={dm} k={k} split={split} ref0={ref} />
          <ul className="am-legend" aria-hidden="true">
            <li style={{ color: 'var(--atlas-ink)' }}>
              <span className="swatch" /> E(d)
            </li>
            <li style={{ color: 'var(--atlas-verified)' }}>
              <span className="swatch is-dashed" /> integer-charge limit{dm.Q ? 's' : ''}
            </li>
            {lim.half !== undefined ? (
              <li style={{ color: 'var(--atlas-fail)' }}>
                <span className="swatch is-dashed" /> two fragments at Q/2
              </li>
            ) : null}
          </ul>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${id}-d`}>
            <span>separation d</span>
            <input id={`${id}-d`} type="range" min={0} max={dm.rows.length - 1} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            <output>{row.d.toFixed(1)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>charge on {names.a} / {names.b}</dt>
            <dd>
              {row.fragQ[0].toFixed(4)} / {row.fragQ[1].toFixed(4)} e
            </dd>
          </div>
          <div>
            <dt>closest atoms of the two fragments</dt>
            <dd>
              {minGap(dm, row.d).toFixed(2)} Å <small>{minGap(dm, row.d) > HP.rc ? '(separate graphs)' : '(within r_c)'}</small>
            </dd>
          </div>
          <div>
            <dt>share of Σf on {names.a} (pass 2)</dt>
            <dd>{(last.f[0] / (last.f[0] + last.f[1])).toFixed(3)}</dd>
          </div>
          <div className={dm.Q !== 0 && row.E - ref < -0.3 ? 'is-alert' : undefined}>
            <dt>E − lowest integer limit</dt>
            <dd>{(row.E - ref).toFixed(3)} eV</dd>
          </div>
          <div>
            <dt>U_local · U_Coul · U_disp (shifted)</dt>
            <dd>
              {(row.U_local - dm.rows[dm.rows.length - 1]!.U_local).toFixed(3)} · {row.U_coul.toFixed(3)} · {row.U_disp.toFixed(3)} eV
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Precomputed with the released ωB97M-D3 checkpoint (ensemble member 0) on rigid monomers. The dashed green line{dm.Q ? 's are' : ' is'} the
          model’s own energy of the isolated fragments with integer charges{dm.Q ? ` (${names.a}⁺ + ${names.b}, and ${names.a} + ${names.b}⁺)` : ''}; the
          exact dissociation limit of a Q = {dm.Q} dimer is the lower of these. Once the closest atoms are more than {HP.rc} Å apart the two
          molecules sit on separate graphs: the local energy and the charges stop changing, and only the Coulomb tail of whatever charge each
          fragment kept still depends on d.
        </p>
      </div>
    </div>
  );
}

function chargeColour(theme: SceneTheme, q: number, qMax: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, q / qMax));
  const base = linear(theme.dark ? '#4a5260' : '#d5d8de');
  return mixRGB(base, linear(t < 0 ? theme.arch : theme.fail), Math.sqrt(Math.abs(t)));
}

function DimerScene({ theme, dm, positions, q, qMax }: { theme: SceneTheme; dm: Dimer; positions: Vec3[]; q: number[]; qMax: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 90);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      radii[i] = dm.numbers[i] === 1 ? 0.3 : 0.45;
      colors.set(chargeColour(theme, q[i]!, qMax), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.15), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, q, qMax, dm, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.ruleStrong);
    const out: Segment[] = [];
    for (let i = 0; i < positions.length; i++)
      for (let j = i + 1; j < positions.length; j++) {
        if (dm.fragment[i] !== dm.fragment[j]) continue;
        const a = positions[i]!;
        const b = positions[j]!;
        if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1.6) out.push({ a, b, color: c, radius: 0.07 });
      }
    return out;
  }, [positions, dm, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
    </>
  );
}

const W = 380;
const M = { l: 48, r: 10, t: 10, b: 28 };
const D0 = DATA.dimers[0]!.rows[0]!.d;
const D1 = DATA.dimers[0]!.rows[DATA.dimers[0]!.rows.length - 1]!.d;
const xOf = (d: number) => M.l + ((d - D0) / (D1 - D0)) * (W - M.l - M.r);

function Frame({ H, ticks, y, fmt, label, k, dm, split }: { H: number; ticks: number[]; y: (v: number) => number; fmt: (v: number) => string; label: string; k: number; dm: Dimer; split: number | null }) {
  return (
    <>
      {split !== null ? <rect className="am-band" x={xOf(split)} y={M.t} width={W - M.r - xOf(split)} height={H - M.t - M.b} /> : null}
      {ticks.map((v) => (
        <g key={v}>
          <line className={Math.abs(v) < 1e-12 ? 'axis' : 'grid'} x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} />
          <text className="tick" x={M.l - 4} y={y(v) + 3} textAnchor="end">
            {fmt(v)}
          </text>
        </g>
      ))}
      {[5, 10, 15, 20].map((d) => (
        <text key={d} className="tick" x={xOf(d)} y={H - M.b + 13} textAnchor="middle">
          {d}
        </text>
      ))}
      {split !== null ? (
        <text className="am-mark" x={xOf(split) + 3} y={M.t + 9}>
          separate graphs
        </text>
      ) : null}
      <line className="axis" x1={xOf(dm.rows[k]!.d)} x2={xOf(dm.rows[k]!.d)} y1={M.t} y2={H - M.b} />
      <text className="axis-label" x={W - M.r} y={H - 2} textAnchor="end">
        d (Å)
      </text>
      <text className="axis-label" x={11} y={M.t + 2} transform={`rotate(-90 11 ${M.t + 2})`} textAnchor="end">
        {label}
      </text>
    </>
  );
}

function ChargePlot({ dm, k, split }: { dm: Dimer; k: number; split: number | null }) {
  const H = 150;
  const vals = dm.rows.flatMap((r) => [r.fragQ[0], r.passes[0]!.q[0]]);
  let lo = Math.min(0, ...vals);
  let hi = Math.max(dm.Q, ...vals);
  const pad = 0.08 * (hi - lo || 0.05);
  lo -= pad;
  hi += pad;
  const y = (v: number) => M.t + ((hi - v) / (hi - lo)) * (H - M.t - M.b);
  const ticks = dm.Q ? [0, 0.5, 1] : [lo + pad, 0, hi - pad].map((v) => +v.toFixed(3));
  const path = (f: (r: Dimer['rows'][number]) => number) => dm.rows.map((r, i) => `${i ? 'L' : 'M'}${xOf(r.d).toFixed(1)},${y(f(r)).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot am-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Charge on the first fragment against separation, after the first and the final charge equilibration.">
        <Frame H={H} ticks={ticks} y={y} fmt={(v) => (dm.Q ? v.toFixed(1) : v.toFixed(3))} label="q (e)" k={k} dm={dm} split={split} />
        <path className="am-curve is-dashed" d={path((r) => r.passes[0]!.q[0])} style={{ stroke: 'var(--atlas-caveat)' }} />
        <path className="am-curve" d={path((r) => r.fragQ[0])} style={{ stroke: 'var(--atlas-fail)' }} />
      </svg>
    </figure>
  );
}

function EnergyPlot({ dm, k, split, ref0 }: { dm: Dimer; k: number; split: number | null; ref0: number }) {
  const H = 160;
  const lim = integerLimits(dm);
  const lines = [lim.plusA, lim.plusB, ...(lim.half !== undefined ? [lim.half] : [])].map((v) => v - ref0);
  const vals = [...dm.rows.map((r) => r.E - ref0), ...lines.filter((v) => v < 3)];
  const lo = Math.floor(Math.min(...vals) * 2) / 2 - 0.25;
  const hi = Math.ceil(Math.max(...vals) * 2) / 2 + 0.25;
  const y = (v: number) => M.t + ((hi - Math.min(hi, v)) / (hi - lo)) * (H - M.t - M.b);
  const step = hi - lo > 4 ? 1 : 0.5;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(+v.toFixed(2));
  const d = dm.rows.map((r, i) => `${i ? 'L' : 'M'}${xOf(r.d).toFixed(1)},${y(r.E - ref0).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot am-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Released-model energy against separation, relative to the lower integer-charge dissociation limit, with that limit and, for the symmetric dimer, the limit of two half-charged fragments.">
        <Frame H={H} ticks={ticks} y={y} fmt={(v) => v.toFixed(step < 1 ? 1 : 0)} label="E − limit (eV)" k={k} dm={dm} split={split} />
        {[lim.plusA, lim.plusB].map((v, i) =>
          v - ref0 < hi ? <line key={i} className="am-curve is-dashed" x1={M.l} x2={W - M.r} y1={y(v - ref0)} y2={y(v - ref0)} style={{ stroke: 'var(--atlas-verified)' }} /> : null,
        )}
        {lim.half !== undefined ? <line className="am-curve is-dashed" x1={M.l} x2={W - M.r} y1={y(lim.half - ref0)} y2={y(lim.half - ref0)} style={{ stroke: 'var(--atlas-fail)' }} /> : null}
        <path className="am-curve" d={d} style={{ stroke: 'var(--atlas-ink)' }} />
      </svg>
    </figure>
  );
}
