import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Aimnet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Sticks, linear, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { DATA, TD_THETA, distancesFrom, firstPassInvariants, invariantDistance, maxVector, methaneD2d } from '~/lib/models/aimnet2';

// Fig. 1: CH4 twisted from tetrahedral (θ = 54.7°) to square planar (θ = 90°) along the D2d path at fixed C–H length.
// Live: carbon's first-pass invariants (src/lib/models/aimnet2.ts) never change, the hydrogens' do.
// Precomputed: the released AIMNet2 checkpoint's own features per pass and its energy along the path.

const TD = methaneD2d(TD_THETA);
const TD_C = firstPassInvariants(TD.numbers, TD.positions, 0);
const TD_H = firstPassInvariants(TD.numbers, TD.positions, 1);
const ROWS = DATA.methane;

export default function AimnetMethane() {
  const theme = useSceneTheme();
  const id = useId();
  const [theta, setTheta] = useState(76);
  const mol = useMemo(() => methaneD2d(theta), [theta]);
  const live = useMemo(() => {
    const c = firstPassInvariants(mol.numbers, mol.positions, 0);
    const h = firstPassInvariants(mol.numbers, mol.positions, 1);
    return { dC: invariantDistance(c, TD_C), vC: maxVector(c), dH: invariantDistance(h, TD_H), hh: distancesFrom(mol.positions, 1).slice(1) };
  }, [mol]);
  const row = ROWS.find((r) => r.theta === theta)!;
  const view = useMemo(() => ({ position: [7.5, -9, 5] as [number, number, number], target: [0, 0, 0] as [number, number, number] }), []);
  const shape = Math.abs(theta - TD_THETA) < 0.5 ? 'tetrahedral' : theta === 90 ? 'square planar' : 'D2d';

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage lab-stage--short">
          <Stage view={view} resetKey={0} background={theme.canvas} label={`Methane at θ = ${theta}°, ${shape}. Carbon at the centre, four hydrogens 1.087 Å away; thin lines join the hydrogens.`} minDistance={4} maxDistance={40}>
            <MethaneScene theme={theme} positions={mol.positions} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: 'var(--atlas-ink)' }} /> C
            </li>
            <li>
              <span className="dot" style={{ background: 'var(--atlas-surface)' }} /> H
            </li>
            <li>θ = {theta}° · {shape}</li>
          </ul>
        </div>
        <div className="am-side">
          <p className="am-h">Released checkpoint: how much each pass sees</p>
          <FeaturePlot theta={theta} />
          <ul className="am-legend" aria-hidden="true">
            <li style={{ color: 'var(--atlas-phys)' }}>
              <span className="swatch" /> C, pass 1
            </li>
            <li style={{ color: 'var(--atlas-arch)' }}>
              <span className="swatch" /> C, pass 2
            </li>
            <li style={{ color: 'var(--atlas-lit)' }}>
              <span className="swatch" /> C, pass 3
            </li>
            <li style={{ color: 'var(--atlas-caveat)' }}>
              <span className="swatch is-dashed" /> H, pass 1
            </li>
          </ul>
          <EnergyPlot theta={theta} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${id}-t`}>
            <span>twist θ</span>
            <input id={`${id}-t`} type="range" min={30} max={90} step={1} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>{theta}°</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className="is-good">
            <dt>carbon, first-pass invariants vs Td (live)</dt>
            <dd>{live.dC < 1e-12 ? 'identical' : live.dC.toExponential(1)}</dd>
          </div>
          <div>
            <dt>carbon, largest l = 1 term |Σ g r̂|² (live)</dt>
            <dd>{live.vC < 1e-12 ? '0' : live.vC.toExponential(1)}</dd>
          </div>
          <div className={live.dH > 1e-6 ? 'is-alert' : undefined}>
            <dt>hydrogen, first-pass invariants vs Td (live)</dt>
            <dd>{live.dH < 1e-12 ? 'identical' : `changed (max ${live.dH.toFixed(3)})`}</dd>
          </div>
          <div>
            <dt>H–H distances from H₁</dt>
            <dd>{live.hh.map((d) => d.toFixed(2)).join(', ')} Å</dd>
          </div>
          <div>
            <dt>model energy − E(Td) (precomputed)</dt>
            <dd>{row.E.toFixed(2)} eV</dd>
          </div>
          <div>
            <dt>model charge on C (precomputed)</dt>
            <dd>{row.qC.toFixed(3)} e</dd>
          </div>
        </dl>
        <p className="lab-note">
          Every C–H distance stays 1.087 Å and the four bond directions always add up to zero, so the l = 0 sums (per shell) and the l = 1 sums
          (per shell, before squaring) that carbon’s first pass is made of are the same at every θ: the readout checks this live, and the
          released model’s own first-pass features for carbon change by less than 10⁻⁶ relative (flat line). The hydrogens do see the twist,
          because their H–H distances change; from the second pass on, carbon reads its neighbours’ updated features and the change reaches it.
          The energy is the released model’s, for a geometry far outside its training data; it is not compared with DFT here.
        </p>
      </div>
    </div>
  );
}

function MethaneScene({ theme, positions }: { theme: SceneTheme; positions: Array<[number, number, number]> }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 30);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      radii[i] = i === 0 ? 0.5 : 0.32;
      colors.set(linear(i === 0 ? (theme.dark ? '#9aa3ae' : '#4a525e') : theme.dark ? '#e8e6e1' : '#fbfaf7'), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.15), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, theme]);
  const sticks = useMemo<Segment[]>(() => positions.slice(1).map((p) => ({ a: positions[0]!, b: p, color: linear(theme.ruleStrong), radius: 0.08 })), [positions, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.caveat);
    const out: ColoredLine[] = [];
    for (let i = 1; i < 5; i++) for (let j = i + 1; j < 5; j++) out.push({ a: positions[i]!, b: positions[j]!, rgba: [c[0], c[1], c[2], 0.55] });
    return out;
  }, [positions, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      <Lines lines={lines} />
    </>
  );
}

const W = 380;
const M = { l: 44, r: 10, t: 10, b: 30 };
const xOf = (th: number, w = W) => M.l + ((th - 30) / 60) * (w - M.l - M.r);

function Axes({ H, yTicks, y, yLabel, theta }: { H: number; yTicks: number[]; y: (v: number) => number; yLabel: string; theta: number }) {
  return (
    <>
      {yTicks.map((v) => (
        <g key={v}>
          <line className={v === 0 ? 'axis' : 'grid'} x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} />
          <text className="tick" x={M.l - 4} y={y(v) + 3} textAnchor="end">
            {v}
          </text>
        </g>
      ))}
      {[30, 45, 60, 75, 90].map((t) => (
        <text key={t} className="tick" x={xOf(t)} y={H - M.b + 13} textAnchor="middle">
          {t}°
        </text>
      ))}
      <line className="am-now" x1={xOf(TD_THETA)} x2={xOf(TD_THETA)} y1={M.t} y2={H - M.b} />
      <text className="am-mark" x={xOf(TD_THETA) + 3} y={M.t + 8}>
        Td
      </text>
      <line className="axis" x1={xOf(theta)} x2={xOf(theta)} y1={M.t} y2={H - M.b} />
      <text className="axis-label" x={W - M.r} y={H - 3} textAnchor="end">
        θ
      </text>
      <text className="axis-label" x={11} y={M.t + 2} transform={`rotate(-90 11 ${M.t + 2})`} textAnchor="end">
        {yLabel}
      </text>
    </>
  );
}

function FeaturePlot({ theta }: { theta: number }) {
  const H = 170;
  const yMax = 140;
  const y = (v: number) => M.t + (1 - v / yMax) * (H - M.t - M.b);
  const path = (f: (r: (typeof ROWS)[number]) => number) => ROWS.map((r, i) => `${i ? 'L' : 'M'}${xOf(r.theta).toFixed(1)},${y(Math.min(yMax, 100 * f(r))).toFixed(1)}`).join('');
  const series = [
    { key: 'c0', d: path((r) => r.conv[0]!.C), color: 'var(--atlas-phys)', dashed: false },
    { key: 'c1', d: path((r) => r.conv[1]!.C), color: 'var(--atlas-arch)', dashed: false },
    { key: 'c2', d: path((r) => r.conv[2]!.C), color: 'var(--atlas-lit)', dashed: false },
    { key: 'h0', d: path((r) => r.conv[0]!.H), color: 'var(--atlas-caveat)', dashed: true },
  ];
  return (
    <figure className="lab-plot am-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Relative change of the released model's environment features against the twist angle: zero for carbon in pass 1 at every angle, non-zero for hydrogen in pass 1 and for carbon in passes 2 and 3.">
        <Axes H={H} yTicks={[0, 50, 100]} y={y} yLabel="change (%)" theta={theta} />
        {series.map((s) => (
          <path key={s.key} className={`am-curve${s.dashed ? ' is-dashed' : ''}`} d={s.d} style={{ stroke: s.color }} />
        ))}
      </svg>
      <figcaption className="am-cap">‖x(θ) − x(Td)‖ / ‖x(Td)‖ for each pass’s environment features x (the input of its MLP from the geometry and neighbour features).</figcaption>
    </figure>
  );
}

function EnergyPlot({ theta }: { theta: number }) {
  const H = 140;
  const eMax = Math.max(...ROWS.map((r) => r.E));
  const top = Math.ceil(eMax);
  const y = (v: number) => M.t + (1 - v / top) * (H - M.t - M.b);
  const d = ROWS.map((r, i) => `${i ? 'L' : 'M'}${xOf(r.theta).toFixed(1)},${y(r.E).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot am-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Released-model energy relative to tetrahedral methane against the twist angle, zero at Td and rising on both sides.">
        <Axes H={H} yTicks={Array.from({ length: top + 1 }, (_, k) => k)} y={y} yLabel="E − E(Td) (eV)" theta={theta} />
        <path className="am-curve" d={d} style={{ stroke: 'var(--atlas-ink)' }} />
      </svg>
      <figcaption className="am-cap">Total energy of the released checkpoint along the path (rigid C–H; not a minimum-energy path).</figcaption>
    </figure>
  );
}
