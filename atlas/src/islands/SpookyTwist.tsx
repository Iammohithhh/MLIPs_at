import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Spooky.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Sticks, linear, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor } from '~/lib/structure/elements';
import { MAGIC_THETA, makeToySpooky, powerSpectrum, runToy, trigonalTwist, type Vec3 } from '~/lib/models/spookynet';
import { Legend, LinePlot } from './spooky/LinePlot';
import { cssColor } from './spooky/colors';

// Fig. 2: an FeO₆ core twisted from an octahedron (τ = 60°) to a trigonal prism (τ = 0°). Live: the
// metal's power-spectrum invariants a_l (l = 0 … 4) and a toy SpookyNet (fixed random weights) run
// with one and two interaction modules.

const MODEL = makeToySpooky();
const TAUS = Array.from({ length: 61 }, (_, i) => i);
const L_NAMES = ['s', 'p', 'd', 'f', 'g'];

const rel = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]!)) / Math.hypot(...b);

function sweep(theta: number) {
  const ref = trigonalTwist(60, theta);
  const r1 = runToy(MODEL, ref, { modules: 1 });
  const r2 = runToy(MODEL, ref, { modules: 2 });
  return TAUS.map((tau) => {
    const mol = trigonalTwist(tau, theta);
    const a = runToy(MODEL, mol, { modules: 1 });
    const b = runToy(MODEL, mol, { modules: 2 });
    return {
      tau,
      m1: rel(a.xs[0]![0]!, r1.xs[0]![0]!),
      m2: rel(b.xs[1]![0]!, r2.xs[1]![0]!),
      l1: rel(a.xs[0]![1]!, r1.xs[0]![1]!),
    };
  });
}

export default function SpookyTwist() {
  const theme = useSceneTheme();
  const id = useId();
  const [tau, setTau] = useState(30);
  const [theta, setTheta] = useState(MAGIC_THETA);
  const mol = useMemo(() => trigonalTwist(tau, theta), [tau, theta]);
  const spectrum = useMemo(() => [0, 1, 2, 3, 4].map((l) => powerSpectrum(mol.positions.slice(1), l)), [mol]);
  const octa = useMemo(() => [0, 1, 2, 3, 4].map((l) => powerSpectrum(trigonalTwist(60, theta).positions.slice(1), l)), [theta]);
  const rows = useMemo(() => sweep(theta), [theta]);
  const row = rows[tau]!;
  const yMax = Math.max(0.02, ...rows.flatMap((r) => [r.m2, r.l1])) * 1.1;
  const magic = theta === MAGIC_THETA;
  const shape = tau === 60 ? (magic ? 'octahedron' : 'trigonal antiprism') : tau === 0 ? 'trigonal prism' : 'twisted';
  const view = useMemo(() => ({ position: [8.5, -10, 5.5] as [number, number, number], target: [0, 0, 0] as [number, number, number] }), []);
  const barMax = Math.max(...spectrum, ...octa);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage lab-stage--short">
          <Stage view={view} resetKey={0} background={theme.canvas} label={`FeO6 core, ${shape}: the upper triangle of oxygens is turned by ${60 - tau} degrees from the octahedron, every Fe–O distance 2.0 Å.`} minDistance={5} maxDistance={40}>
            <TwistScene theme={theme} positions={mol.positions} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: cssColor('Fe', theme.dark) }} /> Fe
            </li>
            <li>
              <span className="dot" style={{ background: cssColor('O', theme.dark) }} /> O
            </li>
            <li>
              τ = {tau}° · θ = {theta.toFixed(1)}° · {shape}
            </li>
          </ul>
        </div>
        <div className="sp-side">
          <p className="sp-h">Metal’s invariants a_l (live)</p>
          <div className="sp-bars">
            {spectrum.map((v, l) => (
              <div className="sp-bar" key={l}>
                <span>
                  {L_NAMES[l]} (l = {l})
                </span>
                <span className="track">
                  <span className="fill" style={{ width: `${(100 * v) / barMax}%`, background: l <= 2 ? 'var(--atlas-phys)' : 'var(--atlas-caveat)' }} />
                </span>
                <output>{v < 1e-12 ? '0' : v.toFixed(2)}</output>
              </div>
            ))}
          </div>
          <p className="sp-cap">Green: resolved by one module (l ≤ 2). Brown: not.</p>
          <p className="sp-h">Toy SpookyNet: change from the octahedron (live)</p>
          <LinePlot
            H={170}
            x={{ min: 0, max: 60, ticks: [0, 15, 30, 45, 60], label: 'τ (°)', fmt: (v) => `${v}°` }}
            y={{ min: 0, max: yMax, ticks: [0, yMax / 2, yMax].map((v) => +v.toPrecision(2)), label: 'relative change', fmt: (v) => `${(100 * v).toFixed(v < 0.1 ? 1 : 0)}%` }}
            series={[
              { key: 'l1', points: rows.map((r) => [r.tau, r.l1] as [number, number]), color: 'var(--atlas-caveat)', dashed: true },
              { key: 'm2', points: rows.map((r) => [r.tau, r.m2] as [number, number]), color: 'var(--atlas-arch)' },
              { key: 'm1', points: rows.map((r) => [r.tau, r.m1] as [number, number]), color: 'var(--atlas-phys)' },
            ]}
            cursor={tau}
            ariaLabel="Relative change of the toy model's features against the twist angle: zero for the metal after one module at every angle, non-zero for an oxygen after one module and for the metal after two."
          />
          <Legend
            items={[
              { label: 'Fe, after module 1', color: 'var(--atlas-phys)' },
              { label: 'Fe, after module 2', color: 'var(--atlas-arch)' },
              { label: 'O, after module 1', color: 'var(--atlas-caveat)', dashed: true },
            ]}
          />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${id}-t`}>
            <span>twist τ</span>
            <input id={`${id}-t`} type="range" min={0} max={60} step={1} value={tau} onChange={(e) => setTau(Number(e.target.value))} />
            <output>{tau}°</output>
          </label>
          <label htmlFor={`${id}-p`}>
            <span>polar angle θ</span>
            <input id={`${id}-p`} type="range" min={42} max={68} step={0.01} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>
              {theta.toFixed(2)}° {magic ? <small>(octahedral)</small> : null}
            </output>
          </label>
          <div className="seg" role="group" aria-label="Polar angle preset">
            <button type="button" aria-pressed={theta === MAGIC_THETA} onClick={() => setTheta(MAGIC_THETA)}>
              θ = arccos(1/√3)
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className="is-good">
            <dt>Fe after one module vs octahedron</dt>
            <dd>{row.m1 < 1e-12 ? 'identical' : row.m1.toExponential(1)}</dd>
          </div>
          <div className={row.m2 > 1e-9 ? 'is-alert' : undefined}>
            <dt>Fe after two modules vs octahedron</dt>
            <dd>{row.m2 < 1e-12 ? 'identical' : `${(100 * row.m2).toFixed(2)}%`}</dd>
          </div>
          <div>
            <dt>f invariant a₃ (prism → octahedron)</dt>
            <dd>
              {spectrum[3]! < 1e-12 ? '0' : spectrum[3]!.toFixed(3)} <small>∝ 1 + cos 3τ</small>
            </dd>
          </div>
          <div>
            <dt>shortest O–O between the triangles</dt>
            <dd>{interTriangle(mol.positions).toFixed(3)} Å</dd>
          </div>
        </dl>
        <p className="lab-note">
          <strong>Toy model:</strong> F = 12 features, K = 8 radial functions, fixed random weights (not trained); what it shows holds for any
          weights. The metal’s six neighbours are all oxygen at 2.0 Å, so in the first module every one of them contributes the same features and
          the same radial factor; the metal can only see Σ<sub>j</sub> Y<sub>l</sub><sup>m</sup>(r̂<sub>j</sub>) for l ≤ 2. Turning one triangle
          leaves those sums unchanged at every polar angle (the θ slider changes the d sum, the twist never does). The oxygens do feel the twist,
          because their distances to the other triangle change, and the metal reads that in module 2.
        </p>
      </div>
    </div>
  );
}

function interTriangle(p: Vec3[]): number {
  let m = Infinity;
  for (let i = 1; i <= 3; i++) for (let j = 4; j <= 6; j++) m = Math.min(m, Math.hypot(p[i]![0] - p[j]![0], p[i]![1] - p[j]![1], p[i]![2] - p[j]![2]));
  return m;
}

function TwistScene({ theme, positions }: { theme: SceneTheme; positions: Vec3[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 30);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      radii[i] = i === 0 ? 0.55 : 0.42;
      colors.set(i === 0 ? elementColor('Fe', theme.dark ? 'dark' : 'light') : elementColor('O', theme.dark ? 'dark' : 'light'), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.15), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, theme]);
  const sticks = useMemo<Segment[]>(() => positions.slice(1).map((p) => ({ a: positions[0]!, b: p, color: linear(theme.ruleStrong), radius: 0.08 })), [positions, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const tri = linear(theme.phys);
    const link = linear(theme.caveat);
    const out: ColoredLine[] = [];
    const edges: Array<[number, number]> = [
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 5],
      [5, 6],
      [6, 4],
    ];
    for (const [a, b] of edges)
      out.push({ a: positions[a]!, b: positions[b]!, rgba: [tri[0], tri[1], tri[2], 0.8] });
    for (let i = 1; i <= 3; i++) out.push({ a: positions[i]!, b: positions[i + 3]!, rgba: [link[0], link[1], link[2], 0.7] });
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
