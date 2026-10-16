import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, Sticks, linear, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import {
  axisRot,
  dot,
  equigradExperiment,
  features,
  featureRotJacobian,
  fitReadout,
  makeToyOrb,
  rotateAll,
  scoreFit,
  swSite,
  type EquigradExperiment,
  type Vec3,
} from '~/lib/models/orb';

// Equigrad (Rhodes et al. 2025, Eq. 1): Δ_rot = ∂E(r·exp(G − Gᵀ))/∂G at G = 0 vanishes for an
// invariant energy. A frozen toy Orb GNS (random weights, Orb-v3 edge features) is given a linear
// last layer fitted, in closed form, to Stillinger–Weber energies of rattled Si₅ clusters, with
// and without rotated copies of the data and with an L2 penalty λ‖Δ_rot‖² on every sample.
// Everything below is computed in the browser.

const AXIS: Vec3 = [1, 2, 3];
const LAMBDAS = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000];
const THETAS = Array.from({ length: 73 }, (_, k) => -180 + k * 5);

export default function OrbEquigrad({ base }: { base: Vec3[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [set, setSet] = useState<'fixed' | 'rotated'>('fixed');
  const [li, setLi] = useState(0);
  const [theta, setTheta] = useState(60);
  const [ex, setEx] = useState<EquigradExperiment | null>(null);
  const model = useMemo(() => makeToyOrb({ d: 64 }), []);

  // ~1 s of work: defer it until after the first paint.
  useEffect(() => {
    const t = window.setTimeout(() => setEx(equigradExperiment(model, base)), 30);
    return () => window.clearTimeout(t);
  }, [model, base]);

  const lambda = LAMBDAS[li]!;
  const fit = useMemo(() => (ex ? fitReadout(ex.train[set], lambda) : null), [ex, set, lambda]);
  const score = useMemo(() => (ex && fit ? scoreFit(fit, ex.train[set], ex.test) : null), [ex, fit, set]);
  const probe = ex?.test[0]?.vectors ?? base;
  const at = (th: number) => rotateAll(probe, axisRot(AXIS, (th * Math.PI) / 180));
  const curveFeatures = useMemo(() => (ex ? THETAS.map((t) => features(model, at(t))) : null), [ex, model]);
  const curve = curveFeatures && fit ? curveFeatures.map((f) => dot(fit, f)) : null;
  const reference = useMemo(() => swSite(probe), [probe]);
  const current = useMemo(() => at(theta), [probe, theta]);
  // Net torque T = (−Δ₁₂, Δ₀₂, −Δ₀₁) of the fitted conservative model at the current orientation.
  const torque = useMemo<Vec3 | null>(() => {
    if (!fit) return null;
    const [d01, d02, d12] = featureRotJacobian(model, current).map((g) => dot(fit, g));
    return [-d12!, d02!, -d01!];
  }, [fit, model, current]);
  const eNow = fit ? dot(fit, features(model, current)) : null;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [8, -11, 6.5], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`A rattled Si5 cluster rotated by ${theta} degrees about the (1, 2, 3) axis. The arrow is the net torque that the fitted model's forces exert on the isolated cluster; an invariant model exerts none.`}
            minDistance={3}
            maxDistance={20}
          >
            <Scene theme={theme} vectors={current} torque={torque} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Training set">
              <button type="button" aria-pressed={set === 'fixed'} onClick={() => setSet('fixed')}>
                one orientation (120)
              </button>
              <button type="button" aria-pressed={set === 'rotated'} onClick={() => setSet('rotated')}>
                augmented: 3 random rotations each (360)
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.fail }} /> net torque Σ r × F (scaled)
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.caveat }} /> rotation axis (1, 2, 3)
            </li>
          </ul>
        </div>
        <Chart theme={theme} curve={curve} reference={reference} theta={theta} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-l`}>
            <span>Equigrad weight λ</span>
            <input id={`${ids}-l`} type="range" min={0} max={LAMBDAS.length - 1} step={1} value={li} onChange={(e) => setLi(Number(e.target.value))} />
            <output>{lambda === 0 ? 'off' : lambda >= 1 ? lambda.toFixed(0) : lambda.toExponential(0)}</output>
          </label>
          <label htmlFor={`${ids}-t`}>
            <span>rotation of the test cluster</span>
            <input id={`${ids}-t`} type="range" min={-180} max={180} step={5} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>{theta}°</output>
          </label>
        </div>
        {score && eNow !== null && torque ? (
          <dl className="lab-readout" aria-live="polite">
            <div>
              <dt>training RMSE</dt>
              <dd>{(1000 * score.trainRmse).toFixed(0)} meV</dd>
            </div>
            <div className={score.rmse > 3 * score.trainRmse ? 'is-alert' : undefined}>
              <dt>test RMSE, random orientations</dt>
              <dd>{(1000 * score.rmse).toFixed(0)} meV</dd>
            </div>
            <div>
              <dt>spread over orientations</dt>
              <dd>{(1000 * score.spread).toFixed(score.spread < 0.01 ? 1 : 0)} meV</dd>
            </div>
            <div>
              <dt>mean ‖Δ<sub>rot</sub>‖ on test</dt>
              <dd>{score.deltaRot.toExponential(1)} eV</dd>
            </div>
            <div>
              <dt>|torque| here</dt>
              <dd>{Math.hypot(...torque).toExponential(1)} eV</dd>
            </div>
          </dl>
        ) : (
          <p className="lab-note" aria-live="polite">Featurising 640 clusters and fitting…</p>
        )}
        <p className="lab-note">
          Only the last, linear layer is fitted; everything before it is frozen and random. That makes the λ-penalised fit a ridge
          regression, but it also means the toy cannot become more invariant without giving up accuracy. A fully trained network
          can change its features as well. Reference: Stillinger–Weber site energy of the central Si atom (eV). Spread = standard
          deviation of the prediction over 8 random orientations, averaged over 20 test clusters.
        </p>
      </div>
    </div>
  );
}

function Scene({ theme, vectors, torque }: { theme: SceneTheme; vectors: Vec3[]; torque: Vec3 | null }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 6, 22);
  const buffers = useMemo<AtomBuffers>(() => {
    const atoms: Vec3[] = [[0, 0, 0], ...vectors];
    const n = atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n).fill(0.5);
    const colors = new Float32Array(n * 3);
    atoms.forEach((a, i) => {
      pos.set(a, 3 * i);
      colors.set(linear(i === 0 ? theme.ink : theme.arch), 3 * i);
    });
    radii[0] = 0.56;
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [vectors, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.inkMuted);
    const u = Math.hypot(...AXIS);
    const ax = AXIS.map((x) => (3.2 * x) / u) as Vec3;
    return [
      ...vectors.map((v) => ({ a: [0, 0, 0] as Vec3, b: v, color: c, radius: 0.09 })),
      { a: ax.map((x) => -x) as Vec3, b: ax, color: linear(theme.caveat), radius: 0.03 },
    ];
  }, [vectors, theme]);
  const arrows = useMemo<Segment[]>(() => {
    if (!torque) return [];
    const t = Math.hypot(...torque);
    if (t < 1e-9) return [];
    // Length grows like log10 so both large and tiny torques stay readable.
    const L = Math.max(0.35, Math.min(3, 1.5 + 0.6 * Math.log10(t)));
    return [{ a: [0, 0, 0], b: torque.map((x) => (L * x) / t) as Vec3, color: linear(theme.fail), radius: 0.08 }];
  }, [torque, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      {arrows.length ? <Arrows arrows={arrows} uniforms={uniforms} /> : null}
    </>
  );
}

function Chart({ theme, curve, reference, theta }: { theme: SceneTheme; curve: number[] | null; reference: number; theta: number }) {
  const W = 520;
  const H = 340;
  const M = { l: 60, r: 16, t: 26, b: 42 };
  const vals = [...(curve ?? []), reference];
  const lo = Math.min(...vals, reference - 0.05);
  const hi = Math.max(...vals, reference + 0.05);
  const pad = 0.08 * (hi - lo);
  const x = (t: number) => M.l + ((t + 180) / 360) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - M.t - M.b);
  const path = curve ? THETAS.map((t, k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)},${y(curve[k]!).toFixed(1)}`).join('') : '';
  const k = THETAS.indexOf(theta);
  const yt = (() => {
    const span = hi - lo;
    const step = span > 1 ? 0.5 : span > 0.4 ? 0.1 : span > 0.2 ? 0.05 : 0.02;
    const out: number[] = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return out;
  })();
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Predicted energy of one test cluster as it is rotated through 360 degrees, against the rotation-invariant reference energy.">
        <text x={M.l} y={M.t - 10} className="axis-label">energy of one test cluster (eV)</text>
        {yt.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toFixed(2)}</text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={y(reference)} y2={y(reference)} style={{ stroke: theme.phys, strokeWidth: 2, strokeDasharray: '6 4' }} />
        <text x={W - M.r - 4} y={y(reference) - 6} textAnchor="end" className="tick">Stillinger–Weber (invariant)</text>
        {curve ? <path d={path} className="curve" style={{ stroke: theme.arch }} /> : null}
        <line x1={x(0)} x2={x(0)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
        <text x={x(0) + 4} y={M.t + 12} className="tick">orientation of the training data</text>
        <line x1={x(theta)} x2={x(theta)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.1 }} />
        {curve && k >= 0 ? <circle cx={x(theta)} cy={y(curve[k]!)} r={5} style={{ fill: theme.arch }} /> : null}
        {[-180, -90, 0, 90, 180].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 16} textAnchor="middle" className="tick">{t}°</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">rotation about (1, 2, 3)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item">
          <span className="swatch" style={{ background: theme.arch }} /> fitted toy Orb
        </span>
        <span className="item">
          <span className="swatch" style={{ background: theme.phys }} /> reference
        </span>
      </figcaption>
    </figure>
  );
}
