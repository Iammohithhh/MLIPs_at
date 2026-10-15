import { useEffect, useId, useMemo, useRef, useState } from 'react';
import './PhysicsLab.css';
import ethanol from '~/data/structures/ethanol.json';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  ImpostorAtoms,
  Sticks,
  linear,
  useShadingUniforms,
  type AtomBuffers,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, usePrefersReducedMotion, type SceneTheme } from '~/three/theme';
import { covalentRadius, elementColor, hexToLinearRgb } from '~/lib/structure/elements';
import type { Vec3 } from '~/lib/structure/types';
import {
  apply,
  centroid,
  distort,
  energy,
  equivarianceError,
  forces,
  rotateAbout,
  rotation,
  springModel,
  type ModelId,
} from '~/lib/physics/rotation-toy';

// Physics Lab P4: rotate a molecule in the lab frame and compare what an invariant
// model and a raw-coordinate model predict. Solid arrows: forces computed on the
// rotated molecule, F(QR). Overlay lines: the original forces rotated, Q·F(R).

type AxisId = 'x' | 'y' | 'z' | 'tilted';
const AXES: Record<AxisId, { v: Vec3; label: string }> = {
  x: { v: [1, 0, 0], label: 'x' },
  y: { v: [0, 1, 0], label: 'y' },
  z: { v: [0, 0, 1], label: 'z' },
  tilted: { v: [0.45, 0.6, 0.66], label: 'tilted' },
};
const MODELS: Array<{ id: ModelId; label: string }> = [
  { id: 'invariant', label: 'Distance-only model' },
  { id: 'coordinate', label: 'Raw-coordinate model' },
];

const SYMBOLS = ethanol.symbols;
const REF = ethanol.positions as Vec3[];
const SPRINGS = springModel(SYMBOLS, REF);
const R0 = (() => {
  const d = distort(REF);
  const c = centroid(d);
  return d.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]] as Vec3);
})();
const F_SCALE = 0.3; // Å of arrow per eV/Å

const deg = (t: number) => (t * Math.PI) / 180;

export default function RotationLab() {
  const theme = useSceneTheme();
  const reduced = usePrefersReducedMotion();
  const ids = useId();
  const [model, setModel] = useState<ModelId>('invariant');
  const [axis, setAxis] = useState<AxisId>('tilted');
  const [theta, setTheta] = useState(70);
  const [spinning, setSpinning] = useState(false);

  // Spin: advance θ continuously (off under reduced motion).
  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!spinning || reduced) return;
    let last = performance.now();
    const step = (t: number) => {
      const dt = t - last;
      last = t;
      setTheta((v) => (v + dt * 0.04) % 360);
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [spinning, reduced]);

  const Q = useMemo(() => rotation(AXES[axis].v, deg(theta)), [axis, theta]);
  const R = useMemo(() => rotateAbout(R0, Q), [Q]);
  const F = useMemo(() => forces(model, SPRINGS, R), [model, R]);
  const QF = useMemo(() => forces(model, SPRINGS, R0).map((v) => apply(Q, v)), [model, Q]);
  const E = energy(model, SPRINGS, R);
  const E0 = energy(model, SPRINGS, R0);
  const err = equivarianceError(model, SPRINGS, R0, Q);
  const dE = E - E0;
  const invariant = Math.abs(dE) < 1e-6;
  const equivariant = err < 1e-6;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [5.6, -9.4, 4.9], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Ethanol rotated by ${theta.toFixed(0)}° about the ${AXES[axis].label} axis, with force arrows from the ${model === 'invariant' ? 'distance-only' : 'raw-coordinate'} toy model.`}
            minDistance={6}
            maxDistance={40}
          >
            <RotationScene theme={theme} R={R} F={F} QF={QF} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Model">
              {MODELS.map((m) => (
                <button key={m.id} type="button" aria-pressed={model === m.id} onClick={() => setModel(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.phys }} /> F(QR): forces on the rotated molecule
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.lit }} /> Q·F(R): original forces, rotated (thin)
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> lab x axis
            </li>
          </ul>
        </div>
        <SymmetryPlot axis={axis} theta={theta} model={model} theme={theme} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-t`} data-sym="Q">
            <span>
              rotation <i>θ</i>
            </span>
            <input
              id={`${ids}-t`}
              type="range"
              min={0}
              max={360}
              step={1}
              value={Math.round(theta)}
              onChange={(e) => {
                setSpinning(false);
                setTheta(Number(e.target.value));
              }}
            />
            <output>{theta.toFixed(0)}°</output>
          </label>
          <label htmlFor={`${ids}-axis`}>
            <span>about axis</span>
            <select id={`${ids}-axis`} value={axis} onChange={(e) => setAxis(e.target.value as AxisId)}>
              <option value="tilted">tilted (0.45, 0.60, 0.66)</option>
              <option value="x">lab x</option>
              <option value="y">lab y</option>
              <option value="z">lab z</option>
            </select>
          </label>
          {!reduced && (
            <button type="button" className="lab-btn" aria-pressed={spinning} onClick={() => setSpinning((s) => !s)}>
              {spinning ? 'Stop' : 'Spin the molecule'}
            </button>
          )}
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div data-sym="E">
            <dt className="math">Energy E(QR)</dt>
            <dd>{E.toFixed(4)} eV</dd>
          </div>
          <div className={invariant ? 'is-good' : 'is-alert'}>
            <dt>Change vs θ = 0</dt>
            <dd>
              {dE >= 0 ? '+' : '−'}
              {Math.abs(dE).toFixed(4)} eV
            </dd>
          </div>
          <div className={equivariant ? 'is-good' : 'is-alert'} data-sym="F">
            <dt className="math">max |F(QR) − Q·F(R)|</dt>
            <dd>{err < 1e-9 ? '0' : err.toFixed(3)} eV/Å</dd>
          </div>
        </dl>
        <p className="lab-note">
          {model === 'invariant' ? (
            <>
              The distance-only model sees nothing but bond lengths, so turning the molecule changes nothing it can see:
              the energy stays put and every force arrow turns with the molecule — the thin purple arrows sit exactly inside
              the teal ones.
            </>
          ) : axis === 'x' ? (
            <>
              Rotating about the lab <i>x</i> axis leaves every bond’s <i>x</i>-component unchanged, so even the
              raw-coordinate model looks symmetric here. Pick another axis to expose it.
            </>
          ) : (
            <>
              The raw-coordinate model reads bond vectors along the lab <i>x</i> axis, so the same molecule gets a
              different energy in a different orientation, and its forces no longer turn with it: the thin purple arrows
              peel away from the teal ones.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function RotationScene({ theme, R, F, QF }: { theme: SceneTheme; R: Vec3[]; F: Vec3[]; QF: Vec3[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 10, 34);
  const mode = theme.dark ? 'dark' : 'light';
  const colors = useMemo(() => SYMBOLS.map((s) => hexToLinear(elementColor(s, mode))), [mode]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = R.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const col = new Float32Array(n * 3);
    R.forEach((p, i) => {
      positions.set(p, 3 * i);
      radii[i] = Math.max(0.22, 0.45 * covalentRadius(SYMBOLS[i]!).r);
      col.set(colors[i]!, 3 * i);
    });
    return { positions, radii, colors: col, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [R, colors]);

  const sticks = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    for (const [i, j] of SPRINGS.bonds) {
      const a = R[i]!;
      const b = R[j]!;
      const mid: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      out.push({ a, b: mid, color: colors[i]!, radius: 0.11 });
      out.push({ a: mid, b, color: colors[j]!, radius: 0.11 });
    }
    return out;
  }, [R, colors]);

  const phys = linear(theme.phys);
  const arrows = useMemo<Segment[]>(
    () =>
      F.flatMap((f, i) => {
        const len = Math.hypot(f[0], f[1], f[2]);
        if (len < 1e-6) return [];
        const p = R[i]!;
        return [{ a: p, b: [p[0] + f[0] * F_SCALE, p[1] + f[1] * F_SCALE, p[2] + f[2] * F_SCALE] as Vec3, color: phys, radius: 0.05 }];
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [F, R, theme.phys],
  );

  const lit = linear(theme.lit);
  // Thin purple arrows for Q·F(R): hidden inside the teal arrows when the model is
  // equivariant, visibly separate when it is not.
  const expected = useMemo<Segment[]>(
    () =>
      QF.flatMap((f, i) => {
        const len = Math.hypot(f[0], f[1], f[2]);
        if (len < 1e-6) return [];
        const p = R[i]!;
        const tip: Vec3 = [p[0] + f[0] * F_SCALE, p[1] + f[1] * F_SCALE, p[2] + f[2] * F_SCALE];
        return [{ a: p, b: tip, color: lit, radius: 0.024 }];
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [QF, R, theme.lit],
  );

  // The fixed lab x axis, drawn under the molecule: the one direction the raw-coordinate model reads.
  const caveat = linear(theme.caveat);
  const triad = useMemo<Segment[]>(
    () => [{ a: [-2.8, 0, -1.9], b: [2.4, 0, -1.9], color: caveat, radius: 0.04 }],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [theme.caveat],
  );

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.22} />}
      {expected.length > 0 && <Arrows arrows={expected} uniforms={uniforms} head={0.2} />}
      <Arrows arrows={triad} uniforms={uniforms} head={0.3} />
    </>
  );
}

function hexToLinear(rgb: [number, number, number]): [number, number, number] {
  // elementColor returns sRGB in [0, 1]; convert to linear for the shader.
  const hex = '#' + rgb.map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, '0')).join('');
  return hexToLinearRgb(hex);
}

function SymmetryPlot({ axis, theta, model, theme }: { axis: AxisId; theta: number; model: ModelId; theme: SceneTheme }) {
  const W = 520;
  const PH = 130;
  const GAP = 34;
  const M = { l: 58, r: 14, t: 18, b: 36 };
  const H = M.t + 2 * PH + GAP + M.b;
  const samples = useMemo(() => {
    const thetas = Array.from({ length: 181 }, (_, k) => k * 2);
    const run = (m: ModelId) => {
      const e0 = energy(m, SPRINGS, R0);
      return thetas.map((t) => {
        const Q = rotation(AXES[axis].v, deg(t));
        return { t, dE: energy(m, SPRINGS, rotateAbout(R0, Q)) - e0, err: equivarianceError(m, SPRINGS, R0, Q) };
      });
    };
    return { invariant: run('invariant'), coordinate: run('coordinate') };
  }, [axis]);

  const all = [...samples.invariant, ...samples.coordinate];
  const eMax = Math.max(0.05, ...all.map((s) => Math.abs(s.dE))) * 1.15;
  const errMax = Math.max(0.05, ...all.map((s) => s.err)) * 1.15;
  const x = (t: number) => M.l + (t / 360) * (W - M.l - M.r);
  const y1 = (v: number) => M.t + PH / 2 - (v / eMax) * (PH / 2);
  const top2 = M.t + PH + GAP;
  const y2 = (v: number) => top2 + PH - (v / errMax) * PH;
  const line = (pts: Array<{ t: number; v: number }>, y: (v: number) => number) =>
    pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join('');

  const series = (m: ModelId) => ({
    e: line(samples[m].map((s) => ({ t: s.t, v: s.dE })), y1),
    q: line(samples[m].map((s) => ({ t: s.t, v: s.err })), y2),
  });
  const inv = series('invariant');
  const coo = series('coordinate');

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy change and equivariance error against rotation angle about the ${AXES[axis].label} axis. The distance-only model is flat at zero in both; the raw-coordinate model varies${axis === 'x' ? ' only for other axes' : ''}.`}>
        <text x={M.l} y={M.t - 6} className="axis-label">
          ΔE(θ) = E(QR) − E(R) [eV]
        </text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={PH} className="frame" />
        <line x1={M.l} x2={W - M.r} y1={y1(0)} y2={y1(0)} className="axis" />
        {[-eMax / 1.15, eMax / 1.15].map((v) => (
          <text key={v} x={M.l - 8} y={y1(v) + 4} textAnchor="end" className="tick">
            {v > 0 ? '+' : '−'}
            {Math.abs(v).toFixed(2)}
          </text>
        ))}
        <text x={M.l - 8} y={y1(0) + 4} textAnchor="end" className="tick">
          0
        </text>
        <path d={coo.e} className={`curve broken${model === 'coordinate' ? ' is-active' : ''}`} />
        <path d={inv.e} className={`curve sym${model === 'invariant' ? ' is-active' : ''}`} />

        <text x={M.l} y={top2 - 6} className="axis-label">
          max |F(QR) − Q·F(R)| [eV/Å]
        </text>
        <rect x={M.l} y={top2} width={W - M.l - M.r} height={PH} className="frame" />
        <text x={M.l - 8} y={y2(0) + 4} textAnchor="end" className="tick">
          0
        </text>
        <text x={M.l - 8} y={y2(errMax / 1.15) + 4} textAnchor="end" className="tick">
          {(errMax / 1.15).toFixed(2)}
        </text>
        <path d={coo.q} className={`curve broken${model === 'coordinate' ? ' is-active' : ''}`} />
        <path d={inv.q} className={`curve sym${model === 'invariant' ? ' is-active' : ''}`} />

        {[0, 90, 180, 270, 360].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 18} textAnchor="middle" className="tick">
            {t}°
          </text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">
          θ
        </text>
        <line x1={x(theta)} x2={x(theta)} y1={M.t} y2={top2 + PH} className="r-line" />
      </svg>
      <figcaption className="lab-legend">
        <span className="item sym">
          <span className="swatch" style={{ background: theme.phys }} /> distance-only model
        </span>
        <span className="item broken">
          <span className="swatch" style={{ background: theme.caveat }} /> raw-coordinate model
        </span>
      </figcaption>
    </figure>
  );
}
