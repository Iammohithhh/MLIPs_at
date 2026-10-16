import { useDeferredValue, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import {
  applyMat,
  axisRotation,
  collinearSwitch,
  ecse,
  makeToyPet,
  pairFrame,
  petCutoff,
  petForward,
  rotateEnv,
  rotationSpread,
  type EcseOptions,
  type Environment,
  type Vec3,
} from '~/lib/models/pet';
import { toEnvironment, type PetEnvironmentData } from './PetAttention';

// ECSE (Pozdnyakov & Ceriotti 2023, Sec. 5): evaluate a non-invariant backbone in the local
// frame of every ordered pair of close neighbours and take a smoothly weighted average.
// Each frame turns with the structure, so the average cannot depend on orientation.

const PET = { rc: 4.5, width: 0.5 };
const AXIS: Vec3 = [1, 2, 3];
const THETAS = Array.from({ length: 73 }, (_, k) => k * 5);
const THETAS_S = Array.from({ length: 9 }, (_, k) => k * 45);
const STRETCH_RIN = 2.6;
const DS = Array.from({ length: 51 }, (_, k) => 1.9 + (1.4 * k) / 50);
const OMEGA = 0.05;
const D_OMEGA = 0.1;
const RIN_WIDTH = 0.5;

export default function PetEcse({ data }: { data: PetEnvironmentData }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [theta, setTheta] = useState(40);
  const [rinInput, setRin] = useState(2.6);
  const rin = useDeferredValue(rinInput);
  const model = useMemo(() => makeToyPet({ species: 3 }), []);
  const { env } = useMemo(() => toEnvironment(data), [data]);
  const opts: EcseOptions = { rin, rinWidth: RIN_WIDTH, omega: OMEGA, dOmega: D_OMEGA };
  const at = (e: Environment, th: number) => rotateEnv(e, axisRotation(AXIS, (th * Math.PI) / 180));

  const backbone = useMemo(() => THETAS.map((t) => petForward(model, at(env, t), PET).energy), [model, env]);
  const sym = useMemo(() => THETAS_S.map((t) => ecse(model, at(env, t), PET, opts).energy), [model, env, rin]);
  const spread = useMemo(() => rotationSpread((e) => petForward(model, e, PET).energy, env, 100), [model, env]);
  const symStd = useMemo(() => {
    const mean = sym.reduce((a, b) => a + b, 0) / sym.length;
    return Math.sqrt(sym.reduce((a, y) => a + (y - mean) ** 2, 0) / sym.length);
  }, [sym]);
  const frames = useMemo(() => ecse(model, env, PET, opts).frames, [model, env, rin]);

  // The shortest Zn–O bond stretches through a fixed r_in: smooth weights versus a plain average.
  const stretchCurves = useMemo(() => {
    const bond = env.neighbours.reduce((b, n, k) => (Math.hypot(...n.v) < Math.hypot(...env.neighbours[b]!.v) ? k : b), 0);
    const stretch = (dd: number): Environment => ({
      ...env,
      neighbours: env.neighbours.map((n, k) => {
        if (k !== bond) return n;
        const r = Math.hypot(...n.v);
        return { ...n, v: n.v.map((c) => (c * dd) / r) as Vec3 };
      }),
    });
    const o: EcseOptions = { rin: STRETCH_RIN, rinWidth: RIN_WIDTH, omega: OMEGA, dOmega: D_OMEGA };
    return (['smooth', 'plain'] as const).map((w) => DS.map((dd) => ecse(model, stretch(dd), PET, { ...o, weights: w }).energy));
  }, [model, env]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [10, -16, 9], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`The ZnO environment rotated by ${theta}° about the (1, 2, 3) axis. ${frames} local frames from pairs of neighbours inside r_in = ${rin.toFixed(2)} Å; the heaviest frame's axes are drawn and turn with the structure.`}
            minDistance={6}
            maxDistance={60}
          >
            <Scene theme={theme} env={at(env, theta)} rin={rin} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.fail }} /> x̂ = r̂<sub>j</sub>
            </li>
            <li>
              <span className="bar" style={{ background: theme.verified }} /> ŷ ∝ r̂<sub>j</sub> × r̂<sub>j′</sub>
            </li>
            <li>
              <span className="bar" style={{ background: theme.arch }} /> ẑ = x̂ × ŷ
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.caveat }} /> r<sub>in</sub>
            </li>
          </ul>
        </div>
        <RotationChart theme={theme} backbone={backbone} sym={sym} theta={theta} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-t`}>
            <span>rotate the structure</span>
            <input id={`${ids}-t`} type="range" min={0} max={360} step={5} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>{theta}°</output>
          </label>
          <label htmlFor={`${ids}-r`}>
            <span>frame cutoff r<sub>in</sub></span>
            <input id={`${ids}-r`} type="range" min={2.3} max={3.6} step={0.05} value={rinInput} onChange={(e) => setRin(Number(e.target.value))} />
            <output>{rinInput.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>frames = backbone evaluations</dt>
            <dd>{frames}</dd>
          </div>
          <div className="is-alert">
            <dt>backbone spread over 100 random rotations</dt>
            <dd>{spread.std.toExponential(2)}</dd>
          </div>
          <div>
            <dt>ECSE spread over the 9 rotations plotted</dt>
            <dd>{symStd < 1e-12 ? '< 10⁻¹² (round-off)' : symStd.toExponential(1)}</dd>
          </div>
        </dl>
        <StretchChart theme={theme} ds={DS} curves={stretchCurves} rin={STRETCH_RIN} />
        <p className="lab-note">
          In the rotation chart the backbone reads Cartesian components, so turning the structure changes its answer; the ECSE average is flat
          to round-off. In the bond chart one Zn–O bond is stretched through r<sub>in</sub> = {STRETCH_RIN} Å. With the paper’s weights
          w<sub>jj′</sub> = f<sub>c</sub>(r<sub>j</sub>) f<sub>c</sub>(r<sub>j′</sub>) q<sub>c</sub>(|r̂<sub>j</sub> × r̂<sub>j′</sub>|²) the frames of that
          bond fade out; with a plain average they vanish at once and the prediction jumps (the paper’s Fig. 1b). Each frame
          costs one backbone call, which is why the frame cutoff is kept short.
        </p>
      </div>
    </div>
  );
}

function heaviestFrame(env: Environment, rin: number): { j: number; k: number } | null {
  let best: { j: number; k: number; w: number } | null = null;
  env.neighbours.forEach((a, j) =>
    env.neighbours.forEach((b, k) => {
      if (j === k) return;
      const ra = Math.hypot(...a.v);
      const rb = Math.hypot(...b.v);
      if (ra >= rin || rb >= rin) return;
      const c = applyMat(pairFrame(a.v, b.v), b.v);
      const z = (c[2] / rb) ** 2; // |r̂_j × r̂_j′|² = sin² of the angle between them
      const w = petCutoff(ra, rin, RIN_WIDTH) * petCutoff(rb, rin, RIN_WIDTH) * collinearSwitch(z, OMEGA, D_OMEGA);
      if (w > (best?.w ?? 0) + 1e-9) best = { j, k, w };
    }),
  );
  return best;
}

function Scene({ theme, env, rin }: { theme: SceneTheme; env: Environment; rin: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 16, 60);
  const buffers = useMemo<AtomBuffers>(() => {
    const atoms = [{ v: [0, 0, 0] as Vec3, s: env.centre }, ...env.neighbours];
    const n = atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const tint = [linear(theme.arch), linear(theme.lit)];
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    atoms.forEach((a, i) => {
      pos.set(a.v, 3 * i);
      const r = Math.hypot(...a.v);
      if (i === 0) {
        radii[i] = 0.6;
        colors.set(linear(theme.ink), 0);
      } else {
        radii[i] = r < rin ? 0.45 : 0.28;
        colors.set(r < rin ? tint[a.s % 2]! : mixRGB(tint[a.s % 2]!, stone, 0.6), 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [env, rin, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const out: ColoredLine[] = [];
    const arch = linear(theme.arch);
    for (const n of env.neighbours) if (Math.hypot(...n.v) < rin) out.push({ a: [0, 0, 0], b: n.v, rgba: [arch[0], arch[1], arch[2], 0.35] });
    const f = heaviestFrame(env, rin);
    if (f) {
      const F = pairFrame(env.neighbours[f.j]!.v, env.neighbours[f.k]!.v);
      const L = 2.4;
      [theme.fail, theme.verified, theme.arch].forEach((col, a) => {
        const c = linear(col);
        out.push({ a: [0, 0, 0], b: [F[3 * a]! * L, F[3 * a + 1]! * L, F[3 * a + 2]! * L], rgba: [c[0], c[1], c[2], 1] });
      });
    }
    return out;
  }, [env, rin, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} overlay />
      <Shell center={[0, 0, 0]} radius={rin} color={theme.caveat} opacity={0.25} />
    </>
  );
}

function RotationChart({ theme, backbone, sym, theta }: { theme: SceneTheme; backbone: number[]; sym: number[]; theta: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 56, r: 16, t: 26, b: 44 };
  const all = [...backbone, ...sym];
  const pad = 0.1 * (Math.max(...all) - Math.min(...all) || 1);
  const lo = Math.min(...all) - pad;
  const hi = Math.max(...all) + pad;
  const x = (t: number) => M.l + (t / 360) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const line = (ts: number[], ys: number[]) => ts.map((t, k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)},${y(ys[k]!).toFixed(1)}`).join('');
  const k = Math.round(theta / 5);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Prediction against rotation angle: the backbone oscillates, the ECSE-symmetrised prediction is a flat line.">
        <text x={M.l} y={M.t - 10} className="axis-label">centre atom’s energy contribution (arb. units)</text>
        <path d={line(THETAS, backbone)} className="curve" style={{ stroke: theme.fail }} />
        <path d={line(THETAS_S, sym)} className="curve" style={{ stroke: theme.arch }} />
        {THETAS_S.map((t, j) => (
          <circle key={t} cx={x(t)} cy={y(sym[j]!)} r={2.6} style={{ fill: theme.arch }} />
        ))}
        <line x1={x(theta)} x2={x(theta)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.2 }} />
        <circle cx={x(theta)} cy={y(backbone[k]!)} r={5} style={{ fill: theme.fail }} />
        {[0, 90, 180, 270, 360].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 16} textAnchor="middle" className="tick">{t}°</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">rotation about (1, 2, 3)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> backbone y<sub>0</sub></span>
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> ECSE y<sub>S</sub></span>
      </figcaption>
    </figure>
  );
}

function StretchChart({ theme, ds, curves, rin }: { theme: SceneTheme; ds: number[]; curves: number[][]; rin: number }) {
  const W = 520;
  const H = 220;
  const M = { l: 56, r: 16, t: 22, b: 40 };
  const all = curves.flat().filter(Number.isFinite);
  const pad = 0.1 * (Math.max(...all) - Math.min(...all) || 1);
  const lo = Math.min(...all) - pad;
  const hi = Math.max(...all) + pad;
  const d0 = ds[0]!;
  const d1 = ds.at(-1)!;
  const x = (d: number) => M.l + ((d - d0) / (d1 - d0)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const piece = (ys: number[], a: number, b: number) =>
    ds
      .map((d, k) => [d, ys[k]!] as const)
      .filter(([d, v]) => d >= a && d < b && Number.isFinite(v))
      .map(([d, v], k) => `${k ? 'L' : 'M'}${x(d).toFixed(1)},${y(v).toFixed(1)}`)
      .join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`ECSE prediction while one Zn–O bond is stretched through the frame cutoff r_in = ${rin.toFixed(2)} Å: continuous with smooth weights, a step with a plain average.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">ECSE prediction while one Zn–O bond stretches</text>
        <rect x={x(rin - RIN_WIDTH)} y={M.t} width={x(rin) - x(rin - RIN_WIDTH)} height={H - M.t - M.b} style={{ fill: theme.caveat, opacity: 0.12 }} />
        <line x1={x(rin)} x2={x(rin)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
        <path d={piece(curves[1]!, d0, rin)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 4' }} />
        <path d={piece(curves[1]!, rin, d1 + 1)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 4' }} />
        <path d={piece(curves[0]!, d0, d1 + 1)} className="curve" style={{ stroke: theme.arch }} />
        {[2, 2.5, 3, 3.5, 4].filter((d) => d >= d0 && d <= d1).map((d) => (
          <text key={d} x={x(d)} y={H - M.b + 16} textAnchor="middle" className="tick">{d} Å</text>
        ))}
        <text x={x(rin) + 4} y={M.t + 12} className="tick">r<tspan baselineShift="sub" fontSize="0.75em">in</tspan></text>
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">Zn–O bond length</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> smooth weights w<sub>jj′</sub></span>
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> plain average</span>
      </figcaption>
    </figure>
  );
}
