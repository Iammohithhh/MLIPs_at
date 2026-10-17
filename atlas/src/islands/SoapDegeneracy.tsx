import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Gap.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, linear, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { DEFAULT_SOAP, bispectrum, degeneratePair, normalise, pairAngles, powerSpectrum, type SoapOptions, type Vec3 } from '~/lib/models/soap';

// Pozdnyakov et al. (PRL 2020, Fig. 1a): two four-neighbour environments with the same distances and
// the same angles, hence the same SOAP power spectrum, that are not rotations or mirror images of each
// other. A four-body invariant (a truncated bispectrum) tells them apart. Computed live in
// src/lib/models/soap.ts.

const R0 = 2.0;
const CENTRES = [1.5, 2.0, 2.5];

// |p̂ − p̂′| = √(2 − 2 p̂·p̂′), computed from the difference so that identical vectors give rounding-level values, not √ε.
function distance(a: Float64Array, b: Float64Array): number {
  const x = normalise(a);
  const y = normalise(b);
  let s = 0;
  for (let k = 0; k < x.length; k++) s += (x[k]! - y[k]!) ** 2;
  return Math.sqrt(s);
}

export default function SoapDegeneracy() {
  const theme = useSceneTheme();
  const ids = useId();
  const [lift, setLift] = useState(0);
  const [lmax, setLmax] = useState(6);
  const o: SoapOptions = useMemo(() => ({ ...DEFAULT_SOAP, rc: 4, lmax }), [lmax]);
  const { plus, minus } = useMemo(() => degeneratePair(R0, lift), [lift]);
  const d2 = useMemo(() => distance(powerSpectrum(plus, o), powerSpectrum(minus, o)), [plus, minus, o]);
  const o3 = useMemo(() => ({ ...o, lmax: Math.min(lmax, 4) }), [o, lmax]);
  const d3 = useMemo(() => distance(bispectrum(plus, o3, CENTRES), bispectrum(minus, o3, CENTRES)), [plus, minus, o3]);
  const angP = pairAngles(plus);
  const angM = pairAngles(minus);
  const same = angP.every((a, k) => Math.abs(a - angM[k]!) < 1e-4); // acos loses ~10⁻⁶° next to 180°
  const fmt = (v: number) => (v < 1e-12 ? '< 10⁻¹²' : v.toExponential(2));

  return (
    <div className="lab">
      <div className="gap-pair">
        <div className="lab-stage lab-stage--short">
          <span className="gap-pair-label">X⁺ · neighbours at 0°, 45°, 135°, 180°</span>
          <Stage view={{ position: [3.2, -6.5, 6.5], target: [0, 0, 0] }} background={theme.canvas} label="Environment X plus: four neighbours on a circle at 0, 45, 135 and 180 degrees." minDistance={3} maxDistance={30}>
            <Scene theme={theme} env={plus} />
          </Stage>
        </div>
        <div className="lab-stage lab-stage--short">
          <span className="gap-pair-label">X⁻ · neighbours at 0°, 45°, 90°, 225°</span>
          <Stage view={{ position: [3.2, -6.5, 6.5], target: [0, 0, 0] }} background={theme.canvas} label="Environment X minus: four neighbours on a circle at 0, 45, 90 and 225 degrees." minDistance={3} maxDistance={30}>
            <Scene theme={theme} env={minus} />
          </Stage>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-z`}>
            <span>lift the 0° neighbour</span>
            <input id={`${ids}-z`} type="range" min={0} max={0.6} step={0.01} value={lift} onChange={(e) => setLift(Number(e.target.value))} />
            <output>{lift.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-l`}>
            <span>power spectrum l<sub>max</sub></span>
            <input id={`${ids}-l`} type="range" min={1} max={10} step={1} value={lmax} onChange={(e) => setLmax(Number(e.target.value))} />
            <output>{lmax}</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" onClick={() => setLift(0)} disabled={lift === 0}>
              back to the plane
            </button>
          </div>
        </div>
        <div className="gap-side" aria-live="polite">
          <Meter theme={theme} label="3-body: SOAP power spectrum" v={d2} color={theme.fail} text={fmt(d2)} />
          <Meter theme={theme} label="4-body: bispectrum" v={d3} color={theme.verified} text={fmt(d3)} />
          <p className="gap-angles">
            Angles between neighbours, X⁺: {angP.map((a) => a.toFixed(1)).join(', ')}°<br />
            X⁻: {angM.map((a) => a.toFixed(1)).join(', ')}° {same ? '— identical lists' : '— lists differ'}
          </p>
        </div>
        <p className="lab-note">
          Distances are √(2 − 2 p̂·p̂′) between the normalised feature vectors, on a log scale from 10⁻¹² to 1. In the plane every
          SOAP component agrees to rounding error, at every l<sub>max</sub>: the power spectrum is a function of the triangles
          (two distances and the angle between them), and the two environments have the same triangles. The bispectrum (three
          density coefficients coupled with Gaunt coefficients, l ≤ 4, three radial channels) sees tetrahedra, which differ. Lift
          the 0° neighbour and the angle lists separate, so the power spectrum starts to tell the two apart too.
        </p>
      </div>
    </div>
  );
}

function Meter({ theme, label, v, color, text }: { theme: SceneTheme; label: string; v: number; color: string; text: string }) {
  const t = v <= 1e-12 ? 0 : Math.max(0, Math.min(1, (Math.log10(v) + 12) / 12));
  return (
    <div className="gap-meter">
      <span>{label}</span>
      <span className="track" aria-hidden="true">
        <span className="fill" style={{ display: 'block', width: `${(100 * t).toFixed(1)}%`, background: color }} />
      </span>
      <output style={{ color: v <= 1e-12 ? theme.fail : theme.ink }}>{text}</output>
    </div>
  );
}

function Scene({ theme, env }: { theme: SceneTheme; env: Vec3[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 4, 30);
  const buffers = useMemo<AtomBuffers>(() => {
    const ghosts: Vec3[] = Array.from({ length: 8 }, (_, k) => [R0 * Math.cos((k * Math.PI) / 4), R0 * Math.sin((k * Math.PI) / 4), 0]);
    const n = 1 + env.length + ghosts.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    radii[0] = 0.32;
    colors.set(linear(theme.ink), 0);
    env.forEach((v, k) => {
      pos.set(v, 3 * (k + 1));
      radii[k + 1] = 0.3;
      colors.set(linear(k === 0 ? theme.caveat : theme.arch), 3 * (k + 1));
    });
    ghosts.forEach((v, k) => {
      const i = 1 + env.length + k;
      pos.set(v, 3 * i);
      radii[i] = 0.06;
      colors.set(linear(theme.inkMuted), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.4), ao: new Float32Array(n).fill(1), count: n };
  }, [env, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.arch);
    const g = linear(theme.inkMuted);
    const ring: ColoredLine[] = Array.from({ length: 64 }, (_, k) => {
      const a = (2 * Math.PI * k) / 64;
      const b = (2 * Math.PI * (k + 1)) / 64;
      return { a: [R0 * Math.cos(a), R0 * Math.sin(a), 0], b: [R0 * Math.cos(b), R0 * Math.sin(b), 0], rgba: [g[0], g[1], g[2], 0.35] };
    });
    return [...ring, ...env.map((v) => ({ a: [0, 0, 0] as Vec3, b: v, rgba: [c[0], c[1], c[2], 0.7] as [number, number, number, number] }))];
  }, [env, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
    </>
  );
}
