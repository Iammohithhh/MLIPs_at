import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import si from '~/data/structures/si-diamond.json';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { neighboursOf } from '~/lib/structure/geometry';
import type { Structure } from '~/lib/structure/types';

// A TOY MODEL of potential-energy-surface softening: near a minimum the energy along a
// displacement is ½kx²; a softened model has curvature s·k (s < 1), so its restoring
// forces and phonon frequencies (∝ √k) are too small. The scales 0.859 → 0.965 are the
// CHGNet example reported by Deng et al. for a one-scalar correction; the curvature k
// itself is illustrative.

const K = 10; // eV/Å² (illustrative)
const PUBLISHED = { before: 0.859, after: 0.965 };

export default function SofteningLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [x, setX] = useState(0.25);
  const [s, setS] = useState(PUBLISHED.before);
  const fRef = -K * x;
  const fModel = -s * K * x;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [6, -11, 6], target: [0, 0, 0] }} background={theme.canvas} label={`Silicon atom displaced by ${x.toFixed(2)} Å with restoring forces from the reference and from a model softened by a factor ${s.toFixed(3)}.`} minDistance={5} maxDistance={40}>
            <SoftScene theme={theme} x={x} s={s} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.verified }} /> reference restoring force
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> softened model
            </li>
          </ul>
        </div>
        <CurvePlot theme={theme} x={x} s={s} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-x`}>
            <span>displacement x</span>
            <input id={`${ids}-x`} type="range" min={-0.4} max={0.4} step={0.01} value={x} onChange={(e) => setX(Number(e.target.value))} />
            <output>{x.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-s`}>
            <span>softening scale s</span>
            <input id={`${ids}-s`} type="range" min={0.7} max={1} step={0.001} value={s} onChange={(e) => setS(Number(e.target.value))} />
            <output>{s.toFixed(3)}</output>
          </label>
          <div className="seg" role="group" aria-label="Published example">
            <button type="button" aria-pressed={s === PUBLISHED.before} onClick={() => setS(PUBLISHED.before)}>CHGNet example 0.859</button>
            <button type="button" aria-pressed={s === PUBLISHED.after} onClick={() => setS(PUBLISHED.after)}>after one-scalar fix 0.965</button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Force, reference</dt>
            <dd>{fRef.toFixed(2)} eV/Å</dd>
          </div>
          <div className="is-alert">
            <dt>Force, softened</dt>
            <dd>{fModel.toFixed(2)} eV/Å</dd>
          </div>
          <div className={Math.sqrt(s) < 0.98 ? 'is-alert' : 'is-good'}>
            <dt>Frequency ratio √s</dt>
            <dd>{Math.sqrt(s).toFixed(3)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          A harmonic frequency goes as the square root of the curvature, so a surface that is {((1 - s) * 100).toFixed(0)}%
          too soft gives frequencies {((1 - Math.sqrt(s)) * 100).toFixed(1)}% too low — and restoring forces too weak at every
          displacement. The two preset scales are the CHGNet example from the source; the curvature itself is a toy value.
        </p>
      </div>
    </div>
  );
}

function SoftScene({ theme, x, s }: { theme: SceneTheme; x: number; s: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 34);
  const S = si as unknown as Structure;
  const env = useMemo(() => neighboursOf(S, 0, 4.0).map((n) => [n.position[0] - S.positions[0]![0], n.position[1] - S.positions[0]![1], n.position[2] - S.positions[0]![2]]), [S]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = env.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const base = linear('#c8b88f');
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    pos.set([x, 0, 0], 0);
    radii[0] = 0.62;
    colors.set(base, 0);
    env.forEach((d, k) => {
      pos.set(d, 3 * (k + 1));
      const near = Math.hypot(d[0]!, d[1]!, d[2]!) < 2.5;
      radii[k + 1] = near ? 0.55 : 0.4;
      colors.set(near ? mixRGB(base, stone, 0.35) : stone, 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [env, x, theme]);
  const arrows = useMemo<Segment[]>(() => {
    if (Math.abs(x) < 0.01) return [];
    const scale = 1.15;
    const ref = -K * x * scale;
    const mod = -s * K * x * scale;
    return [
      { a: [x, 0, 0.4], b: [x + ref, 0, 0.4], color: linear(theme.verified), radius: 0.08 },
      { a: [x, 0, -0.4], b: [x + mod, 0, -0.4], color: linear(theme.caveat), radius: 0.08 },
    ];
  }, [x, s, theme.verified, theme.caveat]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.28} />}
    </>
  );
}

function CurvePlot({ theme, x, s }: { theme: SceneTheme; x: number; s: number }) {
  const W = 520;
  const H = 320;
  const M = { l: 52, r: 16, t: 22, b: 40 };
  const xs = Array.from({ length: 161 }, (_, k) => -0.4 + (k * 0.8) / 160);
  const emax = 0.5 * K * 0.16 * 1.08;
  const X = (v: number) => M.l + ((v + 0.4) / 0.8) * (W - M.l - M.r);
  const Y = (e: number) => M.t + (1 - e / emax) * (H - M.t - M.b);
  const path = (k: number) => xs.map((v, i) => `${i ? 'L' : 'M'}${X(v).toFixed(1)},${Y(0.5 * k * v * v).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy against displacement for the reference curvature and a model softened by ${s.toFixed(3)}.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">energy along the displacement [eV]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {[0, 0.4, 0.8].map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={Y(e)} y2={Y(e)} className="grid" />
            <text x={M.l - 6} y={Y(e) + 4} textAnchor="end" className="tick">{e.toFixed(1)}</text>
          </g>
        ))}
        <path d={path(K)} className="curve" style={{ stroke: theme.verified }} />
        <path d={path(s * K)} className="curve" style={{ stroke: theme.caveat, strokeDasharray: '6 3' }} />
        <line x1={X(x)} x2={X(x)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={X(x)} cy={Y(0.5 * K * x * x)} r={4.5} style={{ fill: theme.verified }} />
        <circle cx={X(x)} cy={Y(0.5 * s * K * x * x)} r={4.5} style={{ fill: theme.caveat }} />
        {[-0.4, -0.2, 0, 0.2, 0.4].map((v) => (
          <text key={v} x={X(v)} y={H - M.b + 16} textAnchor="middle" className="tick">{v.toFixed(1)}</text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">x [Å]</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.verified }} /> reference ½kx²</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> softened ½skx²</span>
      </figcaption>
    </figure>
  );
}
