import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, ShadedSurface, Sticks, linear, useShadingUniforms, type AtomBuffers, type Segment, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { signalMesh } from '~/lib/scenes/sh-surface';
import {
  ESEN_RCUT,
  applyBlocks,
  axisRotation,
  edgeFrame,
  edgeMessage,
  equivarianceError,
  idx,
  makeToyEscn,
  matVec,
  randomFeature,
  rollDependence,
  so2Terms,
  tensorProductTerms,
  transpose,
  wignerBlocks,
  type EdgeMode,
  type Mat3,
  type Vec3,
} from '~/lib/models/escn';

// eSEN's edge message as an exact toy (src/lib/models/escn.ts): rotate the source and target
// features into the bond frame, apply SO(2) convolutions (eSCN Eq. 14) with a gate between them,
// rotate back, multiply by the envelope. Turning the frame about the bond (the roll fairchem draws at
// random) leaves the message unchanged; an unrestricted linear map in the same frame does not.

// Roughly perpendicular to the default view, so the bond is not foreshortened.
const BASE_DIR: Vec3 = [0.82, 0.55, 0.15];
const SPIN_AXIS: Vec3 = [1, 1, 1];
const TEST_R: Mat3 = axisRotation([0.3, -0.7, 0.65], 2.1);

export default function EsenEdge() {
  const theme = useSceneTheme();
  const ids = useId();
  const [L, setL] = useState(3);
  const [mmax, setMmax] = useState(2);
  const [mode, setMode] = useState<EdgeMode>('so2');
  const [roll, setRoll] = useState(0);
  const [spin, setSpin] = useState(0);
  const [r, setR] = useState(2.6);
  const M = Math.min(mmax, L);

  const model = useMemo(() => makeToyEscn({ L }), [L]);
  const base = useMemo(() => ({ xs: randomFeature(L, 41), xt: randomFeature(L, 42) }), [L]);
  const R = useMemo(() => axisRotation(SPIN_AXIS, (spin * Math.PI) / 180), [spin]);
  const D = useMemo(() => wignerBlocks(L, R), [L, R]);
  const xs = useMemo(() => applyBlocks(D, base.xs, L), [D, base, L]);
  const xt = useMemo(() => applyBlocks(D, base.xt, L), [D, base, L]);
  const dir = useMemo(() => matVec(R, BASE_DIR), [R]);
  const rvec = useMemo(() => dir.map((c) => (c * r) / Math.hypot(...BASE_DIR)) as Vec3, [dir, r]);
  const opts = { mmax: M, gamma: (roll * Math.PI) / 180, mode };
  const res = useMemo(() => edgeMessage(model, xs, xt, rvec, opts), [model, xs, xt, rvec, M, roll, mode]);
  const rollDep = useMemo(() => rollDependence(model, xs, xt, rvec, opts), [model, xs, xt, rvec, M, mode]);
  const eqErr = useMemo(() => equivarianceError(model, xs, xt, rvec, TEST_R, opts), [model, xs, xt, rvec, M, roll, mode]);
  const frame = useMemo(() => edgeFrame(rvec, opts.gamma), [rvec, roll]);
  const tp = tensorProductTerms(L);
  const s2 = so2Terms(L, M);
  const tiny = (v: number) => v < 1e-9;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [5.6, -7.8, 4.4], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Two atoms ${r.toFixed(1)} Å apart. A polar plot of the source atom's spherical-harmonic feature (degree ≤ ${L}) sits on the source; the message it sends, after the ${mode === 'so2' ? 'SO(2)' : 'dense'} convolution with orders up to ${M}, is drawn on the target. Small axes show the bond frame, rolled by ${roll}°.`}
            minDistance={4}
            maxDistance={30}
          >
            <Scene theme={theme} L={L} xs={xs} msg={res.message} rvec={rvec} frame={frame} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Edge-frame layer">
              <button type="button" aria-pressed={mode === 'so2'} onClick={() => setMode('so2')}>
                SO(2) convolution
              </button>
              <button type="button" aria-pressed={mode === 'dense'} onClick={() => setMode('dense')}>
                control: dense linear map
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.phys }} /> f &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> f &lt; 0
            </li>
            <li>
              <span className="bar" style={{ background: theme.arch }} /> bond (polar axis of the frame)
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.fail }} /> frame axes after the roll
            </li>
          </ul>
        </div>
        <Ladders theme={theme} L={L} M={M} input={res.edgeIn[0]} output={res.edgeOut} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Maximum degree L">
            <span className="seg-label">L</span>
            {[1, 2, 3, 4].map((k) => (
              <button key={k} type="button" aria-pressed={L === k} onClick={() => setL(k)}>
                {k}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Maximum order M">
            <span className="seg-label">M<sub>max</sub></span>
            {Array.from({ length: L + 1 }, (_, k) => k).map((k) => (
              <button key={k} type="button" aria-pressed={M === k} onClick={() => setMmax(k)}>
                {k}
              </button>
            ))}
          </div>
          <label htmlFor={`${ids}-g`}>
            <span>roll of the bond frame γ</span>
            <input id={`${ids}-g`} type="range" min={0} max={360} step={5} value={roll} onChange={(e) => setRoll(Number(e.target.value))} />
            <output>{roll}°</output>
          </label>
          <label htmlFor={`${ids}-s`}>
            <span>rotate the whole pair</span>
            <input id={`${ids}-s`} type="range" min={0} max={360} step={5} value={spin} onChange={(e) => setSpin(Number(e.target.value))} />
            <output>{spin}°</output>
          </label>
          <label htmlFor={`${ids}-r`}>
            <span>bond length</span>
            <input id={`${ids}-r`} type="range" min={1.2} max={6.4} step={0.05} value={r} onChange={(e) => setR(Number(e.target.value))} />
            <output>{r.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={tiny(rollDep) ? 'is-good' : 'is-alert'}>
            <dt>change when the roll is redrawn</dt>
            <dd>{tiny(rollDep) ? '< 10⁻⁹' : rollDep.toFixed(2)}</dd>
          </div>
          <div className={tiny(eqErr) ? 'is-good' : 'is-alert'}>
            <dt>equivariance error</dt>
            <dd>{tiny(eqErr) ? '< 10⁻⁹' : eqErr.toFixed(2)}</dd>
          </div>
          <div>
            <dt>envelope at {r.toFixed(2)} Å</dt>
            <dd>{res.envelope.toFixed(3)}</dd>
          </div>
          <div>
            <dt>terms: SO(3) tensor product</dt>
            <dd>{tp.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>terms: rotate + SO(2) + rotate back</dt>
            <dd>{s2.total.toLocaleString('en')}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Roll the frame about the bond: in SO(2) mode the coefficients in the ladder move between m and −m, but the message on the
          target does not change, because each (m, −m) block of the convolution commutes with a rotation about the bond. Switch to the
          dense control and the same roll changes the message. Its output then depends on an arbitrary choice of frame, so it is not
          equivariant. With M<sub>max</sub> = 0 the control is again exact, because only roll-invariant m = 0 coefficients survive.
          Term counts are multiply–adds per channel pair for one edge (definitions in <code>escn.ts</code>), not timings. Beyond{' '}
          {ESEN_RCUT} Å the envelope removes the edge.
        </p>
      </div>
    </div>
  );
}

function surface(theme: SceneTheme, mesh: ReturnType<typeof signalMesh>): SurfaceData {
  const pos = linear(theme.phys);
  const neg = linear(theme.caveat);
  const colors = new Float32Array(mesh.signs.length * 3);
  mesh.signs.forEach((s, k) => colors.set(s > 0 ? pos : neg, 3 * k));
  return { positions: mesh.positions, colors, indices: mesh.indices };
}

function Scene({ theme, L, xs, msg, rvec, frame }: { theme: SceneTheme; L: number; xs: Float64Array; msg: Float64Array; rvec: Vec3; frame: Mat3 }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 30);
  const s: Vec3 = rvec.map((c) => -c / 2) as Vec3;
  const t: Vec3 = rvec.map((c) => c / 2) as Vec3;
  const src = useMemo(() => surface(theme, signalMesh(xs, L, s, 0.85)), [theme, xs, L, s[0], s[1], s[2]]);
  const allZero = msg.every((v) => v === 0);
  const dst = useMemo(() => (allZero ? null : surface(theme, signalMesh(msg, L, t, 0.85))), [theme, msg, L, t[0], t[1], t[2], allZero]);
  const atoms = useMemo<AtomBuffers>(
    () => ({
      positions: new Float32Array([...s, ...t]),
      radii: new Float32Array([0.2, 0.2]),
      colors: new Float32Array([...linear(theme.ink), ...linear(theme.ink)]),
      metal: new Float32Array(2),
      ao: new Float32Array([1, 1]),
      count: 2,
    }),
    [theme, s[0], s[1], s[2]],
  );
  const bond = useMemo<Segment[]>(() => [{ a: s, b: t, color: linear(theme.arch), radius: 0.035 }], [theme, s[0], s[1], s[2]]);
  // Frame axes (rows of Q are the frame's x, y, z in global coordinates); z is the bond.
  const axes = useMemo<Segment[]>(() => {
    const Qt = transpose(frame);
    const e1: Vec3 = [Qt[0]!, Qt[3]!, Qt[6]!];
    const e2: Vec3 = [Qt[1]!, Qt[4]!, Qt[7]!];
    const c = linear(theme.fail);
    const o: Vec3 = [0, 0, 0];
    return [
      { a: o, b: e1.map((v) => 0.9 * v) as Vec3, color: c, radius: 0.025 },
      { a: o, b: e2.map((v) => 0.9 * v) as Vec3, color: c, radius: 0.025 },
    ];
  }, [frame, theme]);
  return (
    <>
      <ShadedSurface data={src} uniforms={uniforms} />
      {dst ? <ShadedSurface data={dst} uniforms={uniforms} /> : null}
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Sticks segments={bond} uniforms={uniforms} />
      <Arrows arrows={axes} uniforms={uniforms} head={0.16} />
    </>
  );
}

function Ladders({ theme, L, M, input, output }: { theme: SceneTheme; L: number; M: number; input: Float64Array; output: Float64Array }) {
  const W = 520;
  const H = 360;
  const cell = Math.min(26, 200 / (2 * L + 1));
  const panel = (2 * L + 1) * cell;
  const x0 = [52, 52 + panel + 56];
  const y0 = 44;
  const maxIn = Math.max(...input.map(Math.abs)) || 1;
  const maxOut = Math.max(...output.map(Math.abs)) || 1;
  const color = (v: number) => (v >= 0 ? theme.phys : theme.caveat);
  const opacity = (v: number, mx: number) => Math.min(1, 0.12 + 0.88 * Math.sqrt(Math.abs(v) / mx));
  const ladder = (x: Float64Array, mx: number, ox: number, title: string) => (
    <g>
      <text x={ox} y={y0 - 14} className="axis-label">{title}</text>
      {Array.from({ length: L + 1 }, (_, l) => (
        <g key={l}>
          <text x={ox - 8} y={y0 + l * cell + cell * 0.66} textAnchor="end" className="tick">l={l}</text>
          {Array.from({ length: 2 * L + 1 }, (_, k) => {
            const m = k - L;
            if (Math.abs(m) > l) return null;
            const dropped = Math.abs(m) > M;
            const v = x[idx(l, m)]!;
            return (
              <rect
                key={k}
                x={ox + k * cell + 1}
                y={y0 + l * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={2}
                style={dropped ? { fill: 'none', stroke: theme.ruleStrong, strokeDasharray: '2 2' } : { fill: color(v), fillOpacity: opacity(v, mx) }}
              >
                <title>{`l = ${l}, m = ${m}: ${dropped ? 'dropped (|m| > M_max)' : v.toFixed(3)}`}</title>
              </rect>
            );
          })}
        </g>
      ))}
      {Array.from({ length: 2 * L + 1 }, (_, k) => (
        <text key={k} x={ox + k * cell + cell / 2} y={y0 + (L + 1) * cell + 14} textAnchor="middle" className="tick">
          {k - L}
        </text>
      ))}
      <text x={ox + panel / 2} y={y0 + (L + 1) * cell + 30} textAnchor="middle" className="tick">m (about the bond)</text>
    </g>
  );
  // Term counts against L, log scale.
  const Ls = [1, 2, 3, 4, 5, 6, 8];
  const cy0 = y0 + (L + 1) * cell + 64;
  const ch = H - cy0 - 34;
  const cx0 = 60;
  const cw = W - cx0 - 20;
  const maxT = Math.log10(tensorProductTerms(8));
  const px = (l: number) => cx0 + ((l - 1) / 7) * cw;
  const py = (v: number) => cy0 + ch - (Math.log10(v) / maxT) * ch;
  const line = (f: (l: number) => number) => Ls.map((l, k) => `${k ? 'L' : 'M'}${px(l).toFixed(1)},${py(f(l)).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Feature coefficients in the bond frame, by degree l and order m, before and after the convolution; and the number of terms of a full tensor product versus the SO(2) route as L grows.">
        {ladder(input, maxIn, x0[0]!, 'source feature, bond frame')}
        {ladder(output, maxOut, x0[1]!, 'message, bond frame')}
        {ch > 40 ? (
          <g>
            <text x={cx0} y={cy0 - 8} className="axis-label">multiply–adds per channel pair (log)</text>
            {[1, 2, 3, 4, 5].map((e) => (e <= maxT ? <g key={e}><line x1={cx0} x2={cx0 + cw} y1={py(10 ** e)} y2={py(10 ** e)} className="grid" /><text x={cx0 - 6} y={py(10 ** e) + 4} textAnchor="end" className="tick">10{['⁰', '¹', '²', '³', '⁴', '⁵'][e]}</text></g> : null))}
            <path d={line(tensorProductTerms)} className="curve" style={{ stroke: theme.fail }} />
            <path d={line((l) => so2Terms(l, Math.min(2, l)).total)} className="curve" style={{ stroke: theme.arch }} />
            <circle cx={px(L)} cy={py(tensorProductTerms(L))} r={4} style={{ fill: theme.fail }} />
            <circle cx={px(L)} cy={py(so2Terms(L, Math.min(2, L)).total)} r={4} style={{ fill: theme.arch }} />
            {Ls.map((l) => (
              <text key={l} x={px(l)} y={cy0 + ch + 14} textAnchor="middle" className="tick">{l}</text>
            ))}
            <text x={cx0 + cw} y={cy0 + ch + 28} textAnchor="end" className="tick">L</text>
          </g>
        ) : null}
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> full SO(3) tensor product, O(L⁶)</span>
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> rotate + SO(2), M<sub>max</sub> = 2, O(L³)</span>
        <span className="item">dashed cells: dropped orders</span>
      </figcaption>
    </figure>
  );
}
