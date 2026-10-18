import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, ShadedSurface, linear, useShadingUniforms, type AtomBuffers, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { signalMesh } from '~/lib/scenes/sh-surface';
import { applyBlocks, axisRotation, wignerBlocks, type Mat3, type Vec3 } from '~/lib/models/escn';
import {
  ACTIVATIONS,
  activate,
  activationError,
  eqv2DefaultGrid,
  makeGrid,
  randomCoefs,
  silu,
  swigluExactGrid,
  toGrid,
  type Activation,
} from '~/lib/models/equiformer';

// Activations on the sphere (src/lib/models/equiformer.ts): sample a degree-≤L feature on e3nn's
// latitude–longitude grid, apply a pointwise function, project back. SiLU on the grid
// (EquiformerV2) aliases, so the result depends on how the grid sits relative to the feature;
// SwiGLU-S² (EquiformerV3) is a product of two band-limited functions and becomes exact on a
// large enough grid. The gate is exact on any grid because it never uses one.

const KINDS: Activation[] = ['gate', 's2', 'sep-s2', 'swiglu-s2'];
const SPIN_AXIS: Vec3 = [0.4, 1, 0.3];
const TEST_R: Mat3 = axisRotation([0.3, -0.7, 0.65], 2.1);
const NS = Array.from({ length: 19 }, (_, k) => 4 + 2 * k); // 4…40
const SCALAR = 0.3;
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const toSup = (s: string) => s.replace('+', '').split('').map((c) => SUP[c] ?? c).join('');
function fmt(v: number): string {
  if (v < 1e-12) return '< 10⁻¹²';
  const [m, e] = v.toExponential(1).split('e');
  return `${m} × 10${toSup(e!)}`;
}

export default function EquiformerS2() {
  const theme = useSceneTheme();
  const ids = useId();
  const [kind, setKind] = useState<Activation>('s2');
  const [L, setL] = useState(4);
  const [nBeta, setNBeta] = useState(10);
  const [nAlpha, setNAlpha] = useState(11);
  const [amp, setAmp] = useState(1.5);
  const [spin, setSpin] = useState(0);

  const x1 = useMemo(() => randomCoefs(L, 21, amp), [L, amp]);
  const x2 = useMemo(() => randomCoefs(L, 22, amp), [L, amp]);
  const grid = useMemo(() => makeGrid(L, nBeta, nAlpha), [L, nBeta, nAlpha]);
  const R = useMemo(() => axisRotation(SPIN_AXIS, (spin * Math.PI) / 180), [spin]);
  const D = useMemo(() => wignerBlocks(L, R), [L, R]);
  const xin = useMemo(() => applyBlocks(D, x1, L), [D, x1, L]);
  const xin2 = useMemo(() => applyBlocks(D, x2, L), [D, x2, L]);
  // Output computed from the rotated input; and the unrotated output, rotated afterwards.
  const out = useMemo(() => activate(kind, grid, xin, xin2, SCALAR), [kind, grid, xin, xin2]);
  const ref = useMemo(() => applyBlocks(D, activate(kind, grid, x1, x2, SCALAR), L), [kind, grid, x1, x2, D, L]);
  const err = useMemo(() => activationError(kind, grid, x1, x2, SCALAR, TEST_R), [kind, grid, x1, x2]);
  const samples = useMemo(() => {
    const a = toGrid(grid, xin);
    if (kind === 'swiglu-s2') {
      const b = toGrid(grid, xin2);
      return a.map((v, p) => v * b[p]!);
    }
    return kind === 'gate' ? null : a.map(silu);
  }, [grid, xin, xin2, kind]);
  const curves = useMemo(
    () =>
      Object.fromEntries(
        KINDS.map((k) => [k, NS.map((n) => activationError(k, makeGrid(L, n, n), x1, x2, SCALAR, TEST_R))]),
      ) as Record<Activation, number[]>,
    [L, x1, x2],
  );
  const v2 = eqv2DefaultGrid(L);
  const exact = swigluExactGrid(L);
  const drift = useMemo(() => {
    let num = 0;
    let den = 0;
    out.forEach((v, c) => {
      num += (v - ref[c]!) ** 2;
      den += ref[c]! ** 2;
    });
    return Math.sqrt(num / (den || 1));
  }, [out, ref]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [0.5, -16.5, 5.6], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Left: a degree-${L} feature drawn as a polar plot, with the ${nBeta} × ${nAlpha} sample points of the grid on a sphere around it. Right: the output of the ${ACTIVATIONS[kind].name}. The feature is turned by ${spin}° about a fixed axis while the grid stays put.`}
            minDistance={5}
            maxDistance={30}
          >
            <Scene theme={theme} L={L} xin={xin} out={out} dirs={kind === 'gate' ? [] : grid.dirs} samples={samples} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Activation">
              {KINDS.map((k) => (
                <button key={k} type="button" aria-pressed={kind === k} onClick={() => setKind(k)}>
                  {ACTIVATIONS[k].name}
                </button>
              ))}
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
              <span className="dot" style={{ background: theme.arch }} /> grid point, size ∝ |value there|
            </li>
          </ul>
        </div>
        <ErrorPlot theme={theme} L={L} curves={curves} kind={kind} v2={v2.nBeta} exact={exact.nBeta} n={nBeta === nAlpha ? nBeta : -1} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Maximum degree L">
            <span className="seg-label">L</span>
            {[2, 3, 4, 6].map((k) => (
              <button key={k} type="button" aria-pressed={L === k} onClick={() => setL(k)}>
                {k}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Grid presets">
            <span className="seg-label">grid</span>
            <button type="button" aria-pressed={nBeta === v2.nBeta && nAlpha === v2.nAlpha} onClick={() => (setNBeta(v2.nBeta), setNAlpha(v2.nAlpha))}>
              V2 default {v2.nBeta} × {v2.nAlpha}
            </button>
            <button type="button" aria-pressed={nBeta === 18 && nAlpha === 18} onClick={() => (setNBeta(18), setNAlpha(18))}>
              V2 on OC20 18 × 18
            </button>
            <button type="button" aria-pressed={nBeta === exact.nBeta && nAlpha === exact.nAlpha} onClick={() => (setNBeta(exact.nBeta), setNAlpha(exact.nAlpha))}>
              SwiGLU exact {exact.nBeta} × {exact.nAlpha}
            </button>
          </div>
          <label htmlFor={`${ids}-b`}>
            <span>latitudes N<sub>β</sub> (even)</span>
            <input id={`${ids}-b`} type="range" min={4} max={40} step={2} value={nBeta} onChange={(e) => setNBeta(Number(e.target.value))} />
            <output>{nBeta}</output>
          </label>
          <label htmlFor={`${ids}-a`}>
            <span>longitudes N<sub>α</sub></span>
            <input id={`${ids}-a`} type="range" min={3} max={40} step={1} value={nAlpha} onChange={(e) => setNAlpha(Number(e.target.value))} />
            <output>{nAlpha}</output>
          </label>
          <label htmlFor={`${ids}-m`}>
            <span>feature amplitude</span>
            <input id={`${ids}-m`} type="range" min={0.1} max={4} step={0.1} value={amp} onChange={(e) => setAmp(Number(e.target.value))} />
            <output>{amp.toFixed(1)}</output>
          </label>
          <label htmlFor={`${ids}-s`}>
            <span>rotate the feature</span>
            <input id={`${ids}-s`} type="range" min={0} max={360} step={5} value={spin} onChange={(e) => setSpin(Number(e.target.value))} />
            <output>{spin}°</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={err < 1e-9 ? 'is-good' : 'is-alert'}>
            <dt>equivariance error (one fixed rotation)</dt>
            <dd>{fmt(err)}</dd>
          </div>
          <div className={drift < 1e-9 ? 'is-good' : 'is-alert'}>
            <dt>output at this angle vs rotated output at 0°</dt>
            <dd>{fmt(drift)}</dd>
          </div>
          <div>
            <dt>grid points</dt>
            <dd>{kind === 'gate' ? 'none used' : (nBeta * nAlpha).toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>SwiGLU-S² exact from</dt>
            <dd>
              N<sub>β</sub> ≥ {3 * L + 1}, N<sub>α</sub> ≥ {3 * L + 1}
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Turn the feature with the grid fixed. With the gate the output turns rigidly with it. SiLU on a coarse grid gives an
          output whose shape changes with the angle, because the grid samples a different part of a function that has
          more detail than degree {L} can hold. Refine the grid and the error falls but never reaches zero. Raise the amplitude
          and SiLU gets more nonlinear, so the error grows. SwiGLU-S² is exact, to rounding, once both resolutions reach 3L + 1.
          The errors depend on the random input. They show trends, not the size of the errors in a trained model.
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

const LEFT: Vec3 = [-2.05, 0, 0];
const RIGHT: Vec3 = [2.05, 0, 0];
const SHELL = 1.6;

function Scene({ theme, L, xin, out, dirs, samples }: { theme: SceneTheme; L: number; xin: Float64Array; out: Float64Array; dirs: Vec3[]; samples: Float64Array | null }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 30);
  const a = useMemo(() => surface(theme, signalMesh(xin, L, LEFT, 1.2)), [theme, xin, L]);
  const b = useMemo(() => (out.every((v) => v === 0) ? null : surface(theme, signalMesh(out, L, RIGHT, 1.2))), [theme, out, L]);
  const points = useMemo<AtomBuffers>(() => {
    const n = dirs.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const mx = samples ? Math.max(...samples.map(Math.abs)) || 1 : 1;
    const c = linear(theme.arch);
    dirs.forEach((u, p) => {
      positions.set([LEFT[0] + SHELL * u[0], LEFT[1] + SHELL * u[1], LEFT[2] + SHELL * u[2]], 3 * p);
      radii[p] = 0.018 + 0.06 * Math.sqrt(Math.abs(samples?.[p] ?? 0) / mx);
      colors.set(c, 3 * p);
    });
    return { positions, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [dirs, samples, theme]);
  return (
    <>
      <ShadedSurface data={a} uniforms={uniforms} />
      {b ? <ShadedSurface data={b} uniforms={uniforms} /> : null}
      {points.count ? <ImpostorAtoms buffers={points} version={points.count} uniforms={uniforms} /> : null}
    </>
  );
}

function ErrorPlot({ theme, L, curves, kind, v2, exact, n }: { theme: SceneTheme; L: number; curves: Record<Activation, number[]>; kind: Activation; v2: number; exact: number; n: number }) {
  const W = 520;
  const H = 360;
  const x0 = 58;
  const y0 = 30;
  const w = W - x0 - 18;
  const h = H - y0 - 70;
  const lo = -16;
  const hi = 0;
  const px = (k: number) => x0 + ((k - 4) / 36) * w;
  const py = (v: number) => y0 + ((hi - Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-16))))) / (hi - lo)) * h;
  const color: Record<Activation, string> = { gate: theme.verified, s2: theme.fail, 'sep-s2': theme.caveat, 'swiglu-s2': theme.arch };
  const path = (vals: number[]) => vals.map((v, k) => `${k ? 'L' : 'M'}${px(NS[k]!).toFixed(1)},${py(v).toFixed(1)}`).join('');
  const marks: [number, string][] = [
    [v2, 'V2 default'],
    [18, '18'],
    [exact, 'SwiGLU exact'],
  ];
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Equivariance error against grid resolution N (an N × N grid) for L = ${L}: the gate is exact at every resolution, SiLU on the grid falls slowly, SwiGLU-S² drops to rounding error at N = ${exact}.`}>
        <text x={x0} y={y0 - 12} className="axis-label">equivariance error (log), N × N grid, L = {L}</text>
        {[0, -4, -8, -12, -16].map((e) => (
          <g key={e}>
            <line x1={x0} x2={x0 + w} y1={py(10 ** e)} y2={py(10 ** e)} className="grid" />
            <text x={x0 - 6} y={py(10 ** e) + 4} textAnchor="end" className="tick">
              {e === 0 ? '1' : `10${toSup(String(e))}`}
            </text>
          </g>
        ))}
        {marks.map(([m, t], k) => (
          <g key={t}>
            <line x1={px(m)} x2={px(m)} y1={y0} y2={y0 + h} className="rc-line" />
            <text x={px(m) + 3} y={y0 + 10 + 12 * k} className="tick">{t}</text>
          </g>
        ))}
        {KINDS.map((k) => (
          <path key={k} d={path(curves[k])} className={`curve${k === kind ? ' is-active' : ''}`} style={{ stroke: color[k], opacity: k === kind ? 1 : 0.55 }} />
        ))}
        {n >= 4 && n <= 40 ? <circle cx={px(n)} cy={py(curves[kind][(n - 4) / 2] ?? 1)} r={4.5} style={{ fill: color[kind] }} /> : null}
        {NS.filter((k) => k % 8 === 0 || k === 4).map((k) => (
          <text key={k} x={px(k)} y={y0 + h + 16} textAnchor="middle" className="tick">{k}</text>
        ))}
        <text x={x0 + w} y={y0 + h + 32} textAnchor="end" className="tick">N (= N_β = N_α)</text>
      </svg>
      <figcaption className="lab-legend">
        {KINDS.map((k) => (
          <span key={k} className="item">
            <span className="swatch" style={{ background: color[k] }} /> {ACTIVATIONS[k].name} ({ACTIVATIONS[k].model})
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
