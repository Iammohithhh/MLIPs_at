import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Grace.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, ShadedSurface, linear, useShadingUniforms, type AtomBuffers, type ColoredLine, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { signalMesh } from '~/lib/scenes/sh-surface';
import { mulberry32, randomRotation } from '~/lib/chem/rng';
import { GRACE_RCUT, atomicBase, costs, explicitInvariant, productInvariant, rattle, rotate, type Invariant, type Vec3 } from '~/lib/models/grace';

// The ACE "density trick" on a real environment (src/lib/models/grace.ts): project the neighbours
// onto g_n(r) Y_lm(r̂) once (the atomic base A), then form invariants as products of A's. The same
// number is an explicit sum over ν-tuples of neighbours, and that sum includes tuples that repeat a
// neighbour. Those self-interaction terms are lower body-order functions; the bar shows them.

const NF = 4;
const LMAX_BASE = 6;
const PRESETS: Array<{ id: string; b: Invariant; label: string }> = [
  { id: 'a', b: { n: [1, 2], l: [1, 1] }, label: 'ν = 2 · n = (1, 2) · l = 1' },
  { id: 'b', b: { n: [2, 2], l: [2, 2] }, label: 'ν = 2 · n = (2, 2) · l = 2' },
  { id: 'c', b: { n: [1, 2, 3], l: [1, 1, 2] }, label: 'ν = 3 · n = (1, 2, 3) · l = (1, 1, 2)' },
  { id: 'd', b: { n: [2, 2, 2], l: [2, 2, 2] }, label: 'ν = 3 · n = (2, 2, 2) · l = (2, 2, 2)' },
  { id: 'e', b: { n: [1, 3, 4], l: [1, 2, 3] }, label: 'ν = 3 · n = (1, 3, 4) · l = (1, 2, 3)' },
];

export default function GraceDensity({ neighbours }: { neighbours: Vec3[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [rc, setRc] = useState(4.0);
  const [sigma, setSigma] = useState(0.08);
  const [scramble, setScramble] = useState(false);
  const [n, setN] = useState(1);
  const [L, setL] = useState(4);
  const [preset, setPreset] = useState('c');
  const b = PRESETS.find((p) => p.id === preset)!.b;
  const nu = b.n.length;

  const env = useMemo(() => {
    let vs = rattle(neighbours, sigma, 3);
    if (scramble) {
      const rand = mulberry32(17);
      vs = vs.map((v) => rotate([v], randomRotation(rand))[0]!);
    }
    return vs.filter((v) => Math.hypot(...v) < rc);
  }, [neighbours, sigma, scramble, rc]);
  const A = useMemo(() => atomicBase(env, NF, LMAX_BASE, rc), [env, rc]);
  const product = useMemo(() => productInvariant(A, b), [A, b]);
  const parts = useMemo(() => explicitInvariant(env, b, NF, rc), [env, b, rc]);
  const N = env.length;
  const cost = costs(N, nu, NF, LMAX_BASE);
  const coeffs = useMemo(() => A[n - 1]!.slice(0, (L + 1) ** 2), [A, n, L]);
  // Relative to the size of the terms that are summed, since they partly cancel.
  const scale = Math.abs(parts.distinct) + Math.abs(parts.partial) + Math.abs(parts.diagonal);
  const rel = Math.abs(product - parts.total) / Math.max(scale, 1e-300);

  const rows = [
    { key: 'distinct', label: `all ${nu} neighbours distinct: ${nu + 1}-body`, v: parts.distinct, c: theme.arch },
    ...(nu === 3 ? [{ key: 'partial', label: 'one neighbour repeated: 3-body', v: parts.partial, c: theme.caveat }] : []),
    { key: 'diagonal', label: `the same neighbour ${nu} times: 2-body`, v: parts.diagonal, c: theme.fail },
  ];
  const vmax = Math.max(...rows.map((r) => Math.abs(r.v)), Math.abs(product), 1e-30);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [13, -18, 10.5], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`A silicon atom with ${N} neighbours inside ${rc.toFixed(1)} Å. The surface around it is the neighbour density projected on radial channel ${n} and spherical harmonics up to degree ${L}.`}
            minDistance={4}
            maxDistance={60}
          >
            <Scene theme={theme} env={env} coeffs={coeffs} L={L} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Radial channel">
              {[1, 2, 3, 4].map((k) => (
                <button key={k} type="button" aria-pressed={n === k} onClick={() => setN(k)}>
                  n = {k}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.arch }} /> neighbours within r<sub>cut</sub>
            </li>
            <li>
              <span className="bar" style={{ background: theme.phys }} /> ρ<sub>n</sub>(r̂) &gt; 0
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> ρ<sub>n</sub>(r̂) &lt; 0
            </li>
          </ul>
        </div>
        <div className="grace-split">
          <h4>One invariant, two ways</h4>
          <div className="seg" role="group" aria-label="Invariant">
            {PRESETS.map((p) => (
              <button key={p.id} type="button" aria-pressed={preset === p.id} onClick={() => setPreset(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
          <table className="lab-table">
            <thead>
              <tr>
                <th scope="col">explicit sum over neighbour tuples</th>
                <th scope="col">value</th>
                <th scope="col" aria-label="signed bar" style={{ width: '38%' }}>
                  <span aria-hidden="true" />
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key}>
                  <td>
                    <span className="swatch" style={{ background: r.c }} />
                    {r.label}
                  </td>
                  <td>{r.v.toExponential(3)}</td>
                  <td>
                    <SignedBar v={r.v} vmax={vmax} color={r.c} theme={theme} />
                  </td>
                </tr>
              ))}
              <tr>
                <td>
                  <strong>sum of all {(N ** nu).toLocaleString('en')} tuples</strong>
                </td>
                <td>
                  <strong>{parts.total.toExponential(3)}</strong>
                </td>
                <td>
                  <SignedBar v={parts.total} vmax={vmax} color={theme.ink} theme={theme} />
                </td>
              </tr>
              <tr>
                <td>
                  <strong>product of atomic bases</strong> (Eq. 2)
                </td>
                <td>
                  <strong>{product.toExponential(3)}</strong>
                </td>
                <td>
                  <SignedBar v={product} vmax={vmax} color={theme.ink} theme={theme} />
                </td>
              </tr>
            </tbody>
          </table>
          <CostChart theme={theme} N={N} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-rc`}>
            <span>cutoff r<sub>cut</sub></span>
            <input id={`${ids}-rc`} type="range" min={2.4} max={GRACE_RCUT} step={0.05} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-s`}>
            <span>rattle σ</span>
            <input id={`${ids}-s`} type="range" min={0} max={0.25} step={0.01} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
            <output>{sigma.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-L`}>
            <span>degrees shown in ρ<sub>n</sub></span>
            <input id={`${ids}-L`} type="range" min={0} max={LMAX_BASE} step={1} value={L} onChange={(e) => setL(Number(e.target.value))} />
            <output>l ≤ {L}</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" aria-pressed={scramble} onClick={() => setScramble((s) => !s)}>
              {scramble ? 'scrambled: bond angles randomised' : 'scramble bond angles, keep lengths'}
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>neighbours N</dt>
            <dd>{N}</dd>
          </div>
          <div className={rel < 1e-12 ? 'is-good' : 'is-alert'}>
            <dt>product vs tuple sum (relative)</dt>
            {/* Rounding-level values differ between engines; bucket them so server and browser render the same text. */}
            <dd>{rel < 1e-12 ? '< 10⁻¹²' : rel.toExponential(1)}</dd>
          </div>
          <div>
            <dt>tuples summed (N^ν)</dt>
            <dd>{cost.explicit.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>atomic-base work (N × {NF} × {(LMAX_BASE + 1) ** 2})</dt>
            <dd>{cost.ace.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>self-interaction share</dt>
            <dd>{((100 * (Math.abs(parts.partial) + Math.abs(parts.diagonal))) / (Math.abs(parts.distinct) + Math.abs(parts.partial) + Math.abs(parts.diagonal) || 1)).toFixed(0)} %</dd>
          </div>
        </dl>
        <p className="lab-note">
          The product of atomic bases and the explicit sum agree to rounding error for every setting. Scramble the angles: each
          neighbour keeps its distance but moves to a random direction. The all-distinct term changes, while the term that uses the
          same neighbour {nu} times does not, because it depends on bond lengths only. A linear ACE absorbs such terms into its
          lower-order coefficients. GRACE keeps them on purpose (see text). The atomic-base work is one pass over the neighbours for
          all (n, l, m) at once; it is shared by every invariant, while the tuple sum is paid again for each one.
        </p>
      </div>
    </div>
  );
}

function SignedBar({ v, vmax, color, theme }: { v: number; vmax: number; color: string; theme: SceneTheme }) {
  const w = 100;
  const half = w / 2;
  // Rounded: server and browser maths can differ in the last bit, which would break hydration.
  const len = Math.round((Math.abs(v) / vmax) * half * 100) / 100;
  return (
    <svg viewBox={`0 0 ${w} 12`} width="100%" height="12" aria-hidden="true" preserveAspectRatio="none">
      <line x1={half} x2={half} y1={0} y2={12} style={{ stroke: theme.ruleStrong }} />
      <rect x={v >= 0 ? half : half - len} y={2} width={Math.max(0.5, len)} height={8} rx={1.5} style={{ fill: color }} />
    </svg>
  );
}

function CostChart({ theme, N }: { theme: SceneTheme; N: number }) {
  const W = 420;
  const H = 190;
  const M = { l: 46, r: 12, t: 14, b: 34 };
  const n0 = 2;
  const n1 = 128;
  const y0 = 1;
  const y1 = 1e8;
  const x = (v: number) => M.l + ((Math.log10(v) - Math.log10(n0)) / (Math.log10(n1) - Math.log10(n0))) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.log10(Math.min(Math.max(v, y0), y1)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - M.t - M.b);
  const Ns = Array.from({ length: 61 }, (_, k) => n0 * (n1 / n0) ** (k / 60));
  const path = (f: (n: number) => number) => Ns.map((v, k) => `${k ? 'L' : 'M'}${x(v).toFixed(1)},${y(f(v)).toFixed(1)}`).join('');
  const series = [
    { nu: 2, c: theme.arch },
    { nu: 3, c: theme.caveat },
    { nu: 4, c: theme.fail },
  ];
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Work per environment against the number of neighbours, log–log: explicit sums over ν-tuples grow as N to the power ν; the atomic base grows linearly.">
        {[1, 1e2, 1e4, 1e6, 1e8].map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end" className="tick">
              10{t === 1 ? '⁰' : t === 1e2 ? '²' : t === 1e4 ? '⁴' : t === 1e6 ? '⁶' : '⁸'}
            </text>
          </g>
        ))}
        {[2, 4, 8, 16, 32, 64, 128].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 15} textAnchor="middle" className="tick">
            {t}
          </text>
        ))}
        {series.map((s) => (
          <g key={s.nu}>
            <path d={path((v) => v ** s.nu)} className="curve" style={{ stroke: s.c }} />
            <text x={x(n1) - 2} y={y(n1 ** s.nu) + (s.nu === 4 ? 12 : -4)} textAnchor="end" className="tick" style={{ fill: s.c }}>
              N{s.nu === 2 ? '²' : s.nu === 3 ? '³' : '⁴'}
            </text>
          </g>
        ))}
        <path d={path((v) => v * NF * (LMAX_BASE + 1) ** 2)} className="curve" style={{ stroke: theme.phys, strokeDasharray: '6 3' }} />
        <text x={x(2.2)} y={y(2.2 * NF * 49) - 10} className="tick" style={{ fill: theme.phys }}>
          atomic base, all invariants
        </text>
        <line x1={x(Math.max(N, n0))} x2={x(Math.max(N, n0))} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeDasharray: '2 3' }} />
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          neighbours N
        </text>
      </svg>
    </figure>
  );
}

function surface(theme: SceneTheme, mesh: ReturnType<typeof signalMesh>): SurfaceData {
  const pos = linear(theme.phys);
  const neg = linear(theme.caveat);
  const colors = new Float32Array(mesh.signs.length * 3);
  mesh.signs.forEach((s, k) => colors.set(s > 0 ? pos : neg, 3 * k));
  return { positions: mesh.positions, colors, indices: mesh.indices };
}

function Scene({ theme, env, coeffs, L }: { theme: SceneTheme; env: Vec3[]; coeffs: Float64Array; L: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 12, 60);
  const allZero = coeffs.every((c) => Math.abs(c) < 1e-14);
  const surf = useMemo(() => (allZero ? null : surface(theme, signalMesh(coeffs, L, [0, 0, 0], 1.7, 48, 96))), [theme, coeffs, L, allZero]);
  const atoms = useMemo<AtomBuffers>(() => {
    const n = env.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    radii[0] = 0.3;
    colors.set(linear(theme.ink), 0);
    env.forEach((v, k) => {
      pos.set(v, 3 * (k + 1));
      radii[k + 1] = 0.36;
      colors.set(linear(theme.arch), 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.5), ao: new Float32Array(n).fill(1), count: n };
  }, [env, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.arch);
    return env.map((v) => ({ a: [0, 0, 0], b: v, rgba: [c[0], c[1], c[2], 0.25] }));
  }, [env, theme]);
  return (
    <>
      {surf ? <ShadedSurface data={surf} uniforms={uniforms} /> : null}
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
    </>
  );
}
