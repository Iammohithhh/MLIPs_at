import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './LongRange.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, ShadedSurface, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { mulberry32, randomRotation } from '~/lib/chem/rng';
import si from '~/data/structures/si-diamond.json';
import zno from '~/data/structures/zno-wurtzite.json';
import cu from '~/data/structures/cu-fcc.json';
import quartz from '~/data/structures/sio2-alpha-quartz.json';
import type { Structure } from '~/lib/structure/types';
import {
  b2,
  b2Direct,
  b3,
  cartesianA,
  environment,
  momentSurface,
  monomials,
  neighboursOf,
  powerToLegendre,
  rotateAll,
  sphericalPower,
  type Neighbour,
  type Vec3,
} from '~/lib/models/long-range';

// CACE's Cartesian angular basis on real environments (src/lib/models/long-range.ts): the moments
// A_l = Σ_j R(r_j) x^lx y^ly z^lz of the neighbour unit vectors (Cheng 2024, Eqs. 2, 6), their
// contraction into invariants with the multinomial weights of Eq. (4), and the Legendre mix that ties
// each Cartesian invariant to the spherical-harmonic ones.

const ENVS: Array<{ id: string; name: string; s: unknown; site: number }> = [
  { id: 'si', name: 'Si (diamond)', s: si, site: 0 },
  { id: 'zno', name: 'Zn in ZnO', s: zno, site: 0 },
  { id: 'quartz', name: 'Si in α-quartz', s: quartz, site: 0 },
  { id: 'cu', name: 'Cu (fcc)', s: cu, site: 0 },
];
const LMAX = 4;
const SUP = ['⁰', '¹', '²', '³', '⁴'];

export default function LrCace() {
  const theme = useSceneTheme();
  const ids = useId();
  const [envId, setEnvId] = useState('zno');
  const [rc, setRc] = useState(3);
  const [l, setL] = useState(3);
  const [seed, setSeed] = useState(0);
  const e = ENVS.find((x) => x.id === envId)!;
  const { vectors } = useMemo(() => environment(e.s as Structure, e.site, rc), [e, rc]);
  const rotated = useMemo(() => (seed === 0 ? vectors : rotateAll(vectors, randomRotation(mulberry32(seed)))), [vectors, seed]);
  const env = useMemo(() => neighboursOf(rotated, rc), [rotated, rc]);
  const base = useMemo(() => neighboursOf(vectors, rc), [vectors, rc]);
  const rows = useMemo(
    () =>
      Array.from({ length: LMAX + 1 }, (_, k) => {
        const c = powerToLegendre(k);
        return {
          l: k,
          cart: monomials(k).length,
          sph: 2 * k + 1,
          contracted: b2(env, k),
          direct: b2Direct(env, k),
          mix: c.map((ci, lp) => ({ lp, ci, p: ci ? sphericalPower(env, lp) : 0 })).filter((x) => x.ci),
        };
      }),
    [env],
  );
  const A = useMemo(() => cartesianA(env, l), [env, l]);
  const A0 = useMemo(() => cartesianA(base, l), [base, l]);
  const aChange = Math.max(...A.map((v, i) => Math.abs(v - A0[i]!)));
  const bChange = Math.max(
    ...Array.from({ length: LMAX + 1 }, (_, k) => Math.abs(b2(env, k) - b2(base, k)) / Math.max(1e-12, Math.abs(b2(base, k)))),
    Math.abs(b3(env, 1, 2) - b3(base, 1, 2)) / Math.max(1e-12, Math.abs(b3(base, 1, 2))),
  );
  const view = useMemo(() => ({ position: [rc * 2.1, -rc * 2.4, rc * 1.6] as [number, number, number], target: [0, 0, 0] as [number, number, number] }), [rc]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={Math.round(rc * 10)}
            background={theme.canvas}
            label={`${e.name}: ${vectors.length} neighbours within ${rc} Å and the surface f(n) = Σ R(r) (n · r̂)^${l}, drawn at radius |f|, red where positive and blue where negative.`}
            minDistance={3}
            maxDistance={40}
          >
            <Scene theme={theme} vectors={rotated} env={env} rc={rc} l={l} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Environment">
              {ENVS.map((x) => (
                <button key={x.id} type="button" aria-pressed={envId === x.id} onClick={() => setEnvId(x.id)}>
                  {x.name}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.fail }} /> f &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> f &lt; 0
            </li>
            <li>neighbours shaded by R(r)</li>
          </ul>
        </div>
        <div className="lr-side">
          <h4 className="lr-h">
            A<sub>l</sub> for l = {l}: {monomials(l).length} Cartesian components
          </h4>
          <div className="lr-mono" aria-label={`Cartesian moments of order ${l}`}>
            {monomials(l).map((m, i) => (
              <span key={m.join('')}>
                x{SUP[m[0]]}y{SUP[m[1]]}z{SUP[m[2]]} {fmt(A[i]!)}
              </span>
            ))}
          </div>
          <table className="lr-table">
            <thead>
              <tr>
                <th>l</th>
                <th>Cart.</th>
                <th>2l+1</th>
                <th>Σ C(l) A²</th>
                <th>Σ cos^l θ</th>
                <th>= Legendre mix</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.l} style={r.l === l ? { background: 'color-mix(in oklab, var(--atlas-arch) 8%, transparent)' } : undefined}>
                  <td>{r.l}</td>
                  <td>{r.cart}</td>
                  <td>{r.sph}</td>
                  <td>{fmt(r.contracted)}</td>
                  <td className={Math.abs(r.contracted - r.direct) <= 1e-9 * Math.max(1, Math.abs(r.direct)) ? 'is-match' : undefined}>{fmt(r.direct)}</td>
                  <td>
                    {r.mix.map((x, i) => (
                      <span key={x.lp}>
                        {i ? ' + ' : ''}
                        {frac(x.ci)}·p<sub>{x.lp}</sub>
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="lr-cap">
            Σ C(l) A<sub>l</sub>² (Eq. 7) equals the double sum Σ<sub>jk</sub> R<sub>j</sub>R<sub>k</sub> cos<sup>l</sup> θ<sub>jk</sub> (Eq. 8; green
            when equal to 10⁻⁹). p<sub>l</sub> = Σ<sub>jk</sub> R<sub>j</sub>R<sub>k</sub> P<sub>l</sub>(cos θ<sub>jk</sub>) is the spherical-harmonic power
            spectrum.
          </p>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-l`}>
            <span>angular order l</span>
            <input id={`${ids}-l`} type="range" min={0} max={LMAX} step={1} value={l} onChange={(ev) => setL(Number(ev.target.value))} />
            <output>{l}</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>cutoff r<sub>c</sub></span>
            <input id={`${ids}-rc`} type="range" min={2.5} max={5.5} step={0.1} value={rc} onChange={(ev) => setRc(Number(ev.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" onClick={() => setSeed((s) => s + 1)}>
              rotate the environment
            </button>
            {seed ? (
              <button type="button" className="lab-btn" onClick={() => setSeed(0)}>
                reset
              </button>
            ) : null}
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>neighbours within r<sub>c</sub></dt>
            <dd>{vectors.length}</dd>
          </div>
          <div className={seed && aChange > 1e-9 ? 'is-alert' : undefined}>
            <dt>largest change of an A<sub>l</sub> component after rotating</dt>
            <dd>{seed ? fmt(aChange) : '—'}</dd>
          </div>
          <div className="is-good">
            <dt>largest relative change of B⁽²⁾ (l ≤ 4) and B⁽³⁾₁₂</dt>
            <dd>{seed ? (bChange < 1e-12 ? '0 (to rounding)' : bChange.toExponential(1)) : '—'}</dd>
          </div>
        </dl>
        <p className="lab-note">
          A<sub>l</sub> has (l+1)(l+2)/2 components where a spherical-harmonic block has 2l+1: the extra ones are traces, lower orders hidden in
          x² + y² + z² = 1. Contracting with the multinomial weights turns them into cos<sup>l</sup> θ, which is not a single Legendre
          polynomial but a fixed mix of P<sub>l</sub>, P<sub>l−2</sub>, …; so each Cartesian invariant up to l spans the same space as the spherical
          ones up to l, the sense in which CACE is equivalent to ACE. Rotating changes every A<sub>l</sub> (they are equivariant) and no
          invariant. Radial weights R(r) are Behler’s cosine cutoff, not CACE’s trained Bessel channels.
        </p>
      </div>
    </div>
  );
}

function fmt(x: number): string {
  if (Math.abs(x) < 5e-13) return '0';
  const a = Math.abs(x);
  return a >= 100 ? x.toFixed(1) : a >= 0.01 ? x.toFixed(3) : x.toExponential(1);
}

function frac(c: number): string {
  for (const den of [1, 3, 5, 7, 15, 35]) {
    const num = c * den;
    if (Math.abs(num - Math.round(num)) < 1e-9) return den === 1 ? String(Math.round(num)) : `${Math.round(num)}/${den}`;
  }
  return c.toFixed(3);
}

function surface(env: Neighbour[], l: number, radius: number, theme: SceneTheme, nT = 48, nP = 96): SurfaceData {
  const nv = (nT + 1) * (nP + 1);
  const vals = new Float64Array(nv);
  const dirs: Vec3[] = [];
  let max = 0;
  for (let i = 0; i <= nT; i++) {
    const th = (Math.PI * i) / nT;
    for (let j = 0; j <= nP; j++) {
      const ph = (2 * Math.PI * j) / nP;
      const d: Vec3 = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const f = momentSurface(env, l, d);
      vals[dirs.length] = f;
      dirs.push(d);
      max = Math.max(max, Math.abs(f));
    }
  }
  const s = radius / (max || 1);
  const positions = new Float32Array(nv * 3);
  const colors = new Float32Array(nv * 3);
  const pos = linear(theme.fail);
  const neg = linear(theme.arch);
  const mid = linear(theme.dark ? '#4a5260' : '#d5d8de');
  dirs.forEach((d, k) => {
    const r = Math.abs(vals[k]!) * s;
    positions.set([d[0] * r, d[1] * r, d[2] * r], 3 * k);
    colors.set(mixRGB(mid, vals[k]! >= 0 ? pos : neg, Math.min(1, (0.35 + Math.abs(vals[k]!) / (max || 1)))), 3 * k);
  });
  const indices = new Uint32Array(nT * nP * 6);
  let t = 0;
  for (let i = 0; i < nT; i++)
    for (let j = 0; j < nP; j++) {
      const a = i * (nP + 1) + j;
      const b = a + nP + 1;
      indices.set([a, b, a + 1, a + 1, b, b + 1], t);
      t += 6;
    }
  return { positions, colors, indices };
}

function Scene({ theme, vectors, env, rc, l }: { theme: SceneTheme; vectors: Vec3[]; env: Neighbour[]; rc: number; l: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 10, 40);
  const surf = useMemo(() => surface(env, l, 0.5 * rc, theme), [env, l, rc, theme]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = vectors.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    const arch = linear(theme.phys);
    pos.set([0, 0, 0], 0);
    radii[0] = 0.28;
    colors.set(linear(theme.ink), 0);
    vectors.forEach((v, k) => {
      pos.set(v, 3 * (k + 1));
      radii[k + 1] = 0.3;
      colors.set(mixRGB(stone, arch, 0.2 + 0.8 * env[k]!.w), 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.3), ao: new Float32Array(n).fill(1), count: n };
  }, [vectors, env, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.ruleStrong);
    return vectors.map((v, k) => ({ a: [0, 0, 0] as Vec3, b: v, color: c, radius: 0.02 + 0.04 * env[k]!.w }));
  }, [vectors, env, theme]);
  return (
    <>
      <ShadedSurface data={surf} uniforms={uniforms} />
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
    </>
  );
}
