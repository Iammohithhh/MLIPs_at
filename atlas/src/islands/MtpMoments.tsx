import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Mtp.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, ShadedSurface, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { mulberry32 } from '~/lib/chem/rng';
import { latticeEnv } from '~/lib/models/soap';
import { axisRotation, type Mat3 } from '~/lib/models/escn';
import { basisValues, describe, enumerateBasis, momentMesh, momentName, momentTensor, radialValue, tensorNorm, type Vec3 } from '~/lib/models/mtp';

// Moments of one atom's neighbourhood (src/lib/models/mtp.ts): the tensor M_{μ,ν} drawn as the polynomial
// n̂ ↦ M[n̂, …, n̂] on the sphere, the neighbours coloured by the radial weight f_μ, and the invariants of
// level ≤ 10 with their change under a rotation of the whole neighbourhood.

const D = 2.35; // Si nearest-neighbour distance, Å
const O = { rmin: 1.9, rcut: 5.0 };
const AXIS: Vec3 = [0.4, 1, 0.3];
const BASIS = enumerateBasis(10);

type EnvKind = 'diamond' | 'vacancy' | 'surface' | 'fcc' | 'hcp' | 'bcc';
const ENVS: Record<EnvKind, { name: string; note: string }> = {
  diamond: { name: 'diamond', note: 'tetrahedral site: no inversion centre' },
  vacancy: { name: 'next to a vacancy', note: 'diamond with one nearest neighbour removed' },
  surface: { name: '(001) surface', note: 'bulk-terminated diamond, everything above the atom removed' },
  fcc: { name: 'fcc', note: 'centrosymmetric, cubic' },
  hcp: { name: 'hcp', note: 'ideal c/a; no inversion centre at the site' },
  bcc: { name: 'bcc', note: 'centrosymmetric, cubic' },
};

function environment(kind: EnvKind, rattle: number): Vec3[] {
  let env: Vec3[];
  if (kind === 'vacancy' || kind === 'surface' || kind === 'diamond') {
    env = latticeEnv('diamond', D, O.rcut);
    const a = (4 * D) / Math.sqrt(3);
    if (kind === 'vacancy') env = env.filter((v) => Math.hypot(v[0] - a / 4, v[1] - a / 4, v[2] - a / 4) > 1e-6);
    if (kind === 'surface') env = env.filter((v) => v[2] < 1e-6);
  } else env = latticeEnv(kind, D, O.rcut);
  const r = mulberry32(17);
  return env.map((v) => [v[0] + rattle * (2 * r() - 1), v[1] + rattle * (2 * r() - 1), v[2] + rattle * (2 * r() - 1)]);
}

const apply = (R: Mat3, v: Vec3): Vec3 => [
  R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2],
  R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2],
  R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2],
];

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
function fmt(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1e5) return v.toPrecision(4);
  const [m, e] = v.toExponential(2).split('e');
  return `${m} × 10${e!.replace('+', '').split('').map((c) => SUP[c] ?? c).join('')}`;
}

export default function MtpMoments() {
  const theme = useSceneTheme();
  const ids = useId();
  const [kind, setKind] = useState<EnvKind>('diamond');
  const [mu, setMu] = useState(0);
  const [nu, setNu] = useState(3);
  const [rattle, setRattle] = useState(0);
  const [angle, setAngle] = useState(0);

  const env0 = useMemo(() => environment(kind, rattle), [kind, rattle]);
  const R = useMemo(() => axisRotation(AXIS, (angle * Math.PI) / 180), [angle]);
  const env = useMemo(() => env0.map((v) => apply(R, v)), [env0, R]);
  const mesh = useMemo(() => momentMesh(env, mu, nu, O, 1.45), [env, mu, nu]);
  const norm = useMemo(() => tensorNorm(momentTensor(env, mu, nu, O)), [env, mu, nu]);
  const scale = useMemo(() => tensorNorm(momentTensor(env, mu, 0, O)), [env, mu]);
  const b0 = useMemo(() => basisValues(env0, BASIS, O), [env0]);
  const b = useMemo(() => basisValues(env, BASIS, O), [env]);
  const inv = useMemo(() => Math.max(...BASIS.map((_, k) => Math.abs(b[k]! - b0[k]!) / Math.max(1e-300, Math.abs(b0[k]!) || 1))), [b, b0]);
  const zero = norm < 1e-9 * Math.max(1, scale);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [6, -24, 10], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`An atom's neighbourhood within ${O.rcut} Å (${ENVS[kind].name}), neighbours sized and coloured by the radial weight f${mu}, and at the centre the moment tensor M ${mu},${nu} drawn as a surface whose distance from the centre in each direction is the size of the tensor in that direction.`}
            minDistance={8}
            maxDistance={60}
          >
            <Scene theme={theme} env={env} mu={mu} mesh={mesh} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Environment">
              {(Object.keys(ENVS) as EnvKind[]).map((k) => (
                <button key={k} type="button" aria-pressed={kind === k} onClick={() => setKind(k)}>
                  {ENVS[k].name}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.phys }} /> M[n̂,…,n̂] &gt; 0, f<sub>μ</sub> &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> &lt; 0
            </li>
            <li>
              <span className="ring" style={{ borderColor: theme.ruleStrong }} /> cutoff {O.rcut} Å
            </li>
          </ul>
        </div>
        <div className="mtp-side">
          <div className="lab-controls">
            <div className="seg" role="group" aria-label="Radial index μ">
              <span className="seg-label">μ</span>
              {[0, 1, 2].map((k) => (
                <button key={k} type="button" aria-pressed={mu === k} onClick={() => setMu(k)}>
                  {k}
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Tensor rank ν">
              <span className="seg-label">ν</span>
              {[0, 1, 2, 3, 4].map((k) => (
                <button key={k} type="button" aria-pressed={nu === k} onClick={() => setNu(k)}>
                  {k}
                </button>
              ))}
            </div>
            <label htmlFor={`${ids}-r`}>
              <span>rattle every neighbour by up to</span>
              <input id={`${ids}-r`} type="range" min={0} max={0.3} step={0.01} value={rattle} onChange={(e) => setRattle(Number(e.target.value))} />
              <output>{rattle.toFixed(2)} Å</output>
            </label>
            <label htmlFor={`${ids}-a`}>
              <span>rotate the neighbourhood</span>
              <input id={`${ids}-a`} type="range" min={0} max={360} step={5} value={angle} onChange={(e) => setAngle(Number(e.target.value))} />
              <output>{angle}°</output>
            </label>
          </div>
          <dl className="lab-readout" aria-live="polite">
            <div className={zero ? 'is-alert' : undefined}>
              <dt>
                |{momentName(mu, nu)}| (Frobenius norm)
              </dt>
              <dd>{zero ? '0 (cancels by symmetry)' : fmt(norm)}</dd>
            </div>
            <div>
              <dt>neighbours inside the cutoff</dt>
              <dd>{env.length}</dd>
            </div>
            <div className={inv < 1e-10 ? 'is-good' : 'is-alert'}>
              <dt>largest change of the {BASIS.length} invariants under the rotation</dt>
              <dd>{angle === 0 ? 'turn the slider' : inv < 1e-15 ? '< 10⁻¹⁵ (relative)' : `${fmt(inv)} (relative)`}</dd>
            </div>
          </dl>
          <div className="mtp-table-wrap">
            <table className="mtp-table">
              <caption>All {BASIS.length} basis functions of level ≤ 10 for this neighbourhood</caption>
              <thead>
                <tr>
                  <th scope="col">basis function</th>
                  <th scope="col">level</th>
                  <th scope="col">value</th>
                </tr>
              </thead>
              <tbody>
                {BASIS.map((f, k) => {
                  const z = Math.abs(b[k]!) < 1e-9 * Math.max(1, Math.abs(b[0]!) ** f.mu.length);
                  return (
                    <tr key={f.key} className={z ? 'is-zero' : undefined}>
                      <td>{describe(f)}</td>
                      <td>{f.level}</td>
                      <td>{z ? '0' : fmt(b[k]!)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <p className="lab-note">
        {ENVS[kind].name}: {ENVS[kind].note}. Odd ν changes sign under inversion, so M<sub>μ,1</sub> and M<sub>μ,3</sub> vanish at the
        centrosymmetric fcc and bcc sites and every invariant built from them is zero there; the tetrahedral diamond site has no vector
        moment but a rank-3 one. At the cubic sites M<sub>μ,2</sub> is a sphere: the second moment is isotropic. Remove a neighbour or
        cut a surface and odd moments appear. Rotate the neighbourhood: the surfaces turn with it, the invariants do not change.
      </p>
    </div>
  );
}

function Scene({ theme, env, mu, mesh }: { theme: SceneTheme; env: Vec3[]; mu: number; mesh: ReturnType<typeof momentMesh> }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const surface = useMemo<SurfaceData>(() => {
    const pos = linear(theme.phys);
    const neg = linear(theme.caveat);
    const colors = new Float32Array(mesh.signs.length * 3);
    mesh.signs.forEach((s, k) => colors.set(s > 0 ? pos : neg, 3 * k));
    return { positions: mesh.positions, colors, indices: mesh.indices };
  }, [mesh, theme]);
  const atoms = useMemo<AtomBuffers>(() => {
    const n = env.length + 1;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const f = env.map((v) => radialValue(Math.hypot(...v), mu, O));
    const mx = Math.max(...f.map(Math.abs)) || 1;
    const pos = linear(theme.phys);
    const neg = linear(theme.caveat);
    const pale = linear(theme.ruleStrong);
    env.forEach((v, k) => {
      positions.set(v, 3 * k);
      const w = Math.abs(f[k]!) / mx;
      radii[k] = 0.12 + 0.3 * Math.sqrt(w);
      colors.set(mixRGB(pale, f[k]! >= 0 ? pos : neg, 0.25 + 0.75 * Math.sqrt(w)), 3 * k);
    });
    // the central atom, drawn small so the surface stays visible
    positions.set([0, 0, 0], 3 * env.length);
    radii[env.length] = 0.18;
    colors.set(linear(theme.ink), 3 * env.length);
    return { positions, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [env, mu, theme]);
  return (
    <>
      {mesh.maxAbs > 0 ? <ShadedSurface data={surface} uniforms={uniforms} /> : null}
      <ImpostorAtoms buffers={atoms} version={atoms.count * 1000 + mu} uniforms={uniforms} />
      <Shell center={[0, 0, 0]} radius={O.rcut} color={theme.ruleStrong} opacity={0.18} grid />
    </>
  );
}
