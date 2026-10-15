import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './MaceWalkthrough.css';
import si from '~/data/structures/si-diamond.json';
import zno from '~/data/structures/zno-wurtzite.json';
import ethanol from '~/data/structures/ethanol.json';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  ImpostorAtoms,
  Lines,
  Shell,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { covalentRadius, elementColor, hexToLinearRgb } from '~/lib/structure/elements';
import { neighboursOf } from '~/lib/structure/geometry';
import type { Structure, Vec3 } from '~/lib/structure/types';
import { realSH } from '~/lib/physics/spherical-harmonics';
import { atomicBasis, powerSpectrum, productCount, radial } from '~/lib/models/mace-features';

// The MACE layer, step by step, on a real atomic environment. Equations are typeset
// at build time and passed in; every number shown is computed from the structure.

export interface WalkStep {
  id: 'embed' | 'radial' | 'sh' | 'A' | 'B' | 'message' | 'readout';
  title: string;
  /** Pre-rendered KaTeX HTML. */
  eq: string;
  shape: string;
  intuition: string;
  technical: string;
}

type SysId = 'si' | 'zno' | 'ethanol';
const SYSTEMS: Record<SysId, { label: string; s: Structure }> = {
  si: { label: 'Si (diamond)', s: si as unknown as Structure },
  zno: { label: 'ZnO (wurtzite)', s: zno as unknown as Structure },
  ethanol: { label: 'Ethanol', s: ethanol as unknown as Structure },
};

function colorOf(sym: string, dark: boolean): [number, number, number] {
  const c = elementColor(sym, dark ? 'dark' : 'light');
  return hexToLinearRgb('#' + c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join(''));
}

interface EnvAtom {
  sym: string;
  /** Position relative to atom i (Å). */
  d: Vec3;
  r: number;
}

function environment(sys: SysId, rc: number): { centre: string; atoms: EnvAtom[] } {
  const s = SYSTEMS[sys].s;
  const i = sys === 'ethanol' ? 1 : 0;
  const pi = s.positions[i]!;
  const atoms = neighboursOf(s, i, 2 * rc).map((n) => {
    const d: Vec3 = [n.position[0] - pi[0], n.position[1] - pi[1], n.position[2] - pi[2]];
    return { sym: s.symbols[n.j]!, d, r: n.distance };
  });
  return { centre: s.symbols[i]!, atoms };
}

export default function MaceWalkthrough({ steps }: { steps: WalkStep[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [stepId, setStepId] = useState<WalkStep['id']>('A');
  const [sys, setSys] = useState<SysId>('zno');
  const [rc, setRc] = useState(5);
  const [n, setN] = useState(1);
  const [l, setL] = useState(1);
  const [m, setM] = useState(0);
  const step = steps.find((s) => s.id === stepId)!;
  const mm = Math.max(-l, Math.min(l, m));

  const env = useMemo(() => environment(sys, rc), [sys, rc]);
  const inside = useMemo(() => env.atoms.filter((a) => a.r < rc), [env, rc]);
  const A = useMemo(() => atomicBasis(inside.map((a) => a.d), rc, n, 3), [inside, rc, n]);
  const P = powerSpectrum(A);

  return (
    <div className="lab mw">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={sys}
            view={sys === 'ethanol' ? { position: [5, -8, 5], target: [0, 0, 0] } : { position: [17, -28, 16], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Atomic environment of the central ${env.centre} atom in ${SYSTEMS[sys].label}, cutoff ${rc.toFixed(1)} Å, showing the ${step.title} step of a MACE layer.`}
            minDistance={4}
            maxDistance={70}
          >
            <EnvScene theme={theme} centre={env.centre} atoms={env.atoms} rc={rc} step={stepId} n={n} l={l} m={mm} A={A} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {(Object.keys(SYSTEMS) as SysId[]).map((k) => (
                <button key={k} type="button" aria-pressed={sys === k} onClick={() => setSys(k)}>
                  {SYSTEMS[k].label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="ring" /> cutoff r<sub>c</sub> = {rc.toFixed(1)} Å
            </li>
            <li>{inside.length} neighbours inside</li>
            {stepId === 'message' && <li>{env.atoms.length} within 2r<sub>c</sub> (two layers)</li>}
          </ul>
        </div>

        <div className="mw-side">
          <ol className="mw-flow" aria-label="Steps of one MACE layer">
            {steps.map((s, k) => (
              <li key={s.id}>
                <button type="button" aria-pressed={s.id === stepId} onClick={() => setStepId(s.id)}>
                  <span className="mw-k">{k + 1}</span>
                  <span className="mw-t">{s.title}</span>
                  <span className="mw-shape">{s.shape}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="mw-eq" dangerouslySetInnerHTML={{ __html: step.eq }} />
          <p className="mw-intuition">{step.intuition}</p>
          <p className="mw-technical">{step.technical}</p>
        </div>
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-rc`} data-sym="rc">
            <span>
              cutoff r<sub>c</sub>
            </span>
            <input id={`${ids}-rc`} type="range" min={3} max={6} step={0.1} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-n`}>
            <span>radial function n</span>
            <input id={`${ids}-n`} type="range" min={1} max={8} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} />
            <output>{n}</output>
          </label>
          {stepId === 'sh' && (
            <div className="seg mw-lm" role="group" aria-label="Spherical harmonic">
              <span className="seg-label">l</span>
              {[0, 1, 2, 3].map((k) => (
                <button key={k} type="button" aria-pressed={l === k} onClick={() => { setL(k); setM((v) => Math.max(-k, Math.min(k, v))); }}>
                  {k}
                </button>
              ))}
              <span className="seg-label">m</span>
              {Array.from({ length: 2 * l + 1 }, (_, k) => k - l).map((k) => (
                <button key={k} type="button" aria-pressed={mm === k} onClick={() => setM(k)}>
                  {k > 0 ? `+${k}` : k}
                </button>
              ))}
            </div>
          )}
        </div>
        <InvariantPanel P={P} stepId={stepId} sys={sys} />
      </div>
    </div>
  );
}

function InvariantPanel({ P, stepId, sys }: { P: number[]; stepId: WalkStep['id']; sys: SysId }) {
  const max = Math.max(...P, 1e-9);
  const D = 16;
  return (
    <div className="mw-panel" aria-live="polite">
      {stepId === 'B' ? (
        <table className="mw-count">
          <thead>
            <tr>
              <th>correlation order ν</th>
              <th>body order of message</th>
              <th>products of the 16 A components</th>
            </tr>
          </thead>
          <tbody>
            {[1, 2, 3].map((nu) => (
              <tr key={nu}>
                <td>{nu}</td>
                <td>{nu + 1}-body</td>
                <td className="num">{productCount(D, nu).toLocaleString('en-GB')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <>
          <p className="mw-ptitle">
            Invariants of this environment: P<sub>l</sub> = Σ<sub>m</sub> A<sub>lm</sub>² (unchanged by any rotation)
          </p>
          <div className="mw-bars">
            {P.map((p, l) => (
              <div key={l} className="mw-bar">
                <span className="mw-bl">l = {l}</span>
                <span className="mw-track">
                  <span className="mw-fill" style={{ width: `${(p / max) * 100}%` }} />
                </span>
                <span className="mw-bv">{p < 1e-12 ? '0 (by symmetry)' : p.toExponential(2)}</span>
              </div>
            ))}
          </div>
          {sys === 'si' && <p className="mw-note">Diamond: tetrahedral symmetry cancels l = 1 and l = 2 exactly; the first non-zero block is l = 3.</p>}
        </>
      )}
    </div>
  );
}

function EnvScene({
  theme,
  centre,
  atoms,
  rc,
  step,
  n,
  l,
  m,
  A,
}: {
  theme: SceneTheme;
  centre: string;
  atoms: EnvAtom[];
  rc: number;
  step: WalkStep['id'];
  n: number;
  l: number;
  m: number;
  A: number[][];
}) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const showSecond = step === 'message';
  const pos = linear(theme.phys);
  const neg = linear(theme.caveat);
  const stone = linear(theme.dark ? theme.canvas : '#b7bec9');

  const buffers = useMemo<AtomBuffers>(() => {
    const list = [{ sym: centre, d: [0, 0, 0] as Vec3, r: 0 }, ...atoms];
    const N = list.length;
    const positions = new Float32Array(N * 3);
    const radii = new Float32Array(N);
    const colors = new Float32Array(N * 3);
    const rad = atoms.filter((a) => a.r < rc).map((a) => radial(n, a.r, rc));
    const rmax = Math.max(1e-9, ...rad.map(Math.abs));
    list.forEach((a, k) => {
      positions.set(a.d, 3 * k);
      const base = colorOf(a.sym, theme.dark);
      const rr = Math.max(0.28, 0.5 * covalentRadius(a.sym).r);
      if (k === 0) {
        radii[k] = rr * 1.25;
        colors.set(base, 3 * k);
        return;
      }
      const inCut = a.r < rc;
      if (!inCut) {
        radii[k] = showSecond ? rr * 0.75 : 0;
        colors.set(mixRGB(base, stone, 0.35), 3 * k);
        return;
      }
      radii[k] = rr;
      if (step === 'radial') {
        const v = radial(n, a.r, rc) / rmax;
        colors.set(mixRGB(stone, v >= 0 ? pos : neg, Math.min(1, Math.abs(v))), 3 * k);
      } else if (step === 'sh') {
        const v = realSH(l, m, a.d) / Math.sqrt((2 * l + 1) / (4 * Math.PI));
        colors.set(mixRGB(stone, v >= 0 ? pos : neg, Math.min(1, Math.abs(v))), 3 * k);
      } else colors.set(base, 3 * k);
    });
    return { positions, radii, colors, metal: new Float32Array(N), ao: new Float32Array(N).fill(1), count: N };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, centre, rc, step, n, l, m, theme]);

  // Edges i → j for neighbours inside the cutoff (the graph a layer sees).
  const edges = useMemo<ColoredLine[]>(() => {
    if (step === 'embed' || step === 'readout') return [];
    const c = linear(theme.inkMuted);
    return atoms.filter((a) => a.r < rc).map((a) => ({ a: [0, 0, 0] as [number, number, number], b: a.d as [number, number, number], rgba: [c[0], c[1], c[2], 0.45] as [number, number, number, number] }));
  }, [atoms, rc, step, theme.inkMuted]);

  // Second-layer edges (neighbours of neighbours), only for the message step.
  const edges2 = useMemo<ColoredLine[]>(() => {
    if (!showSecond) return [];
    const c = linear(theme.lit);
    const inner = atoms.filter((a) => a.r < rc);
    const outer = atoms.filter((a) => a.r >= rc);
    const out: ColoredLine[] = [];
    for (const o of outer) {
      const via = inner.find((a) => Math.hypot(o.d[0] - a.d[0], o.d[1] - a.d[1], o.d[2] - a.d[2]) < rc);
      if (via) out.push({ a: via.d as [number, number, number], b: o.d as [number, number, number], rgba: [c[0], c[1], c[2], 0.35] });
    }
    return out;
  }, [atoms, rc, showSecond, theme.lit]);

  // Atomic-basis l = 1 block as a vector (A_{1,−1}, A_{1,0}, A_{1,+1}) ∝ (y, z, x).
  const aVec = useMemo<Segment[]>(() => {
    if (step !== 'A') return [];
    const b = A[1]!;
    const v: Vec3 = [b[2]!, b[0]!, b[1]!];
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len < 1e-6) return [];
    const s = Math.min(3.6, 2.3 + 40 * len) / len; // at least 2.3 Å so it clears the central atom
    return [{ a: [0, 0, 0], b: [v[0] * s, v[1] * s, v[2] * s], color: linear(theme.arch), radius: 0.11 }];
  }, [A, step, theme.arch]);

  // A few three-body triangles i–j–k for the product-basis step.
  const tris = useMemo<ColoredLine[]>(() => {
    if (step !== 'B') return [];
    const c = linear(theme.caveat);
    const inner = atoms.filter((a) => a.r < rc).slice(0, 8);
    const out: ColoredLine[] = [];
    for (let a = 0; a < inner.length; a++)
      for (let b = a + 1; b < inner.length; b += 3) {
        out.push({ a: inner[a]!.d as [number, number, number], b: inner[b]!.d as [number, number, number], rgba: [c[0], c[1], c[2], 0.55] });
      }
    return out;
  }, [atoms, rc, step, theme.caveat]);

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {edges.length > 0 && <Lines lines={edges} />}
      {edges2.length > 0 && <Lines lines={edges2} />}
      {tris.length > 0 && <Lines lines={tris} />}
      {aVec.length > 0 && <Arrows arrows={aVec} uniforms={uniforms} head={0.45} />}
      <Shell center={[0, 0, 0]} radius={rc} color={theme.phys} opacity={step === 'radial' ? 0.55 : 0.32} grid={step !== 'radial'} />
      {showSecond && <Shell center={[0, 0, 0]} radius={2 * rc} color={theme.lit} opacity={0.22} grid />}
      {step === 'readout' && <Shell center={[0, 0, 0]} radius={0.95} color={theme.verified} opacity={0.9} />}
    </>
  );
}
