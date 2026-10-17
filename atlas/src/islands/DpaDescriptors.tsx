import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dpa.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { mulberry32, randomRotation } from '~/lib/chem/rng';
import {
  CU_D,
  DPA31_3M,
  RCS_RATIO,
  closePacked,
  dpa1Descriptor,
  dpa3Descriptor,
  makeToy,
  pyramid,
  relDiff,
  rotate,
  sPoly,
  seDescriptor,
  shuffle,
  square,
  tShape,
  tetrahedron,
  type EnvOptions,
  type Vec3,
} from '~/lib/models/dpa';

// Three descriptor constructions of the Deep Potential family on pairs of environments
// (src/lib/models/dpa.ts): DeepPot-SE's D = Gᵀ R̃ R̃ᵀ G (NeurIPS 2018, Eq. 11), DPA-1's gated
// attention before the same product (Eqs. 6–10), and a DPA-3-style angle embedding φ(cos θ)
// (Eq. 19). All three are exactly invariant; what differs is which environments they separate.
// Weights are random and untrained, so the size of a non-zero difference means nothing; whether it
// is zero does.

type Pair = { id: string; name: string; a: string; b: string; crystal: boolean };
const PAIRS: Pair[] = [
  { id: 'fcc-hcp', name: 'fcc vs hcp', a: 'fcc Cu', b: 'hcp Cu (ideal c/a)', crystal: true },
  { id: 'pyr-t', name: 'pyramid vs T', a: 'pyramid', b: 'T-shape', crystal: false },
  { id: 'tet-sq', name: 'tetrahedron vs square', a: 'tetrahedron', b: 'square', crystal: false },
];
const BOND = 2.2;
const MOL_RC = 4;
const RCS = Array.from({ length: 67 }, (_, k) => 2.7 + 0.05 * k);
const FLOOR = 1e-16;
const SAME = 1e-12;

const toy = makeToy();

function environments(pair: string, rc: number): [Vec3[], Vec3[]] {
  if (pair === 'fcc-hcp') return [closePacked('fcc', CU_D, rc + 1), closePacked('hcp', CU_D, rc + 1)];
  if (pair === 'pyr-t') return [pyramid(BOND), tShape(BOND)];
  return [tetrahedron(BOND), square(BOND)];
}

function opts(rc: number): EnvOptions {
  return { rc, rcs: RCS_RATIO * rc, rcAngle: DPA31_3M.rcAngle };
}

interface Compared {
  se: number;
  dpa1: number;
  dpa3: number;
}

function compare(a: Vec3[], b: Vec3[], o: EnvOptions): Compared {
  return {
    se: relDiff(seDescriptor(toy, a, o).D, seDescriptor(toy, b, o).D),
    dpa1: relDiff(dpa1Descriptor(toy, a, o).D, dpa1Descriptor(toy, b, o).D),
    dpa3: relDiff(dpa3Descriptor(toy, a, o).v, dpa3Descriptor(toy, b, o).v),
  };
}

const ROWS: Array<{ key: keyof Compared; label: string; sub: string; cls: string }> = [
  { key: 'se', label: 'DeepPot-SE', sub: 'D = Gᵀ R̃ R̃ᵀ G', cls: 'is-se' },
  { key: 'dpa1', label: 'DPA-1', sub: 'gated attention ×2', cls: 'is-dpa1' },
  { key: 'dpa3', label: 'DPA-3-style', sub: 'embeds cos θ', cls: 'is-dpa3' },
];

export default function DpaDescriptors() {
  const theme = useSceneTheme();
  const ids = useId();
  const [pair, setPair] = useState('fcc-hcp');
  const [side, setSide] = useState<'a' | 'b'>('a');
  const [rcCrystal, setRc] = useState(3.2);
  const [seed, setSeed] = useState(1);
  const p = PAIRS.find((x) => x.id === pair)!;
  const rc = p.crystal ? rcCrystal : MOL_RC;
  const o = useMemo(() => opts(rc), [rc]);
  const [envA, envB] = useMemo(() => environments(pair, rc), [pair, rc]);
  const diff = useMemo(() => compare(envA, envB, o), [envA, envB, o]);
  const env = side === 'a' ? envA : envB;
  const se = useMemo(() => seDescriptor(toy, env, o), [env, o]);
  const att = useMemo(() => dpa1Descriptor(toy, env, o), [env, o]);
  const d3 = useMemo(() => dpa3Descriptor(toy, env, o), [env, o]);
  // Invariance check: rotate and relabel the shown environment, compare with itself.
  const invariance = useMemo(() => {
    const moved = shuffle(rotate(env, randomRotation(mulberry32(seed))), seed + 7);
    return Math.max(
      relDiff(seDescriptor(toy, env, o).D, seDescriptor(toy, moved, o).D),
      relDiff(dpa1Descriptor(toy, env, o).D, dpa1Descriptor(toy, moved, o).D),
      relDiff(dpa3Descriptor(toy, env, o).v, dpa3Descriptor(toy, moved, o).v),
    );
  }, [env, o, seed]);
  const curve = useMemo(() => RCS.map((r) => {
    const [a, b] = environments('fcc-hcp', r);
    return compare(a, b, opts(r));
  }), []);
  const dipole = useMemo(() => {
    const v: Vec3 = [0, 0, 0];
    for (const q of env) {
      const r = Math.hypot(...q);
      if (r >= rc) continue;
      const s = sPoly(r, o.rcs, rc);
      for (let k = 0; k < 3; k++) v[k]! += (s * q[k]!) / r;
    }
    return v;
  }, [env, rc, o]);
  const dip = Math.hypot(...dipole);
  const extent = p.crystal ? rc + 1 : 3.2;
  const view = useMemo(() => {
    const k = 3.4 * extent;
    return { position: [0.62 * k, -0.7 * k, 0.5 * k] as [number, number, number], target: [0, 0, 0] as [number, number, number] };
  }, [extent]);
  const name = side === 'a' ? p.a : p.b;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={100 * PAIRS.indexOf(p) + Math.round(extent)}
            background={theme.canvas}
            label={`${name}: the central atom and its neighbours within ${rc.toFixed(2)} Å, shaded by the radial weight s(r). The arrow is the weighted dipole Σ s(r) r̂, ${dip < 1e-9 ? 'which is zero here' : `of length ${dip.toFixed(3)}`}.`}
            minDistance={4}
            maxDistance={80}
          >
            <Scene theme={theme} env={env} rc={rc} rcs={o.rcs} dipole={dipole} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Environment shown">
              <button type="button" aria-pressed={side === 'a'} onClick={() => setSide('a')}>
                A: {p.a}
              </button>
              <button type="button" aria-pressed={side === 'b'} onClick={() => setSide('b')}>
                B: {p.b}
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.ink }} /> central atom
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> inside r<sub>c</sub>, darker = larger s(r)
            </li>
            <li>
              <span className="bar" style={{ background: theme.fail }} /> Σ s(r) r̂ {dip < 1e-9 ? <small>(zero)</small> : null}
            </li>
          </ul>
        </div>
        <div className="dpa-side">
          <div className="seg dpa-pairs" role="group" aria-label="Pair of environments">
            {PAIRS.map((x) => (
              <button key={x.id} type="button" aria-pressed={pair === x.id} onClick={() => setPair(x.id)}>
                {x.name}
              </button>
            ))}
          </div>
          <DiffBars theme={theme} diff={diff} />
          {p.crystal ? <DiffCurve theme={theme} curve={curve} rc={rc} onPick={setRc} /> : <PairNote pair={pair} />}
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          {p.crystal ? (
            <label htmlFor={`${ids}-rc`}>
              <span>cutoff r<sub>c</sub></span>
              <input id={`${ids}-rc`} type="range" min={RCS[0]} max={RCS.at(-1)} step={0.05} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
              <output>{rc.toFixed(2)} Å</output>
            </label>
          ) : (
            <p className="dpa-fixed">Neighbours at {BOND} Å, r<sub>c</sub> = {MOL_RC} Å.</p>
          )}
          <div className="presets">
            <button type="button" className="lab-btn" onClick={() => setSeed((s) => s + 1)}>
              rotate and relabel {side.toUpperCase()}
            </button>
            {p.crystal ? (
              <button type="button" className="lab-btn" onClick={() => setRc(6)}>
                DPA-3’s 6 Å
              </button>
            ) : null}
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>neighbours inside r<sub>c</sub> ({side.toUpperCase()})</dt>
            <dd>
              {se.neighbours} <small>{d3.angles.toLocaleString('en')} angles within {Math.min(rc, DPA31_3M.rcAngle).toFixed(1)} Å</small>
            </dd>
          </div>
          <div className={Math.max(...se.dipoleNorms) < 1e-9 ? 'is-alert' : undefined}>
            <dt>largest filtered dipole |Σ G<sub>a</sub> s r̂|</dt>
            <dd>{fmt(Math.max(...se.dipoleNorms))}</dd>
          </div>
          <div className={Math.max(...att.updateNorm) < 1e-9 ? 'is-alert' : undefined}>
            <dt>DPA-1 attention update, before LayerNorm</dt>
            <dd>{fmt(Math.max(...att.updateNorm))}</dd>
          </div>
          <div className="is-good">
            <dt>after rotating and relabelling, largest change</dt>
            <dd>{fmt(invariance)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Each bar is ‖D<sub>A</sub> − D<sub>B</sub>‖ / ‖D‖ on a log scale; anything at the floor is zero to rounding, so that
          descriptor cannot tell A from B whatever its weights. In fcc and hcp every shell of neighbours has zero dipole, so
          DeepPot-SE keeps only radial sums, and DPA-1’s gated attention adds nothing: both stay blind until a cutoff of 4.17 Å takes in
          hcp’s two atoms at 1.633 d, a shell fcc does not have. An embedding of cos θ itself sees the different angles of the first
          shell at once. The pyramid and the T have the same |Σ r̂|², which is all DeepPot-SE sees of their angles, but different
          alignments of each bond with that sum, which DPA-1’s gate picks up.
        </p>
      </div>
    </div>
  );
}

function fmt(x: number): string {
  if (x < SAME) return '0 (to rounding)';
  return x < 1e-2 ? x.toExponential(1) : x.toFixed(3);
}

function logPos(x: number): number {
  // 1e-16 → 0, 1 → 1
  return Math.max(0, Math.min(1, (Math.log10(Math.max(x, FLOOR)) + 16) / 16));
}

function DiffBars({ theme, diff }: { theme: SceneTheme; diff: Compared }) {
  const W = 360;
  const rowH = 34;
  const m = { l: 128, r: 62, t: 8, b: 26 };
  const H = m.t + ROWS.length * rowH + m.b;
  const x = (v: number) => m.l + logPos(v) * (W - m.l - m.r);
  const colour = { se: theme.arch, dpa1: theme.phys, dpa3: theme.caveat };
  return (
    <figure className="lab-plot dpa-bars">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Relative descriptor difference between A and B: DeepPot-SE ${fmt(diff.se)}, DPA-1 ${fmt(diff.dpa1)}, DPA-3-style ${fmt(diff.dpa3)}.`}>
        {[-16, -12, -8, -4, 0].map((e) => (
          <g key={e}>
            <line className="grid" x1={x(10 ** e)} x2={x(10 ** e)} y1={m.t} y2={H - m.b} />
            <text className="tick" x={x(10 ** e)} y={H - m.b + 14} textAnchor="middle">
              {e === 0 ? '1' : `1e${e}`}
            </text>
          </g>
        ))}
        <rect className="dpa-same" x={m.l} y={m.t} width={x(SAME) - m.l} height={H - m.t - m.b} />
        {ROWS.map((r, k) => {
          const v = diff[r.key];
          const y = m.t + k * rowH + 6;
          return (
            <g key={r.key}>
              <text className="dpa-row" x={m.l - 8} y={y + 10} textAnchor="end">
                {r.label}
              </text>
              <text className="dpa-row-sub" x={m.l - 8} y={y + 22} textAnchor="end">
                {r.sub}
              </text>
              <rect x={m.l} y={y + 2} width={Math.max(2, x(v) - m.l)} height={14} rx={3} style={{ fill: colour[r.key], opacity: v < SAME ? 0.35 : 1 }} />
              <text className="dpa-val" x={W - m.r + 6} y={y + 14}>
                {v < SAME ? 'identical' : v.toExponential(1)}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="dpa-cap">‖D<sub>A</sub> − D<sub>B</sub>‖ / ‖D‖, log scale; the grey band is zero to rounding.</figcaption>
    </figure>
  );
}

function DiffCurve({ theme, curve, rc, onPick }: { theme: SceneTheme; curve: Compared[]; rc: number; onPick: (r: number) => void }) {
  const W = 320;
  const H = 190;
  const m = { l: 40, r: 10, t: 12, b: 36 };
  const x = (r: number) => m.l + ((r - RCS[0]!) / (RCS.at(-1)! - RCS[0]!)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - logPos(v)) * (H - m.t - m.b);
  const path = (key: keyof Compared) => curve.map((c, i) => `${i ? 'L' : 'M'}${x(RCS[i]!).toFixed(1)},${y(c[key]).toFixed(1)}`).join('');
  const colour = { se: theme.arch, dpa1: theme.phys, dpa3: theme.caveat };
  const shells = [
    { r: CU_D * Math.SQRT2, t: '√2 d' },
    { r: CU_D * Math.sqrt(8 / 3), t: '1.633 d (hcp)' },
    { r: CU_D * Math.sqrt(3), t: '√3 d' },
  ];
  return (
    <figure className="lab-plot dpa-curve">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Descriptor difference between fcc and hcp copper against the cutoff, for the three descriptors, on a log scale.">
        {[-16, -8, 0].map((e) => (
          <g key={e}>
            <line className="grid" x1={m.l} x2={W - m.r} y1={y(10 ** e)} y2={y(10 ** e)} />
            <text className="tick" x={m.l - 4} y={y(10 ** e) + 3} textAnchor="end">
              {e === 0 ? '1' : `1e${e}`}
            </text>
          </g>
        ))}
        {[3, 4, 5, 6].map((r) => (
          <text key={r} className="tick" x={x(r)} y={H - m.b + 13} textAnchor="middle">
            {r}
          </text>
        ))}
        {shells.map((s) => (
          <g key={s.t}>
            <line className="dpa-shell" x1={x(s.r)} x2={x(s.r)} y1={m.t} y2={H - m.b} />
          </g>
        ))}
        <text className="dpa-shell-label" x={x(shells[1]!.r) + 3} y={m.t + 9}>
          {shells[1]!.t}
        </text>
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">
          cutoff r_c (Å)
        </text>
        {(['dpa3', 'dpa1', 'se'] as const).map((k) => (
          <path key={k} d={path(k)} fill="none" style={{ stroke: colour[k], strokeWidth: k === 'dpa1' ? 3.5 : 2, strokeDasharray: k === 'se' ? '5 3' : undefined }} />
        ))}
        <line className="rc-line is-active" x1={x(rc)} x2={x(rc)} y1={m.t} y2={H - m.b} />
        <rect
          x={m.l}
          y={m.t}
          width={W - m.l - m.r}
          height={H - m.t - m.b}
          fill="transparent"
          style={{ cursor: 'pointer' }}
          onClick={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const px = ((e.clientX - box.left) / box.width) * W;
            const k = Math.round(((px - m.l) / (W - m.l - m.r)) * (RCS.length - 1));
            onPick(RCS[Math.max(0, Math.min(RCS.length - 1, k))]!);
          }}
        />
      </svg>
      <figcaption className="dpa-cap">
        fcc vs hcp against the cutoff. DeepPot-SE (dashed) and DPA-1 coincide; click to set r<sub>c</sub>.
      </figcaption>
    </figure>
  );
}

function PairNote({ pair }: { pair: string }) {
  return pair === 'pyr-t' ? (
    <p className="dpa-pairnote">
      Three bonds of 2.2 Å. The pyramid’s bonds make cos θ = 1/3 with its axis; the T has two opposite bonds and one at right angles.
      Both have |Σ r̂|² = 1.
    </p>
  ) : (
    <p className="dpa-pairnote">
      Four bonds of 2.2 Å: tetrahedral (cos θ = −1/3 between every pair) and square planar (0, 0, −1). Both have Σ r̂ = 0.
    </p>
  );
}

function Scene({ theme, env, rc, rcs, dipole }: { theme: SceneTheme; env: Vec3[]; rc: number; rcs: number; dipole: Vec3 }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const sMax = useMemo(() => Math.max(...env.map((p) => sPoly(Math.hypot(...p), rcs, rc)), 1e-9), [env, rc, rcs]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = env.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const arch = linear(theme.arch);
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    pos.set([0, 0, 0], 0);
    radii[0] = 0.5;
    colors.set(linear(theme.ink), 0);
    env.forEach((p, k) => {
      const r = Math.hypot(...p);
      const s = sPoly(r, rcs, rc) / sMax;
      pos.set(p, 3 * (k + 1));
      radii[k + 1] = r < rc ? 0.36 : 0.2;
      colors.set(r < rc ? mixRGB(stone, arch, 0.25 + 0.75 * s) : mixRGB(stone, linear(theme.canvas), 0.4), 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.4), ao: new Float32Array(n).fill(1), count: n };
  }, [env, rc, rcs, sMax, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.arch);
    return env
      .filter((p) => Math.hypot(...p) < rc)
      .map((p) => ({ a: [0, 0, 0] as Vec3, b: p, color: c, radius: 0.03 + 0.05 * (sPoly(Math.hypot(...p), rcs, rc) / sMax) }));
  }, [env, rc, rcs, sMax, theme]);
  const arrows = useMemo<Segment[]>(() => {
    const L = Math.hypot(...dipole);
    if (L < 1e-9) return [];
    // Drawn 3 Å per unit of s (1/Å): only direction and relative length matter.
    const k = 3 / Math.max(L, 1e-9) * Math.min(1, L / sMax);
    return [{ a: [0, 0, 0], b: [dipole[0] * k, dipole[1] * k, dipole[2] * k], color: linear(theme.fail), radius: 0.09 }];
  }, [dipole, sMax, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      <Arrows arrows={arrows} uniforms={uniforms} head={0.4} />
      <Shell center={[0, 0, 0]} radius={rc} color={theme.arch} opacity={0.08} />
    </>
  );
}
