import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Grace.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import type { Structure } from '~/lib/structure/types';
import { shellAround } from '~/lib/models/three-body';
import { GRACE_RCUT, TOPOLOGIES, buildCluster, countEmbeddings, depth, hopLayers, nodes, reach, sampleEmbedding, topology, type Cluster, type Topology, type Vec3 } from '~/lib/models/grace';

// Graph ACE's cluster functions live on trees; local ACE (GRACE-1L) keeps only stars, a T-layer
// GRACE evaluates trees of depth ≤ T (src/lib/models/grace.ts). Every count and highlighted graph
// below is computed live from a perfect crystal: atoms are nodes, and an edge needs r < r_cut
// because the single-particle function on it vanishes beyond.

export interface CrystalInput {
  id: string;
  name: string;
  structure: Structure;
}

const SHOWN = ['11', '111', '12', '112', '122', '123'];
// Depth 3 needs complete neighbour lists up to 2 r_cut and atoms out to 3 r_cut.
const BALL = 3 * GRACE_RCUT + 0.2;

export default function GraceGraphs({ crystals }: { crystals: CrystalInput[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [crystal, setCrystal] = useState(crystals[0]!.id);
  const [topo, setTopo] = useState('123');
  const [rc, setRc] = useState(2.6);
  const [seed, setSeed] = useState(1);
  const cluster = useMemo(() => {
    const s = crystals.find((c) => c.id === crystal)!.structure;
    const ball: Vec3[] = [[0, 0, 0], ...shellAround(s, BALL).map((n) => n.p as Vec3)];
    return buildCluster(ball, GRACE_RCUT);
  }, [crystal, crystals]);
  const t = topology(topo);
  const counts = useMemo(() => new Map(TOPOLOGIES.map((x) => [x.id, countEmbeddings(cluster, x, rc)])), [cluster, rc]);
  const example = useMemo(() => sampleEmbedding(cluster, t, rc, seed), [cluster, t, rc, seed]);
  const hops = useMemo(() => hopLayers(cluster, rc, 3), [cluster, rc]);
  const reaches = useMemo(() => [1, 2, 3].map((T) => reach(cluster, rc, T)), [cluster, rc]);
  const name = crystals.find((c) => c.id === crystal)!.name;
  const d = depth(t);
  const span = example ? maxSpan(cluster, example) : 0;
  // Frame the atoms that are shown (those within d edges); re-frame when that extent changes by ≥ 1 Å.
  const extent = Math.max(3, Math.round(reaches[d - 1]!.rmax));
  const view = useMemo(() => {
    const k = 4.6 * extent;
    return { position: [0.55 * k, -0.72 * k, 0.42 * k] as [number, number, number], target: [0, 0, 0] as [number, number, number] };
  }, [extent]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={extent}
            background={theme.canvas}
            label={`${name}: a root atom and the atoms reachable within ${d} edge${d > 1 ? 's' : ''} shorter than ${rc.toFixed(2)} Å, coloured by how many edges they are from the root. One ${t.label} graph is highlighted with arrows from the root outwards.`}
            minDistance={6}
            maxDistance={160}
          >
            <Scene theme={theme} cluster={cluster} hops={hops} maxHop={d} example={example} topo={t} rc={rc} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Crystal">
              {crystals.map((c) => (
                <button key={c.id} type="button" aria-pressed={crystal === c.id} onClick={() => setCrystal(c.id)}>
                  {c.name}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.ink }} /> root
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> 1 edge
            </li>
            {d >= 2 ? (
              <li>
                <span className="dot" style={{ background: theme.phys }} /> 2 edges
              </li>
            ) : null}
            {d >= 3 ? (
              <li>
                <span className="dot" style={{ background: theme.lit }} /> 3 edges
              </li>
            ) : null}
            <li>
              <span className="bar" style={{ background: theme.fail }} /> highlighted graph
            </li>
          </ul>
        </div>
        <TopologyPicker theme={theme} counts={counts} selected={topo} onSelect={setTopo} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-rc`}>
            <span>edge cutoff r<sub>cut</sub></span>
            <input id={`${ids}-rc`} type="range" min={2.4} max={GRACE_RCUT} step={0.05} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(2)} Å</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" onClick={() => setSeed((s) => s + 1)} disabled={!example}>
              another {t.label} graph
            </button>{' '}
            <button type="button" className="lab-btn" onClick={() => setRc(GRACE_RCUT)}>
              GRACE’s 6 Å
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>{t.label} graphs on this root</dt>
            <dd>{(counts.get(topo) ?? 0).toLocaleString('en')}</dd>
          </div>
          <div className={d > 1 ? 'is-alert' : undefined}>
            <dt>needs</dt>
            <dd>{d === 1 ? 'GRACE-1L (a star)' : `${d} layers (GRACE-${d}L)`}</dd>
          </div>
          <div>
            <dt>highlighted: farthest pair</dt>
            <dd>{example ? `${span.toFixed(2)} Å` : '—'}</dd>
          </div>
          {reaches.map((r, k) => (
            <div key={k} className={k + 1 === d ? 'is-good' : undefined}>
              <dt>atoms within {k + 1} edge{k ? 's' : ''}</dt>
              <dd>
                {r.atoms.toLocaleString('en')} <small>to {r.rmax.toFixed(2)} Å</small>
              </dd>
            </div>
          ))}
        </dl>
        <p className="lab-note">
          A star’s nodes are all within r<sub>cut</sub> of the root, so two of them are at most 2r<sub>cut</sub> apart. A tree
          with the same number of nodes can stretch along a chain of short bonds, which is how two- and three-layer GRACE models
          see beyond one cutoff. With the first-neighbour cutoff in Si or ZnO (try 2.6 Å) a star has only its four bonds to work
          with; a (123) chain already reaches third neighbours. At 6 Å the counts explode, which is why the models never
          enumerate graphs: the layer recursion of Eqs. (5)–(6) sums over all of them at once, self-interactions included. Counts
          exclude repeated atoms.
        </p>
      </div>
    </div>
  );
}

function maxSpan(c: Cluster, atoms: number[]): number {
  let m = 0;
  for (const a of atoms)
    for (const b of atoms) {
      const p = c.positions[a]!;
      const q = c.positions[b]!;
      m = Math.max(m, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  return m;
}

function Scene({ theme, cluster, hops, maxHop, example, topo, rc }: { theme: SceneTheme; cluster: Cluster; hops: Int32Array; maxHop: number; example: number[] | null; topo: Topology; rc: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 110);
  const buffers = useMemo<AtomBuffers>(() => {
    const shown: number[] = [];
    hops.forEach((h, i) => {
      if (h >= 0 && h <= maxHop) shown.push(i);
    });
    const inGraph = new Set(example ?? []);
    const n = shown.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const layer = [linear(theme.ink), linear(theme.arch), linear(theme.phys), linear(theme.lit)];
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    const fail = linear(theme.fail);
    shown.forEach((i, k) => {
      pos.set(cluster.positions[i]!, 3 * k);
      const h = hops[i]!;
      if (i === 0) {
        radii[k] = 0.62;
        colors.set(layer[0]!, 3 * k);
      } else if (inGraph.has(i)) {
        radii[k] = 0.55;
        colors.set(fail, 3 * k);
      } else {
        radii[k] = example ? 0.3 : 0.38;
        colors.set(mixRGB(layer[h]!, stone, example ? 0.45 : 0.15), 3 * k);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.5), ao: new Float32Array(n).fill(1), count: n };
  }, [cluster, hops, maxHop, example, theme]);
  const arrows = useMemo<Segment[]>(() => {
    if (!example) return [];
    const c = linear(theme.fail);
    return topo.parent.map((p, k) => {
      const a = cluster.positions[example[p]!]!;
      const b = cluster.positions[example[k + 1]!]!;
      const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      // Stop short of the target atom so the head stays visible.
      const s = (len - 0.55) / len;
      return { a, b: [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])], color: c, radius: 0.08 };
    });
  }, [cluster, example, topo, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Arrows arrows={arrows} uniforms={uniforms} head={0.4} />
      <Shell center={[0, 0, 0]} radius={rc} color={theme.arch} opacity={0.1} />
    </>
  );
}

/** Small tree diagrams in the paper's style: root at the left, nodes placed by distance from it. */
function TopologyPicker({ theme, counts, selected, onSelect }: { theme: SceneTheme; counts: Map<string, number>; selected: string; onSelect: (id: string) => void }) {
  return (
    <div className="grace-topos" role="group" aria-label="Graph topology">
      <p className="grace-topos-head">
        Cluster graphs with 3–4 atoms, rooted at the centre. Stars (grey) are local ACE; the rest are trees.
      </p>
      {SHOWN.map((id) => {
        const t = topology(id);
        const d = depth(t);
        return (
          <button key={id} type="button" aria-pressed={selected === id} className={d === 1 ? 'is-star' : undefined} onClick={() => onSelect(id)}>
            <Glyph theme={theme} t={t} />
            <span className="grace-topo-label">
              {t.label}
              <small>
                {nodes(t)} atoms · {d === 1 ? 'star · 1L' : `depth ${d} · ${d}L`}
              </small>
            </span>
            <span className="grace-topo-count">{(counts.get(id) ?? 0).toLocaleString('en')}</span>
          </button>
        );
      })}
    </div>
  );
}

function Glyph({ theme, t }: { theme: SceneTheme; t: Topology }) {
  // Lay nodes out by depth (x) and spread siblings vertically (y).
  const dep = [0];
  t.parent.forEach((p, k) => (dep[k + 1] = dep[p]! + 1));
  const children = (i: number) => t.parent.map((p, k) => (p === i ? k + 1 : -1)).filter((k) => k >= 0);
  const y: number[] = [];
  let leaf = 0;
  const place = (i: number): number => {
    const ch = children(i);
    if (!ch.length) return (y[i] = leaf++);
    const ys = ch.map(place);
    return (y[i] = ys.reduce((a, b) => a + b, 0) / ys.length);
  };
  place(0);
  const W = 74;
  const H = 40;
  const px = (i: number) => 8 + dep[i]! * 22;
  const py = (i: number) => (leaf <= 1 ? H / 2 : 7 + (y[i]! / (leaf - 1)) * (H - 14));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-hidden="true">
      {t.parent.map((p, k) => (
        <line key={k} x1={px(p)} y1={py(p)} x2={px(k + 1)} y2={py(k + 1)} style={{ stroke: theme.inkMuted, strokeWidth: 1.4 }} />
      ))}
      {dep.map((_, i) => (
        <circle key={i} cx={px(i)} cy={py(i)} r={i === 0 ? 4.6 : 3.6} style={{ fill: i === 0 ? theme.ink : theme.arch }} />
      ))}
    </svg>
  );
}
