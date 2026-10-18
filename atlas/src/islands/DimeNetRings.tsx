import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dimenet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, linear, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { forward, hexagonAndTriangles, hexagonSecondNeighbour, makeToyDimeNet, relDiff, ring, type Vec3 } from '~/lib/models/dimenet';
import { forward as schnetForward, makeToySchNet } from '~/lib/models/schnet';
import { ZeroLaneChart } from './dimenet/ZeroLaneChart';

// DimeNet App. A, Fig. 6: a hexagon and two triangles with the same bond length. Below the
// hexagon's second-neighbour distance d√3 every atom in both has two neighbours at d, so a
// distance-only network (here a toy SchNet) gives identical features at any depth; DimeNet's
// messages carry the 120° vs 60° angle from the first interaction block.

const D = 1.5;
const RC = Array.from({ length: 50 }, (_, k) => 1.55 + 0.05 * k);
const Z6 = [6, 6, 6, 6, 6, 6];

function scan() {
  const { hexagon, triangles } = hexagonAndTriangles(D);
  const out = { schnet1: [] as number[], schnet3: [] as number[], dime1: [] as number[], dime3: [] as number[] };
  for (const rc of RC) {
    const s = makeToySchNet({ rc, elements: [6], blocks: 3 });
    const sa = schnetForward(s, Z6, hexagon);
    const sb = schnetForward(s, Z6, triangles);
    out.schnet1.push(relDiff(sa[1]![0]!, sb[1]![0]!));
    out.schnet3.push(relDiff(sa[3]![0]!, sb[3]![0]!));
    const m = makeToyDimeNet({ rc, elements: [6], blocks: 3 });
    const a = forward(m, Z6, hexagon);
    const b = forward(m, Z6, triangles);
    out.dime1.push(relDiff(a.atomFeat[1]![0]!, b.atomFeat[1]![0]!));
    out.dime3.push(relDiff(a.atomFeat[3]![0]!, b.atomFeat[3]![0]!));
  }
  return out;
}

export default function DimeNetRings() {
  const theme = useSceneTheme();
  const ids = useId();
  const [rc, setRc] = useState(2.2);
  const data = useMemo(scan, []);
  const d2 = hexagonSecondNeighbour(D);
  const same = rc < d2;
  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [0, -4, 22], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Left: six atoms in a hexagon; right: two triangles, all bonds ${D} Å. With a ${rc.toFixed(2)} Å cutoff the graph ${same ? 'gives every atom exactly two neighbours at the bond length in both structures' : 'also links second neighbours across the hexagon'}.`}
            minDistance={8}
            maxDistance={50}
          >
            <RingScene theme={theme} rc={rc} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar thin" style={{ background: theme.ink }} /> edges in the graph (r &lt; r<sub>c</sub>)
            </li>
            <li>angles: 120° (hexagon) vs 60° (triangles)</li>
            <li>triangles drawn together, computed 40 Å apart</li>
          </ul>
        </div>
        <ZeroLaneChart
          ariaLabel={`Relative difference between an atom's features in the hexagon and in the triangles, against the cutoff. The toy SchNet curves are exactly zero below ${d2.toFixed(2)} Å; the toy DimeNet curves are non-zero from the first block.`}
          series={[
            { key: 's1', label: 'SchNet, 1 block', color: theme.fail, xs: RC, ys: data.schnet1, width: 3.4 },
            { key: 's3', label: 'SchNet, 3 blocks', color: theme.caveat, xs: RC, ys: data.schnet3, dashed: true },
            { key: 'd1', label: 'DimeNet, 1 block', color: theme.arch, xs: RC, ys: data.dime1 },
            { key: 'd3', label: 'DimeNet, 3 blocks', color: theme.phys, xs: RC, ys: data.dime3, dashed: true },
          ]}
          xMin={1.5}
          xMax={4.05}
          xTicks={[1.5, 2, 2.5, 3, 3.5, 4]}
          xFmt={(v) => `${v} Å`}
          xLabel="cutoff r_c"
          yLabel="‖h(hexagon) − h(triangles)‖ / ‖h(triangles)‖ (log)"
          logLo={-8}
          logHi={0}
          cursor={rc}
          bands={[{ from: 1.5, to: d2, label: 'r_c < d√3', tone: 'fail' }]}
        />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-rc`}>
            <span>
              cutoff r<sub>c</sub>
            </span>
            <input id={`${ids}-rc`} type="range" min={1.55} max={4} step={0.05} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>hexagon second neighbour d√3</dt>
            <dd>{d2.toFixed(2)} Å</dd>
          </div>
          <div className={same ? 'is-alert' : 'is-good'}>
            <dt>distance-only graphs identical</dt>
            <dd>{same ? 'yes' : 'no'}</dd>
          </div>
          <div className="is-good">
            <dt>DimeNet triplets per atom</dt>
            <dd>{same ? '2 in both (α = 120° vs 60°)' : 'more in the hexagon'}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Below d√3 every atom, in both structures, sees two carbon neighbours at 1.5 Å and nothing else, so the multiset of
          distances a SchNet block sums over is the same, and so is everything built from it at any depth (red and orange
          on the zero lane). DimeNet’s message from j to i is updated by the message from k to j together with the angle at
          j: 120° in the hexagon, 60° in a triangle, so the features differ from the first interaction block. The
          embedding block alone sees only distances and agrees in both, as the test suite checks. Small values just above
          1.5 Å come from the envelope, which is close to zero when the bonds sit near the cutoff.
        </p>
      </div>
    </div>
  );
}

function RingScene({ theme, rc }: { theme: SceneTheme; rc: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 18, 60);
  const hex = useMemo(() => ring(6, D, [-3, 0, 0]), []);
  const tris = useMemo(() => [...ring(3, D, [2.6, 1.3, 0], Math.PI / 2), ...ring(3, D, [2.6, -1.9, 0], Math.PI / 2)], []);
  const atoms = useMemo<AtomBuffers>(() => {
    const all = [...hex, ...tris];
    const p = new Float32Array(all.length * 3);
    const colors = new Float32Array(all.length * 3);
    all.forEach((v, k) => {
      p.set(v, 3 * k);
      colors.set(linear(k < 6 ? theme.arch : theme.phys), 3 * k);
    });
    return { positions: p, radii: new Float32Array(all.length).fill(0.34), colors, metal: new Float32Array(all.length), ao: new Float32Array(all.length).fill(1), count: all.length };
  }, [hex, tris, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const ink = linear(theme.ink);
    const out: ColoredLine[] = [];
    const add = (set: Vec3[], groups: number) => {
      for (let a = 0; a < set.length; a++)
        for (let b = a + 1; b < set.length; b++) {
          if (Math.floor(a / groups) !== Math.floor(b / groups)) continue;
          const r = Math.hypot(set[a]![0] - set[b]![0], set[a]![1] - set[b]![1], set[a]![2] - set[b]![2]);
          if (r < rc) out.push({ a: set[a] as [number, number, number], b: set[b] as [number, number, number], rgba: [ink[0], ink[1], ink[2], r > D + 0.01 ? 0.45 : 1] });
        }
    };
    add(hex, 6);
    add(tris, 3);
    return out;
  }, [hex, tris, rc, theme]);
  return (
    <>
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
    </>
  );
}
