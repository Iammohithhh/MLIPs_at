import { useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import {
  AtomPicker,
  ImpostorAtoms,
  Lines,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { STRATA, STRATUM_GAP, type StarNode } from '~/lib/constellation';

// SCHEMATIC: the record graph in 3D. Position comes from the deterministic 2D layout; height from the
// kind of record (physics → models and data → benchmarks and sources → evidence and use).

const TONE: Record<string, keyof SceneTheme> = {
  phys: 'phys',
  arch: 'arch',
  lit: 'lit',
  fail: 'fail',
  neutral: 'inkMuted',
};

export default function Constellation({
  nodes,
  edges,
  tones,
}: {
  nodes: StarNode[];
  edges: Array<{ source: string; target: string }>;
  tones: Record<string, { label: string; tone: string }>;
}) {
  const theme = useSceneTheme();
  const [sel, setSel] = useState<number | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const index = useMemo(() => new Map(nodes.map((n, i) => [n.id, i])), [nodes]);
  const neighbours = useMemo(() => {
    const m = new Map<number, number[]>();
    for (const e of edges) {
      const a = index.get(e.source);
      const b = index.get(e.target);
      if (a == null || b == null) continue;
      m.set(a, [...(m.get(a) ?? []), b]);
      m.set(b, [...(m.get(b) ?? []), a]);
    }
    return m;
  }, [edges, index]);
  const visible = (i: number) => !hidden.has(nodes[i]!.stratum);
  const selNode = sel != null ? nodes[sel] : null;
  const near = sel != null ? (neighbours.get(sel) ?? []).filter(visible) : [];

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage" style={{ minHeight: 520 }}>
          <Stage
            view={{ position: [150, -235, 120], target: [0, 0, 4] }}
            background={theme.canvas}
            label={`The Atlas record graph in 3D: ${nodes.length} records in four strata.`}
            minDistance={30}
            maxDistance={400}
          >
            <Scene
              theme={theme}
              nodes={nodes}
              edges={edges}
              tones={tones}
              index={index}
              sel={sel}
              setSel={setSel}
              near={near}
              visible={visible}
              hidden={hidden}
            />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Strata">
              {STRATA.map((s, k) => (
                <button
                  key={s.name}
                  type="button"
                  aria-pressed={!hidden.has(k)}
                  onClick={() =>
                    setHidden((h) => {
                      const n = new Set(h);
                      if (n.has(k)) n.delete(k);
                      else n.add(k);
                      return n;
                    })
                  }
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            {(['phys', 'arch', 'lit', 'fail', 'neutral'] as const).map((t) => (
              <li key={t}>
                <span className="dot" style={{ background: theme[TONE[t]!] as string }} />{' '}
                {Object.values(tones)
                  .filter((x) => x.tone === t)
                  .map((x) => x.label)
                  .join(', ')}
              </li>
            ))}
          </ul>
        </div>
        <aside className="lab-panel" style={{ margin: 0, alignContent: 'start' }} aria-live="polite">
          {selNode ? (
            <>
              <p className="eyebrow">
                {tones[selNode.collection]?.label} · {selNode.status}
              </p>
              <h3
                style={{
                  margin: '0.3rem 0 0.6rem',
                  font: '600 1.15rem/1.35 var(--font-serif)',
                }}
              >
                <a href={selNode.url}>{selNode.title}</a>
              </h3>
              <p className="lab-note">
                {near.length} linked record{near.length === 1 ? '' : 's'} in view:
              </p>
              <ul
                style={{
                  margin: 0,
                  paddingLeft: '1.1rem',
                  font: '0.88rem/1.5 var(--font-sans)',
                  maxHeight: 360,
                  overflowY: 'auto',
                }}
              >
                {near.slice(0, 40).map((j) => (
                  <li key={j}>
                    <button
                      type="button"
                      onClick={() => setSel(j)}
                      style={{
                        all: 'unset',
                        cursor: 'pointer',
                        color: 'var(--atlas-arch)',
                        textDecoration: 'underline dotted',
                      }}
                    >
                      {nodes[j]!.title.slice(0, 90)}
                    </button>{' '}
                    <span style={{ color: 'var(--atlas-ink-faint)' }}>({tones[nodes[j]!.collection]?.label})</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <>
              <p className="lab-note">
                Every record of the Atlas is a star; every reference between records is a line. Physics sits at the bottom, models and data
                above it, then benchmarks and papers, and the claims, failure cases and applications built on them at the top. Click a star
                to see what it is connected to; drag to rotate.
              </p>
              <dl className="lab-readout">
                {STRATA.map((st, k) => (
                  <div key={st.name}>
                    <dt>{st.name}</dt>
                    <dd>{nodes.filter((n) => n.stratum === k).length} records</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

function Scene({
  theme,
  nodes,
  edges,
  tones,
  index,
  sel,
  setSel,
  near,
  visible,
  hidden,
}: {
  theme: SceneTheme;
  nodes: StarNode[];
  tones: Record<string, { label: string; tone: string }>;
  edges: Array<{ source: string; target: string }>;
  index: Map<string, number>;
  sel: number | null;
  setSel: (i: number | null) => void;
  near: number[];
  visible: (i: number) => boolean;
  hidden: Set<number>;
}) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 220, 900);
  const nearSet = useMemo(() => new Set(near), [near]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = nodes.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const bg = linear(theme.canvas);
    nodes.forEach((nd, i) => {
      pos.set(nd.p, 3 * i);
      const on = visible(i);
      radii[i] = on ? nd.r * (sel === i ? 1.6 : 1) : 0.001;
      const tone = theme[TONE[tones[nd.collection]?.tone ?? 'neutral'] ?? 'inkMuted'] as string;
      const c = linear(tone);
      const dim = sel != null && sel !== i && !nearSet.has(i);
      colors.set(dim ? mixRGB(c, bg, 0.7) : c, 3 * i);
    });
    return {
      positions: pos,
      radii,
      colors,
      metal: new Float32Array(n),
      ao: new Float32Array(n).fill(1),
      count: n,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, sel, nearSet, theme, hidden, tones]);
  const lines = useMemo<ColoredLine[]>(() => {
    const out: ColoredLine[] = [];
    const faint = linear(theme.inkMuted);
    const hot = linear(theme.ink);
    for (const e of edges) {
      const a = index.get(e.source);
      const b = index.get(e.target);
      if (a == null || b == null || !visible(a) || !visible(b)) continue;
      const on = sel != null && (a === sel || b === sel);
      if (sel != null && !on) continue;
      out.push({
        a: nodes[a]!.p,
        b: nodes[b]!.p,
        rgba: on ? [hot[0], hot[1], hot[2], 0.8] : [faint[0], faint[1], faint[2], 0.12],
      });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edges, index, nodes, sel, theme, hidden]);
  const z0 = ((STRATA.length - 1) * STRATUM_GAP) / 2;
  // Strata as outlines only: filled planes wash out the stars behind them.
  const planes = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.inkMuted);
    const h = 55;
    const out: ColoredLine[] = [];
    STRATA.forEach((_, k) => {
      if (hidden.has(k)) return;
      const z = k * STRATUM_GAP - z0 - 4;
      const cs: Array<[number, number]> = [
        [-h, -h],
        [h, -h],
        [h, h],
        [-h, h],
      ];
      cs.forEach((p, i) => {
        const q = cs[(i + 1) % 4]!;
        out.push({ a: [p[0], p[1], z], b: [q[0], q[1], z], rgba: [c[0], c[1], c[2], 0.35] });
      });
    });
    return out;
  }, [hidden, theme.inkMuted, z0]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
      <Lines lines={planes} />
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={buffers.count} onPick={(i) => setSel(i)} />
    </>
  );
}
