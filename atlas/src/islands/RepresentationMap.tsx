import { useEffect, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './RepresentationMap.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor } from '~/lib/structure/elements';
import { CLASS_LABELS, elementClass, nnOverlap, pairOf, projection, type ElementClass, type MapModel, type RepresentationMap as Data } from '~/lib/representation';

// Representation Map: real MACE latent features of 552 reference environments, precomputed on CPU by
// pipeline/atlas_pipeline/representation.py, shown as linked 2D maps (one per model) and a 3D view of the
// selected environment. Nothing is computed in the browser except selections and colours.

type ColourBy = 'class' | 'kind' | 'cn';
const SPACES = [
  { id: 'raw', label: 'Raw features' },
  { id: '3', label: 'Anchored, K = 3' },
  { id: '10', label: 'K = 10' },
  { id: '30', label: 'K = 30' },
  { id: '100', label: 'K = 100' },
];
const CLASS_ORDER: ElementClass[] = ['alkali', 'alkaline-earth', 'transition', 'p-metal', 'metalloid', 'nonmetal', 'halogen'];
const KIND_ORDER = ['crystal', 'surface', 'defect', 'molecule'] as const;
const R_MAX = 6; // Å, MACE-MP cutoff (read from the checkpoints)

function palette(theme: SceneTheme): string[] {
  return [theme.fail, theme.caveat, theme.arch, theme.lit, theme.phys, theme.inkMuted, theme.ink];
}

export default function RepresentationMap() {
  const theme = useSceneTheme();
  const [data, setData] = useState<Data | null>(null);
  const [space, setSpace] = useState('100');
  const [colourBy, setColourBy] = useState<ColourBy>('class');
  const [sel, setSel] = useState(0);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    import('~/data/representation/map.json').then((m) => {
      if (!live) return;
      const d = (m.default ?? m) as unknown as Data;
      setData(d);
      // Start on a rattled ZnO oxygen: an environment with a clear chemical identity.
      const start = d.environments.findIndex((e) => e.el === 'O' && d.structures[e.s]!.id.startsWith('ZnO-rattle'));
      setSel(Math.max(0, start));
    });
    return () => {
      live = false;
    };
  }, []);

  const colourOf = useMemo(() => {
    if (!data) return () => theme.inkMuted;
    const pal = palette(theme);
    return (e: number): string => {
      const env = data.environments[e]!;
      if (colourBy === 'class') return pal[CLASS_ORDER.indexOf(elementClass(env.el)!)]!;
      if (colourBy === 'kind') return pal[KIND_ORDER.indexOf(data.structures[env.s]!.kind)]!;
      const t = Math.min(1, env.cn / 12);
      return `color-mix(in oklab, ${theme.arch} ${Math.round(100 * t)}%, ${theme.caveat})`;
    };
  }, [data, colourBy, theme]);

  if (!data) return <div className="lab rep-loading">Loading 552 environments × 3 models…</div>;

  const env = data.environments[sel]!;
  const st = data.structures[env.s]!;
  const anchors = space === 'raw' ? [] : data.anchors[space]!;
  const [medium, small, random] = data.models as [MapModel, MapModel, MapModel];
  const show = hover ?? sel;
  const showEnv = data.environments[show]!;
  const showSt = data.structures[showEnv.s]!;

  return (
    <div className="lab rep">
      <div className="rep-bar">
        <div className="seg" role="group" aria-label="Feature space">
          {SPACES.map((s) => (
            <button key={s.id} type="button" aria-pressed={space === s.id} onClick={() => setSpace(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Colour by">
          <button type="button" aria-pressed={colourBy === 'class'} onClick={() => setColourBy('class')}>
            element class
          </button>
          <button type="button" aria-pressed={colourBy === 'kind'} onClick={() => setColourBy('kind')}>
            structure type
          </button>
          <button type="button" aria-pressed={colourBy === 'cn'} onClick={() => setColourBy('cn')}>
            coordination
          </button>
        </div>
      </div>
      <Legend theme={theme} colourBy={colourBy} />
      <div className="rep-maps">
        {data.models.map((m) => (
          <MapPanel
            key={m.id}
            data={data}
            model={m}
            space={space}
            sel={sel}
            hover={hover}
            anchors={anchors}
            colourOf={colourOf}
            theme={theme}
            onSelect={setSel}
            onHover={setHover}
          />
        ))}
      </div>
      <p className="rep-hint" aria-live="polite">
        {hover !== null ? 'Hovering: ' : 'Selected: '}
        <strong>{showEnv.el}</strong> in {showSt.title} {showEnv.mult > 1 ? `(${showEnv.mult} identical atoms)` : ''} · coordination {showEnv.cn}.
        Click a point to select it in all three maps; rings mark its ten nearest neighbours in each model’s raw feature space. Points beyond the 1st–99th percentile are drawn at the edge.
      </p>
      <div className="lab-split lab-split--wide">
        <div className="lab-stage lab-stage--short">
          <Stage
            view={viewFor(st)}
            resetKey={env.s}
            background={theme.canvas}
            label={`${st.title}: the selected ${env.el} atom is enlarged and outlined, with a ${R_MAX} Å sphere for one MACE interaction cutoff.`}
            minDistance={3}
            maxDistance={80}
          >
            <StructureScene theme={theme} data={data} sel={sel} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="ring" /> {R_MAX} Å: one interaction (two give 12 Å)
            </li>
          </ul>
        </div>
        <div className="rep-side">
          <h4 className="rep-h">Nearest neighbours of the selected environment</h4>
          {data.models.map((m) => (
            <div key={m.id} className="rep-nn">
              <p className="rep-nn-head">
                <span>{shortName(m)}</span>
                {m.id !== 'medium' ? <small>shares {Math.round(nnOverlap(medium, m, sel) * m.nn[sel]!.length)} of 10 with medium</small> : null}
              </p>
              <ul>
                {m.nn[sel]!.slice(0, 5).map((j) => {
                  const e = data.environments[j]!;
                  return (
                    <li key={j}>
                      <button type="button" onClick={() => setSel(j)}>
                        {e.el} · {data.structures[e.s]!.title}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
          <Agreement data={data} space={space} medium={medium} small={small} random={random} />
        </div>
      </div>
    </div>
  );
}

function shortName(m: MapModel): string {
  return m.trained ? m.name : 'Untrained MACE';
}

function viewFor(st: Data['structures'][number]) {
  const c = st.positions.reduce((a, p) => [a[0] + p[0] / st.positions.length, a[1] + p[1] / st.positions.length, a[2] + p[2] / st.positions.length], [0, 0, 0]);
  const ext = Math.max(...st.positions.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])), 3);
  const k = 3.2 * ext + 14;
  return { position: [c[0] + 0.6 * k, c[1] - 0.75 * k, c[2] + 0.5 * k] as [number, number, number], target: c as [number, number, number] };
}

function robustBounds(xy: [number, number][], skip: Set<number>) {
  const pts = xy.filter((_, i) => !skip.has(i));
  const q = (v: number[], f: number) => {
    const s = [...v].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.max(0, Math.round(f * (s.length - 1))))]!;
  };
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const b = { x0: q(xs, 0.01), x1: q(xs, 0.99), y0: q(ys, 0.01), y1: q(ys, 0.99) };
  const padX = 0.04 * (b.x1 - b.x0 || 1);
  const padY = 0.04 * (b.y1 - b.y0 || 1);
  return { x0: b.x0 - padX, x1: b.x1 + padX, y0: b.y0 - padY, y1: b.y1 + padY };
}

function Legend({ theme, colourBy }: { theme: SceneTheme; colourBy: ColourBy }) {
  const pal = palette(theme);
  const items =
    colourBy === 'class'
      ? CLASS_ORDER.map((c, i) => ({ label: CLASS_LABELS[c], colour: pal[i]! }))
      : colourBy === 'kind'
        ? KIND_ORDER.map((k, i) => ({ label: k, colour: pal[i]! }))
        : [
            { label: 'coordination 0', colour: theme.caveat },
            { label: '6', colour: `color-mix(in oklab, ${theme.arch} 50%, ${theme.caveat})` },
            { label: '12+', colour: theme.arch },
          ];
  return (
    <ul className="rep-legend" aria-label="Colour key">
      {items.map((it) => (
        <li key={it.label}>
          <span className="dot" style={{ background: it.colour }} /> {it.label}
        </li>
      ))}
      <li>
        <span className="diamond" /> anchor
      </li>
    </ul>
  );
}

function MapPanel({
  data,
  model,
  space,
  sel,
  hover,
  anchors,
  colourOf,
  theme,
  onSelect,
  onHover,
}: {
  data: Data;
  model: MapModel;
  space: string;
  sel: number;
  hover: number | null;
  anchors: number[];
  colourOf: (e: number) => string;
  theme: SceneTheme;
  onSelect: (e: number) => void;
  onHover: (e: number | null) => void;
}) {
  const p = projection(model, space);
  const W = 300;
  const H = 260;
  const pad = 12;
  // Bounds from the 1st to 99th percentile of non-anchor points: an anchor is maximally similar to itself,
  // so in a space where everything else is alike (the untrained model) anchors alone would set the scale.
  const b = useMemo(() => robustBounds(p.xy, new Set(anchors)), [p, anchors]);
  const sx = (W - 2 * pad) / (b.x1 - b.x0 || 1);
  const sy = (H - 2 * pad) / (b.y1 - b.y0 || 1);
  const s = Math.min(sx, sy);
  const ox = (W - s * (b.x1 - b.x0)) / 2;
  const oy = (H - s * (b.y1 - b.y0)) / 2;
  const X = (x: number) => ox + (Math.max(b.x0, Math.min(b.x1, x)) - b.x0) * s;
  const Y = (y: number) => H - (oy + (Math.max(b.y0, Math.min(b.y1, y)) - b.y0) * s);
  const nn = new Set(model.nn[sel]);
  return (
    <figure className="rep-map">
      <figcaption>
        <strong>{shortName(model)}</strong>
        <small>
          PC1 {Math.round(100 * p.var[0])}%, PC2 {Math.round(100 * p.var[1])}%
          {space !== 'raw' && model.id !== 'medium' ? ` · Procrustes vs medium ${p.procrustes!.toFixed(2)}` : ''}
        </small>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${model.name}: ${data.environments.length} environments projected to two dimensions (${space === 'raw' ? 'raw features' : `anchored with K = ${space}`}).`}
        onMouseLeave={() => onHover(null)}
      >
        <rect x={0} y={0} width={W} height={H} className="rep-bg" />
        {anchors.map((e) => {
          const [x, y] = p.xy[e]!;
          return <rect key={`a${e}`} x={X(x) - 5} y={Y(y) - 5} width={10} height={10} transform={`rotate(45 ${X(x)} ${Y(y)})`} className="rep-anchor" style={{ stroke: colourOf(e) }} onClick={() => onSelect(e)} />;
        })}
        {p.xy.map(([x, y], e) => (
            <circle
              key={e}
              cx={X(x)}
              cy={Y(y)}
              r={2.2 + 0.5 * Math.sqrt(data.environments[e]!.mult)}
              style={{ fill: colourOf(e), opacity: hover !== null && hover !== e ? 0.55 : 0.85 }}
              onMouseEnter={() => onHover(e)}
              onClick={() => onSelect(e)}
            />
        ))}
        {[...nn].map((e) => {
          const [x, y] = p.xy[e]!;
          return <circle key={`n${e}`} cx={X(x)} cy={Y(y)} r={5.5} className="rep-nn-ring" style={{ stroke: theme.ink }} />;
        })}
        {hover !== null ? <circle cx={X(p.xy[hover]![0])} cy={Y(p.xy[hover]![1])} r={7} className="rep-hover" style={{ stroke: theme.inkMuted }} /> : null}
        <circle cx={X(p.xy[sel]![0])} cy={Y(p.xy[sel]![1])} r={8} className="rep-sel" style={{ stroke: theme.fail }} />
      </svg>
    </figure>
  );
}

function Agreement({ data, space, medium, small, random }: { data: Data; space: string; medium: MapModel; small: MapModel; random: MapModel }) {
  const rows: Array<[MapModel, MapModel]> = [
    [medium, small],
    [medium, random],
    [small, random],
  ];
  const k = space === 'raw' ? '100' : space;
  return (
    <table className="rep-table">
      <thead>
        <tr>
          <th>pair</th>
          <th>CKA (raw)</th>
          <th>mutual 10-NN, raw</th>
          <th>mutual 10-NN, K = {k}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([a, b]) => {
          const p = pairOf(data, a.id, b.id)!;
          return (
            <tr key={`${a.id}-${b.id}`}>
              <td>
                {a.id} / {b.id}
              </td>
              <td className={a.trained && b.trained ? 'is-high' : undefined}>{p.cka.toFixed(2)}</td>
              <td>{p.mknn_raw.toFixed(2)}</td>
              <td>{p.mknn_anchored[k]!.toFixed(2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

const isOrthogonal = (c: NonNullable<Data['structures'][number]['cell']>) =>
  [c[0][1], c[0][2], c[1][0], c[1][2], c[2][0], c[2][1]].every((v) => Math.abs(v) < 1e-6);

function StructureScene({ theme, data, sel }: { theme: SceneTheme; data: Data; sel: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 90);
  const env = data.environments[sel]!;
  const st = data.structures[env.s]!;
  const centre = st.positions[env.i]!;
  const buffers = useMemo<AtomBuffers>(() => {
    const n = st.positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const bg = linear(theme.canvas);
    st.positions.forEach((p, k) => {
      pos.set(p, 3 * k);
      const d = Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
      radii[k] = k === env.i ? 0.62 : 0.4;
      const c = elementColor(st.symbols[k]!, theme.dark ? 'dark' : 'light');
      colors.set(k === env.i || d <= R_MAX ? c : mixRGB(c, bg, 0.6), 3 * k);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.25), ao: new Float32Array(n).fill(1), count: n };
  }, [st, env.i, centre, theme]);
  return (
    <>
      {st.cell && isOrthogonal(st.cell) ? <CellBox origin={[0, 0, 0]} size={[st.cell[0][0], st.cell[1][1], st.cell[2][2]]} color={theme.ruleStrong} /> : null}
      <ImpostorAtoms buffers={buffers} version={sel} uniforms={uniforms} />
      <Shell center={centre} radius={0.85} color={theme.fail} opacity={0.35} />
      <Shell center={centre} radius={R_MAX} color={theme.phys} opacity={0.06} />
    </>
  );
}
