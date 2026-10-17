import { useEffect, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './RepresentationMap.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor } from '~/lib/structure/elements';
import { CLASS_LABELS, elementClass, logPosition, nnOverlap, pairMatrix, projection, type ElementClass, type MapModel, type PairMetric, type RepresentationMap as Data } from '~/lib/representation';

// Representation Map: real latent features of six models (three MACE, CHGNet, SevenNet, Orb) for 562 reference
// environments, precomputed on CPU by pipeline/atlas_pipeline/representation.py, shown as linked 2D maps (one
// per model), an agreement matrix, an invariance strip and a 3D view of the selected environment. Nothing is
// computed in the browser except selections and colours.

type ColourBy = 'class' | 'kind' | 'cn' | 'rot';
// Log axis of the invariance strip (10^-15 … 10^2 d_ref: float64 rounding to Orb's mirror outliers) and of the
// 'rotation drift' colouring (10^-6 … 10^1: float32 rounding stays grey).
const LOG_LO = -15;
const LOG_HI = 2;
const COL_LO = -6;
const COL_HI = 1;
const SPACES = [
  { id: 'raw', label: 'Raw features' },
  { id: '3', label: 'Anchored, K = 3' },
  { id: '10', label: 'K = 10' },
  { id: '30', label: 'K = 30' },
  { id: '100', label: 'K = 100' },
];
const CLASS_ORDER: ElementClass[] = ['alkali', 'alkaline-earth', 'transition', 'p-metal', 'metalloid', 'nonmetal', 'halogen'];
const KIND_ORDER = ['crystal', 'surface', 'defect', 'molecule'] as const;
const R_MAX = 6; // Å: MACE-MP, CHGNet 0.3.0 and Orb v2 atom-graph cutoff (SevenNet-0 uses 5 Å)

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
    return (e: number, m: MapModel): string => {
      const env = data.environments[e]!;
      if (colourBy === 'rot') {
        const t = logPosition(m.invariance.rot_env[e]!, COL_LO, COL_HI);
        return `color-mix(in oklab, ${theme.fail} ${Math.round(100 * t)}%, ${theme.ruleStrong})`;
      }
      if (colourBy === 'class') return pal[CLASS_ORDER.indexOf(elementClass(env.el)!)]!;
      if (colourBy === 'kind') return pal[KIND_ORDER.indexOf(data.structures[env.s]!.kind)]!;
      const t = Math.min(1, env.cn / 12);
      return `color-mix(in oklab, ${theme.arch} ${Math.round(100 * t)}%, ${theme.caveat})`;
    };
  }, [data, colourBy, theme]);

  if (!data) return <div className="lab rep-loading">Loading 562 environments × 6 models…</div>;

  const env = data.environments[sel]!;
  const st = data.structures[env.s]!;
  const anchors = space === 'raw' ? [] : data.anchors[space]!;
  const medium = data.models[0]!;
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
          <button type="button" aria-pressed={colourBy === 'rot'} onClick={() => setColourBy('rot')}>
            rotation drift
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
        Click a point to select it in all six maps; rings mark its ten nearest neighbours in each model’s raw feature space. Points beyond the 1st–99th percentile are drawn at the edge.
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
                {m.nn[sel]!.slice(0, 3).map((j) => {
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
        </div>
      </div>
      <div className="rep-lower">
        <Agreement data={data} space={space} theme={theme} />
        <Invariance data={data} sel={sel} theme={theme} onSelect={setSel} />
      </div>
    </div>
  );
}

const SHORT: Record<string, string> = { medium: 'MACE medium', small: 'MACE small', random: 'Untrained MACE', chgnet: 'CHGNet', sevennet: 'SevenNet-0', orb: 'Orb v2' };
function shortName(m: MapModel): string {
  return SHORT[m.id] ?? m.name;
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
        : colourBy === 'cn'
          ? [
              { label: 'coordination 0', colour: theme.caveat },
              { label: '6', colour: `color-mix(in oklab, ${theme.arch} 50%, ${theme.caveat})` },
              { label: '12+', colour: theme.arch },
            ]
          : [
              { label: `moves ≤ 10⁻⁶ d_ref under rotation`, colour: theme.ruleStrong },
              { label: '10⁻³ d_ref', colour: `color-mix(in oklab, ${theme.fail} ${Math.round(100 * logPosition(1e-3, COL_LO, COL_HI))}%, ${theme.ruleStrong})` },
              { label: '≥ 10 d_ref (each map coloured by its own model)', colour: theme.fail },
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
  colourOf: (e: number, m: MapModel) => string;
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
          {space !== 'raw' && model.id !== 'medium' ? ` · Procrustes vs MACE medium ${p.procrustes!.toFixed(2)}` : ` · ${model.dim} features`}
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
          return <rect key={`a${e}`} x={X(x) - 5} y={Y(y) - 5} width={10} height={10} transform={`rotate(45 ${X(x)} ${Y(y)})`} className="rep-anchor" style={{ stroke: colourOf(e, model) }} onClick={() => onSelect(e)} />;
        })}
        {p.xy.map(([x, y], e) => (
            <circle
              key={e}
              cx={X(x)}
              cy={Y(y)}
              r={2.2 + 0.5 * Math.sqrt(data.environments[e]!.mult)}
              style={{ fill: colourOf(e, model), opacity: hover !== null && hover !== e ? 0.55 : 0.85 }}
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

const METRICS: Array<{ id: 'cka' | 'mknn_raw' | 'mknn_k'; label: string }> = [
  { id: 'cka', label: 'linear CKA' },
  { id: 'mknn_raw', label: 'mutual 10-NN, raw' },
  { id: 'mknn_k', label: 'mutual 10-NN, anchored' },
];

function Agreement({ data, space, theme }: { data: Data; space: string; theme: SceneTheme }) {
  const [metric, setMetric] = useState<'cka' | 'mknn_raw' | 'mknn_k'>('cka');
  const k = space === 'raw' ? '100' : space;
  const key: PairMetric = metric === 'mknn_k' ? `mknn_${k}` : metric;
  const M = pairMatrix(data, key);
  const n = data.models.length;
  const C = 34;
  const L = 98;
  const T = 74;
  return (
    <section className="rep-panel" aria-labelledby="rep-agree-h">
      <h4 className="rep-h" id="rep-agree-h">
        Agreement between models
      </h4>
      <div className="seg" role="group" aria-label="Agreement metric">
        {METRICS.map((m) => (
          <button key={m.id} type="button" aria-pressed={metric === m.id} onClick={() => setMetric(m.id)}>
            {m.id === 'mknn_k' ? `${m.label}, K = ${k}` : m.label}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${L + n * C + 40} ${T + n * C + 4}`} className="rep-matrix" role="img" aria-label={`Matrix of ${key} between the ${n} models.`}>
        {data.models.map((m, i) => (
          <g key={m.id}>
            <text x={L - 6} y={T + i * C + C / 2 + 4} textAnchor="end" className="rep-mx-label">
              {shortName(m)}
            </text>
            <text x={0} y={0} transform={`translate(${L + i * C + C / 2 + 4} ${T - 6}) rotate(-50)`} className="rep-mx-label">
              {shortName(m)}
            </text>
          </g>
        ))}
        {M.map((row, i) =>
          row.map((v, j) => (
            <g key={`${i}-${j}`}>
              <rect
                x={L + j * C + 1}
                y={T + i * C + 1}
                width={C - 2}
                height={C - 2}
                rx={3}
                style={{ fill: i === j ? theme.ruleStrong : `color-mix(in oklab, ${theme.arch} ${Math.round(100 * Math.max(0, v))}%, ${theme.canvas})` }}
              >
                <title>{`${data.models[i]!.name} / ${data.models[j]!.name}: ${v.toFixed(3)}`}</title>
              </rect>
              {i !== j ? (
                <text x={L + j * C + C / 2} y={T + i * C + C / 2 + 4} textAnchor="middle" className="rep-mx-val" style={{ fill: v > 0.55 ? theme.canvas : theme.ink }}>
                  {v.toFixed(2).replace(/^0/, '')}
                </text>
              ) : null}
            </g>
          )),
        )}
      </svg>
      <p className="rep-note">
        {metric === 'cka'
          ? 'Linear CKA of the raw features (Eq. 2): unchanged by rotating or scaling either space; 1 for identical geometry.'
          : metric === 'mknn_raw'
            ? 'Fraction of each environment’s ten nearest neighbours the two models share, in raw feature space.'
            : `The same overlap after the anchor transform of Eq. (1) with K = ${k} anchors (follows the feature-space buttons above).`}
      </p>
    </section>
  );
}

function Invariance({ data, sel, theme, onSelect }: { data: Data; sel: number; theme: SceneTheme; onSelect: (e: number) => void }) {
  const [test, setTest] = useState<'rot' | 'ref' | 'equiv'>('rot');
  const W = 460;
  const L = 90;
  const R = 12;
  const ROW = 30;
  const T = 8;
  const H = T + data.models.length * ROW + 30;
  const X = (v: number) => L + logPosition(v, LOG_LO, LOG_HI) * (W - L - R);
  const ticks = [];
  for (let k = LOG_LO; k <= LOG_HI; k++) if (k % 3 === 0 || k === LOG_HI) ticks.push(k);
  const rot = data.source.rotation;
  return (
    <section className="rep-panel" aria-labelledby="rep-inv-h">
      <h4 className="rep-h" id="rep-inv-h">
        Do environments that should not move, move?
      </h4>
      <div className="seg" role="group" aria-label="Invariance test">
        <button type="button" aria-pressed={test === 'rot'} onClick={() => setTest('rot')}>
          rigid rotation ({rot.angle_deg}°)
        </button>
        <button type="button" aria-pressed={test === 'ref'} onClick={() => setTest('ref')}>
          mirror image
        </button>
        <button type="button" aria-pressed={test === 'equiv'} onClick={() => setTest('equiv')}>
          symmetry-equivalent atoms
        </button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="rep-strip" role="img" aria-label="Feature displacement per environment and model, on a log scale in units of the median nearest-neighbour distance.">
        {ticks.map((k) => (
          <g key={k}>
            <line x1={X(10 ** k)} x2={X(10 ** k)} y1={T} y2={H - 22} className={k === 0 ? 'rep-strip-one' : 'rep-strip-grid'} />
            <text x={X(10 ** k)} y={H - 8} textAnchor="middle" className="rep-mx-label">
              {k === 0 ? '1' : k === 1 ? '10' : k === 2 ? '100' : `1e${k}`}
            </text>
          </g>
        ))}
        {data.models.map((m, r) => {
          const vals = test === 'rot' ? m.invariance.rot_env : test === 'ref' ? m.invariance.ref_env : m.invariance.equiv_env;
          const y = T + r * ROW + ROW / 2;
          const summ = m.invariance[test];
          return (
            <g key={m.id}>
              <text x={L - 8} y={y + 4} textAnchor="end" className="rep-mx-label">
                {shortName(m)}
              </text>
              {vals.map((v, e) =>
                test === 'equiv' && data.environments[e]!.mult === 1 ? null : (
                  <circle
                    key={e}
                    cx={X(v)}
                    cy={y + (((e * 7919) % 13) - 6) * 0.9}
                    r={2}
                    style={{ fill: e === sel ? theme.fail : theme.arch, opacity: e === sel ? 1 : 0.4 }}
                    onClick={() => onSelect(e)}
                  >
                    <title>{`${data.environments[e]!.el} in ${data.structures[data.environments[e]!.s]!.title}: ${v.toExponential(1)} d_ref`}</title>
                  </circle>
                ),
              )}
              {summ.n > 0 ? <line x1={X(summ.median)} x2={X(summ.median)} y1={y - 9} y2={y + 9} className="rep-strip-med" style={{ stroke: theme.ink }} /> : null}
              <circle cx={X(vals[sel]!)} cy={y} r={5} className="rep-sel" style={{ stroke: theme.fail, fill: 'none' }} />
            </g>
          );
        })}
      </svg>
      <p className="rep-note">
        {test === 'rot'
          ? `Each dot is one environment: how far its feature vector moves when the whole structure (and cell) is rotated by ${rot.angle_deg}° about (${rot.axis.join(', ')}). `
          : test === 'ref'
            ? 'Each dot is one environment: how far its feature vector moves when the structure is replaced by its mirror image (z → −z). Distances are unchanged; only handedness flips. '
            : 'Each dot is one environment with symmetry-equivalent copies in its structure (identical neighbour lists within 25 Å, mirror images included): the largest distance between their feature vectors. '}
        Unit d_ref: the median distance from an environment to its nearest distinct one in the same model. Black tick: median; red ring:
        the selected environment. Near 10⁻¹³: float64 rounding (MACE); near 10⁻⁵: float32 rounding (CHGNet, SevenNet). Exact zeros sit at the left edge. Clicking a dot selects it.
      </p>
    </section>
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
