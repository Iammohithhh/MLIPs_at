import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './MaceWalkthrough.css';
import './XrayExplorer.css';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  AtomPicker,
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
import {
  CHANNELS,
  channelMatrix,
  channelValues,
  channelVectors,
  dataflowShapes,
  energyParts,
  incomingEdges,
  mostVariedChannel,
  rotatedView,
  rotationCheck,
  tensorLs,
  type TensorId,
  type Vec3,
  type XrayDoc,
  type XrayGeometry,
} from '~/lib/models/xray';

// Architecture X-ray: the real intermediate tensors of one MACE-MP-0a forward pass,
// precomputed by the Atlas pipeline. Nothing here is a toy: every colour, arrow and bar
// is read from the stored tensors. Equations are typeset at build time and passed in.

export interface XrayStep {
  id: TensorId;
  title: string;
  /** Pre-rendered KaTeX HTML. */
  eq: string;
  what: string;
  code: string;
}

const EDGE_STEPS: TensorId[] = ['radial', 'sh'];
const SIGNED = (id: TensorId, l: number) => l === 0 && (id === 'embed' || id === 'h1' || id === 'h2');

function colorOf(sym: string, dark: boolean): [number, number, number] {
  const c = elementColor(sym, dark ? 'dark' : 'light');
  return hexToLinearRgb('#' + c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join(''));
}

export interface SceneAtom {
  /** Parent atom in the structure (periodic images share their parent's features). */
  parent: number;
  sym: string;
  /** Position relative to the view centre (Å). */
  p: Vec3;
  /** Incoming edge into the centre atom, for edge tensors. */
  edge: number | null;
}

/** Atoms to draw: a molecule whole; a crystal as the centre atom and its neighbours within r_max. */
export function sceneAtoms(sys: XrayGeometry, centre: number): SceneAtom[] {
  if (!sys.cell) {
    const c = sys.positions.reduce<Vec3>((a, p) => [a[0] + p[0] / sys.n, a[1] + p[1] / sys.n, a[2] + p[2] / sys.n], [0, 0, 0]);
    const inc = new Map(incomingEdges(sys, centre).map((e) => [e.j, e.e]));
    return sys.positions.map((p, i) => ({ parent: i, sym: sys.symbols[i]!, p: [p[0] - c[0], p[1] - c[1], p[2] - c[2]], edge: inc.get(i) ?? null }));
  }
  const pc = sys.positions[centre]!;
  const rel = (q: Vec3): Vec3 => [q[0] - pc[0], q[1] - pc[1], q[2] - pc[2]];
  return [
    { parent: centre, sym: sys.symbols[centre]!, p: [0, 0, 0], edge: null },
    ...incomingEdges(sys, centre).map((e) => ({ parent: e.j, sym: sys.symbols[e.j]!, p: rel(e.from), edge: e.e })),
  ];
}

export const fmt = (x: number, d = 3) => (Math.abs(x) < 5e-13 ? '0' : Math.abs(x) >= 1e-3 && Math.abs(x) < 1e4 ? x.toFixed(d) : x.toExponential(2));

export default function XrayExplorer({ steps }: { steps: XrayStep[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [doc, setDoc] = useState<XrayDoc | null>(null);
  const [failed, setFailed] = useState(false);
  const [sysId, setSysId] = useState('ethanol');
  const [stepId, setStepId] = useState<TensorId>('h1');
  const [l, setL] = useState(1);
  const [channel, setChannel] = useState<number | null>(null);
  const [atom, setAtom] = useState(1);
  const [rotated, setRotated] = useState(false);
  const [n, setN] = useState(1);
  const [shK, setShK] = useState(2); // k = l² + l + m; 2 → (l = 1, m = 0)

  useEffect(() => {
    let live = true;
    import('~/data/xray/mace-mp-0a-medium.json')
      .then((m) => live && setDoc(m.default as unknown as XrayDoc))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);

  if (failed) return <p className="stage-fallback">The precomputed tensors could not be loaded.</p>;
  if (!doc) return <div className="lab xr xr-loading" aria-busy="true">Loading the precomputed forward pass…</div>;
  return (
    <XrayInner
      doc={doc}
      steps={steps}
      theme={theme}
      ids={ids}
      state={{ sysId, stepId, l, channel, atom, rotated, n, shK }}
      set={{ setSysId, setStepId, setL, setChannel, setAtom, setRotated, setN, setShK }}
    />
  );
}

interface State {
  sysId: string;
  stepId: TensorId;
  l: number;
  channel: number | null;
  atom: number;
  rotated: boolean;
  n: number;
  shK: number;
}

function XrayInner({
  doc,
  steps,
  theme,
  ids,
  state,
  set,
}: {
  doc: XrayDoc;
  steps: XrayStep[];
  theme: SceneTheme;
  ids: string;
  state: State;
  set: {
    setSysId: (v: string) => void;
    setStepId: (v: TensorId) => void;
    setL: (v: number) => void;
    setChannel: (v: number | null) => void;
    setAtom: (v: number) => void;
    setRotated: (v: boolean) => void;
    setN: (v: number) => void;
    setShK: (v: number) => void;
  };
}) {
  const { sysId, stepId, atom, n, shK } = state;
  const base = doc.systems.find((s) => s.id === sysId) ?? doc.systems[0]!;
  const canRotate = !!base.rotated;
  const isEdge = EDGE_STEPS.includes(stepId);
  const rotated = state.rotated && canRotate && !isEdge;
  const sys = rotated ? rotatedView(base)! : base;
  const step = steps.find((s) => s.id === stepId)!;
  const shapes = dataflowShapes(doc, base);
  const hp = doc.model.hyperparameters;

  const ls = tensorLs(base, stepId);
  const l = ls.some((b) => b.l === state.l && b.live) ? state.l : 0;
  const channel = state.channel ?? (ls.length ? mostVariedChannel(base, stepId, l) : 0);
  const centre = Math.min(atom, base.n - 1);

  const values = ls.length ? channelValues(sys, stepId, l, channel) : null;
  const vectors = l === 1 ? channelVectors(sys, stepId, channel) : null;
  const matrix = useMemo(() => (ls.length ? channelMatrix(sys, stepId, l) : null), [sys, stepId, l, ls.length]);
  const parts = energyParts(doc, sys);
  const check = useMemo(() => rotationCheck(base), [base]);

  const atoms = useMemo(() => sceneAtoms(base, centre), [base, centre]);
  const shown = useMemo<SceneAtom[]>(() => {
    if (!rotated || !base.rotated) return atoms;
    // Rotated pass (molecule only): same atoms, rotated positions about the same centroid.
    const P = base.rotated.positions;
    const c = P.reduce<Vec3>((a, p) => [a[0] + p[0] / base.n, a[1] + p[1] / base.n, a[2] + p[2] / base.n], [0, 0, 0]);
    return atoms.map((a) => ({ ...a, p: [P[a.parent]![0] - c[0], P[a.parent]![1] - c[1], P[a.parent]![2] - c[2]] as Vec3 }));
  }, [atoms, rotated, base]);

  const edgeValue = (e: number) => (stepId === 'radial' ? base.edges.radial[e]![n - 1]! : base.edges.sh[e]![shK]!);
  const shL = Math.floor(Math.sqrt(shK));
  const shM = shK - shL * shL - shL;

  const pickStep = (id: TensorId) => {
    set.setStepId(id);
    set.setChannel(null);
    const avail = tensorLs(base, id).filter((b) => b.live);
    if (avail.length && !avail.some((b) => b.l === state.l)) set.setL(avail[avail.length - 1]!.l);
  };

  const centreName = `${base.symbols[centre]}${centre + 1}`;
  const view = base.cell ? { position: [14, -24, 13] as Vec3, target: [0, 0, 0] as Vec3 } : { position: [6, -10, 6] as Vec3, target: [0, 0, 0] as Vec3 };

  return (
    <div className="lab mw xr">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={sysId}
            view={view}
            background={theme.canvas}
            label={`${base.label}: ${step.title} of the MACE-MP-0a forward pass${ls.length ? `, channel ${channel}, l = ${l}` : ''}${rotated ? ', rotated input' : ''}.`}
            minDistance={4}
            maxDistance={70}
          >
            <XrayScene
              theme={theme}
              atoms={shown}
              centre={centre}
              energyShell={stepId === 'energy'}
              values={values}
              signed={SIGNED(stepId, l)}
              vectors={vectors}
              forces={stepId === 'energy' ? sys.forces : null}
              energies={stepId === 'energy' ? parts.map((p) => p.interaction) : null}
              edgeValue={isEdge ? edgeValue : null}
              rmax={hp.r_max}
              onPick={(i) => i !== null && set.setAtom(shown[i]!.parent)}
            />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {doc.systems.map((s) => (
                <button key={s.id} type="button" aria-pressed={s.id === sysId} onClick={() => { set.setSysId(s.id); set.setChannel(null); set.setAtom(s.id === 'ethanol' ? 1 : 0); }}>
                  {s.label}
                </button>
              ))}
            </div>
            {canRotate && (
              <div className="seg" role="group" aria-label="Input orientation">
                <button type="button" aria-pressed={!rotated} onClick={() => set.setRotated(false)}>
                  Input
                </button>
                <button type="button" aria-pressed={rotated} disabled={isEdge} title={isEdge ? 'Edge tensors are stored for the unrotated pass only' : undefined} onClick={() => set.setRotated(true)}>
                  Rotated {doc.provenance.rotation.angleDeg}°
                </button>
              </div>
            )}
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>centre {centreName} (click an atom)</li>
            {isEdge && <li>{incomingEdges(base, centre).length} incoming edges, r &lt; {hp.r_max} Å</li>}
            {vectors && <li>arrows: l = 1 block of channel {channel}</li>}
            {stepId === 'energy' && <li>arrows: forces −∇E</li>}
          </ul>
        </div>

        <div className="mw-side">
          <ol className="mw-flow" aria-label="Tensors of the MACE forward pass">
            {steps.map((s, k) => (
              <li key={s.id}>
                <button type="button" aria-pressed={s.id === stepId} onClick={() => pickStep(s.id)}>
                  <span className="mw-k">{k + 1}</span>
                  <span className="mw-t">{s.title}</span>
                  <span className="mw-shape">{shapes[s.id]}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="mw-eq" dangerouslySetInnerHTML={{ __html: step.eq }} />
          <p className="mw-technical">{step.what}</p>
          <p className="xr-code">
            <span>hook</span> <code>{step.code}</code>
          </p>
        </div>
      </div>

      <div className="lab-panel xr-panel">
        <div className="lab-controls">
          {ls.length > 0 && (
            <div className="seg" role="group" aria-label="Irrep">
              <span className="seg-label">irrep</span>
              {ls.map((b) => (
                <button key={b.l} type="button" aria-pressed={b.l === l} disabled={!b.live} title={b.live ? undefined : 'Vanishes in this checkpoint (see text)'} onClick={() => set.setL(b.l)}>
                  l = {b.l}
                  {!b.live && ' ≈ 0'}
                </button>
              ))}
            </div>
          )}
          {ls.length > 0 && (
            <label htmlFor={`${ids}-c`}>
              <span>channel k</span>
              <input id={`${ids}-c`} type="range" min={0} max={CHANNELS - 1} step={1} value={channel} onChange={(e) => set.setChannel(Number(e.target.value))} />
              <output>{channel}</output>
            </label>
          )}
          {stepId === 'radial' && (
            <label htmlFor={`${ids}-n`}>
              <span>Bessel function n</span>
              <input id={`${ids}-n`} type="range" min={1} max={hp.num_bessel} step={1} value={n} onChange={(e) => set.setN(Number(e.target.value))} />
              <output>{n}</output>
            </label>
          )}
          {stepId === 'sh' && (
            <label htmlFor={`${ids}-sh`}>
              <span>
                Y<sub>l</sub>
                <sup>m</sup>
              </span>
              <input id={`${ids}-sh`} type="range" min={0} max={(hp.max_ell + 1) ** 2 - 1} step={1} value={shK} onChange={(e) => set.setShK(Number(e.target.value))} />
              <output>
                l = {shL}, m = {shM}
              </output>
            </label>
          )}
        </div>

        {matrix ? (
          <TensorStrip
            matrix={matrix}
            symbols={sys.symbols}
            signed={SIGNED(stepId, l)}
            channel={channel}
            atom={centre}
            onPick={(i, c) => {
              set.setAtom(i);
              set.setChannel(c);
            }}
            caption={`${steps.find((s) => s.id === stepId)!.title}: ${SIGNED(stepId, l) ? 'value' : 'norm'} of the l = ${l} block, atoms × ${CHANNELS} channels`}
          />
        ) : isEdge ? (
          <EdgeTable sys={base} centre={centre} value={edgeValue} label={stepId === 'radial' ? `R${n}(r) · f_cut(r)` : `Y(l=${shL}, m=${shM})`} />
        ) : (
          <EnergyBars parts={parts} total={sys.energy} centre={centre} />
        )}

        {canRotate && check && (
          <RotationReadout check={check} rotated={rotated} doc={doc} />
        )}
        {values && (
          <p className="xr-readout" aria-live="polite">
            {centreName}, channel {channel}, l = {l}: {SIGNED(stepId, l) ? 'value' : '|block|'} = <b>{fmt(values[centre]!, 4)}</b>
            {vectors && (
              <>
                {' '}· vector ({vectors[centre]!.map((x) => fmt(x, 4)).join(', ')})
              </>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

export function TensorStrip({
  matrix,
  symbols,
  signed,
  channel,
  atom,
  onPick,
  caption,
}: {
  matrix: number[][];
  symbols: string[];
  signed: boolean;
  channel: number;
  atom: number;
  onPick: (i: number, c: number) => void;
  caption: string;
}) {
  const rows = matrix.length;
  const channels = matrix[0]?.length ?? CHANNELS;
  const max = Math.max(1e-12, ...matrix.flat().map(Math.abs));
  const W = 640;
  const left = 34;
  const cw = (W - left) / channels;
  const ch = Math.max(9, Math.min(16, 150 / rows));
  const H = rows * ch + 18;
  return (
    <figure className="xr-strip">
      <figcaption>{caption}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${caption}. Largest magnitude ${max.toExponential(2)}.`}>
        {matrix.map((row, i) => (
          <g key={i}>
            <text x={left - 6} y={i * ch + ch * 0.72} className="xr-rowlab" textAnchor="end">
              {symbols[i]}
              {i + 1}
            </text>
            {row.map((v, c) => {
              const t = v / max;
              const cls = signed ? (t >= 0 ? 'pos' : 'neg') : 'pos';
              return (
                <rect
                  key={c}
                  x={left + c * cw}
                  y={i * ch}
                  width={cw + 0.3}
                  height={ch - 1}
                  className={`xr-cell ${cls}`}
                  style={{ opacity: v === 0 ? 0.06 : 0.08 + 0.92 * Math.min(1, Math.abs(t)) ** 0.8 }}
                  onClick={() => onPick(i, c)}
                >
                  <title>
                    {symbols[i]}
                    {i + 1}, k = {c}: {v.toPrecision(4)}
                  </title>
                </rect>
              );
            })}
          </g>
        ))}
        <rect x={left + channel * cw - 0.5} y={0} width={cw + 1} height={rows * ch} className="xr-sel" />
        <rect x={left - 1} y={atom * ch - 0.5} width={W - left + 1} height={ch} className="xr-sel" />
        <text x={left} y={H - 4} className="xr-rowlab">k = 0</text>
        <text x={W} y={H - 4} className="xr-rowlab" textAnchor="end">
          k = {channels - 1}
        </text>
      </svg>
    </figure>
  );
}

export function EdgeTable({ sys, centre, value, label }: { sys: XrayGeometry; centre: number; value: (e: number) => number; label: string }) {
  const inc = incomingEdges(sys, centre);
  const max = Math.max(1e-12, ...inc.map((e) => Math.abs(value(e.e))));
  const shown = inc.slice(0, 14);
  return (
    <div className="mw-panel">
      <p className="mw-ptitle">
        {label} on the {inc.length} edges into {sys.symbols[centre]}
        {centre + 1}
        {inc.length > shown.length && ` (nearest ${shown.length} listed)`}
      </p>
      <div className="mw-bars xr-edges">
        {shown.map((e) => {
          const v = value(e.e);
          return (
            <div key={e.e} className="mw-bar">
              <span className="mw-bl">
                {sys.symbols[e.j]}
                {e.j + 1} · {e.r.toFixed(2)} Å
              </span>
              <span className="mw-track xr-track">
                <span className={`mw-fill ${v < 0 ? 'neg' : ''}`} style={{ width: `${(Math.abs(v) / max) * 50}%`, marginLeft: v < 0 ? `${50 - (Math.abs(v) / max) * 50}%` : '50%' }} />
              </span>
              <span className="mw-bv">{fmt(v, 4)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EnergyBars({ parts, total, centre }: { parts: ReturnType<typeof energyParts>; total: number; centre: number }) {
  const max = Math.max(1e-9, ...parts.map((p) => Math.abs(p.interaction)));
  return (
    <div className="mw-panel">
      <p className="mw-ptitle">
        Site energy E<sub>i</sub> = E<sub>0</sub>(z<sub>i</sub>) + s·(readout₁ + readout₂) + shift. Bars show the learned part (everything but
        E<sub>0</sub>); total E = <b>{total.toFixed(4)} eV</b>.
      </p>
      <div className="mw-bars xr-edges">
        {parts.map((p, i) => (
          <div key={i} className={`mw-bar ${i === centre ? 'xr-cur' : ''}`}>
            <span className="mw-bl">
              {p.sym}
              {i + 1}
            </span>
            <span className="mw-track xr-track">
              <span className={`mw-fill ${p.interaction < 0 ? 'neg' : ''}`} style={{ width: `${(Math.abs(p.interaction) / max) * 50}%`, marginLeft: p.interaction < 0 ? `${50 - (Math.abs(p.interaction) / max) * 50}%` : '50%' }} />
            </span>
            <span className="mw-bv" title={`E0 ${p.e0.toFixed(3)} eV; readouts ${p.readouts.map((r) => r.toFixed(3)).join(', ')} eV; shift ${p.shift.toFixed(3)} eV`}>
              {p.interaction.toFixed(3)} eV
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RotationReadout({ check, rotated, doc }: { check: NonNullable<ReturnType<typeof rotationCheck>>; rotated: boolean; doc: XrayDoc }) {
  const r = doc.provenance.rotation;
  const pct = (x: number | null) => (x === null ? '—' : x < 1e-6 ? '0' : `${(100 * x).toFixed(3)} %`);
  return (
    <div className={`xr-rot ${rotated ? 'on' : ''}`}>
      <p className="mw-ptitle">
        Rotation test (two real forward passes; R = {r.angleDeg}° about ({r.axis.join(', ')})). Largest deviation relative to the largest feature:
      </p>
      <table className="mw-count">
        <thead>
          <tr>
            <th>quantity</th>
            <th>test</th>
            <th>layer 1</th>
            <th>layer 2</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>h, l = 0</td>
            <td>unchanged</td>
            <td className="num">{pct(check.layers[0]!.l0)}</td>
            <td className="num">{pct(check.layers[1]!.l0)}</td>
          </tr>
          <tr>
            <td>h, l = 1</td>
            <td>h′ = R h</td>
            <td className="num">{pct(check.layers[0]!.hl1)}</td>
            <td className="num">{pct(check.layers[1]!.hl1)}</td>
          </tr>
          <tr>
            <td>A, l = 1</td>
            <td>A′ = R A</td>
            <td className="num">{pct(check.layers[0]!.Al1)}</td>
            <td className="num">{pct(check.layers[1]!.Al1)}</td>
          </tr>
          <tr>
            <td>E; F</td>
            <td>E′ = E; F′ = R F</td>
            <td className="num" colSpan={2}>
              |ΔE| = {check.dE.toExponential(1)} eV; F: {pct(check.forces)}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="xr-note">Non-zero entries are the 4–5 significant-figure rounding of the stored file; layer 2 carries no l = 1 block (—).</p>
    </div>
  );
}

export function XrayScene({
  theme,
  atoms,
  centre,
  energyShell,
  values,
  signed,
  vectors,
  forces,
  energies,
  edgeValue,
  rmax,
  onPick,
}: {
  theme: SceneTheme;
  atoms: SceneAtom[];
  centre: number;
  energyShell: boolean;
  values: number[] | null;
  signed: boolean;
  vectors: Vec3[] | null;
  forces: Vec3[] | null;
  energies: number[] | null;
  edgeValue: ((e: number) => number) | null;
  rmax: number;
  onPick: (i: number | null) => void;
}) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const pos = linear(theme.phys);
  const neg = linear(theme.caveat);
  const seq = linear(theme.arch);
  const stone = linear(theme.dark ? theme.canvas : '#b7bec9');

  const buffers = useMemo<AtomBuffers>(() => {
    const N = atoms.length;
    const positions = new Float32Array(N * 3);
    const radii = new Float32Array(N);
    const colors = new Float32Array(N * 3);
    const field = values ?? energies;
    const fmax = field ? Math.max(1e-12, ...field.map(Math.abs)) : 1;
    const emax = edgeValue ? Math.max(1e-12, ...atoms.filter((a) => a.edge !== null).map((a) => Math.abs(edgeValue(a.edge!)))) : 1;
    atoms.forEach((a, k) => {
      positions.set(a.p, 3 * k);
      const isCentre = a.parent === centre && a.edge === null;
      const rr = Math.max(0.3, 0.55 * covalentRadius(a.sym).r) * (isCentre ? 1.3 : 1);
      radii[k] = rr;
      let col: [number, number, number] = colorOf(a.sym, theme.dark);
      if (field) {
        const v = field[a.parent]! / fmax;
        const tgt = signed || energies ? (v >= 0 ? pos : neg) : seq;
        col = mixRGB(stone, tgt, Math.min(1, Math.abs(v)) ** 0.8);
      } else if (edgeValue) {
        if (a.edge === null) col = isCentre ? colorOf(a.sym, theme.dark) : mixRGB(stone, colorOf(a.sym, theme.dark), 0.25);
        else {
          const v = edgeValue(a.edge) / emax;
          col = mixRGB(stone, v >= 0 ? pos : neg, Math.min(1, Math.abs(v)) ** 0.8);
        }
      }
      colors.set(col, 3 * k);
    });
    return { positions, radii, colors, metal: new Float32Array(N), ao: new Float32Array(N).fill(1), count: N };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, centre, values, energies, signed, edgeValue, theme]);

  const c0 = atoms.find((a) => a.parent === centre && a.edge === null)?.p ?? [0, 0, 0];

  const edges = useMemo<ColoredLine[]>(() => {
    if (!edgeValue) return [];
    const emax = Math.max(1e-12, ...atoms.filter((a) => a.edge !== null).map((a) => Math.abs(edgeValue(a.edge!))));
    return atoms
      .filter((a) => a.edge !== null)
      .map((a) => {
        const v = edgeValue(a.edge!) / emax;
        const c = v >= 0 ? pos : neg;
        return { a: a.p as [number, number, number], b: c0 as [number, number, number], rgba: [c[0], c[1], c[2], 0.15 + 0.75 * Math.min(1, Math.abs(v))] as [number, number, number, number] };
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, edgeValue, c0[0], c0[1], c0[2], theme]);

  const arrows = useMemo<Segment[]>(() => {
    const field = vectors ?? forces;
    if (!field) return [];
    const len = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
    const max = Math.max(...field.map(len));
    if (max < 1e-9) return [];
    const scale = 1.6 / max;
    const color = linear(vectors ? theme.arch : theme.fail);
    return atoms
      .filter((a) => len(field[a.parent]!) * scale > 0.45)
      .map((a) => {
        const v = field[a.parent]!;
        const r = 0.5;
        const d = len(v);
        const u: Vec3 = [v[0] / d, v[1] / d, v[2] / d];
        const start: Vec3 = [a.p[0] + u[0] * r, a.p[1] + u[1] * r, a.p[2] + u[2] * r];
        return { a: start, b: [start[0] + v[0] * scale, start[1] + v[1] * scale, start[2] + v[2] * scale], color, radius: 0.06 };
      });
  }, [vectors, forces, atoms, theme.arch, theme.fail]);

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={buffers.count} onPick={onPick} />
      {edges.length > 0 && <Lines lines={edges} />}
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.3} />}
      {edgeValue && <Shell center={c0 as [number, number, number]} radius={rmax} color={theme.phys} opacity={0.22} grid />}
      {energyShell && <Shell center={c0 as [number, number, number]} radius={0.9} color={theme.verified} opacity={0.8} />}
    </>
  );
}
