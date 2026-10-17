import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './MaceWalkthrough.css';
import './XrayExplorer.css';
import { Stage } from '~/three/Stage';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { EdgeTable, TensorStrip, XrayScene, fmt, sceneAtoms, type SceneAtom } from './XrayExplorer';
import { incomingEdges } from '~/lib/models/xray';
import {
  channelMatrix,
  channelValues,
  channelVectors,
  dataflowShapes,
  energyParts,
  isSigned,
  mostVariedChannel,
  symmetryCheck,
  tensorBlocks,
  transformedView,
  type NequipDoc,
  type NequipSystem,
  type NequipTensorId,
  type Vec3,
} from '~/lib/models/xray-nequip';

// Architecture X-ray for NequIP: the real intermediate tensors of one NequIP-OAM-S 0.1 forward
// pass, precomputed by the Atlas pipeline. Every colour, arrow and bar is read from the stored
// tensors; equations are typeset at build time and passed in.

export interface NequipStep {
  id: NequipTensorId;
  title: string;
  /** Pre-rendered KaTeX HTML. */
  eq: string;
  what: string;
  code: string;
}

type Orientation = 'input' | 'rotated' | 'mirrored';
const EDGE_STEPS: NequipTensorId[] = ['radial', 'sh'];
const SH_LABELS = ['Y₀⁰', 'Y₁ (x)', 'Y₁ (y)', 'Y₁ (z)'];

export default function NequipXray({ steps }: { steps: NequipStep[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [doc, setDoc] = useState<NequipDoc | null>(null);
  const [failed, setFailed] = useState(false);
  const [sysId, setSysId] = useState('ethanol');
  const [stepId, setStepId] = useState<NequipTensorId>('h1');
  const [key, setKey] = useState(1);
  const [channel, setChannel] = useState<number | null>(null);
  const [atom, setAtom] = useState(1);
  const [orient, setOrient] = useState<Orientation>('input');
  const [n, setN] = useState(1);
  const [shK, setShK] = useState(3);

  useEffect(() => {
    let live = true;
    import('~/data/xray/nequip-oam-s.json')
      .then((m) => live && setDoc(m.default as unknown as NequipDoc))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);

  if (failed) return <p className="stage-fallback">The precomputed tensors could not be loaded.</p>;
  if (!doc) return <div className="lab xr xr-loading" aria-busy="true">Loading the precomputed forward pass…</div>;

  const base = doc.systems.find((s) => s.id === sysId) ?? doc.systems[0]!;
  return (
    <Inner
      doc={doc}
      base={base}
      steps={steps}
      theme={theme}
      ids={ids}
      state={{ stepId, key, channel, atom, orient, n, shK }}
      set={{
        sys: (id: string) => {
          setSysId(id);
          setChannel(null);
          setAtom(id === 'ethanol' ? 1 : 0);
          setOrient('input');
        },
        step: setStepId,
        key: setKey,
        channel: setChannel,
        atom: setAtom,
        orient: setOrient,
        n: setN,
        shK: setShK,
      }}
    />
  );
}

interface State {
  stepId: NequipTensorId;
  key: number;
  channel: number | null;
  atom: number;
  orient: Orientation;
  n: number;
  shK: number;
}

function Inner({
  doc,
  base,
  steps,
  theme,
  ids,
  state,
  set,
}: {
  doc: NequipDoc;
  base: NequipSystem;
  steps: NequipStep[];
  theme: SceneTheme;
  ids: string;
  state: State;
  set: {
    sys: (id: string) => void;
    step: (v: NequipTensorId) => void;
    key: (v: number) => void;
    channel: (v: number | null) => void;
    atom: (v: number) => void;
    orient: (v: Orientation) => void;
    n: (v: number) => void;
    shK: (v: number) => void;
  };
}) {
  const { stepId, atom, n, shK } = state;
  const hp = doc.model.hyperparameters;
  const isEdge = EDGE_STEPS.includes(stepId);
  const canTransform = !!base.rotated;
  const orient: Orientation = canTransform && !isEdge ? state.orient : 'input';
  const sys = orient === 'input' ? base : transformedView(base, orient)!;
  const step = steps.find((s) => s.id === stepId)!;
  const shapes = dataflowShapes(doc, base);

  const blocks = tensorBlocks(base, stepId);
  const blk = blocks.find((b) => b.key === state.key) ?? blocks[blocks.length - 1] ?? null;
  const key = blk?.key ?? 0;
  const channel = Math.min(state.channel ?? (blk ? mostVariedChannel(base, stepId, key) : 0), (blk?.channels ?? 1) - 1);
  const centre = Math.min(atom, base.n - 1);
  const signed = isSigned(stepId, key);

  const values = blk ? channelValues(sys, stepId, key, channel) : null;
  const vectors = blk?.l === 1 ? channelVectors(sys, stepId, key, channel) : null;
  const matrix = useMemo(() => (blk ? channelMatrix(sys, stepId, key) : null), [sys, stepId, key, blk]);
  const parts = energyParts(sys);
  const rot = useMemo(() => symmetryCheck(base, 'rotated'), [base]);
  const mir = useMemo(() => symmetryCheck(base, 'mirrored'), [base]);

  const atoms = useMemo(() => sceneAtoms(base, centre), [base, centre]);
  const shown = useMemo<SceneAtom[]>(() => {
    if (orient === 'input') return atoms;
    const P = sys.positions;
    const c = P.reduce<Vec3>((a, p) => [a[0] + p[0] / base.n, a[1] + p[1] / base.n, a[2] + p[2] / base.n], [0, 0, 0]);
    return atoms.map((a) => ({ ...a, p: [P[a.parent]![0] - c[0], P[a.parent]![1] - c[1], P[a.parent]![2] - c[2]] as Vec3 }));
  }, [atoms, orient, sys, base.n]);

  const edgeValue = (e: number) => (stepId === 'radial' ? base.edges.radial[e]![n - 1]! : base.edges.sh[e]![shK]!);

  const pickStep = (id: NequipTensorId) => {
    set.step(id);
    set.channel(null);
    const avail = tensorBlocks(base, id);
    if (avail.length && !avail.some((b) => b.key === state.key)) set.key(avail[avail.length - 1]!.key);
  };

  const centreName = `${base.symbols[centre]}${centre + 1}`;
  const view = base.cell ? { position: [12, -20, 11] as Vec3, target: [0, 0, 0] as Vec3 } : { position: [6, -10, 6] as Vec3, target: [0, 0, 0] as Vec3 };
  const orientLabel = orient === 'rotated' ? `, rotated ${doc.provenance.rotation.angleDeg}°` : orient === 'mirrored' ? ', mirrored z → −z' : '';

  return (
    <div className="lab mw xr">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={base.id}
            view={view}
            background={theme.canvas}
            label={`${base.label}: ${step.title} of the NequIP-OAM-S forward pass${blk ? `, ${blk.label} block, channel ${channel}` : ''}${orientLabel}.`}
            minDistance={4}
            maxDistance={70}
          >
            <XrayScene
              theme={theme}
              atoms={shown}
              centre={centre}
              energyShell={stepId === 'energy'}
              values={values}
              signed={signed}
              vectors={vectors}
              forces={stepId === 'energy' ? sys.forces : null}
              energies={stepId === 'energy' ? parts.map((p) => p.learned + p.zbl) : null}
              edgeValue={isEdge ? edgeValue : null}
              rmax={hp.r_max}
              onPick={(i) => i !== null && set.atom(shown[i]!.parent)}
            />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {doc.systems.map((s) => (
                <button key={s.id} type="button" aria-pressed={s.id === base.id} onClick={() => set.sys(s.id)}>
                  {s.label}
                </button>
              ))}
            </div>
            {canTransform && (
              <div className="seg" role="group" aria-label="Input orientation">
                {(['input', 'rotated', 'mirrored'] as const).map((o) => (
                  <button
                    key={o}
                    type="button"
                    aria-pressed={orient === o}
                    disabled={o !== 'input' && isEdge}
                    title={o !== 'input' && isEdge ? 'Edge tensors are stored for the input orientation only' : undefined}
                    onClick={() => set.orient(o)}
                  >
                    {o === 'input' ? 'Input' : o === 'rotated' ? `Rotated ${doc.provenance.rotation.angleDeg}°` : 'Mirrored'}
                  </button>
                ))}
              </div>
            )}
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>centre {centreName} (click an atom)</li>
            {isEdge && <li>{incomingEdges(base, centre).length} incoming edges, r &lt; {hp.r_max} Å</li>}
            {vectors && <li>arrows: l = 1 vectors of channel {channel}</li>}
            {stepId === 'energy' && <li>colour: learned + ZBL site energy; arrows: forces</li>}
          </ul>
        </div>

        <div className="mw-side">
          <ol className="mw-flow" aria-label="Tensors of the NequIP forward pass">
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
          {blocks.length > 0 && (
            <div className="seg" role="group" aria-label={stepId.startsWith('m') ? 'Tensor-product path' : 'Irrep'}>
              <span className="seg-label">{stepId.startsWith('m') ? 'path' : 'block'}</span>
              {blocks.map((b) => (
                <button key={b.key} type="button" aria-pressed={b.key === key} onClick={() => { set.key(b.key); set.channel(null); }}>
                  {b.label}
                </button>
              ))}
            </div>
          )}
          {blk && (
            <label htmlFor={`${ids}-c`}>
              <span>channel</span>
              <input id={`${ids}-c`} type="range" min={0} max={blk.channels - 1} step={1} value={channel} onChange={(e) => set.channel(Number(e.target.value))} />
              <output>{channel}</output>
            </label>
          )}
          {stepId === 'radial' && (
            <label htmlFor={`${ids}-n`}>
              <span>Bessel function n</span>
              <input id={`${ids}-n`} type="range" min={1} max={hp.num_bessels} step={1} value={n} onChange={(e) => set.n(Number(e.target.value))} />
              <output>
                {n} (w = {hp.bessel_weights[n - 1]!.toFixed(2)})
              </output>
            </label>
          )}
          {stepId === 'sh' && (
            <label htmlFor={`${ids}-sh`}>
              <span>component</span>
              <input id={`${ids}-sh`} type="range" min={0} max={3} step={1} value={shK} onChange={(e) => set.shK(Number(e.target.value))} />
              <output>{SH_LABELS[shK]}</output>
            </label>
          )}
        </div>

        {matrix ? (
          <TensorStrip
            matrix={matrix}
            symbols={sys.symbols}
            signed={signed}
            channel={channel}
            atom={centre}
            onPick={(i, c) => {
              set.atom(i);
              set.channel(c);
            }}
            caption={`${step.title}, ${blk!.label}: ${signed ? 'value' : 'norm'} per channel, atoms × ${blk!.channels} channels`}
          />
        ) : isEdge ? (
          <EdgeTable sys={base} centre={centre} value={edgeValue} label={stepId === 'radial' ? `B${n}(r) · f_cut(r) · 2π/r_c²` : `${SH_LABELS[shK]} of r̂ (receiving atom → neighbour)`} />
        ) : (
          <EnergySplit parts={parts} total={sys.energy} centre={centre} />
        )}

        {canTransform && rot && mir && <SymmetryReadout rot={rot} mir={mir} orient={orient} doc={doc} />}
        {values && (
          <p className="xr-readout" aria-live="polite">
            {centreName}, {blk!.label}, channel {channel}: {signed ? 'value' : '|block|'} = <b>{fmt(values[centre]!, 4)}</b>
            {vectors && <> · vector ({vectors[centre]!.map((x) => fmt(x, 4)).join(', ')})</>}
          </p>
        )}
      </div>
    </div>
  );
}

function EnergySplit({ parts, total, centre }: { parts: ReturnType<typeof energyParts>; total: number; centre: number }) {
  const max = Math.max(1e-9, ...parts.flatMap((p) => [Math.abs(p.learned), Math.abs(p.zbl)]));
  const bar = (v: number, cls: string) => (
    <span className="mw-track xr-track">
      <span className={`mw-fill ${v < 0 ? 'neg' : ''} ${cls}`} style={{ width: `${(Math.abs(v) / max) * 50}%`, marginLeft: v < 0 ? `${50 - (Math.abs(v) / max) * 50}%` : '50%' }} />
    </span>
  );
  return (
    <div className="mw-panel">
      <p className="mw-ptitle">
        E<sub>i</sub> = s(z<sub>i</sub>)·readout<sub>i</sub> (learned) + ZBL<sub>i</sub> + μ(z<sub>i</sub>) (isolated-atom shift). Upper bar: learned; lower bar: ZBL. Total E ={' '}
        <b>{total.toFixed(4)} eV</b>.
      </p>
      <div className="mw-bars xr-edges nq-split">
        {parts.map((p, i) => (
          <div key={i} className={`mw-bar ${i === centre ? 'xr-cur' : ''}`}>
            <span className="mw-bl">
              {p.sym}
              {i + 1}
            </span>
            <span className="nq-pair">
              {bar(p.learned, '')}
              {bar(p.zbl, 'nq-zbl')}
            </span>
            <span className="mw-bv" title={`shift μ = ${p.shift.toFixed(4)} eV; site energy ${p.total.toFixed(4)} eV`}>
              {p.learned.toFixed(2)} / +{p.zbl.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SymmetryReadout({
  rot,
  mir,
  orient,
  doc,
}: {
  rot: NonNullable<ReturnType<typeof symmetryCheck>>;
  mir: NonNullable<ReturnType<typeof symmetryCheck>>;
  orient: Orientation;
  doc: NequipDoc;
}) {
  const r = doc.provenance.rotation;
  const pct = (x: number | null) => (x === null ? '—' : x === 0 ? '0' : x < 1e-6 ? '< 10⁻⁶' : `${(100 * x).toFixed(3)} %`);
  return (
    <div className={`xr-rot ${orient !== 'input' ? 'on' : ''}`}>
      <p className="mw-ptitle">
        Three real forward passes of ethanol: as given, rotated (R = {r.angleDeg}° about ({r.axis.join(', ')})) and mirrored (z → −z). Largest deviation
        relative to the largest value:
      </p>
      <table className="mw-count">
        <thead>
          <tr>
            <th>quantity</th>
            <th>test (Q = R or M)</th>
            <th>rotated</th>
            <th>mirrored</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>h⁽¹⁾, h⁽²⁾ scalars</td>
            <td>unchanged</td>
            <td className="num">{pct(Math.max(...rot.layers.map((L) => L.l0)))}</td>
            <td className="num">{pct(Math.max(...mir.layers.map((L) => L.l0)))}</td>
          </tr>
          <tr>
            <td>h⁽¹⁾ vectors</td>
            <td>h′ = Q h</td>
            <td className="num">{pct(rot.layers[0]!.hl1)}</td>
            <td className="num">{pct(mir.layers[0]!.hl1)}</td>
          </tr>
          <tr>
            <td>m⁽¹⁾ vectors</td>
            <td>m′ = Q m</td>
            <td className="num">{pct(rot.layers[0]!.ml1)}</td>
            <td className="num">{pct(mir.layers[0]!.ml1)}</td>
          </tr>
          <tr>
            <td>E; F</td>
            <td>E′ = E; F′ = Q F</td>
            <td className="num">
              {rot.dE.toExponential(0)} eV; {pct(rot.forces)}
            </td>
            <td className="num">
              {mir.dE === 0 ? '0' : mir.dE.toExponential(0)} eV; {pct(mir.forces)}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="xr-note">
        Non-zero rotated entries are the 4–5 significant-figure rounding of the stored file (the rotated coordinates are not exactly representable). The mirror z → −z only
        flips signs, so it is exact even after rounding.
      </p>
    </div>
  );
}
