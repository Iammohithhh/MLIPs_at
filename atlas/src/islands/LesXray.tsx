import { useEffect, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './MaceWalkthrough.css';
import './XrayExplorer.css';
import { Stage } from '~/three/Stage';
import { useSceneTheme } from '~/three/theme';
import { XrayScene, fmt, sceneAtoms, type SceneAtom } from './XrayExplorer';
import {
  becSumRule,
  netCharge,
  rotationCheck,
  stepValues,
  type LesDoc,
  type LesStepId,
  type LesSystem,
  type Vec3,
} from '~/lib/models/xray-les';

// Architecture X-ray for MACELES-OFF: per-atom tensors of one real forward pass (precomputed on
// CPU by pipeline/atlas_pipeline/xray_les.py). Colours, arrows and bars are read from the stored
// values; equations are typeset at build time and passed in.

export interface LesStep {
  id: LesStepId;
  title: string;
  /** Pre-rendered KaTeX HTML. */
  eq: string;
  what: string;
  code: string;
  parts?: string[];
}

type Orientation = 'input' | 'rotated';
type Arrows = 'none' | 'sr' | 'lr';

export default function LesXray({ steps }: { steps: LesStep[] }) {
  const [doc, setDoc] = useState<LesDoc | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    import('~/data/xray/maceles-off.json')
      .then((m) => live && setDoc(m.default as unknown as LesDoc))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);
  if (failed) return <p className="stage-fallback">The precomputed forward pass could not be loaded.</p>;
  if (!doc) return <div className="lab xr xr-loading" aria-busy="true">Loading the precomputed forward pass…</div>;
  return <Inner doc={doc} steps={steps} />;
}

function rotatedSystem(sys: LesSystem): LesSystem | null {
  const r = sys.rotated;
  if (!r) return null;
  return { ...sys, positions: r.positions, q: r.q, fSr: r.fSr, fLr: r.fLr, P: r.P, eSr: r.eSr, eLr: r.eLr };
}

function Inner({ doc, steps }: { doc: LesDoc; steps: LesStep[] }) {
  const theme = useSceneTheme();
  const [sysId, setSysId] = useState(doc.systems[0]!.id);
  const [stepId, setStepId] = useState<LesStepId>('q');
  const [part, setPart] = useState<0 | 1 | 2>(2);
  const [orient, setOrient] = useState<Orientation>('input');
  const [arrows, setArrows] = useState<Arrows>('lr');
  const [atom, setAtom] = useState(0);

  const base = doc.systems.find((s) => s.id === sysId) ?? doc.systems[0]!;
  const sys = orient === 'rotated' ? (rotatedSystem(base) ?? base) : base;
  const step = steps.find((s) => s.id === stepId)!;
  const parts = step.parts ?? [];
  const p = (parts.length ? Math.min(part, parts.length - 1) : 0) as 0 | 1 | 2;
  // In the rotated view only the invariant per-atom values the pipeline stored again are shown.
  const { values: v0, signed, unit } = stepValues(base, stepId, p);
  const values = orient === 'rotated' && stepId === 'q' && p === 2 ? sys.q : v0;
  const centre = Math.min(atom, base.n - 1);
  const rot = useMemo(() => rotationCheck(base), [base]);

  const atoms = useMemo(() => sceneAtoms(base, centre), [base, centre]);
  const shown = useMemo<SceneAtom[]>(() => {
    const P = sys.positions;
    const c = P.reduce<Vec3>((a, q) => [a[0] + q[0] / base.n, a[1] + q[1] / base.n, a[2] + q[2] / base.n], [0, 0, 0]);
    return atoms.map((a) => ({ ...a, p: [P[a.parent]![0] - c[0], P[a.parent]![1] - c[1], P[a.parent]![2] - c[2]] as Vec3 }));
  }, [atoms, sys, base.n]);
  const forces = arrows === 'sr' ? sys.fSr : arrows === 'lr' ? sys.fLr : null;

  const name = (i: number) => `${base.symbols[i]}${i + 1}`;
  const vmax = Math.max(1e-12, ...values.map(Math.abs));
  const pickStep = (id: LesStepId) => {
    setStepId(id);
    setPart(id === 'q' ? 2 : 0);
  };

  return (
    <div className="lab mw xr">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={base.id}
            view={{ position: [6, -10, 6] as Vec3, target: [0, 0, 0] as Vec3 }}
            background={theme.canvas}
            label={`${base.label}: ${step.title} of the MACELES-OFF forward pass${parts.length ? `, ${parts[p]}` : ''}${orient === 'rotated' ? ', rotated' : ''}; arrows: ${arrows === 'lr' ? 'long-range forces' : arrows === 'sr' ? 'local forces' : 'none'}.`}
            minDistance={4}
            maxDistance={70}
          >
            <XrayScene
              theme={theme}
              atoms={shown}
              centre={centre}
              energyShell={false}
              values={values}
              signed={signed}
              vectors={null}
              forces={forces}
              energies={null}
              edgeValue={null}
              rmax={doc.model.hyperparameters.r_max}
              onPick={(i) => i !== null && setAtom(shown[i]!.parent)}
            />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {doc.systems.map((s) => (
                <button key={s.id} type="button" aria-pressed={s.id === base.id} onClick={() => { setSysId(s.id); setOrient('input'); setAtom(0); }}>
                  {s.label}
                </button>
              ))}
            </div>
            {base.rotated && (
              <div className="seg" role="group" aria-label="Orientation">
                {(['input', 'rotated'] as const).map((o) => (
                  <button key={o} type="button" aria-pressed={orient === o} onClick={() => setOrient(o)}>
                    {o === 'input' ? 'as input' : `rotated ${(doc.provenance.rotation as { angleDeg: number }).angleDeg}°`}
                  </button>
                ))}
              </div>
            )}
            <div className="seg" role="group" aria-label="Force arrows">
              <span className="seg-label">arrows</span>
              {(['lr', 'sr', 'none'] as const).map((a) => (
                <button key={a} type="button" aria-pressed={arrows === a} onClick={() => setArrows(a)}>
                  {a === 'lr' ? '−∇E_lr' : a === 'sr' ? '−∇E_sr' : 'off'}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mw-side">
          <ol className="mw-flow" aria-label="Steps of the MACELES forward pass">
            {steps.map((s, k) => (
              <li key={s.id}>
                <button type="button" aria-pressed={s.id === stepId} onClick={() => pickStep(s.id)}>
                  <span className="mw-k">{k + 1}</span>
                  <span className="mw-t">{s.title}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="mw-eq" dangerouslySetInnerHTML={{ __html: step.eq }} />
          <p className="mw-technical">{step.what}</p>
          <p className="xr-code">
            <span>computed by</span> <code>{step.code}</code>
          </p>
        </div>
      </div>

      <div className="lab-panel xr-panel">
        {parts.length > 1 && (
          <div className="lab-controls">
            <div className="seg" role="group" aria-label="Part">
              {parts.map((label, k) => (
                <button key={label} type="button" aria-pressed={k === p} onClick={() => setPart(k as 0 | 1 | 2)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="mw-bars xr-edges les-bars" role="list" aria-label={`${step.title} per atom (${unit})`}>
          {values.map((v, i) => (
            <button key={i} type="button" role="listitem" className={`mw-bar ${i === centre ? 'xr-cur' : ''}`} onClick={() => setAtom(i)}>
              <span className="mw-bl">{name(i)}</span>
              <span className="mw-track xr-track">
                <span
                  className={`mw-fill ${v < 0 ? 'neg' : ''}`}
                  style={
                    signed
                      ? { width: `${(Math.abs(v) / vmax) * 50}%`, marginLeft: v < 0 ? `${50 - (Math.abs(v) / vmax) * 50}%` : '50%' }
                      : { width: `${(Math.abs(v) / vmax) * 100}%` }
                  }
                />
              </span>
              <span className="mw-bv">{fmt(v, 3)}</span>
            </button>
          ))}
        </div>
        <p className="xr-readout" aria-live="polite">
          {name(centre)}: {step.title.toLowerCase()} = <b>{fmt(values[centre]!, 4)}</b> {unit} · |F_sr| = {fmt(Math.hypot(...sys.fSr[centre]!), 3)} eV/Å, |F_lr| ={' '}
          {fmt(Math.hypot(...sys.fLr[centre]!), 3)} eV/Å
        </p>
        <Totals sys={sys} base={base} centre={centre} />
        {orient === 'rotated' && rot && (
          <p className="xr-note">
            Rotation test on the stored passes: |ΔE| = {fmt(rot.dE, 2)} eV, max |Δq| = {fmt(rot.dq, 2)}, rotated forces match R·F to {fmt(Math.max(rot.dFsr, rot.dFlr), 2)} eV/Å.
          </p>
        )}
      </div>
    </div>
  );
}

function Totals({ sys, base, centre }: { sys: LesSystem; base: LesSystem; centre: number }) {
  const Z = base.Z[centre]!;
  const Zd = base.dft?.Z[centre];
  return (
    <dl className="les-totals">
      <div>
        <dt>E_sr (local, incl. E₀)</dt>
        <dd>{sys.eSr.toFixed(4)} eV</dd>
      </div>
      <div>
        <dt>E_lr (latent Coulomb)</dt>
        <dd>{sys.eLr.toFixed(4)} eV</dd>
      </div>
      <div>
        <dt>Σ q (net latent charge)</dt>
        <dd>{netCharge(sys.q).toFixed(4)}</dd>
      </div>
      <div>
        <dt>P = Σ (q − q̄) r</dt>
        <dd>({sys.P.map((x) => x.toFixed(3)).join(', ')}) e·Å</dd>
      </div>
      {base.dft && (
        <div>
          <dt>DFT dipole μ</dt>
          <dd>({base.dft.dipole.map((x) => x.toFixed(3)).join(', ')})</dd>
        </div>
      )}
      <div>
        <dt>Z* diagonal, atom {centre + 1}</dt>
        <dd>
          ({[0, 1, 2].map((a) => Z[a]![a]!.toFixed(2)).join(', ')}){Zd && <> · DFT ({[0, 1, 2].map((a) => Zd[a]![a]!.toFixed(2)).join(', ')})</>}
        </dd>
      </div>
      <div>
        <dt>max |Σᵢ Z*ᵢ|</dt>
        <dd>{fmt(becSumRule(base.Z), 2)}</dd>
      </div>
    </dl>
  );
}
