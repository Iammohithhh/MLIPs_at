import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Spooky.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { PAIRS, RC_PAPER, electrostatics, fragmentPair, makeToySpooky, runToy, subset, type Molecule, type NonlocalMode, type PairKind, type Vec3 } from '~/lib/models/spookynet';
import { Legend, LinePlot, niceTicks } from './spooky/LinePlot';

// Fig. 3: two fragments pulled apart in a toy SpookyNet (fixed random weights), with and without the
// nonlocal block; live. Energies are the network part Σ E_i in toy units; charges from Eq. (24).

const MODEL = makeToySpooky();
const D_MAX = 14;
const KINDS = Object.keys(PAIRS) as PairKind[];
const NAMES: Record<PairKind, [string, string]> = {
  LiF: ['Li', 'F'],
  NaCl: ['Na', 'Cl'],
  HF: ['H', 'F'],
  'benzene-water': ['benzene', 'water'],
  'benzene-benzene': ['benzene 1', 'benzene 2'],
};

const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);

function parts(mol: Molecule, nA: number): [Molecule, Molecule] {
  return [subset(mol, [...Array(nA).keys()]), subset(mol, Array.from({ length: mol.numbers.length - nA }, (_, i) => i + nA))];
}

function minGap(mol: Molecule, nA: number): number {
  let m = Infinity;
  for (let i = 0; i < nA; i++)
    for (let j = nA; j < mol.positions.length; j++) {
      const a = mol.positions[i]!;
      const b = mol.positions[j]!;
      m = Math.min(m, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
  return m;
}

function curves(kind: PairKind, Q: number) {
  const { nA, dMin } = PAIRS[kind];
  const ds: number[] = [];
  for (let d = dMin; d <= D_MAX + 1e-9; d += 0.2) ds.push(+d.toFixed(2));
  const modes: NonlocalMode[] = ['off', 'exact'];
  const out = modes.map((nonlocal) => {
    const [A, B] = parts(fragmentPair(kind, D_MAX), nA);
    const limA = runToy(MODEL, A, { Q, nonlocal }).energy + runToy(MODEL, B, { Q: 0, nonlocal }).energy;
    const limB = runToy(MODEL, A, { Q: 0, nonlocal }).energy + runToy(MODEL, B, { Q, nonlocal }).energy;
    const half = kind === 'benzene-benzene' && Q !== 0 ? 2 * runToy(MODEL, A, { Q: Q / 2, nonlocal }).energy : null;
    const rows = ds.map((d) => {
      const mol = fragmentPair(kind, d);
      const r = runToy(MODEL, mol, { Q, nonlocal });
      return { d, E: r.energy, qA: sum(r.q.slice(0, nA)), gap: minGap(mol, nA) };
    });
    return { nonlocal, limA, limB, half, rows };
  });
  const split = out[0]!.rows.find((r) => r.gap >= RC_PAPER)?.d ?? null;
  return { ds, out, split };
}

export default function SpookyFragments() {
  const theme = useSceneTheme();
  const id = useId();
  const [kind, setKind] = useState<PairKind>('LiF');
  const [Q, setQ] = useState(0);
  const [nl, setNl] = useState<NonlocalMode>('exact');
  const [d, setD] = useState(4);
  const { nA, dMin } = PAIRS[kind];
  const dd = Math.max(dMin, d);
  const mol = useMemo(() => fragmentPair(kind, dd), [kind, dd]);
  const run = useMemo(() => runToy(MODEL, mol, { Q, nonlocal: nl }), [mol, Q, nl]);
  const data = useMemo(() => curves(kind, Q), [kind, Q]);
  const cur = data.out.find((o) => o.nonlocal === nl)!;
  const ref = Math.min(cur.limA, cur.limB);
  const names = NAMES[kind];
  const nTot = mol.numbers.length;
  const aA = sum(run.aQ.slice(0, nA));
  const far = cur.rows[cur.rows.length - 1]!;
  const gapNow = minGap(mol, nA);
  const eEle = useMemo(() => {
    const [A, B] = parts(mol, nA);
    return electrostatics(run.q, mol.positions) - electrostatics(run.q.slice(0, nA), A.positions) - electrostatics(run.q.slice(nA), B.positions);
  }, [mol, run, nA]);

  const eSeries = data.out.map((o) => ({
    key: o.nonlocal,
    points: o.rows.map((r) => [r.d, r.E - Math.min(o.limA, o.limB)] as [number, number]),
    color: o.nonlocal === 'off' ? 'var(--atlas-phys)' : 'var(--atlas-arch)',
    dashed: o.nonlocal !== nl,
    thin: o.nonlocal !== nl,
  }));
  const limits = [cur.limA, cur.limB].map((v) => v - ref);
  const eVals = [...eSeries.flatMap((s) => s.points.filter(([x]) => x > dMin + 0.6).map((p) => p[1])), ...limits, ...(cur.half !== null ? [cur.half - ref] : [])];
  const eLo = Math.min(...eVals);
  const eHi = Math.max(...eVals);
  const ePad = 0.12 * (eHi - eLo || 0.1);

  const qSeries = data.out.map((o) => ({
    key: o.nonlocal,
    points: o.rows.map((r) => [r.d, r.qA] as [number, number]),
    color: o.nonlocal === 'off' ? 'var(--atlas-phys)' : 'var(--atlas-arch)',
    dashed: o.nonlocal !== nl,
    thin: o.nonlocal !== nl,
  }));
  const qVals = [...qSeries.flatMap((s) => s.points.map((p) => p[1])), aA, 0, Q];
  const qLo = Math.min(...qVals);
  const qHi = Math.max(...qVals);
  const qPad = 0.12 * (qHi - qLo || 0.1);
  const xTicks = niceTicks(0, D_MAX, 5).filter((v) => v >= dMin - 0.5);
  const view = useMemo(
    () =>
      kind === 'LiF' || kind === 'NaCl' || kind === 'HF'
        ? { position: [4, -30, 10] as [number, number, number], target: [4, 0, 0] as [number, number, number] }
        : { position: [26, -24, 16] as [number, number, number], target: [0, 0, 4.5] as [number, number, number] },
    [kind],
  );

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={view} resetKey={kind.length} background={theme.canvas} label={`${names[0]} and ${names[1]} ${dd.toFixed(1)} Å apart, total charge ${Q}, atoms coloured by the toy model's charges: red positive, blue negative.`} minDistance={5} maxDistance={70}>
            <PairScene theme={theme} mol={mol} nA={nA} q={run.q} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Pair">
              {KINDS.map((k) => (
                <button key={k} type="button" aria-pressed={kind === k} onClick={() => setKind(k)}>
                  {PAIRS[k].label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: 'var(--atlas-fail)' }} /> q &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: 'var(--atlas-arch)' }} /> q &lt; 0
            </li>
            <li>
              d = {dd.toFixed(1)} Å · Q = {Q > 0 ? `+${Q}` : Q}
            </li>
          </ul>
        </div>
        <div className="sp-side">
          <p className="sp-h">Network energy against the integer limits</p>
          <LinePlot
            H={165}
            x={{ min: dMin, max: D_MAX, ticks: xTicks, label: 'd (Å)' }}
            y={{ min: eLo - ePad, max: eHi + ePad, ticks: niceTicks(eLo - ePad, eHi + ePad, 3), label: 'E − limit (toy)', fmt: (v) => v.toFixed(Math.abs(eHi - eLo) < 0.5 ? 2 : 1) }}
            series={eSeries}
            hlines={[...limits.map((v) => ({ v, color: 'var(--atlas-ink-faint)' })), ...(cur.half !== null ? [{ v: cur.half - ref, color: 'var(--atlas-fail)' }] : [])]}
            band={data.split ?? undefined}
            bandLabel="separate graphs"
            cursor={dd}
            ariaLabel="Network energy of the toy model against separation, with and without nonlocal attention, each relative to the lower of its own integer-charge limits; both curves are flat once the fragments are farther apart than the cutoff."
          />
          <Legend
            items={[
              { label: 'local only', color: 'var(--atlas-phys)' },
              { label: 'with attention', color: 'var(--atlas-arch)' },
              { label: Q ? 'integer limits (selected model)' : 'E(A) + E(B)', color: 'var(--atlas-ink-faint)', dashed: true },
              ...(cur.half !== null ? [{ label: '2 E(A, Q/2)', color: 'var(--atlas-fail)', dashed: true }] : []),
            ]}
          />
          <p className="sp-h">Charge on {names[0]}</p>
          <LinePlot
            H={140}
            x={{ min: dMin, max: D_MAX, ticks: xTicks, label: 'd (Å)' }}
            y={{ min: qLo - qPad, max: qHi + qPad, ticks: niceTicks(qLo - qPad, qHi + qPad, 3), label: 'q (e)', fmt: (v) => v.toFixed(2) }}
            series={qSeries}
            hlines={Q !== 0 ? [{ v: aA, color: 'var(--atlas-caveat)' }] : []}
            band={data.split ?? undefined}
            cursor={dd}
            ariaLabel="Charge on the first fragment against separation; it stops changing once the fragments are on separate graphs, and the share of the charge embedding is the same at every distance."
          />
          <Legend
            items={[
              { label: 'Σ_A q_i, Eq. (24)', color: 'var(--atlas-arch)' },
              ...(Q !== 0 ? [{ label: 'Σ_A a_i, Eq. (10)', color: 'var(--atlas-caveat)', dashed: true }] : []),
            ]}
          />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${id}-d`}>
            <span>separation d</span>
            <input id={`${id}-d`} type="range" min={dMin} max={D_MAX} step={0.2} value={dd} onChange={(e) => setD(Number(e.target.value))} />
            <output>{dd.toFixed(1)} Å</output>
          </label>
          <div className="seg" role="group" aria-label="Total charge Q">
            {[-1, 0, 1, 2].map((v) => (
              <button key={v} type="button" aria-pressed={Q === v} onClick={() => setQ(v)}>
                Q = {v > 0 ? `+${v}` : v}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Nonlocal block">
            <button type="button" aria-pressed={nl === 'exact'} onClick={() => setNl('exact')}>
              with attention
            </button>
            <button type="button" aria-pressed={nl === 'off'} onClick={() => setNl('off')}>
              local only
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>closest atoms of the two fragments</dt>
            <dd>
              {gapNow.toFixed(2)} Å <small>{gapNow >= RC_PAPER ? '(separate graphs)' : '(within r_cut)'}</small>
            </dd>
          </div>
          <div>
            <dt>charge embedding on {names[0]}, Σ_A a_i</dt>
            <dd>
              {Q === 0 ? '0 (e_Q = 0)' : `${aA.toFixed(3)} e`} <small>at every d</small>
            </dd>
          </div>
          <div>
            <dt>charge on {names[0]} / {names[1]}</dt>
            <dd>
              {sum(run.q.slice(0, nA)).toFixed(3)} / {sum(run.q.slice(nA)).toFixed(3)} e
            </dd>
          </div>
          <div className={[cur.limA, cur.limB].some((v) => Math.abs(far.E - v) < 1e-9) ? 'is-good' : 'is-alert'}>
            <dt>far limit − lower integer limit (toy units)</dt>
            <dd>{Math.abs(far.E - ref) < 1e-10 ? '0 (adds up)' : (far.E - ref).toFixed(4)}</dd>
          </div>
          <div>
            <dt>E_ele between the fragments, Eq. (23)</dt>
            <dd>{eEle.toFixed(4)} eV</dd>
          </div>
          <div>
            <dt>uniform charge correction on {names[0]}</dt>
            <dd>
              N_A/N = {nA}/{nTot}
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          <strong>Toy model:</strong> fixed random weights, so the energies are in toy units and the particular values mean nothing; what is
          shown structurally holds for any weights. Beyond the cutoff both curves are flat. Without attention and with Q = 0 the far limit is
          exactly E(A) + E(B). With attention, unlike fragments keep a constant offset at any distance; two identical benzenes do not. A total
          charge is shared through the charge embedding by element counts alone, so two identical fragments always get Q/2 each and the far limit
          is 2E(A, Q/2) (red), neither integer limit (grey). Each curve is drawn relative to the lower integer limit of its own model. E_ele uses the toy charges with Eq. (23) and is reported, not added to the curves.
        </p>
      </div>
    </div>
  );
}

function chargeColour(theme: SceneTheme, q: number, qMax: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, q / qMax));
  const base = linear(theme.dark ? '#4a5260' : '#d5d8de');
  return mixRGB(base, linear(t < 0 ? theme.arch : theme.fail), Math.sqrt(Math.abs(t)));
}

function PairScene({ theme, mol, nA, q }: { theme: SceneTheme; mol: Molecule; nA: number; q: number[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 70);
  const qMax = Math.max(0.05, ...q.map(Math.abs));
  const buffers = useMemo<AtomBuffers>(() => {
    const n = mol.positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    mol.positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      const Z = mol.numbers[i]!;
      radii[i] = Z === 1 ? 0.3 : Z === 6 || Z === 8 || Z === 9 ? 0.45 : 0.6;
      colors.set(chargeColour(theme, q[i]!, qMax), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.15), ao: new Float32Array(n).fill(1), count: n };
  }, [mol, q, qMax, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.ruleStrong);
    const out: Segment[] = [];
    const bonded = (i: number, j: number) => (i < nA) === (j < nA);
    for (let i = 0; i < mol.positions.length; i++)
      for (let j = i + 1; j < mol.positions.length; j++) {
        const a = mol.positions[i]! as Vec3;
        const b = mol.positions[j]! as Vec3;
        const r = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        if ((bonded(i, j) && r < 1.6) || (nA === 1 && r < 2.6)) out.push({ a, b, color: c, radius: 0.07 });
      }
    return out;
  }, [mol, nA, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
    </>
  );
}
