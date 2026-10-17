import { useMemo, useState } from 'react';
import './PhysicsLab.css';
import './LongRange.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { RC_4G, carbonChain, compareChain, makeAtomicNet, type ChargeModel, type Molecule, type Vec3 } from '~/lib/models/long-range';

// Who can see a change beyond the cutoff (src/lib/models/long-range.ts). The same toy atomic network,
// fed by radial symmetry functions inside 4.23 Å, predicts a charge directly (3G-HDNNP; LES's latent
// charge), the same charge shifted to the right total (3G "scaled"), or an electronegativity that charge
// equilibration turns into charges (4G-HDNNP). Protonating the right end of C₁₀H₂ changes the
// environments of the atoms near it only; untrained weights, so only zero versus non-zero is meaningful.

const net = makeAtomicNet();
const N = 10;

const MODELS: Array<{ id: ChargeModel; name: string; sub: string }> = [
  { id: 'local', name: 'Local charge', sub: '3G-HDNNP; LES latent q' },
  { id: 'scaled', name: 'Local, rescaled', sub: '3G “scaled”: shift to Σq = Q' },
  { id: 'qeq', name: 'Charge equilibration', sub: '4G-HDNNP: local χ, global q' },
];
const ZERO = 1e-12;

export default function LrChain() {
  const theme = useSceneTheme();
  const [model, setModel] = useState<ChargeModel>('qeq');
  const [cation, setCation] = useState(true);
  const neutralMol = useMemo(() => carbonChain(N, false), []);
  const cationMol = useMemo(() => carbonChain(N, true), []);
  const all = useMemo(() => Object.fromEntries(MODELS.map((m) => [m.id, compareChain(m.id, net, N)])) as Record<ChargeModel, ReturnType<typeof compareChain>>, []);
  const cmp = all[model];
  const mol = cation ? cationMol : neutralMol;
  const q = cation ? cmp.cation : cmp.neutral;
  const unchanged = cmp.unchanged;
  const farIdx = unchanged.map((u, i) => (u ? i : -1)).filter((i) => i >= 0);
  const farMax = Math.max(...farIdx.map((i) => Math.abs(cmp.delta[i]!)));
  const sum = (x: number[]) => x.reduce((a, b) => a + b, 0);
  const qMax = useMemo(() => Math.max(...Object.values(all).flatMap((c) => [...c.neutral, ...c.cation].map(Math.abs))), [all]);
  const view = useMemo(() => ({ position: [5.8, -30, 11] as [number, number, number], target: [5.8, 0, 0] as [number, number, number] }), []);
  const m = MODELS.find((x) => x.id === model)!;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={0}
            background={theme.canvas}
            label={`${cation ? 'C10H3+' : 'C10H2'} coloured by the ${m.name.toLowerCase()} charges: blue negative, red positive. The spheres of radius ${RC_4G} Å around the right-hand hydrogens mark the atoms whose environment protonation changes.`}
            minDistance={4}
            maxDistance={60}
          >
            <ChainScene theme={theme} mol={mol} q={q} qMax={qMax} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Molecule">
              <button type="button" aria-pressed={!cation} onClick={() => setCation(false)}>
                C₁₀H₂
              </button>
              <button type="button" aria-pressed={cation} onClick={() => setCation(true)}>
                C₁₀H₃⁺ (protonated)
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.arch }} /> q &lt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.fail }} /> q &gt; 0
            </li>
            <li>
              <span className="ring" /> r<sub>c</sub> = {RC_4G} Å around the changed end
            </li>
          </ul>
        </div>
        <div className="lr-side">
          <div className="seg lr-models" role="group" aria-label="Charge model">
            {MODELS.map((x) => (
              <button key={x.id} type="button" aria-pressed={model === x.id} onClick={() => setModel(x.id)}>
                {x.name}
              </button>
            ))}
          </div>
          <p className="lr-sub">{m.sub}</p>
          <ChargeBars theme={theme} neutral={cmp.neutral} cation={cmp.cation} qMax={qMax} />
          <DeltaStrip theme={theme} all={all} unchanged={unchanged} active={model} />
        </div>
      </div>
      <div className="lab-panel">
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>atoms with identical environments in both molecules</dt>
            <dd>
              {farIdx.length} <small>(1–{farIdx.length})</small>
            </dd>
          </div>
          <div className={farMax < ZERO ? 'is-alert' : undefined}>
            <dt>largest change of charge on those atoms</dt>
            <dd>{farMax < ZERO ? '0 (cannot change)' : `${farMax.toFixed(3)} e`}</dd>
          </div>
          <div>
            <dt>total charge, neutral / cation</dt>
            <dd>
              {sum(cmp.neutral).toFixed(3)} / {sum(cmp.cation).toFixed(3)} e
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Adding a proton changes the environments of the atoms inside the rings only; the {farIdx.length} atoms on the left see exactly
          the same neighbours within {RC_4G} Å in both molecules. A charge predicted from the environment cannot change there. Rescaling to
          the right total charge moves every atom by the same amount, wherever the charge physically goes. Charge equilibration
          minimises one global energy with the total charge fixed, so the extra charge spreads along the whole chain, unevenly. The
          weights are random: the pattern of Δq means nothing, whether it is zero does.
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

function ChainScene({ theme, mol, q, qMax }: { theme: SceneTheme; mol: Molecule; q: number[]; qMax: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 14, 60);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = mol.positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    mol.positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      radii[i] = mol.symbols[i] === 'C' ? 0.42 : 0.3;
      colors.set(chargeColour(theme, q[i]!, qMax), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.2), ao: new Float32Array(n).fill(1), count: n };
  }, [mol, q, qMax, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const c = linear(theme.ruleStrong);
    const out: Segment[] = [];
    for (let i = 0; i < mol.positions.length; i++)
      for (let j = i + 1; j < mol.positions.length; j++) {
        const a = mol.positions[i]!;
        const b = mol.positions[j]!;
        if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1.5) out.push({ a, b, color: c, radius: 0.07 });
      }
    return out;
  }, [mol, theme]);
  // The cation's two hydrogens (or the neutral's one) are the changed end.
  const ends: Vec3[] = mol.positions.slice(N + 1);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      {ends.map((p, k) => (
        <Shell key={k} center={p} radius={RC_4G} color={theme.phys} opacity={0.07} />
      ))}
    </>
  );
}

function ChargeBars({ theme, neutral, cation, qMax }: { theme: SceneTheme; neutral: number[]; cation: number[]; qMax: number }) {
  const W = 380;
  const H = 170;
  const m = { l: 40, r: 8, t: 10, b: 30 };
  const n = cation.length;
  const bw = (W - m.l - m.r) / n;
  const y = (v: number) => m.t + ((qMax - v) / (2 * qMax)) * (H - m.t - m.b);
  const labels = ['H', ...Array.from({ length: N }, () => 'C'), 'H', 'H'];
  return (
    <figure className="lab-plot lr-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Charge on each atom, hollow for C10H2 and filled for C10H3+.">
        {[-qMax, 0, qMax].map((v) => (
          <g key={v}>
            <line className={v === 0 ? 'axis' : 'grid'} x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={m.l - 4} y={y(v) + 3} textAnchor="end">
              {v === 0 ? '0' : v.toFixed(2)}
            </text>
          </g>
        ))}
        {cation.map((qc, i) => {
          const x0 = m.l + i * bw;
          const qn = neutral[i];
          return (
            <g key={i}>
              {qn !== undefined ? (
                <rect x={x0 + bw * 0.12} y={Math.min(y(qn), y(0))} width={bw * 0.36} height={Math.max(1, Math.abs(y(qn) - y(0)))} fill="none" style={{ stroke: theme.inkMuted }} />
              ) : null}
              <rect x={x0 + bw * 0.52} y={Math.min(y(qc), y(0))} width={bw * 0.36} height={Math.max(1, Math.abs(y(qc) - y(0)))} style={{ fill: qc < 0 ? theme.arch : theme.fail }} />
              <text className="tick" x={x0 + bw / 2} y={H - m.b + 12} textAnchor="middle">
                {labels[i]}
              </text>
              <text className="tick lr-id" x={x0 + bw / 2} y={H - m.b + 23} textAnchor="middle">
                {i + 1}
              </text>
            </g>
          );
        })}
        <text className="axis-label" x={12} y={m.t + 4} transform={`rotate(-90 12 ${m.t + 4})`} textAnchor="end">
          q (e)
        </text>
      </svg>
      <figcaption className="lr-cap">Charge per atom: hollow C₁₀H₂, filled C₁₀H₃⁺ (atom 13 exists only in the cation).</figcaption>
    </figure>
  );
}

function DeltaStrip({ theme, all, unchanged, active }: { theme: SceneTheme; all: Record<ChargeModel, ReturnType<typeof compareChain>>; unchanged: boolean[]; active: ChargeModel }) {
  const W = 380;
  const rowH = 26;
  const m = { l: 120, r: 8, t: 6, b: 18 };
  const n = unchanged.length;
  const H = m.t + MODELS.length * rowH + m.b;
  const bw = (W - m.l - m.r) / n;
  const dMax = Math.max(...MODELS.flatMap((x) => all[x.id].delta.map(Math.abs)));
  const lastSame = unchanged.lastIndexOf(true);
  return (
    <figure className="lab-plot lr-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Change of charge on protonation, Δq, for atoms 1 to 11 under the three charge models; the shaded atoms have identical environments in both molecules.">
        <rect className="lr-same" x={m.l} y={m.t} width={(lastSame + 1) * bw} height={MODELS.length * rowH} />
        {MODELS.map((x, k) => {
          const yc = m.t + k * rowH + rowH / 2;
          return (
            <g key={x.id} opacity={x.id === active ? 1 : 0.55}>
              <text className="lr-row" x={m.l - 6} y={yc + 4} textAnchor="end">
                {x.name}
              </text>
              <line className="axis" x1={m.l} x2={W - m.r} y1={yc} y2={yc} />
              {all[x.id].delta.map((d, i) => {
                const h = (Math.abs(d) / dMax) * (rowH / 2 - 2);
                return Math.abs(d) < ZERO ? (
                  <circle key={i} cx={m.l + (i + 0.5) * bw} cy={yc} r={1.6} style={{ fill: theme.inkMuted }} />
                ) : (
                  <rect key={i} x={m.l + i * bw + bw * 0.2} y={d > 0 ? yc - h : yc} width={bw * 0.6} height={Math.max(1, h)} style={{ fill: d < 0 ? theme.arch : theme.fail }} />
                );
              })}
            </g>
          );
        })}
        {unchanged.map((_, i) => (
          <text key={i} className="tick" x={m.l + (i + 0.5) * bw} y={H - 5} textAnchor="middle">
            {i + 1}
          </text>
        ))}
      </svg>
      <figcaption className="lr-cap">
        Δq = q(C₁₀H₃⁺) − q(C₁₀H₂) for atoms 1–11; shaded: same environment in both. A dot is exactly zero.
      </figcaption>
    </figure>
  );
}
