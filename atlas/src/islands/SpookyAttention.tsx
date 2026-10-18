import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Spooky.css';
import { carbonChain } from '~/lib/models/long-range';
import { attentionExact, attentionFavor, drawOmega, makeToySpooky, maxDiff, runToy, type FeatureKind, type Matrix, type Molecule } from '~/lib/models/spookynet';
import { Legend, LinePlot } from './spooky/LinePlot';

// Fig. 4: exact softmax attention (Eq. 19) against its FAVOR+ estimate (Eq. 20) on the queries, keys and values
// of the toy SpookyNet's second nonlocal block for C₁₀H₂ and C₁₀H₃⁺; live.

const MODEL = makeToySpooky();
const MS = [6, 12, 24, 48, 96, 192, 384, 768];
const DRAWS = 16;
const Z_OF: Record<string, number> = { H: 1, C: 6 };

function chain(protonated: boolean): { mol: Molecule; labels: string[] } {
  const c = carbonChain(10, protonated);
  return { mol: { numbers: c.symbols.map((s) => Z_OF[s]!), positions: c.positions }, labels: c.symbols };
}

export default function SpookyAttention() {
  const id = useId();
  const [prot, setProt] = useState(true);
  const [mi, setMi] = useState(2);
  const [seed, setSeed] = useState(1);
  const [kind, setKind] = useState<FeatureKind>('positive');
  const [beta, setBeta] = useState(2);
  const { mol, labels } = useMemo(() => chain(prot), [prot]);
  const qkv = useMemo(() => runToy(MODEL, mol, { Q: prot ? 1 : 0 }).attention[1]!, [mol, prot]);
  const F = qkv.Q[0]!.length;
  const scaled = useMemo(() => ({ Q: qkv.Q.map((q) => q.map((v) => beta * v)), K: qkv.K.map((k) => k.map((v) => beta * v)), V: qkv.V }), [qkv, beta]);
  const exact = useMemo(() => attentionExact(scaled.Q, scaled.K, scaled.V), [scaled]);
  const m = MS[mi]!;
  const approx = useMemo(() => attentionFavor(scaled.Q, scaled.K, scaled.V, drawOmega(1000 * seed + m, m, F), kind), [scaled, m, seed, kind, F]);
  const curve = useMemo(
    () =>
      (['positive', 'trig'] as FeatureKind[]).map((k) => ({
        k,
        pts: MS.map((mm) => {
          let e = 0;
          let neg = 0;
          for (let s = 0; s < DRAWS; s++) {
            const r = attentionFavor(scaled.Q, scaled.K, scaled.V, drawOmega(77 + 31 * s + mm, mm, F), k);
            e += Math.min(10, maxDiff(r.weights, exact.weights)) / DRAWS;
            if (r.minDenominator <= 0) neg++;
          }
          return { mm, e, neg };
        }),
      })),
    [scaled, exact, F],
  );
  const err = maxDiff(approx.weights, exact.weights);
  const N = mol.numbers.length;
  const wMax = Math.max(...exact.weights.flat());
  const wMin = Math.min(...exact.weights.flat());
  const pos0 = curve[0]!.pts[0]!.e;
  const negTrig = curve[1]!.pts.reduce((s, p) => s + p.neg, 0);

  return (
    <div className="lab">
      <div className="sp-grid3">
        <Heatmap title="Exact softmax (Eq. 19)" w={exact.weights} labels={labels} vmax={wMax} vmin={wMin} />
        <Heatmap title={`${kind === 'positive' ? 'FAVOR+' : 'Trigonometric'} estimate, m = ${m} (Eq. 20)`} w={approx.weights} labels={labels} vmax={wMax} vmin={wMin} />
        <div>
          <p className="sp-h">Mean largest weight error, {DRAWS} draws</p>
          <LinePlot
            H={200}
            x={{ min: 6, max: 768, ticks: [6, 24, 96, 384], label: 'random features m', log: true }}
            y={{ min: 1e-3, max: 10, ticks: [1e-3, 1e-2, 1e-1, 1, 10], label: 'max |Δa_ij|', log: true, fmt: (v) => (v >= 1 ? String(v) : v.toExponential(0)) }}
            series={[
              { key: 'guide', points: MS.map((mm) => [mm, pos0 * Math.sqrt(6 / mm)] as [number, number]), color: 'var(--atlas-ink-faint)', dashed: true, thin: true },
              { key: 'trig', points: curve[1]!.pts.map((p) => [p.mm, Math.max(1e-3, p.e)] as [number, number]), color: 'var(--atlas-fail)' },
              { key: 'pos', points: curve[0]!.pts.map((p) => [p.mm, Math.max(1e-3, p.e)] as [number, number]), color: 'var(--atlas-arch)' },
            ]}
            cursor={m}
            ariaLabel="Mean largest error of the estimated attention weights against the number of random features, on log axes, for positive and trigonometric features, with a one-over-square-root-of-m guide."
          />
          <Legend
            items={[
              { label: 'positive (FAVOR+)', color: 'var(--atlas-arch)' },
              { label: 'trigonometric', color: 'var(--atlas-fail)' },
              { label: '∝ 1/√m', color: 'var(--atlas-ink-faint)', dashed: true },
            ]}
          />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Molecule">
            <button type="button" aria-pressed={!prot} onClick={() => setProt(false)}>
              C₁₀H₂
            </button>
            <button type="button" aria-pressed={prot} onClick={() => setProt(true)}>
              C₁₀H₃⁺
            </button>
          </div>
          <label htmlFor={`${id}-m`}>
            <span>random features m</span>
            <input id={`${id}-m`} type="range" min={0} max={MS.length - 1} step={1} value={mi} onChange={(e) => setMi(Number(e.target.value))} />
            <output>{m}</output>
          </label>
          <label htmlFor={`${id}-b`}>
            <span>sharpen q, k ×</span>
            <input id={`${id}-b`} type="range" min={1} max={4} step={0.25} value={beta} onChange={(e) => setBeta(Number(e.target.value))} />
            <output>{beta.toFixed(2)}</output>
          </label>
          <div className="seg" role="group" aria-label="Random features">
            <button type="button" aria-pressed={kind === 'positive'} onClick={() => setKind('positive')}>
              positive
            </button>
            <button type="button" aria-pressed={kind === 'trig'} onClick={() => setKind('trig')}>
              sin/cos
            </button>
          </div>
          <div className="seg" role="group" aria-label="Random draw">
            <button type="button" aria-pressed={false} onClick={() => setSeed((s) => s + 1)}>
              redraw ω (draw {seed})
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={err > 0.1 ? 'is-alert' : undefined}>
            <dt>largest |estimate − exact| weight</dt>
            <dd>
              {err.toFixed(3)} <small>largest exact weight {wMax.toFixed(3)}</small>
            </dd>
          </div>
          <div className={approx.minDenominator <= 0 ? 'is-alert' : 'is-good'}>
            <dt>smallest normaliser φ(q_i)·Σ_j φ(k_j)</dt>
            <dd>{approx.minDenominator.toPrecision(3)}</dd>
          </div>
          <div>
            <dt>draws with a normaliser ≤ 0 (all m, sin/cos)</dt>
            <dd>
              {negTrig} / {DRAWS * MS.length}
            </dd>
          </div>
          <div>
            <dt>multiply–adds per head, exact vs FAVOR+</dt>
            <dd>
              2N²F = {(2 * N * N * F).toLocaleString('en-GB')} vs ≈ 4NmF = {(4 * N * m * F).toLocaleString('en-GB')} <small>(N = {N})</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          <strong>Toy model:</strong> queries, keys and values come from the toy SpookyNet’s second nonlocal block (F = {F}), so the pattern
          is not a trained model’s; the comparison between estimators does not depend on it. Exact attention builds the N × N matrix; Eq. (20)
          never does (it forms K̂ᵀV, m × F, once), so its cost grows as N: about 4NmF multiply–adds (two projections, K̂ᵀV, Q̂ times that) against 2N²F, fewer once N exceeds about 2m. Both estimators are unbiased for
          each kernel entry exp(q·k/√F); positive features keep every estimated weight and normaliser positive, while sin/cos features can
          return negative weights and a normaliser near or below zero once attention sharpens (slider; SpookyNet itself has no such knob). The
          paper uses m = F and draws ω once, at initialisation.
        </p>
      </div>
    </div>
  );
}

function Heatmap({ title, w, labels, vmax, vmin }: { title: string; w: Matrix; labels: string[]; vmax: number; vmin: number }) {
  const n = w.length;
  const S = 260;
  const pad = 22;
  const c = (S - pad) / n;
  return (
    <div>
      <p className="sp-h">{title}</p>
      <svg className="sp-heat" viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`${title}: attention weight of atom i (row) on atom j (column).`}>
        {w.map((row, i) =>
          row.map((v, j) => (
            <rect
              key={`${i}-${j}`}
              x={pad + j * c}
              y={pad + i * c}
              width={c + 0.3}
              height={c + 0.3}
              style={{ fill: v < 0 ? 'var(--atlas-fail)' : 'var(--atlas-arch)', fillOpacity: v < 0 ? Math.min(1, -v / (vmax - vmin)) : Math.max(0.04, Math.min(1, (v - vmin) / (vmax - vmin))) }}
            />
          )),
        )}
        {labels.map((l, i) => (
          <g key={i}>
            <text className="tick" x={pad - 4} y={pad + (i + 0.5) * c + 3} textAnchor="end">
              {l}
            </text>
            <text className="tick" x={pad + (i + 0.5) * c} y={pad - 5} textAnchor="middle">
              {l}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
