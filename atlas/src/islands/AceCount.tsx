import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Ace.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { countBasis, countInvariants, environmentBasis, multisetBasis, randomlyRotated, type AceBasis, type NL, type Vec3 } from '~/lib/models/ace';
import { mulberry32 } from '~/lib/chem/rng';

// Build the invariants of one (n, l) tuple the way Dusson et al. do (coupling trees, then a rank
// test on random environments), compare with the character formula and with the paper's Table 1,
// and check invariance on a rotated neighbourhood. Right: how the complete basis grows with degree.

interface Preset {
  label: string;
  l: number[];
  n: number[];
  /** Dusson et al. (2022) Table 1: #RPI for this (l, n), and #RI for this l. */
  rpi: number;
  ri: number;
}

// n is 0-based in the paper; shown and used here as n + 1.
const PRESETS: Preset[] = [
  { label: 'l = 2222, n = 0001', l: [2, 2, 2, 2], n: [0, 0, 0, 1], rpi: 1, ri: 5 },
  { label: 'l = 2222, n = 0012', l: [2, 2, 2, 2], n: [0, 0, 1, 2], rpi: 3, ri: 5 },
  { label: 'l = 2233, n = 0000', l: [2, 2, 3, 3], n: [0, 0, 0, 0], rpi: 3, ri: 5 },
  { label: 'l = 11222, n = 00012', l: [1, 1, 2, 2, 2], n: [0, 0, 0, 1, 2], rpi: 6, ri: 9 },
  { label: 'l = 11222, n = 11001', l: [1, 1, 2, 2, 2], n: [1, 1, 0, 0, 1], rpi: 4, ri: 9 },
  { label: 'l = 22222, n = 00000', l: [2, 2, 2, 2, 2], n: [0, 0, 0, 0, 0], rpi: 1, ri: 16 },
];

const LMAX = 4;
const NMAX = 4;

function invarianceError(nls: NL[]): { kept: number; err: number } {
  const { functions } = multisetBasis(nls);
  if (!functions.length) return { kept: 0, err: 0 };
  const nmax = Math.max(...nls.map((v) => v.n));
  const lmax = Math.max(...nls.map((v) => v.l));
  const basis: AceBasis = { spec: { nmax, lmax, numax: nls.length, degree: [], wL: 1 }, functions, reports: [] };
  const rand = mulberry32(17);
  const env: Vec3[] = Array.from({ length: 10 }, () => [4 * rand() - 2, 4 * rand() - 2, 4 * rand() - 2]);
  const R = { rc: 5, lambda: 2 };
  const a = environmentBasis(basis, env, R);
  const b = environmentBasis(basis, randomlyRotated(env, 5), R);
  let err = 0;
  let scale = 0;
  a.forEach((x, k) => ((err = Math.max(err, Math.abs(x - b[k]!))), (scale = Math.max(scale, Math.abs(x)))));
  return { kept: functions.length, err: err / (scale || 1) };
}

export default function AceCount() {
  const theme = useSceneTheme();
  const ids = useId();
  const [tuple, setTuple] = useState<NL[]>(PRESETS[1]!.l.map((l, k) => ({ l, n: PRESETS[1]!.n[k]! + 1 })));
  const [preset, setPreset] = useState<number | null>(1);
  const nu = tuple.length;
  const sorted = useMemo(() => [...tuple].sort((a, b) => a.n - b.n || a.l - b.l), [tuple]);
  const built = useMemo(() => multisetBasis(sorted).report, [sorted]);
  const check = useMemo(() => invarianceError(sorted), [sorted]);
  const ri = useMemo(() => countInvariants(sorted.map((v, k) => ({ n: k + 1, l: v.l }))), [sorted]);
  const products = sorted.reduce((p, v) => p * (2 * v.l + 1), 1);
  const lsum = sorted.reduce((s, v) => s + v.l, 0);
  const p = preset !== null ? PRESETS[preset]! : null;

  const setSlot = (k: number, v: Partial<NL>) => {
    setPreset(null);
    setTuple((t) => t.map((x, q) => (q === k ? { ...x, ...v } : x)));
  };
  const setNu = (m: number) => {
    setPreset(null);
    setTuple((t) => (m > t.length ? [...t, ...Array.from({ length: m - t.length }, () => ({ n: 1, l: 1 }))] : t.slice(0, m)));
  };

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="ace-count-wrap">
          <div className="ace-presets" role="group" aria-label="Rows of Dusson et al. Table 1">
            {PRESETS.map((q, k) => (
              <button
                key={q.label}
                type="button"
                aria-pressed={preset === k}
                onClick={() => {
                  setPreset(k);
                  setTuple(q.l.map((l, i) => ({ l, n: q.n[i]! + 1 })));
                }}
              >
                {q.label}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Correlation order ν">
            <span className="seg-label">ν</span>
            {[2, 3, 4, 5].map((m) => (
              <button key={m} type="button" aria-pressed={nu === m} onClick={() => setNu(m)}>
                {m}
              </button>
            ))}
          </div>
          <div className="ace-tuple">
            {tuple.map((v, k) => (
              <div className="ace-slot" key={k}>
                <label htmlFor={`${ids}-n${k}`}>
                  n
                  <select id={`${ids}-n${k}`} value={v.n} onChange={(e) => setSlot(k, { n: Number(e.target.value) })}>
                    {Array.from({ length: NMAX }, (_, i) => i + 1).map((x) => (
                      <option key={x} value={x}>
                        {x}
                      </option>
                    ))}
                  </select>
                </label>
                <label htmlFor={`${ids}-l${k}`}>
                  l
                  <select id={`${ids}-l${k}`} value={v.l} onChange={(e) => setSlot(k, { l: Number(e.target.value) })}>
                    {Array.from({ length: LMAX + 1 }, (_, i) => i).map((x) => (
                      <option key={x} value={x}>
                        {x}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ))}
          </div>
          <dl className="ace-steps" aria-live="polite">
            <div>
              <dt>products Π A</dt>
              <dd>
                {products.toLocaleString('en')}
                <small>one per m-tuple, Π(2l + 1)</small>
              </dd>
            </div>
            <div>
              <dt>coupling trees</dt>
              <dd>
                {built.trees}
                <small>{lsum % 2 ? 'Σl odd: no O(3) invariant' : `${built.candidates} real candidates`}</small>
              </dd>
            </div>
            <div>
              <dt>#RI</dt>
              <dd>
                {ri}
                <small>rotation invariants if every n differed</small>
              </dd>
            </div>
            <div className="is-key">
              <dt>#RPI kept</dt>
              <dd>
                {built.kept}
                <small>
                  independent after the rank test; characters predict {built.predicted}
                </small>
              </dd>
            </div>
          </dl>
          <p className="ace-paper">
            {p ? (
              <>
                Dusson et al. Table 1 for this row: #RI = {p.ri}, #RPI = {p.rpi}.{' '}
                {p.ri === ri && p.rpi === built.kept ? <strong>Both reproduced.</strong> : <span className="is-miss">Not reproduced.</span>}{' '}
              </>
            ) : null}
            Rotation test on a random 10-atom neighbourhood: largest relative change{' '}
            {check.kept ? check.err.toExponential(1) : '— (no invariant)'}.
          </p>
        </div>
        <Growth theme={theme} />
      </div>
      <p className="lab-note">
        Repeating a label shrinks the space: products of identical factors are symmetric under swapping them, so coupling
        trees that differ only by such a swap give the same function. The rank test finds this numerically, and the
        character integral (Eq. 5) predicts it exactly. With every (n, l) different the two counts agree (Dusson et al.,
        Proposition 7).
      </p>
    </div>
  );
}

const DEGREES = Array.from({ length: 13 }, (_, k) => k + 2);

function Growth({ theme }: { theme: SceneTheme }) {
  const rows = useMemo(
    () => DEGREES.map((D) => ({ D, c: countBasis({ nmax: D, lmax: D, numax: 4, degree: [D, D, D, D], wL: 1 }) })),
    [],
  );
  const W = 400;
  const H = 300;
  const M = { l: 50, r: 64, t: 18, b: 38 };
  const x = (D: number) => M.l + ((D - DEGREES[0]!) / (DEGREES[DEGREES.length - 1]! - DEGREES[0]!)) * (W - M.l - M.r);
  const y0 = 0;
  const y1 = 5;
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(1, v)) - y0) / (y1 - y0)) * (H - M.t - M.b);
  const colors = [theme.inkMuted, theme.phys, theme.arch, theme.lit];
  return (
    <figure className="lab-plot ace-count-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Number of independent ACE basis functions against total degree for correlation orders 1 to 4, logarithmic scale; dashed lines count the products before symmetrisation.">
        {[1, 10, 100, 1000, 10000, 100000].map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {t.toLocaleString('en')}
            </text>
          </g>
        ))}
        {DEGREES.filter((D) => D % 2 === 0).map((D) => (
          <text key={D} x={x(D)} y={H - M.b + 14} textAnchor="middle" className="tick">
            {D}
          </text>
        ))}
        {[0, 1, 2, 3].map((k) => {
          const pts = rows.filter((r) => r.c[k]!.invariants > 0);
          const path = (f: (r: (typeof rows)[number]) => number) => pts.map((r, i) => `${i ? 'L' : 'M'}${x(r.D).toFixed(1)},${y(f(r)).toFixed(1)}`).join('');
          const last = pts[pts.length - 1];
          return (
            <g key={k}>
              <path d={path((r) => r.c[k]!.products)} className="curve" style={{ stroke: colors[k], strokeDasharray: '4 3', opacity: 0.7 }} />
              <path d={path((r) => r.c[k]!.invariants)} className="curve" style={{ stroke: colors[k] }} />
              {last ? (
                <text x={x(last.D) + 4} y={y(last.c[k]!.invariants) + 4} className="tick" style={{ fill: colors[k] }}>
                  ν = {k + 1}: {last.c[k]!.invariants.toLocaleString('en')}
                </text>
              ) : null}
            </g>
          );
        })}
        <text x={14} y={(H - M.b + M.t) / 2} transform={`rotate(-90 14 ${(H - M.b + M.t) / 2})`} textAnchor="middle" className="axis-label">
          basis functions (log)
        </text>
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          total degree D = max Σ(n + l)
        </text>
      </svg>
      <p className="ace-paper">Solid: independent invariants. Dashed: symmetric products Π A before coupling. One species, n ≥ 1.</p>
    </figure>
  );
}
