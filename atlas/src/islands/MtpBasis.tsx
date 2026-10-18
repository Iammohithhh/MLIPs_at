import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Mtp.css';
import templates from '~/data/models/mtp-templates.json';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { describe, enumerateBasis, maxMomentsUsed, momentComponents, radialFunctionsUsed, type BasisFunction } from '~/lib/models/mtp';

// Basis functions as contraction graphs (src/lib/models/mtp.ts): one node per moment M_{μ,ν}, one edge per
// pair of contracted indices. The Atlas enumeration (≤ 4 moments, μ ≤ 5, ν ≤ 10) runs live; the bars are the
// sizes stored in mlip-2's untrained templates (atlas/src/data/models/mtp-templates.json).

const LEVELS = templates.templates.map((t) => t.level);
const LIVE_MAX = 16;

export default function MtpBasis() {
  const theme = useSceneTheme();
  const ids = useId();
  const [lev, setLev] = useState(10);
  const [onlyNew, setOnlyNew] = useState(true);
  const basis = useMemo(() => enumerateBasis(lev), [lev]);
  const live = useMemo(() => LEVELS.filter((L) => L <= LIVE_MAX).map((L) => enumerateBasis(L).length), []);
  const t = templates.templates.find((x) => x.level === lev)!;
  const shown = onlyNew ? basis.filter((b) => b.level === lev) : basis;
  const params = t.alpha_scalar_moments + 1 + t.radial_funcs_count * t.radial_basis_size;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <CountPlot theme={theme} lev={lev} live={live} />
        <div className="mtp-side">
          <div className="lab-controls">
            <label htmlFor={`${ids}-l`}>
              <span>
                lev<sub>max</sub>
              </span>
              <input id={`${ids}-l`} type="range" min={2} max={LIVE_MAX} step={2} value={lev} onChange={(e) => setLev(Number(e.target.value))} />
              <output>{lev}</output>
            </label>
            <div className="seg" role="group" aria-label="Which basis functions to draw">
              <button type="button" aria-pressed={onlyNew} onClick={() => setOnlyNew(true)}>
                new at this level
              </button>
              <button type="button" aria-pressed={!onlyNew} onClick={() => setOnlyNew(false)}>
                all ≤ lev<sub>max</sub>
              </button>
            </div>
          </div>
          <dl className="lab-readout" aria-live="polite">
            <div className={basis.length === t.alpha_scalar_moments ? 'is-good' : 'is-alert'}>
              <dt>basis functions: Atlas enumeration / mlip-2 template</dt>
              <dd>
                {basis.length} / {t.alpha_scalar_moments}
              </dd>
            </div>
            <div className={momentComponents(basis) === t.alpha_index_basic_count ? 'is-good' : 'is-alert'}>
              <dt>moment components to compute / template</dt>
              <dd>
                {momentComponents(basis)} / {t.alpha_index_basic_count}
              </dd>
            </div>
            <div>
              <dt>radial functions (values of μ)</dt>
              <dd>{radialFunctionsUsed(basis)}</dd>
            </div>
            <div>
              <dt>free parameters (single species)</dt>
              <dd>
                {params}
                <small>
                  {' '}
                  = {t.alpha_scalar_moments} + 1 + {t.radial_funcs_count} × {t.radial_basis_size}
                </small>
              </dd>
            </div>
            <div>
              <dt>highest body order</dt>
              <dd>{maxMomentsUsed(basis) + 1}</dd>
            </div>
          </dl>
        </div>
      </div>
      <ul className="mtp-cards" aria-label={`${shown.length} basis functions drawn as contraction graphs`}>
        {shown.map((b) => (
          <li key={b.key}>
            <MiniGraph b={b} theme={theme} />
            <span className="mtp-card-name">{describe(b)}</span>
            <span className="mtp-card-level">level {b.level} · {b.mu.length + 1}-body</span>
          </li>
        ))}
      </ul>
      <p className="lab-note">
        Each node is a moment M<sub>μ,ν</sub>, each line one pair of contracted indices, so a node has as many lines as its rank ν and
        every index is used: the result is a scalar. Separate pieces multiply. Raise lev<sub>max</sub> and the count grows roughly
        geometrically; the bars are read from the template files of mlip-2 (commit {templates.source.commit.slice(0, 8)}), the dots are
        this page’s enumeration, which matches them at every level.
      </p>
    </div>
  );
}

function CountPlot({ theme, lev, live }: { theme: SceneTheme; lev: number; live: number[] }) {
  const W = 520;
  const H = 340;
  const x0 = 56;
  const y0 = 28;
  const w = W - x0 - 16;
  const h = H - y0 - 58;
  const lo = 0;
  const hi = 4;
  const bw = w / LEVELS.length;
  const px = (k: number) => x0 + bw * (k + 0.5);
  const py = (v: number) => y0 + ((hi - Math.log10(Math.max(v, 1))) / (hi - lo)) * h;
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Number of MTP basis functions against level, on a log scale, from 1 at level 2 to ${templates.templates.at(-1)!.alpha_scalar_moments} at level 28.`}>
        <text x={x0} y={y0 - 12} className="axis-label">basis functions (log scale)</text>
        {[0, 1, 2, 3, 4].map((e) => (
          <g key={e}>
            <line x1={x0} x2={x0 + w} y1={py(10 ** e)} y2={py(10 ** e)} className="grid" />
            <text x={x0 - 6} y={py(10 ** e) + 4} textAnchor="end" className="tick">
              {(10 ** e).toLocaleString('en')}
            </text>
          </g>
        ))}
        {templates.templates.map((t, k) => (
          <g key={t.level}>
            <rect
              x={px(k) - bw * 0.36}
              width={bw * 0.72}
              y={py(t.alpha_scalar_moments)}
              height={y0 + h - py(t.alpha_scalar_moments)}
              style={{ fill: t.level === lev ? theme.arch : theme.ruleStrong, opacity: t.level === lev ? 0.9 : 0.6 }}
            />
            <text x={px(k)} y={y0 + h + 15} textAnchor="middle" className="tick">
              {t.level}
            </text>
            {t.level === lev ? (
              <text x={px(k)} y={py(t.alpha_scalar_moments) - 8} textAnchor="middle" className="tick">
                {t.alpha_scalar_moments}
              </text>
            ) : null}
          </g>
        ))}
        {live.map((n, k) => (
          <circle key={k} cx={px(k)} cy={py(n)} r={3.6} style={{ fill: theme.verified, stroke: theme.canvas, strokeWidth: 1.2 }} />
        ))}
        <text x={x0 + w} y={y0 + h + 34} textAnchor="end" className="tick">
          lev<tspan dy="3" fontSize="9">max</tspan>
        </text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item">
          <span className="swatch" style={{ background: theme.ruleStrong }} /> mlip-2 template (alpha_scalar_moments)
        </span>
        <span className="item">
          <span className="swatch" style={{ background: theme.verified }} /> this page’s enumeration (to level {LIVE_MAX})
        </span>
      </figcaption>
    </figure>
  );
}

const SPOTS: Record<number, [number, number][]> = {
  1: [[50, 42]],
  2: [
    [22, 42],
    [78, 42],
  ],
  3: [
    [50, 16],
    [20, 66],
    [80, 66],
  ],
  4: [
    [22, 16],
    [78, 16],
    [78, 68],
    [22, 68],
  ],
};

function MiniGraph({ b, theme }: { b: BasisFunction; theme: SceneTheme }) {
  const k = b.mu.length;
  const P = SPOTS[k]!;
  const pairs: [number, number][] = [];
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) pairs.push([i, j]);
  return (
    <svg viewBox="0 0 100 86" className="mtp-graph" aria-hidden="true">
      {pairs.map(([i, j], e) => {
        const m = b.adj[e]!;
        if (!m) return null;
        const [ax, ay] = P[i]!;
        const [bx, by] = P[j]!;
        const dx = bx - ax;
        const dy = by - ay;
        const L = Math.hypot(dx, dy);
        const nx = -dy / L;
        const ny = dx / L;
        return Array.from({ length: m }, (_, c) => {
          const off = (c - (m - 1) / 2) * 7;
          const cx = (ax + bx) / 2 + nx * off * 1.6;
          const cy = (ay + by) / 2 + ny * off * 1.6;
          return <path key={`${e}-${c}`} d={`M${ax},${ay} Q${cx},${cy} ${bx},${by}`} style={{ fill: 'none', stroke: theme.arch, strokeWidth: 1.4 }} />;
        });
      })}
      {P.map(([x, y], i) => (
        <g key={i}>
          <circle cx={x} cy={y} r={9.5} style={{ fill: theme.surface, stroke: b.nu[i] ? theme.arch : theme.phys, strokeWidth: 1.6 }} />
          <text x={x} y={y + 3.4} textAnchor="middle" style={{ fill: theme.ink, fontSize: 9.5, fontFamily: 'var(--font-sans)' }}>
            {b.mu[i]},{b.nu[i]}
          </text>
        </g>
      ))}
    </svg>
  );
}
