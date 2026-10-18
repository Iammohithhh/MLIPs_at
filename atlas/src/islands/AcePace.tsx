import { useState } from 'react';
import './PhysicsLab.css';
import './Ace.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import data from '~/data/models/ace-pace-cu.json';

// The released PACE copper potential (Lysogorskiy et al. 2021), evaluated on CPU by the authors'
// python-ace (pipeline/atlas_pipeline/ace_pace.py): its basis by rank, and its energy split into the
// linear density ρ₁, the square-root embedding F(ρ₂) and the core repulsion.

type View = 'dimer' | 'eos';

export default function AcePace() {
  const theme = useSceneTheme();
  const [view, setView] = useState<View>('dimer');
  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <Ranks theme={theme} />
        <div className="ace-side">
          <div className="seg" role="group" aria-label="Curve shown">
            <button type="button" aria-pressed={view === 'dimer'} onClick={() => setView('dimer')}>
              Cu₂ dimer, split by term
            </button>
            <button type="button" aria-pressed={view === 'eos'} onClick={() => setView('eos')}>
              fcc, hcp, bcc E(V)
            </button>
          </div>
          {view === 'dimer' ? <Dimer theme={theme} /> : <Eos theme={theme} />}
        </div>
      </div>
      <p className="lab-note">
        {view === 'dimer'
          ? 'At the bond length (2.2 Å here) most of the binding comes from the embedded density F(ρ₂), with ρ₁ adding a little. Inside about 2 Å ρ₁ turns steeply repulsive and builds the short-range wall; the core repulsion is negligible down to about 1 Å.'
          : 'Energy per atom relative to the fcc minimum. The potential orders the close-packed phases fcc < hcp < bcc at their minima on this grid.'}
      </p>
    </div>
  );
}

function Ranks({ theme }: { theme: SceneTheme }) {
  const rows = data.summary.by_rank;
  const W = 400;
  const H = 250;
  const M = { l: 104, r: 128, t: 16, b: 30 };
  const max = Math.max(...rows.map((r) => r.functions));
  const bh = (H - M.t - M.b) / rows.length;
  const x = (v: number) => M.l + (v / max) * (W - M.l - M.r);
  return (
    <figure className="lab-plot ace-count-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Basis functions of the released Cu PACE potential by rank: ${rows.map((r) => `rank ${r.rank}, ${r.functions}`).join('; ')}.`}>
        {rows.map((r, k) => {
          const y = M.t + k * bh;
          return (
            <g key={r.rank}>
              <text x={M.l - 6} y={y + bh / 2 + 4} textAnchor="end" className="tick">
                ν = {r.rank} ({r.rank + 1}-body)
              </text>
              <rect x={M.l} y={y + 3} width={Math.max(2, x(r.functions) - M.l)} height={bh - 6} rx={3} style={{ fill: theme.arch, opacity: 0.8 }} />
              <text x={x(r.functions) + 5} y={y + bh / 2 + 4} className="tick">
                {r.functions} · n ≤ {r.n_max}, l ≤ {r.l_max}
              </text>
            </g>
          );
        })}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">
          basis functions ({data.summary.total_functions} in all, two coefficients each)
        </text>
      </svg>
      <p className="ace-paper">
        {data.summary.radial_basis} radial basis (λ = {data.summary.radial_parameter_lambda}), {data.summary.nradbase} functions for ν = 1 and{' '}
        {data.summary.nradmax} for ν ≥ 2, cutoff {data.summary.cutoff} Å, l<sub>max</sub> = {data.summary.lmax}; embedding{' '}
        {data.summary.embedding} with E = ρ₁ + F(ρ₂), m = {data.summary.densities[1]!.m}; core repulsion {data.summary.core_repulsion[0]} eV·Å, λ ={' '}
        {data.summary.core_repulsion[1]} Å⁻².
      </p>
    </figure>
  );
}

function Frame({
  theme,
  xs,
  series,
  lo,
  hi,
  xLabel,
  yLabel,
  label,
}: {
  theme: SceneTheme;
  xs: number[];
  series: Array<{ name: string; ys: number[]; color: string; dash?: string; xs?: number[] }>;
  lo: number;
  hi: number;
  xLabel: string;
  yLabel: string;
  label: string;
}) {
  const W = 420;
  const H = 250;
  const M = { l: 46, r: 10, t: 26, b: 34 };
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const x = (v: number) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - M.t - M.b);
  const ticks = Array.from({ length: 5 }, (_, k) => lo + (k * (hi - lo)) / 4);
  const xt = Array.from({ length: 5 }, (_, k) => x0 + (k * (x1 - x0)) / 4);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {Number(t.toFixed(2))}
            </text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 14} textAnchor="middle" className="tick">
            {Number(t.toFixed(1))}
          </text>
        ))}
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} style={{ stroke: theme.ruleStrong }} />
        {series.map((s) => (
          <path key={s.name} d={s.ys.map((v, k) => `${k ? 'L' : 'M'}${x((s.xs ?? xs)[k]!).toFixed(1)},${y(v).toFixed(1)}`).join('')} className="curve" style={{ stroke: s.color, strokeDasharray: s.dash }} />
        ))}
        <text x={M.l} y={14} className="tick">
          {series.map((s, k) => (
            <tspan key={s.name} dx={k ? 10 : 0} style={{ fill: s.color }}>
              {s.dash ? '- -' : '—'} {s.name}
            </tspan>
          ))}
        </text>
        <text x={12} y={(H - M.b + M.t) / 2} transform={`rotate(-90 12 ${(H - M.b + M.t) / 2})`} textAnchor="middle" className="axis-label">
          {yLabel}
        </text>
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          {xLabel}
        </text>
      </svg>
    </figure>
  );
}

function Dimer({ theme }: { theme: SceneTheme }) {
  const pts = data.dimer.filter((p) => p.r >= 1.0);
  return (
    <Frame
      theme={theme}
      xs={pts.map((p) => p.r)}
      lo={-6}
      hi={8}
      xLabel="r (Å)"
      yLabel="E − 2E_atom (eV)"
      label="Cu dimer energy of the released PACE potential against distance, with its linear density, square-root embedding and core-repulsion parts."
      series={[
        { name: 'total', ys: pts.map((p) => p.total), color: theme.ink },
        { name: 'ρ₁', ys: pts.map((p) => p.linear), color: theme.phys, dash: '5 3' },
        { name: 'F(ρ₂)', ys: pts.map((p) => p.embedding), color: theme.arch, dash: '5 3' },
        { name: 'core', ys: pts.map((p) => p.core), color: theme.fail, dash: '2 3' },
      ]}
    />
  );
}

function Eos({ theme }: { theme: SceneTheme }) {
  const e0 = Math.min(...data.eos.fcc.map((p) => p.total));
  const kinds = [
    ['fcc', theme.arch],
    ['hcp', theme.phys],
    ['bcc', theme.fail],
  ] as const;
  const xs = kinds.flatMap(([k]) => data.eos[k].map((p) => p.volume));
  return (
    <Frame
      theme={theme}
      xs={xs}
      lo={-0.02}
      hi={0.3}
      xLabel="volume per atom (Å³)"
      yLabel="E − E_fcc,min (eV/atom)"
      label="Energy per atom against volume for fcc, ideal hcp and bcc copper from the released PACE potential."
      series={kinds.map(([k, color]) => ({ name: k, xs: data.eos[k].map((p) => p.volume), ys: data.eos[k].map((p) => p.total - e0), color }))}
    />
  );
}
