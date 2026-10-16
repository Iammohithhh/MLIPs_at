import { useMemo, useState } from 'react';
import './PhysicsLab.css';
import data from '~/data/benchmarks/catalysis-kempen2025.json';
import { useSceneTheme, type SceneTheme } from '~/three/theme';

// PUBLISHED RESULT: Supplementary Tables S2–S15 of Kempen, Cheula & Andersen (arXiv:2512.16702v1),
// snapshotted by pipeline/atlas_pipeline/catalysis.py. Colours are ranks *within one task*; tasks are
// never averaged into an overall ranking.

const SHORT: Record<string, string> = {
  s2: 'Perovskite vacancy',
  s3: 'ZrO₂ Eₐ',
  s4: 'ZrO₂ ΔE',
  s5: 'ZrO₂ adsorbates',
  s6: 'ZrO₂ TSs',
  s7: 'RWGS Eₐ',
  s8: 'RWGS ΔE',
  s9: 'RWGS adsorbates',
  s10: 'RWGS TSs',
  s11: 'Cluster ZPE',
  s12: 'Formate ads.',
  s13: 'Formate ads., relaxed',
  s14: 'CO₂+H Eₐ',
  s15: 'CO₂+H ΔE',
};

const GROUPS: Array<{ key: string; label: string; test: (t: string) => boolean }> = [
  { key: 'omat-combined', label: 'OMat24 → MPtrj + sAlex', test: (t) => t.startsWith('OMat24 →') },
  { key: 'omat', label: 'OMat24', test: (t) => t.startsWith('OMat24') },
  { key: 'mp-alex', label: 'MPtrj + (s)Alex', test: (t) => /^MPTrj \+ s?Alex/.test(t) },
  { key: 'mp', label: 'MPtrj', test: (t) => t === 'MPTrj' },
  { key: 'oc', label: 'Open Catalyst (OC20/22/25, AQCat)', test: (t) => /OC2|AQCat/.test(t) },
  { key: 'other', label: 'Other', test: () => true },
];
const groupOf = (training: string) => GROUPS.find((g) => g.test(training))!.key;

type Task = (typeof data.tasks)[number];
const rmseOf = (t: Task) => t.rmse as unknown as Record<string, number | undefined>;
const maxaeOf = (t: Task) => t.maxae as unknown as Record<string, number | null | undefined>;
const MLIPS = data.models.filter((m) => m.kind === 'mlip');

function groupColor(t: SceneTheme, g: string) {
  return ({ omat: t.phys, 'omat-combined': t.arch, 'mp-alex': t.lit, mp: t.caveat, oc: t.fail } as Record<string, string>)[g] ?? t.inkMuted;
}

/** Rank percentile of each model within one task (0 = lowest RMSE). */
function percentiles(task: Task) {
  const rm = rmseOf(task);
  const keys = MLIPS.map((m) => m.key).filter((k) => rm[k] != null);
  keys.sort((a, b) => rm[a]! - rm[b]!);
  const out = new Map<string, number>();
  keys.forEach((k, i) => out.set(k, keys.length > 1 ? i / (keys.length - 1) : 0));
  return { out, best: keys[0]!, n: keys.length };
}

export default function CatalysisHeatmap() {
  const theme = useSceneTheme();
  const [sortBy, setSortBy] = useState('s2');
  const [hover, setHover] = useState<{ key: string; task: string } | null>(null);
  const ranks = useMemo(() => Object.fromEntries(data.tasks.map((t) => [t.id, percentiles(t)])), []);
  const sortTask = data.tasks.find((t) => t.id === sortBy)!;
  const rows = useMemo(() => {
    const rm = rmseOf(sortTask);
    return [...MLIPS].sort((a, b) => (rm[a.key] ?? Infinity) - (rm[b.key] ?? Infinity));
  }, [sortTask]);

  const CW = 40;
  const RH = 12;
  const LW = 250;
  const HEAD = 118;
  const W = LW + data.tasks.length * CW + 12;
  const H = HEAD + rows.length * RH + 18;
  // Diverging: teal (best) → neutral surface (median) → red (worst) within each task.
  const ramp = (p: number) =>
    p < 0.5
      ? `color-mix(in oklab, ${theme.phys} ${Math.round((1 - 2 * p) * 100)}%, ${theme.ruleStrong})`
      : `color-mix(in oklab, ${theme.fail} ${Math.round((2 * p - 1) * 100)}%, ${theme.ruleStrong})`;
  const hm = hover ? MLIPS.find((m) => m.key === hover.key) : null;
  const ht = hover ? data.tasks.find((t) => t.id === hover.task) : null;
  const label = (m: (typeof MLIPS)[number]) => `${m.family}${m.variant ? ' ' + m.variant : ''}`;

  return (
    <div className="lab">
      <div className="lab-plot" style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ minWidth: 720 }} role="img" aria-label="Rank of each of 80 foundation potentials within each of 14 catalysis tasks (Kempen et al., Tables S2–S15). Rows sorted by the selected task.">
          {data.tasks.map((t, j) => {
            const x = LW + j * CW + CW / 2;
            const on = t.id === sortBy;
            return (
              <g key={t.id} style={{ cursor: 'pointer' }} onClick={() => setSortBy(t.id)} tabIndex={0} role="button" aria-pressed={on} aria-label={`Sort by ${t.label}`} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSortBy(t.id)}>
                <rect x={x - CW / 2 + 1} y={2} width={CW - 2} height={HEAD - 8} rx={4} style={{ fill: on ? theme.rule : 'transparent' }} />
                <text transform={`translate(${x + 4},${HEAD - 12}) rotate(-60)`} className="tick" style={{ fontWeight: on ? 700 : 400, fill: on ? theme.ink : theme.inkMuted }}>
                  {SHORT[t.id] ?? t.id}
                </text>
              </g>
            );
          })}
          {rows.map((m, i) => {
            const y = HEAD + i * RH;
            const g = groupOf(m.training);
            return (
              <g key={m.key}>
                <rect x={0} y={y + 2} width={4} height={RH - 3} style={{ fill: groupColor(theme, g) }} />
                <text x={10} y={y + RH - 3} className="tick" style={{ fontSize: 9, fill: hover?.key === m.key ? theme.ink : theme.inkMuted }}>
                  {label(m).slice(0, 26)} · {m.training.slice(0, 20)}
                </text>
                {data.tasks.map((t, j) => {
                  const r = ranks[t.id]!;
                  const p = r.out.get(m.key);
                  const best = r.best === m.key;
                  return (
                    <rect
                      key={t.id}
                      x={LW + j * CW + 1}
                      y={y + 1}
                      width={CW - 2}
                      height={RH - 2}
                      style={{ fill: p == null ? 'transparent' : ramp(p), stroke: best ? theme.ink : 'none', strokeWidth: best ? 2.5 : 0, opacity: hover && hover.key !== m.key && hover.task !== t.id ? 0.55 : 1 }}
                      onMouseEnter={() => setHover({ key: m.key, task: t.id })}
                      onMouseLeave={() => setHover(null)}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="lab-legend" style={{ marginTop: '0.5rem' }}>
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> lowest RMSE in that task</span>
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> highest</span>
        <span className="item"><span className="swatch" style={{ background: 'transparent', outline: `1.5px solid ${theme.ink}` }} /> best in that task</span>
        {GROUPS.map((g) => (
          <span key={g.key} className="item"><span className="swatch" style={{ background: groupColor(theme, g.key) }} /> {g.label}</span>
        ))}
      </div>
      <p className="lab-note" aria-live="polite">
        {hm && ht
          ? `${label(hm)} (${hm.training}) — ${ht.label}: RMSE ${rmseOf(ht)[hm.key] ?? '—'} eV, max error ${maxaeOf(ht)[hm.key] ?? '—'} eV (${ht.table}).`
          : `Sorted by “${sortTask.label}” (${sortTask.table}); click a column to sort by another task. Outlined cells mark each task’s lowest RMSE — follow them and the winner keeps changing.`}
      </p>
    </div>
  );
}
