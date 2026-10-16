import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import {
  UMA_EXPERTS,
  UMA_TASKS,
  atomCount,
  makeToyMole,
  mergedApply,
  mergedWeights,
  mixtureApply,
  moleCost,
  relFrob,
  routingCoefficients,
  scaled,
  toyEnergy,
  type Composition,
  type SystemSpec,
  type UmaTask,
} from '~/lib/models/mole';

// UMA's Mixture of Linear Experts (Wood et al. 2025, Eqs. 1–2) with fairchem's routing inputs, as
// an exact toy (src/lib/models/mole.ts): expert coefficients from composition, task, charge and
// spin; experts merged into one matrix per system; what a vacancy and the UMA-S-1.2 composition
// mean do to the merged weights.

interface Preset {
  key: string;
  label: string;
  composition: Composition;
  task: UmaTask;
  vacancy: string;
}
const PRESETS: Preset[] = [
  { key: 'nacl', label: 'NaCl, rock salt (8-atom cell)', composition: { Na: 4, Cl: 4 }, task: 'omat', vacancy: 'Na' },
  { key: 'zno', label: 'ZnO, wurtzite (4-atom cell)', composition: { Zn: 2, O: 2 }, task: 'omat', vacancy: 'O' },
  { key: 'cu', label: 'Cu, fcc (4-atom cell)', composition: { Cu: 4 }, task: 'omat', vacancy: 'Cu' },
  { key: 'co-pt', label: 'CO on a Pt slab (16 Pt)', composition: { Pt: 16, C: 1, O: 1 }, task: 'oc20', vacancy: 'Pt' },
  { key: 'water', label: 'water molecule', composition: { H: 2, O: 1 }, task: 'omol', vacancy: 'H' },
];
const NS = [1, 2, 3, 4, 5, 6, 7, 8];
const fmtComp = (c: Composition) => Object.entries(c).filter(([, k]) => k > 0).map(([z, k]) => `${z}${k === 1 ? '' : toSub(k)}`).join('');
const toSub = (n: number) => String(n).split('').map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');

export default function UmaMole() {
  const theme = useSceneTheme();
  const ids = useId();
  const model = useMemo(() => makeToyMole(), []);
  const [pi, setPi] = useState(0);
  const preset = PRESETS[pi]!;
  const [task, setTask] = useState<UmaTask>(preset.task);
  const [charge, setCharge] = useState(0);
  const [spin, setSpin] = useState(1);
  const [n, setN] = useState(1);
  const [vacancy, setVacancy] = useState(false);
  const [bug, setBug] = useState(false);

  const pristine = useMemo(() => scaled(preset.composition, n), [preset, n]);
  const composition = useMemo(() => {
    const c = { ...pristine };
    if (vacancy) c[preset.vacancy] = c[preset.vacancy]! - 1;
    return c;
  }, [pristine, vacancy, preset]);
  const csd = task === 'omol' ? { charge, spin } : {};
  const spec: SystemSpec = { composition, task, ...csd };
  const alpha = useMemo(() => routingCoefficients(model, spec, bug), [model, composition, task, charge, spin, bug]);
  const alphaRef = useMemo(() => routingCoefficients(model, { composition: pristine, task, ...csd }, bug), [model, pristine, task, charge, spin, bug]);
  const W = useMemo(() => mergedWeights(model, alpha), [model, alpha]);
  const Wref = useMemo(() => mergedWeights(model, alphaRef), [model, alphaRef]);
  const mergeErr = useMemo(() => {
    let e = 0;
    for (const z of Object.keys(composition)) {
      const x = model.env.get(z)!;
      const a = mixtureApply(model, alpha, x);
      const b = mergedApply(model, W, x);
      for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i]! - b[i]!));
    }
    return e;
  }, [model, alpha, W, composition]);
  const shift = vacancy ? relFrob(W, Wref) : 0;
  const N = atomCount(composition);
  const curves = useMemo(
    () =>
      ([false, true] as const).map((self) =>
        NS.map((k) => {
          const c = scaled(preset.composition, k);
          return toyEnergy(model, { composition: c, task, ...csd }, self) / atomCount(c);
        }),
      ),
    [model, preset, task, charge, spin],
  );
  const cost = moleCost(128, UMA_EXPERTS, Math.max(1, N) * 30);
  const sum = alpha.reduce((a, v) => a + v, 0);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <Bars theme={theme} alpha={alpha} ghost={vacancy ? alphaRef : null} title={`${fmtComp(composition)} · task ${task}${task === 'omol' ? ` · charge ${charge}, spin ${spin}` : ''}`} />
        <PerAtom theme={theme} curves={curves} n={n} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-p`}>
            <span>system</span>
            <select
              id={`${ids}-p`}
              value={pi}
              onChange={(e) => {
                const k = Number(e.target.value);
                setPi(k);
                setTask(PRESETS[k]!.task);
                setN(1);
                setVacancy(false);
              }}
            >
              {PRESETS.map((p, k) => (
                <option key={p.key} value={k}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <div className="seg" role="group" aria-label="Task (DFT setting)">
            <span className="seg-label">task</span>
            {UMA_TASKS.map((t) => (
              <button key={t} type="button" aria-pressed={task === t} onClick={() => setTask(t)}>
                {t}
              </button>
            ))}
          </div>
          {task === 'omol' ? (
            <>
              <label htmlFor={`${ids}-q`}>
                <span>total charge</span>
                <input id={`${ids}-q`} type="range" min={-2} max={2} step={1} value={charge} onChange={(e) => setCharge(Number(e.target.value))} />
                <output>{charge > 0 ? `+${charge}` : charge}</output>
              </label>
              <label htmlFor={`${ids}-m`}>
                <span>spin multiplicity</span>
                <input id={`${ids}-m`} type="range" min={1} max={4} step={1} value={spin} onChange={(e) => setSpin(Number(e.target.value))} />
                <output>{spin}</output>
              </label>
            </>
          ) : null}
          <label htmlFor={`${ids}-n`}>
            <span>supercell (copies of the cell)</span>
            <input id={`${ids}-n`} type="range" min={1} max={8} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} />
            <output>×{n}</output>
          </label>
          <div className="seg" role="group" aria-label="Defects and variants">
            <button type="button" aria-pressed={vacancy} onClick={() => setVacancy((v) => !v)}>
              remove one {preset.vacancy}
            </button>
            <button type="button" aria-pressed={bug} onClick={() => setBug((v) => !v)}>
              UMA-S-1.2 mean (÷ N + 1)
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>atoms</dt>
            <dd>{N}</dd>
          </div>
          <div>
            <dt>sum of the 32 coefficients</dt>
            <dd>{sum.toFixed(3)}</dd>
          </div>
          <div className="is-good">
            <dt>merged − expert by expert</dt>
            <dd>{mergeErr < 1e-12 ? '< 10⁻¹²' : mergeErr.toExponential(1)}</dd>
          </div>
          <div className={shift > 0 ? 'is-alert' : undefined}>
            <dt>‖ΔW*‖ / ‖W*‖ from the vacancy</dt>
            <dd>{vacancy ? (shift === 0 ? '0 (same reduced composition)' : shift.toExponential(1)) : '—'}</dd>
          </div>
          <div>
            <dt>cost, expert by expert ÷ merged</dt>
            <dd>{(cost.mixture / cost.merged).toFixed(0)}×</dd>
          </div>
        </dl>
        <p className="lab-note">
          Random, untrained weights: which experts dominate means nothing, but the structure is UMA’s. The coefficients depend on the
          reduced composition, the task and, for omol, charge and spin, never on positions, so a supercell routes exactly like its
          unit cell and an MD run keeps one merged matrix. A vacancy in a compound changes the reduced composition, so the defect cell
          gets different weights from the bulk, by an amount that shrinks like 1/N. Removing an atom from an element changes nothing.
          Cost: one MoLE layer of width 128 applied to 30 edges per atom; merging costs one pass over the 32 experts.
        </p>
      </div>
    </div>
  );
}

function Bars({ theme, alpha, ghost, title }: { theme: SceneTheme; alpha: Float64Array; ghost: Float64Array | null; title: string }) {
  const W = 520;
  const H = 300;
  const M = { l: 44, r: 12, t: 34, b: 40 };
  const mx = Math.max(...alpha, ...(ghost ?? []), 0.05);
  const bw = (W - M.l - M.r) / alpha.length;
  const y = (v: number) => M.t + (1 - v / mx) * (H - M.t - M.b);
  const ticks = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1].filter((v) => v <= mx);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Expert mixing coefficients for ${title}.`}>
        <text x={M.l} y={16} className="axis-label">expert coefficients α_k — {title}</text>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toFixed(2)}</text>
          </g>
        ))}
        {Array.from(alpha, (a, k) => (
          <rect key={k} x={M.l + k * bw + 1} y={y(a)} width={bw - 2} height={y(0) - y(a)} style={{ fill: theme.arch }}>
            <title>{`expert ${k + 1}: α = ${a.toFixed(4)}`}</title>
          </rect>
        ))}
        {ghost
          ? Array.from(ghost, (a, k) => (
              <line key={k} x1={M.l + k * bw + 1} x2={M.l + (k + 1) * bw - 1} y1={y(a)} y2={y(a)} style={{ stroke: theme.fail, strokeWidth: 2 }} />
            ))
          : null}
        {[1, 8, 16, 24, 32].map((k) => (
          <text key={k} x={M.l + (k - 0.5) * bw} y={H - M.b + 16} textAnchor="middle" className="tick">{k}</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">expert k (of 32)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> this system</span>
        {ghost ? <span className="item"><span className="swatch" style={{ background: theme.fail }} /> without the vacancy</span> : null}
      </figcaption>
    </figure>
  );
}

function PerAtom({ theme, curves, n }: { theme: SceneTheme; curves: number[][]; n: number }) {
  const W = 520;
  const H = 300;
  const M = { l: 64, r: 16, t: 34, b: 40 };
  const all = curves.flat();
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (hi - lo < 1e-3) {
    lo -= 5e-4;
    hi += 5e-4;
  }
  const pad = 0.1 * (hi - lo);
  const x = (k: number) => M.l + ((k - 1) / 7) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - M.t - M.b);
  const path = (c: number[]) => c.map((v, k) => `${k ? 'L' : 'M'}${x(k + 1).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const span = hi - lo + 2 * pad;
  const step = 10 ** Math.floor(Math.log10(span / 3));
  const ticks: number[] = [];
  for (let v = Math.ceil((lo - pad) / step) * step; v <= hi + pad; v += step) ticks.push(v);
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Toy energy per atom of perfect supercells of increasing size, with the composition mean and with the mean that divides by N plus one.">
        <text x={M.l} y={16} className="axis-label">toy energy per atom of a perfect supercell (arb. units)</text>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toFixed(digits)}</text>
          </g>
        ))}
        <path d={path(curves[1]!)} className="curve" style={{ stroke: theme.fail }} />
        <path d={path(curves[0]!)} className="curve" style={{ stroke: theme.phys }} />
        <line x1={x(n)} x2={x(n)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1 }} strokeDasharray="3 3" />
        {NS.map((k) => (
          <text key={k} x={x(k)} y={H - M.b + 16} textAnchor="middle" className="tick">×{k}</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">supercell size</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> mean over atoms (intensive)</span>
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> sum ÷ (N + 1), as in UMA-S-1.2</span>
      </figcaption>
    </figure>
  );
}
