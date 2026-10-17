// "Trace the change" (brief §9): for the selected evaluated frame, reveal stage by stage which atoms'
// representations differ from the baseline, plotted against each atom's baseline distance from the moved
// atom. Every point is a stored float64 difference from the precomputed run; zeros (at the stored
// precision, see trace.ts) are drawn on their own line, not as small values. The step order is computational dependency, not runtime.
import { useMemo, useState } from 'react';
import type { Manifest } from '../lib/experiment';
import { baselineDistances, stageRange, summarise, tracePoints, TRACE_STAGES, type TracePoint, type TraceStageKey } from '../lib/trace';

interface Question {
  ask: string;
  /** Atoms the question is about. */
  group: (r0: number) => boolean;
  groupLabel: string;
}

function questions(rMax: number): Partial<Record<TraceStageKey, Question>> {
  return {
    layer1: { ask: `Will the layer-1 features of atoms farther than r_max = ${rMax} Å from the moved atom change?`, group: (r) => r >= rMax, groupLabel: `atoms beyond ${rMax} Å` },
    layer2: { ask: `Will the layer-2 features of atoms between ${rMax} and ${2 * rMax} Å change?`, group: (r) => r >= rMax && r < 2 * rMax, groupLabel: `atoms between ${rMax} and ${2 * rMax} Å` },
    force: { ask: `The farthest atoms in this cell (beyond ${2 * rMax} Å) keep the same site energy (the architecture cannot reach them). Will the force on them change?`, group: (r) => r >= 2 * rMax, groupLabel: `atoms beyond ${2 * rMax} Å` },
  };
}

const W = 640;
const LANE = 74;
const PAD = { l: 132, r: 14, t: 22, b: 44 };

export function TraceChange({ m, frame, atom, onFrame, onAtom }: { m: Manifest; frame: number; atom: number; onFrame: (k: number) => void; onAtom: (i: number) => void }) {
  const r0s = useMemo(() => baselineDistances(m), [m]);
  const ranges = useMemo(() => Object.fromEntries(TRACE_STAGES.map((s) => [s.key, stageRange(m, s.key, r0s)])) as Record<TraceStageKey, [number, number] | null>, [m, r0s]);
  const points = useMemo(() => Object.fromEntries(TRACE_STAGES.map((s) => [s.key, tracePoints(m, frame, s.key, r0s)])) as Record<TraceStageKey, TracePoint[]>, [m, frame, r0s]);
  const [step, setStep] = useState(0);
  const [predictOn, setPredictOn] = useState(true);
  const [guess, setGuess] = useState<Record<string, 'yes' | 'no' | undefined>>({});
  const B = m.displacement.baselineIndex;
  const d = m.displacement.values[frame] ?? 0;
  const rMaxShown = Math.max(...r0s) + 0.4;
  const x = (r: number) => PAD.l + ((r + 0.35) / (rMaxShown + 0.35)) * (W - PAD.l - PAD.r);
  const H = PAD.t + TRACE_STAGES.length * LANE + PAD.b;
  const qs = questions(m.rMax);
  const cur = TRACE_STAGES[step]!;
  const q = qs[cur.key];
  const qKey = `${frame}:${cur.key}`;
  const hidden = predictOn && q && guess[qKey] === undefined && frame !== B;

  const go = (k: number) => setStep(Math.max(0, Math.min(TRACE_STAGES.length - 1, k)));

  return (
    <section className="trace" aria-labelledby="trace-title">
      <header className="trace__head">
        <h3 id="trace-title">Trace the change</h3>
        <span className="label status-pre">Precomputed · frame {frame + 1}, d = {d >= 0 ? '+' : '−'}{Math.abs(d).toFixed(3)} Å</span>
      </header>
      <p className="trace__lede">
        Follow the displacement through the model’s dependency order: geometry, then layer 1, layer 2, site energies, forces. Each dot is one of
        the {m.structure.n} atoms in the periodic cell, placed by its distance from the moved atom.
      </p>
      {frame === B ? (
        <p className="trace__empty">
          At the baseline (d = 0) every difference is zero. <button onClick={() => onFrame(m.frames.length - 1)}>Move to d = +{m.displacement.values.at(-1)!.toFixed(3)} Å</button>{' '}
          <button onClick={() => onFrame(0)}>Move to d = −{Math.abs(m.displacement.values[0]!).toFixed(3)} Å</button>
        </p>
      ) : null}
      <div className="trace__controls" role="group" aria-label="Trace steps">
        <button onClick={() => go(step - 1)} disabled={step === 0} aria-label="Previous trace step">
          ←
        </button>
        {TRACE_STAGES.map((s, i) => (
          <button key={s.key} aria-pressed={i === step} className={i === step ? 'is-on' : i < step ? 'is-done' : ''} onClick={() => go(i)}>
            <span className="trace__n">{i + 1}</span> {s.title}
          </button>
        ))}
        <button onClick={() => go(step + 1)} disabled={step === TRACE_STAGES.length - 1} aria-label="Next trace step">
          →
        </button>
        <label className="trace__predict-toggle">
          <input type="checkbox" checked={predictOn} onChange={(e) => setPredictOn(e.target.checked)} /> Predict before revealing
        </label>
      </div>
      <p className="trace__sentence">
        <strong>{cur.title}.</strong> {cur.sentence}
      </p>
      {q && predictOn && frame !== B && (
        <Predict q={q} answer={guess[qKey]} onAnswer={(a) => setGuess((g) => ({ ...g, [qKey]: a }))} pts={points[cur.key]} />
      )}
      <figure className="trace__fig">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Change by stage against distance from the moved atom, revealed up to stage ${step + 1}`}>
          {[m.rMax, 2 * m.rMax].map((r, i) => (
            <g key={r}>
              <line x1={x(r)} x2={x(r)} y1={PAD.t - 8} y2={H - PAD.b} className="trace__cut" />
              <text x={x(r)} y={PAD.t - 11} textAnchor="middle" className="trace__cut-label">
                {i === 0 ? 'r_max' : '2 r_max'} = {r} Å
              </text>
            </g>
          ))}
          {TRACE_STAGES.map((s, li) => {
            const y0 = PAD.t + li * LANE;
            const revealed = li < step || (li === step && !hidden);
            const range = ranges[s.key];
            const sum = summarise(points[s.key]);
            const ly = (v: number) => {
              if (!range) return y0 + LANE / 2;
              const [lo, hi] = range;
              const t = (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1);
              return y0 + 8 + (1 - t) * (LANE - 30);
            };
            const zeroY = y0 + LANE - 12;
            return (
              <g key={s.key} className={`trace__lane${li === step ? ' is-on' : ''}${revealed ? '' : ' is-hidden'}`}>
                <rect x={PAD.l - 4} y={y0} width={W - PAD.l - PAD.r + 8} height={LANE - 4} className="trace__bg" />
                <text x={6} y={y0 + 16} className="trace__lane-title">
                  {li + 1}. {s.title}
                </text>
                <text x={6} y={y0 + 30} className="trace__lane-unit">
                  log scale, {s.unit}
                </text>
                <text x={6} y={zeroY + 3} className="trace__lane-unit">
                  zero →
                </text>
                {revealed && (
                  <text x={6} y={y0 + 46} className="trace__lane-sum">
                    {sum.changed} changed
                  </text>
                )}
                {revealed &&
                  points[s.key].map((p) =>
                    p.value === null ? (
                      <text key={p.atom} x={x(p.r0)} y={zeroY + 3} className="trace__missing" textAnchor="middle">
                        ×
                      </text>
                    ) : (
                      <circle
                        key={p.atom}
                        cx={x(p.r0)}
                        cy={p.value === 0 ? zeroY : ly(p.value)}
                        r={p.atom === atom ? 4.2 : p.value === 0 ? 1.8 : 2.6}
                        className={`${p.value === 0 ? 'trace__zero' : 'trace__pt'}${p.atom === atom ? ' is-sel' : ''}${p.atom === m.displacement.atom ? ' is-moved' : ''}`}
                        onClick={() => onAtom(p.atom)}
                      >
                        <title>
                          atom {p.atom}, {p.r0.toFixed(2)} Å: {p.value === 0 ? `0 (${s.zero})` : `${p.value.toPrecision(3)} ${s.unit}`}
                        </title>
                      </circle>
                    ),
                  )}
                {!revealed && (
                  <text x={(PAD.l + W) / 2} y={y0 + LANE / 2} textAnchor="middle" className="trace__lane-unit">
                    {li === step ? 'answer the question above to reveal' : 'not revealed yet'}
                  </text>
                )}
              </g>
            );
          })}
          {[0, 2, 4, 6, 8, 10, 12].map((r) => (
            <text key={r} x={x(r)} y={H - 24} textAnchor="middle" className="trace__tick">
              {r}
            </text>
          ))}
          <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" className="trace__tick">
            Å from the moved atom (baseline, minimum image)
          </text>
        </svg>
        <figcaption>
          {cur.quantity}. {sumLine(summarise(points[cur.key]))} Each lane has its own fixed log axis (smallest to largest non-zero value
          over all {m.frames.length - 1} displaced frames); lanes are different quantities and are not compared by height. Dots on the lower line are zero: {cur.zero}. Click a dot to select that atom.
        </figcaption>
      </figure>
      <p className="trace__note">
        The steps follow the order in which the model computes these quantities. The reveal is a presentation sequence: it does not measure run
        time or how fast anything travels. The reach is set by the architecture (two message-passing layers, cutoff r_max = {m.rMax} Å), and
        layer 2 here carries only scalar (l = 0) features.
      </p>
    </section>
  );
}

function sumLine(s: ReturnType<typeof summarise>): string {
  const parts = [`${s.changed} atoms changed`, `${s.exactZero} unchanged`];
  if (s.missing) parts.push(`${s.missing} without data`);
  return `${parts.join(', ')}${s.reach !== null ? `; farthest changed atom ${s.reach.toFixed(2)} Å away` : ''}.`;
}

function Predict({ q, answer, onAnswer, pts }: { q: Question; answer: 'yes' | 'no' | undefined; onAnswer: (a: 'yes' | 'no') => void; pts: TracePoint[] }) {
  const group = pts.filter((p) => q.group(p.r0));
  const s = summarise(group);
  const truth = s.changed > 0 ? 'yes' : 'no';
  return (
    <div className="trace__predict" role="group" aria-label="Predict before revealing">
      <p>{q.ask}</p>
      {answer === undefined ? (
        <div className="trace__predict-btns">
          <button onClick={() => onAnswer('yes')}>Yes, they change</button>
          <button onClick={() => onAnswer('no')}>No, unchanged</button>
        </div>
      ) : (
        <p className="trace__predict-result" role="status">
          You said {answer === 'yes' ? '“yes”' : '“no”'}. Computed: {s.changed} of {group.length} {q.groupLabel} changed
          {s.missing ? ` (${s.missing} without data)` : ''}, so the answer for this frame is <strong>{truth}</strong>. Which atoms can respond follows from the
          architecture (cutoff and number of layers); the sizes are what this checkpoint computed for this one displacement.
        </p>
      )}
    </div>
  );
}
