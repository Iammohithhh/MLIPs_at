import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './SevenNet.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { TOY_RC, dimerCurves, evaluateSerial, makeSystem, makeToyModel, neighbourList, type FidelityMode, type System2D } from '~/lib/models/sevennet';

// SevenNet-MF concatenates a one-hot fidelity to the scalar input of its linear layers and uses
// fidelity-wise energy shifts (Kim et al. 2024, Fig. 1c,d). A one-hot into a linear layer is a
// fidelity-dependent bias. Here the same toy network (src/lib/models/sevennet.ts) evaluates one
// structure at two fidelities with that bias either only in the energy shift or in every linear
// layer. The fidelity weights are random, not fitted: the point is what they can change.

const sys: System2D = makeSystem();
const nbrs = neighbourList(sys, TOY_RC);
const S = 30;
const RS = Array.from({ length: 56 }, (_, k) => 0.7 + k * 0.02);

export default function SevenNetFidelity() {
  const theme = useSceneTheme();
  const ids = useId();
  const [mode, setMode] = useState<FidelityMode>('all');
  const model = useMemo(() => makeToyModel(3), []);
  const lo = useMemo(() => evaluateSerial(model, sys, 0, mode, nbrs), [model, mode]);
  const hi = useMemo(() => evaluateSerial(model, sys, 1, mode, nbrs), [model, mode]);
  const dimer = useMemo(() => dimerCurves(model, [0, 1], RS, mode), [model, mode]);
  let fmax = 0;
  for (const f of lo.forces) fmax = Math.max(fmax, Math.abs(f));
  let dmax = 0;
  for (let i = 0; i < sys.n; i++) dmax = Math.max(dmax, Math.hypot(hi.forces[2 * i]! - lo.forces[2 * i]!, hi.forces[2 * i + 1]! - lo.forces[2 * i + 1]!));
  // The largest change is drawn one spacing long (no arrows at all when nothing changes).
  const scale = dmax > 0 ? S / dmax : 0;
  const W = sys.width * S;
  const H = (sys.height + 0.4) * S;

  return (
    <div className="lab">
      <div className="sn-fid">
        <div className="sn-canvas">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Arrows show how much each atom's force changes between fidelity 0 and fidelity 1. Largest change ${dmax.toPrecision(2)}.`}>
            <defs>
              <marker id={`${ids}-h`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M0,0 L6,3 L0,6 z" style={{ fill: theme.caveat }} />
              </marker>
            </defs>
            {Array.from({ length: sys.n }, (_, i) => {
              const x = sys.pos[2 * i]! * S;
              const y = (0.2 + sys.pos[2 * i + 1]!) * S;
              const dx = (hi.forces[2 * i]! - lo.forces[2 * i]!) * scale;
              const dy = (hi.forces[2 * i + 1]! - lo.forces[2 * i + 1]!) * scale;
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r={(sys.species[i] ? 0.3 : 0.24) * S} style={{ fill: sys.species[i] ? theme.arch : theme.dark ? '#5b6676' : '#9aa3af' }} />
                  {Math.hypot(dx, dy) > 0.5 ? <line x1={x} y1={y} x2={x + dx} y2={y + dy} markerEnd={`url(#${ids}-h)`} style={{ stroke: theme.caveat, strokeWidth: 1.8 }} /> : null}
                </g>
              );
            })}
          </svg>
          <p className="sn-caption-line">
            Arrows: F(fidelity 1) − F(fidelity 0), the largest drawn one spacing long. Large blue and small grey discs are the two
            species.
          </p>
        </div>
        <DimerPlot theme={theme} lo={dimer.lo} hi={dimer.hi} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Where the fidelity enters">
            <button type="button" aria-pressed={mode === 'shift'} onClick={() => setMode('shift')}>
              energy shift only
            </button>
            <button type="button" aria-pressed={mode === 'all'} onClick={() => setMode('all')}>
              every linear layer + shift (SevenNet-MF)
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>E(fidelity 1) − E(fidelity 0)</dt>
            <dd>
              {(hi.energy - lo.energy).toFixed(3)} <small>{((hi.energy - lo.energy) / sys.n).toFixed(4)} per atom</small>
            </dd>
          </div>
          <div className={mode === 'shift' ? 'is-alert' : 'is-good'}>
            <dt>largest force change</dt>
            <dd>
              {dmax === 0 ? '0 (exactly)' : dmax.toPrecision(2)} <small>{((100 * dmax) / fmax).toFixed(1)} % of the largest force</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          A per-species shift adds the same constant to an atom’s energy wherever it sits, so the two fidelities differ by a constant per
          composition and their forces are identical, exactly, as the right panel’s parallel curves show. That is all a one-hot can do in
          a model linear in its descriptors, as Kim et al. note for a simplified Behler–Parrinello network. Inside the message-passing
          layers the bias passes through nonlinearities and is mixed with geometry, so fidelity changes the shape of the energy surface
          and the forces. Fidelity weights here are random; in SevenNet-MF they start at zero and are trained on each database.
        </p>
      </div>
    </div>
  );
}

function DimerPlot({ theme, lo, hi }: { theme: SceneTheme; lo: number[]; hi: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 300;
  const H = 230;
  const m = { l: 46, r: 12, t: 14, b: 38 };
  const all = [...lo, ...hi];
  const ymin = Math.min(...all);
  const ymax = Math.max(...all);
  const pad = 0.08 * (ymax - ymin);
  const x = (r: number) => m.l + ((r - RS[0]!) / (RS.at(-1)! - RS[0]!)) * (W - m.l - m.r);
  const y = (e: number) => m.t + (1 - (e - ymin + pad) / (ymax - ymin + 2 * pad)) * (H - m.t - m.b);
  const path = (v: number[]) => v.map((e, i) => `${i ? 'L' : 'M'}${x(RS[i]!).toFixed(1)},${y(e).toFixed(1)}`).join('');
  const ticks = [0.8, 1.2, 1.6, 1.8];
  const h = hover;
  return (
    <figure className="lab-plot sn-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Energy of a two-atom system against separation, at fidelity 0 and fidelity 1.">
        {ticks.map((r) => (
          <g key={r}>
            <line className="grid" x1={x(r)} x2={x(r)} y1={m.t} y2={H - m.b} />
            <text className="tick" x={x(r)} y={H - m.b + 14} textAnchor="middle">
              {r === TOY_RC ? 'r_c' : r.toFixed(1)}
            </text>
          </g>
        ))}
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">
          dimer separation (toy units)
        </text>
        <text className="axis-label" x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
          E (arb. units)
        </text>
        <path d={path(lo)} fill="none" style={{ stroke: theme.arch, strokeWidth: 2 }} />
        <path d={path(hi)} fill="none" style={{ stroke: theme.caveat, strokeWidth: 2 }} />
        <text x={x(RS.at(-1)!) - 2} y={y(lo.at(-1)!) + (lo.at(-1)! < hi.at(-1)! ? 14 : -6)} textAnchor="end" className="sn-direct">
          fidelity 0
        </text>
        <text x={x(RS.at(-1)!) - 2} y={y(hi.at(-1)!) + (hi.at(-1)! <= lo.at(-1)! ? 14 : -6)} textAnchor="end" className="sn-direct">
          fidelity 1
        </text>
        {h !== null ? (
          <g>
            <line x1={x(RS[h]!)} x2={x(RS[h]!)} y1={m.t} y2={H - m.b} style={{ stroke: theme.ruleStrong }} />
            <circle cx={x(RS[h]!)} cy={y(lo[h]!)} r={4} style={{ fill: theme.arch, stroke: theme.canvas, strokeWidth: 2 }} />
            <circle cx={x(RS[h]!)} cy={y(hi[h]!)} r={4} style={{ fill: theme.caveat, stroke: theme.canvas, strokeWidth: 2 }} />
          </g>
        ) : null}
        <rect
          x={m.l}
          y={m.t}
          width={W - m.l - m.r}
          height={H - m.t - m.b}
          fill="transparent"
          onMouseMove={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const px = ((e.clientX - box.left) / box.width) * W;
            const k = Math.round(((px - m.l) / (W - m.l - m.r)) * (RS.length - 1));
            setHover(Math.max(0, Math.min(RS.length - 1, k)));
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <figcaption className="sn-tip" aria-live="polite">
        {h !== null ? (
          <>
            r = {RS[h]!.toFixed(2)}: E₀ = {lo[h]!.toFixed(3)}, E₁ = {hi[h]!.toFixed(3)}, difference {(hi[h]! - lo[h]!).toFixed(3)}
          </>
        ) : (
          <>A dimer of the two species. Hover for values.</>
        )}
      </figcaption>
    </figure>
  );
}
