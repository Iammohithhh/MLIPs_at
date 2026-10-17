import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dpa.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { BIAS_DATASETS, FRAME_SPECS, fitBias, framesFor, type BiasFit, type Frame } from '~/lib/models/dpa';

// The energy bias of DPA-2 (Eqs. 5–6) and DPA-3 (Eq. 3): per-element reference energies e(Z)
// solved by least squares from frame compositions and total energies. Each frame is a line
// c_H e(H) + c_O e(O) = E in the (e(H), e(O)) plane; the fit is where the lines meet
// (src/lib/models/dpa.ts). The two "DFT setups" and their energies are invented.

const WATER = FRAME_SPECS.filter((f) => f.counts[0] === 2 * f.counts[1]);
const EXTRAS = FRAME_SPECS.filter((f) => f.counts[0] !== 2 * f.counts[1]);
type Mode = 'a' | 'b' | 'both';

export default function DpaBias() {
  const theme = useSceneTheme();
  const ids = useId();
  const [mode, setMode] = useState<Mode>('a');
  const [extra, setExtra] = useState<Set<string>>(() => new Set());
  const specs = useMemo(() => [...WATER, ...EXTRAS.filter((f) => extra.has(f.label))], [extra]);
  const sets = mode === 'both' ? BIAS_DATASETS : BIAS_DATASETS.filter((d) => d.id === mode);
  const frames = useMemo(() => sets.map((d) => framesFor(d, specs)), [sets, specs]);
  const fit = useMemo(() => fitBias(frames.flat()), [frames]);
  const ref = sets[0]!;
  const h2 = framesFor(ref, FRAME_SPECS.filter((f) => f.label === 'H₂'))[0]!;
  // If the network had learned the interaction part exactly, the whole error is in the bias.
  const h2err = (h2.counts[0]! * (fit.bias[0]! - ref.mu[0]) + h2.counts[1]! * (fit.bias[1]! - ref.mu[1])) / 2;
  const toggle = (label: string) =>
    setExtra((s) => {
      const n = new Set(s);
      if (n.has(label)) n.delete(label);
      else n.add(label);
      return n;
    });

  return (
    <div className="lab">
      <div className="lab-split">
        <div className="dpa-side">
          <BiasPlane theme={theme} frames={frames} fit={fit} mode={mode} />
        </div>
        <div className="dpa-side">
          <div className="seg" role="group" aria-label="Training data">
            <button type="button" aria-pressed={mode === 'a'} onClick={() => setMode('a')}>
              setup A
            </button>
            <button type="button" aria-pressed={mode === 'b'} onClick={() => setMode('b')}>
              setup B
            </button>
            <button type="button" aria-pressed={mode === 'both'} onClick={() => setMode('both')}>
              A + B, one bias
            </button>
          </div>
          <fieldset className="dpa-frames">
            <legend>Frames (water clusters always in)</legend>
            {EXTRAS.map((f) => (
              <label key={f.label} htmlFor={`${ids}-${f.label}`}>
                <input id={`${ids}-${f.label}`} type="checkbox" checked={extra.has(f.label)} onChange={() => toggle(f.label)} />
                {f.label}
              </label>
            ))}
          </fieldset>
          <dl className="lab-readout" aria-live="polite">
            <div className={fit.rank < 2 ? 'is-alert' : 'is-good'}>
              <dt>rank of the composition matrix</dt>
              <dd>
                {fit.rank} of 2 <small>{fit.rank < 2 ? 'e(H) and e(O) not identifiable' : 'unique solution'}</small>
              </dd>
            </div>
            <div>
              <dt>fitted e(H), e(O)</dt>
              <dd>
                {fit.bias[0]!.toFixed(2)}, {fit.bias[1]!.toFixed(2)} eV
              </dd>
            </div>
            <div>
              <dt>reference energies of {mode === 'both' ? 'A / B' : ref.label}</dt>
              <dd>{sets.map((d) => `${d.mu[0]}, ${d.mu[1]}`).join(' / ')} eV</dd>
            </div>
            <div className={fit.rmsPerAtom > 5 ? 'is-alert' : undefined}>
              <dt>RMS residual of the fitted frames</dt>
              <dd>{fit.rmsPerAtom.toFixed(2)} eV/atom</dd>
            </div>
            <div className={Math.abs(h2err) > 5 ? 'is-alert' : undefined}>
              <dt>H₂ error from the bias alone ({ref.label})</dt>
              <dd>{h2err.toFixed(2)} eV/atom</dd>
            </div>
          </dl>
        </div>
      </div>
      <div className="lab-panel">
        <p className="lab-note">
          Water clusters all have two H per O, so every one gives a line of the same slope: only 2 e(H) + e(O) is fixed, and least
          squares picks the shortest bias on that line, here wildly wrong per element. The residual stays small because the training
          frames never ask the parts separately; an H₂ molecule does. Add H₂, O₂ or H₂O₂ and the lines cross. With two setups in one fit
          the lines of A and B cannot meet, which is why DPA-2 gives each pre-training dataset its own fitting network and DPA-3 its own
          bias e<sub>m</sub>(Z). The residual left with the right bias is the interaction energy, which the network learns.
        </p>
      </div>
    </div>
  );
}

function BiasPlane({ theme, frames, fit, mode }: { theme: SceneTheme; frames: Frame[][]; fit: BiasFit; mode: Mode }) {
  const ids = useId();
  const W = 340;
  const H = 300;
  const m = { l: 54, r: 12, t: 12, b: 38 };
  const pts = [fit.bias as [number, number], ...BIAS_DATASETS.filter((d) => mode === 'both' || d.id === mode).map((d) => d.mu)];
  let [x0, x1] = [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0]))];
  let [y0, y1] = [Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))];
  const span = Math.max(x1 - x0, y1 - y0, 6);
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  [x0, x1] = [cx - 0.62 * span, cx + 0.62 * span];
  [y0, y1] = [cy - 0.62 * span, cy + 0.62 * span];
  const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v: number) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const line = (f: Frame) => {
    const [a, b] = f.counts as [number, number];
    if (Math.abs(b) >= Math.abs(a)) return [x0, (f.energy - a * x0) / b, x1, (f.energy - a * x1) / b];
    return [(f.energy - b * y0) / a, y0, (f.energy - b * y1) / a, y1];
  };
  const tick = (lo: number, hi: number) => {
    const step = 10 ** Math.floor(Math.log10((hi - lo) / 3));
    const k = [1, 2, 5].map((s) => s * step).find((s) => (hi - lo) / s <= 5) ?? 10 * step;
    const out: number[] = [];
    for (let v = Math.ceil(lo / k) * k; v <= hi; v += k) out.push(v);
    return out;
  };
  const col = [theme.arch, theme.caveat];
  return (
    <figure className="lab-plot dpa-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Each training frame as a line c_H e(H) + c_O e(O) = E. Fitted bias at e(H) = ${fit.bias[0]!.toFixed(1)} eV, e(O) = ${fit.bias[1]!.toFixed(1)} eV.`}>
        <defs>
          <clipPath id={`${ids}-clip`}>
            <rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} />
          </clipPath>
        </defs>
        {tick(x0, x1).map((v) => (
          <g key={`x${v}`}>
            <line className="grid" x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} />
            <text className="tick" x={X(v)} y={H - m.b + 13} textAnchor="middle">
              {+v.toFixed(3)}
            </text>
          </g>
        ))}
        {tick(y0, y1).map((v) => (
          <g key={`y${v}`}>
            <line className="grid" x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} />
            <text className="tick" x={m.l - 4} y={Y(v) + 3} textAnchor="end">
              {+v.toFixed(3)}
            </text>
          </g>
        ))}
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle">
          e(H) (eV)
        </text>
        <text className="axis-label" x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
          e(O) (eV)
        </text>
        <g clipPath={`url(#${ids}-clip)`}>
          {frames.map((fs, d) =>
            fs.map((f, k) => {
              const [ax, ay, bx, by] = line(f);
              return <line key={`${d}-${k}`} className="dpa-frame-line" x1={X(ax!)} y1={Y(ay!)} x2={X(bx!)} y2={Y(by!)} style={{ stroke: col[mode === 'b' ? 1 : d]!, opacity: 0.75 }} />;
            }),
          )}
          {BIAS_DATASETS.filter((d) => mode === 'both' || d.id === mode).map((d) => (
            <g key={d.id}>
              <circle cx={X(d.mu[0])} cy={Y(d.mu[1])} r={6} style={{ fill: 'none', stroke: theme.ink, strokeWidth: 1.6 }} />
              <text className="dpa-pt-label" x={X(d.mu[0]) + (X(d.mu[0]) > W / 2 ? -9 : 9)} y={Y(d.mu[1]) - 9} textAnchor={X(d.mu[0]) > W / 2 ? 'end' : 'start'}>
                {d.label} reference
              </text>
            </g>
          ))}
          <circle cx={X(fit.bias[0]!)} cy={Y(fit.bias[1]!)} r={5} style={{ fill: theme.fail, stroke: theme.canvas, strokeWidth: 1.5 }} />
          <text className="dpa-pt-label" x={X(fit.bias[0]!) + (X(fit.bias[0]!) > W / 2 ? -8 : 8)} y={Y(fit.bias[1]!) + 16} textAnchor={X(fit.bias[0]!) > W / 2 ? 'end' : 'start'} style={{ fill: theme.fail }}>
            least-squares fit
          </text>
        </g>
      </svg>
      <figcaption className="dpa-cap">One line per frame: every bias on it reproduces that frame’s energy up to its interaction part.</figcaption>
    </figure>
  );
}
