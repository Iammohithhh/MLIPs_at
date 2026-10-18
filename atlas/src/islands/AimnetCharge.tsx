import { useId, useState } from 'react';
import './PhysicsLab.css';
import './Aimnet.css';
import { DATA, chord, fragmentSplit } from '~/lib/models/aimnet2';

// Fig. 3: (left) the released checkpoint's energy of one benzene as a function of the total charge Q it is told,
// against the straight lines between integers that exact theory requires (precomputed); (right) NQE on two
// separated fragments, live, starting from the checkpoint's own fragment sums.

const EQ = DATA.benzeneEQ;
const bzw = DATA.dimers.find((d) => d.id === 'bzw')!;
const bz2 = DATA.dimers.find((d) => d.id === 'bz2')!;
const lastPass = (d: typeof bzw) => d.rows[d.rows.length - 1]!.passes[d.rows[d.rows.length - 1]!.passes.length - 1]!;
const PRESETS = [
  { id: 'bzw', label: 'C₆H₆·H₂O⁺', names: ['benzene', 'water'], p: lastPass(bzw), Q: 1 },
  { id: 'bz2', label: '(C₆H₆)₂⁺', names: ['benzene 1', 'benzene 2'], p: lastPass(bz2), Q: 1 },
];

export default function AimnetCharge() {
  const id = useId();
  const [Q, setQ] = useState(0.5);
  const e = EQ.find((r) => Math.abs(r.Q - Q) < 1e-6)!;
  const gap = e.E - chord(EQ, Q);

  const [preset, setPreset] = useState(0);
  const base = PRESETS[preset]!;
  const [qtA, setQtA] = useState(base.p.q_tilde[0]);
  const [qtB, setQtB] = useState(base.p.q_tilde[1]);
  const [share, setShare] = useState(base.p.f[0] / (base.p.f[0] + base.p.f[1]));
  const [Qt, setQt] = useState(base.Q);
  const load = (k: number) => {
    const p = PRESETS[k]!;
    setPreset(k);
    setQtA(p.p.q_tilde[0]);
    setQtB(p.p.q_tilde[1]);
    setShare(p.p.f[0] / (p.p.f[0] + p.p.f[1]));
    setQt(p.Q);
  };
  const [qa, qb] = fragmentSplit(qtA, qtB, share, 1 - share, Qt, 0);

  return (
    <div className="lab">
      <div className="am-grid2">
        <div>
          <p className="am-h">One benzene told its charge is Q · precomputed</p>
          <EQPlot Q={Q} />
          <ul className="am-legend" aria-hidden="true">
            <li style={{ color: 'var(--atlas-ink)' }}>
              <span className="swatch" /> released checkpoint, E(Q)
            </li>
            <li style={{ color: 'var(--atlas-verified)' }}>
              <span className="swatch is-dashed" /> straight lines between integers (exact)
            </li>
          </ul>
          <div className="lab-controls">
            <label htmlFor={`${id}-q`}>
              <span>total charge Q</span>
              <input id={`${id}-q`} type="range" min={-1} max={2} step={0.05} value={Q} onChange={(ev) => setQ(Number(ev.target.value))} />
              <output>{Q.toFixed(2)} e</output>
            </label>
          </div>
          <dl className="lab-readout" aria-live="polite">
            <div className={gap < -0.1 ? 'is-alert' : undefined}>
              <dt>E(Q) − straight line</dt>
              <dd>{gap.toFixed(2)} eV</dd>
            </div>
            <div>
              <dt>charge on each C / H</dt>
              <dd>
                {e.qC.toFixed(3)} / {e.qH.toFixed(3)} e
              </dd>
            </div>
          </dl>
        </div>
        <div>
          <p className="am-h">NQE on two separated fragments · live</p>
          <div className="seg" role="group" aria-label="Start from the checkpoint's sums for">
            {PRESETS.map((p, k) => (
              <button key={p.id} type="button" aria-pressed={preset === k} onClick={() => load(k)}>
                {p.label} at 20 Å
              </button>
            ))}
          </div>
          <p className="am-formula">
            q<sub>A</sub> = Q̃<sub>A</sub> + F<sub>A</sub>/(F<sub>A</sub> + F<sub>B</sub>) · (Q − Q̃<sub>A</sub> − Q̃<sub>B</sub>)
          </p>
          <div className="lab-controls">
            <label htmlFor={`${id}-a`}>
              <span>Q̃ on {base.names[0]}</span>
              <input id={`${id}-a`} type="range" min={-1} max={1.5} step={0.01} value={qtA} onChange={(ev) => setQtA(Number(ev.target.value))} />
              <output>{qtA.toFixed(2)}</output>
            </label>
            <label htmlFor={`${id}-b`}>
              <span>Q̃ on {base.names[1]}</span>
              <input id={`${id}-b`} type="range" min={-1} max={1.5} step={0.01} value={qtB} onChange={(ev) => setQtB(Number(ev.target.value))} />
              <output>{qtB.toFixed(2)}</output>
            </label>
            <label htmlFor={`${id}-f`}>
              <span>F share of {base.names[0]}</span>
              <input id={`${id}-f`} type="range" min={0} max={1} step={0.01} value={share} onChange={(ev) => setShare(Number(ev.target.value))} />
              <output>{share.toFixed(2)}</output>
            </label>
            <label htmlFor={`${id}-Q`}>
              <span>total Q</span>
              <input id={`${id}-Q`} type="range" min={-2} max={2} step={1} value={Qt} onChange={(ev) => setQt(Number(ev.target.value))} />
              <output>{Qt}</output>
            </label>
          </div>
          <div className="am-bars" aria-live="polite">
            {[
              [base.names[0], qa],
              [base.names[1], qb],
            ].map(([n, v]) => (
              <div className="am-bar" key={n as string}>
                <span>{n}</span>
                <span className="track">
                  <span className="fill" style={{ width: `${Math.min(100, Math.abs(v as number) * 50)}%`, background: (v as number) < 0 ? 'var(--atlas-arch)' : 'var(--atlas-fail)' }} />
                </span>
                <output>{(v as number).toFixed(3)} e</output>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="lab-panel">
        <p className="lab-note">
          Left: the checkpoint was trained on molecules with integer charges. Asked for a fractional Q it still returns an energy, and for benzene
          that energy sags below the straight line between neighbouring integers. In exact theory E is linear there, which is why two benzenes
          sharing one positive charge at infinite separation cost exactly E(C₆H₆) + E(C₆H₆⁺); the model’s separated (C₆H₆)₂⁺ instead costs
          2E(½), the sag counted twice. Right: once two fragments are on separate graphs, NQE knows each only through two numbers, its summed
          raw charge Q̃ and its summed weight F. The split is whatever those give, never forced to an integer; for identical fragments it is
          exactly half. The presets load the checkpoint’s own sums from its last charge update at 20 Å, and reproduce its fragment charges.
        </p>
      </div>
    </div>
  );
}

function EQPlot({ Q }: { Q: number }) {
  const W = 380;
  const H = 200;
  const m = { l: 44, r: 10, t: 10, b: 28 };
  const ref = EQ.find((r) => r.Q === 0)!.E;
  const es = EQ.map((r) => r.E - ref);
  const lo = Math.floor(Math.min(...es)) - 1;
  const hi = Math.ceil(Math.max(...es)) + 1;
  const x = (q: number) => m.l + ((q + 1) / 3) * (W - m.l - m.r);
  const y = (v: number) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
  const curve = EQ.map((r, i) => `${i ? 'L' : 'M'}${x(r.Q).toFixed(1)},${y(r.E - ref).toFixed(1)}`).join('');
  const ints = [-1, 0, 1, 2];
  const chords = ints.slice(1).map((b, i) => {
    const a = ints[i]!;
    return `M${x(a).toFixed(1)},${y(chord(EQ, a) - ref).toFixed(1)}L${x(b).toFixed(1)},${y(chord(EQ, b) - ref).toFixed(1)}`;
  });
  const now = EQ.find((r) => Math.abs(r.Q - Q) < 1e-6)!;
  const step = hi - lo > 20 ? 10 : 5;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return (
    <figure className="lab-plot am-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Released-model energy of benzene against its total charge from −1 to +2, relative to the neutral molecule, with straight chords between integer charges; the curve lies below the chords between integers.">
        {ticks.map((v) => (
          <g key={v}>
            <line className={v === 0 ? 'axis' : 'grid'} x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={m.l - 4} y={y(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {ints.map((q) => (
          <text key={q} className="tick" x={x(q)} y={H - m.b + 13} textAnchor="middle">
            {q > 0 ? `+${q}` : q}
          </text>
        ))}
        {chords.map((d, i) => (
          <path key={i} className="am-curve is-dashed" d={d} style={{ stroke: 'var(--atlas-verified)' }} />
        ))}
        <path className="am-curve" d={curve} style={{ stroke: 'var(--atlas-ink)' }} />
        <line className="am-gap" x1={x(Q)} x2={x(Q)} y1={y(now.E - ref)} y2={y(chord(EQ, Q) - ref)} style={{ stroke: 'var(--atlas-fail)' }} />
        <circle cx={x(Q)} cy={y(now.E - ref)} r={3.5} style={{ fill: 'var(--atlas-fail)' }} />
        <text className="axis-label" x={W - m.r} y={H - 2} textAnchor="end">
          Q (e)
        </text>
        <text className="axis-label" x={11} y={m.t + 2} transform={`rotate(-90 11 ${m.t + 2})`} textAnchor="end">
          E − E(0) (eV)
        </text>
      </svg>
    </figure>
  );
}
