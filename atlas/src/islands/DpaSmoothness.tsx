import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dpa.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { SWITCHES, cutoffScan, derivs, jumpsAt, makeToy, type SoftmaxKind, type SwitchDef, type Vec3 } from '~/lib/models/dpa';

// Smoothness at the cutoff in the Deep Potential family (src/lib/models/dpa.ts). Left: the four
// radial weights and their first two derivatives, with the jumps at r_c measured from one-sided
// fits. Right: a toy attention energy while one neighbour leaves through r_c, with a plain
// softmax, a softmax multiplied by the switch, and DPA-2's softmax* (Eq. 35).

const RC = 6;
const toy = makeToy();
const FIXED: Vec3[] = [
  [2.4, 0.3, 0.1],
  [-0.5, 2.6, 0.4],
  [0.2, -0.6, 3.1],
  [-2.9, -1.2, 0.5],
  [1.1, 3.3, -2.2],
];
const DIR: Vec3 = [0.6, 0.64, 0.48];
type Order = 0 | 1 | 2;
const ORDER_LABEL = ['s(r)', 's′(r)', 's″(r)'];
const KINDS: Array<{ id: SoftmaxKind; label: string }> = [
  { id: 'plain', label: 'plain softmax' },
  { id: 'switched', label: 'w · softmax' },
  { id: 'smooth', label: 'softmax* (DPA-2)' },
];

function colours(theme: SceneTheme): Record<SwitchDef['id'], string> {
  return { hard: theme.fail, 'deeppot-se': theme.caveat, poly: theme.arch, dpa3: theme.phys };
}

export default function DpaSmoothness() {
  const theme = useSceneTheme();
  const ids = useId();
  const [width, setWidth] = useState(0.7);
  const [order, setOrder] = useState<Order>(2);
  const rcs = RC - width;
  const jumps = useMemo(() => SWITCHES.map((s) => ({ s, j: jumpsAt((r) => s.f(r, rcs, RC), RC) })), [rcs]);
  const scanR = useMemo(() => Array.from({ length: 241 }, (_, k) => 4.5 + (k * 2) / 240).filter((r) => Math.abs(r - RC) > 1e-6), []);
  const scan = useMemo(() => cutoffScan(toy, FIXED, DIR, scanR, rcs, RC), [scanR, rcs]);
  const c = colours(theme);

  return (
    <div className="lab">
      <div className="dpa-two">
        <div>
          <div className="dpa-head">
            <h4>Radial weights at the cutoff</h4>
            <div className="seg" role="group" aria-label="Derivative shown">
              {([0, 1, 2] as Order[]).map((o) => (
                <button key={o} type="button" aria-pressed={order === o} onClick={() => setOrder(o)}>
                  {ORDER_LABEL[o]}
                </button>
              ))}
            </div>
          </div>
          <SwitchPlot rcs={rcs} order={order} colours={c} />
          <ul className="dpa-legend">
            {SWITCHES.map((s) => (
              <li key={s.id} style={{ color: c[s.id] }}>
                <i className={s.id === 'hard' ? 'dash' : undefined} />
                <span style={{ color: 'var(--atlas-ink-muted)' }}>{s.label}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="dpa-head">
            <h4>One neighbour leaves through r<sub>c</sub></h4>
          </div>
          <ScanPlot theme={theme} r={scan.r} energy={scan.energy} rcs={rcs} />
          <ul className="dpa-legend">
            {KINDS.map((k) => (
              <li key={k.id} style={{ color: k.id === 'plain' ? theme.fail : k.id === 'switched' ? theme.caveat : theme.arch }}>
                <i className={k.id === 'smooth' ? undefined : 'dash'} />
                <span style={{ color: 'var(--atlas-ink-muted)' }}>{k.label}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-w`}>
            <span>switching width r<sub>c</sub> − r<sub>cs</sub></span>
            <input id={`${ids}-w`} type="range" min={0.3} max={2} step={0.05} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
            <output>{width.toFixed(2)} Å</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" onClick={() => setWidth(0.7)}>
              DPA-3’s 5.3 → 6 Å
            </button>
          </div>
          <table className="dpa-table" aria-label="Jumps across the cutoff">
            <thead>
              <tr>
                <th>jump at r<sub>c</sub> = 6 Å</th>
                <th>s</th>
                <th>s′</th>
                <th>s″</th>
              </tr>
            </thead>
            <tbody>
              {jumps.map(({ s, j }) => (
                <tr key={s.id}>
                  <td>{s.label}</td>
                  <JumpCell v={j.value} tol={1e-12} />
                  <JumpCell v={j.first} tol={1e-9} />
                  <JumpCell v={j.second} tol={1e-5} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="lab-readout" aria-live="polite">
          {KINDS.map((k) => (
            <div key={k.id} className={Math.abs(scan.jump[k.id]) > 1e-6 ? 'is-alert' : 'is-good'}>
              <dt>energy jump as the atom crosses r<sub>c</sub>, {k.label}</dt>
              <dd>{fmtJump(scan.jump[k.id])}</dd>
            </div>
          ))}
        </dl>
        <p className="lab-note">
          A jump in s itself makes the energy jump; a jump in s′ makes forces jump; a jump in s″ makes the force derivatives (the
          Hessian behind phonons and elastic constants) jump. The cosine of DeepPot-SE is C¹; DPA-1 and DPA-2 use a quintic that is C²,
          which DPA-2 requires for energy conservation. DPA-3’s double exponential is not exactly zero at r<sub>c</sub>: with its own
          settings (5.3 → 6 Å, C = 20) it is still 8 × 10⁻⁷ there before dropping to zero, far too small to matter, but a narrower
          switch makes the step grow quickly. In the attention panel, multiplying a softmax by the
          switch does not help: the new neighbour still enters every denominator at once. softmax* pushes its logit to −20 first, so it
          enters with weight e⁻²⁰. Energies in arbitrary units; untrained toy.
        </p>
      </div>
    </div>
  );
}

function fmtJump(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  return a < 1e-3 ? v.toExponential(1) : v.toFixed(3);
}

function JumpCell({ v, tol }: { v: number; tol: number }) {
  const a = Math.abs(v);
  if (a <= tol) return <td>0</td>;
  return <td className={a > 1e-4 ? 'is-jump' : 'is-tiny'}>{fmtJump(v)}</td>;
}

function SwitchPlot({ rcs, order, colours: c }: { rcs: number; order: Order; colours: Record<SwitchDef['id'], string> }) {
  const W = 340;
  const H = 230;
  const m = { l: 46, r: 10, t: 12, b: 34 };
  const r0 = Math.min(rcs - 0.4, 4.6);
  const r1 = RC + 0.4;
  const h = 1e-4;
  // Each side of r_c is sampled separately so a discontinuity is drawn as a gap, not a spike.
  const curves = SWITCHES.map((s) => {
    const f = (r: number) => s.f(r, rcs, RC);
    const side = (a: number, b: number, n: number) => Array.from({ length: n }, (_, k) => a + ((b - a) * k) / (n - 1)).map((r) => [r, derivs(f, r, h)[order]] as [number, number]);
    return { id: s.id, left: side(r0, RC - 3 * h, 160), right: side(RC + 3 * h, r1, 20) };
  });
  const ys = curves.flatMap((cv) => [...cv.left, ...cv.right].map((p) => p[1]));
  let ymin = Math.min(0, ...ys);
  let ymax = Math.max(...ys);
  if (ymax - ymin < 1e-9) ymax = ymin + 1;
  const pad = 0.06 * (ymax - ymin);
  ymin -= pad;
  ymax += pad;
  const x = (r: number) => m.l + ((r - r0) / (r1 - r0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const path = (pts: Array<[number, number]>) => pts.map(([r, v], i) => `${i ? 'L' : 'M'}${x(r).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const ticks = [Math.ceil(r0 * 2) / 2];
  while (ticks.at(-1)! + 0.5 <= r1) ticks.push(ticks.at(-1)! + 0.5);
  return (
    <figure className="lab-plot dpa-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${ORDER_LABEL[order]} for four radial weights near the cutoff at 6 Å, switching from ${rcs.toFixed(2)} Å.`}>
        {ticks.map((r) => (
          <g key={r}>
            <line className="grid" x1={x(r)} x2={x(r)} y1={m.t} y2={H - m.b} />
            <text className="tick" x={x(r)} y={H - m.b + 13} textAnchor="middle">
              {r.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="axis" x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} />
        <line className="rc-line is-active" x1={x(RC)} x2={x(RC)} y1={m.t} y2={H - m.b} />
        <line className="dpa-shell" x1={x(rcs)} x2={x(rcs)} y1={m.t} y2={H - m.b} />
        <text className="dpa-shell-label" x={x(rcs) + 3} y={m.t + 9}>
          r_cs
        </text>
        <text className="dpa-shell-label" x={x(RC) + 3} y={m.t + 9}>
          r_c
        </text>
        <text className="tick" x={m.l - 4} y={y(ymax - pad) + 3} textAnchor="end">
          {(ymax - pad).toPrecision(2)}
        </text>
        <text className="tick" x={m.l - 4} y={y(0) + 3} textAnchor="end">
          0
        </text>
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle">
          r (Å)
        </text>
        {curves.map((cv) => (
          <g key={cv.id} style={{ stroke: c[cv.id], strokeWidth: cv.id === 'poly' ? 2.6 : 1.8, strokeDasharray: cv.id === 'hard' ? '5 3' : undefined }} fill="none">
            <path d={path(cv.left)} />
            <path d={path(cv.right)} />
          </g>
        ))}
      </svg>
    </figure>
  );
}

function ScanPlot({ theme, r, energy, rcs }: { theme: SceneTheme; r: number[]; energy: Record<SoftmaxKind, number[]>; rcs: number }) {
  const W = 340;
  const H = 230;
  const m = { l: 46, r: 10, t: 12, b: 34 };
  const r0 = r[0]!;
  const r1 = r.at(-1)!;
  const all = [...energy.plain, ...energy.switched, ...energy.smooth];
  const ymin = Math.min(...all);
  const ymax = Math.max(...all);
  const pad = 0.08 * (ymax - ymin || 1);
  const x = (v: number) => m.l + ((v - r0) / (r1 - r0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - ymin + pad) / (ymax - ymin + 2 * pad)) * (H - m.t - m.b);
  const split = (vals: number[]) => {
    const inside = r.map((v, i) => [v, vals[i]!] as [number, number]).filter(([v]) => v < RC);
    const outside = r.map((v, i) => [v, vals[i]!] as [number, number]).filter(([v]) => v > RC);
    const p = (pts: Array<[number, number]>) => pts.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('');
    return [p(inside), p(outside)];
  };
  const col = { plain: theme.fail, switched: theme.caveat, smooth: theme.arch };
  return (
    <figure className="lab-plot dpa-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Toy attention energy against the distance of a neighbour moving out through the cutoff, for three kinds of softmax.">
        {[4.5, 5, 5.5, 6, 6.5].map((v) => (
          <g key={v}>
            <line className="grid" x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} />
            <text className="tick" x={x(v)} y={H - m.b + 13} textAnchor="middle">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="rc-line is-active" x1={x(RC)} x2={x(RC)} y1={m.t} y2={H - m.b} />
        <line className="dpa-shell" x1={x(rcs)} x2={x(rcs)} y1={m.t} y2={H - m.b} />
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle">
          distance of the moving neighbour (Å)
        </text>
        <text className="axis-label" x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
          E (arb. units)
        </text>
        {(['plain', 'switched', 'smooth'] as SoftmaxKind[]).map((k) => {
          const [a, b] = split(energy[k]);
          return (
            <g key={k} fill="none" style={{ stroke: col[k], strokeWidth: k === 'smooth' ? 2.6 : 1.8, strokeDasharray: k === 'smooth' ? undefined : '5 3' }}>
              <path d={a} />
              <path d={b} />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
