import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import water from '~/data/structures/h2o.json';
import ethanol from '~/data/structures/ethanol.json';
import { Stage } from '~/three/Stage';
import { AtomPicker, ImpostorAtoms, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { covalentRadius } from '~/lib/structure/elements';
import type { Vec3 } from '~/lib/structure/types';
import { QEQ_TOY, dimerLimit, qeq, type QEqParams } from '~/lib/physics/qeq';

// Physics Lab P8: charge equilibration. Atoms are coloured by their equilibrated
// charge; bars compare electronegativities before and after charge flows.

type Scenario = 'water' | 'ethanol' | 'nacl';
const SCENARIOS: Array<{ id: Scenario; label: string }> = [
  { id: 'water', label: 'Water' },
  { id: 'ethanol', label: 'Ethanol' },
  { id: 'nacl', label: 'Na–Cl pair' },
];

function centred(ps: Vec3[]): Vec3[] {
  const c = ps.reduce<Vec3>((a, p) => [a[0] + p[0] / ps.length, a[1] + p[1] / ps.length, a[2] + p[2] / ps.length], [0, 0, 0]);
  return ps.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
}

function system(s: Scenario, r: number): { symbols: string[]; positions: Vec3[] } {
  if (s === 'water') return { symbols: water.symbols, positions: centred(water.positions as Vec3[]) };
  if (s === 'ethanol') return { symbols: ethanol.symbols, positions: centred(ethanol.positions as Vec3[]) };
  return { symbols: ['Na', 'Cl'], positions: [[-r / 2, 0, 0], [r / 2, 0, 0]] };
}

export default function QEqLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [scenario, setScenario] = useState<Scenario>('water');
  const [Q, setQ] = useState(0);
  const [r, setR] = useState(2.4);
  const [picked, setPicked] = useState(0);
  const [shifts, setShifts] = useState<Record<string, number>>({});

  const sys = useMemo(() => system(scenario, r), [scenario, r]);
  const params = useMemo<QEqParams[]>(
    () => sys.symbols.map((s, i) => ({ chi: QEQ_TOY[s]!.chi + (shifts[`${scenario}:${i}`] ?? 0), J: QEQ_TOY[s]!.J })),
    [sys, shifts, scenario],
  );
  const Qeff = scenario === 'nacl' ? 0 : Q;
  const res = useMemo(() => qeq(sys.positions, params, Qeff), [sys, params, Qeff]);
  const pick = Math.min(picked, sys.symbols.length - 1);
  const shiftKey = `${scenario}:${pick}`;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={scenario}
            view={scenario === 'nacl' ? { position: [0, -18, 6], target: [0, 0, 0] } : { position: [3.1, -5.2, 2.7], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`${SCENARIOS.find((s) => s.id === scenario)!.label} with atoms coloured by their equilibrated charge: blue positive, amber negative.`}
            minDistance={3}
            maxDistance={60}
          >
            <QEqScene theme={theme} symbols={sys.symbols} positions={sys.positions} q={res.q} picked={pick} onPick={(i) => i !== null && setPicked(i)} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="System">
              {SCENARIOS.map((s) => (
                <button key={s.id} type="button" aria-pressed={scenario === s.id} onClick={() => { setScenario(s.id); setPicked(0); }}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.arch }} /> q &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> q &lt; 0
            </li>
            <li>click an atom to edit its χ</li>
          </ul>
        </div>
        <QEqPlots theme={theme} symbols={sys.symbols} params={params} res={res} scenario={scenario} r={r} picked={pick} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          {scenario === 'nacl' ? (
            <label htmlFor={`${ids}-r`}>
              <span>separation</span>
              <input id={`${ids}-r`} type="range" min={2} max={30} step={0.1} value={r} onChange={(e) => setR(Number(e.target.value))} />
              <output>{r.toFixed(1)} Å</output>
            </label>
          ) : (
            <label htmlFor={`${ids}-Q`}>
              <span>total charge Q</span>
              <input id={`${ids}-Q`} type="range" min={-1} max={1} step={1} value={Q} onChange={(e) => setQ(Number(e.target.value))} />
              <output>{Q > 0 ? '+1' : Q < 0 ? '−1' : '0'} e</output>
            </label>
          )}
          <label htmlFor={`${ids}-chi`}>
            <span>
              χ of atom {pick + 1} ({sys.symbols[pick]})
            </span>
            <input
              id={`${ids}-chi`}
              type="range"
              min={-3}
              max={3}
              step={0.1}
              value={shifts[shiftKey] ?? 0}
              onChange={(e) => setShifts((s) => ({ ...s, [shiftKey]: Number(e.target.value) }))}
            />
            <output>
              {(shifts[shiftKey] ?? 0) >= 0 ? '+' : '−'}
              {Math.abs(shifts[shiftKey] ?? 0).toFixed(1)} eV
            </output>
          </label>
          <button type="button" className="lab-btn" onClick={() => setShifts({})}>
            Reset electronegativities
          </button>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt className="math">μ (equalised)</dt>
            <dd>{res.mu.toFixed(3)} eV</dd>
          </div>
          <div>
            <dt>
              Charge on atom {pick + 1} ({sys.symbols[pick]})
            </dt>
            <dd>
              {res.q[pick]! >= 0 ? '+' : '−'}
              {Math.abs(res.q[pick]!).toFixed(3)} e
            </dd>
          </div>
          <div>
            <dt className="math">Σ q</dt>
            <dd>{res.q.reduce((a, b) => a + b, 0).toFixed(3)} e</dd>
          </div>
        </dl>
        <p className="lab-note">
          {scenario === 'nacl' ? (
            <>
              Pull the ions apart. Real Na and Cl atoms separate as neutral atoms, but this model keeps transferring charge:
              at infinite separation the Coulomb term vanishes and the charges settle at ±
              {dimerLimit(QEQ_TOY.Na!, QEQ_TOY.Cl!).toFixed(2)} e, set only by χ and J. That is a known flaw of plain charge
              equilibration.
            </>
          ) : (
            <>
              Charge flows from atoms with low electronegativity to atoms with high electronegativity until every atom’s
              effective electronegativity equals the same μ (the flat line on the right). Make an atom more electronegative
              and watch it pull charge from its neighbours.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function chargeColor(theme: SceneTheme, q: number): [number, number, number] {
  const t = Math.min(1, Math.abs(q) / 0.6);
  const neutral = linear(theme.dark ? '#8d96a3' : '#c3c8cf');
  return mixRGB(neutral, linear(q >= 0 ? theme.arch : theme.caveat), 0.25 + 0.75 * t);
}

function QEqScene({
  theme,
  symbols,
  positions,
  q,
  picked,
  onPick,
}: {
  theme: SceneTheme;
  symbols: string[];
  positions: Vec3[];
  q: number[];
  picked: number;
  onPick: (i: number | null) => void;
}) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 40);
  const ionic = symbols.length === 2 && symbols[0] === 'Na';
  const buffers = useMemo<AtomBuffers>(() => {
    const n = symbols.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      radii[i] = ionic ? (symbols[i] === 'Na' ? 0.82 : 1.1) : Math.max(0.24, 0.48 * covalentRadius(symbols[i]!).r);
      colors.set(chargeColor(theme, q[i]!), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [symbols, positions, q, theme, ionic]);

  const sticks = useMemo<Segment[]>(() => {
    if (ionic) return [];
    const out: Segment[] = [];
    for (let i = 0; i < positions.length; i++)
      for (let j = i + 1; j < positions.length; j++) {
        const a = positions[i]!;
        const b = positions[j]!;
        const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        if (d > 1.15 * (covalentRadius(symbols[i]!).r + covalentRadius(symbols[j]!).r)) continue;
        const mid: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
        out.push({ a, b: mid, color: chargeColor(theme, q[i]!), radius: 0.1 });
        out.push({ a: mid, b, color: chargeColor(theme, q[j]!), radius: 0.1 });
      }
    return out;
  }, [positions, symbols, q, theme, ionic]);

  const p = positions[picked]!;
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      <Shell center={p} radius={buffers.radii[picked]! * 1.55} color={theme.ink} opacity={0.5} />
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={symbols.length} onPick={onPick} />
    </>
  );
}

function QEqPlots({
  theme,
  symbols,
  params,
  res,
  scenario,
  r,
  picked,
}: {
  theme: SceneTheme;
  symbols: string[];
  params: QEqParams[];
  res: ReturnType<typeof qeq>;
  scenario: Scenario;
  r: number;
  picked: number;
}) {
  const W = 520;
  const PH = 120;
  const GAP = 50;
  const M = { l: 52, r: 14, t: 22, b: 40 };
  const H = M.t + 2 * PH + GAP + M.b;
  const n = symbols.length;
  const bw = (W - M.l - M.r) / n;

  // Top: charges.
  const qMax = Math.max(0.6, ...res.q.map((v) => Math.abs(v))) * 1.1;
  const yq = (v: number) => M.t + PH / 2 - (v / qMax) * (PH / 2);

  // Bottom: electronegativity before/after, or the dissociation curve.
  const top2 = M.t + PH + GAP;
  const chiAll = [...params.map((p) => p.chi), res.mu];
  const cLo = Math.min(...chiAll) - 0.8;
  const cHi = Math.max(...chiAll) + 0.8;
  const yc = (v: number) => top2 + PH - ((v - cLo) / (cHi - cLo)) * PH;

  const curve = useMemo(() => {
    if (scenario !== 'nacl') return null;
    return Array.from({ length: 141 }, (_, k) => 2 + (k * 28) / 140).map((d) => ({
      d,
      q: qeq([[0, 0, 0], [d, 0, 0]], [QEQ_TOY.Na!, QEQ_TOY.Cl!]).q[1]!,
    }));
  }, [scenario]);
  const xd = (d: number) => M.l + ((d - 2) / 28) * (W - M.l - M.r);
  const yd = (v: number) => top2 + Math.min(1, Math.max(0, -v)) * PH; // 0 at top, −1 at bottom (clipped)
  const lim = -dimerLimit(QEQ_TOY.Na!, QEQ_TOY.Cl!);

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Equilibrated charges per atom: ${res.q.map((v, i) => `${symbols[i]} ${v.toFixed(2)}`).join(', ')}.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">charge q [e]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={PH} className="frame" />
        <line x1={M.l} x2={W - M.r} y1={yq(0)} y2={yq(0)} className="axis" />
        {res.q.map((v, i) => (
          <g key={i}>
            <rect
              x={M.l + i * bw + bw * 0.22}
              y={Math.min(yq(v), yq(0))}
              width={bw * 0.56}
              height={Math.abs(yq(v) - yq(0))}
              style={{ fill: v >= 0 ? theme.arch : theme.caveat, opacity: i === picked ? 1 : 0.6 }}
            />
            <text x={M.l + i * bw + bw / 2} y={M.t + PH + 14} textAnchor="middle" className={`tick${i === picked ? ' strong' : ''}`}>
              {symbols[i]}
              {n > 3 ? i + 1 : ''}
            </text>
          </g>
        ))}
        {[qMax / 1.1, -qMax / 1.1].map((v) => (
          <text key={v} x={M.l - 6} y={yq(v) + 4} textAnchor="end" className="tick">
            {v > 0 ? '+' : '−'}
            {Math.abs(v).toFixed(1)}
          </text>
        ))}

        {scenario === 'nacl' && curve ? (
          <>
            <text x={M.l} y={top2 - 8} className="axis-label">charge on Cl vs separation [e]</text>
            <rect x={M.l} y={top2} width={W - M.l - M.r} height={PH} className="frame" />
            <line x1={M.l} x2={W - M.r} y1={yd(0)} y2={yd(0)} className="ew-total qeq-true" />
            <line x1={M.l} x2={W - M.r} y1={yd(lim)} y2={yd(lim)} className="qeq-limit" />
            <path d={curve.map((p, k) => `${k ? 'L' : 'M'}${xd(p.d).toFixed(1)},${yd(p.q).toFixed(1)}`).join('')} className="curve" style={{ stroke: theme.caveat }} />
            <circle cx={xd(r)} cy={yd(res.q[1]!)} r={4.5} style={{ fill: theme.caveat }} />
            <text x={W - M.r - 4} y={yd(0) + 13} textAnchor="end" className="tick">neutral atoms (correct limit)</text>
            <text x={W - M.r - 4} y={yd(lim) - 5} textAnchor="end" className="tick">QEq limit {lim.toFixed(2)}</text>
            {[2, 10, 20, 30].map((d) => (
              <text key={d} x={xd(d)} y={top2 + PH + 14} textAnchor="middle" className="tick">{d} Å</text>
            ))}
          </>
        ) : (
          <>
            <text x={M.l} y={top2 - 8} className="axis-label">electronegativity before (χ) and after (all equal μ) [eV]</text>
            <rect x={M.l} y={top2} width={W - M.l - M.r} height={PH} className="frame" />
            {params.map((p, i) => (
              <g key={i}>
                <line x1={M.l + i * bw + bw * 0.2} x2={M.l + i * bw + bw * 0.8} y1={yc(p.chi)} y2={yc(p.chi)} className="qeq-before" />
                <circle cx={M.l + i * bw + bw / 2} cy={yc(res.chiEff[i]!)} r={4} style={{ fill: theme.verified }} />
              </g>
            ))}
            <line x1={M.l} x2={W - M.r} y1={yc(res.mu)} y2={yc(res.mu)} className="qeq-mu" />
            <text x={W - M.r - 4} y={yc(res.mu) - 5} textAnchor="end" className="tick">μ = {res.mu.toFixed(2)}</text>
          </>
        )}
      </svg>
      <figcaption className="lab-legend">
        {scenario === 'nacl' ? (
          <>
            <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> QEq charge on Cl</span>
            <span className="item"><span className="swatch" style={{ background: theme.verified }} /> neutral atoms</span>
          </>
        ) : (
          <>
            <span className="item"><span className="swatch" style={{ background: theme.inkMuted }} /> χ before charge flows</span>
            <span className="item"><span className="swatch" style={{ background: theme.verified }} /> effective χ after (= μ)</span>
          </>
        )}
      </figcaption>
    </figure>
  );
}
