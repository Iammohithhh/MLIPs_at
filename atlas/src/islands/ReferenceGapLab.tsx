import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, linear, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor } from '~/lib/structure/elements';
import nacl from '~/data/structures/nacl-rocksalt.json';
import { FUNCTIONALS, densityBudget, latticeFromDensity, type FunctionalId } from '~/lib/physics/reference-gap';

// THEORY + TOY MODEL: the measured cell, the cell the training functional would give, and the cell
// of a model that fits that functional with a given error. Lattice errors compose multiplicatively;
// density follows as a⁻³. The 3D cells are drawn with the strain magnified so 1% is visible.

const A0 = nacl.cell[0]![0]!;
const ATOMS = nacl.symbols.map((s, i) => ({ s, p: nacl.positions[i] as [number, number, number] }));
const REPS = 2; // 2 × 2 × 2 conventional cells

export default function ReferenceGapLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [fn, setFn] = useState<FunctionalId>('pbe');
  const [fit, setFit] = useState(0);
  const [tol, setTol] = useState(2.5);
  const [mag, setMag] = useState(8);
  const f = FUNCTIONALS.find((x) => x.id === fn)!;
  const b = densityBudget(f.a0MRE, fit);
  const latTol = [100 * latticeFromDensity(tol / 100), 100 * latticeFromDensity(-tol / 100)];
  const ok = Math.abs(b.total) <= tol;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [26, -30, 18], target: [A0, A0, A0] }}
            background={theme.canvas}
            label={`Rock-salt cells: measured, ${f.name} reference (${f.a0MRE > 0 ? '+' : ''}${f.a0MRE}% lattice) and model (${fmtS(b.totalLattice)}% lattice), strain drawn ${mag} times larger.`}
            minDistance={10}
            maxDistance={120}
          >
            <CellsScene theme={theme} refPct={f.a0MRE} modelPct={b.totalLattice} mag={mag} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="bar" style={{ background: theme.ink }} /> measured cell</li>
            <li><span className="bar" style={{ background: theme.arch }} /> {f.name} reference</li>
            <li><span className="bar" style={{ background: theme.caveat }} /> model (atoms drawn here)</li>
          </ul>
        </div>
        <BudgetPlot theme={theme} b={b} tol={tol} refName={f.name} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="presets" role="group" aria-label="Training functional">
            <span className="seg-label">Training functional (mean lattice error, 44 solids):</span>
            {FUNCTIONALS.map((x) => (
              <button key={x.id} type="button" className="lab-btn" aria-pressed={fn === x.id} onClick={() => setFn(x.id)}>
                {x.name} {x.a0MRE > 0 ? '+' : ''}{x.a0MRE}%
              </button>
            ))}
          </div>
          <label htmlFor={`${ids}-fit`}>
            <span>model’s lattice error relative to its functional</span>
            <input id={`${ids}-fit`} type="range" min={-3} max={3} step={0.05} value={fit} onChange={(e) => setFit(Number(e.target.value))} />
            <output>{fmtS(fit)}%</output>
          </label>
          <label htmlFor={`${ids}-tol`}>
            <span>density tolerance for the application</span>
            <input id={`${ids}-tol`} type="range" min={0.5} max={10} step={0.5} value={tol} onChange={(e) => setTol(Number(e.target.value))} />
            <output>±{tol.toFixed(1)}%</output>
          </label>
          <label htmlFor={`${ids}-mag`}>
            <span>draw strain magnified (3D only)</span>
            <input id={`${ids}-mag`} type="range" min={1} max={15} step={1} value={mag} onChange={(e) => setMag(Number(e.target.value))} />
            <output>×{mag}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Density error from the functional alone</dt>
            <dd>{fmtS(b.reference)}%</dd>
          </div>
          <div>
            <dt>Added by the fit</dt>
            <dd>{fmtS(b.fit)}%</dd>
          </div>
          <div className={ok ? undefined : 'is-alert'}>
            <dt>Model vs experiment</dt>
            <dd>{fmtS(b.total)}% {ok ? '(within tolerance)' : '(outside tolerance)'}</dd>
          </div>
          <div>
            <dt>Lattice tolerance that implies</dt>
            <dd>{fmtS(latTol[0]!)}% to {fmtS(latTol[1]!)}%</dd>
          </div>
        </dl>
        <p className="lab-note">
          With the fit error at zero the model is a perfect copy of its functional, and the readout is the functional’s own
          error. A fit error of the opposite sign can cancel it; that is luck, not accuracy, and it does not carry over to
          other properties.
        </p>
      </div>
    </div>
  );
}

const fmtS = (x: number, d = 2) => `${x > 0.0005 ? '+' : x < -0.0005 ? '−' : ''}${Math.abs(x).toFixed(d)}`;

function cubeEdges(L: number, rgba: [number, number, number, number]): ColoredLine[] {
  const c: Array<[number, number, number]> = [];
  for (const x of [0, L]) for (const y of [0, L]) for (const z of [0, L]) c.push([x, y, z]);
  const out: ColoredLine[] = [];
  for (let i = 0; i < 8; i++)
    for (let j = i + 1; j < 8; j++) {
      const d = c[i]!.reduce((s, v, k) => s + (v !== c[j]![k] ? 1 : 0), 0);
      if (d === 1) out.push({ a: c[i]!, b: c[j]!, rgba });
    }
  return out;
}

function CellsScene({ theme, refPct, modelPct, mag }: { theme: SceneTheme; refPct: number; modelPct: number; mag: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 120);
  const sRef = 1 + (mag * refPct) / 100;
  const sModel = 1 + (mag * modelPct) / 100;
  const L = REPS * A0;
  const lines = useMemo<ColoredLine[]>(() => {
    const ink = linear(theme.ink);
    const arch = linear(theme.arch);
    const cav = linear(theme.caveat);
    return [
      ...cubeEdges(L, [ink[0], ink[1], ink[2], 0.9]),
      ...cubeEdges(L * sRef, [arch[0], arch[1], arch[2], 0.95]),
      ...cubeEdges(L * sModel, [cav[0], cav[1], cav[2], 1]),
    ];
  }, [L, sRef, sModel, theme]);
  const buffers = useMemo<AtomBuffers>(() => {
    const pts: Array<{ s: string; p: [number, number, number] }> = [];
    for (let i = 0; i < REPS; i++)
      for (let j = 0; j < REPS; j++)
        for (let k = 0; k < REPS; k++)
          for (const a of ATOMS) pts.push({ s: a.s, p: [(a.p[0] + i * A0) * sModel, (a.p[1] + j * A0) * sModel, (a.p[2] + k * A0) * sModel] });
    const n = pts.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    pts.forEach((a, i) => {
      pos.set(a.p, 3 * i);
      radii[i] = a.s === 'Na' ? 0.62 : 0.9;
      colors.set(elementColor(a.s, theme.dark ? 'dark' : 'light'), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [sModel, theme.dark]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} overlay />
    </>
  );
}

function BudgetPlot({ theme, b, tol, refName }: { theme: SceneTheme; b: ReturnType<typeof densityBudget>; tol: number; refName: string }) {
  const W = 520;
  const H = 300;
  const M = { l: 150, r: 24, t: 34, b: 44 };
  const span = Math.max(6, Math.ceil(Math.max(Math.abs(b.reference), Math.abs(b.total), Math.abs(b.fit), tol) + 1));
  const X = (v: number) => M.l + ((v + span) / (2 * span)) * (W - M.l - M.r);
  const rows = [
    { label: `${refName} vs experiment`, v: b.reference, from: 0, color: theme.arch },
    { label: 'model vs its functional', v: b.fit, from: b.reference, color: theme.caveat },
    { label: 'model vs experiment', v: b.total, from: 0, color: Math.abs(b.total) <= tol ? theme.verified : theme.fail },
  ];
  const rowH = (H - M.t - M.b) / rows.length;
  const ticks = Array.from({ length: 2 * span + 1 }, (_, i) => i - span).filter((t) => t % (span > 8 ? 4 : 2) === 0);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Density error budget: ${refName} ${b.reference.toFixed(2)}%, fit ${b.fit.toFixed(2)}%, total ${b.total.toFixed(2)}% against a ±${tol}% tolerance.`}>
        <text x={M.l} y={M.t - 14} className="axis-label">density error against experiment [%]</text>
        <rect x={X(-tol)} y={M.t} width={X(tol) - X(-tol)} height={H - M.t - M.b} style={{ fill: theme.verified, opacity: 0.1 }} />
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={M.t} y2={H - M.b} className={t === 0 ? 'axis' : 'grid'} />
            <text x={X(t)} y={H - M.b + 16} textAnchor="middle" className="tick">{t > 0 ? `+${t}` : t}</text>
          </g>
        ))}
        <text x={X(tol)} y={M.t + 12} dx={4} className="tick" style={{ fill: theme.verified }}>±{tol}%</text>
        {rows.map((r, i) => {
          const y = M.t + i * rowH + rowH * 0.25;
          const x1 = X(r.from);
          const x2 = X(r.from + r.v);
          return (
            <g key={r.label}>
              <text x={M.l - 10} y={y + rowH * 0.3} textAnchor="end" className="tick" style={{ fontWeight: i === 2 ? 600 : 400 }}>{r.label}</text>
              <rect x={Math.min(x1, x2)} y={y} width={Math.max(1.5, Math.abs(x2 - x1))} height={rowH * 0.5} rx={3} style={{ fill: r.color, opacity: 0.85 }} />
              <text x={r.v < 0 ? Math.min(x1, x2) - 6 : Math.max(x1, x2) + 6} y={y + rowH * 0.32} textAnchor={r.v < 0 ? 'end' : 'start'} className="tick" style={{ fill: r.color, fontWeight: 600 }}>
                {fmtS(r.v)}%
              </text>
            </g>
          );
        })}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">ρ = (1 + ε)⁻³ − 1, isotropic strain</text>
      </svg>
    </figure>
  );
}
