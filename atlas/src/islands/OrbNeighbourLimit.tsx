import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ORB_K20, ORB_KINF, ORB_RMAX, limitCut, makeToyOrb, orbForward, withRadial, type EdgeMode, type LimitCurve, type Vec3 } from '~/lib/models/orb';

// Orb keeps at most k neighbours per atom inside 6 Å: k = 20 for the "-20" checkpoints and 120
// for "-inf". In fcc Cu the 20th place falls inside the 24-atom third shell, so 22 symmetry-
// equivalent atoms are dropped by ordering alone. Moving one of them inward makes it an edge;
// at the shell distance the graph changes discontinuously. Curves are computed at build time
// by the same tested module; the 3D view and readouts are computed live.

export type LimitKey = `${EdgeMode}-${'20' | 'inf'}`;

export interface NeighbourLimitData {
  vectors: Vec3[];
  moved: number;
  curves: Record<LimitKey, LimitCurve>;
}

const R_MIN = 4.15;
const R_MAX = 4.7;

export default function OrbNeighbourLimit({ data }: { data: NeighbourLimitData }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [r, setR] = useState(4.36);
  const [limit, setLimit] = useState<'20' | 'inf'>('20');
  const [mode, setMode] = useState<EdgeMode>('orb');
  const model = useMemo(() => makeToyOrb({ mode }), [mode]);
  const k = limit === '20' ? ORB_K20 : ORB_KINF;
  const vectors = useMemo(() => withRadial(data.vectors, data.moved, r), [data, r]);
  const now = useMemo(() => orbForward(model, vectors, k), [model, vectors, k]);
  const r0 = Math.hypot(...data.vectors[data.moved]!);
  const cut = useMemo(() => limitCut(data.vectors.map((v) => Math.hypot(...v)), ORB_K20), [data]);
  const rank = vectors
    .map((v, i) => [Math.hypot(...v), i] as const)
    .filter(([d]) => d < ORB_RMAX)
    .sort((a, b) => a[0] - b[0])
    .findIndex(([, i]) => i === data.moved);
  const isEdge = now.edges.includes(data.moved);
  const key: LimitKey = `${mode}-${limit}`;
  const curve = data.curves[key];
  const step = (c: LimitCurve, field: 'energy' | 'force') => {
    const i = c.r.findIndex((x) => x > r0);
    return c[field][i]! - c[field][i - 1]!;
  };
  const fNow = interp(curve.r, curve.force, r);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [16, -24, 14], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`A copper atom in fcc Cu with ${vectors.filter((v) => Math.hypot(...v) < ORB_RMAX).length} neighbours inside 6 Å, of which ${now.edges.length} become edges. One third-shell atom has been moved to ${r.toFixed(3)} Å; it is ${isEdge ? '' : 'not '}an edge.`}
            minDistance={8}
            maxDistance={80}
          >
            <Scene theme={theme} vectors={vectors} edges={now.edges} gates={now.gates} moved={data.moved} r0={r0} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Neighbour limit">
              <button type="button" aria-pressed={limit === '20'} onClick={() => setLimit('20')}>
                k = 20 (“-20”)
              </button>
              <button type="button" aria-pressed={limit === 'inf'} onClick={() => setLimit('inf')}>
                k = 120 (“-inf”)
              </button>
            </div>
            <div className="seg" role="group" aria-label="Edge features">
              <button type="button" aria-pressed={mode === 'orb'} onClick={() => setMode('orb')}>
                Bessel × Y<sub>lm</sub> (Orb-v3)
              </button>
              <button type="button" aria-pressed={mode === 'distance'} onClick={() => setMode('distance')}>
                distance only
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.arch }} /> edge (opacity = gate ψ)
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> third shell, 4.43 Å (24 atoms)
            </li>
            <li>
              <span className="bar" style={{ background: theme.fail }} /> moved atom
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys }} /> r<sub>max</sub> = 6 Å
            </li>
          </ul>
        </div>
        <Charts theme={theme} curves={data.curves} mode={mode} limit={limit} r={r} r0={r0} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-r`}>
            <span>moved atom’s distance</span>
            <input id={`${ids}-r`} type="range" min={R_MIN} max={R_MAX} step={0.001} value={r} onChange={(e) => setR(Number(e.target.value))} />
            <output>{r.toFixed(3)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>edges kept / inside 6 Å</dt>
            <dd>
              {now.edges.length} / {cut.inside}
            </dd>
          </div>
          <div>
            <dt>tied at the 20th distance</dt>
            <dd>
              {cut.tied} atoms, {cut.tiedKept} kept
            </dd>
          </div>
          <div className={limit === '20' && !isEdge ? 'is-alert' : undefined}>
            <dt>moved atom: rank, edge?</dt>
            <dd>
              #{rank + 1}, {isEdge ? 'edge' : 'dropped'}
            </dd>
          </div>
          <div>
            <dt>radial force on it</dt>
            <dd>{fNow.toExponential(2)}</dd>
          </div>
          <div className={limit === '20' ? 'is-alert' : undefined}>
            <dt>steps at {r0.toFixed(3)} Å: ΔE, ΔF</dt>
            <dd>
              {step(curve, 'energy').toExponential(1)}, {step(curve, 'force').toExponential(1)}
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          Same random, untrained weights in every setting; energies and forces in arbitrary units. With k = 20 the moved atom is an
          edge only while it is nearer than its 23 equivalent neighbours. At 4.43 Å it swaps with one of them. With distance-only
          features that swap leaves the energy unchanged but the force on the atom drops to zero. With Orb-v3’s direction-dependent
          features the energy steps as well. With k = 120 all 78 neighbours are edges and both curves are smooth.
        </p>
      </div>
    </div>
  );
}

function interp(xs: number[], ys: number[], x: number): number {
  const i = Math.min(xs.length - 2, Math.max(0, xs.findIndex((v) => v > x) - 1));
  const t = (x - xs[i]!) / (xs[i + 1]! - xs[i]!);
  return ys[i]! + t * (ys[i + 1]! - ys[i]!);
}

function Scene({ theme, vectors, edges, gates, moved, r0 }: { theme: SceneTheme; vectors: Vec3[]; edges: number[]; gates: number[]; moved: number; r0: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 22, 80);
  const buffers = useMemo<AtomBuffers>(() => {
    const atoms: Vec3[] = [[0, 0, 0], ...vectors];
    const n = atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const kept = new Set(edges);
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    const arch = linear(theme.arch);
    const shell = linear(theme.caveat);
    atoms.forEach((a, i) => {
      pos.set(a, 3 * i);
      const j = i - 1;
      if (i === 0) {
        radii[i] = 0.62;
        colors.set(linear(theme.ink), 0);
      } else if (j === moved) {
        radii[i] = 0.55;
        colors.set(linear(theme.fail), 3 * i);
      } else {
        const inShell = Math.abs(Math.hypot(...a) - r0) < 1e-6;
        const base = inShell ? shell : arch;
        radii[i] = kept.has(j) ? 0.48 : 0.3;
        colors.set(kept.has(j) ? base : mixRGB(base, stone, inShell ? 0.35 : 0.75), 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.6), ao: new Float32Array(n).fill(1), count: n };
  }, [vectors, edges, moved, r0, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const arch = linear(theme.arch);
    const fail = linear(theme.fail);
    const gmax = Math.max(...gates, 1e-12);
    return edges.map((j, i) => {
      const c = j === moved ? fail : arch;
      return { a: [0, 0, 0], b: vectors[j]!, rgba: [c[0], c[1], c[2], Math.min(1, 0.08 + 0.92 * (gates[i]! / gmax))] };
    });
  }, [vectors, edges, gates, moved, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
      <Shell center={[0, 0, 0]} radius={ORB_RMAX} color={theme.phys} opacity={0.12} />
      <Shell center={[0, 0, 0]} radius={r0} color={theme.caveat} opacity={0.14} />
    </>
  );
}

function Charts({ theme, curves, mode, limit, r, r0 }: { theme: SceneTheme; curves: Record<LimitKey, LimitCurve>; mode: EdgeMode; limit: '20' | 'inf'; r: number; r0: number }) {
  const W = 520;
  const H = 380;
  const M = { l: 60, r: 16, t: 24, b: 40 };
  const gap = 34;
  const ph = (H - M.t - M.b - gap) / 2;
  const x = (v: number) => M.l + ((v - R_MIN) / (R_MAX - R_MIN)) * (W - M.l - M.r);
  const c20 = curves[`${mode}-20`];
  const cinf = curves[`${mode}-inf`];
  // Energies are shifted to agree at the left end so the two curves share one scale.
  const shifted = (c: LimitCurve): LimitCurve => ({ ...c, energy: c.energy.map((e) => e - c.energy[0]!) });
  const s20 = shifted(c20);
  const sinf = shifted(cinf);
  const panel = (field: 'energy' | 'force', top: number) => {
    const a = field === 'energy' ? s20 : c20;
    const b = field === 'energy' ? sinf : cinf;
    const all = [...a[field], ...b[field]];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = 0.1 * (hi - lo || 1);
    const y = (v: number) => top + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * ph;
    const path = (c: LimitCurve, from: number, to: number) =>
      c.r
        .map((rr, k) => [rr, c[field][k]!] as const)
        .filter(([rr]) => rr >= from && rr < to)
        .map(([rr, v], k) => `${k ? 'L' : 'M'}${x(rr).toFixed(1)},${y(v).toFixed(1)}`)
        .join('');
    const at20 = interp(a.r, a[field], r);
    const atInf = interp(b.r, b[field], r);
    return (
      <g>
        <rect x={M.l} y={top} width={W - M.l - M.r} height={ph} style={{ fill: 'none', stroke: theme.rule }} />
        <text x={M.l} y={top - 8} className="axis-label">
          {field === 'energy' ? 'centre atom’s energy, relative to 4.15 Å (arb. units)' : 'radial force on the moved atom (arb. units)'}
        </text>
        <path d={path(b, 0, 99)} className="curve" style={{ stroke: theme.arch, opacity: limit === 'inf' ? 1 : 0.45 }} />
        <path d={path(a, 0, r0)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 3', opacity: limit === '20' ? 1 : 0.45 }} />
        <path d={path(a, r0, 99)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 3', opacity: limit === '20' ? 1 : 0.45 }} />
        <circle cx={x(r)} cy={y(atInf)} r={4.5} style={{ fill: theme.arch }} />
        <circle cx={x(r)} cy={y(at20)} r={4.5} style={{ fill: theme.fail }} />
      </g>
    );
  };
  const ticks = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7];
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Energy of the centre atom and radial force on the moved atom against its distance, for a 20-neighbour limit (a step at the third-shell distance) and for 120 neighbours (smooth).">
        <line x1={x(r0)} x2={x(r0)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
        <text x={x(r0) + 4} y={H - M.b - 6} className="tick">third shell</text>
        {panel('energy', M.t)}
        {panel('force', M.t + ph + gap)}
        <line x1={x(r)} x2={x(r)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.1 }} />
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 16} textAnchor="middle" className="tick">
            {t.toFixed(1)}
          </text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">moved atom’s distance from the centre (Å)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item">
          <span className="swatch" style={{ background: theme.fail }} /> k = 20
        </span>
        <span className="item">
          <span className="swatch" style={{ background: theme.arch }} /> k = 120 (all 78 neighbours)
        </span>
      </figcaption>
    </figure>
  );
}
