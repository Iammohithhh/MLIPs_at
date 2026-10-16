import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import type { Vec3 } from '~/lib/structure/types';
import type { GraphCounts } from '~/lib/models/three-body';

// Atom graph versus angle graph around one atom of a real crystal. Every pair of bonds
// shorter than the three-body cutoff is one angle the model sums over; their number
// grows roughly as the square of the neighbour count, which is why M3GNet (4 Å) and
// CHGNet (3 Å) use a shorter cutoff for angles than for the atom graph (5 Å).

export interface ThreeBodyCrystal {
  id: string;
  label: string;
  /** Neighbours of atom 0 within the atom-graph cutoff, relative to it. */
  shell: Array<{ p: Vec3; r: number; symbol: string }>;
  centreSymbol: string;
  /** Per-atom averages over the cell, on the CUTOFFS grid. */
  counts: GraphCounts[];
}

export const ATOM_CUTOFF = 5;
const MARKS = [
  { r: 3, name: 'CHGNet bond graph', dy: 12 },
  { r: 4, name: 'M3GNet three-body', dy: 28 },
];

export default function ThreeBodyGraph({ crystals }: { crystals: ThreeBodyCrystal[] }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [ci, setCi] = useState(0);
  const [r3, setR3] = useState(4);
  const c = crystals[ci]!;
  const inside = c.shell.filter((a) => a.r < r3);
  const nb = inside.length;
  const angles = (nb * (nb - 1)) / 2;
  const atomBonds = c.shell.filter((a) => a.r < ATOM_CUTOFF).length;

  return (
    <div className="lab">
      <div className="lab-toolbar">
        <div className="seg" role="group" aria-label="Crystal">
          {crystals.map((x, k) => (
            <button key={x.id} type="button" aria-pressed={k === ci} onClick={() => setCi(k)}>
              {x.label}
            </button>
          ))}
        </div>
      </div>
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [16, -20, 12], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`${c.label}: one ${c.centreSymbol} atom with ${atomBonds} neighbours inside the ${ATOM_CUTOFF} Å atom-graph cutoff; ${nb} of them are inside the ${r3.toFixed(1)} Å three-body cutoff, forming ${angles} bond angles.`}
            minDistance={6}
            maxDistance={70}
          >
            <Scene theme={theme} c={c} r3={r3} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.arch }} /> bonds inside r<sub>3</sub>
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.caveat }} /> one line per bond angle j–i–k
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys }} /> atom graph, r &lt; {ATOM_CUTOFF} Å
            </li>
          </ul>
        </div>
        <CountChart theme={theme} counts={c.counts} r3={r3} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-r3`}>
            <span>three-body cutoff r<sub>3</sub></span>
            <input id={`${ids}-r3`} type="range" min={2} max={5} step={0.05} value={r3} onChange={(e) => setR3(Number(e.target.value))} />
            <output>{r3.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>bonds in the atom graph (5 Å)</dt>
            <dd>{atomBonds}</dd>
          </div>
          <div>
            <dt>bonds inside r<sub>3</sub></dt>
            <dd>{nb}</dd>
          </div>
          <div className={angles > 200 ? 'is-alert' : undefined}>
            <dt>angles at this atom, n(n − 1)/2</dt>
            <dd>{angles.toLocaleString('en-GB')}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Messages in the atom graph cost one term per bond; three-body terms cost one per <em>pair</em> of bonds. Opening
          the angle cutoff from 3 to 5 Å multiplies the angle count by far more than the bond count — the reason both models
          keep a shorter cutoff for angles. Counts are exact for these ideal crystals; atom 0 of each cell is shown.
        </p>
      </div>
    </div>
  );
}

function Scene({ theme, c, r3 }: { theme: SceneTheme; c: ThreeBodyCrystal; r3: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 70);
  const buffers = useMemo<AtomBuffers>(() => {
    const atoms = [{ p: [0, 0, 0] as Vec3, r: 0, symbol: c.centreSymbol }, ...c.shell.filter((a) => a.r < ATOM_CUTOFF)];
    const n = atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const species = Array.from(new Set(atoms.map((a) => a.symbol)));
    const tint = [theme.arch, theme.lit];
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    atoms.forEach((a, i) => {
      pos.set(a.p, 3 * i);
      const base = linear(tint[species.indexOf(a.symbol) % 2]!);
      if (i === 0) {
        radii[i] = 0.75;
        colors.set(linear(theme.ink), 3 * i);
      } else if (a.r < r3) {
        radii[i] = 0.55;
        colors.set(base, 3 * i);
      } else {
        radii[i] = 0.3;
        colors.set(mixRGB(base, stone, 0.6), 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [c, r3, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const inside = c.shell.filter((a) => a.r < r3);
    const out: ColoredLine[] = [];
    const arch = linear(theme.arch);
    const cav = linear(theme.caveat);
    const phys = mixRGB(linear(theme.phys), linear(theme.canvas), 0.45);
    for (const a of c.shell) if (a.r >= r3 && a.r < ATOM_CUTOFF) out.push({ a: [0, 0, 0], b: a.p, rgba: [phys[0], phys[1], phys[2], 0.5] });
    for (const a of inside) out.push({ a: [0, 0, 0], b: a.p, rgba: [arch[0], arch[1], arch[2], 1] });
    const alpha = Math.max(0.12, Math.min(0.7, 18 / Math.max(1, inside.length)));
    for (let j = 0; j < inside.length; j++)
      for (let k = j + 1; k < inside.length; k++) out.push({ a: inside[j]!.p, b: inside[k]!.p, rgba: [cav[0], cav[1], cav[2], alpha] });
    return out;
  }, [c, r3, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
      <Shell center={[0, 0, 0]} radius={r3} color={theme.caveat} opacity={0.3} />
      <Shell center={[0, 0, 0]} radius={ATOM_CUTOFF} color={theme.phys} opacity={0.18} />
    </>
  );
}

function CountChart({ theme, counts, r3 }: { theme: SceneTheme; counts: GraphCounts[]; r3: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 56, r: 16, t: 26, b: 44 };
  const lo = -0.25;
  const hi = 3.5;
  const r0 = counts[0]!.cutoff;
  const r1 = counts.at(-1)!.cutoff;
  const x = (r: number) => M.l + ((r - r0) / (r1 - r0)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(v, 1)) - lo) / (hi - lo)) * (H - M.t - M.b);
  // Step curves: counts change only when a shell is crossed.
  const step = (key: 'bondsPerAtom' | 'anglesPerAtom') =>
    counts.map((c, k) => `${k ? `L${x(c.cutoff)},${y(counts[k - 1]![key])}L` : 'M'}${x(c.cutoff)},${y(c[key])}`).join('');
  const at = counts.reduce((best, c) => (Math.abs(c.cutoff - r3) < Math.abs(best.cutoff - r3) ? c : best), counts[0]!);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Bonds and bond angles per atom against cutoff radius. At ${r3.toFixed(1)} Å: ${at.bondsPerAtom.toFixed(1)} bonds and ${at.anglesPerAtom.toFixed(0)} angles per atom on average.`}>
        <text x={M.l} y={M.t - 10} className="axis-label">per atom, averaged over the cell (log scale)</text>
        {[1, 10, 100, 1000].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toLocaleString('en-GB')}</text>
          </g>
        ))}
        {MARKS.map((m) => (
          <g key={m.r}>
            <line x1={x(m.r)} x2={x(m.r)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
            <text x={x(m.r) + 4} y={M.t + m.dy} className="tick">{m.name}</text>
          </g>
        ))}
        <line x1={x(ATOM_CUTOFF)} x2={x(ATOM_CUTOFF)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
        <text x={x(ATOM_CUTOFF) + 4} y={M.t + 44} className="tick">atom graph</text>
        <path d={step('bondsPerAtom')} className="curve" style={{ stroke: theme.phys }} />
        <path d={step('anglesPerAtom')} className="curve" style={{ stroke: theme.caveat }} />
        <line x1={x(r3)} x2={x(r3)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.2 }} />
        <circle cx={x(r3)} cy={y(at.bondsPerAtom)} r={5} style={{ fill: theme.phys }} />
        <circle cx={x(r3)} cy={y(at.anglesPerAtom)} r={5} style={{ fill: theme.caveat }} />
        {[2, 3, 4, 5, 6].filter((r) => r >= r0 && r <= r1).map((r) => (
          <text key={r} x={x(r)} y={H - M.b + 16} textAnchor="middle" className="tick">{r} Å</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">cutoff radius</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> bonds per atom</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> angles per atom</span>
      </figcaption>
    </figure>
  );
}
