import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dimenet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import si from '~/data/structures/si-diamond.json';
import cu from '~/data/structures/cu-fcc.json';
import type { Structure } from '~/lib/structure/types';
import { neighboursOf } from '~/lib/structure/geometry';
import { gemnetOcBounds, interactionCounts, type InteractionCounts, type Vec3 } from '~/lib/models/dimenet';

// How many edges, triplets (DimeNet, GemNet-T) and quadruplets (GemNet-Q) one atom's
// messages need in two elemental crystals, counted exactly from their neighbour lists.

const CRYSTALS = {
  cu: { label: 'Cu (fcc)', s: cu as unknown as Structure },
  si: { label: 'Si (diamond)', s: si as unknown as Structure },
} as const;
type Key = keyof typeof CRYSTALS;
const MAXR = 10;
const CS = Array.from({ length: 61 }, (_, k) => 2 + 0.1 * k);

function vectorsOf(s: Structure): Vec3[] {
  const o = s.positions[0]!;
  return neighboursOf(s, 0, MAXR + 1e-6).map((n) => [n.position[0] - o[0], n.position[1] - o[1], n.position[2] - o[2]] as Vec3);
}

export default function GemNetCost() {
  const theme = useSceneTheme();
  const ids = useId();
  const [key, setKey] = useState<Key>('cu');
  const [c, setC] = useState(5);
  const vecs = useMemo(() => vectorsOf(CRYSTALS[key].s), [key]);
  const within = useMemo(() => (r: number) => vecs.filter((v) => Math.hypot(...v) < r), [vecs]);
  const curve = useMemo(() => CS.map((x) => interactionCounts(within, x)), [within]);
  const now = interactionCounts(within, c);
  const gem = useMemo(() => interactionCounts(within, 5, 10), [within]);
  const oc = gemnetOcBounds();
  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [0, -30, 16], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`${CRYSTALS[key].label}: the ${now.edges} neighbours of one atom within ${c.toFixed(1)} Å.`}
            minDistance={10}
            maxDistance={90}
          >
            <ShellScene theme={theme} vecs={within(c)} c={c} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Crystal">
              {(Object.keys(CRYSTALS) as Key[]).map((k) => (
                <button key={k} type="button" aria-pressed={key === k} onClick={() => setKey(k)}>
                  {CRYSTALS[k].label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.fail }} /> centre atom a
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> embedding neighbours (edges into a)
            </li>
          </ul>
        </div>
        <CountPlot theme={theme} curve={curve} c={c} oc={oc} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-c`}>
            <span>cutoff (c_emb = c_int)</span>
            <input id={`${ids}-c`} type="range" min={2} max={8} step={0.1} value={c} onChange={(e) => setC(Number(e.target.value))} />
            <output>{c.toFixed(1)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>edges into one atom</dt>
            <dd>{now.edges.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>triplets (DimeNet, GemNet-T)</dt>
            <dd>{now.triplets.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>quadruplets (GemNet-Q)</dt>
            <dd>{now.quadruplets.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>GemNet-Q, c_emb 5 Å, c_int 10 Å</dt>
            <dd>{gem.quadruplets.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>GemNet-OC limits (30 / 8 neighbours)</dt>
            <dd>≤ {oc.quadruplets.toLocaleString('en')}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Counts are per atom and per interaction block, for the messages pointing into that atom; every atom in these two
          crystals is equivalent, so they are exact. With k neighbours an atom has k(k − 1) triplets; a quadruplet adds a
          neighbour of a neighbour, so the count grows roughly as k<sub>int</sub>·k<sub>emb</sub>². GemNet’s molecular
          setting puts the quadruplet cutoff at 10 Å; GemNet-OC instead caps the graph at 30 nearest neighbours and the
          quadruplet interaction at 8, which bounds the work whatever the density.
        </p>
      </div>
    </div>
  );
}

function ShellScene({ theme, vecs, c }: { theme: SceneTheme; vecs: Vec3[]; c: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 80);
  const atoms = useMemo<AtomBuffers>(() => {
    const n = vecs.length + 1;
    const p = new Float32Array(3 * n);
    const colors = new Float32Array(3 * n);
    const radii = new Float32Array(n);
    colors.set(linear(theme.fail), 0);
    radii[0] = 0.55;
    const arch = linear(theme.arch);
    const faint = mixRGB(arch, linear(theme.canvas), 0.35);
    vecs.forEach((v, k) => {
      p.set(v, 3 * (k + 1));
      colors.set(Math.hypot(...v) < 3 ? arch : faint, 3 * (k + 1));
      radii[k + 1] = 0.36;
    });
    return { positions: p, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [vecs, theme]);
  const spokes = useMemo<ColoredLine[]>(() => {
    const ink = linear(theme.ink);
    return vecs.length <= 60 ? vecs.map((v) => ({ a: [0, 0, 0] as [number, number, number], b: v as [number, number, number], rgba: [ink[0], ink[1], ink[2], 0.35] as [number, number, number, number] })) : [];
  }, [vecs, theme]);
  return (
    <>
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Lines lines={spokes} />
      <Shell center={[0, 0, 0]} radius={c} color={theme.phys} opacity={0.16} />
    </>
  );
}

function CountPlot({ theme, curve, c, oc }: { theme: SceneTheme; curve: InteractionCounts[]; c: number; oc: InteractionCounts }) {
  const W = 520;
  const H = 330;
  const M = { l: 52, r: 14, t: 26, b: 44 };
  const LO = 0;
  const HI = 7;
  const x = (v: number) => M.l + ((v - 2) / 6) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(v, 1)) - LO) / (HI - LO)) * (H - M.t - M.b);
  const series: Array<{ k: keyof InteractionCounts; label: string; color: string }> = [
    { k: 'edges', label: 'edges', color: theme.phys },
    { k: 'triplets', label: 'triplets', color: theme.arch },
    { k: 'quadruplets', label: 'quadruplets', color: theme.fail },
  ];
  return (
    <figure className="lab-plot dn-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Edges, triplets and quadruplets per atom against the cutoff on a log scale, with GemNet-OC's neighbour-limit bounds as dashed lines.">
        <text x={M.l} y={M.t - 10} className="axis-label">
          count per atom and block (log)
        </text>
        {Array.from({ length: HI - LO + 1 }, (_, e) => e + LO).map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={y(10 ** e)} y2={y(10 ** e)} className="grid" />
            <text x={M.l - 6} y={y(10 ** e) + 4} textAnchor="end" className="tick">
              10<tspan dy="-5" fontSize="8">{e}</tspan>
            </text>
          </g>
        ))}
        {series.map((s) => (
          <g key={s.k}>
            <path d={curve.map((p, i) => `${i ? 'L' : 'M'}${x(CS[i]!).toFixed(1)},${y(p[s.k]).toFixed(1)}`).join('')} className="curve" style={{ stroke: s.color, strokeWidth: 2.4, fill: 'none' }} />
            <line x1={M.l} x2={W - M.r} y1={y(oc[s.k])} y2={y(oc[s.k])} style={{ stroke: s.color, strokeDasharray: '5 4', strokeWidth: 1.2, opacity: 0.8 }} />
          </g>
        ))}
        <text x={W - M.r - 2} y={y(oc.quadruplets) - 4} textAnchor="end" className="tick">
          GemNet-OC bounds (dashed)
        </text>
        <line x1={x(c)} x2={x(c)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.2 }} />
        {[2, 3, 4, 5, 6, 7, 8].map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 16} textAnchor="middle" className="tick">
            {v} Å
          </text>
        ))}
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          cutoff
        </text>
      </svg>
      <figcaption className="lab-legend">
        {series.map((s) => (
          <span key={s.k} className="item">
            <span className="swatch" style={{ background: s.color }} /> {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
