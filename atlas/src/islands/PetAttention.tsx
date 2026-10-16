import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { emptiestDirection, makeToyPet, petCutoff, petForward, withProbe, type AttentionMode, type Environment, type Vec3 } from '~/lib/models/pet';

// One PET block on a real environment (Zn in wurtzite ZnO) with random, untrained weights.
// Every neighbour inside R_c becomes a token; the transformer's attention decides how much
// each token contributes. A probe atom moves outward through the cutoff: with PET's
// cutoff-modulated attention its weight fades to zero; with a plain softmax it does not,
// so the prediction jumps when the probe leaves the token list.

export interface PetEnvironmentData {
  centreSymbol: string;
  neighbours: Array<{ p: Vec3; symbol: string }>;
}

const RC = 4.5;
const WIDTH = 0.5;
const D_MIN = 2.8;
const D_MAX = 5.5;
// 0.01 Å grid plus dense points around R_c for the zoom inset.
const DS = [...Array.from({ length: 271 }, (_, k) => D_MIN + ((D_MAX - D_MIN) * k) / 270), ...Array.from({ length: 121 }, (_, k) => RC - 0.06 + k * 0.001)].sort((a, b) => a - b);
const PROBE_SPECIES = 2;

export function toEnvironment(data: PetEnvironmentData): { env: Environment; species: string[] } {
  const species = Array.from(new Set([data.centreSymbol, ...data.neighbours.map((n) => n.symbol)]));
  return {
    species,
    env: {
      centre: species.indexOf(data.centreSymbol),
      neighbours: data.neighbours.map((n) => ({ v: [...n.p] as Vec3, s: species.indexOf(n.symbol) })),
    },
  };
}

export default function PetAttention({ data }: { data: PetEnvironmentData }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [d, setD] = useState(4.3);
  const [mode, setMode] = useState<AttentionMode>('pet');
  const model = useMemo(() => makeToyPet({ species: 3 }), []);
  const { env } = useMemo(() => toEnvironment(data), [data]);
  const u = useMemo(() => emptiestDirection(env, 3.5, D_MAX), [env]);
  const curves = useMemo(
    () =>
      (['pet', 'plain'] as const).map((m) => DS.map((x) => petForward(model, withProbe(env, u, x, PROBE_SPECIES), { rc: RC, width: WIDTH, mode: m }).energy)),
    [model, env, u],
  );
  const probeEnv = withProbe(env, u, d, PROBE_SPECIES);
  const now = petForward(model, probeEnv, { rc: RC, width: WIDTH, mode });
  const probeIdx = probeEnv.neighbours.length - 1;
  const probeAttn = now.tokens.at(-1) === probeIdx ? now.attention.at(-1)! : 0;
  const fc = petCutoff(d, RC, WIDTH);
  const jump = Math.abs(curves[1]![DS.findIndex((x) => x >= RC) - 1]! - curves[1]![DS.findIndex((x) => x >= RC)]!);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [14, -22, 12], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`A zinc atom in wurtzite ZnO with ${now.tokens.length} neighbour tokens inside the ${RC} Å cutoff. A probe atom sits ${d.toFixed(2)} Å away; the centre token gives it ${(100 * probeAttn).toFixed(2)} % of its attention.`}
            minDistance={8}
            maxDistance={70}
          >
            <Scene theme={theme} env={probeEnv} tokens={now.tokens} attention={now.attention} probe={probeIdx} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Attention">
              <button type="button" aria-pressed={mode === 'pet'} onClick={() => setMode('pet')}>
                PET attention (× f<sub>c</sub>)
              </button>
              <button type="button" aria-pressed={mode === 'plain'} onClick={() => setMode('plain')}>
                plain softmax
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.arch }} /> centre token’s attention (opacity)
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys }} /> R<sub>c</sub> = {RC} Å
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.caveat }} /> R<sub>c</sub> − ΔR<sub>c</sub>
            </li>
            <li>
              <span className="bar" style={{ background: theme.fail }} /> probe atom
            </li>
          </ul>
        </div>
        <EnergyChart theme={theme} curves={curves} d={d} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-d`}>
            <span>probe distance</span>
            <input id={`${ids}-d`} type="range" min={D_MIN} max={D_MAX} step={0.01} value={d} onChange={(e) => setD(Number(e.target.value))} />
            <output>{d.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>tokens (neighbours with r &lt; R<sub>c</sub>)</dt>
            <dd>{now.tokens.length}</dd>
          </div>
          <div>
            <dt>f<sub>c</sub>(probe)</dt>
            <dd>{fc.toFixed(3)}</dd>
          </div>
          <div className={mode === 'plain' && d < RC && d > RC - WIDTH ? 'is-alert' : undefined}>
            <dt>centre’s attention to the probe</dt>
            <dd>{(100 * probeAttn).toFixed(2)} %</dd>
          </div>
          <div>
            <dt>plain-softmax jump at R<sub>c</sub></dt>
            <dd>{jump.toFixed(3)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Both curves use the same weights and the same f<sub>c</sub> on the bond heads; only the attention differs. A plain
          softmax keeps a finite share for the probe until the moment it stops being a token, so the energy steps and the
          force on the probe is infinite there. Multiplying the attention by f<sub>c</sub> and renormalising lets the token fade
          out. Energies are in arbitrary units (untrained weights).
        </p>
      </div>
    </div>
  );
}

function Scene({ theme, env, tokens, attention, probe }: { theme: SceneTheme; env: Environment; tokens: number[]; attention: number[]; probe: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 70);
  const buffers = useMemo<AtomBuffers>(() => {
    const atoms = [{ v: [0, 0, 0] as Vec3, s: env.centre }, ...env.neighbours];
    const n = atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const tint = [linear(theme.arch), linear(theme.lit)];
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    const inside = new Set(tokens);
    atoms.forEach((a, i) => {
      pos.set(a.v, 3 * i);
      if (i === 0) {
        radii[i] = 0.7;
        colors.set(linear(theme.ink), 0);
      } else if (i - 1 === probe) {
        radii[i] = 0.55;
        colors.set(linear(theme.fail), 3 * i);
      } else {
        const base = tint[a.s % 2]!;
        radii[i] = inside.has(i - 1) ? 0.5 : 0.3;
        colors.set(inside.has(i - 1) ? base : mixRGB(base, stone, 0.6), 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [env, tokens, probe, theme]);
  const lines = useMemo<ColoredLine[]>(() => {
    const arch = linear(theme.arch);
    const fail = linear(theme.fail);
    const amax = Math.max(...attention, 1e-12);
    return tokens.map((j, k) => {
      const c = j === probe ? fail : arch;
      return { a: [0, 0, 0], b: env.neighbours[j]!.v, rgba: [c[0], c[1], c[2], Math.min(1, 0.06 + 0.94 * (attention[k]! / amax))] };
    });
  }, [env, tokens, attention, probe, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
      <Shell center={[0, 0, 0]} radius={RC} color={theme.phys} opacity={0.2} />
      <Shell center={[0, 0, 0]} radius={RC - WIDTH} color={theme.caveat} opacity={0.22} />
    </>
  );
}

function EnergyChart({ theme, curves, d }: { theme: SceneTheme; curves: number[][]; d: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 56, r: 16, t: 26, b: 44 };
  const all = curves.flat();
  const pad = 0.08 * (Math.max(...all) - Math.min(...all) || 1);
  const lo = Math.min(...all) - pad;
  const hi = Math.max(...all) + pad;
  const x = (r: number) => M.l + ((r - D_MIN) / (D_MAX - D_MIN)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  // Draw the plain curve as two pieces so the step at R_c is a visible gap, not a slope.
  const path = (ys: number[], from: number, to: number) =>
    DS.map((r, k) => [r, ys[k]!] as const)
      .filter(([r]) => r >= from && r <= to)
      .map(([r, v], k) => `${k ? 'L' : 'M'}${x(r).toFixed(1)},${y(v).toFixed(1)}`)
      .join('');
  const k = DS.reduce((best, r, i) => (Math.abs(r - d) < Math.abs(DS[best]! - d) ? i : best), 0);
  const ticks = [3, 3.5, 4, 4.5, 5, 5.5];
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Atomic energy contribution of the centre atom against probe distance, for PET attention (smooth through the cutoff) and plain softmax attention (a step at the cutoff).">
        <text x={M.l} y={M.t - 10} className="axis-label">centre atom’s energy contribution (arb. units)</text>
        <rect x={x(RC - WIDTH)} y={M.t} width={x(RC) - x(RC - WIDTH)} height={H - M.t - M.b} style={{ fill: theme.caveat, opacity: 0.12 }} />
        <line x1={x(RC)} x2={x(RC)} y1={M.t} y2={H - M.b} className="axis" strokeDasharray="2 4" />
        <text x={x(RC) + 4} y={M.t + 14} className="tick">R<tspan baselineShift="sub" fontSize="0.75em">c</tspan></text>
        <text x={x(RC - WIDTH) + 4} y={H - M.b - 8} className="tick">switching region</text>
        <path d={path(curves[1]!, D_MIN, RC - 1e-9)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 4' }} />
        <path d={path(curves[1]!, RC, D_MAX)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '5 4' }} />
        <path d={path(curves[0]!, D_MIN, D_MAX)} className="curve" style={{ stroke: theme.arch }} />
        <line x1={x(d)} x2={x(d)} y1={M.t} y2={H - M.b} style={{ stroke: theme.ink, strokeWidth: 1.2 }} />
        <Zoom theme={theme} curves={curves} box={{ x: 388, y: 44, w: 124, h: 110 }} />
        <circle cx={x(d)} cy={y(curves[0]![k]!)} r={5} style={{ fill: theme.arch }} />
        <circle cx={x(d)} cy={y(curves[1]![k]!)} r={5} style={{ fill: theme.fail }} />
        {ticks.map((r) => (
          <text key={r} x={x(r)} y={H - M.b + 16} textAnchor="middle" className="tick">{r} Å</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">probe distance from the centre atom</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> PET: α<sub>ij</sub> f<sub>c</sub>(r<sub>j</sub>), renormalised</span>
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> plain softmax</span>
      </figcaption>
    </figure>
  );
}

// Inset: the curves within ±0.06 Å of R_c, rescaled so the plain-softmax step is visible.
function Zoom({ theme, curves, box }: { theme: SceneTheme; curves: number[][]; box: { x: number; y: number; w: number; h: number } }) {
  const lo = RC - 0.06;
  const hi = RC + 0.06;
  const pts = (ys: number[]) => DS.map((r, k) => [r, ys[k]!] as const).filter(([r]) => r >= lo && r <= hi);
  const vals = [...pts(curves[0]!), ...pts(curves[1]!)].map(([, v]) => v);
  const vmin = Math.min(...vals);
  const vmax = Math.max(...vals);
  const pad = 0.15 * (vmax - vmin || 1);
  const x = (r: number) => box.x + ((r - lo) / (hi - lo)) * box.w;
  const y = (v: number) => box.y + (1 - (v - vmin + pad) / (vmax - vmin + 2 * pad)) * box.h;
  const path = (ys: number[], a: number, b: number) =>
    pts(ys)
      .filter(([r]) => r >= a && r < b)
      .map(([r, v], k) => `${k ? 'L' : 'M'}${x(r).toFixed(1)},${y(v).toFixed(1)}`)
      .join('');
  return (
    <g aria-hidden="true">
      <rect x={box.x} y={box.y} width={box.w} height={box.h} style={{ fill: theme.canvas, stroke: theme.ruleStrong }} />
      <line x1={x(RC)} x2={x(RC)} y1={box.y} y2={box.y + box.h} className="axis" strokeDasharray="2 3" />
      <path d={path(curves[1]!, lo, RC)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '4 3' }} />
      <path d={path(curves[1]!, RC, hi + 1)} className="curve" style={{ stroke: theme.fail, strokeDasharray: '4 3' }} />
      <path d={path(curves[0]!, lo, hi + 1)} className="curve" style={{ stroke: theme.arch }} />
      <text x={box.x + 4} y={box.y + 12} className="tick">R<tspan baselineShift="sub" fontSize="0.75em">c</tspan> ± 0.06 Å</text>
    </g>
  );
}
