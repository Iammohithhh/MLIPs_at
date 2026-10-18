import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, Sticks, linear, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import type { Vec3 } from '~/lib/models/escn';
import { CUTOFF_MODES, attend, attentionScene, cutoffJump, makeToyAttention, type CutoffMode } from '~/lib/models/equiformer';

// Graph attention near the radius cutoff (src/lib/models/equiformer.ts). One centre atom attends
// over its neighbours with softmax weights from an MLP of invariant edge features; one neighbour
// moves out through r_c. A hard cutoff (EquiformerV2) and an envelope on the message alone both make
// the output jump, because the softmax denominator loses a finite term; an envelope inside the
// softmax (EquiformerV3, Eq. 4) does not.

const MODES: CutoffMode[] = ['hard', 'message', 'smooth'];
const R_MIN = 3.5;
const R_MAX = 7.5;
const RS = Array.from({ length: 321 }, (_, k) => R_MIN + ((R_MAX - R_MIN) * k) / 320);

export default function EquiformerAttention() {
  const theme = useSceneTheme();
  const ids = useId();
  const [mode, setMode] = useState<CutoffMode>('message');
  const [r, setR] = useState(5.6);
  const [ln, setLn] = useState(true);
  const [logScale, setLogScale] = useState(0);
  const scale = 10 ** logScale;
  const t = useMemo(() => makeToyAttention(), []);
  const scene = useMemo(() => attentionScene(r), [r]);
  const dists = scene.positions.map((p) => Math.hypot(...p));
  const res = useMemo(() => attend(t, dists, mode, { ln, scale }), [t, r, mode, ln, scale]);
  const curves = useMemo(
    () =>
      Object.fromEntries(
        MODES.map((m) => {
          const pts = RS.map((rr) => {
            const s = attentionScene(rr);
            const a = attend(t, s.positions.map((p) => Math.hypot(...p)), m, { ln, scale });
            return { y: a.y, w: a.weights[s.mover]! };
          });
          return [m, pts];
        }),
      ) as Record<CutoffMode, { y: number; w: number }[]>,
    [t, ln, scale],
  );
  const jump = useMemo(() => cutoffJump(t, attentionScene(t.rcut).positions.map((p) => Math.hypot(...p)), scene.mover, mode, { ln, scale }), [t, mode, scene.mover, ln, scale]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [4, -21, 11], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`A centre atom and ${scene.mover + 1} neighbours; bond thickness shows each attention weight. The highlighted neighbour is ${r.toFixed(2)} Å away; the translucent sphere is the ${t.rcut} Å cutoff.`}
            minDistance={6}
            maxDistance={40}
          >
            <Scene theme={theme} positions={scene.positions} mover={scene.mover} weights={res.weights} rcut={t.rcut} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="How the cutoff enters">
              {MODES.map((m) => (
                <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)}>
                  {CUTOFF_MODES[m].model}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.arch }} /> attention weight (thickness)
            </li>
            <li>
              <span className="dot" style={{ background: theme.fail }} /> moving neighbour
            </li>
          </ul>
        </div>
        <CutoffPlot theme={theme} curves={curves} mode={mode} r={r} rcut={t.rcut} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-r`}>
            <span>distance of the moving neighbour</span>
            <input id={`${ids}-r`} type="range" min={R_MIN} max={R_MAX} step={0.01} value={r} onChange={(e) => setR(Number(e.target.value))} />
            <output>{r.toFixed(2)} Å</output>
          </label>
          <div className="seg" role="group" aria-label="Attention re-normalisation">
            <span className="seg-label">LN before the nonlinearity</span>
            <button type="button" aria-pressed={ln} onClick={() => setLn(true)}>
              on (V2)
            </button>
            <button type="button" aria-pressed={!ln} onClick={() => setLn(false)}>
              off (Equiformer)
            </button>
          </div>
          <label htmlFor={`${ids}-s`}>
            <span>scale of the invariant features</span>
            <input id={`${ids}-s`} type="range" min={-1} max={1.6} step={0.05} value={logScale} onChange={(e) => setLogScale(Number(e.target.value))} />
            <output>×{scale < 10 ? scale.toFixed(2) : scale.toFixed(0)}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={Math.abs(jump) < 1e-9 ? 'is-good' : 'is-alert'}>
            <dt>jump in the output at r = r<sub>c</sub></dt>
            <dd>{Math.abs(jump) < 1e-9 ? '0 (to rounding)' : jump.toFixed(3)}</dd>
          </div>
          <div>
            <dt>weight of the moving neighbour</dt>
            <dd>{res.weights[scene.mover]!.toFixed(3)}</dd>
          </div>
          <div>
            <dt>output y</dt>
            <dd>{res.y.toFixed(3)}</dd>
          </div>
          <div className={res.entropy < 0.2 ? 'is-alert' : ''}>
            <dt>attention entropy (uniform: {Math.log(scene.mover + 1).toFixed(2)})</dt>
            <dd>{res.entropy.toFixed(2)} nats</dd>
          </div>
        </dl>
        <p className="lab-note">
          Move the highlighted neighbour out through {t.rcut} Å. With a hard cutoff its weight drops from a finite value to zero, and
          the other weights jump up to fill the gap. An envelope on the message removes its own contribution smoothly, but its
          weight still leaves the softmax denominator all at once, so the output still jumps (the jump readout is computed at
          the cutoff for the mode you picked). With the envelope inside the softmax its weight fades to zero first. Switch
          LayerNorm off and scale the features up: the logits grow with the scale and the softmax collapses onto one neighbour.
          With LayerNorm the logits do not change. The toy cutoff is 6 Å; EquiformerV2 used 12 Å on OC20 and V3 6 Å on OMat24.
        </p>
      </div>
    </div>
  );
}

function Scene({ theme, positions, mover, weights, rcut }: { theme: SceneTheme; positions: Vec3[]; mover: number; weights: number[]; rcut: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 10, 45);
  const atoms = useMemo<AtomBuffers>(() => {
    const n = positions.length + 1;
    const pos = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const radii = new Float32Array(n).fill(0.32);
    positions.forEach((p, k) => {
      pos.set(p, 3 * (k + 1));
      colors.set(linear(k === mover ? theme.fail : theme.inkMuted), 3 * (k + 1));
    });
    colors.set(linear(theme.ink), 0);
    radii[0] = 0.42;
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, mover, theme]);
  const bonds = useMemo<Segment[]>(
    () =>
      positions
        .map((p, k) => ({ p, w: weights[k]! }))
        .filter((b) => b.w > 1e-4)
        .map((b) => ({ a: [0, 0, 0] as Vec3, b: b.p, color: linear(theme.arch), radius: 0.02 + 0.2 * Math.sqrt(b.w) })),
    [positions, weights, theme],
  );
  return (
    <>
      <Shell center={[0, 0, 0]} radius={rcut} color={theme.arch} opacity={0.08} />
      <ImpostorAtoms buffers={atoms} version={positions.length} uniforms={uniforms} />
      {bonds.length ? <Sticks segments={bonds} uniforms={uniforms} /> : null}
    </>
  );
}

function CutoffPlot({ theme, curves, mode, r, rcut }: { theme: SceneTheme; curves: Record<CutoffMode, { y: number; w: number }[]>; mode: CutoffMode; r: number; rcut: number }) {
  const W = 520;
  const H = 360;
  const x0 = 56;
  const w = W - x0 - 18;
  const top = { y0: 30, h: 150 };
  const bot = { y0: 225, h: 90 };
  const px = (v: number) => x0 + ((v - R_MIN) / (R_MAX - R_MIN)) * w;
  const ys = MODES.flatMap((m) => curves[m].map((p) => p.y));
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);
  const pad = 0.08 * (yHi - yLo || 1);
  const pyTop = (v: number) => top.y0 + ((yHi + pad - v) / (yHi - yLo + 2 * pad)) * top.h;
  const pyBot = (v: number) => bot.y0 + (1 - v) * bot.h;
  const color: Record<CutoffMode, string> = { hard: theme.fail, message: theme.caveat, smooth: theme.arch };
  // Break each curve at the cutoff so a jump is drawn as a gap, not a vertical line.
  const path = (vals: number[], py: (v: number) => number) =>
    vals
      .map((v, k) => {
        const rr = RS[k]!;
        const prev = k > 0 ? RS[k - 1]! : rr;
        const move = k === 0 || (prev < rcut && rr >= rcut);
        return `${move ? 'M' : 'L'}${px(rr).toFixed(1)},${py(v).toFixed(1)}`;
      })
      .join('');
  const k = Math.round(((r - R_MIN) / (R_MAX - R_MIN)) * 320);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Top: the aggregated output as the moving neighbour goes from 3.5 to 7.5 Å, for a hard cutoff, an envelope on the message only, and an envelope inside the softmax. Bottom: that neighbour's attention weight.">
        <text x={x0} y={top.y0 - 12} className="axis-label">output y = Σ a_ij v_ij</text>
        <text x={x0} y={bot.y0 - 12} className="axis-label">weight of the moving neighbour</text>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={x0} x2={x0 + w} y1={pyBot(v)} y2={pyBot(v)} className="grid" />
            <text x={x0 - 6} y={pyBot(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <line x1={px(rcut)} x2={px(rcut)} y1={top.y0} y2={bot.y0 + bot.h} className="rc-line is-active" />
        <text x={px(rcut) + 4} y={top.y0 + 10} className="tick">r_c</text>
        <line x1={px(r)} x2={px(r)} y1={top.y0} y2={bot.y0 + bot.h} className="grid" style={{ strokeDasharray: '3 3' }} />
        {MODES.map((m) => (
          <g key={m} style={{ opacity: m === mode ? 1 : 0.45 }}>
            <path d={path(curves[m].map((p) => p.y), pyTop)} className={`curve${m === mode ? ' is-active' : ''}`} style={{ stroke: color[m] }} />
            <path d={path(curves[m].map((p) => p.w), pyBot)} className={`curve${m === mode ? ' is-active' : ''}`} style={{ stroke: color[m] }} />
          </g>
        ))}
        <circle cx={px(r)} cy={pyTop(curves[mode][k]!.y)} r={4.5} style={{ fill: color[mode] }} />
        <circle cx={px(r)} cy={pyBot(curves[mode][k]!.w)} r={4.5} style={{ fill: color[mode] }} />
        {[4, 5, 6, 7].map((v) => (
          <text key={v} x={px(v)} y={bot.y0 + bot.h + 16} textAnchor="middle" className="tick">{v} Å</text>
        ))}
      </svg>
      <figcaption className="lab-legend">
        {MODES.map((m) => (
          <span key={m} className="item">
            <span className="swatch" style={{ background: color[m] }} /> {CUTOFF_MODES[m].name} ({CUTOFF_MODES[m].model})
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
