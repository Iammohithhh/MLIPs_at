import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import {
  ImpostorAtoms,
  Lines,
  Shell,
  Sticks,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor, hexToLinearRgb } from '~/lib/structure/elements';
import type { Vec3 } from '~/lib/structure/types';
import { SW_CUTOFF, SW_SI, phi2, phi3, screen3, swCluster, triplet } from '~/lib/physics/stillinger-weber';

// Physics Lab P3: three silicon atoms under the Stillinger–Weber potential. A model
// built from pairs alone has no opinion about the bond angle; the three-body term
// supplies one, with a minimum at the tetrahedral angle.

type Model = 'pairs' | 'sw';
const TETRA = (Math.acos(-1 / 3) * 180) / Math.PI;

function siColor(dark: boolean): [number, number, number] {
  const c = elementColor('Si', dark ? 'dark' : 'light');
  return hexToLinearRgb('#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join(''));
}

export default function BodyOrderLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [theta, setTheta] = useState(80);
  const [r1, setR1] = useState(2.35);
  const [r2, setR2] = useState(2.35);
  const [model, setModel] = useState<Model>('sw');

  const pos = useMemo(() => triplet(r1, r2, theta), [r1, r2, theta]);
  const e = useMemo(() => swCluster(pos, SW_SI, model === 'sw'), [pos, model]);
  const eFull = useMemo(() => swCluster(pos), [pos]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [1.4, -2.2, 9.5], target: [0.9, 0.9, 0] }}
            background={theme.canvas}
            label={`Three silicon atoms: bonds of ${r1.toFixed(2)} and ${r2.toFixed(2)} Å at an angle of ${theta.toFixed(0)}°.`}
            minDistance={4}
            maxDistance={30}
          >
            <TripletScene theme={theme} pos={pos} model={model} theta={theta} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Model">
              <button type="button" aria-pressed={model === 'pairs'} onClick={() => setModel('pairs')}>
                Pairs only
              </button>
              <button type="button" aria-pressed={model === 'sw'} onClick={() => setModel('sw')}>
                Pairs + angles (Stillinger–Weber)
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="ring" /> cutoff aσ = {SW_CUTOFF.toFixed(2)} Å around the centre atom
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> angle strain (three-body energy)
            </li>
          </ul>
        </div>
        <TripletPlots theme={theme} r1={r1} r2={r2} theta={theta} model={model} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-t`}>
            <span>
              angle <i>θ</i>
            </span>
            <input id={`${ids}-t`} type="range" min={45} max={180} step={0.5} value={theta} onChange={(ev) => setTheta(Number(ev.target.value))} />
            <output>{theta.toFixed(1)}°</output>
          </label>
          <label htmlFor={`${ids}-r1`}>
            <span>bond 1</span>
            <input id={`${ids}-r1`} type="range" min={2} max={3.9} step={0.01} value={r1} onChange={(ev) => setR1(Number(ev.target.value))} />
            <output>{r1.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-r2`}>
            <span>bond 2</span>
            <input id={`${ids}-r2`} type="range" min={2} max={3.9} step={0.01} value={r2} onChange={(ev) => setR2(Number(ev.target.value))} />
            <output>{r2.toFixed(2)} Å</output>
          </label>
          <button type="button" className="lab-btn" onClick={() => setTheta(TETRA)}>
            Snap to tetrahedral angle
          </button>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Two-body</dt>
            <dd>
              {e.two.toFixed(3)} eV <small>({e.nPairs} pairs)</small>
            </dd>
          </div>
          <div className={model === 'pairs' ? undefined : 'is-alert'}>
            <dt>Three-body</dt>
            <dd>
              {model === 'pairs' ? '—' : `${e.three.toFixed(3)} eV`} <small>({model === 'pairs' ? 0 : e.nTriplets} triplets)</small>
            </dd>
          </div>
          <div>
            <dt>Total</dt>
            <dd>{e.total.toFixed(3)} eV</dd>
          </div>
        </dl>
        <p className="lab-note">
          {model === 'pairs' ? (
            <>
              With pairs only, the energy is flat for every angle above about 107°, once the outer pair leaves the
              cutoff, and lowest near 60°, where the outer pair forms a third bond. A pair model favours close packing; it
              has no reason to prefer the tetrahedral angle. Switch on the three-body term to give it one.
            </>
          ) : (
            <>
              The three-body term λε(cos θ + ⅓)² vanishes at the tetrahedral angle, {TETRA.toFixed(2)}°, and penalises
              every other angle — here {eFull.three.toFixed(3)} eV of angle strain. Stretch a bond towards the cutoff and the
              smooth screening switches the angle term off with it.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function TripletScene({ theme, pos, model, theta }: { theme: SceneTheme; pos: Vec3[]; model: Model; theta: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 6, 26);
  const si = siColor(theme.dark);
  const buffers = useMemo<AtomBuffers>(
    () => ({
      positions: new Float32Array(pos.flat()),
      radii: new Float32Array([0.5, 0.5, 0.5]),
      colors: new Float32Array([...si, ...si, ...si]),
      metal: new Float32Array(3),
      ao: new Float32Array([1, 1, 1]),
      count: 3,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pos, si[0]],
  );

  const sticks = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    const bond = (a: Vec3, b: Vec3) => {
      const r = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      if (r >= SW_CUTOFF) return;
      const strength = Math.min(1, Math.max(0, -phi2(r) / SW_SI.epsilon));
      out.push({ a, b, color: mixRGB(linear(theme.ruleStrong), linear(theme.phys), strength), radius: 0.06 + 0.08 * strength });
    };
    bond(pos[0]!, pos[1]!);
    bond(pos[0]!, pos[2]!);
    bond(pos[1]!, pos[2]!);
    return out;
  }, [pos, theme.ruleStrong, theme.phys]);

  // Angle arc at the centre atom, coloured by the three-body strain.
  const arc = useMemo<ColoredLine[]>(() => {
    const strain = model === 'sw' ? Math.min(1, phi3(1, 1, Math.cos((theta * Math.PI) / 180)) / 20) : 0;
    const c = mixRGB(linear(theme.ink), linear(theme.caveat), model === 'sw' ? 0.35 + 0.65 * Math.sqrt(strain) : 0);
    const R = 0.95;
    const n = 48;
    const out: ColoredLine[] = [];
    for (let k = 0; k < n; k++) {
      const t0 = ((theta * Math.PI) / 180) * (k / n);
      const t1 = ((theta * Math.PI) / 180) * ((k + 1) / n);
      out.push({ a: [R * Math.cos(t0), R * Math.sin(t0), 0], b: [R * Math.cos(t1), R * Math.sin(t1), 0], rgba: [c[0], c[1], c[2], 1] });
    }
    // Tetrahedral reference tick.
    const tt = (TETRA * Math.PI) / 180;
    const g = linear(theme.verified);
    out.push({ a: [0.75 * Math.cos(tt), 0.75 * Math.sin(tt), 0], b: [1.2 * Math.cos(tt), 1.2 * Math.sin(tt), 0], rgba: [g[0], g[1], g[2], 1] });
    return out;
  }, [theta, model, theme.ink, theme.caveat, theme.verified]);

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      <Lines lines={arc} overlay />
      <Shell center={[0, 0, 0]} radius={SW_CUTOFF} color={theme.phys} opacity={0.25} grid />
    </>
  );
}

function TripletPlots({ theme, r1, r2, theta, model }: { theme: SceneTheme; r1: number; r2: number; theta: number; model: Model }) {
  const W = 520;
  const PH = 140;
  const GAP = 52;
  const M = { l: 52, r: 14, t: 22, b: 40 };
  const H = M.t + 2 * PH + GAP + M.b;
  const thetas = Array.from({ length: 136 }, (_, k) => 45 + k);
  const pairs = thetas.map((t) => swCluster(triplet(r1, r2, t), SW_SI, false).total);
  const full = thetas.map((t) => swCluster(triplet(r1, r2, t)).total);
  const lo = Math.min(...pairs, ...full);
  const hi = Math.min(Math.max(...pairs, ...full), lo + 6);
  const x = (t: number) => M.l + ((t - 45) / 135) * (W - M.l - M.r);
  const y = (v: number) => M.t + PH - ((Math.min(v, hi) - lo) / (hi - lo || 1)) * PH;
  const line = (vals: number[]) => thetas.map((t, k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)},${y(vals[k]!).toFixed(1)}`).join('');

  const top2 = M.t + PH + GAP;
  const rs = Array.from({ length: 200 }, (_, k) => 2 + (k * 2.2) / 199);
  const xr = (r: number) => M.l + ((r - 2) / 2.2) * (W - M.l - M.r);
  const yr = (v: number) => top2 + PH * 0.55 - (v / SW_SI.epsilon) * PH * 0.45;
  const ys = (v: number) => top2 + PH - v * PH * 0.9;
  const cur = swCluster(triplet(r1, r2, theta), SW_SI, model === 'sw').total;

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Energy of the triplet against the bond angle for a pairs-only model (flat over wide angles) and Stillinger–Weber (minimum at 109.47 degrees); and the radial pair energy and three-body screening, both reaching zero smoothly at the cutoff.">
        <text x={M.l} y={M.t - 8} className="axis-label">triplet energy vs angle θ [eV]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={PH} className="frame" />
        <line x1={x(TETRA)} x2={x(TETRA)} y1={M.t} y2={M.t + PH} style={{ stroke: theme.verified, strokeDasharray: '4 3' }} />
        <text x={x(TETRA) + 4} y={M.t + 12} className="tick">109.47°</text>
        <path d={line(pairs)} className="curve" style={{ stroke: theme.ruleStrong, strokeWidth: model === 'pairs' ? 3 : 2 }} />
        <path d={line(full)} className="curve" style={{ stroke: theme.phys, strokeWidth: model === 'sw' ? 3 : 2, opacity: model === 'sw' ? 1 : 0.5 }} />
        <circle cx={x(theta)} cy={y(cur)} r={5} style={{ fill: model === 'sw' ? theme.phys : theme.inkMuted }} />
        {[45, 90, 135, 180].map((t) => (
          <text key={t} x={x(t)} y={M.t + PH + 14} textAnchor="middle" className="tick">{t}°</text>
        ))}

        <text x={M.l} y={top2 - 8} className="axis-label">pair energy φ₂(r) and three-body screening</text>
        <rect x={M.l} y={top2} width={W - M.l - M.r} height={PH} className="frame" />
        <line x1={M.l} x2={W - M.r} y1={yr(0)} y2={yr(0)} className="axis" />
        <path d={rs.map((r, k) => `${k ? 'L' : 'M'}${xr(r).toFixed(1)},${yr(Math.max(-SW_SI.epsilon * 1.05, Math.min(phi2(r), SW_SI.epsilon))).toFixed(1)}`).join('')} className="curve" style={{ stroke: theme.phys }} />
        <path d={rs.map((r, k) => `${k ? 'L' : 'M'}${xr(r).toFixed(1)},${ys(screen3(r) / screen3(2)).toFixed(1)}`).join('')} className="curve" style={{ stroke: theme.caveat, strokeDasharray: '5 3' }} />
        <line x1={xr(SW_CUTOFF)} x2={xr(SW_CUTOFF)} y1={top2} y2={top2 + PH} style={{ stroke: theme.inkMuted, strokeDasharray: '3 3' }} />
        <text x={xr(SW_CUTOFF) + 4} y={top2 + 12} className="tick">aσ (C∞ cutoff)</text>
        <line x1={xr(r1)} x2={xr(r1)} y1={top2} y2={top2 + PH} className="r-line" />
        {[2, 2.5, 3, 3.5, 4].map((r) => (
          <text key={r} x={xr(r)} y={top2 + PH + 14} textAnchor="middle" className="tick">{r} Å</text>
        ))}
        <text x={M.l - 6} y={yr(-SW_SI.epsilon) + 4} textAnchor="end" className="tick">−ε</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.ruleStrong }} /> pairs only</span>
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> Stillinger–Weber / φ₂</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> three-body screening (scaled)</span>
      </figcaption>
    </figure>
  );
}
