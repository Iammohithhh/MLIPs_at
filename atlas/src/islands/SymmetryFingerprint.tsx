import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import si from '~/data/structures/si-diamond.json';
import zno from '~/data/structures/zno-wurtzite.json';
import ethanol from '~/data/structures/ethanol.json';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { neighboursOf } from '~/lib/structure/geometry';
import type { Structure, Vec3 } from '~/lib/structure/types';
import { apply, rotation } from '~/lib/physics/rotation-toy';
import { DEFAULT_SET, evaluate, g2Contributions, g3WithShares, label } from '~/lib/models/symmetry-functions';

// Behler–Parrinello fingerprint of one atom: a fixed vector of symmetry functions.
// Rotating the structure leaves it unchanged; distorting it does not. Select a
// function to colour the neighbours by how much each contributes to it.

const RC = 6;
type SysId = 'si' | 'zno' | 'ethanol';
const SYS: Record<SysId, { label: string; s: Structure; centre: number }> = {
  si: { label: 'Si (diamond)', s: si as unknown as Structure, centre: 0 },
  zno: { label: 'ZnO (wurtzite)', s: zno as unknown as Structure, centre: 0 },
  ethanol: { label: 'Ethanol', s: ethanol as unknown as Structure, centre: 2 },
};

function envOf(id: SysId): Vec3[] {
  const { s, centre } = SYS[id];
  const c = s.positions[centre]!;
  return neighboursOf(s, centre, RC).map((n) => [n.position[0] - c[0], n.position[1] - c[1], n.position[2] - c[2]]);
}

export default function SymmetryFingerprint() {
  const theme = useSceneTheme();
  const ids = useId();
  const [sys, setSys] = useState<SysId>('si');
  const [angle, setAngle] = useState(0);
  const [noise, setNoise] = useState(0);
  const [fn, setFn] = useState(3);

  const ref = useMemo(() => envOf(sys), [sys]);
  const env = useMemo(() => {
    const Q = rotation([0.3, 0.8, 0.5], (angle * Math.PI) / 180);
    return ref.map((d, k) => {
      const shake: Vec3 = [noise * Math.sin(2.1 * k + 0.4), noise * Math.cos(1.7 * k + 1.3), noise * Math.sin(3.3 * k + 2.2)];
      return apply(Q, [d[0] + shake[0], d[1] + shake[1], d[2] + shake[2]]);
    });
  }, [ref, angle, noise]);
  const refVals = useMemo(() => DEFAULT_SET.map((p) => evaluate(ref, p, RC)), [ref]);
  const vals = useMemo(() => DEFAULT_SET.map((p) => evaluate(env, p, RC)), [env]);
  const p = DEFAULT_SET[fn]!;
  const contrib = useMemo(() => (p.kind === 'G2' ? g2Contributions(env, p, RC) : g3WithShares(env, p, RC).shares), [env, p]);
  const maxDev = Math.max(...vals.map((v, k) => Math.abs(v / (refVals[k] || 1) - 1)));

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage key={sys} view={sys === 'ethanol' ? { position: [4.5, -7.5, 4.5], target: [0, 0, 0] } : { position: [9, -15, 8.5], target: [0, 0, 0] }} background={theme.canvas} label={`Neighbours within ${RC} Å of one atom in ${SYS[sys].label}, coloured by their contribution to ${label(p)}.`} minDistance={4} maxDistance={50}>
            <FingerprintScene theme={theme} env={env} contrib={contrib} centreSym={SYS[sys].s.symbols[SYS[sys].centre]!} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {(Object.keys(SYS) as SysId[]).map((k) => (
                <button key={k} type="button" aria-pressed={sys === k} onClick={() => setSys(k)}>
                  {SYS[k].label}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.phys }} /> larger contribution to {label(p)}
            </li>
            <li>
              <span className="ring" /> cutoff {RC} Å (cosine f<sub>c</sub>)
            </li>
          </ul>
        </div>
        <FingerprintChart theme={theme} vals={vals} refVals={refVals} sel={fn} onSel={setFn} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-a`}>
            <span>rotate structure</span>
            <input id={`${ids}-a`} type="range" min={0} max={360} step={1} value={angle} onChange={(e) => setAngle(Number(e.target.value))} />
            <output>{angle}°</output>
          </label>
          <label htmlFor={`${ids}-n`}>
            <span>thermal distortion</span>
            <input id={`${ids}-n`} type="range" min={0} max={0.3} step={0.01} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
            <output>{noise.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt className="math">{label(p)}</dt>
            <dd>{vals[fn]!.toFixed(4)}</dd>
          </div>
          <div className={maxDev < 1e-9 ? 'is-good' : 'is-alert'}>
            <dt>Largest change vs reference</dt>
            <dd>{maxDev < 1e-9 ? '0 (unchanged)' : `${(maxDev * 100).toFixed(1)} %`}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Rotate as far as you like: every bar stays at 1, because each function depends only on distances and angles.
          Add distortion and the fingerprint moves. A Behler–Parrinello network sees nothing else: each element has its
          own network that maps this vector to an atomic energy.
        </p>
      </div>
    </div>
  );
}

function FingerprintScene({ theme, env, contrib, centreSym }: { theme: SceneTheme; env: Vec3[]; contrib: number[]; centreSym: string }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 10, 40);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = env.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const max = Math.max(1e-12, ...contrib);
    const stone = linear(theme.dark ? '#4a5568' : '#c3c9d2');
    const hot = linear(theme.phys);
    pos.set([0, 0, 0], 0);
    radii[0] = centreSym === 'H' ? 0.35 : 0.75;
    colors.set(linear(theme.ink), 0);
    env.forEach((d, k) => {
      pos.set(d, 3 * (k + 1));
      const t = Math.min(1, contrib[k]! / max);
      radii[k + 1] = 0.32 + 0.38 * Math.sqrt(t);
      colors.set(mixRGB(stone, hot, Math.sqrt(t)), 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [env, contrib, theme, centreSym]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Shell center={[0, 0, 0]} radius={RC} color={theme.phys} opacity={0.25} grid />
    </>
  );
}

function FingerprintChart({ theme, vals, refVals, sel, onSel }: { theme: SceneTheme; vals: number[]; refVals: number[]; sel: number; onSel: (k: number) => void }) {
  const W = 520;
  const H = 330;
  const M = { l: 44, r: 12, t: 24, b: 90 };
  const n = vals.length;
  const bw = (W - M.l - M.r) / n;
  const ratios = vals.map((v, k) => (refVals[k] ? v / refVals[k]! : 1));
  const top = Math.max(1.6, ...ratios);
  const y = (v: number) => M.t + (1 - v / top) * (H - M.t - M.b);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Fingerprint of ${n} symmetry functions relative to the undistorted structure: ${ratios.map((r) => r.toFixed(3)).join(', ')}.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">fingerprint relative to the reference structure</text>
        {[0, 0.5, 1, 1.5].filter((v) => v <= top).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className={v === 1 ? 'axis' : 'grid'} />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        {ratios.map((r, k) => (
          <g key={k} role="button" tabIndex={0} aria-label={`${label(DEFAULT_SET[k]!)}: ${r.toFixed(3)}`} onClick={() => onSel(k)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSel(k)} style={{ cursor: 'pointer' }}>
            <rect x={M.l + k * bw + bw * 0.18} y={y(r)} width={bw * 0.64} height={Math.max(0, y(0) - y(r))} style={{ fill: k === sel ? theme.phys : DEFAULT_SET[k]!.kind === 'G2' ? theme.arch : theme.lit, opacity: k === sel ? 1 : 0.7 }} />
            <text x={M.l + k * bw + bw / 2} y={H - M.b + 10} transform={`rotate(-55 ${M.l + k * bw + bw / 2} ${H - M.b + 10})`} textAnchor="end" className={`tick${k === sel ? ' strong' : ''}`}>
              {label(DEFAULT_SET[k]!)}
            </text>
          </g>
        ))}
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> radial G²</span>
        <span className="item"><span className="swatch" style={{ background: theme.lit }} /> angular G³</span>
        <span className="item">click a bar to colour the neighbours</span>
      </figcaption>
    </figure>
  );
}
