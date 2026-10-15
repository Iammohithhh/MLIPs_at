import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';

// How far T message-passing layers reach in diamond silicon. Hop counts and reach
// were computed at build time on a cluster large enough to have no boundary effect.

export interface DepthData {
  rc: number;
  /** Hop count from the centre for each displayed atom (−1 = more than maxT hops). */
  hops: number[];
  /** Atoms reachable within T hops, T = 1..maxT (full cluster, not just those drawn). */
  reach: number[];
}

const MARKS: Array<{ T: number; label: string }> = [
  { T: 2, label: 'MACE' },
  { T: 6, label: 'NequIP (periodic)' },
];

export default function ReceptiveDepth({ positions, data, maxT, drawRadius }: { positions: number[][]; data: DepthData[]; maxT: number; drawRadius: number }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [rcIdx, setRcIdx] = useState(1);
  const [T, setT] = useState(2);
  const d = data[rcIdx]!;
  const reachT = d.reach[T - 1]!;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [24, -38, 22], target: [0, 0, 0] }} background={theme.canvas} label={`Diamond silicon around one atom; atoms that can influence it after ${T} layers with a ${d.rc} Å cutoff are coloured by hop count.`} minDistance={10} maxDistance={110}>
            <DepthScene theme={theme} positions={positions} hops={d.hops} T={T} rc={d.rc} drawRadius={drawRadius} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.arch }} /> reached in ≤ {T} hops (darker = fewer)
            </li>
            <li>drawn: atoms within {drawRadius} Å</li>
          </ul>
        </div>
        <ReachChart theme={theme} data={d} maxT={maxT} T={T} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-T`}>
            <span>layers T</span>
            <input id={`${ids}-T`} type="range" min={1} max={maxT} step={1} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <output>{T}</output>
          </label>
          <div className="seg" role="group" aria-label="Cutoff radius">
            <span className="seg-label">r_c</span>
            {data.map((x, k) => (
              <button key={x.rc} type="button" aria-pressed={rcIdx === k} onClick={() => setRcIdx(k)}>
                {x.rc} Å
              </button>
            ))}
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Atoms that can influence the centre</dt>
            <dd>{reachT.toLocaleString('en-GB')}</dd>
          </div>
          <div>
            <dt className="math">Formal radius T·r_c</dt>
            <dd>{(T * d.rc).toFixed(0)} Å</dd>
          </div>
        </dl>
        <p className="lab-note">
          Each layer passes information one cutoff further, so the number of atoms that can affect one prediction grows
          roughly with the cube of the depth. Two-body message-passing models need several layers to build up angular and
          many-body information; in a dense solid that means thousands of atoms per prediction.
        </p>
      </div>
    </div>
  );
}

function DepthScene({ theme, positions, hops, T, rc, drawRadius }: { theme: SceneTheme; positions: number[][]; hops: number[]; T: number; rc: number; drawRadius: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 90);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const near = linear(theme.arch);
    const far = linear(theme.lit);
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      const h = hops[i]!;
      if (i === 0) {
        radii[i] = 0.95;
        colors.set(linear(theme.ink), 3 * i);
      } else if (h > 0 && h <= T) {
        radii[i] = 0.62;
        colors.set(mixRGB(near, far, T > 1 ? (h - 1) / (T - 1) : 0), 3 * i);
      } else {
        radii[i] = 0.3;
        colors.set(mixRGB(stone, linear(theme.canvas), 0.3), 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, hops, T, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Shell center={[0, 0, 0]} radius={rc} color={theme.phys} opacity={0.4} />
      {T * rc <= drawRadius + 2 && T > 1 && <Shell center={[0, 0, 0]} radius={T * rc} color={theme.arch} opacity={0.22} grid />}
    </>
  );
}

function ReachChart({ theme, data, maxT, T }: { theme: SceneTheme; data: DepthData; maxT: number; T: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 56, r: 16, t: 20, b: 46 };
  const lo = 1;
  const hi = Math.log10(Math.max(...data.reach) * 1.6);
  const x = (t: number) => M.l + ((t - 0.5) / maxT) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(v, 10)) - lo) / (hi - lo)) * (H - M.t - M.b);
  const bw = ((W - M.l - M.r) / maxT) * 0.56;
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Atoms within reach after 1 to ${maxT} layers with r_c = ${data.rc} Å: ${data.reach.join(', ')}.`}>
        <text x={M.l} y={M.t - 6} className="axis-label">atoms that can influence one prediction (log scale)</text>
        {[10, 100, 1000, 10000].filter((v) => Math.log10(v) <= hi).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toLocaleString('en-GB')}</text>
          </g>
        ))}
        {data.reach.map((r, k) => (
          <g key={k}>
            <rect x={x(k + 1) - bw / 2} y={y(r)} width={bw} height={y(10) - y(r)} style={{ fill: k + 1 === T ? theme.arch : theme.ruleStrong }} />
            <text x={x(k + 1)} y={y(r) - 5} textAnchor="middle" className="tick">{r.toLocaleString('en-GB')}</text>
            <text x={x(k + 1)} y={H - M.b + 16} textAnchor="middle" className={`tick${k + 1 === T ? ' strong' : ''}`}>T = {k + 1}</text>
          </g>
        ))}
        {MARKS.map((m) => (
          <text key={m.label} x={x(m.T)} y={H - M.b + 32} textAnchor="middle" className="tick" style={{ fill: theme.lit }}>
            {m.label}
          </text>
        ))}
      </svg>
    </figure>
  );
}
