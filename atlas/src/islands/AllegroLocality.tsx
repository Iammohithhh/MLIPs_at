import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';

// Allegro keeps its features on ordered pairs ij. A layer updates pair ij from the
// directions to every neighbour k of atom i inside the same cutoff (Eq. 16 of the
// paper) — nothing outside atom i's cutoff sphere ever enters. Compared here with the
// growing reach of message passing.

export interface LocalityData {
  rc: number;
  /** Atoms reachable by message passing within T hops, T = 1..maxT. */
  mpReach: number[];
}

export default function AllegroLocality({ positions, data, maxT }: { positions: number[][]; data: LocalityData; maxT: number }) {
  const theme = useSceneTheme();
  const ids = useId();
  const rc = data.rc;
  const inside = useMemo(
    () => positions.map((p, i) => ({ i, r: Math.hypot(p[0]!, p[1]!, p[2]!) })).filter((a) => a.i > 0 && a.r < rc).map((a) => a.i),
    [positions, rc],
  );
  const [jIdx, setJIdx] = useState(0);
  const [T, setT] = useState(3);
  const j = inside[Math.min(jIdx, inside.length - 1)]!;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [14, -22, 12], target: [0, 0, 0] }} background={theme.canvas} label={`Diamond silicon: the pair from the central atom to neighbour ${jIdx + 1} and the ${inside.length - 1} other neighbours whose directions update it, all inside one ${rc} Å cutoff.`} minDistance={6} maxDistance={70}>
            <PairScene theme={theme} positions={positions} inside={inside} j={j} rc={rc} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.arch }} /> pair ij (carries the features)
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys }} /> directions r̂ᵢₖ used to update it
            </li>
          </ul>
        </div>
        <LocalityChart theme={theme} data={data} maxT={maxT} T={T} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-j`}>
            <span>pair i → j</span>
            <input id={`${ids}-j`} type="range" min={0} max={inside.length - 1} step={1} value={jIdx} onChange={(e) => setJIdx(Number(e.target.value))} />
            <output>j = {jIdx + 1}</output>
          </label>
          <label htmlFor={`${ids}-T`}>
            <span>layers</span>
            <input id={`${ids}-T`} type="range" min={1} max={maxT} step={1} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <output>{T}</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className="is-good">
            <dt>Allegro: atoms that can affect Eᵢ</dt>
            <dd>{inside.length}</dd>
          </div>
          <div>
            <dt>Message passing, {T} layers</dt>
            <dd>{data.mpReach[T - 1]!.toLocaleString('en-GB')}</dd>
          </div>
        </dl>
        <p className="lab-note">
          However many Allegro layers you stack, atom <i>i</i>’s energy depends only on atoms inside its own cutoff sphere,
          because pair features are refined with directions from atom <i>i</i>’s neighbourhood and never passed between
          atoms. That is what lets each atom’s energy be computed independently on separate processors.
        </p>
      </div>
    </div>
  );
}

function PairScene({ theme, positions, inside, j, rc }: { theme: SceneTheme; positions: number[][]; inside: number[]; j: number; rc: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 70);
  const inSet = useMemo(() => new Set(inside), [inside]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = positions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const stone = linear(theme.dark ? '#4a5568' : '#b9c0ca');
    positions.forEach((p, i) => {
      pos.set(p, 3 * i);
      if (i === 0) {
        radii[i] = 0.9;
        colors.set(linear(theme.ink), 3 * i);
      } else if (i === j) {
        radii[i] = 0.8;
        colors.set(linear(theme.arch), 3 * i);
      } else if (inSet.has(i)) {
        radii[i] = 0.62;
        colors.set(linear(theme.phys), 3 * i);
      } else {
        radii[i] = 0.32;
        colors.set(stone, 3 * i);
      }
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [positions, inSet, j, theme]);
  const pair = useMemo<Segment[]>(() => [{ a: [0, 0, 0], b: positions[j] as [number, number, number], color: linear(theme.arch), radius: 0.13 }], [positions, j, theme.arch]);
  const dirs = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.phys);
    const soft = mixRGB(c, linear(theme.canvas), 0.2);
    return inside.filter((k) => k !== j).map((k) => ({ a: [0, 0, 0], b: positions[k] as [number, number, number], rgba: [soft[0], soft[1], soft[2], 0.85] }));
  }, [inside, j, positions, theme.phys, theme.canvas]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={pair} uniforms={uniforms} />
      <Lines lines={dirs} />
      <Shell center={[0, 0, 0]} radius={rc} color={theme.phys} opacity={0.45} grid />
    </>
  );
}

function LocalityChart({ theme, data, maxT, T }: { theme: SceneTheme; data: LocalityData; maxT: number; T: number }) {
  const W = 520;
  const H = 340;
  const M = { l: 58, r: 16, t: 22, b: 44 };
  const allegro = data.mpReach[0]!;
  const hi = Math.log10(Math.max(...data.mpReach) * 1.6);
  const x = (t: number) => M.l + ((t - 1) / (maxT - 1)) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(v, 10)) - 1) / (hi - 1)) * (H - M.t - M.b);
  const ts = Array.from({ length: maxT }, (_, k) => k + 1);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Atoms that can affect one atomic energy against the number of layers: Allegro stays at ${allegro}; message passing grows to ${data.mpReach.at(-1)}.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">atoms that can affect one atomic energy (log scale), r_c = {data.rc} Å</text>
        {[10, 100, 1000, 10000].filter((v) => Math.log10(v) <= hi).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} textAnchor="end" className="tick">{v.toLocaleString('en-GB')}</text>
          </g>
        ))}
        <path d={ts.map((t, k) => `${k ? 'L' : 'M'}${x(t)},${y(data.mpReach[t - 1]!)}`).join('')} className="curve" style={{ stroke: theme.lit }} />
        <path d={ts.map((t, k) => `${k ? 'L' : 'M'}${x(t)},${y(allegro)}`).join('')} className="curve" style={{ stroke: theme.phys, strokeWidth: 3 }} />
        {ts.map((t) => (
          <g key={t}>
            <circle cx={x(t)} cy={y(data.mpReach[t - 1]!)} r={t === T ? 5.5 : 3.5} style={{ fill: theme.lit }} />
            <circle cx={x(t)} cy={y(allegro)} r={t === T ? 5.5 : 3.5} style={{ fill: theme.phys }} />
            <text x={x(t)} y={H - M.b + 16} textAnchor="middle" className={`tick${t === T ? ' strong' : ''}`}>{t}</text>
          </g>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">layers</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> Allegro (strictly local)</span>
        <span className="item"><span className="swatch" style={{ background: theme.lit }} /> message passing</span>
      </figcaption>
    </figure>
  );
}
