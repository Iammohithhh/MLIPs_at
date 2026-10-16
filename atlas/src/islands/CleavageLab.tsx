import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { A_CU, MORSE_CU, cleavageEnergy, coordination, crossPairs, fcc111 } from '~/lib/physics/cleavage';

// TOY MODEL: pull an fcc(111) crystal apart rigidly along its middle plane. Atoms at the new
// surfaces lose neighbours; the bonds across the gap stretch far beyond anything an
// equilibrium structure contains before they break.

const RC = 7;
const S = fcc111(A_CU, 4, 2, 8);
const PAIRS = crossPairs(S, RC);
const MID = S.layers / 2;
const NN_CUT = 3.0;
const COPIES = [-1, 0, 1].flatMap((u) => [-1, 0].map((v) => [u, v + 0.5] as const));
const DS = Array.from({ length: 121 }, (_, i) => (i * 6) / 120);

export default function CleavageLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [d, setD] = useState(1.2);
  const curve = useMemo(() => DS.map((x) => cleavageEnergy(S, PAIRS, x, MORSE_CU, RC)), []);
  const e = cleavageEnergy(S, PAIRS, d, MORSE_CU, RC);
  const eInf = cleavageEnergy(S, PAIRS, 20, MORSE_CU, RC);
  const coord = useMemo(() => coordination(S, d, NN_CUT), [d]);
  const surf = S.atoms.map((a, i) => (a.layer === MID - 1 || a.layer === MID ? coord[i]! : 12));
  const minCoord = Math.min(...surf);
  const bondLen = Math.sqrt(S.nn ** 2 - (S.dz ** 2) + (S.dz + d) ** 2);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [12, -58, 7], target: [0, 0, 0] }} background={theme.canvas} label={`A copper-like fcc crystal pulled apart along a (111) plane with a gap of ${d.toFixed(2)} Å.`} minDistance={12} maxDistance={110}>
            <CleaveScene theme={theme} d={d} coord={surf} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: theme.arch }} /> fewer than 12 neighbours</li>
            <li><span className="bar" style={{ background: theme.arch }} /> bonds across the gap (fade as they break)</li>
          </ul>
        </div>
        <CleavePlot theme={theme} curve={curve} d={d} e={e} eInf={eInf} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-d`}>
            <span>gap opened along (111)</span>
            <input id={`${ids}-d`} type="range" min={0} max={6} step={0.05} value={d} onChange={(ev) => setD(Number(ev.target.value))} />
            <output>{d.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Energy to open the gap</dt>
            <dd>{e.toFixed(0)} meV/Å²</dd>
          </div>
          <div>
            <dt>Fully separated (toy 2γ)</dt>
            <dd>{eInf.toFixed(0)} meV/Å²</dd>
          </div>
          <div>
            <dt>Neighbours of a surface atom</dt>
            <dd>{minCoord} of 12</dd>
          </div>
          <div className={bondLen > 1.25 * S.nn ? 'is-alert' : undefined}>
            <dt>Stretched bond across the gap</dt>
            <dd>{bondLen.toFixed(2)} Å ({((bondLen / S.nn - 1) * 100).toFixed(0)}%)</dd>
          </div>
        </dl>
        <p className="lab-note">
          A bulk crystal near equilibrium never contains a bond stretched by tens of percent or an atom with nine neighbours.
          Predicting a surface energy asks a model trained on such crystals to extrapolate along this whole path.
        </p>
      </div>
    </div>
  );
}

function CleaveScene({ theme, d, coord }: { theme: SceneTheme; d: number; coord: number[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 100);
  const [Lx, Ly] = S.cell;
  const zc = (MID - 0.5) * S.dz;
  const pos = (k: number, u: number, v: number): [number, number, number] => {
    const a = S.atoms[k]!;
    return [a.p[0] + u * Lx - Lx / 2, a.p[1] + v * Ly - Ly / 2, a.p[2] - zc + (a.layer >= MID ? d / 2 : -d / 2)];
  };
  const buffers = useMemo<AtomBuffers>(() => {
    const n = S.atoms.length * COPIES.length;
    const P = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const cu = linear(theme.dark ? '#c98a5e' : '#b87348');
    const hot = linear(theme.arch);
    let m = 0;
    for (const [u, v] of COPIES)
      S.atoms.forEach((_, k) => {
        P.set(pos(k, u, v), 3 * m);
        radii[m] = 0.78;
        const loss = Math.min(1, (12 - coord[k]!) / 3);
        colors.set(mixRGB(cu, hot, loss), 3 * m);
        metal[m] = 0.85 * (1 - loss);
        m++;
      });
    return { positions: P, radii, colors, metal, ao: new Float32Array(n).fill(1), count: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, coord, theme]);
  const sticks = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    const hot = linear(theme.arch);
    for (const q of PAIRS) {
      const r0 = Math.hypot(q.dx, q.dy, q.dz0);
      if (r0 > NN_CUT) continue;
      const r = Math.hypot(q.dx, q.dy, q.dz0 + d);
      const s = (r - r0) / r0;
      if (s > 0.6) continue;
      const a = pos(q.i, 0, 0);
      const b: [number, number, number] = [a[0] + q.dx, a[1] + q.dy, a[2] + q.dz0 + d];
      out.push({ a, b, color: hot, radius: 0.16 * (1 - s / 0.6) + 0.03 });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, theme.arch]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
    </>
  );
}

function CleavePlot({ theme, curve, d, e, eInf }: { theme: SceneTheme; curve: number[]; d: number; e: number; eInf: number }) {
  const W = 520;
  const H = 340;
  const M = { l: 56, r: 16, t: 24, b: 40 };
  const top = Math.ceil((eInf * 1.15) / 50) * 50;
  const X = (x: number) => M.l + (x / 6) * (W - M.l - M.r);
  const Y = (v: number) => H - M.b - (v / top) * (H - M.t - M.b);
  const path = DS.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(curve[i]!).toFixed(1)}`).join('');
  const ticks = Array.from({ length: Math.floor(top / 50) + 1 }, (_, i) => i * 50);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy per area to separate the crystal against the gap; ${e.toFixed(0)} meV/Å² at ${d.toFixed(2)} Å.`}>
        <text x={M.l} y={M.t - 9} className="axis-label">energy per area [meV/Å²]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className={v === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={Y(eInf)} y2={Y(eInf)} className="rc-line" />
        <text x={W - M.r - 6} y={Y(eInf) - 6} textAnchor="end" className="tick">two new surfaces, 2γ</text>
        <path d={path} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.4 }} />
        <line x1={X(d)} x2={X(d)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={X(d)} cy={Y(e)} r={5} style={{ fill: theme.phys, stroke: theme.canvas, strokeWidth: 1.5 }} />
        {[0, 1, 2, 3, 4, 5, 6].map((x) => (
          <text key={x} x={X(x)} y={H - M.b + 16} textAnchor="middle" className="tick">{x}</text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">gap d [Å]</text>
      </svg>
    </figure>
  );
}
