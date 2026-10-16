import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { angleScan, blindAngle, makeToySchNet, outerDistance, triatomic, type Vec3 } from '~/lib/models/schnet';

// A real SchNet forward pass (random, untrained weights) on three silicon atoms.
// SchNet's filters see only distances, so the centre atom learns the bond angle only
// through the j–k edge: after one block it is blind to the angle, and if j and k are
// outside each other's cutoff it stays blind however many blocks are stacked.

const R = 2.3517; // diamond Si nearest-neighbour distance, Å
const BLOCKS = 3;
const THETAS = Array.from({ length: 151 }, (_, k) => 30 + k);
const LOG_LO = -6;
const LOG_HI = -1;

export default function SchNetAngle() {
  const theme = useSceneTheme();
  const ids = useId();
  const [theta, setTheta] = useState(109.47);
  const [rc, setRc] = useState(4.5);
  const model = useMemo(() => makeToySchNet({ rc, blocks: BLOCKS, elements: [14] }), [rc]);
  const scan = useMemo(() => angleScan(model, [14, 14, 14], R, THETAS, theta), [model, theta]);
  const rjk = outerDistance(R, theta);
  const seen = rjk < rc;
  const thStar = blindAngle(R, rc);
  const colors = [theme.fail, theme.caveat, theme.arch];

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [1.2, -3, 26], target: [1.2, 1.2, 0] }}
            background={theme.canvas}
            label={`Three silicon atoms: a centre atom bonded to two neighbours ${R} Å away at a ${theta.toFixed(1)}° angle. The neighbours are ${rjk.toFixed(2)} Å apart, ${seen ? 'inside' : 'outside'} each other's ${rc} Å cutoff.`}
            minDistance={8}
            maxDistance={60}
          >
            <TriScene theme={theme} pos={triatomic(R, theta)} rc={rc} seen={seen} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="bar" style={{ background: theme.ink }} /> bonds i–j, i–k (seen from block 1)
            </li>
            <li>
              <span className="bar thin" style={{ background: seen ? theme.phys : theme.ruleStrong }} /> edge j–k {seen ? '(in the graph)' : '(beyond r_c: absent)'}
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys, opacity: 0.5 }} /> cutoff sphere around j
            </li>
          </ul>
        </div>
        <AngleChart theme={theme} thetas={THETAS} diff={scan.diff} theta={theta} thStar={thStar} colors={colors} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-th`}>
            <span>bond angle θ</span>
            <input id={`${ids}-th`} type="range" min={30} max={180} step={0.5} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>{theta.toFixed(1)}°</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>cutoff r<sub>c</sub></span>
            <input id={`${ids}-rc`} type="range" min={3} max={6} step={0.1} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>r<sub>jk</sub> = 2r sin(θ/2)</dt>
            <dd>{rjk.toFixed(2)} Å</dd>
          </div>
          <div className={seen ? 'is-good' : 'is-alert'}>
            <dt>edge j–k in the graph</dt>
            <dd>{seen ? 'yes' : 'no'}</dd>
          </div>
          <div>
            <dt>angles a SchNet of any depth confuses</dt>
            <dd>{thStar >= 180 ? 'none (r_c ≥ 2r)' : `all θ > ${thStar.toFixed(1)}°`}</dd>
          </div>
        </dl>
        <p className="lab-note">
          The chart asks: after <i>T</i> blocks, how different are the centre atom’s features at every other angle from
          those at the angle you set? Block 1 sums filters of <i>r</i><sub>ij</sub> and <i>r</i><sub>ik</sub> only, so the
          answer is exactly zero everywhere. From block 2, neighbour <i>j</i> brings back what it learned from <i>k</i>{' '}
          through <i>r</i><sub>jk</sub>, which encodes the angle — but only while <i>r</i><sub>jk</sub> &lt; <i>r</i><sub>c</sub>.
          The tetrahedral angle puts the two neighbours 3.84 Å apart.
        </p>
      </div>
    </div>
  );
}

function TriScene({ theme, pos, rc, seen }: { theme: SceneTheme; pos: Vec3[]; rc: number; seen: boolean }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 18, 60);
  const buffers = useMemo<AtomBuffers>(() => {
    const p = new Float32Array(9);
    const colors = new Float32Array(9);
    const radii = new Float32Array([0.62, 0.55, 0.55]);
    pos.forEach((v, i) => p.set(v, 3 * i));
    colors.set(linear(theme.ink), 0);
    colors.set(linear(theme.arch), 3);
    colors.set(linear(theme.arch), 6);
    return { positions: p, radii, colors, metal: new Float32Array(3), ao: new Float32Array(3).fill(1), count: 3 };
  }, [pos, theme]);
  const bonds = useMemo<Segment[]>(() => {
    const c = mixRGB(linear(theme.ink), linear(theme.canvas), 0.25);
    return [1, 2].map((k) => ({ a: pos[0] as [number, number, number], b: pos[k] as [number, number, number], color: c, radius: 0.1 }));
  }, [pos, theme]);
  const jk = useMemo<ColoredLine[]>(() => {
    const c = linear(seen ? theme.phys : theme.ruleStrong);
    return [{ a: pos[1] as [number, number, number], b: pos[2] as [number, number, number], rgba: [c[0], c[1], c[2], seen ? 1 : 0.6] }];
  }, [pos, seen, theme]);
  const jkStick = useMemo<Segment[]>(
    () => (seen ? [{ a: pos[1] as [number, number, number], b: pos[2] as [number, number, number], color: linear(theme.phys), radius: 0.05 }] : []),
    [pos, seen, theme],
  );
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={bonds} uniforms={uniforms} />
      {seen ? <Sticks segments={jkStick} uniforms={uniforms} /> : <Lines lines={jk} />}
      <Shell center={pos[1] as [number, number, number]} radius={rc} color={theme.phys} opacity={0.22} />
    </>
  );
}

function AngleChart({
  theme,
  thetas,
  diff,
  theta,
  thStar,
  colors,
}: {
  theme: SceneTheme;
  thetas: number[];
  diff: number[][];
  theta: number;
  thStar: number;
  colors: string[];
}) {
  const W = 520;
  const H = 360;
  const M = { l: 52, r: 14, t: 26, b: 70 };
  const zeroY = H - M.b + 26;
  const x = (th: number) => M.l + ((th - 30) / 150) * (W - M.l - M.r);
  const y = (v: number) => {
    if (v < 1e-12) return zeroY;
    const l = Math.min(Math.max(Math.log10(v), LOG_LO), LOG_HI);
    return M.t + (1 - (l - LOG_LO) / (LOG_HI - LOG_LO)) * (H - M.t - M.b);
  };
  // Zeros (to round-off) drop to their own lane under the axis; tiny non-zeros clamp to its floor.
  const path = (row: number[]) => row.map((v, k) => `${k ? 'L' : 'M'}${x(thetas[k]!)},${y(v)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Relative change of the centre atom's features between ${theta.toFixed(1)}° and every other angle, after 1, 2 and 3 SchNet blocks. Block 1 is identically zero; blocks 2 and 3 are non-zero only below ${thStar.toFixed(1)}°.`}>
        <text x={M.l} y={M.t - 10} className="axis-label">‖x_i(θ′) − x_i(θ)‖ / ‖x_i(θ)‖ (log scale)</text>
        {thStar < 180 && (
          <g>
            <rect x={x(thStar)} y={M.t} width={x(180) - x(thStar)} height={zeroY + 8 - M.t} style={{ fill: theme.fail, opacity: 0.08 }} />
            <text x={(x(thStar) + x(180)) / 2} y={M.t + 14} textAnchor="middle" className="tick">r_jk &gt; r_c</text>
          </g>
        )}
        {Array.from({ length: LOG_HI - LOG_LO + 1 }, (_, k) => LOG_LO + k).map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={y(10 ** e)} y2={y(10 ** e)} className="grid" />
            <text x={M.l - 6} y={y(10 ** e) + 4} textAnchor="end" className="tick">
              10<tspan dy="-5" fontSize="8">{e}</tspan>
            </text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={zeroY} y2={zeroY} className="axis" strokeDasharray="3 3" />
        <text x={M.l - 6} y={zeroY + 4} textAnchor="end" className="tick">0</text>
        <text x={M.l + 4} y={zeroY - 6} className="tick">exactly zero: these angles look identical</text>
        {diff.map((row, t) => (
          <path key={t} d={path(row)} className="curve" style={{ stroke: colors[t], strokeWidth: t === 0 ? 3.4 : 2.2, opacity: 0.95 }} />
        ))}
        <line x1={x(theta)} x2={x(theta)} y1={M.t} y2={zeroY + 8} style={{ stroke: theme.ink, strokeWidth: 1.2 }} />
        <line x1={x(109.47)} x2={x(109.47)} y1={zeroY + 8} y2={zeroY + 14} className="axis" />
        {[30, 60, 90, 120, 150, 180].map((th) => (
          <text key={th} x={x(th)} y={zeroY + 26} textAnchor="middle" className="tick">
            {th}°
          </text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">other angle θ′</text>
      </svg>
      <figcaption className="lab-legend">
        {colors.map((c, t) => (
          <span key={t} className="item">
            <span className="swatch" style={{ background: c }} /> after {t + 1} block{t ? 's' : ''}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
