import { useId, useMemo, useState } from 'react';
import { Html } from '@react-three/drei';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, linear, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { coulomb, shortRangeToy } from '~/lib/physics/pair';
import { useLinkedSymbol } from '~/lib/useLinkedSymbol';

// Physics Lab P1, part 1: a Na⁺/Cl⁻ pair, a cutoff sphere, and the Coulomb tail a
// strictly local pair term cannot represent.

const NA_X = -5;
const R_MIN = 2.4;
const R_MAX = 14;

export default function PairCutoffLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [r, setR] = useState(7);
  const [rc, setRc] = useState(5);
  const [active, setSym] = useLinkedSymbol('pair-cutoff');

  const eC = coulomb(r, 1, -1);
  const eSr = shortRangeToy(r, 1, -1, rc);
  const missing = eC - eSr;
  const share = Math.abs(missing / eC);

  return (
    <div className="lab">
      <div className="lab-split">
        <div className="lab-stage lab-stage--short">
          <Stage
            view={{ position: [0.5, -31, 8.5], target: [0.5, 0, -0.5] }}
            background={theme.canvas}
            label={`Two ions ${r.toFixed(1)} Å apart; a cutoff sphere of radius ${rc.toFixed(1)} Å around the Na⁺ ion.`}
            minDistance={15}
            maxDistance={90}
          >
            <PairScene theme={theme} r={r} rc={rc} active={active} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: ELEMENTS.Na.color }} /> Na⁺ <small>+1 e</small>
            </li>
            <li>
              <span className="dot" style={{ background: ELEMENTS.Cl.color }} /> Cl⁻ <small>−1 e</small>
            </li>
            <li>
              <span className={`ring${r < rc ? '' : ' out'}`} /> cutoff sphere {r < rc ? '(pair inside)' : '(pair outside)'}
            </li>
          </ul>
        </div>
        <EnergyPlot r={r} rc={rc} active={active} setSym={setSym} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-r`} data-sym="r">
            <span>
              separation <i>r</i>
            </span>
            <input id={`${ids}-r`} type="range" min={R_MIN} max={R_MAX} step={0.05} value={r} onChange={(e) => setR(Number(e.target.value))} />
            <output>{r.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-rc`} data-sym="rc">
            <span>
              cutoff <i>r</i>
              <sub>c</sub>
            </span>
            <input id={`${ids}-rc`} type="range" min={3} max={9} step={0.1} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div data-sym="EC">
            <dt>Coulomb</dt>
            <dd>{eC.toFixed(3)} eV</dd>
          </div>
          <div data-sym="Esr">
            <dt>Toy short-range term</dt>
            <dd>{eSr.toFixed(3)} eV</dd>
          </div>
          <div className={share > 0.999 ? 'is-alert' : undefined}>
            <dt>Missing tail</dt>
            <dd>
              {missing.toFixed(3)} eV <small>({(share * 100).toFixed(0)}%)</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          {r >= rc ? (
            <>
              The ions are farther apart than <i>r</i>
              <sub>c</sub>: the cutoff term is exactly zero, yet they still attract with {Math.abs(eC).toFixed(2)} eV.
            </>
          ) : (
            <>
              Inside the cutoff the toy term recovers {((1 - share) * 100).toFixed(0)}% of the Coulomb energy; the
              envelope switches it off smoothly as <i>r</i> → <i>r</i>
              <sub>c</sub>.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function PairScene({ theme, r, rc, active }: { theme: SceneTheme; r: number; rc: number; active: string | null }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 90);
  const clX = NA_X + r;
  const buffers = useMemo<AtomBuffers>(
    () => ({
      positions: new Float32Array([NA_X, 0, 0, clX, 0, 0]),
      radii: new Float32Array([ELEMENTS.Na.ionicRadius, ELEMENTS.Cl.ionicRadius]),
      colors: new Float32Array([...linear(ELEMENTS.Na.color), ...linear(ELEMENTS.Cl.color)]),
      metal: new Float32Array([0, 0]),
      ao: new Float32Array([1, 1]),
      count: 2,
    }),
    [clX],
  );
  const inside = r < rc;
  const strength = Math.min(1, 2.82 / r);
  const col = linear(theme.arch);
  const lines = useMemo(
    () => [{ a: [NA_X, 0, 0] as [number, number, number], b: [clX, 0, 0] as [number, number, number], rgba: [col[0], col[1], col[2], 0.25 + 0.65 * strength] as [number, number, number, number] }],
    [clX, col[0], col[1], col[2], strength],
  );
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={lines} />
      <Shell center={[NA_X, 0, 0]} radius={rc} color={inside ? theme.phys : theme.caveat} opacity={active === 'rc' ? 0.95 : 0.6} grid />
      <Html position={[NA_X + rc * 0.71, 0, rc * 0.71 + 0.6]} center className="lab-tag lab-tag--soft">
        r<sub>c</sub>
      </Html>
    </>
  );
}

function EnergyPlot({
  r,
  rc,
  active,
  setSym,
}: {
  r: number;
  rc: number;
  active: string | null;
  setSym: (s: string | null) => void;
}) {
  const W = 520;
  const H = 330;
  const M = { l: 54, r: 16, t: 18, b: 40 };
  const yMin = -6.4;
  const yMax = 0.6;
  const x = (v: number) => M.l + ((v - 2) / (R_MAX - 2)) * (W - M.l - M.r);
  const y = (v: number) => M.t + ((yMax - v) / (yMax - yMin)) * (H - M.t - M.b);
  const rs = useMemo(() => Array.from({ length: 400 }, (_, i) => R_MIN + (i / 399) * (R_MAX - R_MIN)), []);
  const path = (f: (r: number) => number) => rs.map((v, i) => `${i ? 'L' : 'M'}${x(v).toFixed(1)},${y(f(v)).toFixed(1)}`).join('');
  const cPath = path((v) => coulomb(v, 1, -1));
  const sPath = path((v) => shortRangeToy(v, 1, -1, rc));
  // Shaded tail between the curves.
  const tail =
    rs.map((v, i) => `${i ? 'L' : 'M'}${x(v).toFixed(1)},${y(coulomb(v, 1, -1)).toFixed(1)}`).join('') +
    [...rs].reverse().map((v) => `L${x(v).toFixed(1)},${y(shortRangeToy(v, 1, -1, rc)).toFixed(1)}`).join('') +
    'Z';
  const hover = (s: string) => ({ onPointerEnter: () => setSym(s), onPointerLeave: () => setSym(null) });
  const eC = coulomb(r, 1, -1);
  const eS = shortRangeToy(r, 1, -1, rc);

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy against separation for a Na⁺–Cl⁻ pair. Coulomb energy and a toy short-range term that vanishes at r_c = ${rc.toFixed(1)} Å; the shaded region is the missing tail.`}>
        {[-6, -4, -2, 0].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className={v === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="tick">
              {v}
            </text>
          </g>
        ))}
        {[2, 4, 6, 8, 10, 12, 14].map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 18} textAnchor="middle" className="tick">
            {v}
          </text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">
          r [Å]
        </text>
        <text x={12} y={M.t + 4} className="axis-label">
          E [eV]
        </text>
        <path d={tail} className="tail" />
        <path d={cPath} className={`curve coulomb${active === 'EC' ? ' is-active' : ''}`} data-sym="EC" {...hover('EC')} />
        <path d={sPath} className={`curve toy${active === 'Esr' || active === 'u' ? ' is-active' : ''}`} data-sym="Esr" {...hover('Esr')} />
        <line x1={x(rc)} x2={x(rc)} y1={M.t} y2={H - M.b} className={`rc-line${active === 'rc' ? ' is-active' : ''}`} {...hover('rc')} />
        <text x={x(rc) + 5} y={M.t + 12} className="rc-label">
          r<tspan dy="4" fontSize="0.75em">c</tspan>
        </text>
        <line x1={x(r)} x2={x(r)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={x(r)} cy={y(eC)} r={4.5} className="dot coulomb" />
        <circle cx={x(r)} cy={y(eS)} r={4.5} className="dot toy" />
      </svg>
      <figcaption className="lab-legend">
        <span className="item coulomb" data-sym="EC">
          <span className="swatch" /> Coulomb <i>E</i>
          <sub>C</sub>
        </span>
        <span className="item toy" data-sym="Esr">
          <span className="swatch" /> toy short-range term <i>E</i>
          <sub>sr</sub>
        </span>
        <span className="item tail">
          <span className="swatch" /> missing tail
        </span>
      </figcaption>
    </figure>
  );
}
