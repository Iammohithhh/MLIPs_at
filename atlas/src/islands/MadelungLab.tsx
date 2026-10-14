import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, TintBox, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { NACL_R0, rockSaltCluster } from '~/lib/chem/structures';
import { MADELUNG_NACL, cubeSeries, madelungEwald, sphereSeries } from '~/lib/physics/madelung';

// Physics Lab P1, part 2: the Madelung sum of rock salt summed over a growing
// sphere or cube, against the Ewald result. Everything is computed in the browser.

type Method = 'sphere' | 'cube' | 'evjen';
const N_SHOW = 5; // 3D cluster half-width (11³ = 1331 ions)
const R_PLOT = 12;

const cluster = rockSaltCluster(N_SHOW);
const SPHERE = sphereSeries(R_PLOT);
const CUBE = cubeSeries(R_PLOT);
const EVJEN = cubeSeries(R_PLOT, true);

interface Region {
  inside: (i: number, j: number, k: number) => number; // weight 0..1
}

function regionFor(method: Method, R: number): Region {
  if (method === 'sphere') return { inside: (i, j, k) => (i * i + j * j + k * k <= R * R + 1e-9 ? 1 : 0) };
  const N = Math.round(R);
  return {
    inside: (i, j, k) => {
      const m = Math.max(Math.abs(i), Math.abs(j), Math.abs(k));
      if (m > N) return 0;
      if (method === 'cube') return 1;
      let w = 1;
      if (Math.abs(i) === N) w *= 0.5;
      if (Math.abs(j) === N) w *= 0.5;
      if (Math.abs(k) === N) w *= 0.5;
      return w;
    },
  };
}

/** Partial sum, ions counted and net (weighted) charge inside the region, for any R. */
function regionStats(method: Method, R: number) {
  const reg = regionFor(method, R);
  const L = Math.ceil(R);
  let sum = 0;
  let count = 0;
  let net = 0;
  for (let i = -L; i <= L; i++)
    for (let j = -L; j <= L; j++)
      for (let k = -L; k <= L; k++) {
        const w = reg.inside(i, j, k);
        if (!w) continue;
        const q = (((i + j + k) % 2) + 2) % 2 === 0 ? 1 : -1;
        net += w * q;
        if (i === 0 && j === 0 && k === 0) continue;
        count++;
        sum += (w * q) / Math.sqrt(i * i + j * j + k * k);
      }
  return { M: -sum, count, net };
}

export default function MadelungLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [method, setMethod] = useState<Method>('sphere');
  const [R, setR] = useState(2.6);
  // Ewald runs in the browser after hydration (timing differs from the server render).
  const [ewald, setEwald] = useState<{ M: number; ms: number | null }>(() => ({ M: MADELUNG_NACL, ms: null }));
  useEffect(() => {
    const t0 = performance.now();
    const M = madelungEwald();
    setEwald({ M, ms: performance.now() - t0 });
  }, []);

  const Reff = method === 'sphere' ? R : Math.max(1, Math.round(R));
  const stats = useMemo(() => regionStats(method, Reff), [method, Reff]);
  const err = stats.M - ewald.M;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [30, -48, 28], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Rock-salt crystal of 1331 ions. Ions inside the summation ${method === 'sphere' ? 'sphere' : 'cube'} are shown in full colour.`}
            minDistance={20}
            maxDistance={120}
          >
            <CrystalScene theme={theme} method={method} R={Reff} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Summation region">
              {(
                [
                  ['sphere', 'Spheres'],
                  ['cube', 'Cubes'],
                  ['evjen', 'Neutral cubes (Evjen)'],
                ] as const
              ).map(([id, label]) => (
                <button key={id} type="button" aria-pressed={method === id} onClick={() => setMethod(id)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <SumPlot method={method} R={Reff} ewald={ewald.M} current={stats.M} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-R`}>
            <span>{method === 'sphere' ? 'sphere radius R' : 'cube half-width N'}</span>
            <input
              id={`${ids}-R`}
              type="range"
              min={1}
              max={N_SHOW}
              step={method === 'sphere' ? 0.02 : 1}
              value={method === 'sphere' ? R : Reff}
              onChange={(e) => setR(Number(e.target.value))}
            />
            <output>
              {Reff.toFixed(method === 'sphere' ? 2 : 0)} r<sub>0</sub> = {(Reff * NACL_R0).toFixed(1)} Å
            </output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Ions summed</dt>
            <dd>{stats.count.toLocaleString('en-GB')}</dd>
          </div>
          <div className={Math.abs(stats.net) > 1.5 ? 'is-alert' : undefined}>
            <dt>Net charge inside</dt>
            <dd>
              {stats.net > 0 ? '+' : stats.net < 0 ? '−' : ''}
              {Math.abs(stats.net).toFixed(method === 'evjen' ? 2 : 0)} e
            </dd>
          </div>
          <div>
            <dt>Partial sum M</dt>
            <dd>{stats.M.toFixed(4)}</dd>
          </div>
          <div className={Math.abs(err) > 0.1 ? 'is-alert' : 'is-good'}>
            <dt>Error vs Ewald</dt>
            <dd>{err >= 0 ? '+' : '−'}{Math.abs(err).toFixed(4)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          {ewald.ms === null ? (
            <>Ewald summation (running in your browser…): </>
          ) : (
            <>Ewald summation, computed in your browser in {ewald.ms.toFixed(1)} ms: </>
          )}
          <strong>M = {ewald.M.toFixed(6)}</strong> (tabulated {MADELUNG_NACL.toFixed(6)}).{' '}
          {method === 'sphere'
            ? 'A sphere cuts through shells unevenly, so it encloses a net charge; that charge, not the far-away detail, drives the swings.'
            : method === 'cube'
              ? 'A cube encloses at most one surplus charge, so the sum settles, slowly, as the cube grows.'
              : 'Evjen weights (½ on faces, ¼ on edges, ⅛ on corners) make every cube neutral, and the sum converges within a few shells.'}
        </p>
      </div>
    </div>
  );
}

function CrystalScene({ theme, method, R }: { theme: SceneTheme; method: Method; R: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 110);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = cluster.atoms.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const reg = regionFor(method, R);
    const stone = linear(theme.dark ? theme.canvas : '#b7bec9');
    const partial = linear(theme.caveat);
    const na = linear(ELEMENTS.Na.color);
    const cl = linear(ELEMENTS.Cl.color);
    cluster.atoms.forEach((a, idx) => {
      positions.set(a.p, 3 * idx);
      const [i, j, k] = a.p.map((v) => Math.round(v / NACL_R0)) as [number, number, number];
      const w = reg.inside(i, j, k);
      const base = a.q > 0 ? na : cl;
      const r = (a.q > 0 ? ELEMENTS.Na.ionicRadius : ELEMENTS.Cl.ionicRadius) * 0.92;
      if (w === 1) {
        colors.set(base, 3 * idx);
        radii[idx] = r;
      } else if (w > 0) {
        colors.set(mixRGB(base, partial, 0.55), 3 * idx);
        radii[idx] = r * (0.55 + 0.45 * w);
      } else {
        colors.set(mixRGB(base, stone, theme.dark ? 0.78 : 0.74), 3 * idx);
        radii[idx] = r * 0.3;
      }
    });
    return { positions, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [theme, method, R]);
  const side = (2 * R + 1) * NACL_R0;
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {method === 'sphere' ? (
        <Shell center={[0, 0, 0]} radius={R * NACL_R0 + 0.35} color={theme.phys} opacity={0.5} />
      ) : (
        <TintBox center={[0, 0, 0]} size={[side - NACL_R0 + 0.8, side - NACL_R0 + 0.8, side - NACL_R0 + 0.8]} color={theme.phys} opacity={0.06} />
      )}
      <Shell center={[0, 0, 0]} radius={ELEMENTS.Na.ionicRadius * 1.6} color={theme.ink} opacity={0.7} />
    </>
  );
}

function SumPlot({ method, R, ewald, current }: { method: Method; R: number; ewald: number; current: number }) {
  const W = 520;
  const H = 360;
  const M = { l: 46, r: 14, t: 16, b: 40 };
  const yMin = -11;
  const yMax = 15;
  const x = (v: number) => M.l + (v / R_PLOT) * (W - M.l - M.r);
  const y = (v: number) => M.t + ((yMax - v) / (yMax - yMin)) * (H - M.t - M.b);
  const clampY = (v: number) => y(Math.max(yMin, Math.min(yMax, v)));
  let sphere = `M${x(0)},${clampY(0)}`;
  let prev = 0;
  for (const p of SPHERE) {
    sphere += `L${x(p.R).toFixed(1)},${clampY(prev).toFixed(1)}L${x(p.R).toFixed(1)},${clampY(p.M).toFixed(1)}`;
    prev = p.M;
  }
  sphere += `L${x(R_PLOT)},${clampY(prev)}`;
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Partial Madelung sums against summation size. Spheres swing between about −10 and 15; cubes approach ${ewald.toFixed(4)}; neutral Evjen cubes reach it within a few shells.`}>
        {[-10, -5, 0, 5, 10, 15].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className={v === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="tick">
              {v}
            </text>
          </g>
        ))}
        {[0, 2, 4, 6, 8, 10, 12].map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 18} textAnchor="middle" className="tick">
            {v}
          </text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">
          R or N [r₀]
        </text>
        <text x={10} y={M.t + 2} className="axis-label">
          M
        </text>
        <rect x={M.l} y={M.t} width={x(N_SHOW) - M.l} height={H - M.t - M.b} className="shown-band" />
        <text x={x(N_SHOW) - 4} y={M.t + 12} textAnchor="end" className="band-label">
          shown in 3D
        </text>
        <line x1={M.l} x2={W - M.r} y1={y(ewald)} y2={y(ewald)} className="ewald" />
        <text x={W - M.r - 2} y={y(ewald) + 16} textAnchor="end" className="ewald-label">
          Ewald M = {ewald.toFixed(4)}
        </text>
        <path d={sphere} className={`curve sphere${method === 'sphere' ? ' is-active' : ''}`} />
        {CUBE.map((p) => (
          <circle key={`c${p.R}`} cx={x(p.R)} cy={clampY(p.M)} r={3.6} className={`pt cube${method === 'cube' ? ' is-active' : ''}`} />
        ))}
        {EVJEN.map((p) => (
          <rect
            key={`e${p.R}`}
            x={x(p.R) - 3.4}
            y={clampY(p.M) - 3.4}
            width={6.8}
            height={6.8}
            transform={`rotate(45 ${x(p.R)} ${clampY(p.M)})`}
            className={`pt evjen${method === 'evjen' ? ' is-active' : ''}`}
          />
        ))}
        <line x1={x(R)} x2={x(R)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={x(R)} cy={clampY(current)} r={6} className={`now ${method}`} />
      </svg>
      <figcaption className="lab-legend">
        <span className="item sphere">
          <span className="swatch" /> spheres
        </span>
        <span className="item cube">
          <span className="swatch" /> cubes
        </span>
        <span className="item evjen">
          <span className="swatch" /> neutral cubes
        </span>
        <span className="item ewald">
          <span className="swatch" /> Ewald
        </span>
      </figcaption>
    </figure>
  );
}
