import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import slab from '~/data/structures/au111-slab.json';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  ImpostorAtoms,
  Lines,
  TintBox,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import type { Vec3 } from '~/lib/structure/types';
import { potentialStep, slabEnergyPerArea, slabProfile, vacuumField } from '~/lib/physics/slab-dipole';

// Physics Lab P9: a polar slab and its periodic images along z. Without a correction
// the images force an artificial field through the vacuum; the dipole correction
// removes it and puts the potential step at the cell boundary instead.

const POS = slab.positions as Vec3[];
const A1 = slab.cell[0] as Vec3;
const A2 = slab.cell[1] as Vec3;
const AREA = Math.abs(A1[0] * A2[1] - A1[1] * A2[0]);
const zs = POS.map((p) => p[2]);
const Z_MIN = Math.min(...zs);
const THICK = Math.max(...zs) - Z_MIN;
const BOTTOM = 3; // vacuum below the slab inside the cell (Å)
const SHEET_D = 1.6;
const ION_H = 2.2;
// In-plane centre of the cell and hollow-site shift for adsorbed ions.
const CX = (A1[0] + A2[0]) / 2;
const CY = (A1[1] + A2[1]) / 2;
const as = Math.hypot(A1[0], A1[1]) / 4;

export default function SlabDipoleLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [vacuum, setVacuum] = useState(16);
  const [p, setP] = useState(0.005);
  const [corrected, setCorrected] = useState(false);
  const [resetKey, setResetKey] = useState(0);

  const Lz = THICK + BOTTOM + vacuum;
  const zTop = BOTTOM + THICK;
  const prof = useMemo(() => slabProfile({ p, Lz, zSheet: zTop, d: SHEET_D }), [p, Lz, zTop]);
  const Evac = vacuumField(p, Lz);
  const step = potentialStep(p);
  const uCell = slabEnergyPerArea(p, Lz) * AREA * 1000; // meV per surface cell

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [52, -150, 26], target: [0, 0, -2] }}
            resetKey={resetKey}
            background={theme.canvas}
            label={`Au(111) slab with a dipole layer, repeated along z with ${vacuum.toFixed(0)} Å of vacuum. ${corrected ? 'Dipole correction on: no field in the vacuum.' : `Periodic images force a field of ${Evac.toFixed(3)} V/Å through the vacuum.`}`}
            minDistance={20}
            maxDistance={260}
          >
            <SlabScene theme={theme} Lz={Lz} zTop={zTop} p={p} corrected={corrected} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Boundary condition">
              <button type="button" aria-pressed={!corrected} onClick={() => setCorrected(false)}>
                Periodic
              </button>
              <button type="button" aria-pressed={corrected} onClick={() => setCorrected(true)}>
                Dipole-corrected
              </button>
            </div>
          </div>
          <button type="button" className="ix-reset lab-reset" onClick={() => setResetKey((k) => k + 1)}>
            Reset view
          </button>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: ELEMENTS.Na.color }} /> adsorbed cations (+σ)
            </li>
            <li>
              <span className="bar" style={{ background: theme.caveat }} /> image charge in the surface (−σ)
            </li>
            <li>
              <span className="bar" style={{ background: corrected ? theme.phys : theme.fail }} />{' '}
              {corrected ? 'compensating dipole layer' : 'artificial field'}
            </li>
          </ul>
        </div>
        <PotentialPlot theme={theme} prof={prof} Lz={Lz} corrected={corrected} vacuum={vacuum} p={p} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-v`}>
            <span>vacuum</span>
            <input id={`${ids}-v`} type="range" min={6} max={60} step={1} value={vacuum} onChange={(e) => setVacuum(Number(e.target.value))} />
            <output>{vacuum.toFixed(0)} Å</output>
          </label>
          <label htmlFor={`${ids}-p`}>
            <span>surface dipole p</span>
            <input id={`${ids}-p`} type="range" min={0} max={0.01} step={0.0002} value={p} onChange={(e) => setP(Number(e.target.value))} />
            <output>{(p * 1000).toFixed(1)} me/Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={corrected ? 'is-good' : 'is-alert'}>
            <dt>Field in vacuum</dt>
            <dd>{corrected ? '0' : Evac.toFixed(4)} V/Å</dd>
          </div>
          <div>
            <dt>Step across the layer</dt>
            <dd>{step.toFixed(3)} V</dd>
          </div>
          <div className={corrected ? 'is-good' : undefined}>
            <dt>Spurious energy / cell</dt>
            <dd>
              {corrected ? 'removed' : `${uCell.toFixed(1)} meV`} <small>({AREA.toFixed(0)} Å²)</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          {corrected ? (
            <>
              With the correction the vacuum is field-free on both sides and the potential steps by p/ε₀ ={' '}
              {step.toFixed(2)} V across the dipole layer, the difference in work function between the two faces. The
              compensating jump sits at the cell boundary, inside the vacuum, where no charge feels it.
            </>
          ) : (
            <>
              Periodicity forces the average field over the cell to vanish, so the slab’s dipole is cancelled by a uniform
              field of p/(ε₀L<sub>z</sub>) = {Evac.toFixed(4)} V/Å running through the vacuum and the slab. Doubling the
              vacuum only halves it: the error decays as 1/L<sub>z</sub>.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function SlabScene({ theme, Lz, zTop, p, corrected }: { theme: SceneTheme; Lz: number; zTop: number; p: number; corrected: boolean }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 80, 220);
  const zc = Lz / 2; // centre the main cell vertically
  const images = [-1, 0, 1];

  const buffers = useMemo<AtomBuffers>(() => {
    const top = POS.filter((q) => Math.abs(q[2] - (Z_MIN + THICK)) < 0.1);
    const ions = top.filter((_, k) => k % 2 === 0).map((q) => [q[0] + as / 2, q[1] + (as * Math.sqrt(3)) / 6, q[2] + ION_H] as Vec3);
    const all = [...POS.map((q) => ({ q, el: 'Au' as const })), ...ions.map((q) => ({ q, el: 'Na' as const }))];
    const n = all.length * images.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const stone = linear(theme.dark ? theme.canvas : '#b7bec9');
    let k = 0;
    for (const im of images)
      for (const a of all) {
        const z = a.q[2] - Z_MIN + BOTTOM + im * Lz - zc;
        positions.set([a.q[0] - CX, a.q[1] - CY, z], 3 * k);
        radii[k] = a.el === 'Au' ? 1.25 : 0.9;
        const base = linear(ELEMENTS[a.el].color);
        colors.set(im === 0 ? base : mixRGB(base, stone, 0.55), 3 * k);
        metal[k] = a.el === 'Au' ? (im === 0 ? 1 : 0.5) : 0;
        k++;
      }
    return { positions, radii, colors, metal, ao: new Float32Array(n).fill(1), count: n };
  }, [theme, Lz, zc]);

  // Artificial field arrows in every vacuum gap (periodic case only).
  const E = vacuumField(p, Lz);
  const arrows = useMemo<Segment[]>(() => {
    if (corrected || p === 0) return [];
    const len = Math.min(9, 600 * E);
    const col = linear(theme.fail);
    const out: Segment[] = [];
    for (const im of [-1, 0, 1]) {
      const gapMid = zTop + (Lz - zTop) / 2 + im * Lz - zc;
      for (const gx of [-1, 0, 1])
        for (const gy of [-1, 0, 1]) {
          const x = gx * 3.6;
          const y = gy * 3.2;
          out.push({ a: [x, y, gapMid - len / 2], b: [x, y, gapMid + len / 2], color: col, radius: 0.12 });
        }
    }
    return out;
  }, [corrected, p, E, theme.fail, Lz, zTop, zc]);

  // Cell edges (hexagonal prism) for the main cell.
  const edges = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.ruleStrong);
    const rgba: [number, number, number, number] = [c[0], c[1], c[2], 0.9];
    const corners = [
      [0, 0],
      [A1[0], A1[1]],
      [A1[0] + A2[0], A1[1] + A2[1]],
      [A2[0], A2[1]],
    ].map(([x, y]) => [x! - CX, y! - CY] as [number, number]);
    const out: ColoredLine[] = [];
    for (const z of [-zc, Lz - zc])
      corners.forEach((p0, i) => {
        const p1 = corners[(i + 1) % 4]!;
        out.push({ a: [p0[0], p0[1], z], b: [p1[0], p1[1], z], rgba });
      });
    corners.forEach((p0) => out.push({ a: [p0[0], p0[1], -zc], b: [p0[0], p0[1], Lz - zc], rgba }));
    return out;
  }, [Lz, zc, theme.ruleStrong]);

  const w = Math.hypot(A1[0] + A2[0], A1[1] + A2[1]) * 0.75;
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={edges} />
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.9} />}
      {images.map((im) => (
        <TintBox key={im} center={[0, 0, zTop + im * Lz - zc]} size={[w, w * 0.8, 0.12]} color={theme.caveat} opacity={im === 0 ? 0.35 : 0.15} />
      ))}
      {corrected && <TintBox center={[0, 0, Lz - zc - 0.6]} size={[w, w * 0.8, 1.2]} color={theme.phys} opacity={0.22} />}
    </>
  );
}

function PotentialPlot({
  theme,
  prof,
  Lz,
  corrected,
  vacuum,
  p,
}: {
  theme: SceneTheme;
  prof: ReturnType<typeof slabProfile>;
  Lz: number;
  corrected: boolean;
  vacuum: number;
  p: number;
}) {
  const W = 520;
  const M = { l: 50, r: 14, t: 22 };
  const PH = 170;
  const P2 = 96;
  const GAP = 54;
  const H = M.t + PH + GAP + P2 + 40;
  const x = (z: number) => M.l + (z / (2 * Lz)) * (W - M.l - M.r);
  const step = potentialStep(Math.max(p, 1e-6));
  const all = [...prof.phiPer, ...prof.phiCor];
  const lo = Math.min(...all, -0.2) - 0.15 * step;
  const hi = Math.max(...all, 0.2) + 0.15 * step;
  const y = (v: number) => M.t + PH - ((v - lo) / (hi - lo)) * PH;
  const tile = (phi: number[], shiftSecond: number) =>
    [0, 1]
      .map((k) =>
        prof.z.map((z, i) => `${i ? 'L' : 'M'}${x(z + k * Lz).toFixed(1)},${y(phi[i]! + k * shiftSecond).toFixed(1)}`).join(''),
      )
      .join('');
  // Corrected potential is periodic up to the jump at the cell boundary.
  const per = tile(prof.phiPer, 0);
  const cor = tile(prof.phiCor, 0);
  const jumpX = x(Lz);

  // Finite-size panel: artificial field vs vacuum thickness.
  const top2 = M.t + PH + GAP;
  const vs = Array.from({ length: 55 }, (_, k) => 6 + k);
  const thick = Lz - vacuum;
  const fmax = vacuumField(Math.max(p, 1e-6), thick + 6) * 1.1;
  const xv = (v: number) => M.l + ((v - 6) / 54) * (W - M.l - M.r);
  const yv = (f: number) => top2 + P2 - (f / fmax) * P2;

  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Planar-averaged electrostatic potential over two cells. Periodic: sawtooth with a vacuum field of ${vacuumField(p, Lz).toFixed(4)} V/Å. Dipole-corrected: flat vacuum and a ${step.toFixed(2)} V step.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">planar-averaged potential φ̄(z) [V], two cells</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={PH} className="frame" />
        <line x1={x(Lz)} x2={x(Lz)} y1={M.t} y2={M.t + PH} className="grid" strokeDasharray="3 3" />
        <path d={per} className={`curve${corrected ? '' : ' is-active'}`} style={{ stroke: theme.fail, opacity: corrected ? 0.35 : 1 }} />
        <path d={cor} className={`curve${corrected ? ' is-active' : ''}`} style={{ stroke: theme.phys, opacity: corrected ? 1 : 0.35 }} />
        {corrected && (
          <line x1={jumpX} x2={jumpX} y1={y(prof.phiCor.at(-1)!)} y2={y(0)} style={{ stroke: theme.phys, strokeWidth: 2, strokeDasharray: '2 2' }} />
        )}
        {[0, Lz, 2 * Lz].map((z, k) => (
          <text key={k} x={x(z)} y={M.t + PH + 14} textAnchor="middle" className="tick">
            {k === 0 ? '0' : k === 1 ? 'L_z' : '2L_z'}
          </text>
        ))}
        <text x={M.l - 6} y={y(0) + 4} textAnchor="end" className="tick">0</text>
        <text x={M.l - 6} y={y(step) + 4} textAnchor="end" className="tick">{step.toFixed(1)}</text>

        <text x={M.l} y={top2 - 8} className="axis-label">artificial field vs vacuum thickness [V/Å]</text>
        <rect x={M.l} y={top2} width={W - M.l - M.r} height={P2} className="frame" />
        <path d={vs.map((v, k) => `${k ? 'L' : 'M'}${xv(v).toFixed(1)},${yv(vacuumField(p, thick + v)).toFixed(1)}`).join('')} className="curve" style={{ stroke: theme.fail }} />
        <circle cx={xv(vacuum)} cy={yv(vacuumField(p, Lz))} r={4.5} style={{ fill: theme.fail }} />
        {[6, 20, 40, 60].map((v) => (
          <text key={v} x={xv(v)} y={top2 + P2 + 14} textAnchor="middle" className="tick">{v} Å</text>
        ))}
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.fail }} /> periodic</span>
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> dipole-corrected</span>
      </figcaption>
    </figure>
  );
}
