import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { minImage } from '~/lib/chem/structures';
import { interfaceProfile } from '~/lib/scenes/interface-potential';
import { ixModel } from '~/lib/scenes/interface-scene';

// TOY MODEL: tilt the dipoles of the first water layer on Au(111) and watch the
// planar-averaged charge density and electrostatic potential respond.

export default function WaterDipoleLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [theta, setTheta] = useState(90);
  const prof = useMemo(() => interfaceProfile(theta), [theta]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [-8, -44, 5], target: [0, 0, -0.5] }} background={theme.canvas} label={`Au(111)/water interface with the first water layer's dipoles tilted ${theta}° from the surface normal.`} minDistance={15} maxDistance={110}>
            <DipoleScene theme={theme} prof={prof} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="bar" style={{ background: theme.phys }} /> first-layer dipoles</li>
            <li>θ = 0°: H up · 90°: flat · 180°: H down</li>
          </ul>
        </div>
        <ProfilePlot theme={theme} prof={prof} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-t`}>
            <span>first-layer tilt θ</span>
            <input id={`${ids}-t`} type="range" min={0} max={180} step={1} value={theta} onChange={(e) => setTheta(Number(e.target.value))} />
            <output>{theta}°</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Potential step across the film</dt>
            <dd>{prof.drop.toFixed(2)} V</dd>
          </div>
          <div>
            <dt>First water layer</dt>
            <dd>{prof.parts.firstLayer.toFixed(2)} V</dd>
          </div>
          <div>
            <dt>Ion pair</dt>
            <dd>{prof.parts.ions.toFixed(2)} V</dd>
          </div>
          <div>
            <dt>Rest of the water</dt>
            <dd>{prof.parts.otherWater.toFixed(2)} V</dd>
          </div>
        </dl>
        <p className="lab-note">
          Each neutral group shifts the potential by its dipole per area over ε₀. A fully aligned first layer moves it by
          volts — far more than at real interfaces, where orientations are only partly ordered and the metal’s electrons
          screen and polarise, which this toy omits. The direction of the effect, not its size, is the lesson.
        </p>
      </div>
    </div>
  );
}

function DipoleScene({ theme, prof }: { theme: SceneTheme; prof: ReturnType<typeof interfaceProfile> }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 40, 120);
  const model = ixModel();
  const s = model.structure;
  const O = model.origin;
  const first = useMemo(() => new Set(prof.firstLayer.flatMap((i) => [i, i + 1, i + 2])), [prof.firstLayer]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = prof.atoms.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const stone = linear(theme.dark ? '#4a5568' : '#c3c9d2');
    prof.atoms.forEach((a, i) => {
      pos.set([a.p[0] - O[0], a.p[1] - O[1], a.p[2] - O[2]], 3 * i);
      const base = linear(ELEMENTS[a.el].color);
      const hot = a.mol < 0 || first.has(i);
      radii[i] = ELEMENTS[a.el].radius * (hot || a.el === 'Au' ? 1 : 0.6);
      colors.set(hot || a.el === 'Au' ? base : mixRGB(base, stone, 0.75), 3 * i);
      metal[i] = ELEMENTS[a.el].metal;
    });
    return { positions: pos, radii, colors, metal, ao: model.ao, count: n };
  }, [prof, first, theme, O, model.ao]);
  const sticks = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    for (const [a, b] of s.bonds) {
      const pa = prof.atoms[a]!.p;
      const pb = prof.atoms[b]!.p;
      const A: [number, number, number] = [pa[0] - O[0], pa[1] - O[1], pa[2] - O[2]];
      const B: [number, number, number] = [pb[0] - O[0], pb[1] - O[1], pb[2] - O[2]];
      const c: [number, number, number] = [buffers.colors[3 * b]!, buffers.colors[3 * b + 1]!, buffers.colors[3 * b + 2]!];
      out.push({ a: A, b: B, color: c, radius: first.has(a) ? 0.13 : 0.09 });
    }
    return out;
  }, [s.bonds, prof.atoms, O, buffers, first]);
  const arrows = useMemo<Segment[]>(() => {
    const col = linear(theme.phys);
    return prof.firstLayer.map((iO) => {
      const o = prof.atoms[iO]!.p;
      const h1 = minImage(o, prof.atoms[iO + 1]!.p, s.cell, s.pbc);
      const h2 = minImage(o, prof.atoms[iO + 2]!.p, s.cell, s.pbc);
      const u = [h1[0] + h2[0], h1[1] + h2[1], h1[2] + h2[2]];
      const l = Math.hypot(u[0]!, u[1]!, u[2]!) || 1;
      const st: [number, number, number] = [o[0] - O[0], o[1] - O[1], o[2] - O[2]];
      return { a: st, b: [st[0] + (u[0]! / l) * 2.4, st[1] + (u[1]! / l) * 2.4, st[2] + (u[2]! / l) * 2.4] as [number, number, number], color: col, radius: 0.15 };
    });
  }, [prof, s.cell, s.pbc, O, theme.phys]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      <Arrows arrows={arrows} uniforms={uniforms} head={0.6} />
    </>
  );
}

function ProfilePlot({ theme, prof }: { theme: SceneTheme; prof: ReturnType<typeof interfaceProfile> }) {
  const W = 520;
  const PH = 140;
  const GAP = 46;
  const M = { l: 56, r: 14, t: 22, b: 40 };
  const H = M.t + 2 * PH + GAP + M.b;
  const z0 = prof.z[0]!;
  const z1 = prof.z.at(-1)!;
  const X = (z: number) => M.l + ((z - z0) / (z1 - z0)) * (W - M.l - M.r);
  const rmax = Math.max(...prof.rho.map(Math.abs)) * 1.1 || 1;
  const Yr = (v: number) => M.t + PH / 2 - (v / rmax) * (PH / 2);
  const top2 = M.t + PH + GAP;
  const plo = Math.min(-6.5, ...prof.phi);
  const phi = Math.max(2.5, ...prof.phi);
  const Yp = (v: number) => top2 + (1 - (v - plo) / (phi - plo)) * PH;
  const line = (vals: number[], Y: (v: number) => number) => prof.z.map((z, i) => `${i ? 'L' : 'M'}${X(z).toFixed(1)},${Y(vals[i]!).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Planar-averaged charge density and electrostatic potential across the water film; the potential step is ${prof.drop.toFixed(2)} V.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">planar-averaged charge density ρ̄(z) [e/Å³]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={PH} className="frame" />
        <rect x={X(z0)} y={M.t} width={X(z0 + 3.6) - X(z0)} height={PH} style={{ fill: theme.phys, opacity: 0.1 }} />
        <rect x={X(z0)} y={top2} width={X(z0 + 3.6) - X(z0)} height={PH} style={{ fill: theme.phys, opacity: 0.1 }} />
        <text x={X(z0 + 1.8)} y={M.t + 13} textAnchor="middle" className="tick" style={{ fill: theme.phys }}>1st layer</text>
        <line x1={M.l} x2={W - M.r} y1={Yr(0)} y2={Yr(0)} className="axis" />
        <path d={line(prof.rho, Yr)} className="curve" style={{ stroke: theme.lit, strokeWidth: 1.4 }} />
        <text x={M.l} y={top2 - 8} className="axis-label">electrostatic potential φ̄(z) [V], zero at the metal surface</text>
        <rect x={M.l} y={top2} width={W - M.l - M.r} height={PH} className="frame" />
        {[-6, -4, -2, 0, 2].filter((v) => v >= plo && v <= phi).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={Yp(v)} y2={Yp(v)} className={v === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 6} y={Yp(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <path d={line(prof.phi, Yp)} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.4 }} />
        <text x={W - M.r - 6} y={Yp(prof.drop) + (prof.drop > (plo + phi) / 2 ? 16 : -8)} textAnchor="end" className="tick" style={{ fill: theme.phys }}>
          Δφ̄ = {prof.drop.toFixed(2)} V
        </text>
        {[0, 5, 10, 15, 20].map((d) => (
          <text key={d} x={X(z0 + d)} y={H - M.b + 16} textAnchor="middle" className="tick">{d}</text>
        ))}
        <text x={W - M.r} y={H - 8} textAnchor="end" className="axis-label">height above the Au surface [Å]</text>
      </svg>
    </figure>
  );
}
