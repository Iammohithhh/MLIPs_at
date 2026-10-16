import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import slab from '~/data/structures/au111-slab.json';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, TintBox, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { DEFAULT_DISCHARGE, discharge, levelDos, shiftPerElectron, type DischargeParams } from '~/lib/physics/newns-anderson';
import type { Vec3 } from '~/lib/structure/types';

// TOY MODEL (model Hamiltonian): a cation's level broadens and crosses the electrode's Fermi level.
// Constant potential holds E_F; constant charge lets E_F fall as the finite electrode gives up electrons.

const POS = slab.positions as Vec3[];
const TOP = Math.max(...POS.map((p) => p[2]));
const CELL = slab.cell as [Vec3, Vec3, Vec3];
const CX = (CELL[0][0] + CELL[1][0]) / 2;
const CY = (CELL[0][1] + CELL[1][1]) / 2;
const ZS = Array.from({ length: 161 }, (_, i) => 2 + (i * 8) / 160);

type Mode = 'potential' | 'charge';

export default function DischargeLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [z, setZ] = useState(4.5);
  const [phi, setPhi] = useState(-1.2);
  const [L, setL] = useState(10);
  const [mode, setMode] = useState<Mode>('charge');
  const p = useMemo<DischargeParams>(() => ({ ...DEFAULT_DISCHARGE, phi }), [phi]);
  const area = L * L;
  const st = discharge(z, p, mode, area);
  const cp = discharge(z, p, 'potential');
  const curves = useMemo(
    () => ({ pot: ZS.map((x) => discharge(x, p, 'potential').charge), chg: ZS.map((x) => discharge(x, p, 'charge', area).charge) }),
    [p, area],
  );

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [8, -30, 9], target: [0, 0, 2.5] }} background={theme.canvas} label={`An ion ${z.toFixed(1)} Å above a gold electrode holding a charge of ${st.charge.toFixed(2)} e.`} minDistance={10} maxDistance={80}>
            <DischargeScene theme={theme} z={z} charge={st.charge} L={L} deficit={mode === 'charge' ? st.n : 0} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Electrode condition">
              <button type="button" aria-pressed={mode === 'potential'} onClick={() => setMode('potential')}>Constant potential</button>
              <button type="button" aria-pressed={mode === 'charge'} onClick={() => setMode('charge')}>Constant charge (finite cell)</button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: ELEMENTS.Na.color }} /> ion, shaded by its charge</li>
            <li><span className="bar" style={{ background: theme.phys }} /> lateral cell, {L} × {L} Å</li>
          </ul>
        </div>
        <div className="lab-plot-stack">
          <LevelDiagram theme={theme} st={st} ef0={cp.ef} mode={mode} />
          <ChargePlot theme={theme} curves={curves} z={z} q={st.charge} mode={mode} L={L} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-z`}>
            <span>ion height above the image plane</span>
            <input id={`${ids}-z`} type="range" min={2} max={10} step={0.05} value={z} onChange={(e) => setZ(Number(e.target.value))} />
            <output>{z.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-phi`}>
            <span>electrode potential vs zero charge</span>
            <input id={`${ids}-phi`} type="range" min={-2} max={0.5} step={0.05} value={phi} onChange={(e) => setPhi(Number(e.target.value))} />
            <output>{phi.toFixed(2)} V</output>
          </label>
          <label htmlFor={`${ids}-L`}>
            <span>lateral cell size (constant charge)</span>
            <input id={`${ids}-L`} type="range" min={6} max={30} step={1} value={L} onChange={(e) => setL(Number(e.target.value))} />
            <output>{L} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Ion charge, {mode === 'charge' ? 'constant charge' : 'constant potential'}</dt>
            <dd>{st.charge.toFixed(2)} e</dd>
          </div>
          <div className={mode === 'charge' && Math.abs(st.charge - cp.charge) > 0.1 ? 'is-alert' : undefined}>
            <dt>Ion charge, potential held</dt>
            <dd>{cp.charge.toFixed(2)} e</dd>
          </div>
          <div>
            <dt>Fermi-level shift</dt>
            <dd>{(st.ef - cp.ef).toFixed(2)} eV</dd>
          </div>
          <div>
            <dt>Shift per electron, this cell</dt>
            <dd>{shiftPerElectron(area).toFixed(2)} eV</dd>
          </div>
        </dl>
        <p className="lab-note">
          Whether the ion is charged depends on where its level sits relative to the electrode’s Fermi level — a quantity an
          energy-and-force potential has no input for. In a small cell without a potentiostat each transferred electron
          moves the Fermi level by tenths of an eV, so the same geometry discharges differently than at constant potential.
        </p>
      </div>
    </div>
  );
}

function DischargeScene({ theme, z, charge, L, deficit }: { theme: SceneTheme; z: number; charge: number; L: number; deficit: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 25, 70);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = POS.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const au = linear(ELEMENTS.Au.color);
    POS.forEach((q, i) => {
      pos.set([q[0] - CX, q[1] - CY, q[2] - TOP], 3 * i);
      radii[i] = 1.05;
      colors.set(au, 3 * i);
      metal[i] = 1;
    });
    pos.set([0, 0, z], 3 * POS.length);
    radii[POS.length] = 0.95 + 0.25 * (1 - charge);
    colors.set(mixRGB(linear(theme.inkMuted), linear(ELEMENTS.Na.color), charge), 3 * POS.length);
    return { positions: pos, radii, colors, metal, ao: new Float32Array(n).fill(1), count: n };
  }, [z, charge, theme]);
  const box = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.phys);
    const h = L / 2;
    const corners: Array<[number, number]> = [[-h, -h], [h, -h], [h, h], [-h, h]];
    return corners.map((a, i) => {
      const b = corners[(i + 1) % 4]!;
      return { a: [a[0], a[1], 0.05], b: [b[0], b[1], 0.05], rgba: [c[0], c[1], c[2], 0.9] };
    });
  }, [L, theme.phys]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={box} overlay />
      {deficit > 0.02 && <TintBox center={[0, 0, -2.4]} size={[Math.min(L, 16), Math.min(L, 14), 4.6]} color={theme.caveat} opacity={0.08 + 0.25 * deficit} />}
      <Shell center={[0, 0, z]} radius={1.3 + 0.9 * charge} color={ELEMENTS.Na.color} opacity={0.25 * charge + 0.05} />
    </>
  );
}

function LevelDiagram({ theme, st, ef0, mode }: { theme: SceneTheme; st: ReturnType<typeof discharge>; ef0: number; mode: Mode }) {
  const W = 520;
  const H = 230;
  const M = { l: 56, r: 16, t: 22, b: 26 };
  const e0 = -8;
  const e1 = -2;
  const Y = (e: number) => M.t + ((e1 - e) / (e1 - e0)) * (H - M.t - M.b);
  const xm = M.l + 150;
  const xd = xm + 40;
  const dmax = 1 / (Math.PI * st.delta);
  const scale = Math.min(220, 220 / (dmax * 1.05)) * Math.min(1, dmax * 1.05);
  const pts = Array.from({ length: 241 }, (_, i) => e0 + ((e1 - e0) * i) / 240);
  const X = (e: number) => xd + (levelDos(e, st.eps, st.delta) / (dmax * 1.05)) * scale;
  const line = pts.map((e, i) => `${i ? 'L' : 'M'}${X(e).toFixed(1)},${Y(e).toFixed(1)}`).join('');
  const filled = pts.filter((e) => e <= st.ef);
  const area = filled.length > 1 ? `M${xd},${Y(filled[0]!)}` + filled.map((e) => `L${X(e).toFixed(1)},${Y(e).toFixed(1)}`).join('') + `L${xd},${Y(st.ef)}Z` : '';
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Energy levels: the ion's broadened level at ${st.eps.toFixed(2)} eV against a Fermi level at ${st.ef.toFixed(2)} eV.`}>
        <text x={M.l} y={M.t - 8} className="axis-label">energy vs vacuum [eV]</text>
        {[-8, -7, -6, -5, -4, -3, -2].map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={Y(e)} y2={Y(e)} className="grid" />
            <text x={M.l - 6} y={Y(e) + 4} textAnchor="end" className="tick">{e}</text>
          </g>
        ))}
        <rect x={M.l + 10} y={Y(st.ef)} width={xm - M.l - 20} height={Y(e0) - Y(st.ef)} style={{ fill: ELEMENTS.Au.color, opacity: 0.55 }} />
        <text x={(M.l + xm) / 2} y={Y(e0) - 8} textAnchor="middle" className="tick">metal (filled)</text>
        <line x1={M.l} x2={W - M.r} y1={Y(st.ef)} y2={Y(st.ef)} style={{ stroke: theme.caveat, strokeWidth: 2 }} />
        <text x={W - M.r - 4} y={Y(st.ef) - 5} textAnchor="end" className="tick" style={{ fill: theme.caveat }}>E_F{mode === 'charge' ? ' (shifted)' : ''}</text>
        {mode === 'charge' && Math.abs(st.ef - ef0) > 0.01 && (
          <line x1={M.l} x2={W - M.r} y1={Y(ef0)} y2={Y(ef0)} style={{ stroke: theme.caveat, strokeWidth: 1, strokeDasharray: '4 3', opacity: 0.7 }} />
        )}
        {area && <path d={area} style={{ fill: theme.phys, opacity: 0.35 }} />}
        <line x1={xd} x2={xd} y1={M.t} y2={H - M.b} className="axis" />
        <path d={line} className="curve" style={{ stroke: theme.phys, strokeWidth: 2 }} />
        <text x={xd + 6} y={Y(st.eps) - 6} className="tick" style={{ fill: theme.phys }}>ion level, {st.n.toFixed(2)} e filled</text>
      </svg>
    </figure>
  );
}

function ChargePlot({ theme, curves, z, q, mode, L }: { theme: SceneTheme; curves: { pot: number[]; chg: number[] }; z: number; q: number; mode: Mode; L: number }) {
  const W = 520;
  const H = 190;
  const M = { l: 56, r: 16, t: 22, b: 34 };
  const X = (x: number) => M.l + ((x - 2) / 8) * (W - M.l - M.r);
  const Y = (v: number) => M.t + (1 - v) * (H - M.t - M.b);
  const path = (v: number[]) => ZS.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(v[i]!).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Ion charge against height for constant potential and constant charge.">
        <text x={M.l} y={M.t - 8} className="axis-label">ion charge [e]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} className="grid" />
            <text x={M.l - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <path d={path(curves.pot)} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.2 }} />
        <path d={path(curves.chg)} className="curve" style={{ stroke: theme.caveat, strokeWidth: 2.2, strokeDasharray: '6 3' }} />
        <line x1={X(z)} x2={X(z)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={X(z)} cy={Y(q)} r={5} style={{ fill: mode === 'potential' ? theme.phys : theme.caveat }} />
        {[2, 4, 6, 8, 10].map((x) => (
          <text key={x} x={X(x)} y={H - M.b + 15} textAnchor="middle" className="tick">{x}</text>
        ))}
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">height z [Å]</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.phys }} /> constant potential</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> constant charge, {L} × {L} Å cell</span>
      </figcaption>
    </figure>
  );
}
