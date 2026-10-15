import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import slab from '~/data/structures/au111-slab.json';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Shell, TintBox, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS, COULOMB_K } from '~/lib/chem/elements';
import { polynomialEnvelope } from '~/lib/cutoffs';
import type { Vec3 } from '~/lib/structure/types';

// TOY MODEL: a cation above an ideal metal surface. The classical image attraction is
// E(z) = −k q²/(4z); a local model can only represent what lies inside its receptive
// field, sketched here as the same curve switched off smoothly at r_c.

const POS = slab.positions as Vec3[];
const zs = POS.map((p) => p[2]);
const TOP = Math.max(...zs);
const CELL = slab.cell as [Vec3, Vec3, Vec3];
const CX = (CELL[0][0] + CELL[1][0]) / 2;
const CY = (CELL[0][1] + CELL[1][1]) / 2;

const image = (z: number) => (-COULOMB_K * 1) / (4 * z);
const local = (z: number, rc: number) => image(z) * polynomialEnvelope(z, rc, 6).f;

export default function ImageChargeLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [z, setZ] = useState(6);
  const [rc, setRc] = useState(5);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [6, -34, 6], target: [0, 0, 1] }} background={theme.canvas} label={`A cation ${z.toFixed(1)} Å above a gold surface, with its mirror image charge below the surface.`} minDistance={12} maxDistance={90}>
            <ImageScene theme={theme} z={z} rc={rc} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: ELEMENTS.Na.color }} /> cation (+1 e)</li>
            <li><span className="dot" style={{ background: theme.caveat }} /> image charge (−1 e, mirrored)</li>
            <li><span className="ring" /> local model’s reach r_c</li>
          </ul>
        </div>
        <ImagePlot theme={theme} z={z} rc={rc} />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-z`}>
            <span>height above surface</span>
            <input id={`${ids}-z`} type="range" min={1.5} max={14} step={0.1} value={z} onChange={(e) => setZ(Number(e.target.value))} />
            <output>{z.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>local reach r_c</span>
            <input id={`${ids}-rc`} type="range" min={3} max={10} step={0.5} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>Image attraction</dt>
            <dd>{image(z).toFixed(3)} eV</dd>
          </div>
          <div className={z >= rc ? 'is-alert' : undefined}>
            <dt>Local model (toy)</dt>
            <dd>{local(z, rc).toFixed(3)} eV</dd>
          </div>
        </dl>
        <p className="lab-note">
          {z >= rc
            ? `Beyond its reach the local model predicts no attraction at all, while the image law still gives ${image(z).toFixed(2)} eV — the kind of qualitative error that aggregate energy and force errors can miss.`
            : 'Inside its reach a trained model can learn some of this attraction from its data; the toy shows the most it could represent, switched off smoothly at r_c.'}
        </p>
      </div>
    </div>
  );
}

function ImageScene({ theme, z, rc }: { theme: SceneTheme; z: number; rc: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 30, 80);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = POS.length + 2;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const au = linear(ELEMENTS.Au.color);
    POS.forEach((p, i) => {
      pos.set([p[0] - CX, p[1] - CY, p[2] - TOP], 3 * i);
      radii[i] = 1.05;
      colors.set(au, 3 * i);
      metal[i] = 1;
    });
    pos.set([0, 0, z], 3 * POS.length);
    radii[POS.length] = 0.95;
    colors.set(linear(ELEMENTS.Na.color), 3 * POS.length);
    pos.set([0, 0, -z], 3 * (POS.length + 1));
    radii[POS.length + 1] = 0.7;
    colors.set(mixRGB(linear(theme.caveat), linear(theme.canvas), 0.25), 3 * (POS.length + 1));
    return { positions: pos, radii, colors, metal, ao: new Float32Array(n).fill(1), count: n };
  }, [z, theme]);
  const link = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.caveat);
    return [{ a: [0, 0, z - 0.95], b: [0, 0, -z + 0.7], rgba: [c[0], c[1], c[2], 0.75] }];
  }, [z, theme.caveat]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={link} overlay />
      <TintBox center={[0, 0, 0.02]} size={[16, 14, 0.06]} color={theme.inkMuted} opacity={0.12} />
      <Shell center={[0, 0, z]} radius={rc} color={theme.phys} opacity={0.3} grid />
      <Shell center={[0, 0, -z]} radius={0.85} color={theme.caveat} opacity={0.95} overlay />
    </>
  );
}

function ImagePlot({ theme, z, rc }: { theme: SceneTheme; z: number; rc: number }) {
  const W = 520;
  const H = 320;
  const M = { l: 54, r: 16, t: 22, b: 40 };
  const zs2 = Array.from({ length: 200 }, (_, k) => 1.5 + (k * 12.5) / 199);
  const lo = image(1.5) * 1.05;
  const X = (v: number) => M.l + ((v - 1.5) / 12.5) * (W - M.l - M.r);
  const Y = (e: number) => M.t + (e / lo) * (H - M.t - M.b);
  const path = (f: (v: number) => number) => zs2.map((v, k) => `${k ? 'L' : 'M'}${X(v).toFixed(1)},${Y(f(v)).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Image-charge energy against height above the surface, and a local toy that switches it off at the cutoff.">
        <text x={M.l} y={M.t - 8} className="axis-label">energy of the ion [eV]</text>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="frame" />
        {[0, -1, -2].map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={Y(e)} y2={Y(e)} className={e === 0 ? 'axis' : 'grid'} />
            <text x={M.l - 6} y={Y(e) + 4} textAnchor="end" className="tick">{e}</text>
          </g>
        ))}
        <path d={path(image)} className="curve" style={{ stroke: theme.arch }} />
        <path d={path((v) => local(v, rc))} className="curve" style={{ stroke: theme.caveat, strokeDasharray: '6 3' }} />
        <line x1={X(rc)} x2={X(rc)} y1={M.t} y2={H - M.b} className="rc-line" />
        <line x1={X(z)} x2={X(z)} y1={M.t} y2={H - M.b} className="r-line" />
        <circle cx={X(z)} cy={Y(image(z))} r={4.5} style={{ fill: theme.arch }} />
        <circle cx={X(z)} cy={Y(local(z, rc))} r={4.5} style={{ fill: theme.caveat }} />
        {[2, 4, 6, 8, 10, 12, 14].map((v) => (
          <text key={v} x={X(v)} y={H - M.b + 16} textAnchor="middle" className="tick">{v}</text>
        ))}
        <text x={W - M.r} y={H - 6} textAnchor="end" className="axis-label">z [Å]</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.arch }} /> image law −kq²/(4z)</span>
        <span className="item"><span className="swatch" style={{ background: theme.caveat }} /> local toy, switched off at r_c</span>
      </figcaption>
    </figure>
  );
}
