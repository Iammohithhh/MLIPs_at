import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dimenet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, ShadedSurface, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment, type SurfaceData } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { PAPER_CUTOFF, besselRoots, envelope, sbfTilde, sphericalBessel, y0 } from '~/lib/models/dimenet';

// DimeNet's 2D spherical Fourier–Bessel basis (Eq. 6) times the envelope (Eq. 8), drawn as
// a relief over the plane of a triplet k → j → i. The basis depends on d_kj and on the angle
// α at j between the directions to k and to i, so it is axially symmetric about the j→i bond.

const C = PAPER_CUTOFF;
const D_JI = 1.5;
const NR = 44;
const NPHI = 120;
const HEIGHT = 1.6;

export default function DimeNetBasis() {
  const theme = useSceneTheme();
  const ids = useId();
  const [l, setL] = useState(2);
  const [n, setN] = useState(2);
  const [d, setD] = useState(2.4);
  const [alphaDeg, setAlpha] = useState(70);
  const alpha = (alphaDeg * Math.PI) / 180;
  const value = (r: number, a: number) => envelope(r / C) * sbfTilde(l, n, r, a, C);
  const peak = useMemo(() => {
    let m = 0;
    for (let a = 0; a <= 40; a++) for (let k = 1; k <= 60; k++) m = Math.max(m, Math.abs(envelope(k / 60) * sbfTilde(l, n, (k / 60) * C, (a / 40) * Math.PI, C)));
    return m || 1;
  }, [l, n]);
  const v = value(d, alpha);
  const z = besselRoots(l, n)[n - 1]!;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [-9, -15, 13], target: [0.6, 0, 0] }}
            background={theme.canvas}
            label={`Relief of the DimeNet basis function with l = ${l}, n = ${n} over the plane of a triplet: atom j at the centre, atom i ${D_JI} Å along the bond axis, atom k ${d.toFixed(2)} Å from j at ${alphaDeg}° from that axis. The basis value at k is ${(v / peak).toFixed(2)} of its peak.`}
            minDistance={8}
            maxDistance={60}
          >
            <BasisScene theme={theme} l={l} n={n} d={d} alpha={alpha} peak={peak} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> basis &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> basis &lt; 0
            </li>
            <li>disc radius = cutoff c = {C} Å</li>
          </ul>
        </div>
        <div className="lab-plot-stack">
          <RadialPlot theme={theme} l={l} n={n} z={z} d={d} />
          <AngularPlot theme={theme} l={l} alpha={alpha} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-l`}>
            <span>angular order l</span>
            <input id={`${ids}-l`} type="range" min={0} max={6} step={1} value={l} onChange={(e) => setL(Number(e.target.value))} />
            <output>{l}</output>
          </label>
          <label htmlFor={`${ids}-n`}>
            <span>radial index n</span>
            <input id={`${ids}-n`} type="range" min={1} max={6} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} />
            <output>{n}</output>
          </label>
          <label htmlFor={`${ids}-d`}>
            <span>
              distance d<sub>kj</sub>
            </span>
            <input id={`${ids}-d`} type="range" min={0.8} max={4.9} step={0.05} value={d} onChange={(e) => setD(Number(e.target.value))} />
            <output>{d.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-a`}>
            <span>angle α at j</span>
            <input id={`${ids}-a`} type="range" min={0} max={180} step={1} value={alphaDeg} onChange={(e) => setAlpha(Number(e.target.value))} />
            <output>{alphaDeg}°</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>
              root z<sub>
                {l}
                {n}
              </sub>{' '}
              of j<sub>{l}</sub>
            </dt>
            <dd>{z.toFixed(4)}</dd>
          </div>
          <div>
            <dt>value at k (fraction of peak)</dt>
            <dd>{(v / peak).toFixed(3)}</dd>
          </div>
          <div>
            <dt>basis size N_SHBF × N_SRBF</dt>
            <dd>7 × 6 = 42</dd>
          </div>
        </dl>
        <p className="lab-note">
          The relief is one of the 42 numbers the interaction block receives for every triplet. l sets how many times the
          function changes sign around j (the Legendre polynomial in cos α), n how many radial nodes it has before the
          cutoff, where every function vanishes because z<sub>ln</sub> is a root of j<sub>l</sub>. The envelope then flattens
          value, slope and curvature at c, so a neighbour crossing the cutoff changes nothing abruptly.
        </p>
      </div>
    </div>
  );
}

function BasisScene({ theme, l, n, d, alpha, peak }: { theme: SceneTheme; l: number; n: number; d: number; alpha: number; peak: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 70);
  const surface = useMemo<SurfaceData>(() => {
    const pos = new Float32Array((NR + 1) * (NPHI + 1) * 3);
    const col = new Float32Array((NR + 1) * (NPHI + 1) * 3);
    const pc = linear(theme.caveat);
    const nc = linear(theme.arch);
    const mid = linear(theme.surface);
    let k = 0;
    for (let a = 0; a <= NR; a++) {
      const r = (a / NR) * C;
      for (let b = 0; b <= NPHI; b++) {
        const phi = (b / NPHI) * 2 * Math.PI;
        const al = phi <= Math.PI ? phi : 2 * Math.PI - phi;
        const v = (envelope(r / C) * sbfTilde(l, n, r, al, C)) / peak;
        pos.set([r * Math.cos(phi), r * Math.sin(phi), HEIGHT * v], 3 * k);
        col.set(mixRGB(mid, v >= 0 ? pc : nc, Math.min(1, Math.abs(v) * 1.4)), 3 * k);
        k++;
      }
    }
    const idx: number[] = [];
    for (let a = 0; a < NR; a++)
      for (let b = 0; b < NPHI; b++) {
        const p = a * (NPHI + 1) + b;
        const q = p + NPHI + 1;
        idx.push(p, q, p + 1, p + 1, q, q + 1);
      }
    return { positions: pos, colors: col, indices: Uint32Array.from(idx) };
  }, [l, n, peak, theme]);
  const k: [number, number, number] = [d * Math.cos(alpha), d * Math.sin(alpha), 0];
  const kz = (HEIGHT * envelope(d / C) * sbfTilde(l, n, d, alpha, C)) / peak;
  const atoms = useMemo<AtomBuffers>(() => {
    const p = new Float32Array([0, 0, 0, D_JI, 0, 0, ...k]);
    const colors = new Float32Array(9);
    colors.set(linear(theme.ink), 0);
    colors.set(linear(theme.phys), 3);
    colors.set(linear(theme.fail), 6);
    return { positions: p, radii: new Float32Array([0.32, 0.28, 0.28]), colors, metal: new Float32Array(3), ao: new Float32Array(3).fill(1), count: 3 };
  }, [d, alpha, theme]);
  const bonds = useMemo<Segment[]>(() => {
    const c = mixRGB(linear(theme.ink), linear(theme.canvas), 0.3);
    return [
      { a: [0, 0, 0], b: [D_JI, 0, 0], color: c, radius: 0.06 },
      { a: [0, 0, 0], b: k, color: c, radius: 0.06 },
    ];
  }, [d, alpha, theme]);
  const guides = useMemo<ColoredLine[]>(() => {
    const ink = linear(theme.ink);
    const ring: ColoredLine[] = [];
    const rule = linear(theme.ruleStrong);
    for (let s = 0; s < 96; s++) {
      const t0 = (s / 96) * 2 * Math.PI;
      const t1 = ((s + 1) / 96) * 2 * Math.PI;
      ring.push({ a: [C * Math.cos(t0), C * Math.sin(t0), 0], b: [C * Math.cos(t1), C * Math.sin(t1), 0], rgba: [rule[0], rule[1], rule[2], 1] });
    }
    return [{ a: k, b: [k[0], k[1], kz], rgba: [ink[0], ink[1], ink[2], 1] }, { a: [-C, 0, 0], b: [C, 0, 0], rgba: [rule[0], rule[1], rule[2], 0.8] }, ...ring];
  }, [d, alpha, kz, theme]);
  return (
    <>
      <ShadedSurface data={surface} uniforms={uniforms} />
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Sticks segments={bonds} uniforms={uniforms} />
      <Lines lines={guides} overlay />
    </>
  );
}

function RadialPlot({ theme, l, n, z, d }: { theme: SceneTheme; l: number; n: number; z: number; d: number }) {
  const W = 380;
  const H = 150;
  const M = { l: 40, r: 10, t: 14, b: 26 };
  const xs = Array.from({ length: 121 }, (_, k) => (k / 120) * C);
  const raw = xs.map((r) => sphericalBessel(l, (z * r) / C));
  const env = xs.map((r, k) => envelope(r / C) * raw[k]!);
  const top = Math.max(...raw.map(Math.abs), 1e-9);
  const x = (v: number) => M.l + (v / C) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v / top + 1) / 2) * (H - M.t - M.b);
  const path = (ys: number[]) => ys.map((v, k) => `${k ? 'L' : 'M'}${x(xs[k]!).toFixed(1)},${y(v).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot dn-mini">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Radial part j_${l}(z_${l}${n} d / c) with and without the envelope, for d from 0 to ${C} Å.`}>
        <text x={M.l} y={M.t - 3} className="axis-label">
          radial part j<tspan dy="3" fontSize="8">{l}</tspan>
          <tspan dy="-3">(z d / c)</tspan>
        </text>
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="axis" />
        <path d={path(raw)} className="curve" style={{ stroke: theme.inkMuted, strokeDasharray: '4 3', strokeWidth: 1.4, fill: 'none' }} />
        <path d={path(env)} className="curve" style={{ stroke: theme.phys, strokeWidth: 2.4, fill: 'none' }} />
        <line x1={x(d)} x2={x(d)} y1={M.t} y2={H - M.b} style={{ stroke: theme.fail, strokeWidth: 1.4 }} />
        {[0, 1, 2, 3, 4, 5].map((v) => (
          <text key={v} x={x(v)} y={H - M.b + 13} textAnchor="middle" className="tick">
            {v}
          </text>
        ))}
        <text x={W - M.r} y={H - 2} textAnchor="end" className="axis-label">
          d (Å)
        </text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item">
          <span className="swatch" style={{ background: theme.inkMuted }} /> without envelope
        </span>
        <span className="item">
          <span className="swatch" style={{ background: theme.phys }} /> × u(d/c), p = 6
        </span>
      </figcaption>
    </figure>
  );
}

function AngularPlot({ theme, l, alpha }: { theme: SceneTheme; l: number; alpha: number }) {
  const W = 380;
  const H = 130;
  const M = { l: 40, r: 10, t: 14, b: 26 };
  const as = Array.from({ length: 121 }, (_, k) => (k / 120) * Math.PI);
  const ys = as.map((a) => y0(l, a));
  const top = Math.max(...ys.map(Math.abs));
  const x = (v: number) => M.l + (v / Math.PI) * (W - M.l - M.r);
  const y = (v: number) => M.t + (1 - (v / top + 1) / 2) * (H - M.t - M.b);
  return (
    <figure className="lab-plot dn-mini">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Angular part Y_${l}^0 of the angle at j, from 0 to 180 degrees.`}>
        <text x={M.l} y={M.t - 3} className="axis-label">
          angular part Y<tspan dy="3" fontSize="8">{l}</tspan>
          <tspan dy="-9" fontSize="8">0</tspan>
          <tspan dy="6">(α)</tspan>
        </text>
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="axis" />
        <path d={ys.map((v, k) => `${k ? 'L' : 'M'}${x(as[k]!).toFixed(1)},${y(v).toFixed(1)}`).join('')} className="curve" style={{ stroke: theme.arch, strokeWidth: 2.4, fill: 'none' }} />
        <line x1={x(alpha)} x2={x(alpha)} y1={M.t} y2={H - M.b} style={{ stroke: theme.fail, strokeWidth: 1.4 }} />
        {[0, 45, 90, 135, 180].map((v) => (
          <text key={v} x={x((v * Math.PI) / 180)} y={H - M.b + 13} textAnchor="middle" className="tick">
            {v}°
          </text>
        ))}
        <text x={W - M.r} y={H - 2} textAnchor="end" className="axis-label">
          α
        </text>
      </svg>
    </figure>
  );
}
