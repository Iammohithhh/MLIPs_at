import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  ImpostorAtoms,
  ShadedSurface,
  Shell,
  linear,
  useShadingUniforms,
  type AtomBuffers,
  type Segment,
  type SurfaceData,
} from '~/three/primitives';
import { useSceneTheme, usePrefersReducedMotion, type SceneTheme } from '~/three/theme';
import { addition, shBlock, type Vec3 } from '~/lib/physics/spherical-harmonics';
import { lobeMesh } from '~/lib/scenes/sh-surface';

// Physics Lab P5, part 1: the real spherical harmonic Y_lm as a 3D polar plot, and
// the l-th feature block Y_l(r̂) of a neighbour direction. Moving the neighbour mixes
// the 2l+1 components; their sum of squares never changes.

const NEIGHBOUR_R = 2.6;

export default function HarmonicsLab() {
  const theme = useSceneTheme();
  const reduced = usePrefersReducedMotion();
  const ids = useId();
  const [l, setL] = useState(2);
  const [m, setM] = useState(0);
  const [polar, setPolar] = useState(52);
  const [azimuth, setAzimuth] = useState(35);
  const [spinning, setSpinning] = useState(false);

  useEffect(() => {
    if (!spinning || reduced) return;
    let raf = 0;
    let last = performance.now();
    const step = (t: number) => {
      setAzimuth((a) => (a + (t - last) * 0.05) % 360);
      last = t;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [spinning, reduced]);

  const mm = Math.max(-l, Math.min(l, m));
  const dir: Vec3 = useMemo(() => {
    const th = (polar * Math.PI) / 180;
    const ph = (azimuth * Math.PI) / 180;
    return [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
  }, [polar, azimuth]);
  const block = shBlock(l, dir);
  const norm2 = block.reduce((s, y) => s + y * y, 0);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [5.2, -7.8, 4.6], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`Polar plot of the real spherical harmonic with l = ${l}, m = ${mm}; positive lobes teal, negative lobes amber. A neighbour atom sits in direction polar ${polar.toFixed(0)}°, azimuth ${azimuth.toFixed(0)}°.`}
            minDistance={4}
            maxDistance={30}
          >
            <LobeScene theme={theme} l={l} m={mm} dir={dir} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Degree l">
              <span className="seg-label">l</span>
              {[0, 1, 2, 3].map((k) => (
                <button key={k} type="button" aria-pressed={l === k} onClick={() => { setL(k); setM((v) => Math.max(-k, Math.min(k, v))); }}>
                  {k}
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Order m">
              <span className="seg-label">m</span>
              {Array.from({ length: 2 * l + 1 }, (_, k) => k - l).map((k) => (
                <button key={k} type="button" aria-pressed={mm === k} onClick={() => setM(k)}>
                  {k > 0 ? `+${k}` : k}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.phys }} /> Y &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> Y &lt; 0
            </li>
            <li>
              <span className="bar" style={{ background: theme.arch }} /> neighbour direction r̂
            </li>
          </ul>
        </div>
        <BlockChart l={l} m={mm} block={block} norm2={norm2} />
      </div>

      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-pol`}>
            <span>neighbour polar</span>
            <input id={`${ids}-pol`} type="range" min={0} max={180} step={1} value={Math.round(polar)} onChange={(e) => setPolar(Number(e.target.value))} />
            <output>{polar.toFixed(0)}°</output>
          </label>
          <label htmlFor={`${ids}-az`}>
            <span>neighbour azimuth</span>
            <input
              id={`${ids}-az`}
              type="range"
              min={0}
              max={360}
              step={1}
              value={Math.round(azimuth)}
              onChange={(e) => {
                setSpinning(false);
                setAzimuth(Number(e.target.value));
              }}
            />
            <output>{azimuth.toFixed(0)}°</output>
          </label>
          {!reduced && (
            <button type="button" className="lab-btn" aria-pressed={spinning} onClick={() => setSpinning((s) => !s)}>
              {spinning ? 'Stop' : 'Orbit the neighbour'}
            </button>
          )}
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt className="math">
              Y<sub>{l},{mm}</sub>(r̂)
            </dt>
            <dd>{block[mm + l]!.toFixed(4)}</dd>
          </div>
          <div className="is-good">
            <dt className="math">Σₘ Yₗₘ(r̂)²</dt>
            <dd>{norm2.toFixed(6)}</dd>
          </div>
          <div>
            <dt className="math">(2l+1)/4π</dt>
            <dd>{addition(l).toFixed(6)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Move the neighbour: the {2 * l + 1} components of the l = {l} block change and trade weight with each other,
          but their sum of squares stays at (2l+1)/4π. A rotation acts inside each block and never moves weight from one
          l to another — which is why equivariant networks keep their features organised by l.
        </p>
      </div>
    </div>
  );
}

function LobeScene({ theme, l, m, dir }: { theme: SceneTheme; l: number; m: number; dir: Vec3 }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 8, 26);
  const mesh = useMemo(() => lobeMesh(l, m), [l, m]);
  const surface = useMemo<SurfaceData>(() => {
    const pos = linear(theme.phys);
    const neg = linear(theme.caveat);
    const colors = new Float32Array(mesh.signs.length * 3);
    mesh.signs.forEach((s, k) => colors.set(s > 0 ? pos : neg, 3 * k));
    return { positions: mesh.positions, colors, indices: mesh.indices };
  }, [mesh, theme.phys, theme.caveat]);

  const nb: [number, number, number] = [dir[0] * NEIGHBOUR_R, dir[1] * NEIGHBOUR_R, dir[2] * NEIGHBOUR_R];
  const atoms = useMemo<AtomBuffers>(() => {
    const ink = linear(theme.inkMuted);
    return {
      positions: new Float32Array([0, 0, 0, ...nb]),
      radii: new Float32Array([0.22, 0.32]),
      colors: new Float32Array([...ink, ...linear(theme.arch)]),
      metal: new Float32Array(2),
      ao: new Float32Array([1, 1]),
      count: 2,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nb[0], nb[1], nb[2], theme.inkMuted, theme.arch]);
  const arrow = useMemo<Segment[]>(
    () => [{ a: [0, 0, 0], b: [dir[0] * (NEIGHBOUR_R - 0.34), dir[1] * (NEIGHBOUR_R - 0.34), dir[2] * (NEIGHBOUR_R - 0.34)], color: linear(theme.arch), radius: 0.035 }],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dir[0], dir[1], dir[2], theme.arch],
  );
  return (
    <>
      <ShadedSurface data={surface} uniforms={uniforms} />
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Arrows arrows={arrow} uniforms={uniforms} head={0.25} />
      <Shell center={[0, 0, 0]} radius={NEIGHBOUR_R} color={theme.ruleStrong} opacity={0.35} grid />
    </>
  );
}

function BlockChart({ l, m, block, norm2 }: { l: number; m: number; block: number[]; norm2: number }) {
  const W = 520;
  const H = 300;
  const M = { l: 48, r: 14, t: 26, b: 44 };
  const lim = Math.sqrt(addition(3)) * 1.05; // fixed scale across l so blocks compare
  const n = block.length;
  const bw = (W - M.l - M.r) / n;
  const y = (v: number) => M.t + ((lim - v) / (2 * lim)) * (H - M.t - M.b);
  const target = addition(l);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`The ${n} components of the l = ${l} block for the current neighbour direction: ${block.map((v) => v.toFixed(3)).join(', ')}. Sum of squares ${norm2.toFixed(4)}.`}>
        <text x={M.l} y={M.t - 10} className="axis-label">
          feature block Yₗ(r̂), l = {l}
        </text>
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="axis" />
        {[-0.6, -0.3, 0.3, 0.6].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end" className="tick">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {block.map((v, k) => {
          const mk = k - l;
          const x0 = M.l + k * bw + bw * 0.2;
          const w = bw * 0.6;
          return (
            <g key={k}>
              <rect
                x={x0}
                y={Math.min(y(v), y(0))}
                width={w}
                height={Math.abs(y(v) - y(0))}
                className={`bar ${v >= 0 ? 'pos' : 'neg'}${mk === m ? ' is-active' : ''}`}
              />
              <text x={x0 + w / 2} y={H - M.b + 18} textAnchor="middle" className={`tick${mk === m ? ' strong' : ''}`}>
                m = {mk > 0 ? `+${mk}` : mk}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="norm-meter" aria-hidden="true">
        <span className="norm-label">Σ Y² / [(2l+1)/4π]</span>
        <span className="norm-track">
          <span className="norm-fill" style={{ width: `${Math.min(100, (norm2 / target) * 100)}%` }} />
        </span>
        <span className="norm-val">{(norm2 / target).toFixed(6)}</span>
      </div>
    </figure>
  );
}
