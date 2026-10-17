import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './SevenNet.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import type { Structure } from '~/lib/structure/types';
import { PAPER_BENCH, SEVENNET0, decompositionCost, domainEdge, forEachAtomInBox, overhang, shellCounts } from '~/lib/models/sevennet';

// One MPI rank's subdomain in α-quartz (the SevenNet paper's benchmark crystal), with the ghost
// atoms it must hold: those within r_c of its box for SevenNet, within T·r_c for the naive scheme
// without communication between layers. Counts are exact enumerations of the infinite crystal
// (src/lib/models/sevennet.ts); the 3D view shows a 5.4 Å slice through the middle of the box.

const SIZES = [250, 500, 1000, 2000, 4608, 10000, 20000, 50000, 100000];
const SLICE = 5.4;
const fmt = (x: number) => x.toLocaleString('en');
const big = (x: number) => (x >= 1e6 ? `${(x / 1e6).toFixed(1)} M` : x >= 1e4 ? `${(x / 1e3).toFixed(0)} k` : fmt(Math.round(x)));

export default function SevenNetDomains({ quartz }: { quartz: Structure }) {
  const theme = useSceneTheme();
  const ids = useId();
  const [si, setSi] = useState(4);
  const [T, setT] = useState<number>(SEVENNET0.layers);
  const [rc, setRc] = useState<number>(SEVENNET0.rc);
  const [scheme, setScheme] = useState<'sevennet' | 'naive'>('sevennet');
  const N = SIZES[si]!;
  const L = domainEdge(quartz, N);
  const counts = useMemo(() => shellCounts(quartz, L, rc, T), [quartz, L, rc, T]);
  const cost = decompositionCost(counts, T, SEVENNET0.featureFloats);
  const curve = useMemo(
    () =>
      SIZES.map((n) => {
        const c = shellCounts(quartz, domainEdge(quartz, n), rc, T);
        const k = decompositionCost(c, T, SEVENNET0.featureFloats);
        return { n, seven: k.ghostsSeven / k.owned, naive: k.ghostsNaive / k.owned, work: k.updatesNaive / k.updatesSeven };
      }),
    [quartz, rc, T],
  );
  const extent = L + 2 * T * rc;
  const view = useMemo(() => {
    // Look down on the slice from above and slightly in front, far enough to frame the deepest shell.
    const k = 1.35 * extent;
    return { position: [L / 2 + 0.1 * k, L / 2 - 0.55 * k, 1.05 * k] as [number, number, number], target: [L / 2, L / 2, 0] as [number, number, number] };
  }, [extent, L]);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={Math.round(extent)}
            background={theme.canvas}
            label={`A ${SLICE} Å slice through one rank's subdomain of α-quartz (${fmt(counts.owned)} atoms, edge ${L.toFixed(1)} Å). Owned atoms are blue, ghosts within r_c green, and the extra ghosts a ${T}-layer model would need without communication are brown.`}
            minDistance={10}
            maxDistance={600}
          >
            <Scene theme={theme} quartz={quartz} L={L} rc={rc} T={T} scheme={scheme} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Scheme">
              <button type="button" aria-pressed={scheme === 'sevennet'} onClick={() => setScheme('sevennet')}>
                SevenNet: r<sub>c</sub> shell
              </button>
              <button type="button" aria-pressed={scheme === 'naive'} onClick={() => setScheme('naive')}>
                naive: T·r<sub>c</sub> shell
              </button>
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.arch }} /> owned
            </li>
            <li>
              <span className="dot" style={{ background: theme.phys }} /> ghost within r<sub>c</sub>
            </li>
            <li>
              <span className="dot" style={{ background: theme.caveat }} /> ghost within T·r<sub>c</sub> {scheme === 'sevennet' ? <small>(not held)</small> : null}
            </li>
          </ul>
        </div>
        <div className="sn-side">
          <ShellPlot theme={theme} curve={curve} current={N} T={T} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-n`}>
            <span>atoms per rank</span>
            <input id={`${ids}-n`} type="range" min={0} max={SIZES.length - 1} step={1} value={si} onChange={(e) => setSi(Number(e.target.value))} />
            <output>{fmt(N)}</output>
          </label>
          <label htmlFor={`${ids}-t`}>
            <span>message-passing layers T</span>
            <input id={`${ids}-t`} type="range" min={1} max={5} step={1} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <output>{T}</output>
          </label>
          <div className="presets">
            <button type="button" className="lab-btn" aria-pressed={rc === SEVENNET0.rc} onClick={() => setRc(SEVENNET0.rc)}>
              r<sub>c</sub> = 5 Å (SevenNet-0)
            </button>{' '}
            <button type="button" className="lab-btn" aria-pressed={rc === PAPER_BENCH.rc} onClick={() => setRc(PAPER_BENCH.rc)}>
              r<sub>c</sub> = 4 Å (scaling benchmarks)
            </button>{' '}
            <button
              type="button"
              className="lab-btn"
              onClick={() => {
                setSi(SIZES.indexOf(PAPER_BENCH.atomsPerGpu));
                setRc(PAPER_BENCH.rc);
              }}
            >
              paper: 4,608 atoms per GPU
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>owned (edge {L.toFixed(1)} Å)</dt>
            <dd>{fmt(cost.owned)}</dd>
          </div>
          <div className="is-good">
            <dt>ghosts, SevenNet (r<sub>c</sub>)</dt>
            <dd>
              {fmt(cost.ghostsSeven)} <small>{(cost.ghostsSeven / cost.owned).toFixed(2)} per owned</small>
            </dd>
          </div>
          <div className={T > 1 ? 'is-alert' : undefined}>
            <dt>ghosts, naive ({(T * rc).toFixed(0)} Å)</dt>
            <dd>
              {fmt(cost.ghostsNaive)} <small>{(cost.ghostsNaive / cost.owned).toFixed(2)} per owned</small>
            </dd>
          </div>
          <div className={T > 1 ? 'is-alert' : undefined}>
            <dt>node updates per force call</dt>
            <dd>
              {big(cost.updatesSeven)} vs {big(cost.updatesNaive)} <small>naive ×{(cost.updatesNaive / cost.updatesSeven).toFixed(2)}</small>
            </dd>
          </div>
          <div>
            <dt>floats received, SevenNet</dt>
            <dd>
              {big(cost.commSeven.total)} <small>in {cost.commSeven.messages} exchanges</small>
            </dd>
          </div>
          <div>
            <dt>floats received, naive</dt>
            <dd>
              {big(cost.commNaive.total)} <small>in 2 exchanges</small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          A strictly local potential (T = 1) needs only the r<sub>c</sub> shell, and both schemes coincide. For a T-layer network the
          naive rank must hold every atom within T·r<sub>c</sub> and recompute layers for atoms that other ranks also compute, while
          SevenNet keeps the thin shell and updates only its own atoms. The price is traffic: each of the T − 1 feature exchanges sends
          {' '}{SEVENNET0.featureFloats} floats per ghost (SevenNet-0’s 128x0e + 64x1e + 32x2e), and the same again for gradients on the
          way back, so SevenNet moves more data in more, smaller steps. Ghost fractions shrink as domains grow, which is why strong
          scaling stalls once each GPU holds too few atoms (Park et al., Sec. 4.2).
        </p>
      </div>
    </div>
  );
}

function Scene({ theme, quartz, L, rc, T, scheme }: { theme: SceneTheme; quartz: Structure; L: number; rc: number; T: number; scheme: 'sevennet' | 'naive' }) {
  const D = T * rc;
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 1.6 * (L + 2 * D), 4 * (L + 2 * D));
  const buffers = useMemo<AtomBuffers>(() => {
    const pos: number[] = [];
    const rad: number[] = [];
    const col: number[] = [];
    const stone = linear(theme.dark ? '#3c4552' : '#c9ced6');
    const owned = linear(theme.arch);
    const ghost = linear(theme.phys);
    const far = linear(theme.caveat);
    forEachAtomInBox(quartz, [-D, -D, -SLICE / 2], [L + D, L + D, SLICE / 2], (p, k) => {
      const d = overhang([p[0], p[1], 0], L);
      const isO = quartz.symbols[k] === 'O';
      pos.push(p[0], p[1], p[2]);
      if (d <= 0) {
        rad.push(isO ? 0.55 : 0.75);
        col.push(...owned);
      } else if (d <= rc) {
        rad.push(isO ? 0.5 : 0.7);
        col.push(...ghost);
      } else {
        rad.push(isO ? 0.42 : 0.6);
        col.push(...(scheme === 'naive' ? mixRGB(far, stone, 0.15) : mixRGB(far, stone, 0.75)));
      }
    });
    const n = rad.length;
    return { positions: Float32Array.from(pos), radii: Float32Array.from(rad), colors: Float32Array.from(col), metal: new Float32Array(n).fill(0.3), ao: new Float32Array(n).fill(1), count: n };
  }, [quartz, L, rc, D, scheme, theme]);
  const z0 = -SLICE / 2;
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <CellBox origin={[0, 0, z0]} size={[L, L, SLICE]} color={theme.arch} opacity={1} />
      <CellBox origin={[-rc, -rc, z0]} size={[L + 2 * rc, L + 2 * rc, SLICE]} color={theme.phys} opacity={0.9} />
      {T > 1 ? <CellBox origin={[-D, -D, z0]} size={[L + 2 * D, L + 2 * D, SLICE]} color={theme.caveat} opacity={scheme === 'naive' ? 0.9 : 0.4} /> : null}
    </>
  );
}

/** Ghosts per owned atom against domain size, both schemes (log–log). */
function ShellPlot({ theme, curve, current, T }: { theme: SceneTheme; curve: Array<{ n: number; seven: number; naive: number; work: number }>; current: number; T: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 440;
  const H = 330;
  const m = { l: 50, r: 16, t: 16, b: 42 };
  const lx = (n: number) => m.l + ((Math.log10(n) - Math.log10(SIZES[0]!)) / (Math.log10(SIZES.at(-1)!) - Math.log10(SIZES[0]!))) * (W - m.l - m.r);
  const ymin = 0.1;
  const ymax = 100;
  const ly = (v: number) => m.t + (1 - (Math.log10(Math.max(v, ymin)) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin))) * (H - m.t - m.b);
  const path = (k: 'seven' | 'naive') => curve.map((p, i) => `${i ? 'L' : 'M'}${lx(p.n).toFixed(1)},${ly(p[k]).toFixed(1)}`).join('');
  const cur = curve.find((p) => p.n === current)!;
  const h = hover !== null ? curve[hover]! : null;
  return (
    <figure className="lab-plot sn-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Ghost atoms per owned atom against atoms per rank, log scales, for SevenNet's r_c shell and the naive ${T}-layer shell.`}>
        {[0.1, 1, 10, 100].map((v) => (
          <g key={v}>
            <line className="grid" x1={m.l} x2={W - m.r} y1={ly(v)} y2={ly(v)} />
            <text className="tick" x={m.l - 6} y={ly(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {[1e3, 1e4, 1e5].map((n) => (
          <text key={n} className="tick" x={lx(n)} y={H - m.b + 14} textAnchor="middle">
            {n >= 1e3 ? `${n / 1e3}k` : n}
          </text>
        ))}
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">
          atoms per rank
        </text>
        <text className="axis-label" x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
          ghosts per owned atom
        </text>
        <line className="now" x1={lx(current)} x2={lx(current)} y1={m.t} y2={H - m.b} style={{ stroke: theme.ruleStrong }} />
        <path d={path('naive')} fill="none" style={{ stroke: theme.caveat, strokeWidth: 2 }} />
        <path d={path('seven')} fill="none" style={{ stroke: theme.phys, strokeWidth: 2 }} />
        {(['naive', 'seven'] as const).map((k) => (
          <circle key={k} cx={lx(cur.n)} cy={ly(cur[k])} r={4.5} style={{ fill: k === 'seven' ? theme.phys : theme.caveat, stroke: theme.canvas, strokeWidth: 2 }} />
        ))}
        {T > 1 ? (
          <text x={lx(curve[1]!.n) + 4} y={ly(curve[1]!.naive) - 8} className="sn-direct">
            naive, {T}·r<tspan baselineShift="sub">c</tspan>
          </text>
        ) : null}
        <text x={lx(curve[1]!.n) + 4} y={ly(curve[1]!.seven) + 16} className="sn-direct">
          {T > 1 ? 'SevenNet, r' : 'both schemes, r'}
          <tspan baselineShift="sub">c</tspan>
        </text>
        {curve.map((p, i) => (
          <rect key={p.n} x={lx(p.n) - 12} y={m.t} width={24} height={H - m.t - m.b} fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
      </svg>
      <figcaption className="sn-tip" aria-live="polite">
        {h ? (
          <>
            {fmt(h.n)} atoms: {h.seven.toFixed(2)} vs {h.naive.toFixed(2)} ghosts per owned atom; naive does ×{h.work.toFixed(2)} the node updates.
          </>
        ) : (
          <>Hover a size for values. Same shells as the 3D view, {T} layer{T > 1 ? 's' : ''}.</>
        )}
      </figcaption>
    </figure>
  );
}
