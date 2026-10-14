import { useEffect, useId, useMemo, useState } from 'react';
import './InterfaceExplorer.css';
import { Stage, type StageView } from '~/three/Stage';
import {
  Arrows,
  AtomPicker,
  CellBox,
  ImpostorAtoms,
  Lines,
  Shell,
  Sticks,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
  type ShadingUniforms,
} from '~/three/primitives';
import { useSceneTheme, usePrefersReducedMotion, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { minImage, type Vec3 } from '~/lib/chem/structures';
import {
  IX_MAX_T,
  describeAtom,
  ixCouplings,
  ixDerive,
  ixModel,
  type IXMode,
} from '~/lib/scenes/interface-scene';
import { useLinkedSymbol } from '~/lib/useLinkedSymbol';

const MODES: Array<{ id: IXMode; label: string }> = [
  { id: 'local', label: 'Local cutoff' },
  { id: 'mp', label: 'Message passing' },
  { id: 'long', label: 'Long range' },
];

const VIEW: StageView = { position: [-21, -67, 16.5], target: [0, 0, 0.5] };

export default function InterfaceExplorer() {
  const model = ixModel();
  const s = model.structure;
  const theme = useSceneTheme();
  const reduced = usePrefersReducedMotion();
  const ids = useId();
  const [active, setSym] = useLinkedSymbol('interface-explorer');

  const [mode, setMode] = useState<IXMode>('mp');
  const [T, setT] = useState(2);
  const [rc, setRc] = useState(4);
  const [center, setCenter] = useState(model.naIndex);
  const [showWater, setShowWater] = useState(true);
  const [showDipoles, setShowDipoles] = useState(false);
  const [showImages, setShowImages] = useState(false);
  const [resetKey, setResetKey] = useState(0);

  const derived = useMemo(() => ixDerive({ mode, T, rc, center }), [mode, T, rc, center]);
  const layers = derived.layers;

  // Reveal the receptive field hop by hop when anything changes.
  const [reveal, setReveal] = useState(layers);
  useEffect(() => {
    if (reduced) {
      setReveal(layers);
      return;
    }
    setReveal(0);
    let k = 0;
    const id = window.setInterval(() => {
      k += 1;
      setReveal(k);
      if (k >= layers) window.clearInterval(id);
    }, 260);
    return () => window.clearInterval(id);
  }, [mode, T, rc, center, layers, reduced]);

  const couplings = useMemo(
    () => (mode === 'long' ? ixCouplings(center, derived.hops, layers) : []),
    [mode, center, derived.hops, layers],
  );

  const world = (p: Vec3): [number, number, number] => [p[0] - model.origin[0], p[1] - model.origin[1], p[2] - model.origin[2]];
  const ci = world(s.atoms[center]!.p);
  const isWater = (i: number) => s.atoms[i]!.mol >= 0;
  const visible = (i: number) => showWater || !isWater(i) || i === center;

  // ---------------------------------------------------------------- atoms
  const buffers = useMemo<AtomBuffers>(() => {
    const n = s.atoms.length;
    const images = showImages ? 8 : 0;
    const total = n * (1 + images);
    const positions = new Float32Array(total * 3);
    const radii = new Float32Array(total);
    const colors = new Float32Array(total * 3);
    const metal = new Float32Array(total);
    const ao = new Float32Array(total);
    const bg = linear(theme.canvas);
    // Faded atoms keep a trace of their element colour. On the light theme they fade
    // towards stone grey (fading towards the near-white page would make them vanish).
    const stone = linear(theme.dark ? theme.canvas : '#aeb6c2');
    const ghost = (c: [number, number, number]) => mixRGB(mixRGB(c, grey(c), 0.7), stone, theme.dark ? 0.5 : 0.55);
    const elColor = Object.fromEntries(Object.entries(ELEMENTS).map(([k, v]) => [k, linear(v.color)]));

    s.atoms.forEach((a, i) => {
      const p = world(a.p);
      positions.set(p, 3 * i);
      const h = derived.hops[i]!;
      const reached = i === center || (h > 0 && h <= Math.min(reveal, layers));
      radii[i] = visible(i) ? ELEMENTS[a.el].radius * (reached ? 1 : 0.8) : 0;
      const base = elColor[a.el]!;
      colors.set(reached ? base : ghost(base), 3 * i);
      metal[i] = ELEMENTS[a.el].metal * (reached ? 1 : 0.5);
      ao[i] = model.ao[i]!;
    });

    if (images) {
      let k = n;
      for (const dx of [-1, 0, 1])
        for (const dy of [-1, 0, 1]) {
          if (dx === 0 && dy === 0) continue;
          s.atoms.forEach((a, i) => {
            const p = world([a.p[0] + dx * s.cell[0], a.p[1] + dy * s.cell[1], a.p[2]]);
            positions.set(p, 3 * k);
            radii[k] = visible(i) ? ELEMENTS[a.el].radius * 0.8 : 0;
            colors.set(mixRGB(ghost(elColor[a.el]!), bg, 0.35), 3 * k);
            metal[k] = ELEMENTS[a.el].metal * 0.5;
            ao[k] = model.ao[i]!;
            k++;
          });
        }
    }
    return { positions, radii, colors, metal, ao, count: total };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, derived.hops, reveal, layers, center, showImages, showWater]);

  // ---------------------------------------------------------------- bonds and dipoles
  const sticks = useMemo<Segment[]>(() => {
    if (!showWater) return [];
    const out: Segment[] = [];
    const col = (i: number) => [buffers.colors[3 * i]!, buffers.colors[3 * i + 1]!, buffers.colors[3 * i + 2]!] as [number, number, number];
    for (const [a, b] of s.bonds) {
      const pa = world(s.atoms[a]!.p);
      const pb = world(s.atoms[b]!.p);
      const mid: [number, number, number] = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
      out.push({ a: pa, b: mid, color: col(a), radius: 0.13 });
      out.push({ a: mid, b: pb, color: col(b), radius: 0.13 });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buffers, showWater]);

  const dipoles = useMemo<Segment[]>(() => {
    if (!showDipoles || !showWater) return [];
    const phys = linear(theme.phys);
    const bg = linear(theme.canvas);
    const out: Segment[] = [];
    s.atoms.forEach((a, i) => {
      if (a.el !== 'O') return;
      const o = a.p;
      const h1 = minImage(o, s.atoms[i + 1]!.p, s.cell, s.pbc);
      const h2 = minImage(o, s.atoms[i + 2]!.p, s.cell, s.pbc);
      const u = [h1[0] + h2[0], h1[1] + h2[1], h1[2] + h2[2]];
      const l = Math.hypot(u[0]!, u[1]!, u[2]!);
      const start = world(o);
      const end: [number, number, number] = [start[0] + (u[0]! / l) * 1.75, start[1] + (u[1]! / l) * 1.75, start[2] + (u[2]! / l) * 1.75];
      const h = derived.hops[i]!;
      const reached = i === center || (h > 0 && h <= Math.min(reveal, layers));
      out.push({ a: start, b: end, color: reached ? phys : mixRGB(phys, bg, 0.75), radius: 0.075 });
    });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDipoles, showWater, theme, derived.hops, reveal, layers, center]);

  // ---------------------------------------------------------------- long-range web
  const lines = useMemo<ColoredLine[]>(() => {
    if (!couplings.length) return [];
    const wMax = couplings[0]!.weight;
    const att = linear(theme.arch);
    const rep = linear(theme.caveat);
    return couplings.map((c) => {
      const t = c.weight / wMax;
      const col = c.attractive ? att : rep;
      return { a: ci, b: world(c.to), rgba: [col[0], col[1], col[2], 0.12 + 0.7 * t] };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [couplings, theme, center]);

  const pick = (i: number | null) => {
    if (i !== null && i < s.atoms.length) setCenter(i);
  };

  const info = describeAtom(center);
  const reachNow = derived.reach[layers - 1]!;
  const ballNow = derived.ball[layers - 1]!;
  const hasCharge = s.atoms[center]!.q !== 0;

  return (
    <div className="ix" data-mode={mode}>
      <div className="ix-stage">
        <StageScene
          theme={theme}
          buffers={buffers}
          sticks={sticks}
          dipoles={dipoles}
          lines={lines}
          center={ci}
          rc={rc}
          T={layers}
          mode={mode}
          active={active}
          cell={s.cell}
          origin={model.origin}
          resetKey={resetKey}
          onPick={pick}
          pickCount={s.atoms.length}
          atomRadius={ELEMENTS[s.atoms[center]!.el].radius}
        />
        <div className="ix-toolbar">
          <div className="seg" role="group" aria-label="What can influence atom i">
            {MODES.map((m) => (
              <button key={m.id} type="button" aria-pressed={mode === m.id} onClick={() => setMode(m.id)}>
                {m.label}
              </button>
            ))}
          </div>
          {mode !== 'local' && (
            <div className="seg" role="group" aria-label="Message-passing layers T">
              <span className="seg-label" data-sym="T" aria-hidden="true">
                T
              </span>
              {Array.from({ length: IX_MAX_T }, (_, k) => k + 1).map((k) => (
                <button key={k} type="button" aria-pressed={T === k} onClick={() => setT(k)}>
                  {k}
                </button>
              ))}
            </div>
          )}
        </div>
        <button type="button" className="ix-reset" onClick={() => setResetKey((k) => k + 1)}>
          Reset view
        </button>
        <p className="ix-hint" aria-hidden="true">
          Drag to rotate · scroll or pinch to zoom · click an atom to make it atom <i>i</i>
        </p>
      </div>

      <div className="ix-panel">
        <div className="ix-readout" id={`${ids}-readout`} aria-live="polite">
          <p className="ix-atom">
            <span className="ix-dot" style={{ background: ELEMENTS[s.atoms[center]!.el].color }} aria-hidden="true" />
            Atom <i>i</i>: <strong>{info.name}</strong>, <span className="ix-q">q = {info.charge}</span>
          </p>
          {mode === 'local' && (
            <p>
              <strong className="num">{reachNow}</strong> atoms lie inside the cutoff sphere{' '}
              <span data-sym="rc">
                <i>r</i>
                <sub>c</sub>
              </span>{' '}
              = {rc.toFixed(1)} Å. A one-layer local model sees only these; every other atom is invisible to it.
            </p>
          )}
          {mode === 'mp' && (
            <p>
              After <span data-sym="T"><i>T</i></span> = {layers} layer{layers > 1 ? 's' : ''},{' '}
              <strong className="num">{reachNow}</strong> atoms can influence atom <i>i</i> (≤ {layers} graph hops). The
              sphere of radius <span data-sym="Trc"><i>T</i>·<i>r</i><sub>c</sub></span> holds{' '}
              <strong className="num">{ballNow}</strong>: distance alone over-counts by {ballNow - reachNow}.
            </p>
          )}
          {mode === 'long' && (
            <p>
              <strong className="num">{derived.beyond}</strong> atoms lie beyond the receptive field after {layers}{' '}
              layer{layers > 1 ? 's' : ''}.{' '}
              {hasCharge ? (
                <>
                  Coulomb’s law couples atom <i>i</i> to every charged one of them; the {lines.length} strongest
                  pairs are drawn (<span className="key att">attractive</span>, <span className="key rep">repulsive</span>,
                  opacity ∝ |q<sub>i</sub>q<sub>j</sub>|/r). Within each distant water the O and H lines nearly
                  cancel; what survives is the dipole field, ∝ 1/r².
                </>
              ) : (
                <>Au carries no charge in this toy, so it has no Coulomb partners here. Pick an ion or a water atom.</>
              )}
            </p>
          )}
        </div>

        <ReachChart reach={derived.reach} ball={derived.ball} layers={layers} total={s.atoms.length - 1} />

        <div className="ix-controls">
          <label htmlFor={`${ids}-rc`} data-sym="rc" onPointerEnter={() => setSym('rc')} onPointerLeave={() => setSym(null)}>
            <span>
              cutoff <i>r</i>
              <sub>c</sub>
            </span>
            <input
              id={`${ids}-rc`}
              type="range"
              min={3}
              max={6}
              step={0.25}
              value={rc}
              onChange={(e) => setRc(Number(e.target.value))}
            />
            <output>{rc.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-atom`}>
            <span>atom i</span>
            <select
              id={`${ids}-atom`}
              value={model.presets.find((p) => p.index === center)?.id ?? 'custom'}
              onChange={(e) => {
                const p = model.presets.find((x) => x.id === e.target.value);
                if (p) setCenter(p.index);
              }}
            >
              {model.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
              {!model.presets.some((p) => p.index === center) && <option value="custom">{info.name}</option>}
            </select>
          </label>
          <div className="ix-toggles" role="group" aria-label="Display">
            <label>
              <input type="checkbox" checked={showWater} onChange={(e) => setShowWater(e.target.checked)} /> Water
            </label>
            <label>
              <input type="checkbox" checked={showDipoles} onChange={(e) => setShowDipoles(e.target.checked)} /> Dipole
              arrows
            </label>
            <label>
              <input type="checkbox" checked={showImages} onChange={(e) => setShowImages(e.target.checked)} /> Periodic
              images
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

function StageScene(props: {
  theme: SceneTheme;
  buffers: AtomBuffers;
  sticks: Segment[];
  dipoles: Segment[];
  lines: ColoredLine[];
  center: [number, number, number];
  rc: number;
  T: number;
  mode: IXMode;
  active: string | null;
  cell: Vec3;
  origin: Vec3;
  resetKey: number;
  onPick: (i: number | null) => void;
  pickCount: number;
  atomRadius: number;
}) {
  const { theme } = props;
  return (
    <Stage
      view={VIEW}
      resetKey={props.resetKey}
      background={theme.canvas}
      label="Three-dimensional illustrative Au(111) slab under water with one Na⁺ and one Cl⁻. Atoms that can influence the selected atom are shown in full colour; the rest are faded."
    >
      <SceneContents {...props} />
    </Stage>
  );
}

function SceneContents(props: Parameters<typeof StageScene>[0]) {
  const { theme, buffers, sticks, dipoles, lines, center, rc, T, mode, active, cell, origin } = props;
  const uniforms: ShadingUniforms = useShadingUniforms(theme.canvas, theme.dark, 40, 120);
  const rcActive = active === 'rc';
  const trcActive = active === 'Trc' || active === 'T';
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Sticks segments={sticks} uniforms={uniforms} />
      {dipoles.length > 0 && <Arrows arrows={dipoles} uniforms={uniforms} />}
      {lines.length > 0 && <Lines lines={lines} />}
      <CellBox
        origin={[-origin[0], -origin[1], -origin[2] - 1.2]}
        size={[cell[0], cell[1], cell[2] - 10]}
        color={theme.ruleStrong}
        opacity={0.55}
      />
      <Shell center={center} radius={props.atomRadius * 1.6} color={theme.phys} opacity={0.9} />
      <Shell center={center} radius={rc} color={theme.phys} opacity={rcActive ? 0.95 : 0.6} />
      {mode !== 'local' && T > 1 && (
        <Shell center={center} radius={T * rc} color={theme.phys} opacity={trcActive ? 0.6 : 0.28} grid />
      )}
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={props.pickCount} onPick={props.onPick} />
    </>
  );
}

/** Perceptual grey of a linear RGB colour (Rec. 709 luminance). */
function grey(c: [number, number, number]): [number, number, number] {
  const y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  return [y, y, y];
}

function ReachChart({ reach, ball, layers, total }: { reach: number[]; ball: number[]; layers: number; total: number }) {
  const W = 300;
  const H = 150;
  const M = { l: 40, r: 8, t: 12, b: 28 };
  const bw = (W - M.l - M.r) / reach.length;
  const y = (v: number) => M.t + (H - M.t - M.b) * (1 - v / total);
  const ticks = [0, Math.round(total / 2), total];
  return (
    <figure className="ix-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Atoms that can influence atom i after T = 1 to 4 layers: ${reach.join(', ')}; atoms within T times r_c: ${ball.join(', ')}; of ${total}.`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 6} y={y(t) + 3} textAnchor="end" className="tick">
              {t}
            </text>
          </g>
        ))}
        {reach.map((r, k) => {
          const x0 = M.l + k * bw + bw * 0.18;
          const w = bw * 0.64;
          const current = k + 1 === layers;
          return (
            <g key={k} className={current ? 'is-current' : undefined}>
              <rect x={x0} y={y(ball[k]!)} width={w} height={y(0) - y(ball[k]!)} className="ball" data-sym="Trc" />
              <rect x={x0 + w * 0.18} y={y(r)} width={w * 0.64} height={y(0) - y(r)} className="reach" />
              <text x={x0 + w / 2} y={H - 10} textAnchor="middle" className="tick">
                T = {k + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        <span className="item">
          <span className="key reach" /> reachable in ≤ <i>T</i> hops
        </span>
        <span className="item" data-sym="Trc">
          <span className="key ball" /> within <i>T</i>·<i>r</i>
          <sub>c</sub>
        </span>
      </figcaption>
    </figure>
  );
}
