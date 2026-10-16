import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, TintBox, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS } from '~/lib/chem/elements';
import { fcc111 } from '~/lib/physics/cleavage';
import { atomCount, concentrationPerPair, potentialPerElectron, relativeCost, sideForConcentration, sideForPotential, sigmaPerElectron } from '~/lib/physics/finite-size';

// THEORY: what one ion pair and one electron mean in an interface cell of lateral size L.

const AU = 4.078;
const REF_L = 10;

export default function SmallCellLab() {
  const theme = useSceneTheme();
  const ids = useId();
  const [L, setL] = useState(12);
  const [h, setH] = useState(20);
  const [logC, setLogC] = useState(-1); // target concentration 10^logC M
  const [dU, setDU] = useState(0.1);
  const cTarget = 10 ** logC;
  const Lc = sideForConcentration(cTarget, h);
  const Lu = sideForPotential(dU);
  const Lneed = Math.max(Lc, Lu);
  const nNow = atomCount(L, h).total;
  const nRef = atomCount(REF_L, h).total;
  const nNeed = atomCount(Lneed, h).total;
  const fmt = (x: number) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toPrecision(2));

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage view={{ position: [42, -70, 46], target: [0, 0, 6] }} background={theme.canvas} label={`An Au(111)/water cell ${L} Å wide with one ion pair, and its periodic images.`} minDistance={20} maxDistance={260}>
            <CellScene theme={theme} L={L} h={h} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: ELEMENTS.Na.color }} /> <span className="dot" style={{ background: ELEMENTS.Cl.color }} /> one ion pair per cell, repeated by periodicity</li>
            <li><span className="bar" style={{ background: theme.phys }} /> the simulated cell</li>
          </ul>
        </div>
        <div className="lab-plot-stack">
          <Multiples theme={theme} L={L} h={h} cTarget={cTarget} dU={dU} Lc={Lc} Lu={Lu} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-L`}>
            <span>cell side L</span>
            <input id={`${ids}-L`} type="range" min={6} max={40} step={0.5} value={L} onChange={(e) => setL(Number(e.target.value))} />
            <output>{L.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-h`}>
            <span>electrolyte thickness h</span>
            <input id={`${ids}-h`} type="range" min={10} max={40} step={1} value={h} onChange={(e) => setH(Number(e.target.value))} />
            <output>{h} Å</output>
          </label>
          <label htmlFor={`${ids}-c`}>
            <span>concentration you want to model</span>
            <input id={`${ids}-c`} type="range" min={-2} max={0.3} step={0.05} value={logC} onChange={(e) => setLogC(Number(e.target.value))} />
            <output>{fmt(cTarget)} M</output>
          </label>
          <label htmlFor={`${ids}-u`}>
            <span>potential resolution you need</span>
            <input id={`${ids}-u`} type="range" min={0.02} max={0.5} step={0.01} value={dU} onChange={(e) => setDU(Number(e.target.value))} />
            <output>{dU.toFixed(2)} V</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>One ion pair in this cell</dt>
            <dd>{fmt(concentrationPerPair(L, h))} M</dd>
          </div>
          <div>
            <dt>One electron on the electrode</dt>
            <dd>{fmt(sigmaPerElectron(L))} μC/cm²</dd>
          </div>
          <div>
            <dt>Potential step per electron</dt>
            <dd>{fmt(potentialPerElectron(L))} V</dd>
          </div>
          <div className={Lneed > L ? 'is-alert' : undefined}>
            <dt>Cell your targets need</dt>
            <dd>
              {fmt(Lneed)} Å · {nNeed.toLocaleString('en-US')} atoms · ×{fmt(relativeCost(nNeed, nRef))} cost
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          This cell has {nNow.toLocaleString('en-US')} atoms. Whole ions and whole electrons come in steps whose size is set by
          the cell area: halving a step doubles the area and the atoms and, at cubic scaling, costs about eight times the
          DFT time. Training data from small cells sample a world of high concentration, large surface charge and coarse potential
          steps — which is why transfer to large cells is an open question rather than a given.
        </p>
      </div>
    </div>
  );
}

function CellScene({ theme, L, h }: { theme: SceneTheme; L: number; h: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 40, 260);
  const s = useMemo(() => {
    const as = AU / Math.SQRT2;
    const nx = Math.max(2, Math.round(L / as));
    const ny = Math.max(1, Math.round(L / (as * Math.sqrt(3))));
    return fcc111(AU, nx, ny, 4);
  }, [L]);
  const [Lx, Ly] = s.cell;
  const top = 3 * s.dz;
  const buffers = useMemo<AtomBuffers>(() => {
    const ions: Array<{ p: [number, number, number]; c: string; r: number }> = [];
    for (let u = -1; u <= 1; u++)
      for (let v = -1; v <= 1; v++) {
        const main = u === 0 && v === 0;
        ions.push({ p: [u * Lx - Lx * 0.2, v * Ly - Ly * 0.15, top + 4], c: ELEMENTS.Na.color, r: main ? 1.1 : 0.7 });
        ions.push({ p: [u * Lx + Lx * 0.22, v * Ly + Ly * 0.2, top + h * 0.6], c: ELEMENTS.Cl.color, r: main ? 1.3 : 0.85 });
      }
    const nm = s.atoms.length * 9;
    const n = nm + ions.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const au = linear(ELEMENTS.Au.color);
    const auImg = mixRGB(au, linear(theme.canvas), 0.55);
    let i = 0;
    for (let u = -1; u <= 1; u++)
      for (let v = -1; v <= 1; v++)
        for (const a of s.atoms) {
          pos.set([a.p[0] - Lx / 2 + u * Lx, a.p[1] - Ly / 2 + v * Ly, a.p[2]], 3 * i);
          radii[i] = 1.2;
          colors.set(u === 0 && v === 0 ? au : auImg, 3 * i);
          metal[i] = u === 0 && v === 0 ? 1 : 0.3;
          i++;
        }
    ions.forEach((ion, k) => {
      const i = nm + k;
      pos.set(ion.p, 3 * i);
      radii[i] = ion.r;
      colors.set(linear(ion.c), 3 * i);
    });
    return { positions: pos, radii, colors, metal, ao: new Float32Array(n).fill(1), count: n };
  }, [s, Lx, Ly, top, h, theme.canvas]);
  const lines = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.phys);
    const g = linear(theme.inkMuted);
    const out: ColoredLine[] = [];
    const box = (x0: number, y0: number, col: [number, number, number], a: number) => {
      const z0 = top + 0.5;
      const z1 = top + h;
      const cs: Array<[number, number]> = [[x0, y0], [x0 + Lx, y0], [x0 + Lx, y0 + Ly], [x0, y0 + Ly]];
      cs.forEach((p, i) => {
        const q = cs[(i + 1) % 4]!;
        out.push({ a: [p[0], p[1], z0], b: [q[0], q[1], z0], rgba: [...col, a] });
        out.push({ a: [p[0], p[1], z1], b: [q[0], q[1], z1], rgba: [...col, a] });
        out.push({ a: [p[0], p[1], z0], b: [p[0], p[1], z1], rgba: [...col, a] });
      });
    };
    for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) if (u || v) box(u * Lx - Lx / 2, v * Ly - Ly / 2, g, 0.25);
    box(-Lx / 2, -Ly / 2, c, 1);
    return out;
  }, [Lx, Ly, top, h, theme.phys, theme.inkMuted]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <TintBox center={[0, 0, top + 0.5 + h / 2]} size={[3 * Lx, 3 * Ly, h]} color={theme.lit} opacity={0.06} />
      <Lines lines={lines} />
    </>
  );
}

function Multiples({ theme, L, h, cTarget, dU, Lc, Lu }: { theme: SceneTheme; L: number; h: number; cTarget: number; dU: number; Lc: number; Lu: number }) {
  const W = 520;
  const PH = 96;
  const M = { l: 58, r: 14, t: 18, b: 6 };
  const GAP = 30;
  const H = M.t + 3 * PH + 2 * GAP + 30;
  const L0 = 6;
  const L1 = 40;
  const X = (x: number) => M.l + ((Math.log(x) - Math.log(L0)) / (Math.log(L1) - Math.log(L0))) * (W - M.l - M.r);
  const xs = Array.from({ length: 80 }, (_, i) => L0 * (L1 / L0) ** (i / 79));
  const panels = [
    { title: 'one ion pair as a concentration [M]', f: (x: number) => concentrationPerPair(x, h), target: cTarget, need: Lc, color: theme.lit },
    { title: 'one electron as a surface charge [μC/cm²]', f: sigmaPerElectron, target: null as number | null, need: null as number | null, color: theme.arch },
    { title: 'potential step per electron [V]', f: (x: number) => potentialPerElectron(x), target: dU, need: Lu, color: theme.caveat },
  ];
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Concentration per ion pair, surface charge per electron and potential step per electron against cell size, on logarithmic axes.">
        {panels.map((p, k) => {
          const top = M.t + k * (PH + GAP);
          const vals = xs.map(p.f);
          const lo = Math.log10(Math.min(...vals, p.target ?? Infinity) * 0.8);
          const hi = Math.log10(Math.max(...vals, p.target ?? 0) * 1.25);
          const Y = (v: number) => top + (1 - (Math.log10(v) - lo) / (hi - lo)) * PH;
          const path = xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(vals[i]!).toFixed(1)}`).join('');
          const decades = Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.ceil(lo) + i);
          return (
            <g key={p.title}>
              <text x={M.l} y={top - 5} className="axis-label">{p.title}</text>
              <rect x={M.l} y={top} width={W - M.l - M.r} height={PH} className="frame" />
              {decades.map((d) => (
                <g key={d}>
                  <line x1={M.l} x2={W - M.r} y1={Y(10 ** d)} y2={Y(10 ** d)} className="grid" />
                  <text x={M.l - 5} y={Y(10 ** d) + 4} textAnchor="end" className="tick">{10 ** d >= 1 ? 10 ** d : (10 ** d).toPrecision(1)}</text>
                </g>
              ))}
              {p.target != null && <line x1={M.l} x2={W - M.r} y1={Y(p.target)} y2={Y(p.target)} style={{ stroke: p.color, strokeDasharray: '5 3', strokeWidth: 1.4 }} />}
              {p.need != null && p.need >= L0 && p.need <= L1 && <line x1={X(p.need)} x2={X(p.need)} y1={top} y2={top + PH} style={{ stroke: p.color, strokeDasharray: '2 3' }} />}
              <path d={path} className="curve" style={{ stroke: p.color, strokeWidth: 2.4 }} />
              <line x1={X(L)} x2={X(L)} y1={top} y2={top + PH} className="r-line" />
              <circle cx={X(L)} cy={Y(p.f(L))} r={4.5} style={{ fill: p.color }} />
            </g>
          );
        })}
        {[6, 10, 15, 20, 30, 40].map((x) => (
          <text key={x} x={X(x)} y={H - 14} textAnchor="middle" className="tick">{x}</text>
        ))}
        <text x={W - M.r} y={H - 1} textAnchor="end" className="axis-label">cell side L [Å] (log)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="swatch" style={{ background: theme.inkMuted }} /> dashed: your target and the cell side that meets it</span>
      </figcaption>
    </figure>
  );
}
