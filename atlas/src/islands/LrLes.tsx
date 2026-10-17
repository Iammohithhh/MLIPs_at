import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './LongRange.css';
import { Stage } from '~/three/Stage';
import { CellBox, ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { COULOMB_K } from '~/lib/chem/elements';
import { DIMER_BOX, LES_KC, dipoleTerm, gaussianPair, kConverged, pairEnergy, pairKernel, type Vec3 } from '~/lib/models/long-range';

// What Eq. (3) computes for the dimer benchmark of the LES paper: two charges in a 30 Å cubic box,
// σ = 1 Å, k_c = 2π/3 Å⁻¹ (Cheng 2025, Sec. I B). Curves from src/lib/models/long-range.ts, checked
// against the direct structure-factor sum in long-range.test.ts.

const L = DIMER_BOX;
const D = Array.from({ length: 141 }, (_, i) => 1 + 0.1 * i); // 1 … 15 Å (half the box)
const KC: Array<{ id: string; label: string; v: number | null }> = [
  { id: 'third', label: 'π/3', v: Math.PI / 3 },
  { id: 'paper', label: '2π/3 (dimers, NaCl)', v: LES_KC },
  { id: 'water', label: 'π (water)', v: Math.PI * 0.9999 },
  { id: 'inf', label: '→ ∞', v: null },
];
const PAIRS = [
  { id: 'cc', label: '+1 / −1', q: [1, -1] as const },
  { id: 'pp', label: '+1 / +1', q: [1, 1] as const },
  { id: 'cp', label: '+1 / −0.3', q: [1, -0.3] as const },
];
const SR_CUT = 5; // Å, CACE short-range model of the dimer benchmark (Sec. I B)

export default function LrLes() {
  const theme = useSceneTheme();
  const ids = useId();
  const [d, setD] = useState(8);
  const [sigma, setSigma] = useState(1);
  const [kcId, setKcId] = useState('paper');
  const [pairId, setPairId] = useState('cc');
  const pair = PAIRS.find((p) => p.id === pairId)!;
  const [q1, q2] = pair.q;
  const kcSel = KC.find((k) => k.id === kcId)!;
  const full = useMemo(() => pairKernel(L, sigma, kConverged(sigma)), [sigma]);
  const kern = useMemo(() => (kcSel.v === null ? full : pairKernel(L, sigma, kcSel.v)), [kcSel, sigma, full]);
  const curves = useMemo(
    () => ({
      les: D.map((x) => pairEnergy(x, q1, q2, kern, full, sigma)),
      full: D.map((x) => pairEnergy(x, q1, q2, full, full, sigma)),
      iso: D.map((x) => gaussianPair(x, q1, q2, sigma)),
      coul: D.map((x) => (COULOMB_K * q1 * q2) / x),
    }),
    [q1, q2, kern, full, sigma],
  );
  const eLes = pairEnergy(d, q1, q2, kern, full, sigma);
  const eFull = pairEnergy(d, q1, q2, full, full, sigma);
  const eIso = gaussianPair(d, q1, q2, sigma);
  const dip = q1 + q2 === 0 ? dipoleTerm([q2 * d, 0, 0], L ** 3) : null;
  const weightAtKc = kcSel.v === null ? 0 : Math.exp((-sigma * sigma * kcSel.v ** 2) / 2);
  const view = useMemo(() => ({ position: [L * 0.5, -L * 2.7, L * 1.7] as [number, number, number], target: [L / 2, L / 2, L / 2] as [number, number, number] }), []);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={view}
            resetKey={0}
            background={theme.canvas}
            label={`Two Gaussian charges ${pair.label} a distance ${d.toFixed(1)} Å apart in a ${L} Å periodic box, with their nearest images along x. The inner shell around each charge has radius 2σ = ${(2 * sigma).toFixed(1)} Å; the outer ring marks the short-range cutoff of ${SR_CUT} Å.`}
            minDistance={10}
            maxDistance={200}
          >
            <BoxScene theme={theme} d={d} sigma={sigma} q={[q1, q2]} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.fail }} /> q &gt; 0
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> q &lt; 0
            </li>
            <li>
              <span className="ring" /> {SR_CUT} Å cutoff <small>faint: images at ±{L} Å</small>
            </li>
          </ul>
        </div>
        <div className="lr-side">
          <div className="seg" role="group" aria-label="Charges">
            {PAIRS.map((p) => (
              <button key={p.id} type="button" aria-pressed={pairId === p.id} onClick={() => setPairId(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
          <Curve theme={theme} curves={curves} d={d} onPick={setD} />
          <Residual theme={theme} curves={curves} d={d} dipole={q1 + q2 === 0 ? D.map((x) => -dipoleTerm([q2 * x, 0, 0], L ** 3)) : null} />
          <ul className="lr-legend" aria-hidden="true">
            <li style={{ color: theme.phys }}>
              <span className="swatch" /> E<sub>lr</sub>, k<sub>c</sub> = {kcSel.label.split(' ')[0]}
            </li>
            <li style={{ color: theme.lit }}>
              <span className="swatch is-dashed" /> k<sub>c</sub> → ∞ (periodic)
            </li>
            <li style={{ color: theme.caveat }}>
              <span className="swatch is-dashed" /> isolated Gaussians
            </li>
            <li style={{ color: theme.inkMuted }}>
              <span className="swatch is-dashed" /> point charges
            </li>
          </ul>
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-d`}>
            <span>separation d</span>
            <input id={`${ids}-d`} type="range" min={1} max={15} step={0.1} value={d} onChange={(e) => setD(Number(e.target.value))} />
            <output>{d.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-s`}>
            <span>smearing σ</span>
            <input id={`${ids}-s`} type="range" min={0.7} max={2} step={0.1} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
            <output>{sigma.toFixed(1)} Å</output>
          </label>
          <div className="seg" role="group" aria-label="Reciprocal-space cutoff k_c (1/Å)">
            {KC.map((k) => (
              <button key={k.id} type="button" aria-pressed={kcId === k.id} onClick={() => setKcId(k.id)}>
                k<sub>c</sub> {k.label}
              </button>
            ))}
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>k vectors with 0 &lt; |k| &lt; k<sub>c</sub></dt>
            <dd>
              {kern.nk.toLocaleString('en')} <small>Gaussian weight at k<sub>c</sub>: {weightAtKc.toFixed(3)}</small>
            </dd>
          </div>
          <div>
            <dt>energy at d: chosen k<sub>c</sub> / converged</dt>
            <dd>
              {eLes.toFixed(3)} / {eFull.toFixed(3)} eV
            </dd>
          </div>
          <div>
            <dt>isolated Gaussian pair</dt>
            <dd>{eIso.toFixed(3)} eV</dd>
          </div>
          <div className={dip !== null ? undefined : 'is-alert'}>
            <dt>periodic minus isolated {dip !== null ? '(tinfoil dipole term)' : '(net charge: uniform background)'}</dt>
            <dd>
              {(eFull - eIso).toFixed(3)} eV {dip !== null ? <small>dipole term −{dip.toFixed(3)}</small> : null}
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          With σ = 1 Å the pair interaction is erf(d/√2σ)/d: finite at contact, and Coulomb’s 1/d to within 0.1% beyond 3.3 Å. A
          short-range model must supply what differs at short range. k<sub>c</sub> truncates the sum where the Gaussian weight is still about
          0.11 at the paper’s 2π/3 Å⁻¹, leaving ripples a few Å long. Large d shows the periodic box: a neutral pair is its own images’
          neighbour, and omitting k = 0 is the tinfoil boundary condition, which removes 2π|M|²/3V. With net charge (+1/+1), k = 0
          omission means a uniform neutralising background. Curves have the per-atom constants (self and truncation terms) removed.
        </p>
      </div>
    </div>
  );
}

type Curves = { les: number[]; full: number[]; iso: number[]; coul: number[] };

function Curve({ theme, curves, d, onPick }: { theme: SceneTheme; curves: Curves; d: number; onPick: (d: number) => void }) {
  const W = 380;
  const H = 240;
  const m = { l: 44, r: 10, t: 12, b: 34 };
  const vals = [...curves.les, ...curves.full, ...curves.iso];
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = 0.08 * (hi - lo || 1);
  const y0 = lo - pad;
  const y1 = hi + pad;
  const x = (r: number) => m.l + ((r - 1) / 14) * (W - m.l - m.r);
  const y = (v: number) => m.t + ((y1 - Math.max(y0, Math.min(y1, v))) / (y1 - y0)) * (H - m.t - m.b);
  const path = (v: number[]) => v.map((e, i) => `${i ? 'L' : 'M'}${x(D[i]!).toFixed(1)},${y(e).toFixed(1)}`).join('');
  const ticks = niceTicks(y0, y1);
  return (
    <figure className="lab-plot lr-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Long-range energy of the pair against separation: truncated latent Ewald sum, converged periodic sum, isolated Gaussian charges and point charges.">
        <rect className="lr-band" x={x(1)} y={m.t} width={x(SR_CUT) - x(1)} height={H - m.t - m.b} />
        <text className="lr-band-label" x={x(1) + 4} y={m.t + 10}>
          inside the {SR_CUT} Å cutoff
        </text>
        <line className="dpa-shell" x1={x(2 * SR_CUT)} x2={x(2 * SR_CUT)} y1={m.t} y2={H - m.b} style={{ stroke: theme.ruleStrong, strokeDasharray: '2 3' }} />
        <text className="lr-band-label" x={x(2 * SR_CUT) + 3} y={m.t + 10}>
          one MP layer: 10 Å
        </text>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
            <text className="tick" x={m.l - 4} y={y(t) + 3} textAnchor="end">
              {Math.abs(t) < 1e-9 ? '0' : t.toFixed(Math.abs(y1 - y0) < 3 ? 1 : 0)}
            </text>
          </g>
        ))}
        {[1, 5, 10, 15].map((r) => (
          <text key={r} className="tick" x={x(r)} y={H - m.b + 13} textAnchor="middle">
            {r}
          </text>
        ))}
        <text className="axis-label" x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">
          d (Å)
        </text>
        <text className="axis-label" x={12} y={m.t + 4} transform={`rotate(-90 12 ${m.t + 4})`} textAnchor="end">
          eV
        </text>
        <path className="lr-curve is-dashed" d={path(curves.coul)} style={{ stroke: theme.inkMuted }} />
        <path className="lr-curve is-dashed" d={path(curves.iso)} style={{ stroke: theme.caveat }} />
        <path className="lr-curve is-dashed" d={path(curves.full)} style={{ stroke: theme.lit }} />
        <path className="lr-curve" d={path(curves.les)} style={{ stroke: theme.phys }} />
        <line className="rc-line is-active" x1={x(d)} x2={x(d)} y1={m.t} y2={H - m.b} />
        <rect
          x={m.l}
          y={m.t}
          width={W - m.l - m.r}
          height={H - m.t - m.b}
          fill="transparent"
          style={{ cursor: 'pointer' }}
          onClick={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const px = ((e.clientX - box.left) / box.width) * W;
            onPick(Math.max(1, Math.min(15, Math.round((1 + ((px - m.l) / (W - m.l - m.r)) * 14) * 10) / 10)));
          }}
        />
      </svg>
      <figcaption className="lr-cap">Pair energy against separation; click to move the charges. Point-charge values beyond the axis are clipped.</figcaption>
    </figure>
  );
}

function Residual({ theme, curves, d, dipole }: { theme: SceneTheme; curves: Curves; d: number; dipole: number[] | null }) {
  const W = 380;
  const H = 130;
  const m = { l: 44, r: 10, t: 10, b: 26 };
  const les = curves.les.map((v, i) => v - curves.iso[i]!);
  const full = curves.full.map((v, i) => v - curves.iso[i]!);
  const vals = [...les, ...full, ...(dipole ?? []), 0];
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = 0.1 * (hi - lo || 0.01);
  const y0 = lo - pad;
  const y1 = hi + pad;
  const x = (r: number) => m.l + ((r - 1) / 14) * (W - m.l - m.r);
  const y = (v: number) => m.t + ((y1 - v) / (y1 - y0)) * (H - m.t - m.b);
  const path = (v: number[]) => v.map((e, i) => `${i ? 'L' : 'M'}${x(D[i]!).toFixed(1)},${y(e).toFixed(1)}`).join('');
  return (
    <figure className="lab-plot lr-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Difference between the latent Ewald energy and the isolated Gaussian pair, against separation, with the tinfoil dipole term for comparison.">
        {niceTicks(y0, y1).map((t) => (
          <g key={t}>
            <line className={Math.abs(t) < 1e-9 ? 'axis' : 'grid'} x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
            <text className="tick" x={m.l - 4} y={y(t) + 3} textAnchor="end">
              {Math.abs(t) < 1e-9 ? '0' : t.toFixed(Math.abs(y1 - y0) < 0.5 ? 2 : 1)}
            </text>
          </g>
        ))}
        {[1, 5, 10, 15].map((r) => (
          <text key={r} className="tick" x={x(r)} y={H - m.b + 13} textAnchor="middle">
            {r}
          </text>
        ))}
        {dipole ? <path className="lr-curve is-thin" d={path(dipole)} style={{ stroke: theme.ruleStrong }} /> : null}
        <path className="lr-curve is-dashed" d={path(full)} style={{ stroke: theme.lit }} />
        <path className="lr-curve" d={path(les)} style={{ stroke: theme.phys }} />
        <line className="rc-line is-active" x1={x(d)} x2={x(d)} y1={m.t} y2={H - m.b} />
      </svg>
      <figcaption className="lr-cap">
        Minus the isolated pair (eV): truncation ripples (solid), periodic images (dashed){dipole ? '; grey: −2π|M|²/3V alone' : ''}.
      </figcaption>
    </figure>
  );
}

function niceTicks(a: number, b: number): number[] {
  const span = b - a;
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => span / s <= 6) ?? 20;
  const out: number[] = [];
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) out.push(Math.round(t * 1e6) / 1e6);
  return out;
}

function BoxScene({ theme, d, sigma, q }: { theme: SceneTheme; d: number; sigma: number; q: readonly [number, number] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 40, 160);
  const c = L / 2;
  const sites: Array<{ p: Vec3; q: number; image: boolean }> = [];
  for (const shift of [-L, 0, L]) {
    sites.push({ p: [c - d / 2 + shift, c, c], q: q[0], image: shift !== 0 });
    sites.push({ p: [c + d / 2 + shift, c, c], q: q[1], image: shift !== 0 });
  }
  const buffers = useMemo<AtomBuffers>(() => {
    const n = sites.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const bg = linear(theme.canvas);
    sites.forEach((s, i) => {
      pos.set(s.p, 3 * i);
      radii[i] = 0.55 + 0.45 * Math.abs(s.q);
      const col = linear(s.q < 0 ? theme.arch : theme.fail);
      colors.set(s.image ? mixRGB(col, bg, 0.6) : col, 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.2), ao: new Float32Array(n).fill(1), count: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, q[0], q[1], theme]);
  return (
    <>
      <CellBox origin={[0, 0, 0]} size={[L, L, L]} color={theme.ruleStrong} />
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {sites
        .filter((s) => !s.image)
        .map((s, k) => (
          <group key={k}>
            <Shell center={s.p} radius={2 * sigma} color={s.q < 0 ? theme.arch : theme.fail} opacity={0.18} />
            <Shell center={s.p} radius={SR_CUT} color={theme.phys} opacity={0.05} />
          </group>
        ))}
    </>
  );
}
