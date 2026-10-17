import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Gap.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { mulberry32 } from '~/lib/chem/rng';
import { DEFAULT_SOAP, fcut, latticeEnv, powerByL, powerSpectrum, normalise, soapKernel, soapLength, type LatticeKind, type SoapOptions, type Vec3 } from '~/lib/models/soap';

// SOAP compares two atoms by the overlap of their neighbour densities, averaged over rotations
// (src/lib/models/soap.ts). Seven environments with the same nearest-neighbour distance, so only the
// arrangement differs; every kernel value is computed live from the power spectra.

const D = 2.556; // Cu's nearest-neighbour distance, a/√2 with a = 3.615 Å
const RMAX = 6.5;

interface EnvDef {
  id: string;
  label: string;
  env: Vec3[];
}

function rattle(vs: Vec3[], amp: number, seed: number): Vec3[] {
  const rand = mulberry32(seed);
  const g = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  return vs.map((v) => [v[0] + amp * g(), v[1] + amp * g(), v[2] + amp * g()]);
}

const ENVS: EnvDef[] = (() => {
  const lat = (k: LatticeKind, d = D) => latticeEnv(k, d, RMAX);
  return [
    { id: 'fcc', label: 'fcc', env: lat('fcc') },
    { id: 'fcc-c', label: 'fcc −3 %', env: lat('fcc', D * 0.97) },
    { id: 'fcc-r', label: 'fcc rattled', env: rattle(lat('fcc'), 0.1, 4) },
    { id: 'hcp', label: 'hcp', env: lat('hcp') },
    { id: 'bcc', label: 'bcc', env: lat('bcc') },
    { id: 'sc', label: 'sc', env: lat('sc') },
    { id: 'dia', label: 'diamond', env: lat('diamond') },
  ];
})();

export default function SoapKernel() {
  const theme = useSceneTheme();
  const ids = useId();
  const [sigma, setSigma] = useState(0.5);
  const [rc, setRc] = useState(5.0);
  const [zeta, setZeta] = useState(4);
  const [lmax, setLmax] = useState(6);
  const [pair, setPair] = useState<[number, number]>([0, 3]);
  const o: SoapOptions = useMemo(() => ({ ...DEFAULT_SOAP, sigma, rc, lmax }), [sigma, rc, lmax]);
  const spectra = useMemo(() => ENVS.map((e) => powerSpectrum(e.env, o)), [o]);
  const unit = useMemo(() => spectra.map(normalise), [spectra]);
  const K = useMemo(() => unit.map((a) => unit.map((b) => soapKernel(a, b, zeta))), [unit, zeta]);
  const dist = K.map((row) => row.map((k) => Math.sqrt(Math.max(0, 2 - 2 * k))));
  const dmax = Math.max(1e-6, ...dist.flat());
  const [ia, ib] = pair;
  const A = ENVS[ia]!;
  const B = ENVS[ib]!;
  const shareA = powerByL(spectra[ia]!, o);
  const shareB = powerByL(spectra[ib]!, o);
  const nIn = (e: Vec3[]) => e.filter((v) => Math.hypot(...v) < rc).length;

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [14, -19, 11], target: [0, 0, 0] }}
            background={theme.canvas}
            label={`The ${A.label} environment: a centre atom and its neighbours inside ${rc.toFixed(1)} Å, each drawn as a Gaussian of width ${sigma.toFixed(2)} Å and faded by the cutoff weight.`}
            minDistance={4}
            maxDistance={60}
          >
            <Scene theme={theme} env={A.env} sigma={sigma} rc={rc} width={o.width} />
          </Stage>
          <div className="lab-toolbar">
            <span className="lab-tag">{A.label}</span>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: theme.ink }} /> centre
            </li>
            <li>
              <span className="dot" style={{ background: theme.arch }} /> neighbour, radius σ
            </li>
            <li>
              <span className="ring" style={{ borderColor: theme.phys }} /> r<sub>c</sub> − d, r<sub>c</sub>
            </li>
          </ul>
        </div>
        <div className="gap-side">
          <h4>Normalised kernel K = (p̂·p̂′)<sup>ζ</sup></h4>
          <Heatmap theme={theme} K={K} dist={dist} dmax={dmax} pair={pair} onPick={setPair} />
          <p className="gap-hint">Rows set the 3D view; pick any cell to compare two environments.</p>
          <ShareChart theme={theme} a={shareA} b={shareB} la={A.label} lb={B.label} />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-s`}>
            <span>Gaussian width σ</span>
            <input id={`${ids}-s`} type="range" min={0.2} max={1.0} step={0.05} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
            <output>{sigma.toFixed(2)} Å</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>cutoff r<sub>c</sub></span>
            <input id={`${ids}-rc`} type="range" min={2.8} max={6} step={0.1} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(1)} Å</output>
          </label>
          <label htmlFor={`${ids}-z`}>
            <span>exponent ζ</span>
            <input id={`${ids}-z`} type="range" min={1} max={8} step={1} value={zeta} onChange={(e) => setZeta(Number(e.target.value))} />
            <output>{zeta}</output>
          </label>
          <label htmlFor={`${ids}-l`}>
            <span>angular channels l<sub>max</sub></span>
            <input id={`${ids}-l`} type="range" min={0} max={8} step={1} value={lmax} onChange={(e) => setLmax(Number(e.target.value))} />
            <output>{lmax}</output>
          </label>
          <div className="presets">
            <button
              type="button"
              className="lab-btn"
              onClick={() => {
                setSigma(0.5);
                setRc(5);
                setZeta(4);
                setLmax(6);
              }}
            >
              Si GAP settings (l<sub>max</sub> 6 here, 12 in the model)
            </button>
          </div>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>
              K({A.label}, {B.label})
            </dt>
            <dd>{K[ia]![ib]!.toFixed(4)}</dd>
          </div>
          <div>
            <dt>kernel distance √(2 − 2K)</dt>
            <dd>{dist[ia]![ib]!.toFixed(4)}</dd>
          </div>
          <div>
            <dt>neighbours inside r<sub>c</sub></dt>
            <dd>
              {nIn(A.env)} / {nIn(B.env)}
            </dd>
          </div>
          <div>
            <dt>power-spectrum length</dt>
            <dd>{soapLength(o).toLocaleString('en')}</dd>
          </div>
        </dl>
        <p className="lab-note">
          The l = 0 channel, the radial distribution of neighbours, carries almost all of |p|², hence the log scale; fcc and the other cubic lattices have no weight at l = 1, 2, 3 or 5, which cubic symmetry forbids. With
          l<sub>max</sub> = 0 only distances count and fcc, hcp and the strained fcc look much alike; each extra angular channel
          adds bond-angle information. Wide Gaussians blur the shells together and raise every similarity; a larger ζ sharpens the
          kernel so that only close matches stay near 1. fcc and hcp share their first two shells (12 at d, 6 at √2 d) and differ in
          bond angles and in later shells, so the cutoff and l<sub>max</sub> decide how different they look.
        </p>
      </div>
    </div>
  );
}

function Heatmap({ theme, K, dist, dmax, pair, onPick }: { theme: SceneTheme; K: number[][]; dist: number[][]; dmax: number; pair: [number, number]; onPick: (p: [number, number]) => void }) {
  const n = ENVS.length;
  const cell = 38;
  const lab = 74;
  const W = lab + n * cell + 28;
  const H = lab + n * cell;
  const lo = linear(theme.surface);
  const hi = linear(theme.arch);
  const css = (c: [number, number, number]) => `rgb(${c.map((x) => Math.round(255 * Math.pow(Math.max(0, Math.min(1, x)), 1 / 2.2))).join(',')})`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="gap-heat" role="group" aria-label="Kernel matrix between the seven environments; darker cells are more similar.">
      {ENVS.map((e, i) => (
        <g key={e.id}>
          <text x={lab - 6} y={lab + i * cell + cell / 2 + 4} textAnchor="end" className="tick">
            {e.label}
          </text>
          <text x={lab + i * cell + cell / 2} y={lab - 8} textAnchor="start" className="tick" transform={`rotate(-50 ${lab + i * cell + cell / 2} ${lab - 8})`}>
            {e.label}
          </text>
        </g>
      ))}
      {K.map((row, i) =>
        row.map((k, j) => {
          const t = 1 - dist[i]![j]! / dmax;
          const fill = css(mixRGB(lo, hi, 0.08 + 0.92 * t));
          const on = pair[0] === i && pair[1] === j;
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={lab + j * cell + 1}
                y={lab + i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={3}
                style={{ fill, stroke: on ? theme.fail : 'none', strokeWidth: on ? 2.5 : 0, cursor: 'pointer' }}
                onClick={() => onPick([i, j])}
                role="button"
                tabIndex={0}
                aria-label={`${ENVS[i]!.label} against ${ENVS[j]!.label}: K = ${k.toFixed(3)}`}
                onKeyDown={(ev) => {
                  if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault();
                    onPick([i, j]);
                  }
                }}
              />
              <text x={lab + j * cell + cell / 2} y={lab + i * cell + cell / 2 + 3.5} textAnchor="middle" className="gap-heat-v" style={{ fill: t > 0.55 ? theme.canvas : theme.ink }} aria-hidden="true">
                {k > 0.9995 ? '1' : k.toFixed(2).replace(/^0/, '')}
              </text>
            </g>
          );
        }),
      )}
    </svg>
  );
}

function ShareChart({ theme, a, b, la, lb }: { theme: SceneTheme; a: number[]; b: number[]; la: string; lb: string }) {
  const W = 420;
  const H = 150;
  const M = { l: 44, r: 10, t: 22, b: 26 };
  const n = a.length;
  const bw = (W - M.l - M.r) / n;
  // Log scale: l = 0 (the radial distribution) carries almost all of |p|²; the angular channels are orders smaller.
  const y0 = 1e-6;
  const y = (v: number) => M.t + (1 - (Math.log10(Math.max(v, y0)) - Math.log10(y0)) / -Math.log10(y0)) * (H - M.t - M.b);
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Share of the power spectrum carried by each angular channel l, for ${la} and ${lb}, on a logarithmic scale.`}>
        {[1e-6, 1e-4, 1e-2, 1].map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 5} y={y(t) + 4} textAnchor="end" className="tick">
              {t === 1 ? '1' : `10${t === 1e-2 ? '⁻²' : t === 1e-4 ? '⁻⁴' : '⁻⁶'}`}
            </text>
          </g>
        ))}
        {a.map((v, l) => (
          <g key={l}>
            <rect x={M.l + l * bw + bw * 0.12} y={y(v)} width={bw * 0.36} height={Math.max(0, y(y0) - y(v))} style={{ fill: theme.arch }} />
            <rect x={M.l + l * bw + bw * 0.52} y={y(b[l]!)} width={bw * 0.36} height={Math.max(0, y(y0) - y(b[l]!))} style={{ fill: theme.caveat }} />
            <text x={M.l + (l + 0.5) * bw} y={H - M.b + 15} textAnchor="middle" className="tick">
              {l}
            </text>
          </g>
        ))}
        <text x={M.l} y={13} className="tick" style={{ fill: theme.arch }}>
          ■ {la}
        </text>
        <text x={M.l + 110} y={13} className="tick" style={{ fill: theme.caveat }}>
          ■ {lb}
        </text>
        <text x={W - M.r} y={13} textAnchor="end" className="axis-label">
          share of |p|² by l
        </text>
      </svg>
    </figure>
  );
}

function Scene({ theme, env, sigma, rc, width }: { theme: SceneTheme; env: Vec3[]; sigma: number; rc: number; width: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 12, 60);
  const buffers = useMemo<AtomBuffers>(() => {
    const inside = env.filter((v) => Math.hypot(...v) < rc);
    const n = inside.length + 1;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const arch = linear(theme.arch);
    const stone = linear(theme.dark ? '#3c4552' : '#d5d9df');
    radii[0] = sigma;
    colors.set(linear(theme.ink), 0);
    inside.forEach((v, k) => {
      pos.set(v, 3 * (k + 1));
      radii[k + 1] = sigma;
      colors.set(mixRGB(stone, arch, fcut(Math.hypot(...v), rc, width)), 3 * (k + 1));
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n).fill(0.4), ao: new Float32Array(n).fill(1), count: n };
  }, [env, sigma, rc, width, theme]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Shell center={[0, 0, 0]} radius={rc} color={theme.phys} opacity={0.08} />
      <Shell center={[0, 0, 0]} radius={Math.max(0.1, rc - width)} color={theme.phys} opacity={0.05} />
    </>
  );
}
