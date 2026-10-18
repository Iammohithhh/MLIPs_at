import { useEffect, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Mtp.css';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { fitSi, spearman, type FitOptions, type FitResult, type SiData } from '~/lib/models/mtp-fit';

// A linear MTP fitted in the browser to the PBE energies of the Zuo et al. Si cells, from basis values
// precomputed on CPU (pipeline/atlas_pipeline/mtp.py → src/data/models/mtp-si.json), with each cell's
// extrapolation grade against a D-optimal active set chosen by maxvol from the training cells.

const GROUPS = ['Elastic', 'AIMD-NVT', 'Vacancy', 'Surface'] as const;
const LEVELS = [8, 10, 12, 14, 16];

function groupColor(theme: SceneTheme, g: string): string {
  return { Elastic: theme.verified, 'AIMD-NVT': theme.arch, Vacancy: theme.lit, Surface: theme.fail }[g] ?? theme.ink;
}

export default function MtpActiveSi() {
  const theme = useSceneTheme();
  const [data, setData] = useState<SiData | null>(null);
  const [opts, setOpts] = useState<FitOptions>({ levelMax: 12, leaveOut: 'Surface', fitOn: 'all', lambda: 1e-10 });
  useEffect(() => {
    let live = true;
    import('~/data/models/mtp-si.json').then((m) => live && setData(m.default as unknown as SiData));
    return () => {
      live = false;
    };
  }, []);
  const res = useMemo(() => (data ? fitSi(data, opts) : null), [data, opts]);
  const set = (p: Partial<FitOptions>) => setOpts((o) => ({ ...o, ...p }));

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        {res ? <Scatter theme={theme} res={res} /> : <p className="lab-note">Loading the precomputed basis values…</p>}
        <div className="mtp-side">
          <div className="lab-controls">
            <div className="seg" role="group" aria-label="MTP level">
              <span className="seg-label">lev<sub>max</sub></span>
              {LEVELS.map((L) => (
                <button key={L} type="button" aria-pressed={opts.levelMax === L} onClick={() => set({ levelMax: L })}>
                  {L}
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Group left out of training">
              <span className="seg-label">leave out</span>
              {[null, ...GROUPS].map((g) => (
                <button key={g ?? 'none'} type="button" aria-pressed={opts.leaveOut === g} onClick={() => set({ leaveOut: g })}>
                  {g ?? 'nothing'}
                </button>
              ))}
            </div>
            <div className="seg" role="group" aria-label="Cells used for the fit">
              <span className="seg-label">fit on</span>
              <button type="button" aria-pressed={opts.fitOn === 'all'} onClick={() => set({ fitOn: 'all' })}>
                all training cells
              </button>
              <button type="button" aria-pressed={opts.fitOn === 'active'} onClick={() => set({ fitOn: 'active' })}>
                the active set only
              </button>
            </div>
          </div>
          {res ? <Readout res={res} opts={opts} theme={theme} /> : null}
        </div>
      </div>
      <p className="lab-note">
        Every point is one DFT cell: across, its extrapolation grade γ against the active set; up, the MTP’s energy error. Training cells sit
        at γ ≤ 1 by construction (maxvol leaves none outside). Leave the surfaces out and they land far to the right with errors tens of
        times the test error: γ flags them without any DFT calculation. Raise the level and the training error falls, but with
        energies alone the larger models fit noise, and at level 16 some choices leave too few independent cells to build an active set at all.
      </p>
    </div>
  );
}

const fmt = (v: number) => (Number.isFinite(v) ? (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) : '—');

function Readout({ res, opts, theme }: { res: FitResult; opts: FitOptions; theme: SceneTheme }) {
  const ev = res.cells.filter((c) => c.role !== 'train' && Number.isFinite(c.gamma));
  const rho = ev.length > 3 ? spearman(ev.map((c) => c.gamma), ev.map((c) => c.error)) : NaN;
  const med = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]! : NaN);
  return (
    <>
      <dl className="lab-readout" aria-live="polite">
        <div>
          <dt>parameters m (basis functions + 1)</dt>
          <dd>
            {res.m} <small>from {res.cells.filter((c) => c.role === 'train').length} training cells</small>
          </dd>
        </div>
        <div className={res.active ? undefined : 'is-alert'}>
          <dt>active set (maxvol)</dt>
          <dd>{res.active ? `${res.active.rows.length} cells, ${res.active.swaps} swaps after the greedy start` : 'none: the training cells do not determine all m coefficients'}</dd>
        </div>
        <div>
          <dt>RMSE train / test{opts.leaveOut ? ' / left out' : ''} (meV/atom)</dt>
          <dd>
            {fmt(res.rmse.train)} / {fmt(res.rmse.test)}
            {opts.leaveOut ? <> / <span style={{ color: theme.fail }}>{fmt(res.rmse.leftOut ?? NaN)}</span></> : null}
          </dd>
        </div>
        <div>
          <dt>rank correlation of <span className="mtp-nocase">γ</span> and error (cells not trained on)</dt>
          <dd>{Number.isFinite(rho) ? rho.toFixed(2) : '—'}</dd>
        </div>
      </dl>
      <table className="mtp-table">
        <caption>Median γ and RMSE (meV/atom) by group, cells not trained on</caption>
        <thead>
          <tr>
            <th scope="col">group</th>
            <th scope="col">cells</th>
            <th scope="col">median <span className="mtp-nocase">γ</span></th>
            <th scope="col">RMSE</th>
          </tr>
        </thead>
        <tbody>
          {GROUPS.map((g) => {
            const c = ev.filter((x) => x.config.group === g);
            const rmse = c.length ? Math.sqrt(c.reduce((s, x) => s + x.error ** 2, 0) / c.length) : NaN;
            return (
              <tr key={g}>
                <td>
                  <span className="mtp-swatch" style={{ background: groupColor(theme, g) }} /> {g}
                  {g === opts.leaveOut ? ' (left out)' : ''}
                </td>
                <td>{c.length}</td>
                <td>{fmt(med(c.map((x) => x.gamma)))}</td>
                <td>{fmt(rmse)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

function Scatter({ theme, res }: { theme: SceneTheme; res: FitResult }) {
  const W = 540;
  const H = 400;
  const x0 = 58;
  const y0 = 26;
  const w = W - x0 - 16;
  const h = H - y0 - 64;
  const fin = res.cells.filter((c) => Number.isFinite(c.gamma) && Number.isFinite(c.error));
  const gx = [-1, Math.min(4, Math.max(2, Math.ceil(Math.log10(Math.max(...fin.map((c) => c.gamma), 1)))))];
  const ey = [Math.max(-3, Math.min(-1, Math.floor(Math.log10(Math.max(1e-3, Math.min(...fin.map((c) => c.error))))))), Math.max(2, Math.min(4, Math.ceil(Math.log10(Math.max(...fin.map((c) => c.error), 1)))))];
  const px = (g: number) => x0 + ((Math.log10(Math.max(g, 10 ** gx[0]!)) - gx[0]!) / (gx[1]! - gx[0]!)) * w;
  const py = (e: number) => y0 + ((ey[1]! - Math.min(ey[1]!, Math.log10(Math.max(e, 10 ** ey[0]!)))) / (ey[1]! - ey[0]!)) * h;
  const sup = (e: number) => (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]).join('');
  const cells = [...res.cells].sort((a, b) => (a.role === 'train' ? -1 : 0) - (b.role === 'train' ? -1 : 0));
  return (
    <figure className="lab-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter plot, both axes logarithmic: extrapolation grade γ of each DFT cell against the MTP energy error per atom, coloured by group. Training cells sit at γ ≤ 1; cells of a left-out group lie to the right with larger errors.">
        <text x={x0} y={y0 - 10} className="axis-label">|E_MTP − E_DFT| per atom (meV, log)</text>
        <rect x={px(10)} width={x0 + w - px(10)} y={y0} height={h} style={{ fill: theme.fail, opacity: 0.07 }} />
        <rect x={px(2)} width={px(10) - px(2)} y={y0} height={h} style={{ fill: theme.caveat, opacity: 0.07 }} />
        {Array.from({ length: ey[1]! - ey[0]! + 1 }, (_, k) => ey[0]! + k).map((e) => (
          <g key={`y${e}`}>
            <line x1={x0} x2={x0 + w} y1={py(10 ** e)} y2={py(10 ** e)} className="grid" />
            <text x={x0 - 6} y={py(10 ** e) + 4} textAnchor="end" className="tick">10{sup(e)}</text>
          </g>
        ))}
        {Array.from({ length: gx[1]! - gx[0]! + 1 }, (_, k) => gx[0]! + k).map((e) => (
          <g key={`x${e}`}>
            <line x1={px(10 ** e)} x2={px(10 ** e)} y1={y0} y2={y0 + h} className="grid" />
            <text x={px(10 ** e)} y={y0 + h + 16} textAnchor="middle" className="tick">10{sup(e)}</text>
          </g>
        ))}
        {[
          [1, 'γ = 1', theme.verified],
          [2, 'select', theme.caveat],
          [10, 'break', theme.fail],
        ].map(([g, t, c]) => (
          <g key={t as string}>
            <line x1={px(g as number)} x2={px(g as number)} y1={y0} y2={y0 + h} style={{ stroke: c as string, strokeWidth: 1.2, strokeDasharray: g === 1 ? undefined : '4 3' }} />
            <text x={px(g as number) + 3} y={y0 + 11} className="tick" style={{ fill: c as string }}>{t}</text>
          </g>
        ))}
        {cells.map((c) => {
          if (!Number.isFinite(c.gamma) || !Number.isFinite(c.error)) return null;
          const col = groupColor(theme, c.config.group);
          const x = px(c.gamma);
          const y = py(c.error);
          if (c.role === 'train')
            return <circle key={c.index} cx={x} cy={y} r={c.active ? 3.4 : 2.4} style={{ fill: c.active ? col : 'none', stroke: col, strokeWidth: 1, opacity: 0.55 }} />;
          if (c.role === 'test') return <circle key={c.index} cx={x} cy={y} r={4.6} style={{ fill: col, stroke: theme.canvas, strokeWidth: 1.2 }} />;
          return <rect key={c.index} x={x - 4.4} y={y - 4.4} width={8.8} height={8.8} transform={`rotate(45 ${x} ${y})`} style={{ fill: col, stroke: theme.canvas, strokeWidth: 1.2 }} />;
        })}
        <text x={x0 + w} y={y0 + h + 36} textAnchor="end" className="tick">extrapolation grade γ (log)</text>
      </svg>
      <figcaption className="lab-legend">
        <span className="item"><span className="mtp-mark mtp-mark--ring" /> training cell (filled: in the active set)</span>
        <span className="item"><span className="mtp-mark mtp-mark--dot" /> test cell</span>
        <span className="item"><span className="mtp-mark mtp-mark--diamond" /> left-out group</span>
        {GROUPS.map((g) => (
          <span key={g} className="item"><span className="mtp-swatch" style={{ background: groupColor(theme, g) }} /> {g}</span>
        ))}
      </figcaption>
    </figure>
  );
}
