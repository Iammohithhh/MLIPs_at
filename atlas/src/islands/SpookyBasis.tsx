import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Spooky.css';
import { BOHR, GAMMA_INIT, RC_PAPER, fcut, radialBasis } from '~/lib/models/spookynet';
import { Legend, LinePlot, niceTicks } from './spooky/LinePlot';

// Fig. 1: SpookyNet's radial basis, Bernstein polynomials of e^{−γr} times f_cut (Eqs. 14–16), live.

const PALETTE = ['var(--atlas-arch)', 'var(--atlas-phys)', 'var(--atlas-lit)', 'var(--atlas-caveat)', 'var(--atlas-fail)', 'var(--atlas-verified)'];

export default function SpookyBasis() {
  const id = useId();
  const [K, setK] = useState(8);
  const [gamma, setGamma] = useState(+GAMMA_INIT.toFixed(2));
  const [rc, setRc] = useState(+RC_PAPER.toFixed(2));
  const rMax = 10;
  const grid = useMemo(() => Array.from({ length: 241 }, (_, i) => (i * rMax) / 240), []);
  const data = useMemo(() => grid.map((r) => radialBasis(r, K, gamma, rc)), [grid, K, gamma, rc]);
  const peaks = useMemo(
    () =>
      Array.from({ length: K }, (_, k) => {
        let best = 0;
        data.forEach((row, i) => {
          if (row[k]! > data[best]![k]!) best = i;
        });
        return grid[best]!;
      }),
    [data, grid, K],
  );
  const series = Array.from({ length: K }, (_, k) => ({
    key: `k${k}`,
    points: grid.map((r, i) => [r, data[i]![k]!] as [number, number]),
    color: PALETTE[k % PALETTE.length]!,
    thin: true,
  }));
  const sumSeries = { key: 'sum', points: grid.map((r, i) => [r, data[i]!.reduce((s, v) => s + v, 0)] as [number, number]), color: 'var(--atlas-ink)' };
  const cutSeries = { key: 'cut', points: grid.map((r) => [r, fcut(r, rc)] as [number, number]), color: 'var(--atlas-ink-faint)', dashed: true };
  const inner = peaks.filter((p) => p > 0.05 && p < rc - 0.05).sort((a, b) => a - b);
  const gaps = inner.slice(1).map((p, i) => p - inner[i]!);

  return (
    <div className="lab">
      <div className="sp-grid2">
        <div>
          <p className="sp-h">The K radial functions and their total</p>
          <LinePlot
            H={190}
            x={{ min: 0, max: rMax, ticks: [0, 2, 4, 6, 8, 10], label: 'r (Å)' }}
            y={{ min: 0, max: 1.05, ticks: [0, 0.5, 1], label: 'ρ_k', fmt: (v) => v.toFixed(1) }}
            series={[...series, cutSeries, sumSeries]}
            band={rc}
            bandLabel="r ≥ r_cut"
            ariaLabel={`The ${K} radial basis functions against distance, crowded at short range and spread out at long range, all vanishing smoothly at the cutoff; their sum equals the cutoff function.`}
          />
          <Legend
            items={[
              { label: 'ρ₀ … ρ_K−1', color: 'var(--atlas-arch)' },
              { label: 'Σ_k ρ_k', color: 'var(--atlas-ink)' },
              { label: 'f_cut', color: 'var(--atlas-ink-faint)', dashed: true },
            ]}
          />
        </div>
        <div>
          <p className="sp-h">Where each function peaks</p>
          <LinePlot
            H={190}
            x={{ min: 0, max: Math.max(1, K - 1), ticks: niceTicks(0, Math.max(1, K - 1), 4), label: 'k' }}
            y={{ min: 0, max: rMax, ticks: [0, 2.5, 5, 7.5, 10], label: 'peak r (Å)', fmt: (v) => v.toFixed(1) }}
            series={[
              { key: 'peaks', points: peaks.map((p, k) => [k, p] as [number, number]), color: 'var(--atlas-arch)' },
              {
                key: 'free',
                points: Array.from({ length: K - 1 }, (_, i) => i + 1).map((k) => [k, Math.min(rMax, Math.log((K - 1) / k) / gamma)] as [number, number]),
                color: 'var(--atlas-caveat)',
                dashed: true,
              },
            ]}
            hlines={[{ v: rc, color: 'var(--atlas-fail)' }]}
            ariaLabel="Position of each basis function's maximum against its index k: high k peak near the nucleus, low k far away, and the cutoff pulls the outer ones in."
          />
          <Legend
            items={[
              { label: 'with f_cut', color: 'var(--atlas-arch)' },
              { label: 'ln((K−1)/k)/γ, no cutoff', color: 'var(--atlas-caveat)', dashed: true },
              { label: 'r_cut', color: 'var(--atlas-fail)', dashed: true },
            ]}
          />
        </div>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${id}-k`}>
            <span>K functions</span>
            <input id={`${id}-k`} type="range" min={2} max={16} step={1} value={K} onChange={(e) => setK(Number(e.target.value))} />
            <output>{K}</output>
          </label>
          <label htmlFor={`${id}-g`}>
            <span>decay γ</span>
            <input id={`${id}-g`} type="range" min={0.3} max={2.5} step={0.01} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
            <output>
              {gamma.toFixed(2)} Å⁻¹ <small>({(gamma * BOHR).toFixed(2)} a₀⁻¹)</small>
            </output>
          </label>
          <label htmlFor={`${id}-c`}>
            <span>r_cut</span>
            <input id={`${id}-c`} type="range" min={3} max={10} step={0.01} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>
              {rc.toFixed(2)} Å <small>({(rc / BOHR).toFixed(1)} a₀)</small>
            </output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className="is-good">
            <dt>max |Σ_k ρ_k − f_cut| on the grid</dt>
            <dd>{Math.max(...sumSeries.points.map(([r, s]) => Math.abs(s - fcut(r, rc)))).toExponential(1)}</dd>
          </div>
          <div>
            <dt>peaks inside r_cut, closest pair / widest pair</dt>
            <dd>{gaps.length ? `${Math.min(...gaps).toFixed(2)} / ${Math.max(...gaps).toFixed(2)} Å` : '—'}</dd>
          </div>
          <div>
            <dt>ρ₀ (rises monotonically without f_cut) at r_cut/2</dt>
            <dd>{radialBasis(rc / 2, K, gamma, rc)[0]!.toFixed(3)}</dd>
          </div>
        </dl>
        <p className="lab-note">
          Bernstein polynomials of one degree add up to one, so the K functions always sum to the cutoff function (black and dashed curves
          coincide): no distance inside the cutoff is unrepresented. Without the cutoff, function k peaks where e<sup>−γr</sup> = k/(K − 1), at
          r = ln((K − 1)/k)/γ; equal steps in k are unequal steps in r, closely spaced near the nucleus and spreading out with distance. The
          defaults are the paper’s initial γ = ½ a₀⁻¹ and r_cut = 10 a₀; γ is learned.
        </p>
      </div>
    </div>
  );
}
