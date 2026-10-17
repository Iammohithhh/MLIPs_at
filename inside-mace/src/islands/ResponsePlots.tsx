// "Which parts of the model respond?": for the selected atom, the feature-block difference norm
// ‖h(frame) − h(baseline)‖ against displacement, one small panel per layer and irrep block, each with its
// own axis (internal features of different layers are not on comparable scales), plus the change in the
// atom's site energy (eV). Points are evaluated frames only.
import { featureChange, siteEnergyChange, type FeatureBlock, type Manifest } from '../lib/experiment';

interface Series {
  key: string;
  title: string;
  unit: string;
  get: (k: number) => number | null;
}

export function ResponsePlots({ m, atom, frame, onFrame }: { m: Manifest; atom: number; frame: number; onFrame: (k: number) => void }) {
  const vals = m.displacement.values;
  const series: Series[] = [];
  m.baseline.norms.forEach((n, t) => {
    (['h0', 'h1'] as FeatureBlock[]).forEach((b) => {
      if (n[b] === null) return;
      series.push({
        key: `${t}${b}`,
        title: `Layer ${t + 1}, ${b === 'h0' ? '128 scalars (l = 0)' : '128 vectors (l = 1)'}: ‖Δh‖`,
        unit: 'model units',
        get: (k) => featureChange(m, k, t, b, atom),
      });
    });
  });
  series.push({ key: 'Ei', title: `Site energy change ΔE${atom === m.displacement.atom ? ' of the moved atom' : `ᵢ, atom ${atom}`}`, unit: 'eV', get: (k) => siteEnergyChange(m, k, atom) });

  const W = 220;
  const H = 110;
  const pad = { l: 8, r: 8, t: 8, b: 8 };
  const x = (v: number) => pad.l + ((v - vals[0]!) / (vals.at(-1)! - vals[0]!)) * (W - pad.l - pad.r);
  return (
    <div className="rp">
      {series.map((sr) => {
        const ys = vals.map((_, k) => sr.get(k));
        const fin = ys.filter((v): v is number => v !== null);
        const lo = Math.min(0, ...fin);
        const hi = Math.max(0, ...fin);
        const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
        const cur = ys[frame];
        return (
          <figure key={sr.key} className="rp__panel">
            <figcaption>
              <span>{sr.title}</span>
              <span className="mono rp__val">{cur === null || cur === undefined ? 'no data' : `${cur.toPrecision(3).replace(/^-/, '−')} ${sr.unit}`}</span>
            </figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${sr.title} against displacement, range ${lo.toPrecision(2)} to ${hi.toPrecision(2)} ${sr.unit}`}>
              <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} className="axis" />
              <line x1={x(0)} x2={x(0)} y1={pad.t} y2={H - pad.b} className="axis" />
              {ys.map((v, k) =>
                v === null ? null : (
                  <circle key={k} cx={x(vals[k]!)} cy={y(v)} r={k === frame ? 3.6 : 2} className={k === frame ? 'pt is-on' : 'pt'} onClick={() => onFrame(k)} />
                ),
              )}
            </svg>
            <p className="rp__range mono">
              {lo.toPrecision(2).replace(/^-/, '−')} … {hi.toPrecision(2).replace(/^-/, '−')} {sr.unit}
            </p>
          </figure>
        );
      })}
      <p className="rp__def">
        ‖Δh‖ is the norm of the difference of the whole feature block between this frame and the baseline (not the difference of two
        norms). Each panel has its own axis; compare shapes, not heights, across panels.
      </p>
    </div>
  );
}
