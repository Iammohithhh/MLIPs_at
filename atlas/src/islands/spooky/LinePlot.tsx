// A small SVG line plot shared by the SpookyNet figures.

export interface Series {
  key: string;
  points: Array<[number, number]>;
  color: string;
  dashed?: boolean;
  thin?: boolean;
}

export interface Axis {
  min: number;
  max: number;
  ticks: number[];
  label: string;
  fmt?: (v: number) => string;
  log?: boolean;
}

const W = 380;
const M = { l: 46, r: 10, t: 10, b: 28 };

export function LinePlot({
  H = 160,
  x,
  y,
  series,
  cursor,
  band,
  bandLabel,
  hlines = [],
  ariaLabel,
}: {
  H?: number;
  x: Axis;
  y: Axis;
  series: Series[];
  cursor?: number;
  /** Shade x ≥ band. */
  band?: number;
  bandLabel?: string;
  hlines?: Array<{ v: number; color: string }>;
  ariaLabel: string;
}) {
  const tx = (v: number) => (x.log ? Math.log10(v) : v);
  const ty = (v: number) => (y.log ? Math.log10(Math.max(v, 1e-300)) : v);
  const xOf = (v: number) => M.l + ((tx(v) - tx(x.min)) / (tx(x.max) - tx(x.min))) * (W - M.l - M.r);
  const yOf = (v: number) => {
    const t = (ty(v) - ty(y.min)) / (ty(y.max) - ty(y.min));
    return M.t + (1 - Math.max(-0.02, Math.min(1.02, t))) * (H - M.t - M.b);
  };
  const fx = x.fmt ?? ((v: number) => String(v));
  const fy = y.fmt ?? ((v: number) => String(v));
  return (
    <figure className="lab-plot sp-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
        {band !== undefined && band < x.max ? (
          <>
            <rect className="sp-band" x={xOf(Math.max(band, x.min))} y={M.t} width={W - M.r - xOf(Math.max(band, x.min))} height={H - M.t - M.b} />
            {bandLabel ? (
              <text className="sp-mark" x={xOf(Math.max(band, x.min)) + 3} y={M.t + 9}>
                {bandLabel}
              </text>
            ) : null}
          </>
        ) : null}
        {y.ticks.map((v) => (
          <g key={`y${v}`}>
            <line className={v === 0 && !y.log ? 'axis' : 'grid'} x1={M.l} x2={W - M.r} y1={yOf(v)} y2={yOf(v)} />
            <text className="tick" x={M.l - 4} y={yOf(v) + 3} textAnchor="end">
              {fy(v)}
            </text>
          </g>
        ))}
        {x.ticks.map((v) => (
          <text key={`x${v}`} className="tick" x={xOf(v)} y={H - M.b + 13} textAnchor="middle">
            {fx(v)}
          </text>
        ))}
        {hlines.map((h, i) => (
          <line key={`h${i}`} className="sp-curve is-dashed is-thin" x1={M.l} x2={W - M.r} y1={yOf(h.v)} y2={yOf(h.v)} style={{ stroke: h.color }} />
        ))}
        {series.map((s) => (
          <path
            key={s.key}
            className={`sp-curve${s.dashed ? ' is-dashed' : ''}${s.thin ? ' is-thin' : ''}`}
            d={s.points.map(([a, b], i) => `${i ? 'L' : 'M'}${xOf(a).toFixed(1)},${yOf(b).toFixed(1)}`).join('')}
            style={{ stroke: s.color }}
          />
        ))}
        {cursor !== undefined ? <line className="axis" x1={xOf(cursor)} x2={xOf(cursor)} y1={M.t} y2={H - M.b} /> : null}
        <text className="axis-label" x={W - M.r} y={H - 2} textAnchor="end">
          {x.label}
        </text>
        <text className="axis-label" x={11} y={M.t + 2} transform={`rotate(-90 11 ${M.t + 2})`} textAnchor="end">
          {y.label}
        </text>
      </svg>
    </figure>
  );
}

export function Legend({ items }: { items: Array<{ label: string; color: string; dashed?: boolean }> }) {
  return (
    <ul className="sp-legend" aria-hidden="true">
      {items.map((it) => (
        <li key={it.label} style={{ color: it.color }}>
          <span className={`swatch${it.dashed ? ' is-dashed' : ''}`} /> {it.label}
        </li>
      ))}
    </ul>
  );
}

/** Evenly spaced ticks covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
