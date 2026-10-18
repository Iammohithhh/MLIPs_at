// Log-scale curves with a separate lane for exact zeros (to round-off), shared by the
// DimeNet/GemNet figures: "exactly zero" means the model cannot tell the two inputs apart.

export interface ZeroSeries {
  key: string;
  label: string;
  color: string;
  xs: number[];
  ys: number[];
  width?: number;
  dashed?: boolean;
}

const ZERO = 1e-12;

export function ZeroLaneChart({
  series,
  xMin,
  xMax,
  xTicks,
  xFmt = (v) => String(v),
  xLabel,
  yLabel,
  logLo = -7,
  logHi = 0,
  cursor,
  marks = [],
  bands = [],
  ariaLabel,
}: {
  series: ZeroSeries[];
  xMin: number;
  xMax: number;
  xTicks: number[];
  xFmt?: (v: number) => string;
  xLabel: string;
  yLabel: string;
  logLo?: number;
  logHi?: number;
  cursor?: number;
  /** Vertical reference lines. */
  marks?: Array<{ x: number; label: string }>;
  /** Shaded x ranges. */
  bands?: Array<{ from: number; to: number; label: string; tone: 'fail' | 'phys' }>;
  ariaLabel: string;
}) {
  const W = 520;
  const H = 330;
  const M = { l: 52, r: 14, t: 26, b: 62 };
  const zeroY = H - M.b + 22;
  const x = (v: number) => M.l + ((v - xMin) / (xMax - xMin)) * (W - M.l - M.r);
  const y = (v: number) => {
    if (v < ZERO) return zeroY;
    const l = Math.min(Math.max(Math.log10(v), logLo), logHi);
    return M.t + (1 - (l - logLo) / (logHi - logLo)) * (H - M.t - M.b);
  };
  const exps = Array.from({ length: logHi - logLo + 1 }, (_, k) => logLo + k).filter((e) => (e - logLo) % (logHi - logLo > 6 ? 2 : 1) === 0);
  return (
    <figure className="lab-plot dn-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
        <text x={M.l} y={M.t - 10} className="axis-label">
          {yLabel}
        </text>
        {bands.map((b) => (
          <g key={b.label}>
            <rect className={`dn-band dn-band--${b.tone}`} x={x(Math.max(b.from, xMin))} y={M.t} width={Math.max(0, x(Math.min(b.to, xMax)) - x(Math.max(b.from, xMin)))} height={zeroY + 8 - M.t} />
            <text x={(x(Math.max(b.from, xMin)) + x(Math.min(b.to, xMax))) / 2} y={M.t + 13} textAnchor="middle" className="tick">
              {b.label}
            </text>
          </g>
        ))}
        {exps.map((e) => (
          <g key={e}>
            <line x1={M.l} x2={W - M.r} y1={y(10 ** e)} y2={y(10 ** e)} className="grid" />
            <text x={M.l - 6} y={y(10 ** e) + 4} textAnchor="end" className="tick">
              10<tspan dy="-5" fontSize="8">{e}</tspan>
            </text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={zeroY} y2={zeroY} className="axis" strokeDasharray="3 3" />
        <text x={M.l - 6} y={zeroY + 4} textAnchor="end" className="tick">
          0
        </text>
        <text x={M.l + 4} y={zeroY - 6} className="tick">
          exactly zero: indistinguishable
        </text>
        {marks.map((m) => (
          <g key={m.label}>
            <line x1={x(m.x)} x2={x(m.x)} y1={M.t} y2={zeroY + 8} className="axis" strokeDasharray="2 4" />
            <text x={x(m.x) + 3} y={M.t + 26} className="tick">
              {m.label}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <path
            key={s.key}
            d={s.xs.map((v, k) => `${k ? 'L' : 'M'}${x(v).toFixed(1)},${y(s.ys[k]!).toFixed(1)}`).join('')}
            className="curve"
            style={{ stroke: s.color, strokeWidth: s.width ?? 2.4, strokeDasharray: s.dashed ? '6 4' : undefined, fill: 'none' }}
          />
        ))}
        {cursor !== undefined && <line x1={x(cursor)} x2={x(cursor)} y1={M.t} y2={zeroY + 8} style={{ stroke: 'var(--atlas-ink)', strokeWidth: 1.2 }} />}
        {xTicks.map((v) => (
          <text key={v} x={x(v)} y={zeroY + 24} textAnchor="middle" className="tick">
            {xFmt(v)}
          </text>
        ))}
        <text x={W - M.r} y={H - 4} textAnchor="end" className="axis-label">
          {xLabel}
        </text>
      </svg>
      <figcaption className="lab-legend">
        {series.map((s) => (
          <span key={s.key} className="item">
            <span className="swatch" style={{ background: s.color }} /> {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
