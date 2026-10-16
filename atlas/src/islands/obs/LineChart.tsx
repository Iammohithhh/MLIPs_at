import { useEffect, useId, useMemo, useRef, useState } from 'react';

// A small SVG line chart for the Observatory: one series per model, colour by model
// identity (fixed order), direct labels at the line ends, a crosshair with a tooltip
// that lists every series at the hovered x, and a table view for screen readers.

export interface Series {
  key: string;
  name: string;
  /** Short name for the direct label at the line end. */
  short?: string;
  /** CSS colour (a model token). */
  color: string;
  y: (number | null)[];
  dashed?: boolean;
}

export interface LineChartProps {
  x: number[];
  series: Series[];
  xLabel: string;
  yLabel: string;
  /** Format a y value for the tooltip and table. */
  fmtY?: (v: number) => string;
  fmtX?: (v: number) => string;
  /** Clip the y range to this window (values outside are drawn at the edge, marked). */
  yClip?: [number, number];
  /** Horizontal reference line (e.g. zero). */
  zero?: boolean;
  /** Vertical marker lines, e.g. a gallery geometry. */
  marks?: { x: number; label: string }[];
  height?: number;
  caption: string;
}

const M = { l: 58, r: 92, t: 14, b: 44 };

function ticks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
  return out;
}

export default function LineChart({
  x,
  series,
  xLabel,
  yLabel,
  fmtY = (v) => v.toFixed(3),
  fmtX = (v) => v.toFixed(2),
  yClip,
  zero,
  marks = [],
  height = 300,
  caption,
}: LineChartProps) {
  const id = useId();
  // Draw at the container's real width so text stays at its CSS size on phones and wide screens.
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.min(780, Math.round(e!.contentRect.width)))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const xs = [Math.min(...x), Math.max(...x)];
  const ys = useMemo(() => {
    const all = series.flatMap((s) => s.y.filter((v): v is number => v !== null && Number.isFinite(v)));
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (yClip) {
      lo = Math.max(lo, yClip[0]);
      hi = Math.min(hi, yClip[1]);
    }
    if (zero) {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    const pad = (hi - lo || 1) * 0.06;
    return [lo - pad, hi + pad] as [number, number];
  }, [series, yClip, zero]);
  const sx = (v: number) => M.l + ((v - xs[0]!) / (xs[1]! - xs[0]! || 1)) * iw;
  const sy = (v: number) => M.t + (1 - (v - ys[0]) / (ys[1] - ys[0])) * ih;
  const clampY = (v: number) => Math.min(ys[1], Math.max(ys[0], v));

  const paths = series.map((s) => {
    let d = '';
    let pen = false;
    s.y.forEach((v, k) => {
      if (v === null || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${sx(x[k]!).toFixed(1)},${sy(clampY(v)).toFixed(1)}`;
      pen = true;
    });
    return d;
  });

  // Direct labels at the right end, nudged apart so they do not collide.
  const labels = useMemo(() => {
    const raw = series.map((s, i) => {
      let k = s.y.length - 1;
      while (k > 0 && (s.y[k] === null || !Number.isFinite(s.y[k]!))) k--;
      return { i, y: sy(clampY(s.y[k] ?? 0)) };
    });
    raw.sort((a, b) => a.y - b.y);
    for (let k = 1; k < raw.length; k++) if (raw[k]!.y - raw[k - 1]!.y < 13) raw[k]!.y = raw[k - 1]!.y + 13;
    return raw;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, ys[0], ys[1]]);

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * iw + M.l;
    let best = 0;
    x.forEach((v, k) => {
      if (Math.abs(sx(v) - px) < Math.abs(sx(x[best]!) - px)) best = k;
    });
    setHover(best);
  };

  const tip = hover !== null ? { k: hover, left: sx(x[hover]!) > W / 2 } : null;

  return (
    <figure className="obs-chart">
      <div className="obs-chart-box" ref={box}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={`${id}-c`}>
          <title id={`${id}-c`}>{caption}</title>
          {ticks(ys[0], ys[1]).map((t) => (
            <g key={`y${t}`}>
              <line className="obs-grid" x1={M.l} x2={M.l + iw} y1={sy(t)} y2={sy(t)} />
              <text className="obs-tick" x={M.l - 6} y={sy(t) + 3.5} textAnchor="end">
                {fmtY(t).replace('-', '−')}
              </text>
            </g>
          ))}
          {ticks(xs[0]!, xs[1]!, 6).map((t) => (
            <text key={`x${t}`} className="obs-tick" x={sx(t)} y={M.t + ih + 16} textAnchor="middle">
              {fmtX(t)}
            </text>
          ))}
          <line className="obs-axis" x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} />
          {zero && ys[0] < 0 && ys[1] > 0 && <line className="obs-zero" x1={M.l} x2={M.l + iw} y1={sy(0)} y2={sy(0)} />}
          {marks.map((m) => (
            <g key={m.label}>
              <line className="obs-mark" x1={sx(m.x)} x2={sx(m.x)} y1={M.t} y2={M.t + ih} />
              <text className="obs-tick" x={sx(m.x) + 4} y={M.t + 10}>
                {m.label}
              </text>
            </g>
          ))}
          <text className="obs-axlab" x={M.l + iw / 2} y={H - 6} textAnchor="middle">
            {xLabel}
          </text>
          <text className="obs-axlab" transform={`translate(13 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">
            {yLabel}
          </text>
          {series.map((s, i) => (
            <path key={s.key} d={paths[i]} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
          ))}
          {labels.map((l) => (
            <text key={series[l.i]!.key} className="obs-dlabel" x={M.l + iw + 6} y={l.y + 3.5}>
              <tspan fill={series[l.i]!.color}>■</tspan> {series[l.i]!.short ?? series[l.i]!.name}
            </text>
          ))}
          {tip && (
            <g pointerEvents="none">
              <line className="obs-cross" x1={sx(x[tip.k]!)} x2={sx(x[tip.k]!)} y1={M.t} y2={M.t + ih} />
              {series.map((s) =>
                s.y[tip.k] !== null && Number.isFinite(s.y[tip.k]!) ? (
                  <circle key={s.key} cx={sx(x[tip.k]!)} cy={sy(clampY(s.y[tip.k]!))} r={4} fill={s.color} className="obs-dot" />
                ) : null,
              )}
            </g>
          )}
          <rect x={M.l} y={M.t} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {tip && (
          <div
            className="obs-tip"
            style={{
              left: tip.left ? undefined : `${(sx(x[tip.k]!) / W) * 100 + 2}%`,
              right: tip.left ? `${(1 - sx(x[tip.k]!) / W) * 100 + 2}%` : undefined,
            }}
          >
            <strong>
              {xLabel.split(' (')[0]}: {fmtX(x[tip.k]!)}
            </strong>
            {series.map((s) => (
              <span key={s.key}>
                <i style={{ background: s.color }} /> {s.name}: {s.y[tip.k] === null || !Number.isFinite(s.y[tip.k]!) ? '—' : fmtY(s.y[tip.k]!)}
              </span>
            ))}
          </div>
        )}
      </div>
      <details className="obs-table">
        <summary>Table view</summary>
        <div className="obs-table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">{xLabel}</th>
                {series.map((s) => (
                  <th key={s.key} scope="col">
                    {s.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {x.map((v, k) => (
                <tr key={k}>
                  <th scope="row">{fmtX(v)}</th>
                  {series.map((s) => (
                    <td key={s.key}>{s.y[k] === null || !Number.isFinite(s.y[k]!) ? '—' : fmtY(s.y[k]!)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
