// Layout for the Papers chronology: x = publication year on a scale that
// compresses the early decades, one horizontal lane per topic, collisions
// resolved by stacking within a lane. Pure and tested.

export interface GraphPaper {
  id: string;
  year: number;
  lane: number;
  label: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  kind: 'cites' | 'typed';
  label?: string;
}

export interface PlacedPaper extends GraphPaper {
  x: number;
  y: number;
  /** Label baseline y, or null when it would collide (shown on hover instead). */
  labelY: number | null;
}

export interface GraphLayout {
  width: number;
  height: number;
  nodes: PlacedPaper[];
  edges: Array<GraphEdge & { d: string }>;
  laneY: number[];
  ticks: Array<{ year: number; x: number }>;
}

export const NODE_R = 7;

export function yearScale(width: number, left: number, right: number) {
  const breakYear = 2000;
  const start = 1915;
  const end = 2027;
  const breakX = left + (width - left - right) * 0.24;
  return (year: number) =>
    year <= breakYear
      ? left + ((year - start) / (breakYear - start)) * (breakX - left)
      : breakX + ((year - breakYear) / (end - breakYear)) * (width - right - breakX);
}

export function layoutPapers(papers: GraphPaper[], edges: GraphEdge[], laneCount: number): GraphLayout {
  const width = 1040;
  const laneH = 74;
  const top = 34;
  const left = 150;
  const right = 24;
  const height = top + laneCount * laneH + 34;
  const x = yearScale(width, left, right);
  const laneY = Array.from({ length: laneCount }, (_, k) => top + k * laneH + laneH / 2);

  const nodes: PlacedPaper[] = [];
  const minGap = 2 * NODE_R + 6;
  for (let lane = 0; lane < laneCount; lane++) {
    const inLane = papers.filter((p) => p.lane === lane).sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
    const placed: PlacedPaper[] = [];
    for (const p of inLane) {
      const px = x(p.year);
      // Try rows 0, −1, +1, −2, +2 around the lane, then shift right a column; first free slot wins.
      search: for (let col = 0; col < 8; col++) {
        for (let k = 0; k < 5; k++) {
          const off = (k % 2 === 0 ? 1 : -1) * Math.ceil(k / 2) * minGap;
          const node: PlacedPaper = { ...p, x: px + col * minGap * 0.8, y: laneY[lane]! + off, labelY: null };
          if (placed.every((q) => Math.hypot(q.x - node.x, q.y - node.y) >= minGap)) {
            placed.push(node);
            nodes.push(node);
            break search;
          }
        }
      }
    }
  }

  // Labels: try above, then below each node; keep only those that collide with no other label or node.
  const boxes: Array<{ x0: number; x1: number; y0: number; y1: number }> = [];
  const nodeBoxes = nodes.map((n) => ({ x0: n.x - NODE_R, x1: n.x + NODE_R, y0: n.y - NODE_R, y1: n.y + NODE_R }));
  const hit = (b: { x0: number; x1: number; y0: number; y1: number }) =>
    [...boxes, ...nodeBoxes].some((o) => b.x0 < o.x1 && o.x0 < b.x1 && b.y0 < o.y1 && o.y0 < b.y1);
  for (const n of [...nodes].sort((a, b) => a.x - b.x)) {
    const w = n.label.length * 5.6 + 4;
    for (const ly of [n.y - NODE_R - 4, n.y + NODE_R + 12]) {
      const b = { x0: n.x - w / 2, x1: n.x + w / 2, y0: ly - 10, y1: ly + 2 };
      if (b.x0 > left - 6 && b.x1 < width && !hit(b)) {
        n.labelY = ly;
        boxes.push(b);
        break;
      }
    }
  }

  const at = new Map(nodes.map((n) => [n.id, n]));
  const placedEdges = edges
    .filter((e) => at.has(e.from) && at.has(e.to))
    .map((e) => {
      const a = at.get(e.from)!;
      const b = at.get(e.to)!;
      // Arc bulging upwards for citations, downwards for typed relations.
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2 + (e.kind === 'cites' ? -1 : 1) * Math.min(60, 12 + Math.abs(a.x - b.x) * 0.12);
      return { ...e, d: `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}` };
    });

  const ticks = [1920, 1940, 1960, 1980, 2000, 2005, 2010, 2015, 2020, 2025].map((year) => ({ year, x: x(year) }));
  return { width, height, nodes, edges: placedEdges, laneY, ticks };
}
