// Geometry for the content-model diagram: one arrow per (from, to) collection
// pair, labelled with every schema field that makes the reference.

import { COLLECTION_META, NODE } from './meta';
import type { CollectionName, Relation } from './schemas';

export interface DiagramEdge {
  from: CollectionName;
  to: CollectionName;
  fields: string[];
  d: string;
  self: boolean;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

const box = (c: CollectionName): Box => ({ ...COLLECTION_META[c].pos, w: NODE.w, h: NODE.h });
const center = (b: Box) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

/** Where the ray from the box centre in direction (dx, dy) leaves the box (plus a gap). */
export function exitPoint(b: Box, dx: number, dy: number, gap = 4): { x: number; y: number } {
  const c = center(b);
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const tx = ux === 0 ? Infinity : b.w / 2 / Math.abs(ux);
  const ty = uy === 0 ? Infinity : b.h / 2 / Math.abs(uy);
  const t = Math.min(tx, ty) + gap;
  return { x: c.x + ux * t, y: c.y + uy * t };
}

export function groupRelations(relations: Relation[]): Array<{ from: CollectionName; to: CollectionName; fields: string[] }> {
  const map = new Map<string, { from: CollectionName; to: CollectionName; fields: string[] }>();
  for (const r of relations) {
    const k = `${r.from}→${r.to}`;
    const g = map.get(k) ?? { from: r.from, to: r.to, fields: [] };
    if (!g.fields.includes(r.field)) g.fields.push(r.field);
    map.set(k, g);
  }
  return [...map.values()];
}

export function diagramEdges(relations: Relation[]): DiagramEdge[] {
  const groups = groupRelations(relations);
  const has = (a: string, b: string) => groups.some((g) => g.from === a && g.to === b);
  return groups.map((g) => {
    if (g.from === g.to) {
      const b = box(g.from);
      const x0 = b.x + b.w - 34;
      const y1 = b.y + 18;
      const d = `M${x0},${b.y - 3} C${x0 + 6},${b.y - 34} ${b.x + b.w + 34},${b.y - 10} ${b.x + b.w + 4},${y1}`;
      return { ...g, d, self: true };
    }
    const A = box(g.from);
    const B = box(g.to);
    const ca = center(A);
    const cb = center(B);
    const dx = cb.x - ca.x;
    const dy = cb.y - ca.y;
    const dist = Math.hypot(dx, dy);
    // Separate the two directions of a mutual pair; bow long edges away from nodes in between.
    const bend = (has(g.to, g.from) ? 22 : 0) + (dist > 420 ? 46 : 0);
    const nx = -dy / dist;
    const ny = dx / dist;
    const mid = { x: (ca.x + cb.x) / 2 + nx * bend, y: (ca.y + cb.y) / 2 + ny * bend };
    const p0 = exitPoint(A, mid.x - ca.x, mid.y - ca.y);
    const p1 = exitPoint(B, mid.x - cb.x, mid.y - cb.y, 7);
    const d = `M${p0.x.toFixed(1)},${p0.y.toFixed(1)} Q${mid.x.toFixed(1)},${mid.y.toFixed(1)} ${p1.x.toFixed(1)},${p1.y.toFixed(1)}`;
    return { ...g, d, self: false };
  });
}
