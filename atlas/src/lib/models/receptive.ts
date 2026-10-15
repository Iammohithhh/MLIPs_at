// Receptive field of message passing in a real crystal: how many atoms can influence
// the central atom after T layers of a cutoff-r_c graph. Computed on a large cluster
// carved from the periodic structure (big enough that the boundary is never reached).

import type { Structure, Vec3 } from '../structure/types';
import { neighboursOf } from '../structure/geometry';

/** Centre atom first, then every atom within R of it (periodic images included). */
export function clusterAround(s: Structure, R: number, centre = 0): Vec3[] {
  const c = s.positions[centre]!;
  return [[0, 0, 0], ...neighboursOf(s, centre, R).map((n) => [n.position[0] - c[0], n.position[1] - c[1], n.position[2] - c[2]] as Vec3)];
}

/** Hop distance from atom 0 over the graph with edges r < rc (−1 = not reached within maxHops). */
export function hopsFromCentre(pos: Vec3[], rc: number, maxHops: number): Int16Array {
  const n = pos.length;
  // Cell list for the neighbour search.
  const cell = rc;
  const key = (p: Vec3) => `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)},${Math.floor(p[2] / cell)}`;
  const grid = new Map<string, number[]>();
  pos.forEach((p, i) => {
    const k = key(p);
    const list = grid.get(k);
    if (list) list.push(i);
    else grid.set(k, [i]);
  });
  const rc2 = rc * rc;
  const neighbours = (i: number): number[] => {
    const p = pos[i]!;
    const cx = Math.floor(p[0] / cell);
    const cy = Math.floor(p[1] / cell);
    const cz = Math.floor(p[2] / cell);
    const out: number[] = [];
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++)
        for (let c = -1; c <= 1; c++) {
          for (const j of grid.get(`${cx + a},${cy + b},${cz + c}`) ?? []) {
            if (j === i) continue;
            const q = pos[j]!;
            const d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2;
            if (d2 < rc2) out.push(j);
          }
        }
    return out;
  };
  const hops = new Int16Array(n).fill(-1);
  hops[0] = 0;
  let frontier = [0];
  for (let h = 1; h <= maxHops && frontier.length; h++) {
    const next: number[] = [];
    for (const u of frontier)
      for (const v of neighbours(u))
        if (hops[v] === -1) {
          hops[v] = h;
          next.push(v);
        }
    frontier = next;
  }
  return hops;
}

/** Atoms (excluding the centre) reachable within T hops, for T = 1..maxHops. */
export function reachByDepth(hops: Int16Array, maxHops: number): number[] {
  const out = new Array<number>(maxHops).fill(0);
  for (const h of hops) if (h > 0) for (let t = h; t <= maxHops; t++) out[t - 1]!++;
  return out;
}
