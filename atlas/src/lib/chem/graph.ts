// Cutoff graph and message-passing reach on a (partly) periodic structure.
// Edges join atoms closer than r_c (minimum image on periodic axes); after T
// message-passing layers atom i can be influenced only by atoms within T hops.

import { minImage, type Structure } from './structures';

export interface Adjacency {
  /** CSR offsets: neighbours of i are indices[offsets[i]..offsets[i+1]). */
  offsets: Int32Array;
  indices: Int32Array;
}

/** All pairs closer than rc. O(N²); fine for the few thousand atoms of a teaching scene. */
export function cutoffGraph(s: Structure, rc: number): Adjacency {
  const n = s.atoms.length;
  const lists: number[][] = Array.from({ length: n }, () => []);
  const rc2 = rc * rc;
  for (let i = 0; i < n; i++) {
    const pi = s.atoms[i]!.p;
    for (let j = i + 1; j < n; j++) {
      const d = minImage(pi, s.atoms[j]!.p, s.cell, s.pbc);
      if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < rc2) {
        lists[i]!.push(j);
        lists[j]!.push(i);
      }
    }
  }
  const offsets = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i]! + lists[i]!.length;
  const indices = new Int32Array(offsets[n]!);
  lists.forEach((l, i) => indices.set(l, offsets[i]!));
  return { offsets, indices };
}

/** Breadth-first hop counts from `start`, up to `maxHops`; −1 means not reached. */
export function hopDistances(g: Adjacency, start: number, maxHops: number): Int16Array {
  const n = g.offsets.length - 1;
  const hops = new Int16Array(n).fill(-1);
  hops[start] = 0;
  let frontier = [start];
  for (let h = 1; h <= maxHops && frontier.length; h++) {
    const next: number[] = [];
    for (const u of frontier) {
      for (let e = g.offsets[u]!; e < g.offsets[u + 1]!; e++) {
        const v = g.indices[e]!;
        if (hops[v] === -1) {
          hops[v] = h;
          next.push(v);
        }
      }
    }
    frontier = next;
  }
  return hops;
}

/** Number of atoms (excluding i) reached within exactly ≤ t hops, for t = 1..maxHops. */
export function reachCounts(hops: Int16Array, maxHops: number): number[] {
  const counts = new Array<number>(maxHops).fill(0);
  for (const h of hops) if (h > 0) for (let t = h; t <= maxHops; t++) counts[t - 1]!++;
  return counts;
}

/** Atoms (excluding i) closer than R to atom i (minimum image). */
export function countWithin(s: Structure, i: number, R: number): number {
  let c = 0;
  const pi = s.atoms[i]!.p;
  s.atoms.forEach((a, j) => {
    if (j === i) return;
    const d = minImage(pi, a.p, s.cell, s.pbc);
    if (Math.hypot(d[0], d[1], d[2]) < R) c++;
  });
  return c;
}
