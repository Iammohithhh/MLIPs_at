// A 2D toy "ionic layer" used by the homepage receptive-field figure (TOY MODEL).
// Square lattice with alternating ±1 charges (a rock-salt-like checkerboard) and
// deterministic thermal-like jitter. Everything here is pure and unit-tested.

export interface Atom2D {
  x: number;
  y: number;
  /** Toy formal charge, +1 or -1. */
  q: 1 | -1;
}

export interface LatticeOptions {
  nx: number;
  ny: number;
  /** Lattice spacing in figure units. */
  a: number;
  /** Jitter amplitude as a fraction of a (uniform in [-j, j] per coordinate). */
  jitter: number;
  seed: number;
  /** Offset of the first site. */
  origin?: { x: number; y: number };
}

/** Mulberry32: small, fast, deterministic PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeIonicLayer(opts: LatticeOptions): Atom2D[] {
  const rand = mulberry32(opts.seed);
  const ox = opts.origin?.x ?? 0;
  const oy = opts.origin?.y ?? 0;
  const atoms: Atom2D[] = [];
  for (let j = 0; j < opts.ny; j++) {
    for (let i = 0; i < opts.nx; i++) {
      const jx = (rand() * 2 - 1) * opts.jitter * opts.a;
      const jy = (rand() * 2 - 1) * opts.jitter * opts.a;
      atoms.push({
        x: ox + i * opts.a + jx,
        y: oy + j * opts.a + jy,
        q: (i + j) % 2 === 0 ? 1 : -1,
      });
    }
  }
  return atoms;
}

export function distance(a: Atom2D, b: Atom2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Undirected graph with an edge for every pair closer than rc (open ball). */
export function cutoffGraph(atoms: Atom2D[], rc: number): number[][] {
  const adj: number[][] = atoms.map(() => []);
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      if (distance(atoms[i]!, atoms[j]!) < rc) {
        adj[i]!.push(j);
        adj[j]!.push(i);
      }
    }
  }
  return adj;
}

/** Breadth-first graph distance from `source`; unreachable nodes get Infinity. */
export function hopDistances(adj: number[][], source: number): number[] {
  const hops = adj.map(() => Infinity);
  hops[source] = 0;
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head]!;
    for (const v of adj[u]!) {
      if (hops[v] === Infinity) {
        hops[v] = hops[u]! + 1;
        queue.push(v);
      }
    }
  }
  return hops;
}

/** Index of the atom nearest to a point. */
export function nearestAtom(atoms: Atom2D[], x: number, y: number): number {
  let best = 0;
  let bestD = Infinity;
  atoms.forEach((a, i) => {
    const d = Math.hypot(a.x - x, a.y - y);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

export interface ReceptiveField {
  /** Atoms (excluding the centre) within T graph hops. */
  inField: number;
  /** Atoms (excluding the centre) inside the Euclidean ball of radius T·rc. */
  inBall: number;
  /** Message-carrying edges [from, to] with hop(to) = hop(from) − 1, toward the centre. */
  flowEdges: Array<[number, number]>;
}

/**
 * Receptive field of `center` after T message-passing layers on the cutoff graph.
 * With one hop per layer, information reaches the centre only from atoms within T
 * hops — always a subset of the ball of radius T·rc, often a strict subset.
 */
export function receptiveField(
  atoms: Atom2D[],
  adj: number[][],
  hops: number[],
  center: number,
  rc: number,
  T: number,
): ReceptiveField {
  const c = atoms[center]!;
  let inField = 0;
  let inBall = 0;
  const flowEdges: Array<[number, number]> = [];
  atoms.forEach((a, i) => {
    if (i === center) return;
    if (hops[i]! <= T) inField++;
    if (distance(a, c) < T * rc) inBall++;
  });
  adj.forEach((nbrs, u) => {
    const hu = hops[u]!;
    if (hu > T || hu === 0) return;
    for (const v of nbrs) {
      if (hops[v] === hu - 1) flowEdges.push([u, v]);
    }
  });
  return { inField, inBall, flowEdges };
}
