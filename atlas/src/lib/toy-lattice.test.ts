import { describe, expect, it } from 'vitest';
import {
  cutoffGraph,
  distance,
  hopDistances,
  makeIonicLayer,
  mulberry32,
  nearestAtom,
  receptiveField,
} from './toy-lattice';

describe('mulberry32', () => {
  it('is deterministic and in [0, 1)', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 1000; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe('makeIonicLayer', () => {
  const atoms = makeIonicLayer({ nx: 6, ny: 4, a: 1, jitter: 0, seed: 1 });

  it('builds nx·ny sites on a square lattice', () => {
    expect(atoms).toHaveLength(24);
    expect(atoms[7]).toMatchObject({ x: 1, y: 1 });
  });

  it('is charge neutral with checkerboard signs (nearest neighbours opposite)', () => {
    expect(atoms.reduce((s, a) => s + a.q, 0)).toBe(0);
    expect(atoms[0]!.q).toBe(-atoms[1]!.q);
    expect(atoms[0]!.q).toBe(-atoms[6]!.q);
    expect(atoms[0]!.q).toBe(atoms[7]!.q);
  });

  it('keeps jitter within the requested amplitude', () => {
    const j = makeIonicLayer({ nx: 10, ny: 10, a: 2, jitter: 0.1, seed: 3 });
    j.forEach((p, k) => {
      expect(Math.abs(p.x - (k % 10) * 2)).toBeLessThanOrEqual(0.2 + 1e-12);
      expect(Math.abs(p.y - Math.floor(k / 10) * 2)).toBeLessThanOrEqual(0.2 + 1e-12);
    });
  });
});

describe('cutoff graph and hops on a perfect square lattice', () => {
  const atoms = makeIonicLayer({ nx: 9, ny: 9, a: 1, jitter: 0, seed: 0 });
  const center = nearestAtom(atoms, 4, 4);

  it('finds the central site', () => {
    expect(atoms[center]).toMatchObject({ x: 4, y: 4 });
  });

  it('rc = 1.5a connects the 4 first and 4 second neighbours', () => {
    const adj = cutoffGraph(atoms, 1.5);
    expect(adj[center]).toHaveLength(8);
    adj[center]!.forEach((j) => expect(distance(atoms[center]!, atoms[j]!)).toBeLessThan(1.5));
  });

  it('graph hops equal the Chebyshev distance when diagonals are edges', () => {
    const adj = cutoffGraph(atoms, 1.5);
    const hops = hopDistances(adj, center);
    atoms.forEach((a, i) => {
      expect(hops[i]).toBe(Math.max(Math.abs(a.x - 4), Math.abs(a.y - 4)));
    });
  });

  it('receptive field after T layers is a (2T+1)² block minus the centre', () => {
    const adj = cutoffGraph(atoms, 1.5);
    const hops = hopDistances(adj, center);
    for (const T of [1, 2, 3]) {
      const rf = receptiveField(atoms, adj, hops, center, 1.5, T);
      expect(rf.inField).toBe((2 * T + 1) ** 2 - 1);
      // The ball of radius T·rc is never smaller than the hop field.
      expect(rf.inBall).toBeGreaterThanOrEqual(rf.inField);
    }
  });

  it('flow edges always point one hop closer to the centre', () => {
    const adj = cutoffGraph(atoms, 1.5);
    const hops = hopDistances(adj, center);
    const rf = receptiveField(atoms, adj, hops, center, 1.5, 2);
    expect(rf.flowEdges.length).toBeGreaterThan(0);
    rf.flowEdges.forEach(([u, v]) => {
      expect(hops[v]).toBe(hops[u]! - 1);
      expect(hops[u]).toBeLessThanOrEqual(2);
    });
  });

  it('marks disconnected atoms as unreachable', () => {
    const adj = cutoffGraph(atoms, 0.5);
    const hops = hopDistances(adj, center);
    expect(hops.filter((h) => h === Infinity)).toHaveLength(80);
  });
});
