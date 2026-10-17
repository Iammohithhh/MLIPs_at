import { describe, expect, it } from 'vitest';
import raw from '~/data/representation/map.json';
import { atomCount, bounds, elementClass, nnOverlap, pairOf, projection, type RepresentationMap } from './representation';

const map = raw as unknown as RepresentationMap;
const n = map.environments.length;

describe('Representation Map data', () => {
  it('has one point per environment in every projection of every model', () => {
    for (const m of map.models) {
      expect(m.raw.xy.length).toBe(n);
      for (const k of Object.keys(map.anchors)) expect(projection(m, k).xy.length).toBe(n);
    }
  });
  it('accounts for every atom of every structure exactly once', () => {
    expect(atomCount(map)).toBe(map.structures.reduce((t, s) => t + s.symbols.length, 0));
    for (const e of map.environments) expect(map.structures[e.s]!.symbols[e.i]).toBe(e.el);
  });
  it('lists ten distinct neighbours per environment, never the environment itself', () => {
    for (const m of map.models)
      m.nn.forEach((row, e) => {
        expect(row.length).toBe(map.source.k_nn);
        expect(new Set(row).size).toBe(row.length);
        expect(row).not.toContain(e);
        for (const j of row) expect(j).toBeLessThan(n);
      });
  });
  it('uses valid, shared anchors and aligns every map to the medium model', () => {
    for (const [k, a] of Object.entries(map.anchors)) {
      expect(a.length).toBe(Number(k));
      for (const i of a) expect(i).toBeLessThan(n);
    }
    expect(map.models[0]!.id).toBe('medium');
    for (const m of map.models) for (const p of Object.values(m.anchored)) expect(p.procrustes).toBeLessThanOrEqual(1 + 1e-9);
  });
  it('classifies every element on the map', () => {
    for (const e of map.environments) expect(elementClass(e.el)).not.toBeNull();
  });
  it('shows the trained models closer to each other than to the untrained one (the page’s reading)', () => {
    const ms = pairOf(map, 'medium', 'small')!;
    expect(ms.cka).toBeGreaterThan(pairOf(map, 'medium', 'random')!.cka);
    expect(ms.cka).toBeGreaterThan(pairOf(map, 'small', 'random')!.cka);
    const small = map.models.find((m) => m.id === 'small')!;
    const random = map.models.find((m) => m.id === 'random')!;
    for (const k of Object.keys(map.anchors)) expect(small.anchored[k]!.procrustes!).toBeGreaterThan(random.anchored[k]!.procrustes!);
  });
  it('computes neighbour overlaps and bounds', () => {
    const [a, b] = map.models;
    expect(nnOverlap(a!, a!, 0)).toBe(1);
    const o = nnOverlap(a!, b!, 5);
    expect(o).toBeGreaterThanOrEqual(0);
    expect(o).toBeLessThanOrEqual(1);
    expect(bounds([[1, 2], [-3, 5]])).toEqual({ x0: -3, x1: 1, y0: 2, y1: 5 });
  });
});
