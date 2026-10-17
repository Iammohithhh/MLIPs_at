import { describe, expect, it } from 'vitest';
import raw from '~/data/representation/map.json';
import { atomCount, bounds, elementClass, logPosition, nnOverlap, pairMatrix, pairOf, projection, type RepresentationMap } from './representation';

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
  it('builds a symmetric agreement matrix with ones on the diagonal', () => {
    for (const metric of ['cka', 'mknn_raw', 'mknn_100'] as const) {
      const M = pairMatrix(map, metric);
      expect(M.length).toBe(map.models.length);
      M.forEach((row, i) => {
        expect(row[i]).toBe(1);
        row.forEach((v, j) => expect(v).toBe(M[j]![i]));
      });
    }
    expect(logPosition(0, -6, 0)).toBe(0);
    expect(logPosition(1e-3, -6, 0)).toBeCloseTo(0.5);
    expect(logPosition(50, -6, 0)).toBe(1);
  });
  it('has invariance data for every environment, and an exactly invariant architecture never fails it badly', () => {
    for (const m of map.models) {
      expect(m.invariance.rot_env.length).toBe(n);
      expect(m.invariance.equiv_env.length).toBe(n);
      expect(m.invariance.ref_env.length).toBe(n);
      m.invariance.equiv_env.forEach((v, e) => {
        if (map.environments[e]!.mult === 1) expect(v).toBe(0);
      });
    }
    // MACE's scalars are O(3)-invariant by construction: rotation and reflection residuals are float64 rounding.
    // Equivalence is decided on distances rounded to 1e-3 Å, so near-symmetric G2 molecules leave ~1e-5.
    for (const id of ['medium', 'small', 'random']) {
      const inv = map.models.find((m) => m.id === id)!.invariance;
      expect(inv.rot.max).toBeLessThan(1e-9);
      expect(inv.ref.max).toBeLessThan(1e-9);
      expect(inv.equiv.max).toBeLessThan(1e-4);
    }
  });
  it('supports the page’s reading of the cross-architecture comparison and the invariance tests', () => {
    const by = (id: string) => map.models.find((m) => m.id === id)!;
    const cka = (a: string, b: string) => pairOf(map, a, b)!.cka;
    // Same data, ordered by architecture: MACE small > SevenNet > CHGNet > Orb in CKA with MACE medium.
    expect(cka('medium', 'small')).toBeGreaterThan(cka('medium', 'sevennet'));
    expect(cka('medium', 'sevennet')).toBeGreaterThan(cka('medium', 'chgnet'));
    expect(cka('medium', 'chgnet')).toBeGreaterThan(cka('medium', 'orb'));
    // Orb is furthest from every other trained model, and below the untrained MACE against medium.
    for (const id of ['medium', 'small', 'chgnet', 'sevennet'])
      for (const other of ['medium', 'small', 'chgnet', 'sevennet'].filter((x) => x !== id)) expect(cka(id, 'orb')).toBeLessThan(cka(id, other));
    expect(cka('medium', 'orb')).toBeLessThan(cka('medium', 'random'));
    // CHGNet: invariant to rounding in all three tests (float32).
    const c = by('chgnet').invariance;
    for (const k of ['rot', 'ref', 'equiv'] as const) expect(c[k].max).toBeLessThan(0.01);
    // SevenNet-0: rotations to rounding, mirror images not.
    const s7 = by('sevennet').invariance;
    expect(s7.rot.max).toBeLessThan(0.01);
    expect(s7.ref.max).toBeGreaterThan(0.03);
    // Orb v2: fails rotation at the scale of chemistry.
    const o = by('orb').invariance;
    expect(o.rot.median).toBeGreaterThan(0.1);
    expect(o.rot.max).toBeGreaterThan(1);
    expect(o.equiv.max).toBeGreaterThan(0.5);
  });
});
