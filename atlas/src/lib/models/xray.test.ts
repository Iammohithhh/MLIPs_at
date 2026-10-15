import { describe, expect, it } from 'vitest';
import raw from '~/data/xray/mace-mp-0a-medium.json';
import {
  channelMatrix,
  channelValues,
  channelVectors,
  dataflowShapes,
  energyParts,
  incomingEdges,
  rotatedView,
  rotationCheck,
  tensorLs,
  type XrayDoc,
} from './xray';

const doc = raw as unknown as XrayDoc;
const sys = (id: string) => doc.systems.find((s) => s.id === id)!;

describe('X-ray data provenance', () => {
  it('names the checkpoint and its hash and is labelled PRECOMPUTED', () => {
    expect(doc.status).toBe('PRECOMPUTED');
    expect(doc.model.checkpoint).toBe('mace-mp-0a-medium');
    expect(doc.model.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(doc.model.hyperparameters.r_max).toBe(6);
    expect(doc.model.hyperparameters.num_interactions).toBe(2);
  });
});

describe('energy bookkeeping', () => {
  it.each(['ethanol', 'zno-wurtzite', 'si-diamond'])('%s: per-atom parts sum to the total energy', (id) => {
    const s = sys(id);
    const parts = energyParts(doc, s);
    parts.forEach((p, i) => expect(p.total).toBeCloseTo(s.nodeEnergy[i]!, 3));
    expect(parts.reduce((a, p) => a + p.total, 0)).toBeCloseTo(s.energy, 2);
  });
});

describe('symmetry in the real network', () => {
  it('rotating ethanol leaves l = 0 features unchanged and co-rotates l = 1 features', () => {
    const r = rotationCheck(sys('ethanol'))!;
    expect(r.dE).toBeLessThan(1e-9);
    expect(r.forces).toBeLessThan(1e-3);
    expect(r.layers[0]!.l0).toBeLessThan(1e-3);
    expect(r.layers[1]!.l0).toBeLessThan(1e-3);
    expect(r.layers[0]!.hl1!).toBeLessThan(2e-3);
    expect(r.layers[0]!.Al1!).toBeLessThan(2e-3);
  });

  it('the rotated pass is a complete system of its own with the same energy split', () => {
    const e = sys('ethanol');
    const r = rotatedView(e)!;
    const a = energyParts(doc, e);
    const b = energyParts(doc, r);
    b.forEach((p, i) => expect(p.total).toBeCloseTo(a[i]!.total, 3));
    expect(r.positions).not.toEqual(e.positions);
    expect(rotatedView(sys('si-diamond'))).toBeNull();
  });

  it('diamond: every l = 1 and l = 2 entry of the first message is exactly zero', () => {
    const s = sys('si-diamond');
    for (const l of [1, 2]) for (let c = 0; c < 128; c += 7) expect(Math.max(...channelValues(s, 'A1', l, c)!)).toBe(0);
    expect(Math.max(...channelMatrix(s, 'A1', 3)!.flat())).toBeGreaterThan(0.1);
    // All eight atoms are equivalent: identical invariant features.
    const m = channelMatrix(s, 'h2', 0)!;
    for (const row of m) row.forEach((x, c) => expect(x).toBeCloseTo(m[0]![c]!, 3));
  });

  it('the second message carries no l > 0 information in this checkpoint', () => {
    for (const s of doc.systems) {
      expect(tensorLs(s, 'A2').map((b) => b.live)).toEqual([true, false, false, false]);
      expect(s.layers[1]!.A.peak.slice(1).every((p) => p < 1e-80)).toBe(true);
      expect(channelVectors(s, 'A2', 0)).toBeNull();
    }
  });
});

describe('graph and shapes', () => {
  it('edges are within the cutoff and incoming edges are listed nearest first', () => {
    const s = sys('zno-wurtzite');
    const inc = incomingEdges(s, 0);
    expect(inc.length).toBeGreaterThan(0);
    inc.forEach((e, k) => {
      expect(e.r).toBeLessThan(doc.model.hyperparameters.r_max);
      if (k > 0) expect(e.r).toBeGreaterThanOrEqual(inc[k - 1]!.r);
    });
    // Wurtzite: four nearest neighbours of Zn are O at ≈ 1.98 Å.
    expect(inc.slice(0, 4).every((e) => s.symbols[e.j] === 'O' && Math.abs(e.r - 1.98) < 0.03)).toBe(true);
  });

  it('dataflow shapes follow the hyperparameters', () => {
    const s = sys('ethanol');
    const sh = dataflowShapes(doc, s);
    expect(sh.A1).toBe('[9, 128, 16]');
    expect(sh.h1).toBe('[9, 512]');
    expect(sh.h2).toBe('[9, 128]');
    expect(sh.radial).toBe(`[${s.edges.count}, 10]`);
  });
});
