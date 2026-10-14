import { describe, expect, it } from 'vitest';
import { RF, atomPosition, cropAround, labelOnCircle, rfModel, rfReadout, rfState } from './receptive-field-figure';
import { distance, nearestAtom } from './toy-lattice';

const model = rfModel();
const center = nearestAtom(model.atoms, RF.width / 2, RF.height / 2);

describe('homepage receptive-field model', () => {
  it('fits the lattice inside the figure', () => {
    for (const a of model.atoms) {
      expect(a.x).toBeGreaterThan(RF.atomRadius);
      expect(a.y).toBeGreaterThan(RF.atomRadius);
      expect(a.x).toBeLessThan(RF.width - RF.atomRadius);
      expect(a.y).toBeLessThan(RF.height - RF.atomRadius);
    }
  });

  it('local view = atoms strictly within r_c', () => {
    const s = rfState(model, center, 'local', 3);
    const c = model.atoms[center]!;
    const within = model.atoms.filter((a, j) => j !== center && distance(a, c) < model.rc).length;
    expect(s.layers).toBe(1);
    expect(s.inField).toBe(within);
    expect(s.longRange).toHaveLength(0);
  });

  it('message passing grows monotonically and never exceeds the T·r_c disc', () => {
    let prev = 0;
    for (let T = 1; T <= RF.maxLayers; T++) {
      const s = rfState(model, center, 'mp', T);
      expect(s.inField).toBeGreaterThan(prev);
      expect(s.inField).toBeLessThanOrEqual(s.inBall);
      prev = s.inField;
    }
  });

  it('every atom in the hop field is inside the T·r_c disc (triangle inequality)', () => {
    const T = 3;
    const s = rfState(model, center, 'mp', T);
    const c = model.atoms[center]!;
    s.hops.forEach((h, j) => {
      if (h <= T) expect(distance(model.atoms[j]!, c)).toBeLessThan(T * model.rc);
    });
  });

  it('long-range view couples to every atom beyond r_c, attraction ⇔ opposite charges', () => {
    const s = rfState(model, center, 'long', 1);
    const local = rfState(model, center, 'local', 1);
    expect(s.longRange.length + local.inField + 1).toBe(model.atoms.length);
    for (const l of s.longRange) {
      expect(l.attractive).toBe(model.atoms[l.j]!.q !== model.atoms[center]!.q);
      expect(l.opacity).toBeGreaterThan(0);
      expect(l.opacity).toBeLessThanOrEqual(0.85);
    }
  });

  it('readouts quote the computed counts', () => {
    const s = rfState(model, center, 'mp', 2);
    expect(rfReadout(s, 'mp', model.atoms.length)).toContain(`${s.inField} atoms`);
  });

  it('keeps circle labels inside the figure', () => {
    for (const [x, y, r] of [
      [10, 10, 60],
      [RF.width / 2, RF.height / 2, 260],
      [RF.width - 5, RF.height - 5, 120],
    ] as const) {
      const p = labelOnCircle(x, y, r);
      expect(p.y).toBeGreaterThanOrEqual(8);
      expect(p.y).toBeLessThanOrEqual(RF.height);
    }
  });
});

describe('long-range readout and narrow-screen crop', () => {
  it('counts atoms beyond r_c that T layers can still reach', () => {
    const T = 2;
    const s = rfState(model, center, 'long', T);
    const mp = rfState(model, center, 'mp', T);
    const local = rfState(model, center, 'local', T);
    // Reachable within T hops minus those already inside r_c.
    expect(s.beyondReachable).toBe(mp.inField - local.inField);
    expect(rfReadout(s, 'long', model.atoms.length)).toContain(`${s.beyondReachable} are reachable`);
    expect(rfReadout(s, 'long', model.atoms.length)).toContain('strictly local model sees none');
  });

  it('crops a square window around atom i, clamped to the figure', () => {
    expect(cropAround(0)).toEqual({ x0: 0, x1: RF.height });
    expect(cropAround(RF.width)).toEqual({ x0: RF.width - RF.height, x1: RF.width });
    const mid = cropAround(RF.width / 2);
    expect(mid.x1 - mid.x0).toBe(RF.height);
  });

  it('places labels inside the visible window', () => {
    const view = cropAround(100);
    const p = labelOnCircle(100, RF.height / 2, 200, view);
    expect(p.x).toBeGreaterThanOrEqual(view.x0);
    expect(p.x).toBeLessThanOrEqual(view.x1);
  });

  it('describes atom i by grid position and charge', () => {
    expect(atomPosition(0, 1)).toBe(`Atom i: row 1 of ${RF.ny}, column 1 of ${RF.nx}, positive ion.`);
    expect(atomPosition(RF.nx + 2, -1)).toContain('row 2');
  });
});
