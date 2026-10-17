import { describe, expect, it } from 'vitest';
import quartz from '~/data/structures/sio2-alpha-quartz.json';
import si from '~/data/structures/si-diamond.json';
import type { Structure } from '~/lib/structure/types';
import {
  PAPER_BENCH,
  SEVENNET0,
  TOY_RC,
  decompose,
  decompositionCost,
  dimerCurves,
  domainEdge,
  envelope,
  evaluateNaive,
  evaluateSerial,
  evaluateSevenNet,
  forEachAtomInBox,
  makeSystem,
  makeToyModel,
  maxForceError,
  numberDensity,
  shellCounts,
  type System2D,
} from './sevennet';

const q = quartz as unknown as Structure;
const s = si as unknown as Structure;
const sys = makeSystem();

function shifted(base: System2D, i: number, axis: number, h: number): System2D {
  const pos = Float64Array.from(base.pos);
  pos[2 * i + axis]! += h;
  return { ...base, pos };
}

describe('SevenNet-0 settings', () => {
  it('carries 480 floats per atom between layers (128x0e + 64x1e + 32x2e)', () => {
    expect(SEVENNET0.featureFloats).toBe(480);
  });
});

describe('ghost shells in a crystal', () => {
  it('quartz has 9 atoms in 113.1 Å³ (2.65 g/cm³)', () => {
    expect(q.symbols.length).toBe(9);
    expect(numberDensity(q)).toBeCloseTo(9 / 113.131, 5);
    const gcm3 = (numberDensity(q) / 3) * (28.0855 + 2 * 15.999) * 1.66054;
    expect(gcm3).toBeCloseTo(2.65, 2);
  });

  it('enumerates the infinite crystal: a box of whole cubic cells holds whole cells of atoms', () => {
    let n = 0;
    forEachAtomInBox(s, [-0.01, -0.01, -0.01], [2 * 5.431 - 0.01, 2 * 5.431 - 0.01, 5.431 - 0.01], () => n++);
    expect(n).toBe(8 * 4);
  });

  it('owned count matches density × volume, and each shell adds atoms', () => {
    const L = domainEdge(q, PAPER_BENCH.atomsPerGpu);
    const c = shellCounts(q, L, PAPER_BENCH.rc, 5);
    expect(Math.abs(c.owned - PAPER_BENCH.atomsPerGpu) / PAPER_BENCH.atomsPerGpu).toBeLessThan(0.01);
    for (let k = 1; k <= 5; k++) expect(c.cum[k]!).toBeGreaterThan(c.cum[k - 1]!);
    // Each cumulative count is close to ρ (L + 2kr_c)³.
    const rho = numberDensity(q);
    for (let k = 0; k <= 5; k++) {
      const expect3 = rho * (L + 2 * k * PAPER_BENCH.rc) ** 3;
      expect(Math.abs(c.cum[k]! - expect3) / expect3).toBeLessThan(0.01);
    }
  });

  it('costs: SevenNet updates T × owned, the naive shell sums the shrinking boxes', () => {
    const c = shellCounts(q, 20, 5, 3);
    const k = decompositionCost(c, 3, 480);
    expect(k.updatesSeven).toBe(3 * c.owned);
    expect(k.updatesNaive).toBe(c.cum[2]! + c.cum[1]! + c.cum[0]!);
    expect(k.ghostsSeven).toBe(c.cum[1]! - c.owned);
    expect(k.commSeven.features).toBe(2 * 480 * k.ghostsSeven);
    expect(k.commSeven.messages).toBe(6);
    // One layer: both schemes are the same strictly local decomposition.
    const one = decompositionCost(c, 1, 480);
    expect(one.updatesNaive).toBe(one.updatesSeven);
    expect(one.ghostsNaive).toBe(one.ghostsSeven);
    expect(one.commSeven.features + one.commSeven.gradients).toBe(0);
  });
});

describe('toy message-passing network', () => {
  const m = makeToyModel(3);

  it('the envelope vanishes smoothly at r_c', () => {
    const [u, du] = envelope(TOY_RC - 1e-6);
    expect(Math.abs(u)).toBeLessThan(1e-10);
    expect(Math.abs(du)).toBeLessThan(1e-8);
    expect(envelope(0.5)[1]).toBeCloseTo((envelope(0.5 + 1e-6)[0] - envelope(0.5 - 1e-6)[0]) / 2e-6, 6);
  });

  it('forces are minus the energy gradient (central differences)', () => {
    const ev = evaluateSerial(m, sys);
    const h = 1e-6;
    for (const i of [0, 47, 151, 299])
      for (const ax of [0, 1]) {
        const fd = -(evaluateSerial(m, shifted(sys, i, ax, h)).energy - evaluateSerial(m, shifted(sys, i, ax, -h)).energy) / (2 * h);
        expect(ev.forces[2 * i + ax]).toBeCloseTo(fd, 6);
      }
  });

  it('is invariant under rotation and translation, and forces rotate with it', () => {
    const th = 0.7;
    const [c, sn] = [Math.cos(th), Math.sin(th)];
    const pos = new Float64Array(sys.pos.length);
    for (let i = 0; i < sys.n; i++) {
      const x = sys.pos[2 * i]!;
      const y = sys.pos[2 * i + 1]!;
      pos[2 * i] = c * x - sn * y + 3.2;
      pos[2 * i + 1] = sn * x + c * y - 1.1;
    }
    const a = evaluateSerial(m, sys);
    const b = evaluateSerial(m, { ...sys, pos });
    expect(b.energy).toBeCloseTo(a.energy, 9);
    for (const i of [5, 120]) {
      expect(b.forces[2 * i]).toBeCloseTo(c * a.forces[2 * i]! - sn * a.forces[2 * i + 1]!, 9);
      expect(b.forces[2 * i + 1]).toBeCloseTo(sn * a.forces[2 * i]! + c * a.forces[2 * i + 1]!, 9);
    }
  });

  it('the energy depends on atoms more than one cutoff away (T layers reach T·r_c)', () => {
    const near = evaluateSerial(m, sys).atomic[0]!;
    // Atom 2 is two spacings (> r_c, < 2 r_c) from atom 0 along the bottom row.
    const moved = evaluateSerial(m, shifted(sys, 2, 1, 0.05)).atomic[0]!;
    expect(Math.abs(moved - near)).toBeGreaterThan(1e-6);
  });
});

describe('parallel evaluation', () => {
  for (const T of [1, 2, 3, 5])
    for (const P of [2, 3, 4])
      it(`SevenNet and naive schemes reproduce the serial result (T = ${T}, P = ${P})`, () => {
        const m = makeToyModel(T);
        const ref = evaluateSerial(m, sys);
        const seven = evaluateSevenNet(m, sys, P);
        const naive = evaluateNaive(m, sys, P);
        expect(Math.abs(seven.energy - ref.energy)).toBeLessThan(1e-10);
        expect(Math.abs(naive.energy - ref.energy)).toBeLessThan(1e-10);
        expect(maxForceError(seven.forces, ref.forces).max).toBeLessThan(1e-10);
        expect(maxForceError(naive.forces, ref.forces).max).toBeLessThan(1e-10);
        // Every atom is owned once.
        expect(seven.domains.reduce((a, d) => a + d.owned.length, 0)).toBe(sys.n);
        expect(seven.updates).toBe(T * sys.n);
        if (T > 1) expect(naive.updates).toBeGreaterThan(seven.updates);
        expect(seven.messages).toBe(2 * (T - 1) + 2);
      });

  it('counts exchanged floats as the paper describes', () => {
    const m = makeToyModel(4);
    const ev = evaluateSevenNet(m, sys, 3);
    const ghosts = decompose(sys, 3, m.rc).reduce((a, d) => a + d.ghosts.length, 0);
    expect(ev.comm.positions).toBe(4 * ghosts);
    expect(ev.comm.features).toBe(3 * m.C * ghosts);
    expect(ev.comm.gradients).toBe(3 * m.C * ghosts);
    expect(ev.comm.forces).toBe(2 * ghosts);
  });

  it('without feature exchange the energy is wrong; without gradient or force exchange only the forces are', () => {
    const m = makeToyModel(3);
    const ref = evaluateSerial(m, sys);
    const noFeat = evaluateSevenNet(m, sys, 3, { features: false, gradients: true, forces: true });
    expect(Math.abs(noFeat.energy - ref.energy)).toBeGreaterThan(1e-6);
    const noGrad = evaluateSevenNet(m, sys, 3, { features: true, gradients: false, forces: true });
    expect(Math.abs(noGrad.energy - ref.energy)).toBeLessThan(1e-10);
    expect(maxForceError(noGrad.forces, ref.forces).max).toBeGreaterThan(1e-4);
    const noForce = evaluateSevenNet(m, sys, 3, { features: true, gradients: true, forces: false });
    expect(Math.abs(noForce.energy - ref.energy)).toBeLessThan(1e-10);
    expect(maxForceError(noForce.forces, ref.forces).max).toBeGreaterThan(1e-4);
    // Errors sit near the strip boundaries.
    const bad = maxForceError(noGrad.forces, ref.forces).atom;
    const x = sys.pos[2 * bad]!;
    const w = sys.width / 3;
    expect(Math.min(Math.abs(x - w), Math.abs(x - 2 * w))).toBeLessThan(3 * m.rc);
  });

  it('with one layer there is nothing to exchange between layers', () => {
    const m = makeToyModel(1);
    const ref = evaluateSerial(m, sys);
    const ev = evaluateSevenNet(m, sys, 4, { features: false, gradients: false, forces: true });
    expect(maxForceError(ev.forces, ref.forces).max).toBeLessThan(1e-10);
  });
});

describe('multi-fidelity', () => {
  const m = makeToyModel(3);

  it('a fidelity in the energy shift only moves energies by a constant per species: forces identical', () => {
    const lo = evaluateSerial(m, sys, 0, 'shift');
    const hi = evaluateSerial(m, sys, 1, 'shift');
    expect(maxForceError(lo.forces, hi.forces).max).toBe(0);
    let expected = 0;
    for (let i = 0; i < sys.n; i++) expected += m.fid.shift[sys.species[i]!]!;
    expect(hi.energy - lo.energy).toBeCloseTo(expected, 9);
  });

  it('a fidelity in every linear layer changes the forces too', () => {
    const lo = evaluateSerial(m, sys, 0, 'all');
    const hi = evaluateSerial(m, sys, 1, 'all');
    expect(maxForceError(lo.forces, hi.forces).max).toBeGreaterThan(1e-3);
    // Fidelity 0 is the plain model.
    expect(lo.energy).toBeCloseTo(evaluateSerial(m, sys).energy, 12);
  });

  it('dimer curves: parallel under a shift, different shapes otherwise', () => {
    const rs = [0.8, 1.0, 1.2, 1.5];
    const a = dimerCurves(m, [0, 1], rs, 'shift');
    const diffs = a.hi.map((h, i) => h - a.lo[i]!);
    for (const d of diffs) expect(d).toBeCloseTo(diffs[0]!, 12);
    const b = dimerCurves(m, [0, 1], rs, 'all');
    const d2 = b.hi.map((h, i) => h - b.lo[i]!);
    expect(Math.max(...d2) - Math.min(...d2)).toBeGreaterThan(1e-4);
  });

  it('the multi-fidelity model still parallelises exactly', () => {
    const ref = evaluateSerial(m, sys, 1, 'all');
    const ev = evaluateSevenNet(m, sys, 3, undefined, 1, 'all');
    expect(maxForceError(ev.forces, ref.forces).max).toBeLessThan(1e-10);
  });
});
