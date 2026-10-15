import { describe, expect, it } from 'vitest';
import water from '~/data/structures/h2o.json';
import type { Vec3 } from '../structure/types';
import { QEQ_TOY, dimerLimit, gamma, qeq } from './qeq';

const W = water.positions as Vec3[];
const Wp = water.symbols.map((s) => QEQ_TOY[s]!);

describe('charge equilibration', () => {
  it('conserves total charge and equalises the chemical potential', () => {
    for (const Q of [0, -1, 1]) {
      const r = qeq(W, Wp, Q);
      expect(r.q.reduce((a, b) => a + b, 0)).toBeCloseTo(Q, 10);
      for (const c of r.chiEff) expect(c).toBeCloseTo(r.mu, 9);
    }
  });
  it('makes oxygen negative and the hydrogens positive and equal in water', () => {
    const r = qeq(W, Wp);
    const iO = water.symbols.indexOf('O');
    expect(r.q[iO]!).toBeLessThan(-0.2);
    const hs = r.q.filter((_, i) => i !== iO);
    expect(hs[0]!).toBeGreaterThan(0);
    expect(hs[0]!).toBeCloseTo(hs[1]!, 9);
  });
  it('is the minimum of the energy at fixed total charge', () => {
    const r = qeq(W, Wp);
    // Moving charge between two atoms (keeping Σq) can only raise the energy.
    const e = (q: number[]) => {
      let s = 0;
      for (let i = 0; i < q.length; i++) {
        s += Wp[i]!.chi * q[i]! + 0.5 * Wp[i]!.J * q[i]! ** 2;
        for (let j = i + 1; j < q.length; j++) {
          const a = W[i]!;
          const b = W[j]!;
          s += gamma(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])) * q[i]! * q[j]!;
        }
      }
      return s;
    };
    expect(e(r.q)).toBeCloseTo(r.energy, 10);
    for (const d of [0.01, -0.02]) {
      const q = [...r.q];
      q[0]! += d;
      q[1]! -= d;
      expect(e(q)).toBeGreaterThan(r.energy);
    }
  });
  it('leaves a separated Na–Cl pair with fractional charges (the QEq dissociation problem)', () => {
    const na = QEQ_TOY.Na!;
    const cl = QEQ_TOY.Cl!;
    const far = qeq([[0, 0, 0], [1e7, 0, 0]], [na, cl]);
    expect(far.q[1]!).toBeCloseTo(-dimerLimit(na, cl), 4);
    expect(Math.abs(far.q[1]!)).toBeGreaterThan(0.3);
  });
});
