import { describe, expect, it } from 'vitest';
import {
  DATA,
  HP,
  K_E,
  TD_THETA,
  chord,
  coulombLR,
  dimerPositions,
  distancesFrom,
  expCutoff,
  firstPassInvariants,
  fragmentSplit,
  integerLimits,
  invariantDistance,
  maxVector,
  methaneD2d,
  nqe,
} from './aimnet2';

const dimer = (id: string) => DATA.dimers.find((d) => d.id === id)!;
const row = (id: string, d: number) => dimer(id).rows.find((r) => Math.abs(r.d - d) < 1e-6)!;

describe('AIMNet2 first-pass features on the methane D2d path', () => {
  const td = methaneD2d(TD_THETA);
  const tdInv = firstPassInvariants(td.numbers, td.positions, 0);

  it('keeps every C–H distance and makes the four bond directions sum to zero', () => {
    for (const th of [30, 45, TD_THETA, 70, 90]) {
      const m = methaneD2d(th);
      expect(distancesFrom(m.positions, 0).every((r) => Math.abs(r - 1.087) < 1e-12)).toBe(true);
      const s = m.positions.slice(1).reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]);
      expect(Math.hypot(...s)).toBeLessThan(1e-12);
    }
  });

  it('gives carbon identical l = 0 and vanishing l = 1 invariants along the whole path', () => {
    expect(maxVector(tdInv)).toBeLessThan(1e-12);
    for (const th of [30, 40, 60, 75, 90]) {
      const m = methaneD2d(th);
      const inv = firstPassInvariants(m.numbers, m.positions, 0);
      expect(invariantDistance(inv, tdInv)).toBeLessThan(1e-12);
      expect(maxVector(inv)).toBeLessThan(1e-12);
    }
  });

  it('changes the hydrogens’ invariants, which is how later passes see the distortion', () => {
    const tdH = firstPassInvariants(td.numbers, td.positions, 1);
    const sq = methaneD2d(90);
    const sqH = firstPassInvariants(sq.numbers, sq.positions, 1);
    expect(invariantDistance(sqH, tdH)).toBeGreaterThan(1e-2);
    // H–H distances: Td 1.775 Å (all), D4h √2 r and 2r
    expect(distancesFrom(td.positions, 1).slice(1)).toEqual(expect.arrayContaining([expect.closeTo(1.087 * Math.sqrt(8 / 3), 9)]));
    const dSq = distancesFrom(sq.positions, 1);
    expect(dSq[1]).toBeCloseTo(1.087 * Math.SQRT2, 9);
    expect(dSq[3]).toBeCloseTo(2 * 1.087, 9);
  });

  it('is what the released model shows: carbon’s pass-0 features constant, later passes not', () => {
    for (const r of DATA.methane) {
      expect(r.conv[0]!.C).toBeLessThan(1e-5);
      if (Math.abs(r.theta - TD_THETA) > 5) {
        expect(r.conv[0]!.H).toBeGreaterThan(1e-2);
        expect(r.conv[1]!.C).toBeGreaterThan(1e-2);
        expect(r.conv[2]!.C).toBeGreaterThan(1e-2);
      }
    }
    const tdRow = DATA.methane.find((r) => r.theta === 55)!;
    expect(Math.abs(tdRow.E)).toBeLessThan(0.01);
    expect(DATA.methane.find((r) => r.theta === 90)!.E).toBeGreaterThan(1);
  });

  it('uses the checkpoint’s own basis: 16 shells from 0.8 Å, cutoff 5 Å', () => {
    expect(HP.shifts).toHaveLength(16);
    expect(HP.shifts[0]).toBeCloseTo(0.8, 6);
    expect(HP.rc).toBe(5);
    expect(HP.eta).toBeCloseTo((16 / 4.2) ** 2, 4);
  });
});

describe('neural charge equilibration', () => {
  it('restores the total charge and moves charge in proportion to f', () => {
    const qt = [0.1, -0.3, 0.05, 0.4];
    const f = [1, 0, 2, 0.5];
    const q = nqe(qt, f, 1, 0);
    expect(q.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(q[1]).toBe(-0.3);
    expect((q[2]! - qt[2]!) / (q[0]! - qt[0]!)).toBeCloseTo(2, 12);
  });

  it('splits Q evenly between identical separated fragments, whatever they predicted', () => {
    const [a, b] = fragmentSplit(-0.2, -0.2, 7.3, 7.3, 1, 0);
    expect(a).toBeCloseTo(0.5, 12);
    expect(b).toBeCloseTo(0.5, 12);
  });

  it('reproduces the recorded fragment charges of the released model from q̃ and f', () => {
    for (const dm of DATA.dimers)
      for (const r of dm.rows)
        for (const p of r.passes) {
          const [a, b] = fragmentSplit(p.q_tilde[0], p.q_tilde[1], p.f[0], p.f[1], dm.Q);
          expect(a).toBeCloseTo(p.q[0], 3);
          expect(b).toBeCloseTo(p.q[1], 3);
        }
  });
});

describe('released-model dimers', () => {
  it('splits (C6H6)2+ exactly in half at every separation', () => {
    for (const r of dimer('bz2').rows) {
      expect(r.fragQ[0]).toBeCloseTo(0.5, 4);
      expect(r.fragQ[1]).toBeCloseTo(0.5, 4);
    }
  });

  it('sums its terms to the total energy', () => {
    for (const dm of DATA.dimers) for (const r of dm.rows) expect(r.U_local + r.U_coul + r.U_disp).toBeCloseTo(r.E, 3);
  });

  it('has a local energy that stops changing once the fragments are beyond the 5 Å cutoff', () => {
    for (const dm of DATA.dimers) {
      const far = dm.rows.filter((r) => r.d >= 5.4);
      for (const r of far) expect(r.U_local).toBeCloseTo(far[0]!.U_local, 3);
      for (const r of far) for (const k of [0, 1]) expect(r.fragQ[k]!).toBeCloseTo(far[0]!.fragQ[k]!, 4);
    }
  });

  it('dissociates (C6H6)2+ into two benzenes of charge ½, not into C6H6+ + C6H6', () => {
    const dm = dimer('bz2');
    const lim = integerLimits(dm);
    const r20 = row('bz2', 20);
    // two +½ charges 20 Å apart repel by k/(4·20); the rest is two isolated half-charged benzenes
    expect(r20.E - K_E / 80).toBeCloseTo(lim.half!, 1);
    expect(lim.half! - lim.plusA).toBeLessThan(-1);
  });

  it('keeps a spurious charge on neutral, separated benzene and water', () => {
    const r = row('bzw0', 20);
    expect(Math.abs(r.fragQ[0])).toBeGreaterThan(1e-4);
    expect(r.fragQ[0] + r.fragQ[1]).toBeCloseTo(0, 5);
  });

  it('recomputes the Coulomb term from the stored charges', () => {
    for (const id of ['bz2', 'bzw']) {
      const dm = dimer(id);
      for (const d of [3.4, 8, 20]) {
        const r = row(id, d);
        expect(coulombLR(r.q, dimerPositions(dm, d))).toBeCloseTo(r.U_coul, 2);
      }
    }
  });
});

describe('fractional charge', () => {
  it('has the exact chord at integers and a convex released-model curve between them', () => {
    const eq = DATA.benzeneEQ;
    for (const Q of [-1, 0, 1, 2]) expect(chord(eq, Q)).toBe(eq.find((r) => r.Q === Q)!.E);
    const half = eq.find((r) => Math.abs(r.Q - 0.5) < 1e-6)!;
    expect(half.E - chord(eq, 0.5)).toBeLessThan(-0.5);
  });

  it('switches the short-range Coulomb envelope from 1 to 0 at 4.6 Å', () => {
    expect(expCutoff(0)).toBeCloseTo(1, 12);
    expect(expCutoff(HP.coulomb_sr_rc)).toBeLessThan(1e-12);
    expect(expCutoff(2.3)).toBeGreaterThan(0);
  });
});
