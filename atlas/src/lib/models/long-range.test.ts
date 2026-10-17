import { describe, expect, it } from 'vitest';
import zno from '~/data/structures/zno-wurtzite.json';
import si from '~/data/structures/si-diamond.json';
import { mulberry32, randomRotation } from '../chem/rng';
import { madelungEwald } from '../physics/madelung';
import { COULOMB_K } from '../chem/elements';
import type { Structure } from '../structure/types';
import {
  b2,
  b2Direct,
  b3,
  b3Direct,
  carbonChain,
  compareChain,
  dimerCurve,
  dipoleTerm,
  environment,
  gaussianPair,
  legendreP,
  lesEnergy,
  makeAtomicNet,
  monoValue,
  monomials,
  multinomial,
  neighboursOf,
  powerToLegendre,
  qeqCharges,
  qeqMatrix,
  realSpaceEwald,
  rotateAll,
  selfTerm,
  sphericalPower,
  type Vec3,
  LES_KC,
  kConverged,
  pairEnergy,
  pairKernel,
} from './long-range';
import { QEQ_TOY } from '../physics/qeq';

const net = makeAtomicNet();

describe('C10H2 / C10H3+: who can see a change beyond the cutoff', () => {
  const a = carbonChain(10, false);
  const b = carbonChain(10, true);
  it('builds both molecules with the same left part', () => {
    expect(a.symbols.length).toBe(12);
    expect(b.symbols.length).toBe(13);
    for (let i = 0; i <= 10; i++) expect(b.positions[i]).toEqual(a.positions[i]);
  });
  const local = compareChain('local', net);
  const scaled = compareChain('scaled', net);
  const qeq = compareChain('qeq', net);
  const far = local.unchanged.map((u, i) => (u ? i : -1)).filter((i) => i >= 0);
  it('leaves the left half (more than 4.23 Å from every changed atom) with identical environments', () => {
    expect(far.length).toBeGreaterThanOrEqual(6);
    expect(far[0]).toBe(0);
    expect(local.unchanged.at(-1)).toBe(false);
  });
  it('gives local (3G / LES) charges that cannot change there', () => {
    for (const i of far) expect(local.delta[i]).toBe(0);
  });
  it('gives scaled 3G charges a uniform shift there, the same for every atom', () => {
    const d0 = scaled.delta[far[0]!]!;
    expect(Math.abs(d0)).toBeGreaterThan(1e-4);
    for (const i of far) expect(scaled.delta[i]).toBeCloseTo(d0, 12);
  });
  it('lets charge equilibration move charge onto those atoms, unevenly', () => {
    const d = far.map((i) => qeq.delta[i]!);
    for (const x of d) expect(Math.abs(x)).toBeGreaterThan(1e-3);
    expect(Math.max(...d) - Math.min(...d)).toBeGreaterThan(1e-3);
  });
  it('conserves the total charge and solves Eq. (5) with one multiplier', () => {
    for (const m of [a, b]) {
      const s = qeqCharges(net, m);
      expect(s.q.reduce((x, y) => x + y, 0)).toBeCloseTo(m.charge, 12);
      const A = qeqMatrix(m, m.symbols.map((x) => QEQ_TOY[x]!.J));
      A.forEach((row, i) => {
        const lhs = row.reduce((t, v, j) => t + v * s.q[j]!, 0) + s.chi[i]!;
        expect(lhs).toBeCloseTo(-s.lambda, 9);
      });
    }
  });
  it('is the minimum of E_Qeq at fixed total charge', () => {
    const s = qeqCharges(net, b);
    const A = qeqMatrix(b, b.symbols.map((x) => QEQ_TOY[x]!.J));
    const E = (q: number[]) => {
      let e = 0;
      q.forEach((qi, i) => {
        e += s.chi[i]! * qi + 0.5 * A[i]![i]! * qi * qi;
        for (let j = 0; j < i; j++) e += A[i]![j]! * qi * q[j]!;
      });
      return e;
    };
    expect(E(s.q)).toBeCloseTo(s.energy, 10);
    const q = [...s.q];
    q[0]! += 0.01;
    q[5]! -= 0.01;
    expect(E(q)).toBeGreaterThan(s.energy);
  });
});

describe('latent Ewald summation', () => {
  const rand = mulberry32(11);
  const L: Vec3 = [9, 10, 11];
  const pos: Vec3[] = Array.from({ length: 6 }, () => [9 * rand(), 10 * rand(), 11 * rand()]);
  const q = [0.7, -0.4, 0.3, -0.9, 0.5, -0.2];
  const kAll = (s: number) => Math.sqrt(2 * 14 * Math.log(10)) / s;
  const self = (s: number) => q.reduce((t, x) => t + selfTerm(x, s), 0);
  const point = (s: number) => lesEnergy(pos, q, L, s, kAll(s)).energy + realSpaceEwald(pos, q, L, s, 9 * s) - self(s);

  it('is the reciprocal part of an Ewald sum: adding the erfc part and removing self terms gives a σ-independent energy', () => {
    expect(point(0.7)).toBeCloseTo(point(1.3), 4);
  });
  it('reproduces the rock-salt Madelung constant that way', () => {
    const a = 5.64;
    const na: Vec3[] = [[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]];
    const cl: Vec3[] = na.map((p) => [p[0] + a / 2, p[1], p[2]]);
    const P = [...na, ...cl];
    const Q = [1, 1, 1, 1, -1, -1, -1, -1];
    const s = 0.8;
    const e = lesEnergy(P, Q, [a, a, a], s, kAll(s)).energy + realSpaceEwald(P, Q, [a, a, a], s, 9 * s) - 8 * selfTerm(1, s);
    expect(e / 4).toBeCloseTo((-madelungEwald() * COULOMB_K) / (a / 2), 3);
  });
  it('is invariant to translating every atom', () => {
    const moved = pos.map((p) => [p[0] + 1.7, p[1] - 0.3, p[2] + 4.2] as Vec3);
    expect(lesEnergy(moved, q, L).energy).toBeCloseTo(lesEnergy(pos, q, L).energy, 9);
  });
  it('counts k vectors symmetrically and none at k = 0', () => {
    const { nk } = lesEnergy(pos, q, [30, 30, 30]);
    expect(nk % 2).toBe(0);
    expect(nk).toBeGreaterThan(0);
  });
  it('gives an isolated pair of Gaussian charges, minus the tinfoil dipole term, in a large box', () => {
    const d = 4;
    const [pt] = dimerCurve(1, -1, [d], 60, 1, 8);
    expect(pt!.lesFull).toBeCloseTo(gaussianPair(d, 1, -1, 1) - dipoleTerm([d, 0, 0], 60 ** 3), 4);
  });
  it('gives the same curve from the fast cubic-box kernel', () => {
    const ds = [1.5, 4, 9.2, 15];
    for (const [q1, q2, kc] of [[1, -1, LES_KC], [1, 1, 3.05], [0.6, -0.3, LES_KC]] as const) {
      const kcK = pairKernel(30, 1, kc);
      const full = pairKernel(30, 1, kConverged(1));
      dimerCurve(q1, q2, ds, 30, 1, kc).forEach((p, i) => {
        expect(pairEnergy(ds[i]!, q1, q2, kcK, full)).toBeCloseTo(p.les, 8);
        expect(pairEnergy(ds[i]!, q1, q2, full, full)).toBeCloseTo(p.lesFull, 8);
      });
    }
  });
  it('approaches bare Coulomb beyond a few σ', () => {
    expect(gaussianPair(8, 1, 1, 1)).toBeCloseTo(COULOMB_K / 8, 6);
    expect(gaussianPair(0.5, 1, 1, 1)).toBeLessThan(COULOMB_K / 0.5 / 2);
  });
});

describe('CACE: Cartesian angular basis', () => {
  const rand = mulberry32(5);
  const env = neighboursOf(
    Array.from({ length: 7 }, () => [3 * rand() - 1.5, 3 * rand() - 1.5, 3 * rand() - 1.5] as Vec3),
    4,
  );
  it('has (l+1)(l+2)/2 monomials of order l', () => {
    for (let l = 0; l <= 6; l++) expect(monomials(l).length).toBe(((l + 1) * (l + 2)) / 2);
  });
  it('contracts monomials into powers of dot products (Eq. 4)', () => {
    const u = env[0]!.u;
    const v = env[1]!.u;
    for (let l = 0; l <= 5; l++) {
      const s = monomials(l).reduce((t, m) => t + multinomial(m) * monoValue(m, u) * monoValue(m, v), 0);
      expect(s).toBeCloseTo((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) ** l, 12);
    }
  });
  it('makes B(2) a sum of w_j w_k cos^l θ_jk and B(3) a chain of two angles (Eqs. 7–9)', () => {
    for (let l = 0; l <= 4; l++) expect(b2(env, l)).toBeCloseTo(b2Direct(env, l), 10);
    for (const [l1, l2] of [[1, 1], [1, 2], [2, 3]] as const) expect(b3(env, l1, l2)).toBeCloseTo(b3Direct(env, l1, l2), 9);
  });
  it('is invariant under rotation, while the A_l themselves are not', () => {
    const R = randomRotation(mulberry32(9));
    const rot = env.map((n) => ({ u: rotateAll([n.u], R)[0]!, w: n.w }));
    for (let l = 1; l <= 4; l++) expect(b2(rot, l)).toBeCloseTo(b2(env, l), 10);
    expect(b3(rot, 2, 1)).toBeCloseTo(b3(env, 2, 1), 10);
  });
  it('expands x^l in Legendre polynomials of order l, l−2, ...', () => {
    for (let l = 0; l <= 6; l++) {
      const c = powerToLegendre(l);
      for (const x of [-0.9, -0.3, 0.2, 0.75]) expect(c.reduce((t, ci, k) => t + ci * legendreP(k, x), 0)).toBeCloseTo(x ** l, 12);
      c.forEach((ci, k) => (k % 2 !== l % 2 ? expect(ci).toBe(0) : expect(ci).toBeGreaterThan(0)));
    }
  });
  it('so each Cartesian B(2)_l is a fixed mix of spherical invariants p_l, p_(l−2), ...', () => {
    for (let l = 0; l <= 4; l++) {
      const c = powerToLegendre(l);
      const mix = c.reduce((t, ci, k) => t + (ci ? ci * sphericalPower(env, k) : 0), 0);
      expect(b2(env, l)).toBeCloseTo(mix, 9);
    }
  });
  it('finds the four tetrahedral neighbours in wurtzite ZnO and diamond Si, whose l = 1 moment vanishes in Si', () => {
    expect(environment(zno as unknown as Structure, 0, 2.5).vectors.length).toBe(4);
    const s = environment(si as unknown as Structure, 0, 2.5);
    expect(s.vectors.length).toBe(4);
    expect(b2(s.neighbours, 1)).toBeCloseTo(0, 12);
  });
});
