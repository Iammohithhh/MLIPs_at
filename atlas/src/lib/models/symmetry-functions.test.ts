import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import type { Structure, Vec3 } from '../structure/types';
import { neighboursOf } from '../structure/geometry';
import { apply, rotation } from '../physics/rotation-toy';
import { cosineCutoff } from '../cutoffs';
import { DEFAULT_SET, evaluate, g2, g3WithShares } from './symmetry-functions';

const S = si as unknown as Structure;
const env: Vec3[] = neighboursOf(S, 0, 6).map((n) => [n.position[0] - S.positions[0]![0], n.position[1] - S.positions[0]![1], n.position[2] - S.positions[0]![2]]);

describe('Behler–Parrinello symmetry functions', () => {
  it('G² with η → 0 counts neighbours weighted by the cutoff', () => {
    const expected = env.reduce((s, d) => s + cosineCutoff(Math.hypot(...d), 6).f, 0);
    expect(g2(env, { kind: 'G2', eta: 0, rs: 0 }, 6)).toBeCloseTo(expected, 12);
  });

  it('every function is invariant under rotation and neighbour permutation', () => {
    const Q = rotation([0.2, 0.7, -0.3], 1.1);
    const rotated = env.map((d) => apply(Q, d));
    const shuffled = [...env].reverse();
    for (const p of DEFAULT_SET) {
      const v = evaluate(env, p, 6);
      expect(evaluate(rotated, p, 6)).toBeCloseTo(v, 10);
      expect(evaluate(shuffled, p, 6)).toBeCloseTo(v, 10);
    }
  });

  it('neighbour shares of G³ add up to its value', () => {
    const r = g3WithShares(env, { kind: 'G3', eta: 0.005, zeta: 2, lambda: 1 }, 6);
    expect(r.shares.reduce((a, b) => a + b, 0)).toBeCloseTo(r.value, 10);
  });

  it('a distortion changes the fingerprint', () => {
    const moved = env.map((d, k) => [d[0] + 0.1 * Math.sin(k), d[1], d[2]] as Vec3);
    expect(Math.abs(evaluate(moved, DEFAULT_SET[3]!, 6) - evaluate(env, DEFAULT_SET[3]!, 6))).toBeGreaterThan(1e-3);
  });
});
