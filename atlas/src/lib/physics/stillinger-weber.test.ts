import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import type { Structure, Vec3 } from '../structure/types';
import { neighboursOf } from '../structure/geometry';
import { SW_CUTOFF, SW_SI, phi2, phi3, screen3, swCluster, swPerAtom, triplet } from './stillinger-weber';

const S = si as unknown as Structure;

describe('Stillinger–Weber silicon', () => {
  it('reproduces the diamond cohesive energy −2ε = −4.3366 eV/atom at a = 5.431 Å', () => {
    const vecs: Vec3[][] = S.positions.map((p, i) =>
      neighboursOf(S, i, SW_CUTOFF).map((n) => [n.position[0] - p[0], n.position[1] - p[1], n.position[2] - p[2]] as Vec3),
    );
    const e = swPerAtom(vecs);
    expect(e.three).toBeCloseTo(0, 10); // every angle is tetrahedral
    expect(e.two).toBeCloseTo(-2 * SW_SI.epsilon, 3);
  });

  it('has its pair minimum near the Si–Si bond length', () => {
    let best = { r: 0, e: Infinity };
    for (let r = 2.0; r < 3.0; r += 0.0005) {
      const e = phi2(r);
      if (e < best.e) best = { r, e };
    }
    expect(best.r).toBeGreaterThan(2.3);
    expect(best.r).toBeLessThan(2.4);
    expect(best.e).toBeCloseTo(-SW_SI.epsilon, 3);
  });

  it('vanishes smoothly at the cutoff (value and slopes → 0)', () => {
    const h = 1e-4;
    const r = SW_CUTOFF - 2e-3;
    expect(Math.abs(phi2(r))).toBeLessThan(1e-50);
    expect(Math.abs((phi2(r + h) - phi2(r - h)) / (2 * h))).toBeLessThan(1e-40);
    expect(phi2(SW_CUTOFF)).toBe(0);
    expect(screen3(SW_CUTOFF + 0.1)).toBe(0);
  });

  it('three-body term is zero at the tetrahedral angle and positive elsewhere', () => {
    expect(phi3(2.35, 2.35, -1 / 3)).toBeCloseTo(0, 12);
    expect(phi3(2.35, 2.35, Math.cos((90 * Math.PI) / 180))).toBeGreaterThan(0);
  });

  it('a pairs-only model cannot prefer the tetrahedral angle in a triplet', () => {
    const at = (th: number) => swCluster(triplet(2.35, 2.35, th), SW_SI, false).total;
    // j–k is beyond the cutoff for wide angles, so the pair energy does not depend on θ there.
    expect(at(109.47)).toBeCloseTo(at(150), 10);
    const full = (th: number) => swCluster(triplet(2.35, 2.35, th)).total;
    expect(full(109.47)).toBeLessThan(full(150));
    expect(full(109.47)).toBeLessThan(full(90));
  });
});
