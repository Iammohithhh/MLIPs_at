import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import type { Structure } from '../structure/types';
import { clusterAround, hopsFromCentre, reachByDepth } from './receptive';

const S = si as unknown as Structure;

describe('receptive field in diamond silicon', () => {
  const pos = clusterAround(S, 4 * 4 + 1);
  const hops = hopsFromCentre(pos, 4, 4);
  const reach = reachByDepth(hops, 4);

  it('one layer with r_c = 4 Å reaches the first two shells: 4 + 12 = 16 atoms', () => {
    expect(reach[0]).toBe(16);
  });

  it('reach grows monotonically and stays inside the ball of radius T·r_c', () => {
    for (let t = 1; t < 4; t++) expect(reach[t]!).toBeGreaterThan(reach[t - 1]!);
    pos.forEach((p, i) => {
      const h = hops[i]!;
      if (h > 0) expect(Math.hypot(p[0], p[1], p[2])).toBeLessThan(4 * h);
    });
  });

  it('a larger cluster gives exactly the same counts (no boundary effect)', () => {
    const big = clusterAround(S, 4 * 4 + 6);
    expect(reachByDepth(hopsFromCentre(big, 4, 4), 4)).toEqual(reach);
  });
});
