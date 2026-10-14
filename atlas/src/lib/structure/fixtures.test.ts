import { describe, expect, it } from 'vitest';
import { ljCluster } from '~/lib/scenes/viewer-gallery';

describe('LJ13 gallery cluster', () => {
  it('has 13 atoms, finite forces that sum to zero, centre–vertex ≈ 2^(1/6)σ', () => {
    const s = ljCluster();
    expect(s.positions).toHaveLength(13);
    const f = s.vectors!.forces!.values;
    [0, 1, 2].forEach((c) => expect(f.reduce((t, v) => t + v[c]!, 0)).toBeCloseTo(0, 9));
    expect(f.some((v) => Math.hypot(...v) > 0.1)).toBe(true);
    const d = Math.hypot(...s.positions[1]!.map((v, k) => v - s.positions[0]![k]!));
    expect(d / 3.4).toBeGreaterThan(1.0);
    expect(d / 3.4).toBeLessThan(1.25);
  });
});
