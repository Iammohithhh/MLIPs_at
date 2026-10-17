import { describe, expect, it } from 'vitest';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import { parseManifest, siteEnergyChange } from './experiment';
import { baselineDistances, stageRange, summarise, tracePoints, TRACE_STAGES } from './trace';

const m = parseManifest(raw);
const B = m.displacement.baselineIndex;
const last = m.frames.length - 1;
const r0s = baselineDistances(m);

describe('trace the change: receptive field of the two-layer, r_max = 6 Å model', () => {
  it('baseline frame: nothing changed at any stage', () => {
    for (const st of TRACE_STAGES) {
      const s = summarise(tracePoints(m, B, st.key, r0s));
      expect(s.changed, st.key).toBe(0);
      expect(s.exactZero).toBe(m.structure.n);
    }
  });

  it('layer-1 features change only within r_max, exactly zero beyond', () => {
    for (const k of [0, last]) {
      for (const p of tracePoints(m, k, 'layer1', r0s)) {
        if (p.r0 >= m.rMax) expect(p.value).toBe(0);
      }
      expect(summarise(tracePoints(m, k, 'layer1', r0s)).reach!).toBeLessThan(m.rMax);
    }
  });

  it('layer-2 features and site energies reach beyond r_max but not beyond 2 r_max', () => {
    for (const st of ['layer2', 'site'] as const) {
      const pts = tracePoints(m, last, st, r0s);
      const s = summarise(pts);
      expect(s.reach!).toBeGreaterThan(m.rMax);
      expect(s.reach!).toBeLessThan(2 * m.rMax);
      for (const p of pts) if (p.r0 >= 2 * m.rMax) expect(p.value).toBe(0);
    }
    // the 6 farthest atoms in this cell (12.74 Å) do exist, so the zero is a real observation
    expect(r0s.filter((r) => r > 2 * m.rMax).length).toBeGreaterThan(0);
  });

  it('forces change even where the site energy is exactly unchanged (F = −∂E_total/∂r)', () => {
    const far = r0s.map((r, i) => (r > 2 * m.rMax ? i : -1)).filter((i) => i >= 0);
    const F = tracePoints(m, last, 'force', r0s);
    for (const i of far) {
      expect(siteEnergyChange(m, last, i)).toBe(0);
      expect(F[i]!.value!).toBeGreaterThan(1e-6); // well above the stored force precision (1e-7 eV/Å)
    }
  });

  it('edge stage: only the moved atom and atoms sharing a graph edge with it change', () => {
    const pts = tracePoints(m, last, 'edges', r0s);
    const s = summarise(pts);
    expect(s.reach!).toBeLessThan(m.rMax + Math.abs(m.displacement.values[last]!));
    expect(pts[m.displacement.atom]!.value!).toBeCloseTo(Math.abs(m.displacement.values[last]!), 2);
  });

  it('fixed log ranges are positive and ordered', () => {
    for (const st of TRACE_STAGES) {
      const r = stageRange(m, st.key, r0s)!;
      expect(r[0]).toBeGreaterThan(0);
      expect(r[1]).toBeGreaterThan(r[0]);
    }
  });
});
