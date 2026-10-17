import { describe, expect, it } from 'vitest';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import { norm, parseManifest } from './experiment';
import { bonds, graphEdges, neighbourhood } from './scene';

const m = parseManifest(raw);
const B = m.displacement.baselineIndex;
const moved = m.displacement.atom;

describe('neighbourhood (periodic images)', () => {
  const base = neighbourhood(m, B, m.rMax)!;

  it('centres on the moved atom and matches the diamond shell counts within r_max', () => {
    expect(base[0]!.index).toBe(moved);
    expect(norm(base[0]!.pos)).toBe(0);
    const count = (r: number) => base.filter((a) => Math.abs(a.r0 - r) < 0.01).length;
    expect(count(m.shells[0]!)).toBe(4);
    expect(count(m.shells[1]!)).toBe(12);
    expect(count(m.shells[2]!)).toBe(12);
    expect(new Set(base.map((a) => a.index)).size).toBe(base.length); // one image per atom (cell 16.3 Å > 2 r_max)
  });

  it('keeps the same atoms and images in every frame; only the moved atom moves', () => {
    for (const f of m.frames) {
      const nb = neighbourhood(m, f.k, m.rMax)!;
      expect(nb.map((a) => a.index)).toEqual(base.map((a) => a.index));
      nb.forEach((a, i) => {
        const b = base[i]!;
        const dd = norm([a.pos[0] - b.pos[0], a.pos[1] - b.pos[1], a.pos[2] - b.pos[2]]);
        expect(dd).toBeCloseTo(a.index === moved ? Math.abs(f.d) : 0, 6);
      });
    }
  });

  it('graph edges are neighbours within r_max, distinct from Si–Si bonds', () => {
    const e = graphEdges(m, B, base, moved);
    expect(e.length).toBe(base.length - 1);
    expect(e.every((x) => x.r < m.rMax)).toBe(true);
    const bondCount = bonds(base).filter(([a, b]) => base[a]!.index === moved || base[b]!.index === moved).length;
    expect(bondCount).toBe(4);
  });
});

import { STAGE_INFO, stagesFor } from './stages';
import { STAGES } from './urlState';

describe('walkthrough stages', () => {
  it('cover the seven URL stages in order, each with an equation and an implementation locator', () => {
    expect(STAGE_INFO.map((s) => s.id)).toEqual([...STAGES]);
    for (const s of STAGE_INFO) {
      expect(s.tex.length).toBeGreaterThan(5);
      expect(s.code.length).toBeGreaterThan(0);
      expect(/[\x07\x08\x0b\x0c]/.test(s.tex)).toBe(false); // no TeX escapes lost to string processing
    }
  });
  it('neighbour count in the stage text matches the data', () => {
    const n = neighbourhood(m, B, m.rMax)!.length - 1;
    expect(stagesFor(m)[1]!.changed).toContain(`${n} neighbours`);
    const hyper = m.provenance.hyperparameters as { num_bessel: number; avg_num_neighbors: number };
    expect(STAGE_INFO[2]!.changed).toContain(`${hyper.num_bessel} radial Bessel values`);
    expect(STAGE_INFO[3]!.plain).toContain(hyper.avg_num_neighbors.toFixed(2));
  });
});
