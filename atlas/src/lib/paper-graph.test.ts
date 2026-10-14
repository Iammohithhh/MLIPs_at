import { describe, expect, it } from 'vitest';
import { NODE_R, layoutPapers, yearScale } from './paper-graph';

describe('papers chronology layout', () => {
  it('is monotonic in year with a compressed early segment', () => {
    const x = yearScale(1000, 100, 20);
    expect(x(1921)).toBeLessThan(x(1987));
    expect(x(1987)).toBeLessThan(x(2007));
    expect(x(2000) - x(1920)).toBeLessThan(x(2026) - x(2000));
  });

  it('never overlaps nodes, even many papers in one year and lane', () => {
    const papers = Array.from({ length: 7 }, (_, k) => ({ id: `p${k}`, year: 2023, lane: 1, label: 'x' }));
    papers.push({ id: 'q', year: 2007, lane: 0, label: 'y' });
    const g = layoutPapers(papers, [{ from: 'p0', to: 'q', kind: 'cites' }], 3);
    expect(g.nodes).toHaveLength(8);
    for (let i = 0; i < g.nodes.length; i++)
      for (let j = i + 1; j < g.nodes.length; j++) {
        const a = g.nodes[i]!;
        const b = g.nodes[j]!;
        expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThanOrEqual(2 * NODE_R);
      }
    g.nodes.forEach((n) => {
      expect(n.y).toBeGreaterThan(0);
      expect(n.y).toBeLessThan(g.height);
    });
    expect(g.edges).toHaveLength(1);
  });

  it('drops edges to papers that are not drawn', () => {
    const g = layoutPapers([{ id: 'a', year: 2000, lane: 0, label: 'a' }], [{ from: 'a', to: 'zzz', kind: 'cites' }], 1);
    expect(g.edges).toHaveLength(0);
  });
});

describe('labels', () => {
  it('never overlap one another', () => {
    const papers = Array.from({ length: 10 }, (_, k) => ({ id: `p${k}`, year: 2020 + (k % 4), lane: k % 2, label: 'Somebody et al. 2023' }));
    const g = layoutPapers(papers, [], 2);
    const shown = g.nodes.filter((n) => n.labelY !== null);
    expect(shown.length).toBeGreaterThan(0);
    for (let i = 0; i < shown.length; i++)
      for (let j = i + 1; j < shown.length; j++) {
        const a = shown[i]!;
        const b = shown[j]!;
        const w = a.label.length * 5.6 + 4;
        const overlapX = Math.abs(a.x - b.x) < w;
        const overlapY = Math.abs(a.labelY! - b.labelY!) < 12;
        expect(overlapX && overlapY).toBe(false);
      }
  });
});
