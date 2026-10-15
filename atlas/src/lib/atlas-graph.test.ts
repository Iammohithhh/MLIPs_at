import { describe, expect, it } from 'vitest';
import { auditContent } from '~/content/integrity';
import { buildAtlasMap } from './atlas-graph';

const { records } = auditContent('content');

describe('atlas map', () => {
  const map = buildAtlasMap(records);

  it('has one node per record and only edges between existing nodes', () => {
    expect(map.nodes).toHaveLength(records.size);
    const ids = new Set(map.nodes.map((n) => n.id));
    for (const e of map.edges) {
      expect(ids.has(e.source)).toBe(true);
      expect(ids.has(e.target)).toBe(true);
      expect(e.source).not.toBe(e.target);
    }
  });

  it('is laid out identically on every build', () => {
    const again = buildAtlasMap(records);
    expect(again.nodes.map((n) => [n.x, n.y])).toEqual(map.nodes.map((n) => [n.x, n.y]));
  });

  it('links MACE to its paper and its checkpoint', () => {
    const has = (a: string, b: string) => map.edges.some((e) => (e.source === a && e.target === b) || (e.source === b && e.target === a));
    expect(has('model-families/mace', 'papers/batatia2022mace')).toBe(true);
    expect(has('checkpoints/mace-mp-0a-medium', 'model-families/mace')).toBe(true);
  });
});
