import { describe, expect, it } from 'vitest';
import type { AtlasMap } from './atlas-graph';
import { STRATA, STRATUM_GAP, buildConstellation, stratumOf } from './constellation';
import { COLLECTION_NAMES } from '~/content/schemas';

const map: AtlasMap = {
  nodes: [
    { id: 'principles/a', collection: 'principles', title: 'A', status: 'draft', url: '/', degree: 1, x: 0, y: 0 },
    { id: 'claims/b', collection: 'claims', title: 'B', status: 'draft', url: '/', degree: 4, x: 100, y: 50 },
  ],
  edges: [{ source: 'principles/a', target: 'claims/b' }],
};

describe('constellation layout', () => {
  it('places every collection in exactly one stratum', () => {
    for (const c of COLLECTION_NAMES) expect(STRATA.filter((s) => s.collections.includes(c))).toHaveLength(1);
  });

  it('is deterministic and keeps strata apart in height', () => {
    const a = buildConstellation(map);
    expect(a).toEqual(buildConstellation(map));
    const [p, c] = a.nodes;
    expect(c!.p[2] - p!.p[2]).toBeGreaterThan(STRATUM_GAP * (stratumOf('claims') - stratumOf('principles')) - 6.01);
    expect(c!.r).toBeGreaterThan(p!.r);
  });
});
