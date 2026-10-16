// The Atlas in 3D: the 2D force layout of the record graph (atlas-graph.ts) lifted into strata, so a
// node's height says what kind of knowledge it is — physics at the bottom, models and data in the
// middle, evidence and its consequences at the top. Deterministic: same records, same picture.

import type { CollectionName } from '~/content/schemas';
import type { AtlasMap, MapNode } from './atlas-graph';

export const STRATA: Array<{ name: string; collections: CollectionName[] }> = [
  { name: 'Physics', collections: ['principles', 'concepts', 'experiments'] },
  { name: 'Models and data', collections: ['model-families', 'implementations', 'checkpoints', 'datasets', 'levels-of-theory'] },
  { name: 'Benchmarks and sources', collections: ['benchmarks', 'benchmark-results', 'papers'] },
  { name: 'Evidence and use', collections: ['claims', 'failure-cases', 'applications'] },
];

export const STRATUM_GAP = 26;

export interface StarNode extends MapNode {
  p: [number, number, number];
  stratum: number;
  r: number;
}

export function stratumOf(c: CollectionName): number {
  const i = STRATA.findIndex((s) => s.collections.includes(c));
  return i < 0 ? 0 : i;
}

/** Hash of an id to [0, 1): a stable per-node offset within its stratum. */
function unit(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return ((h >>> 0) % 10007) / 10007;
}

export function buildConstellation(map: AtlasMap, extent = 90): { nodes: StarNode[]; edges: AtlasMap['edges'] } {
  const xs = map.nodes.map((n) => n.x);
  const ys = map.nodes.map((n) => n.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
  const z0 = ((STRATA.length - 1) * STRATUM_GAP) / 2;
  const nodes = map.nodes.map((n) => {
    const s = stratumOf(n.collection);
    return {
      ...n,
      stratum: s,
      p: [((n.x - cx) / span) * extent, ((n.y - cy) / span) * extent, s * STRATUM_GAP - z0 + (unit(n.id) - 0.5) * 6] as [number, number, number],
      r: 0.9 + 0.55 * Math.sqrt(n.degree),
    };
  });
  return { nodes, edges: map.edges };
}
