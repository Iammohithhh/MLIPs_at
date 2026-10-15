// The Atlas as a graph: every record is a node, every schema reference an edge.
// Laid out at build time with a force simulation driven by a seeded random source,
// so the map is identical on every build.

import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationNodeDatum } from 'd3-force';
import type { CollectionName } from '~/content/schemas';
import type { LoadedRecord } from '~/content/integrity';
import { COLLECTION_META, recordTitle } from '~/content/meta';
import { recordUrl } from './record-urls';
import { mulberry32 } from './chem/rng';

export interface MapNode {
  id: string;
  collection: CollectionName;
  title: string;
  status: string;
  url: string;
  degree: number;
  x: number;
  y: number;
}

export interface MapEdge {
  source: string;
  target: string;
}

export interface AtlasMap {
  nodes: MapNode[];
  edges: MapEdge[];
}

// Collections arranged on a loose ring so related kinds sit near each other.
const ANCHOR: Partial<Record<CollectionName, [number, number]>> = {
  principles: [-0.9, -0.55],
  concepts: [-0.75, 0],
  experiments: [-0.85, 0.6],
  applications: [-0.3, 0.95],
  'failure-cases': [0.15, 0.9],
  claims: [0.25, 0.25],
  papers: [0.85, 0.15],
  'model-families': [-0.2, -0.75],
  implementations: [-0.55, -0.95],
  checkpoints: [0.25, -0.8],
  datasets: [0.75, -0.6],
  'levels-of-theory': [0.95, -0.95],
  benchmarks: [0.9, 0.7],
  'benchmark-results': [0.6, 0.95],
};

export function buildAtlasMap(records: Map<string, LoadedRecord>, size = 1000): AtlasMap {
  const nodes: Array<MapNode & SimulationNodeDatum> = [];
  const index = new Map<string, number>();
  for (const [k, r] of records) {
    index.set(k, nodes.length);
    nodes.push({
      id: k,
      collection: r.collection,
      title: recordTitle(r.data),
      status: r.data.review.status,
      url: recordUrl(r.collection, r.id, r.data),
      degree: 0,
      x: 0,
      y: 0,
    });
  }
  const seen = new Set<string>();
  const edges: MapEdge[] = [];
  for (const [k, r] of records)
    for (const ref of r.refs) {
      const t = `${ref.target}/${ref.id}`;
      if (!index.has(t) || t === k) continue;
      const e = k < t ? `${k}|${t}` : `${t}|${k}`;
      if (seen.has(e)) continue;
      seen.add(e);
      edges.push({ source: k, target: t });
      nodes[index.get(k)!]!.degree++;
      nodes[index.get(t)!]!.degree++;
    }

  // Deterministic start: each node near its collection anchor, spread by a seeded RNG.
  const rand = mulberry32(1914);
  const R = size * 0.42;
  for (const n of nodes) {
    const [ax, ay] = ANCHOR[n.collection] ?? [0, 0];
    n.x = ax * R + (rand() - 0.5) * 60;
    n.y = ay * R + (rand() - 0.5) * 60;
  }
  const links = edges.map((e) => ({ source: index.get(e.source)!, target: index.get(e.target)! }));
  const sim = forceSimulation(nodes)
    .randomSource(mulberry32(7))
    .force('link', forceLink(links).distance(46).strength(0.25))
    .force('charge', forceManyBody().strength(-60).distanceMax(320))
    .force('collide', forceCollide<MapNode & SimulationNodeDatum>((n) => 7 + Math.sqrt(n.degree) * 2.2))
    .force('x', forceX<MapNode & SimulationNodeDatum>((n) => (ANCHOR[n.collection]?.[0] ?? 0) * R).strength(0.07))
    .force('y', forceY<MapNode & SimulationNodeDatum>((n) => (ANCHOR[n.collection]?.[1] ?? 0) * R).strength(0.07))
    .stop();
  for (let i = 0; i < 500; i++) sim.tick();

  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      collection: n.collection,
      title: n.title,
      status: n.status,
      url: n.url,
      degree: n.degree,
      x: Math.round((n.x ?? 0) * 10) / 10,
      y: Math.round((n.y ?? 0) * 10) / 10,
    })),
    edges,
  };
}

export const COLLECTION_TONES = Object.fromEntries(
  Object.entries(COLLECTION_META).map(([c, m]) => [c, { label: m.label, tone: m.tone }]),
) as Record<CollectionName, { label: string; tone: string }>;
