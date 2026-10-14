// Display metadata for each collection: label, semantic tone, one-line role, and
// where its node sits in the content-model diagram (viewBox 1040 × 640).

import type { Tone } from '~/lib/labels';
import type { CollectionName } from './schemas';

export interface CollectionMeta {
  label: string;
  tone: Tone;
  role: string;
  /** Top-left corner of the node in the content-model diagram. */
  pos: { x: number; y: number };
}

export const COLLECTION_META: Record<CollectionName, CollectionMeta> = {
  principles: { label: 'Physical principles', tone: 'phys', role: 'The physics approximated', pos: { x: 30, y: 40 } },
  concepts: { label: 'Concepts', tone: 'phys', role: 'Ideas linking physics and models', pos: { x: 30, y: 210 } },
  experiments: { label: 'Experiments', tone: 'phys', role: 'Labs, figures, scans', pos: { x: 30, y: 380 } },
  applications: { label: 'Applications', tone: 'phys', role: 'Systems people care about', pos: { x: 30, y: 550 } },
  'model-families': { label: 'Model families', tone: 'arch', role: 'Architectures, no weights', pos: { x: 292, y: 40 } },
  implementations: { label: 'Implementations', tone: 'arch', role: 'Code, version, licence', pos: { x: 292, y: 210 } },
  'failure-cases': { label: 'Failure cases', tone: 'fail', role: 'Evidence-graded limits', pos: { x: 292, y: 550 } },
  checkpoints: { label: 'Checkpoints', tone: 'arch', role: 'Specific trained weights', pos: { x: 554, y: 40 } },
  'benchmark-results': { label: 'Benchmark results', tone: 'lit', role: 'One number, one protocol', pos: { x: 554, y: 295 } },
  claims: { label: 'Claims', tone: 'lit', role: 'Statements with locators', pos: { x: 554, y: 550 } },
  datasets: { label: 'Datasets', tone: 'neutral', role: 'What checkpoints learn from', pos: { x: 816, y: 40 } },
  'levels-of-theory': { label: 'Levels of theory', tone: 'neutral', role: 'Functional, corrections, code', pos: { x: 816, y: 210 } },
  benchmarks: { label: 'Benchmarks', tone: 'lit', role: 'Tasks, metrics, protocols', pos: { x: 816, y: 380 } },
  papers: { label: 'Papers', tone: 'lit', role: 'Sources of ideas and results', pos: { x: 816, y: 550 } },
};

export const NODE = { w: 194, h: 62 } as const;

/** A short human-readable title for any record. */
export function recordTitle(data: Record<string, unknown>): string {
  const t = data['name'] ?? data['title'] ?? data['statement'];
  if (typeof t === 'string') return t;
  if (typeof data['benchmark'] === 'string' && typeof data['checkpoint'] === 'string') {
    return `${data['checkpoint']} on ${data['benchmark']} (${String(data['metric'])}, ${String(data['protocol'])})`;
  }
  return '(untitled)';
}
