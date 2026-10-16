// Open data: static JSON dumps of every Atlas record, their JSON Schemas, and the benchmark snapshots
// that pipeline modules copy from published sources. Served at build time under /api/v1/.
import { z } from 'zod';
import { SCHEMAS, type CollectionName } from '~/content/schemas';

export const API_VERSION = 'v1';
export const API_BASE = `/api/${API_VERSION}`;

/** JSON Schema (draft 2020-12) of one collection's records, as authors write them. */
export function collectionSchema(c: CollectionName): Record<string, unknown> {
  const s = z.toJSONSchema(SCHEMAS[c] as z.ZodType, { unrepresentable: 'any', io: 'input' }) as Record<string, unknown>;
  return { ...s, $id: `${API_BASE}/schema/${c}.json`, title: `MLIP Atlas ${c} record` };
}

const SNAPSHOT_FILES = import.meta.glob<{ default: Record<string, unknown> }>('../data/benchmarks/*.json', { eager: true });

export interface Snapshot {
  name: string;
  data: Record<string, unknown>;
  source: unknown;
}

/** Benchmark snapshots written by pipeline modules, keyed by file name without extension. */
export function snapshots(): Snapshot[] {
  return Object.entries(SNAPSHOT_FILES)
    .map(([path, mod]) => {
      const name = path.split('/').pop()!.replace(/\.json$/, '');
      const data = mod.default;
      return { name, data, source: (data as { source?: unknown }).source ?? null };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}
