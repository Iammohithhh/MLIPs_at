// Open-data index: what is available under /api/v1/ and how many records each collection holds.
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { COLLECTION_NAMES } from '~/content/schemas';
import { API_BASE, snapshots } from '~/lib/open-data';

export const GET: APIRoute = async () => {
  const collections = [];
  for (const c of COLLECTION_NAMES) {
    const entries = await getCollection(c);
    collections.push({
      name: c,
      count: entries.length,
      published: entries.filter((e) => (e.data as { review: { status: string } }).review.status === 'published').length,
      records: `${API_BASE}/${c}.json`,
      schema: `${API_BASE}/schema/${c}.json`,
    });
  }
  const body = {
    name: 'MLIP Atlas open data',
    version: 'v1',
    generated: new Date().toISOString(),
    notes: [
      'Records are the Atlas content as authored, validated against the schemas below; review.status says whether a record has passed review.',
      'Snapshots are numbers copied by tested pipeline modules from published sources; each carries its source and retrieval date.',
    ],
    collections,
    snapshots: snapshots().map((s) => ({ name: s.name, url: `${API_BASE}/snapshots/${s.name}.json`, source: s.source })),
  };
  return new Response(JSON.stringify(body, null, 1), { headers: { 'Content-Type': 'application/json' } });
};
