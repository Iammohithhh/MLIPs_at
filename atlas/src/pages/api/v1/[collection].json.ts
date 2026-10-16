// All records of one collection, with their site URL.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';
import { COLLECTION_NAMES, type CollectionName } from '~/content/schemas';
import { recordUrl } from '~/lib/record-urls';
import { API_BASE } from '~/lib/open-data';

export const getStaticPaths: GetStaticPaths = () => COLLECTION_NAMES.map((collection) => ({ params: { collection } }));

export const GET: APIRoute = async ({ params }) => {
  const c = params.collection as CollectionName;
  const records = (await getCollection(c)).map((e) => {
    const d = e.data as Record<string, unknown>;
    return { id: e.id, url: recordUrl(c, e.id, d), ...d };
  });
  const body = { collection: c, schema: `${API_BASE}/schema/${c}.json`, count: records.length, records };
  return new Response(JSON.stringify(body, null, 1), { headers: { 'Content-Type': 'application/json' } });
};
