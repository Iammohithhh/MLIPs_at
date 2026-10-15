// Static search index of every Atlas record (built at build time). The ⌘K
// palette loads it lazily and searches it with Orama alongside Pagefind's
// full-text index of the pages.
import type { APIRoute } from 'astro';
import { recordUrl } from '~/lib/record-urls';
import { getCollection } from 'astro:content';
import { COLLECTION_NAMES, type CollectionName } from '~/content/schemas';
import { COLLECTION_META, recordTitle } from '~/content/meta';

export interface RecordDoc {
  id: string;
  collection: CollectionName;
  kind: string;
  title: string;
  text: string;
  status: string;
  url: string;
}

function textOf(d: Record<string, unknown>): string {
  const fields = ['summary', 'statement', 'question', 'venue', 'notes', 'functional', 'body_order'];
  const parts = fields.map((f) => d[f]).filter((v): v is string => typeof v === 'string');
  if (Array.isArray(d['authors'])) parts.push((d['authors'] as string[]).join('; '));
  if (Array.isArray(d['aliases'])) parts.push((d['aliases'] as string[]).join(' '));
  return parts.join(' — ').slice(0, 600);
}

const urlOf = recordUrl;

export const GET: APIRoute = async () => {
  const docs: RecordDoc[] = [];
  for (const c of COLLECTION_NAMES) {
    for (const e of await getCollection(c)) {
      const d = e.data as Record<string, unknown> & { review: { status: string } };
      docs.push({
        id: e.id,
        collection: c,
        kind: COLLECTION_META[c].label,
        title: recordTitle(d),
        text: textOf(d),
        status: d.review.status,
        url: urlOf(c, e.id, d),
      });
    }
  }
  return new Response(JSON.stringify(docs), { headers: { 'Content-Type': 'application/json' } });
};
