// JSON Schema of one collection's records.
import type { APIRoute, GetStaticPaths } from 'astro';
import { COLLECTION_NAMES, type CollectionName } from '~/content/schemas';
import { collectionSchema } from '~/lib/open-data';

export const getStaticPaths: GetStaticPaths = () => COLLECTION_NAMES.map((collection) => ({ params: { collection } }));

export const GET: APIRoute = ({ params }) =>
  new Response(JSON.stringify(collectionSchema(params.collection as CollectionName), null, 1), { headers: { 'Content-Type': 'application/schema+json' } });
