// One benchmark snapshot exactly as the pipeline wrote it.
import type { APIRoute, GetStaticPaths } from 'astro';
import { snapshots } from '~/lib/open-data';

export const getStaticPaths: GetStaticPaths = () => snapshots().map((s) => ({ params: { name: s.name } }));

export const GET: APIRoute = ({ params }) => {
  const s = snapshots().find((x) => x.name === params.name)!;
  return new Response(JSON.stringify(s.data), { headers: { 'Content-Type': 'application/json' } });
};
