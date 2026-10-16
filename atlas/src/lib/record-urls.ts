// Where each Atlas record is best shown on the site. Used by search and the atlas map.
import type { CollectionName } from '~/content/schemas';
import { modelPageUrl } from './model-pages';

/** Applications with a built lab page under /applications/<id>/. */
export const APPLICATION_LABS = new Set(['electrochemical-interfaces', 'batteries']);

export function recordUrl(c: CollectionName, id: string, d: Record<string, unknown>): string {
  if (c === 'papers') return `/papers/#paper-${id}`;
  if (c === 'experiments' && typeof d['page'] === 'string') return d['page'];
  if (c === 'model-families' && modelPageUrl(id)) return modelPageUrl(id)!;
  if (c === 'datasets') return `/data/#ds-${id}`;
  if (c === 'failure-cases') return `/failures/${id}/`;
  if (c === 'applications') return APPLICATION_LABS.has(id) ? `/applications/${id}/` : `/applications/`;
  return `/method/content-model/#rec-${c}-${id}`;
}
