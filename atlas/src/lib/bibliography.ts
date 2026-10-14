// Bibliographic metadata fetched by pipeline/atlas_pipeline/bibliography.py
// (Crossref, arXiv, OpenAlex) and helpers to format citations. The paper YAML
// records stay the reviewed source of truth; this cache adds full author lists,
// containers and citation links, and checks the records against the sources.

import bib from '~/data/bibliography.json';
import type { Paper } from '~/content/schemas';

export interface BibEntry {
  fetched: string;
  crossref: {
    doi: string;
    title: string;
    authors: Array<{ family: string; given: string }>;
    container: string;
    volume: string | null;
    issue: string | null;
    page: string | null;
    article_number: string | null;
    year: number | null;
    type: string | null;
  } | null;
  arxiv: { id: string; title: string; authors: string[]; published: string; updated: string; primary_category: string | null } | null;
  openalex: { id: string; publication_year: number | null; cited_by_count: number | null; referenced_works: string[] } | null;
  /** Atlas paper ids this paper cites (from OpenAlex reference lists). */
  cites: string[];
  checks: Array<{ level: 'ok' | 'warn'; field: string; message: string }>;
}

export const BIBLIOGRAPHY = bib as unknown as Record<string, BibEntry>;

export function bibEntry(id: string): BibEntry | undefined {
  return BIBLIOGRAPHY[id];
}

/** "Batatia et al." / "Behler & Parrinello" / "Behler". */
export function authorShort(p: Pick<Paper, 'authors' | 'et_al'>): string {
  const fam = (a: string) => a.split(',')[0]!.trim();
  if (p.authors.length === 1 && !p.et_al) return fam(p.authors[0]!);
  if (p.authors.length === 2 && !p.et_al) return `${fam(p.authors[0]!)} & ${fam(p.authors[1]!)}`;
  return `${fam(p.authors[0]!)} et al.`;
}

export function citeLabel(p: Pick<Paper, 'authors' | 'et_al' | 'year'>): string {
  return `${authorShort(p)} ${p.year}`;
}

/** Full author list: from Crossref when available (complete and ordered), else the record. */
export function authorList(p: Pick<Paper, 'authors' | 'et_al'>, b?: BibEntry, max = 8): string {
  const names = b?.crossref?.authors.length
    ? b.crossref.authors.map((a) => `${a.family}, ${a.given.replace(/([A-Z])[a-zà-ÿ]+/g, '$1.').replace(/\.(?=[A-Z])/g, '. ')}`.trim())
    : b?.arxiv?.authors.length
      ? b.arxiv.authors.map((n) => {
          const parts = n.split(' ');
          const fam = parts.pop()!;
          return `${fam}, ${parts.map((x) => `${x[0]}.`).join(' ')}`;
        })
      : p.authors;
  const etAl = (p.et_al && !b?.crossref && !b?.arxiv) || names.length > max;
  return names.slice(0, max).join('; ') + (etAl ? ' et al.' : '');
}

export function primaryUrl(p: Pick<Paper, 'identifiers'>): string | undefined {
  if (p.identifiers.doi) return `https://doi.org/${p.identifiers.doi}`;
  if (p.identifiers.arxiv) return `https://arxiv.org/abs/${p.identifiers.arxiv}`;
  return p.identifiers.url;
}

/** One-line venue: container volume, pages/article (year), falling back to the record. */
export function venueLine(p: Pick<Paper, 'venue' | 'year' | 'identifiers'>, b?: BibEntry): string {
  const c = b?.crossref;
  if (c?.container) {
    const loc = c.article_number ?? c.page;
    return `${c.container}${c.volume ? ` ${c.volume}` : ''}${loc ? `, ${loc}` : ''} (${c.year ?? p.year})`;
  }
  if (p.venue) return `${p.venue} (${p.year})`;
  if (p.identifiers.arxiv) return `arXiv:${p.identifiers.arxiv} (${p.year})`;
  return String(p.year);
}
