import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { BIBLIOGRAPHY, authorList, authorShort, citeLabel, venueLine } from './bibliography';

const papersDir = fileURLToPath(new URL('../../content/papers', import.meta.url));
const ids = readdirSync(papersDir)
  .filter((f) => f.endsWith('.yaml'))
  .map((f) => f.replace(/\.yaml$/, ''));

describe('bibliography cache', () => {
  it('has fetched metadata for every paper record (run the bibliography pipeline)', () => {
    expect(ids.filter((id) => !BIBLIOGRAPHY[id])).toEqual([]);
  });

  it('every paper record agrees with Crossref/arXiv on title, year and first author', () => {
    const warns = ids.flatMap((id) =>
      (BIBLIOGRAPHY[id]?.checks ?? []).filter((c) => c.level === 'warn').map((c) => `${id}: ${c.message}`),
    );
    expect(warns).toEqual([]);
  });

  it('citation links only point at Atlas papers', () => {
    for (const [id, e] of Object.entries(BIBLIOGRAPHY)) {
      for (const c of e.cites) {
        expect(BIBLIOGRAPHY[c], `${id} cites unknown ${c}`).toBeDefined();
        expect(c).not.toBe(id);
      }
    }
  });
});

describe('formatting', () => {
  const one = { authors: ['Behler, J.'], et_al: false, year: 2011 };
  const two = { authors: ['Behler, J.', 'Parrinello, M.'], et_al: false, year: 2007 };
  const many = { authors: ['Batatia, I.', 'Kovács, D. P.', 'Simm, G. N. C.'], et_al: false, year: 2022 };
  it('formats short labels', () => {
    expect(citeLabel(one)).toBe('Behler 2011');
    expect(authorShort(two)).toBe('Behler & Parrinello');
    expect(citeLabel(many)).toBe('Batatia et al. 2022');
  });

  it('uses Crossref for venues and full author lists', () => {
    const p = { authors: ['Behler, J.'], et_al: false, year: 2011, identifiers: { doi: '10.1063/1.3553717' } };
    const b = BIBLIOGRAPHY['behler2011symmetry'];
    expect(venueLine(p, b)).toMatch(/Chemical Physics 134, 074106 \(2011\)/);
    expect(authorList(p, b)).toMatch(/^Behler, J/);
  });
});
