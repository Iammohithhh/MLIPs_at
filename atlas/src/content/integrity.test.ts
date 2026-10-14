import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { auditContent, checkIntegrity, validateRecords, type RawRecord } from './integrity';

const contentRoot = fileURLToPath(new URL('../../content', import.meta.url));
const pagesRoot = fileURLToPath(new URL('../pages', import.meta.url));

describe('the Atlas content', () => {
  it('passes schema and integrity checks', () => {
    const { records, problems } = auditContent(contentRoot, pagesRoot);
    expect(problems).toEqual([]);
    expect(records.size).toBeGreaterThan(0);
  });
});

// Small synthetic graphs, one rule violation each.
const review = (status = 'draft') => ({ status, last_checked: '2026-10-14' });
const paper = (id: string, extra: object = {}): RawRecord => ({
  collection: 'papers',
  id,
  file: `papers/${id}.yaml`,
  raw: {
    title: 'T',
    authors: ['A, B.'],
    year: 2020,
    type: 'article',
    identifiers: { arxiv: '2001.00001' },
    verified_against: 'full-text',
    review: review('published'),
    ...extra,
  },
});
const claim = (id: string, extra: object = {}): RawRecord => ({
  collection: 'claims',
  id,
  file: `claims/${id}.yaml`,
  raw: {
    statement: 'S',
    grade: 'PUBLISHED_RESULT',
    sources: [{ paper: 'p1', locator: 'Fig. 2' }],
    verified_against: 'full-text',
    review: review('published'),
    ...extra,
  },
});

function problemsFor(raws: RawRecord[]) {
  const v = validateRecords(raws);
  return [...v.problems, ...checkIntegrity(v.records, pagesRoot)].map((p) => `${p.file}: ${p.message}`);
}

describe('integrity rules', () => {
  it('a clean graph has no problems', () => {
    expect(problemsFor([paper('p1'), claim('c1')])).toEqual([]);
  });

  it('flags references to records that do not exist', () => {
    expect(problemsFor([claim('c1')])).toEqual(['claims/c1.yaml: unknown papers id "p1"']);
  });

  it('flags published records that rest on unpublished ones', () => {
    const out = problemsFor([paper('p1', { review: review('draft') }), claim('c1')]);
    expect(out).toEqual(['claims/c1.yaml: published record references papers/p1, which is draft']);
  });

  it('flags publication of sources checked only against abstracts', () => {
    const out = problemsFor([paper('p1'), claim('c1', { verified_against: 'abstract' })]);
    expect(out).toEqual(['claims/c1.yaml: cannot be published while only verified against abstract']);
  });

  it('flags published claims whose sources lack locators', () => {
    const out = problemsFor([paper('p1'), claim('c1', { sources: [{ paper: 'p1' }] })]);
    expect(out).toEqual(['claims/c1.yaml: sources.0: published claims need a figure/table/section locator']);
  });

  it('allows drafts to have open TODOs', () => {
    const out = problemsFor([
      paper('p1', { review: review('draft'), verified_against: 'abstract' }),
      claim('c1', { sources: [{ paper: 'p1' }], verified_against: 'abstract', review: review('draft') }),
    ]);
    expect(out).toEqual([]);
  });

  it('flags derivations and experiment pages that do not exist', () => {
    const out = problemsFor([
      claim('c1', { grade: 'THEORY', sources: [], verified_against: undefined, derivation: '/no/such/page/' }),
    ]);
    expect(out).toEqual(['claims/c1.yaml: derivation page "/no/such/page/" does not exist']);
  });

  it('flags benchmark results whose protocol or metric the benchmark does not define', () => {
    const raws: RawRecord[] = [
      paper('p1'),
      {
        collection: 'benchmarks',
        id: 'b1',
        file: 'benchmarks/b1.yaml',
        raw: {
          name: 'B',
          summary: 's',
          introduced_in: 'p1',
          domain: ['molecules'],
          metrics: [{ id: 'mae', name: 'MAE', unit: 'meV/atom', higher_is_better: false }],
          protocols: [{ id: 'standard', name: 'Std', description: 'd' }],
          review: review('draft'),
        },
      },
      {
        collection: 'model-families',
        id: 'f1',
        file: 'model-families/f1.yaml',
        raw: {
          name: 'F',
          paradigms: ['equivariant'],
          introduced_in: 'p1',
          year: 2022,
          summary: 's',
          locality: { kind: 'message-passing' },
          symmetry: 'equivariant',
          review: review('draft'),
        },
      },
      {
        collection: 'levels-of-theory',
        id: 'pbe',
        file: 'levels-of-theory/pbe.yaml',
        raw: { name: 'PBE', method: 'DFT', review: review('draft') },
      },
      {
        collection: 'levels-of-theory',
        id: 'r2scan',
        file: 'levels-of-theory/r2scan.yaml',
        raw: { name: 'r2SCAN', method: 'meta-GGA-DFT', review: review('draft') },
      },
      {
        collection: 'datasets',
        id: 'd1',
        file: 'datasets/d1.yaml',
        raw: {
          name: 'D',
          domain: ['molecules'],
          summary: 's',
          introduced_in: 'p1',
          levels_of_theory: ['pbe'],
          review: review('draft'),
        },
      },
      {
        collection: 'implementations',
        id: 'i1',
        file: 'implementations/i1.yaml',
        raw: {
          name: 'i',
          repo: 'https://example.org/i',
          language: 'Python',
          framework: 'pytorch',
          licence: 'MIT',
          families: ['f1'],
          review: review('draft'),
        },
      },
      {
        collection: 'checkpoints',
        id: 'ck1',
        file: 'checkpoints/ck1.yaml',
        raw: {
          name: 'CK',
          family: 'f1',
          implementation: 'i1',
          release: { source: 'p1' },
          training: [{ stage: 'training', dataset: 'd1', level_of_theory: 'r2scan' }],
          outputs: ['energy', 'forces'],
          licence: 'MIT',
          review: review('draft'),
        },
      },
      {
        collection: 'benchmark-results',
        id: 'r1',
        file: 'benchmark-results/r1.yaml',
        raw: {
          benchmark: 'b1',
          protocol: 'other',
          metric: 'rmse',
          checkpoint: 'ck1',
          value: 1,
          retrieved: '2026-10-14',
          status: 'PUBLISHED_RESULT',
          source: { kind: 'paper', paper: 'p1', locator: 'Table 1' },
          review: review('draft'),
        },
      },
    ];
    expect(problemsFor(raws)).toEqual([
      'checkpoints/ck1.yaml: training.0: dataset "d1" does not provide level of theory "r2scan"',
      'benchmark-results/r1.yaml: protocol "other" is not defined by benchmark "b1"',
      'benchmark-results/r1.yaml: metric "rmse" is not defined by benchmark "b1"',
    ]);
  });

  it('reports schema errors with their field path', () => {
    const out = problemsFor([paper('p1', { year: 'twenty' })]);
    expect(out[0]).toMatch(/^papers\/p1\.yaml: year: /);
  });

  it('rejects non-kebab-case file names', () => {
    const r = paper('p1');
    expect(problemsFor([{ ...r, id: 'Bad_Name' }])).toEqual(['papers/p1.yaml: file name "Bad_Name" is not a kebab-case id']);
  });
});
