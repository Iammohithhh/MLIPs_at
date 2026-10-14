import { describe, expect, it } from 'vitest';
import { COLLECTION_NAMES, SCHEMAS, collectRefs, schemaRelations } from './schemas';

const review = { status: 'draft', last_checked: '2026-10-14' };

describe('schema relations (derived from ref() fields)', () => {
  const rel = schemaRelations();
  const has = (from: string, field: string, to: string) =>
    rel.some((r) => r.from === from && r.field === field && r.to === to);

  it('finds the core chain architecture → checkpoint → dataset → level of theory', () => {
    expect(has('checkpoints', 'family', 'model-families')).toBe(true);
    expect(has('checkpoints', 'training[].dataset', 'datasets')).toBe(true);
    expect(has('checkpoints', 'training[].level_of_theory', 'levels-of-theory')).toBe(true);
    expect(has('datasets', 'levels_of_theory[]', 'levels-of-theory')).toBe(true);
  });

  it('finds refs through unions, optionals and nested arrays', () => {
    expect(has('checkpoints', 'release.source', 'papers')).toBe(true);
    expect(has('claims', 'sources[].paper', 'papers')).toBe(true);
    expect(has('claims', 'reproduction.experiment', 'experiments')).toBe(true);
    expect(has('benchmark-results', 'benchmark', 'benchmarks')).toBe(true);
  });

  it('every collection takes part in at least one relation', () => {
    for (const c of COLLECTION_NAMES) {
      expect(rel.some((r) => r.from === c || r.to === c), c).toBe(true);
    }
  });
});

describe('per-record rules', () => {
  it('collects references while parsing', () => {
    const { result, refs } = collectRefs(() =>
      SCHEMAS.claims.safeParse({
        statement: 'x',
        grade: 'PUBLISHED_RESULT',
        sources: [{ paper: 'p1', locator: 'Fig. 1' }],
        verified_against: 'full-text',
        review,
      }),
    );
    expect(result.success).toBe(true);
    expect(refs).toEqual([{ target: 'papers', id: 'p1' }]);
  });

  it('rejects a PUBLISHED_RESULT claim without sources', () => {
    const r = SCHEMAS.claims.safeParse({ statement: 'x', grade: 'PUBLISHED_RESULT', review });
    expect(r.success).toBe(false);
  });

  it('rejects a THEORY claim with neither source nor derivation', () => {
    expect(SCHEMAS.claims.safeParse({ statement: 'x', grade: 'THEORY', review }).success).toBe(false);
    expect(SCHEMAS.claims.safeParse({ statement: 'x', grade: 'THEORY', derivation: '/', review }).success).toBe(true);
  });

  it('rejects REPRODUCED without a matching reproduction', () => {
    const base = {
      statement: 'x',
      grade: 'REPRODUCED',
      sources: [{ paper: 'p1', locator: 'Fig. 1' }],
      verified_against: 'full-text',
      review,
    };
    expect(SCHEMAS.claims.safeParse(base).success).toBe(false);
    expect(
      SCHEMAS.claims.safeParse({ ...base, reproduction: { experiment: 'e', date: '2026-01-01', matches: false } }).success,
    ).toBe(false);
    expect(
      SCHEMAS.claims.safeParse({ ...base, reproduction: { experiment: 'e', date: '2026-01-01', matches: true } }).success,
    ).toBe(true);
  });

  it('requires verified_against when a claim has sources', () => {
    const r = SCHEMAS.claims.safeParse({
      statement: 'x',
      grade: 'THEORY',
      sources: [{ paper: 'p1' }],
      review,
    });
    expect(r.success).toBe(false);
  });

  it('rejects unknown fields (typos never pass silently)', () => {
    const r = SCHEMAS.claims.safeParse({ statement: 'x', grade: 'THEORY', derivation: '/', review, sorces: [] });
    expect(r.success).toBe(false);
  });

  it('normalises YAML Date objects to ISO dates', () => {
    const r = SCHEMAS.claims.safeParse({
      statement: 'x',
      grade: 'THEORY',
      derivation: '/',
      review: { status: 'draft', last_checked: new Date('2026-10-14T00:00:00Z') },
    });
    expect(r.success && r.data.review.last_checked).toBe('2026-10-14');
  });

  it('papers need at least one identifier and well-formed ones', () => {
    const base = { title: 't', authors: ['A, B.'], year: 2020, type: 'article', verified_against: 'secondary', review };
    expect(SCHEMAS.papers.safeParse({ ...base, identifiers: {} }).success).toBe(false);
    expect(SCHEMAS.papers.safeParse({ ...base, identifiers: { arxiv: '2206.07697' } }).success).toBe(true);
    expect(SCHEMAS.papers.safeParse({ ...base, identifiers: { arxiv: 'arXiv:2206.07697' } }).success).toBe(false);
    expect(SCHEMAS.papers.safeParse({ ...base, identifiers: { doi: 'https://doi.org/10.1/x' } }).success).toBe(false);
  });

  it('leaderboard results need a url; benchmark protocol ids must be unique', () => {
    const r = SCHEMAS['benchmark-results'].safeParse({
      benchmark: 'b',
      protocol: 'p',
      metric: 'm',
      checkpoint: 'c',
      value: 0.9,
      retrieved: '2026-10-14',
      status: 'PUBLISHED_RESULT',
      source: { kind: 'leaderboard' },
      review,
    });
    expect(r.success).toBe(false);
    const b = SCHEMAS.benchmarks.safeParse({
      name: 'B',
      summary: 's',
      introduced_in: 'p',
      domain: ['molecules'],
      metrics: [{ id: 'm', name: 'M', unit: '', higher_is_better: true }],
      protocols: [
        { id: 'p', name: 'P', description: 'd' },
        { id: 'p', name: 'P2', description: 'd' },
      ],
      review,
    });
    expect(b.success).toBe(false);
  });
});
