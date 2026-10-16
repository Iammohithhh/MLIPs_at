import { describe, expect, it } from 'vitest';
import { COLLECTION_NAMES } from '~/content/schemas';
import { collectionSchema, snapshots } from './open-data';

describe('open data', () => {
  it('every collection has a JSON Schema', () => {
    for (const c of COLLECTION_NAMES) {
      const s = collectionSchema(c);
      expect(s['type']).toBe('object');
      expect(s['$id']).toBe(`/api/v1/schema/${c}.json`);
    }
  });

  it('every benchmark snapshot records where it came from', () => {
    const all = snapshots();
    expect(all.length).toBeGreaterThan(0);
    for (const s of all) expect(s.source, s.name).toBeTruthy();
  });
});
