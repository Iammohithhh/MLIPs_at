import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseManifest } from './experiment';
import { baselineNeighbourCount, stagesFor } from './stages';

const load = (id: string, f: string) => JSON.parse(readFileSync(new URL(`../data/experiments/${id}/${f}`, import.meta.url), 'utf8'));

describe('stage texts follow the experiment', () => {
  for (const id of ['si-bond-stretch', 'zno-axial-bond']) {
    const m = parseManifest(load(id, 'manifest.json'));
    const chunk = load(id, `frames/${String(m.displacement.baselineIndex).padStart(2, '0')}.json`);
    it(`${id}: neighbour count equals the stored edges into the moved atom at the baseline`, () => {
      expect(baselineNeighbourCount(m, m.displacement.atom)).toBe(chunk.edgesIntoMoved.src.length);
    });
    it(`${id}: no placeholder or foreign atom count survives`, () => {
      const text = JSON.stringify(stagesFor(m));
      expect(text).not.toMatch(/%(N|COMP|EL|NB)%/);
      expect(text).toContain(`[${m.structure.n}, 3]`);
      expect(text).toContain(`${m.structure.n} atoms`);
      if (m.structure.n !== 216) expect(text).not.toContain('216');
    });
  }
});
