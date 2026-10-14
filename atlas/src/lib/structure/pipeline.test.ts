import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { displayBonds, neighboursOf } from './geometry';
import { structureSchema } from './schema';
import type { Structure } from './types';

const dir = fileURLToPath(new URL('../../data/structures', import.meta.url));
const files = readdirSync(dir).filter((f) => f.endsWith('.json') && f !== 'index.json');
const load = (id: string) => JSON.parse(readFileSync(`${dir}/${id}.json`, 'utf8')) as Structure;

describe('pipeline structures', () => {
  it.each(files)('%s matches the Atlas structure schema', (f) => {
    const r = structureSchema.safeParse(JSON.parse(readFileSync(`${dir}/${f}`, 'utf8')));
    expect(r.success, JSON.stringify(r.error?.issues)).toBe(true);
  });

  it('TS geometry agrees with ASE on rock-salt coordination', () => {
    const s = load('nacl-rocksalt');
    s.positions.forEach((_, i) => {
      const nb = neighboursOf(s, i, 3);
      expect(nb).toHaveLength(6);
      nb.forEach((n) => expect(n.distance).toBeCloseTo(2.82, 5));
    });
  });

  it('draws 4 bonds per Si and none across the slab vacuum', () => {
    expect(displayBonds(load('si-diamond'))).toHaveLength(16);
    const slab = load('au111-slab');
    const z = slab.positions.map((p) => p[2]);
    const span = Math.max(...z) - Math.min(...z);
    for (const b of displayBonds(slab)) {
      expect(b.image[2]).toBe(0);
      expect(b.distance).toBeLessThan(3.2);
    }
    expect(span).toBeLessThan(slab.cell![2][2]);
  });
});
