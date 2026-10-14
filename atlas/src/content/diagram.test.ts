import { describe, expect, it } from 'vitest';
import { diagramEdges, exitPoint, groupRelations } from './diagram';
import { COLLECTION_META, NODE } from './meta';
import { COLLECTION_NAMES, schemaRelations } from './schemas';

describe('content-model diagram', () => {
  it('has a non-overlapping node for every collection inside the viewBox', () => {
    const boxes = COLLECTION_NAMES.map((c) => ({ c, ...COLLECTION_META[c].pos }));
    for (const b of boxes) {
      expect(b.x).toBeGreaterThanOrEqual(0);
      expect(b.y).toBeGreaterThanOrEqual(30);
      expect(b.x + NODE.w).toBeLessThanOrEqual(1040);
      expect(b.y + NODE.h).toBeLessThanOrEqual(640);
    }
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i]!;
        const b = boxes[j]!;
        const overlap = a.x < b.x + NODE.w && b.x < a.x + NODE.w && a.y < b.y + NODE.h && b.y < a.y + NODE.h;
        expect(overlap, `${a.c} overlaps ${b.c}`).toBe(false);
      }
    }
  });

  it('groups relations into one edge per collection pair, keeping every field', () => {
    const rel = schemaRelations();
    const groups = groupRelations(rel);
    const total = groups.reduce((n, g) => n + g.fields.length, 0);
    expect(total).toBe(new Set(rel.map((r) => `${r.from}.${r.field}→${r.to}`)).size);
    const ck = groups.find((g) => g.from === 'checkpoints' && g.to === 'claims');
    expect(ck?.fields).toEqual(expect.arrayContaining(['known_issues[]', 'claims[]']));
  });

  it('draws every edge as a finite path', () => {
    for (const e of diagramEdges(schemaRelations())) {
      expect(e.d).toMatch(/^M[-\d.]+,[-\d.]+ [QC]/);
      expect(e.d).not.toMatch(/NaN|Infinity/);
    }
  });

  it('exit points lie on (just outside) the box boundary', () => {
    const b = { x: 0, y: 0, w: 100, h: 40 };
    expect(exitPoint(b, 1, 0, 0)).toEqual({ x: 100, y: 20 });
    expect(exitPoint(b, 0, -1, 0)).toEqual({ x: 50, y: 0 });
    const p = exitPoint(b, 1, 1, 0);
    expect(p.y).toBeCloseTo(40, 10);
  });
});
