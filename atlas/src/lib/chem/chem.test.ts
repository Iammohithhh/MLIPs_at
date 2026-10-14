import { describe, expect, it } from 'vitest';
import { auWaterInterface, dist, minImage, rockSaltCluster, SPCE, AU_LATTICE, type Vec3 } from './structures';
import { countWithin, cutoffGraph, hopDistances, reachCounts } from './graph';
import { mulberry32, randomRotation } from './rng';

const info = auWaterInterface();
const s = info.structure;

describe('rng', () => {
  it('is deterministic', () => {
    const a = mulberry32(1);
    const b = mulberry32(1);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
  it('randomRotation is orthonormal', () => {
    const R = randomRotation(mulberry32(3));
    const row = (i: number) => [R[3 * i]!, R[3 * i + 1]!, R[3 * i + 2]!];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const d = row(i).reduce((acc, v, k) => acc + v * row(j)[k]!, 0);
        expect(d).toBeCloseTo(i === j ? 1 : 0, 10);
      }
  });
});

describe('Au(111)/water/NaCl interface', () => {
  it('is identical on every build', () => {
    expect(auWaterInterface().structure.atoms).toEqual(s.atoms);
  });

  it('has a charge-neutral toy charge set', () => {
    const q = s.atoms.reduce((acc, a) => acc + a.q, 0);
    expect(Math.abs(q)).toBeLessThan(1e-9);
  });

  it('has the fcc(111) surface nearest-neighbour distance a/√2', () => {
    const au = s.atoms.filter((a) => a.el === 'Au');
    const first = au[0]!.p;
    const nn = Math.min(...au.slice(1).map((a) => dist(first, a.p, s.cell, s.pbc)));
    expect(nn).toBeCloseTo(AU_LATTICE / Math.SQRT2, 6);
  });

  it('builds SPC/E waters (O–H 1.0 Å, H–O–H 109.47°)', () => {
    for (const [o, h] of s.bonds) expect(dist(s.atoms[o]!.p, s.atoms[h]!.p, s.cell, s.pbc)).toBeCloseTo(SPCE.rOH, 9);
    for (const iO of info.oxygens) {
      const o = s.atoms[iO]!.p;
      const a = minImage(o, s.atoms[iO + 1]!.p, s.cell, s.pbc);
      const b = minImage(o, s.atoms[iO + 2]!.p, s.cell, s.pbc);
      const cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (SPCE.rOH * SPCE.rOH);
      expect((Math.acos(cos) * 180) / Math.PI).toBeCloseTo(SPCE.angleDeg, 6);
    }
  });

  it('reaches close to liquid-water density without overlapping atoms', () => {
    const volume = s.cell[0] * s.cell[1] * 13;
    expect(info.waterCount / volume).toBeGreaterThan(0.0334 * 0.85);
    const nonBonded = (i: number, j: number) => s.atoms[i]!.mol === -1 || s.atoms[i]!.mol !== s.atoms[j]!.mol;
    let minHO = Infinity;
    let minOO = Infinity;
    for (let i = 0; i < s.atoms.length; i++)
      for (let j = i + 1; j < s.atoms.length; j++) {
        if (!nonBonded(i, j)) continue;
        const d = dist(s.atoms[i]!.p, s.atoms[j]!.p, s.cell, s.pbc);
        const pair = [s.atoms[i]!.el, s.atoms[j]!.el].sort().join('');
        if (pair === 'HO') minHO = Math.min(minHO, d);
        if (pair === 'OO') minOO = Math.min(minOO, d);
      }
    expect(minOO).toBeGreaterThanOrEqual(2.6);
    expect(minHO).toBeGreaterThanOrEqual(1.55);
  });

  it('orients first-shell waters with oxygen towards Na⁺', () => {
    const na = s.atoms[info.naIndex]!.p;
    for (const iO of info.oxygens) {
      const o = s.atoms[iO]!.p;
      if (dist(o, na, s.cell, s.pbc) >= 3.4) continue;
      const hMid = minImage(o, s.atoms[iO + 1]!.p, s.cell, s.pbc).map(
        (v, k) => (v + minImage(o, s.atoms[iO + 2]!.p, s.cell, s.pbc)[k]!) / 2,
      ) as Vec3;
      const away = minImage(na, o, s.cell, s.pbc);
      expect(hMid[0] * away[0] + hMid[1] * away[1] + hMid[2] * away[2]).toBeGreaterThan(0);
    }
  });
});

describe('cutoff graph and message passing', () => {
  const rc = 4;
  const g = cutoffGraph(s, rc);
  const i = info.naIndex;
  const hops = hopDistances(g, i, 4);

  it('is symmetric', () => {
    for (let u = 0; u < s.atoms.length; u++)
      for (let e = g.offsets[u]!; e < g.offsets[u + 1]!; e++) {
        const v = g.indices[e]!;
        const back = Array.from(g.indices.slice(g.offsets[v]!, g.offsets[v + 1]!));
        expect(back).toContain(u);
      }
  });

  it('one hop equals the atoms inside the cutoff sphere', () => {
    expect(reachCounts(hops, 4)[0]).toBe(countWithin(s, i, rc));
  });

  it('T hops never reach beyond T·r_c (receptive-field bound)', () => {
    s.atoms.forEach((a, j) => {
      const h = hops[j]!;
      if (h > 0) expect(dist(s.atoms[i]!.p, a.p, s.cell, s.pbc)).toBeLessThan(h * rc);
    });
  });

  it('reach grows monotonically with layers', () => {
    const c = reachCounts(hops, 4);
    for (let t = 1; t < 4; t++) expect(c[t]!).toBeGreaterThanOrEqual(c[t - 1]!);
  });
});

describe('rock-salt cluster', () => {
  it('alternates charges and is neutral up to the central ion', () => {
    const c = rockSaltCluster(2);
    expect(c.atoms).toHaveLength(125);
    expect(c.atoms.reduce((a, b) => a + b.q, 0)).toBe(1);
  });
});
