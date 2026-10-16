import { describe, expect, it } from 'vitest';
import si from '~/data/structures/si-diamond.json';
import cu from '~/data/structures/cu-fcc.json';
import type { Structure } from '~/lib/structure/types';
import { mulberry32, randomRotation } from '../chem/rng';
import { realSH } from '../physics/spherical-harmonics';
import { shellAround } from './three-body';
import {
  GRACE_RCUT,
  TOPOLOGIES,
  atomicBase,
  buildCluster,
  chebBasis,
  costs,
  countEmbeddings,
  depth,
  envelope,
  explicitInvariant,
  gaunt,
  gaussLegendre,
  hopLayers,
  isStar,
  productInvariant,
  rattle,
  reach,
  rotate,
  sampleEmbedding,
  selfTermPairForm,
  topology,
  type Invariant,
  type Vec3,
} from './grace';

const ball = (s: unknown, R: number): Vec3[] => [[0, 0, 0], ...shellAround(s as Structure, R).map((n) => n.p as Vec3)];

describe('tree topologies', () => {
  it('depths and stars follow the paper’s notation', () => {
    expect(TOPOLOGIES.map((t) => depth(t))).toEqual([1, 1, 2, 1, 2, 2, 3]);
    expect(TOPOLOGIES.filter(isStar).map((t) => t.id)).toEqual(['1', '11', '111']);
  });
});

describe('embeddings in diamond Si', () => {
  // First shell only (2.35 Å): every atom has exactly four neighbours, so all counts are combinatorial.
  const c = buildCluster(ball(si, 8), 2.5);
  const count = (id: string) => countEmbeddings(c, topology(id), 2.5);
  it('matches closed forms for a four-coordinated network', () => {
    expect(count('1')).toBe(4);
    expect(count('11')).toBe(6); // C(4,2)
    expect(count('111')).toBe(4); // C(4,3)
    expect(count('12')).toBe(12); // 4 × 3
    expect(count('112')).toBe(36); // 4 × 3 × 3
    expect(count('122')).toBe(12); // 4 × C(3,2)
    expect(count('123')).toBe(36); // 4 × 3 × 3 (diamond has no 3-rings)
  });
  it('samples valid embeddings deterministically', () => {
    const t = topology('123');
    const e = sampleEmbedding(c, t, 2.5, 7)!;
    expect(e).toEqual(sampleEmbedding(c, t, 2.5, 7));
    expect(new Set(e).size).toBe(4);
    t.parent.forEach((p, k) => {
      const a = c.positions[e[p]!]!;
      const b = c.positions[e[k + 1]!]!;
      expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeLessThan(2.5);
    });
  });
  it('a star cannot reach beyond one cutoff; a chain of three edges reaches third neighbours', () => {
    const r1 = reach(c, 2.5, 1);
    const r3 = reach(c, 2.5, 3);
    expect(r1.atoms).toBe(4);
    expect(r1.rmax).toBeLessThan(2.5);
    // 4 + 12 + 24 atoms within three bonds: the third hop adds the 12 third neighbours (4.50 Å) and
    // 12 atoms at 5.92 Å, more than twice the cutoff from the root.
    expect(r3.atoms).toBe(40);
    expect(r3.rmax).toBeCloseTo(5.918, 2);
    expect(hopLayers(c, 2.5, 2).filter((h) => h === 2).length).toBe(12);
  });
});

describe('embeddings at GRACE’s 6 Å cutoff', () => {
  it('counts agree with direct formulas for stars and two-edge chains in fcc Cu', () => {
    const c = buildCluster(ball(cu, 13), GRACE_RCUT);
    const n0 = c.adj[0]!.filter((x) => x.d < GRACE_RCUT).length;
    expect(countEmbeddings(c, topology('1'), GRACE_RCUT)).toBe(n0);
    expect(countEmbeddings(c, topology('11'), GRACE_RCUT)).toBe((n0 * (n0 - 1)) / 2);
    expect(countEmbeddings(c, topology('111'), GRACE_RCUT)).toBe((n0 * (n0 - 1) * (n0 - 2)) / 6);
    // Each first-hop atom j has n0 neighbours (a perfect crystal), one of which is the root.
    expect(countEmbeddings(c, topology('12'), GRACE_RCUT)).toBe(n0 * (n0 - 1));
  });
});

describe('radial basis', () => {
  it('the p = 5 envelope has zero value, slope and curvature at the cutoff', () => {
    const h = 1e-4;
    expect(envelope(1 - 1e-12)).toBeCloseTo(0, 10);
    expect((envelope(1) - envelope(1 - h)) / h).toBeCloseTo(0, 6);
    // One-sided second difference has an O(h) error from f''' ≠ 0; it must shrink with h.
    const d2 = (k: number) => (envelope(1) - 2 * envelope(1 - k) + envelope(1 - 2 * k)) / (k * k);
    expect(Math.abs(d2(1e-5))).toBeLessThan(0.01);
    expect(Math.abs(d2(1e-5))).toBeLessThan(Math.abs(d2(1e-4)) / 5);
    expect(envelope(0)).toBe(1);
  });
  it('Chebyshev functions are T_n(2r/rc − 1) times the envelope', () => {
    const g = chebBasis(1.5, 3, 6);
    const t = 2 * 0.25 - 1;
    const f = envelope(0.25);
    expect(g[0]).toBeCloseTo(t * f, 14);
    expect(g[1]).toBeCloseTo((2 * t * t - 1) * f, 14);
    expect(g[2]).toBeCloseTo((4 * t ** 3 - 3 * t) * f, 14);
    expect(chebBasis(6.1, 3, 6)).toEqual([0, 0, 0]);
  });
});

describe('real Gaunt coefficients', () => {
  it('Gauss–Legendre integrates polynomials exactly', () => {
    const { x, w } = gaussLegendre(5);
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 14);
    expect(x.reduce((s, z, i) => s + w[i]! * z ** 8, 0)).toBeCloseTo(2 / 9, 13);
  });
  it('with l3 = 0 they reduce to orthonormality / √(4π)', () => {
    const G = gaunt(2, 2, 0);
    for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) expect(G[a * 5 + b]).toBeCloseTo(a === b ? 1 / Math.sqrt(4 * Math.PI) : 0, 13);
  });
  it('vanish for odd l1 + l2 + l3 and outside the triangle', () => {
    expect(gaunt(1, 1, 1).every((g) => g === 0)).toBe(true);
    expect(gaunt(1, 1, 4).every((g) => g === 0)).toBe(true);
    expect(gaunt(1, 1, 2).some((g) => g !== 0)).toBe(true);
  });
  it('reproduce ∫ Y_10 Y_10 Y_20 dΩ = √(45/4π) · (1 1 2; 0 0 0)² = (2/15)√(45/4π)', () => {
    const G = gaunt(1, 1, 2);
    expect(G[(1 * 3 + 1) * 5 + 2]).toBeCloseTo((2 / 15) * Math.sqrt(45 / (4 * Math.PI)), 13);
    // and the same coefficient straight from the harmonics on a dense midpoint grid
    let s = 0;
    const n = 200;
    for (let i = 0; i < n; i++) {
      const z = -1 + (2 * (i + 0.5)) / n;
      for (let j = 0; j < 2 * n; j++) {
        const ph = (Math.PI * (j + 0.5)) / n;
        const r = Math.sqrt(1 - z * z);
        const v: Vec3 = [r * Math.cos(ph), r * Math.sin(ph), z];
        s += realSH(1, 0, v) ** 2 * realSH(2, 0, v) * (2 / n) * (Math.PI / n);
      }
    }
    expect(G[(1 * 3 + 1) * 5 + 2]).toBeCloseTo(s, 4);
  });
});

describe('density trick and self-interactions', () => {
  const base = shellAround(si as unknown as Structure, 4.6).map((n) => n.p as Vec3);
  const env = rattle(base, 0.1, 3);
  const nf = 4;
  const cases: Invariant[] = [
    { n: [1, 2], l: [1, 1] },
    { n: [2, 3], l: [3, 3] },
    { n: [1, 2, 3], l: [1, 1, 2] },
    { n: [2, 2, 1], l: [2, 2, 2] },
    { n: [1, 3, 4], l: [1, 2, 3] },
  ];
  it.each(cases)('product basis = explicit tuple sum incl. self-interactions (%o)', (b) => {
    const A = atomicBase(env, nf, 3);
    const p = productInvariant(A, b);
    const e = explicitInvariant(env, b, nf);
    expect(e.total).toBeCloseTo(p, 10);
    expect(e.terms).toBe(env.length ** b.n.length);
  });
  it('the ν = 2 self term is the two-body sum Σ_j g g (2l+1)/4π', () => {
    for (const b of cases.filter((x) => x.n.length === 2)) {
      expect(explicitInvariant(env, b, nf).diagonal).toBeCloseTo(selfTermPairForm(env, b, nf), 12);
    }
  });
  it('invariants are rotation-invariant', () => {
    const R = randomRotation(mulberry32(11));
    for (const b of cases) {
      const a = productInvariant(atomicBase(env, nf, 3), b);
      const r = productInvariant(atomicBase(rotate(env, R), nf, 3), b);
      expect(r).toBeCloseTo(a, 10);
    }
  });
  it('the all-equal (diagonal) terms depend on bond lengths only', () => {
    // Rotate every neighbour independently: lengths kept, all angles changed.
    const rand = mulberry32(5);
    const scrambled = env.map((v) => rotate([v], randomRotation(rand))[0]!);
    for (const b of cases) {
      const a = explicitInvariant(env, b, nf);
      const s = explicitInvariant(scrambled, b, nf);
      expect(s.diagonal).toBeCloseTo(a.diagonal, 10);
      expect(Math.abs(s.distinct - a.distinct)).toBeGreaterThan(1e-6);
    }
  });
  it('costs: tuples grow as N^ν, the atomic base linearly', () => {
    const c1 = costs(16, 3, 8, 4);
    const c2 = costs(32, 3, 8, 4);
    expect(c2.explicit / c1.explicit).toBe(8);
    expect(c2.ace / c1.ace).toBe(2);
    expect(c1.distinctOnly).toBe(16 * 15 * 14);
  });
});
