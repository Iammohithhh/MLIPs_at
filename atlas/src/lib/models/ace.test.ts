import { describe, expect, it } from 'vitest';
import { mulberry32 } from '../chem/rng';
import { realSH } from '../physics/spherical-harmonics';
import {
  atomicBase,
  buildBasis,
  clebschGordan,
  countBasis,
  countInvariants,
  couplingTrees,
  environmentBasis,
  evaluateBasisWithDerivative,
  multisetBasis,
  randomlyRotated,
  realFromComplex,
  type AceSpec,
  type Radial,
  type Vec3,
} from './ace';

const R: Radial = { rc: 5, lambda: 5 };
const env = (seed: number, n = 9): Vec3[] => {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const d = 1.8 + 2.8 * r();
    const z = 2 * r() - 1;
    const p = 2 * Math.PI * r();
    const s = Math.sqrt(1 - z * z);
    return [d * s * Math.cos(p), d * s * Math.sin(p), d * z] as Vec3;
  });
};
const SPEC: AceSpec = { nmax: 3, lmax: 3, numax: 4, degree: [3, 6, 7, 7], wL: 1 };

describe('Clebsch–Gordan coefficients', () => {
  it('match tabulated values and are orthonormal', () => {
    expect(clebschGordan(1, 0, 1, 0, 0, 0)).toBeCloseTo(-1 / Math.sqrt(3), 14);
    expect(clebschGordan(1, 1, 1, -1, 0, 0)).toBeCloseTo(1 / Math.sqrt(3), 14);
    expect(clebschGordan(1, 1, 1, 0, 2, 1)).toBeCloseTo(Math.SQRT1_2, 14);
    expect(clebschGordan(2, 1, 1, 0, 2, 1)).toBeCloseTo(1 / Math.sqrt(6), 14);
    for (const [l1, l2] of [[1, 1], [2, 1], [3, 2]] as const)
      for (let L = Math.abs(l1 - l2); L <= l1 + l2; L++)
        for (let Lp = Math.abs(l1 - l2); Lp <= l1 + l2; Lp++) {
          let s = 0;
          for (let m1 = -l1; m1 <= l1; m1++) for (let m2 = -l2; m2 <= l2; m2++) s += clebschGordan(l1, m1, l2, m2, L, 0) * clebschGordan(l1, m1, l2, m2, Lp, 0);
          expect(s).toBeCloseTo(L === Lp ? 1 : 0, 12);
        }
  });

  it('map complex to real harmonics by a unitary matrix', () => {
    for (let l = 0; l <= 4; l++) {
      const { re, im } = realFromComplex(l);
      const d = 2 * l + 1;
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++) {
          let sr = 0;
          let si = 0;
          for (let k = 0; k < d; k++) {
            sr += re[a * d + k]! * re[b * d + k]! + im[a * d + k]! * im[b * d + k]!;
            si += im[a * d + k]! * re[b * d + k]! - re[a * d + k]! * im[b * d + k]!;
          }
          expect(sr).toBeCloseTo(a === b ? 1 : 0, 12);
          expect(si).toBeCloseTo(0, 12);
        }
    }
  });
});

describe('counting invariants with characters', () => {
  const nl = (...p: Array<[number, number]>) => p.map(([n, l]) => ({ n, l }));
  it('agrees with hand counts', () => {
    expect(countInvariants(nl([1, 0]))).toBe(1);
    expect(countInvariants(nl([1, 1]))).toBe(0);
    expect(countInvariants(nl([1, 2], [2, 2]))).toBe(1);
    expect(countInvariants(nl([1, 1], [2, 2]))).toBe(0);
    expect(countInvariants(nl([1, 1], [2, 1], [3, 1]))).toBe(0); // odd Σl: the triple product is a pseudoscalar
    expect(countInvariants(nl([1, 1], [2, 1], [3, 2]))).toBe(1);
    expect(countInvariants(nl([1, 2], [1, 2], [1, 2]))).toBe(1); // tr M³ of a traceless symmetric matrix
    expect(countInvariants(nl([1, 1], [2, 1], [3, 1], [4, 1]))).toBe(3); // (a·b)(c·d) and its two re-pairings
    expect(countInvariants(nl([1, 1], [1, 1], [1, 1], [1, 1]))).toBe(1); // (a·a)²
    expect(countInvariants(nl([1, 2], [1, 2], [1, 2], [1, 2]))).toBe(1); // (tr M²)²
    expect(countInvariants(nl([1, 1], [1, 1], [2, 1], [2, 1]))).toBe(2); // (a·a)(b·b), (a·b)²
  });

  it('enumerates coupling trees with every admissible intermediate', () => {
    // the chain lists L_1 … L_{ν−2}; the last must equal l_ν so that it couples to zero
    expect(couplingTrees([1, 1, 1, 1])).toEqual([[0, 1], [1, 1], [2, 1]]);
    expect(couplingTrees([2, 1, 1])).toEqual([[1]]);
    expect(couplingTrees([1, 2])).toEqual([]);
  });
});

describe('the basis', () => {
  const basis = buildBasis(SPEC);

  it('keeps exactly as many independent functions per multiset as the character formula predicts', () => {
    for (const r of basis.reports) expect(r.kept, JSON.stringify(r.nls)).toBe(r.predicted);
    const counts = countBasis(SPEC);
    expect(counts.reduce((s, c) => s + c.invariants, 0)).toBe(basis.functions.length);
  });

  it('removes real redundancy for repeated labels', () => {
    const r = multisetBasis([{ n: 1, l: 1 }, { n: 1, l: 1 }, { n: 2, l: 1 }, { n: 2, l: 1 }]).report;
    expect(r.trees).toBe(3);
    expect(r.kept).toBe(2);
  });

  it('is invariant under rotations, reflections and permutations of neighbours', () => {
    for (const seed of [1, 2, 3]) {
      const e = env(seed);
      const B0 = environmentBasis(basis, e, R);
      const B1 = environmentBasis(basis, randomlyRotated(e, 100 + seed), R);
      const B2 = environmentBasis(basis, e.map((v) => [-v[0], -v[1], -v[2]] as Vec3), R);
      const B3 = environmentBasis(basis, [...e].reverse(), R);
      const scale = Math.max(...B0.map(Math.abs));
      for (let k = 0; k < B0.length; k++) {
        expect(Math.abs(B1[k]! - B0[k]!) / scale).toBeLessThan(1e-11);
        expect(Math.abs(B2[k]! - B0[k]!) / scale).toBeLessThan(1e-11);
        expect(Math.abs(B3[k]! - B0[k]!) / scale).toBeLessThan(1e-11);
      }
    }
  });

  it('turns ν = 2 into the power spectrum Σ_m A_{n1 l m} A_{n2 l m}', () => {
    const e = env(4);
    const { A } = atomicBase(e, SPEC.nmax, SPEC.lmax, R);
    const K = (SPEC.lmax + 1) ** 2;
    const B = environmentBasis(basis, e, R);
    basis.functions.forEach((f, q) => {
      if (f.nls.length !== 2) return;
      const [a, b] = f.nls;
      let p = 0;
      for (let m = 0; m <= 2 * a!.l; m++) p += A[(a!.n - 1) * K + a!.l ** 2 + m]! * A[(b!.n - 1) * K + b!.l ** 2 + m]!;
      // the coupling is ⟨l m l −m|00⟩ in the complex basis: ±1/√(2l+1) times the real dot product
      expect(Math.abs(B[q]!)).toBeCloseTo(Math.abs(p) / Math.sqrt(2 * a!.l + 1), 10);
    });
  });

  it('builds A from real harmonics', () => {
    const e = env(5, 3);
    const { A } = atomicBase(e, 2, 2, R);
    const K = 9;
    let direct = 0;
    for (const v of e) direct += realSH(2, 1, v) * (0.5 * (1 + Math.cos((Math.PI * Math.hypot(...v)) / R.rc)));
    expect(A[0 * K + 4 + 2 + 1]).toBeCloseTo(direct, 12);
  });

  it('has exact derivatives with respect to neighbour positions', () => {
    const e = env(6, 6);
    const { A, dA } = atomicBase(e, SPEC.nmax, SPEC.lmax, R, true);
    const { B, dBdA } = evaluateBasisWithDerivative(basis, A);
    const h = 1e-5;
    for (const j of [0, 3])
      for (const a of [0, 2]) {
        const p = e.map((v) => [...v] as Vec3);
        const m = e.map((v) => [...v] as Vec3);
        p[j]![a]! += h;
        m[j]![a]! -= h;
        const Bp = environmentBasis(basis, p, R);
        const Bm = environmentBasis(basis, m, R);
        const scale = Math.max(...B.map(Math.abs));
        basis.functions.forEach((_, q) => {
          let an = 0;
          const d = dA![j]![a]!;
          for (let k = 0; k < d.length; k++) an += dBdA[q]![k]! * d[k]!;
          expect(Math.abs(an - (Bp[q]! - Bm[q]!) / (2 * h)) / scale).toBeLessThan(1e-6);
        });
      }
  });
});

describe('Dusson et al. (2022), Table 1', () => {
  // (l, n, #RPI) rows; #RI is the count with every n distinct. n is 0-based in the paper, shifted by 1 here.
  const rows: Array<[number[], number[], number]> = [
    [[2, 2, 2, 2], [0, 0, 0, 1], 1],
    [[2, 2, 2, 2], [0, 0, 0, 2], 1],
    [[2, 2, 2, 2], [0, 0, 1, 2], 3],
    [[2, 2, 2, 4], [0, 0, 0, 0], 1],
    [[2, 2, 3, 3], [0, 0, 0, 0], 3],
    [[1, 1, 2, 2, 2], [0, 0, 0, 1, 2], 6],
    [[1, 1, 2, 2, 2], [0, 1, 0, 0, 0], 2],
    [[1, 1, 2, 2, 2], [1, 1, 0, 0, 1], 4],
    [[1, 2, 2, 2, 3], [0, 0, 0, 0, 0], 3],
    [[2, 2, 2, 2, 2], [0, 0, 0, 0, 0], 1],
  ];
  const ri: Array<[number[], number]> = [[[2, 2, 2, 2], 5], [[2, 2, 2, 4], 3], [[2, 2, 3, 3], 5], [[1, 1, 2, 2, 2], 9], [[1, 2, 2, 2, 3], 12], [[2, 2, 2, 2, 2], 16]];
  const sorted = (ls: number[], ns: number[]) => ls.map((l, k) => ({ n: ns[k]! + 1, l })).sort((a, b) => a.n - b.n || a.l - b.l);

  it('reproduces every #RI and #RPI entry with the character formula', () => {
    for (const [ls, count] of ri) expect(countInvariants(ls.map((l, k) => ({ n: k + 1, l })))).toBe(count);
    for (const [ls, ns, count] of rows) expect(countInvariants(sorted(ls, ns)), `${ls} ${ns}`).toBe(count);
  });

  it('and with the coupling-plus-rank construction', () => {
    for (const [ls, ns, count] of rows.filter((r) => r[0].length === 4)) expect(multisetBasis(sorted(ls, ns)).report.kept).toBe(count);
    expect(multisetBasis(sorted([1, 1, 2, 2, 2], [1, 1, 0, 0, 1])).report.kept).toBe(4);
  });
});
