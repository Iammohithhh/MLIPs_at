import { describe, expect, it } from 'vitest';
import { addition, legendre, realSH, shBlock, tensorProduct, type Vec3 } from './spherical-harmonics';
import { apply, rotation } from './rotation-toy';

// Gauss–Legendre nodes in cos θ (via Newton on P_n) × uniform φ: exact for the
// polynomial degrees used here.
function gaussLegendre(n: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 1; i <= n; i++) {
    let x = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5));
    for (let it = 0; it < 50; it++) {
      let p0 = 1;
      let p1 = x;
      for (let k = 2; k <= n; k++) {
        const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      const dp = (n * (x * p1 - p0)) / (x * x - 1);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    let p0 = 1;
    let p1 = x;
    for (let k = 2; k <= n; k++) {
      const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
      p0 = p1;
      p1 = p2;
    }
    const dp = (n * (x * p1 - p0)) / (x * x - 1);
    out.push([x, 2 / ((1 - x * x) * dp * dp)]);
  }
  return out;
}

function integrate(f: (v: Vec3) => number): number {
  const gl = gaussLegendre(12);
  const nphi = 24;
  let s = 0;
  for (const [z, w] of gl) {
    const rho = Math.sqrt(1 - z * z);
    for (let k = 0; k < nphi; k++) {
      const phi = (2 * Math.PI * k) / nphi;
      s += w * (2 * Math.PI / nphi) * f([rho * Math.cos(phi), rho * Math.sin(phi), z]);
    }
  }
  return s;
}

describe('associated Legendre functions', () => {
  it('match closed forms', () => {
    const x = 0.37;
    expect(legendre(2, 0, x)).toBeCloseTo((3 * x * x - 1) / 2, 12);
    expect(legendre(2, 1, x)).toBeCloseTo(3 * x * Math.sqrt(1 - x * x), 12);
    expect(legendre(3, 3, x)).toBeCloseTo(15 * (1 - x * x) ** 1.5, 12);
  });
});

describe('real spherical harmonics', () => {
  it('Y00 is the constant 1/√(4π)', () => {
    expect(realSH(0, 0, [0.3, -0.2, 0.9])).toBeCloseTo(1 / Math.sqrt(4 * Math.PI), 12);
  });
  it('the l = 1 block is √(3/4π) times (y, z, x)', () => {
    const v: Vec3 = [0.48, -0.6, 0.64];
    const k = Math.sqrt(3 / (4 * Math.PI));
    const b = shBlock(1, v);
    expect(b[0]).toBeCloseTo(k * v[1], 12);
    expect(b[1]).toBeCloseTo(k * v[2], 12);
    expect(b[2]).toBeCloseTo(k * v[0], 12);
  });
  it('are orthonormal for l ≤ 3', () => {
    const lm: Array<[number, number]> = [];
    for (let l = 0; l <= 3; l++) for (let m = -l; m <= l; m++) lm.push([l, m]);
    for (const [l1, m1] of lm)
      for (const [l2, m2] of lm) {
        const v = integrate((d) => realSH(l1, m1, d) * realSH(l2, m2, d));
        expect(v).toBeCloseTo(l1 === l2 && m1 === m2 ? 1 : 0, 9);
      }
  });
  it('satisfy the addition theorem for every direction', () => {
    for (const v of [[1, 0, 0], [0.2, 0.5, -0.84], [-0.7, 0.1, 0.3]] as Vec3[])
      for (let l = 0; l <= 3; l++) {
        const s = shBlock(l, v).reduce((a, y) => a + y * y, 0);
        expect(s).toBeCloseTo(addition(l), 12);
      }
  });
  it('a rotation mixes components within a block but preserves its norm', () => {
    const v: Vec3 = [0.2, 0.5, -0.84];
    const w = apply(rotation([0.3, 0.8, 0.1], 1.1), v);
    for (let l = 1; l <= 3; l++) {
      const a = shBlock(l, v);
      const b = shBlock(l, w);
      expect(a.some((y, k) => Math.abs(y - b[k]!) > 1e-3)).toBe(true);
      expect(b.reduce((s, y) => s + y * y, 0)).toBeCloseTo(a.reduce((s, y) => s + y * y, 0), 12);
    }
  });
  it('have parity (−1)^l', () => {
    const v: Vec3 = [0.2, 0.5, -0.84];
    for (let l = 0; l <= 3; l++)
      for (let m = -l; m <= l; m++)
        expect(realSH(l, m, [-v[0], -v[1], -v[2]])).toBeCloseTo((-1) ** l * realSH(l, m, v), 12);
  });
});

describe('tensor-product selection rules', () => {
  it('1 ⊗ 1 = 0 ⊕ 1 ⊕ 2 with the l = 1 output a pseudovector', () => {
    const tp = tensorProduct(1, 1);
    expect(tp.map((p) => p.l)).toEqual([0, 1, 2]);
    expect(tp.map((p) => p.natural)).toEqual([true, false, true]);
  });
  it('dimensions add up: (2l1+1)(2l2+1) = Σ(2l+1)', () => {
    for (let a = 0; a <= 3; a++)
      for (let b = 0; b <= 3; b++)
        expect(tensorProduct(a, b).reduce((s, p) => s + p.dim, 0)).toBe((2 * a + 1) * (2 * b + 1));
  });
});
