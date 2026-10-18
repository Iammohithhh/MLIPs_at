import { describe, expect, it } from 'vitest';
import templates from '~/data/models/mtp-templates.json';
import si from '~/data/models/mtp-si.json';
import { mulberry32, randomRotation } from '../chem/rng';
import { latticeEnv } from './soap';
import {
  basisValues,
  canonical,
  chebyshevRadial,
  contract,
  describe as describeBasis,
  enumerateBasis,
  maxMomentsUsed,
  momentComponents,
  momentOnSphere,
  momentTensor,
  moments,
  radialFunctionsUsed,
  radialValue,
  type BasisFunction,
  type Vec3,
} from './mtp';

const o = { rmin: 1.9, rcut: 5.0 };
const rotate = (R: number[], v: Vec3): Vec3 => [R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2], R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2], R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2]];
const rattle = (env: Vec3[], amp: number, seed: number): Vec3[] => {
  const r = mulberry32(seed);
  return env.map((v) => [v[0] + amp * (2 * r() - 1), v[1] + amp * (2 * r() - 1), v[2] + amp * (2 * r() - 1)]);
};
const relErr = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num = Math.max(num, Math.abs(a[i]! - b[i]!));
    den = Math.max(den, Math.abs(b[i]!));
  }
  return num / (den || 1);
};

describe('radial basis (mlip-2 RBChebyshev)', () => {
  it('is T_β(ξ)(r − R_cut)² and vanishes with zero slope at the cutoff', () => {
    const r = 3.1;
    const ksi = (2 * r - 6.9) / 3.1;
    const q = chebyshevRadial(r, 1.9, 5, 4);
    const env = (r - 5) ** 2;
    expect(q[0]).toBeCloseTo(env, 14);
    expect(q[1]).toBeCloseTo(ksi * env, 14);
    expect(q[2]).toBeCloseTo((2 * ksi * ksi - 1) * env, 14);
    expect(q[3]).toBeCloseTo((4 * ksi ** 3 - 3 * ksi) * env, 14);
    expect(chebyshevRadial(5, 1.9, 5, 4).every((x) => x === 0)).toBe(true);
    const h = 1e-6;
    expect(Math.abs((radialValue(5 - h, 2, o) - radialValue(5 - 2 * h, 2, o)) / h)).toBeLessThan(1e-4);
  });
});

describe('basis enumeration', () => {
  it('reproduces the nine level-8 basis functions listed by Novikov et al. (2021), Eq. (6)', () => {
    const b = enumerateBasis(8);
    expect(b.map(describeBasis).sort()).toEqual(
      ['M₀,₀', 'M₁,₀', 'M₀,₀ M₀,₀', 'M₀,₁ · M₀,₁', 'M₀,₂ : M₀,₂', 'M₀,₀ M₁,₀', 'M₀,₀ M₀,₀ M₀,₀', 'M₀,₀ (M₀,₁ · M₀,₁)', 'M₀,₀ M₀,₀ M₀,₀ M₀,₀'].sort(),
    );
    expect(b.map((x) => x.level).sort((p, q) => p - q)).toEqual([2, 4, 6, 6, 6, 8, 8, 8, 8]);
  });

  it('matches every untrained template shipped with mlip-2: basis size, moment components, radial functions', () => {
    expect(templates.templates.length).toBe(14);
    for (const t of templates.templates) {
      const b = enumerateBasis(t.level);
      expect(b.length, `level ${t.level}`).toBe(t.alpha_scalar_moments);
      expect(momentComponents(b), `level ${t.level}`).toBe(t.alpha_index_basic_count);
      expect(radialFunctionsUsed(b), `level ${t.level}`).toBe(t.radial_funcs_count);
      expect(Math.max(...b.flatMap((x) => x.nu))).toBe(t.nu_max);
    }
  });

  it('free parameters = basis + 1 constant + radial functions × radial basis size: Zuo et al. Table S10', () => {
    const params = (L: number) => {
      const t = templates.templates.find((x) => x.level === L)!;
      return t.alpha_scalar_moments + 1 + t.radial_funcs_count * t.radial_basis_size;
    };
    // Table S10 of the supporting information: 16 → 125 (Li), 20 → 329 (Ni, Cu, Mo), 24 → 913 (Si, Ge)
    expect([params(16), params(20), params(24)]).toEqual([125, 329, 913]);
  });

  it('caps contractions at four moments (five-body) and has no repeated graphs', () => {
    const b = enumerateBasis(20);
    expect(maxMomentsUsed(b)).toBe(4);
    expect(new Set(b.map((x) => x.key)).size).toBe(b.length);
    // without the cap there would be more
    expect(enumerateBasis(10, { maxMoments: 5, muMax: 5, nuMax: 10 }).length).toBe(17);
  });

  it('labels a graph the same way whatever the order of its nodes', () => {
    // (M₀,₂ M₀,₁) · M₀,₁ written with the matrix in the middle and at either end
    const a = canonical([0, 0, 0], [1, 1, 0]);
    const b = canonical([0, 0, 0], [1, 0, 1]);
    const c = canonical([0, 0, 0], [0, 1, 1]);
    expect(a).toEqual(b);
    expect(b).toEqual(c);
  });

  it('agrees with the pipeline’s enumeration for the precomputed Si basis', () => {
    const b = enumerateBasis(si.settings.level_max);
    expect(si.basis.map((x) => x.key)).toEqual(b.map((x) => x.key));
    expect(si.basis.map((x) => x.level)).toEqual(b.map((x) => x.level));
  });
});

describe('moments and contractions', () => {
  const env = rattle(latticeEnv('diamond', 2.35, 5.0), 0.15, 3);
  const basis = enumerateBasis(14);

  it('every basis function is invariant under rotation, reflection and permutation of neighbours', () => {
    const v0 = basisValues(env, basis, o);
    const R = randomRotation(mulberry32(7));
    expect(relErr(basisValues(env.map((v) => rotate(R, v)), basis, o), v0)).toBeLessThan(1e-11);
    expect(relErr(basisValues(env.map((v) => [-v[0], v[1], v[2]] as Vec3), basis, o), v0)).toBeLessThan(1e-11);
    expect(relErr(basisValues([...env].reverse(), basis, o), v0)).toBeLessThan(1e-11);
  });

  it('contractions equal the direct sums over neighbour pairs and triples', () => {
    const u = env.map((v) => {
      const r = Math.hypot(...v);
      return { f: radialValue(r, 0, o), f1: radialValue(r, 1, o), u: v.map((x) => x / r) as Vec3 };
    });
    const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const by = (k: string) => basis.find((b) => b.key === k)!;
    const M = moments(env, basis, o);
    // M₀,₂ : M₀,₂ = Σ_ij f f (û_i · û_j)²
    let s2 = 0;
    for (const a of u) for (const b of u) s2 += a.f * b.f * dot(a.u, b.u) ** 2;
    expect(contract(by('2|0,0|2'), M)).toBeCloseTo(s2, 9);
    // M₀,₁ · M₁,₁ = Σ_ij f₀ f₁ (û_i · û_j)
    let s1 = 0;
    for (const a of u) for (const b of u) s1 += a.f * b.f1 * dot(a.u, b.u);
    expect(contract(by('2|0,1|1'), M)).toBeCloseTo(s1, 9);
    // (M₀,₂ M₀,₁) · M₀,₁ = Σ_ijk f f f (û_i · û_j)(û_i · û_k)
    let s3 = 0;
    for (const a of u) for (const b of u) for (const c of u) s3 += a.f * b.f * c.f * dot(a.u, b.u) * dot(a.u, c.u);
    expect(contract(by('3|0,0,0|0,1,1'), M)).toBeCloseTo(s3, 8);
  });

  it('odd moments vanish at centrosymmetric sites; diamond’s tetrahedral site keeps M₀,₃', () => {
    const fcc = latticeEnv('fcc', 2.35, 5.0);
    const dia = latticeEnv('diamond', 2.35, 3.0);
    const max = (t: Float64Array) => Math.max(...t.map(Math.abs));
    expect(max(momentTensor(fcc, 0, 1, o))).toBeLessThan(1e-12);
    expect(max(momentTensor(fcc, 0, 3, o))).toBeLessThan(1e-12);
    expect(max(momentTensor(dia, 0, 1, o))).toBeLessThan(1e-12);
    expect(max(momentTensor(dia, 0, 3, o))).toBeGreaterThan(0.1);
    // cubic sites: the second moment is isotropic, M₀,₂ ∝ identity
    const m2 = momentTensor(fcc, 0, 2, o);
    expect(Math.abs(m2[0]! - m2[4]!) + Math.abs(m2[1]!)).toBeLessThan(1e-12);
  });

  it('the polar surface is the tensor contracted with n̂ in every slot', () => {
    const t = momentTensor(env, 1, 3, o);
    const n: Vec3 = [0.36, -0.48, 0.8];
    let s = 0;
    for (let k = 0; k < 27; k++) s += t[k]! * n[k % 3]! * n[((k / 3) | 0) % 3]! * n[((k / 9) | 0) % 3]!;
    expect(momentOnSphere(env, 1, 3, o, n)).toBeCloseTo(s, 12);
  });

  it('neighbours beyond R_cut do not contribute', () => {
    const b: BasisFunction[] = enumerateBasis(10);
    const far: Vec3[] = [...env, [5.2, 0, 0], [0, 0, -6]];
    expect(relErr(basisValues(far, b, o), basisValues(env, b, o))).toBe(0);
  });
});
