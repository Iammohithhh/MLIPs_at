// Real spherical harmonics and tensor-product selection rules (Physics Lab P5).
//
// Convention: real, orthonormal on the unit sphere, Condon–Shortley phase omitted:
//   Y_l0  = N_l^0 P_l^0(cos θ)
//   Y_lm  = √2 N_l^m P_l^m(cos θ) cos(mφ)      (m > 0)
//   Y_l,−m = √2 N_l^m P_l^m(cos θ) sin(mφ)     (m > 0)
//   N_l^m = √[(2l+1)/(4π) · (l−m)!/(l+m)!]
// Equivariant libraries differ in sign and ordering conventions (e.g. e3nn uses its
// own); the invariants shown on the page do not depend on them.
//
// Addition theorem: Σ_m Y_lm(r̂)² = (2l+1)/(4π) for every direction r̂ — the norm of
// the l-th feature block is rotation-invariant even though its components mix.

export type Vec3 = [number, number, number];

function factorial(n: number): number {
  let f = 1;
  for (let k = 2; k <= n; k++) f *= k;
  return f;
}

/** Associated Legendre P_l^m(x), m ≥ 0, without the Condon–Shortley phase. */
export function legendre(l: number, m: number, x: number): number {
  if (m > l) return 0;
  let pmm = 1;
  if (m > 0) {
    const s = Math.sqrt(Math.max(0, 1 - x * x));
    let fact = 1;
    for (let k = 1; k <= m; k++) {
      pmm *= fact * s;
      fact += 2;
    }
  }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  let pll = 0;
  for (let ll = m + 2; ll <= l; ll++) {
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

/** Real spherical harmonic Y_lm evaluated at direction v (need not be normalised). */
export function realSH(l: number, m: number, v: Vec3): number {
  const r = Math.hypot(v[0], v[1], v[2]) || 1;
  const z = v[2] / r;
  const phi = Math.atan2(v[1], v[0]);
  const am = Math.abs(m);
  const N = Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (factorial(l - am) / factorial(l + am)));
  const P = legendre(l, am, z);
  if (m === 0) return N * P;
  return Math.SQRT2 * N * P * (m > 0 ? Math.cos(am * phi) : Math.sin(am * phi));
}

/** The 2l+1 components [Y_l,−l … Y_l,l] for direction v. */
export function shBlock(l: number, v: Vec3): number[] {
  return Array.from({ length: 2 * l + 1 }, (_, k) => realSH(l, k - l, v));
}

export const addition = (l: number) => (2 * l + 1) / (4 * Math.PI);

export interface TPPath {
  l: number;
  /** Parity of the product, (−1)^(l1+l2). */
  parity: 1 | -1;
  /** True when the parity equals (−1)^l, as for a spherical harmonic of order l ("polar"). */
  natural: boolean;
  dim: number;
}

/** Outputs of l1 ⊗ l2 for inputs with natural parity (−1)^l1 and (−1)^l2. */
export function tensorProduct(l1: number, l2: number): TPPath[] {
  const parity = ((l1 + l2) % 2 === 0 ? 1 : -1) as 1 | -1;
  const out: TPPath[] = [];
  for (let l = Math.abs(l1 - l2); l <= l1 + l2; l++) {
    out.push({ l, parity, natural: parity === (l % 2 === 0 ? 1 : -1), dim: 2 * l + 1 });
  }
  return out;
}
