// Lattice sums for rock-salt NaCl, in units of the nearest-neighbour distance r0.
//
// The electrostatic energy of one ion is −M·k·e²/r0, with Madelung constant
// M = −Σ'_n (−1)^(n₁+n₂+n₃)/|n| over the simple-cubic lattice of ions with
// alternating charge. The series is only conditionally convergent: its value
// depends on the order of summation.
// - Expanding spheres: the partial sums do not converge (Borwein, Borwein &
//   Taylor, J. Math. Phys. 26, 2999, 1985).
// - Expanding cubes: converge to M.
// - Evjen weights (charge-neutral cubes, Phys. Rev. 39, 675, 1932) converge fast.
// - Ewald summation (Ann. Phys. 369, 253, 1921) gives M to machine precision.

/** M for rock salt to 12 digits, for display and tests (OEIS A085469). */
export const MADELUNG_NACL = 1.74756459463318;

const sign = (i: number, j: number, k: number) => (((i + j + k) % 2) + 2) % 2 === 0 ? 1 : -1;

export interface PartialSum {
  /** Summation radius (spheres) or half-width (cubes), units of r0. */
  R: number;
  /** Running estimate of M. */
  M: number;
  /** Ions included so far (excluding the reference ion). */
  count: number;
}

/** Partial Madelung sums over expanding spheres, one point per distinct shell |n|² ≤ Rmax². */
export function sphereSeries(Rmax: number): PartialSum[] {
  const N = Math.floor(Rmax);
  const shells = new Map<number, number>();
  for (let i = -N; i <= N; i++)
    for (let j = -N; j <= N; j++)
      for (let k = -N; k <= N; k++) {
        const r2 = i * i + j * j + k * k;
        if (r2 === 0 || r2 > Rmax * Rmax) continue;
        shells.set(r2, (shells.get(r2) ?? 0) + sign(i, j, k));
      }
  const out: PartialSum[] = [];
  let sum = 0;
  let count = 0;
  for (const r2 of [...shells.keys()].sort((a, b) => a - b)) {
    const net = shells.get(r2)!;
    sum += net / Math.sqrt(r2);
    // Count sites on this shell (net charge alone hides them).
    count += sitesOnShell(r2, N);
    out.push({ R: Math.sqrt(r2), M: -sum, count });
  }
  return out;
}

function sitesOnShell(r2: number, N: number): number {
  let c = 0;
  for (let i = -N; i <= N; i++)
    for (let j = -N; j <= N; j++) {
      const rest = r2 - i * i - j * j;
      if (rest < 0) continue;
      const k = Math.round(Math.sqrt(rest));
      if (k * k === rest) c += k === 0 ? 1 : 2;
    }
  return c;
}

/** Partial Madelung sum over the cube max|nᵢ| ≤ N; with `evjen`, boundary sites carry weights ½, ¼, ⅛. */
export function cubeSum(N: number, evjen = false): number {
  let sum = 0;
  for (let i = -N; i <= N; i++)
    for (let j = -N; j <= N; j++)
      for (let k = -N; k <= N; k++) {
        if (i === 0 && j === 0 && k === 0) continue;
        let w = 1;
        if (evjen) {
          if (Math.abs(i) === N) w *= 0.5;
          if (Math.abs(j) === N) w *= 0.5;
          if (Math.abs(k) === N) w *= 0.5;
        }
        sum += (w * sign(i, j, k)) / Math.sqrt(i * i + j * j + k * k);
      }
  return -sum;
}

export function cubeSeries(Nmax: number, evjen = false): PartialSum[] {
  return Array.from({ length: Nmax }, (_, t) => {
    const N = t + 1;
    return { R: N, M: cubeSum(N, evjen), count: (2 * N + 1) ** 3 - 1 };
  });
}

/** Complementary error function, |relative error| < 1.2e-7 (Numerical Recipes erfcc). */
export function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

export interface EwaldParts {
  /** Σ q erfc(αr)/r over images within the real-space cutoff (units e/r0). */
  real: number;
  /** (4π/V) Σ_k e^(−k²/4α²)/k² Σ q cos(k·r) over |m| ≤ kCut. */
  recip: number;
  /** −2α q_i/√π: removes the reference ion's own Gaussian. */
  self: number;
  /** real + recip + self = potential φ at the reference Na⁺; M = −φ. */
  total: number;
  /** Number of real-space pair terms and of reciprocal vectors used. */
  nReal: number;
  nK: number;
}

const CELL_IONS: Array<[number, number, number, number]> = (() => {
  const ions: Array<[number, number, number, number]> = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) ions.push([i, j, k, sign(i, j, k)]);
  return ions;
})();

/**
 * Ewald summation for the potential at the Na⁺ at the origin of rock salt, on the
 * conventional cubic cell (edge L = 2 r0, 4 Na⁺ + 4 Cl⁻), all lengths in units of r0.
 * Spherical cutoffs: real-space terms with r < rCut, reciprocal vectors k = 2πm/L
 * with |m| ≤ kCut.
 */
export function ewaldParts(alpha = 1.2, rCut = 6, kCut = 8): EwaldParts {
  const L = 2;
  const n = Math.ceil(rCut / L) + 1;
  let real = 0;
  let nReal = 0;
  for (const [x, y, z, q] of CELL_IONS)
    for (let a = -n; a <= n; a++)
      for (let b = -n; b <= n; b++)
        for (let c = -n; c <= n; c++) {
          const r = Math.hypot(x + a * L, y + b * L, z + c * L);
          if (r === 0 || r >= rCut) continue;
          real += (q * erfc(alpha * r)) / r;
          nReal++;
        }
  let recip = 0;
  let nK = 0;
  const V = L ** 3;
  const K = Math.floor(kCut);
  for (let a = -K; a <= K; a++)
    for (let b = -K; b <= K; b++)
      for (let c = -K; c <= K; c++) {
        if ((a === 0 && b === 0 && c === 0) || a * a + b * b + c * c > kCut * kCut) continue;
        const kx = (2 * Math.PI * a) / L;
        const ky = (2 * Math.PI * b) / L;
        const kz = (2 * Math.PI * c) / L;
        const k2 = kx * kx + ky * ky + kz * kz;
        let s = 0;
        for (const [x, y, z, q] of CELL_IONS) s += q * Math.cos(kx * x + ky * y + kz * z);
        recip += (Math.exp(-k2 / (4 * alpha * alpha)) / k2) * s;
        nK++;
      }
  recip *= (4 * Math.PI) / V;
  const self = (-2 * alpha) / Math.sqrt(Math.PI);
  return { real, recip, self, total: real + recip + self, nReal, nK };
}

/** Madelung constant by Ewald summation (α in units of 1/r0). */
export function madelungEwald(alpha = 1.2, rCut = 7, kCut = 8): number {
  return -ewaldParts(alpha, rCut, kCut).total;
}
