// SOAP: the smooth overlap of atomic positions (Bartók, Kondor & Csányi, PRB 87, 184115, 2013).
//
// An atom's neighbours become a density of Gaussians,
//   ρ(r) = Σ_j w_j exp(−α |r − r_j|²),  α = 1 / (2σ²),  w_j = f_cut(r_j)
// (the centre atom included with weight 1, as in the Si GAP of Bartók et al., PRX 8, 041048, Eq. 2).
// The SOAP kernel is the overlap of two densities, squared and averaged over rotations,
//   k̃(ρ, ρ') = ∫ dR̂ |∫ dr ρ(r) ρ'(R̂ r)|²       (normalised Haar measure here),
// then normalised and raised to a power ζ: K = δ² [k̃(ρ,ρ') / √(k̃(ρ,ρ) k̃(ρ',ρ'))]^ζ.
//
// Expanding exp(2α r r_j cos γ) = 4π Σ_lm i_l(2α r r_j) Y_lm(r̂) Y_lm(r̂_j) gives the density's
// coefficients on the sphere of radius r,
//   c_lm(r) = 4π Σ_j w_j exp(−α (r² + r_j²)) i_l(2α r r_j) Y_lm(r̂_j),
// and the Haar average (∫ D^l_mm' D^l'_μμ' dR̂ = δ δ δ / (2l+1)) turns the kernel into a dot product of
// power spectra p_l(r, r') = Σ_m c_lm(r) c_lm(r'):
//   k̃ = Σ_l 1/(2l+1) ∫∫ r² r'² dr dr' p_l(r, r') p'_l(r, r').
// We discretise r on a Gauss–Legendre grid (a radial basis of point functions) instead of QUIP's
// radial basis; the radial integrals of the exact kernel are done in closed form in `exactKernel`
// (Weber's second exponential integral), and the tests check that the two agree.
//
// i_l is the modified spherical Bessel function of the first kind; we only ever need e^{−x} i_l(x).

import { realSH, type Vec3 } from '../physics/spherical-harmonics';
import { gaussLegendre, gaunt } from './grace';

export type { Vec3 };

/** The SOAP settings of the Si GAP, read from its released parameter file gp_iter6_sparse9k.xml. */
export const GAP18_SOAP = {
  lmax: 12,
  nmax: 10,
  sigma: 0.5,
  zeta: 4,
  rc: 5.0,
  width: 1.0,
  delta: 3.0,
  nSparse: 9000,
  dims: 716,
} as const;

/** QUIP's cutoff function (Bartók & Csányi, IJQC 2015, Eq. 24): 1 inside r_c − d, a cosine ramp to 0 at r_c. */
export function fcut(r: number, rc: number, d: number): number {
  if (r <= rc - d) return 1;
  if (r > rc) return 0;
  return 0.5 * (Math.cos((Math.PI * (r - rc + d)) / d) + 1);
}

// ---------------------------------------------------------------------------
// Modified spherical Bessel functions, scaled: e^{−x} i_l(x), l = 0 … lmax
// ---------------------------------------------------------------------------

/** e^{−x} i_0(x) = (1 − e^{−2x}) / (2x), with its limit 1 at x = 0. */
function scaledI0(x: number): number {
  if (x < 1e-6) return 1 - x + (2 * x * x) / 3;
  return -Math.expm1(-2 * x) / (2 * x);
}

/**
 * e^{−x} i_l(x) for l = 0 … lmax, x ≥ 0. Power series for small x (each term positive, no
 * cancellation); otherwise Miller's downward recurrence i_{l−1} = i_{l+1} + (2l+1)/x · i_l, which is
 * stable because i_l decreases with l, normalised by the closed form of i_0.
 */
export function scaledBesselI(lmax: number, x: number): Float64Array {
  const out = new Float64Array(lmax + 1);
  if (x === 0) {
    out[0] = 1;
    return out;
  }
  if (x < 2) {
    const ex = Math.exp(-x);
    const h = (x * x) / 2;
    let lead = 1; // x^l / (2l+1)!!
    for (let l = 0; l <= lmax; l++) {
      if (l > 0) lead *= x / (2 * l + 1);
      let term = 1;
      let sum = 1;
      for (let k = 1; k < 60; k++) {
        term *= h / (k * (2 * l + 2 * k + 1));
        sum += term;
        if (term < 1e-17 * sum) break;
      }
      out[l] = ex * lead * sum;
    }
    return out;
  }
  const start = lmax + 20 + Math.ceil(Math.sqrt(80 * x));
  let up = 0; // i_{l+1}
  let cur = 1e-300; // i_l
  const tmp = new Float64Array(lmax + 1);
  for (let l = start; l > 0; l--) {
    const down = up + ((2 * l + 1) / x) * cur;
    up = cur;
    cur = down;
    if (l - 1 <= lmax) tmp[l - 1] = cur;
    if (cur > 1e250) {
      // Rescale everything stored so far.
      up *= 1e-250;
      cur *= 1e-250;
      for (let k = l - 1; k <= lmax; k++) tmp[k]! *= 1e-250;
    }
  }
  const norm = scaledI0(x) / tmp[0]!;
  for (let l = 0; l <= lmax; l++) out[l] = tmp[l]! * norm;
  return out;
}

// ---------------------------------------------------------------------------
// Density coefficients and the power spectrum on a radial grid
// ---------------------------------------------------------------------------

export interface SoapOptions {
  /** Gaussian width σ (Å); α = 1/(2σ²). */
  sigma: number;
  /** Cutoff radius r_c (Å). */
  rc: number;
  /** Width d of the cosine ramp (Å). */
  width: number;
  lmax: number;
  /** Radial grid nodes on 0 ≤ r ≤ r_c + 3σ. */
  nr: number;
  /** Weight of the centre atom's own Gaussian (QUIP's central_weight). */
  centralWeight: number;
}

export const DEFAULT_SOAP: SoapOptions = { sigma: 0.5, rc: 5.0, width: 1.0, lmax: 6, nr: 20, centralWeight: 1 };

const lm = (l: number, m: number) => l * l + l + m;

/** Neighbours that enter the density: inside r_c, each with its cutoff weight. The centre is added separately. */
export function weighted(env: Vec3[], o: Pick<SoapOptions, 'rc' | 'width'>): Array<{ v: Vec3; r: number; w: number }> {
  const out: Array<{ v: Vec3; r: number; w: number }> = [];
  for (const v of env) {
    const r = Math.hypot(v[0], v[1], v[2]);
    if (r < 1e-9 || r >= o.rc) continue;
    out.push({ v, r, w: fcut(r, o.rc, o.width) });
  }
  return out;
}

export interface RadialGrid {
  r: number[];
  w: number[];
}

const gridCache = new Map<string, RadialGrid>();

export function radialGrid(o: Pick<SoapOptions, 'rc' | 'sigma' | 'nr'>): RadialGrid {
  const R = o.rc + 3 * o.sigma;
  const key = `${R},${o.nr}`;
  const hit = gridCache.get(key);
  if (hit) return hit;
  const { x, w } = gaussLegendre(o.nr);
  const g = { r: x.map((t) => (R * (t + 1)) / 2), w: w.map((t) => (R * t) / 2) };
  gridCache.set(key, g);
  return g;
}

/** c_lm(r_a) for every grid node a, flattened as [a][lm]. */
export function densityCoefficients(env: Vec3[], o: SoapOptions): Float64Array {
  const { r: rs } = radialGrid(o);
  const alpha = 1 / (2 * o.sigma * o.sigma);
  const L = o.lmax;
  const nlm = (L + 1) ** 2;
  const c = new Float64Array(rs.length * nlm);
  const atoms = weighted(env, o);
  // Centre atom: only l = 0 survives (i_l(0) = δ_l0), Y_00 = 1/√(4π).
  rs.forEach((r, a) => {
    c[a * nlm]! += o.centralWeight * 4 * Math.PI * Math.exp(-alpha * r * r) / Math.sqrt(4 * Math.PI);
  });
  for (const { v, r: rj, w } of atoms) {
    const Y = new Float64Array(nlm);
    for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) Y[lm(l, m)] = realSH(l, m, v);
    rs.forEach((r, a) => {
      const g = Math.exp(-alpha * (r - rj) ** 2); // exp(−α(r² + r_j²)) e^{2α r r_j}
      if (g < 1e-300) return;
      const I = scaledBesselI(L, 2 * alpha * r * rj);
      const base = a * nlm;
      for (let l = 0; l <= L; l++) {
        const f = 4 * Math.PI * w * g * I[l]!;
        for (let m = -l; m <= l; m++) c[base + lm(l, m)]! += f * Y[lm(l, m)]!;
      }
    });
  }
  return c;
}

/** Length of the power-spectrum vector: (lmax + 1) · nr(nr + 1)/2. */
export const soapLength = (o: Pick<SoapOptions, 'lmax' | 'nr'>) => ((o.lmax + 1) * o.nr * (o.nr + 1)) / 2;

/**
 * The SOAP power spectrum as a vector whose dot product is the Haar-averaged overlap k̃:
 * entries (l, a ≤ b) = √(1/(2l+1)) · √(w_a w_b) r_a r_b · p_l(r_a, r_b), off-diagonal pairs × √2.
 * Ordered l-major, so `powerByL` can read off the share of each l.
 */
export function powerSpectrum(env: Vec3[], o: SoapOptions): Float64Array {
  const { r: rs, w: ws } = radialGrid(o);
  const c = densityCoefficients(env, o);
  const n = rs.length;
  const L = o.lmax;
  const nlm = (L + 1) ** 2;
  const out = new Float64Array(soapLength(o));
  let k = 0;
  for (let l = 0; l <= L; l++) {
    const fl = Math.sqrt(1 / (2 * l + 1));
    for (let a = 0; a < n; a++)
      for (let b = a; b < n; b++) {
        let p = 0;
        for (let m = -l; m <= l; m++) p += c[a * nlm + lm(l, m)]! * c[b * nlm + lm(l, m)]!;
        const s = fl * Math.sqrt(ws[a]! * ws[b]!) * rs[a]! * rs[b]! * (a === b ? 1 : Math.SQRT2);
        out[k++] = s * p;
      }
  }
  return out;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

export function normalise(p: Float64Array): Float64Array {
  const n = Math.sqrt(dot(p, p)) || 1;
  return p.map((x) => x / n);
}

/** The normalised SOAP vector p̂ = p / |p| of one environment. */
export const soapVector = (env: Vec3[], o: SoapOptions) => normalise(powerSpectrum(env, o));

/** K = δ² (p̂·p̂')^ζ for normalised vectors. */
export const soapKernel = (a: ArrayLike<number>, b: ArrayLike<number>, zeta: number, delta = 1) => delta * delta * Math.max(0, dot(a, b)) ** zeta;

/** Kernel distance √(2 − 2K) for δ = 1: 0 for identical environments. */
export const kernelDistance = (a: ArrayLike<number>, b: ArrayLike<number>, zeta: number) => Math.sqrt(Math.max(0, 2 - 2 * soapKernel(a, b, zeta)));

/** Share of |p|² carried by each angular channel l (vector from `powerSpectrum`). */
export function powerByL(p: Float64Array, o: Pick<SoapOptions, 'lmax' | 'nr'>): number[] {
  const per = (o.nr * (o.nr + 1)) / 2;
  const tot = dot(p, p) || 1;
  return Array.from({ length: o.lmax + 1 }, (_, l) => {
    let s = 0;
    for (let k = l * per; k < (l + 1) * per; k++) s += p[k]! * p[k]!;
    return s / tot;
  });
}

// ---------------------------------------------------------------------------
// Exact rotation-averaged overlap (closed-form radial integrals), for tests
// ---------------------------------------------------------------------------

/**
 * k̃(ρ, ρ') = Σ_{l ≤ lmax} 1/(2l+1) Σ_mm' (I^l_mm')², with
 *   I^l_mm' = C Σ_{j j'} w_j w_j' exp(−α (r_j² + r_j'²)/2) i_l(α r_j r_j') Y_lm(r̂_j) Y_lm'(r̂_j'),
 *   C = (4π)² · (1/(8α)) √(π/(2α)),
 * from ∫₀^∞ r² e^{−p r²} i_l(a r) i_l(b r) dr = (1/(4p)) √(π/p) e^{(a²+b²)/(4p)} i_l(ab/(2p)).
 */
export function exactKernel(envA: Vec3[], envB: Vec3[], o: SoapOptions): number {
  const alpha = 1 / (2 * o.sigma * o.sigma);
  const C = 16 * Math.PI * Math.PI * (1 / (8 * alpha)) * Math.sqrt(Math.PI / (2 * alpha));
  const withCentre = (env: Vec3[]) => [{ v: [0, 0, 1] as Vec3, r: 0, w: o.centralWeight }, ...weighted(env, o)];
  const A = withCentre(envA);
  const B = withCentre(envB);
  let k = 0;
  for (let l = 0; l <= o.lmax; l++) {
    const d = 2 * l + 1;
    const I = new Float64Array(d * d);
    for (const a of A)
      for (const b of B) {
        const x = alpha * a.r * b.r;
        const f = C * a.w * b.w * Math.exp(-alpha * ((a.r - b.r) ** 2) / 2) * scaledBesselI(l, x)[l]!; // e^{−α(ra²+rb²)/2} i_l(x) = e^{−α(ra−rb)²/2} e^{−x} i_l(x)
        if (f === 0) continue;
        for (let m = -l; m <= l; m++) {
          const ya = realSH(l, m, a.v);
          for (let mp = -l; mp <= l; mp++) I[(m + l) * d + mp + l]! += f * ya * realSH(l, mp, b.v);
        }
      }
    k += dot(I, I) / d;
  }
  return k;
}

/** The overlap ∫ ρ(r) ρ'(R r) dr for a rotation R (row-major), in closed form: (π/2α)^{3/2} Σ w w' e^{−α|r_j − R r_j'|²/2}. */
export function overlap(envA: Vec3[], envB: Vec3[], R: number[], o: SoapOptions): number {
  const alpha = 1 / (2 * o.sigma * o.sigma);
  const pre = (Math.PI / (2 * alpha)) ** 1.5;
  const withCentre = (env: Vec3[]) => [{ v: [0, 0, 0] as Vec3, w: o.centralWeight }, ...weighted(env, o).map(({ v, w }) => ({ v, w }))];
  const A = withCentre(envA);
  const B = withCentre(envB).map(({ v, w }) => ({
    v: [R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2], R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2], R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2]] as Vec3,
    w,
  }));
  let s = 0;
  for (const a of A) for (const b of B) s += a.w * b.w * Math.exp((-alpha * ((a.v[0] - b.v[0]) ** 2 + (a.v[1] - b.v[1]) ** 2 + (a.v[2] - b.v[2]) ** 2)) / 2);
  return pre * s;
}

// ---------------------------------------------------------------------------
// A truncated bispectrum (four-body), for the degenerate pair
// ---------------------------------------------------------------------------

/**
 * Radial channels A_nlm = ∫ r² dr g_n(r) c_lm(r) with Gaussian g_n centred at `centres`, then
 * b_{n1n2n3,l1l2l3} = Σ_m G(l1m1, l2m2, l3m3) A_{n1l1m1} A_{n2l2m2} A_{n3l3m3} for n1 ≤ n2 ≤ n3,
 * all (l1 ≤ l2 ≤ l3) ≤ lmax allowed by the Gaunt selection rules. Invariant under rotations
 * (each b is a full contraction with Gaunt coefficients) and permutations.
 */
export function bispectrum(env: Vec3[], o: SoapOptions, centres: number[], width = 0.5): Float64Array {
  const { r: rs, w: ws } = radialGrid(o);
  const c = densityCoefficients(env, o);
  const L = o.lmax;
  const nlm = (L + 1) ** 2;
  const A = centres.map((rn) => {
    const out = new Float64Array(nlm);
    rs.forEach((r, a) => {
      const g = ws[a]! * r * r * Math.exp(-((r - rn) ** 2) / (2 * width * width));
      for (let k = 0; k < nlm; k++) out[k]! += g * c[a * nlm + k]!;
    });
    return out;
  });
  const vals: number[] = [];
  const nn = centres.length;
  for (let n1 = 0; n1 < nn; n1++)
    for (let n2 = n1; n2 < nn; n2++)
      for (let n3 = n2; n3 < nn; n3++)
        for (let l1 = 0; l1 <= L; l1++)
          for (let l2 = l1; l2 <= L; l2++)
            for (let l3 = l2; l3 <= Math.min(L, l1 + l2); l3++) {
              if ((l1 + l2 + l3) % 2) continue;
              const G = gaunt(l1, l2, l3);
              const d2 = 2 * l2 + 1;
              const d3 = 2 * l3 + 1;
              let s = 0;
              for (let i = 0; i < 2 * l1 + 1; i++) {
                const a1 = A[n1]![l1 * l1 + i]!;
                if (a1 === 0) continue;
                for (let j = 0; j < d2; j++) {
                  const a12 = a1 * A[n2]![l2 * l2 + j]!;
                  for (let k = 0; k < d3; k++) s += G[(i * d2 + j) * d3 + k]! * a12 * A[n3]![l3 * l3 + k]!;
                }
              }
              vals.push(s);
            }
  return new Float64Array(vals);
}

// ---------------------------------------------------------------------------
// Environments: lattices and Pozdnyakov's degenerate pair
// ---------------------------------------------------------------------------

export type LatticeKind = 'fcc' | 'hcp' | 'bcc' | 'sc' | 'diamond';

/** Neighbours within R of an atom in a monatomic lattice whose nearest-neighbour distance is d. */
export function latticeEnv(kind: LatticeKind, d: number, R: number): Vec3[] {
  const out: Vec3[] = [];
  const push = (p: Vec3) => {
    const r = Math.hypot(p[0], p[1], p[2]);
    if (r > 1e-9 && r < R) out.push(p);
  };
  if (kind === 'hcp') {
    const c = d * Math.sqrt(8 / 3);
    const n = Math.ceil(R / d) + 2;
    for (let i = -n; i <= n; i++)
      for (let j = -n; j <= n; j++)
        for (let k = -n; k <= n; k++)
          for (const b of [[0, 0, 0], [1 / 3, 2 / 3, 0.5]] as const) {
            const u = i + b[0];
            const v = j + b[1];
            push([d * (u - v / 2), d * v * (Math.sqrt(3) / 2), (k + b[2]) * c]);
          }
    return out;
  }
  const basis: Record<Exclude<LatticeKind, 'hcp'>, { a: number; b: Vec3[] }> = {
    fcc: { a: d * Math.SQRT2, b: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] },
    bcc: { a: (2 * d) / Math.sqrt(3), b: [[0, 0, 0], [0.5, 0.5, 0.5]] },
    sc: { a: d, b: [[0, 0, 0]] },
    diamond: {
      a: (4 * d) / Math.sqrt(3),
      b: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]],
    },
  };
  const { a, b } = basis[kind];
  const n = Math.ceil(R / a) + 1;
  for (let i = -n; i <= n; i++) for (let j = -n; j <= n; j++) for (let k = -n; k <= n; k++) for (const s of b) push([a * (i + s[0]), a * (j + s[1]), a * (k + s[2])]);
  return out;
}

/**
 * Pozdnyakov et al. (PRL 125, 166001, 2020), Fig. 1a: four neighbours on a circle of radius r around
 * the centre, at multiples of 45°. X⁺ uses {0, 1, 3, 4} × 45° and X⁻ {0, 1, 2, 5} × 45°; both have the
 * angles 45°, 45°, 90°, 135°, 135°, 180° between neighbours but cannot be superimposed. `lift` moves
 * the first neighbour (at 0°) out of the plane by that many Å in both, which breaks the degeneracy.
 */
export function degeneratePair(r: number, lift = 0): { plus: Vec3[]; minus: Vec3[] } {
  const at = (k: number, z = 0): Vec3 => [r * Math.cos((k * Math.PI) / 4), r * Math.sin((k * Math.PI) / 4), z];
  return {
    plus: [at(0, lift), at(1), at(3), at(4)],
    minus: [at(0, lift), at(1), at(2), at(5)],
  };
}

/** Sorted angles (degrees) between all pairs of neighbours, seen from the centre. */
export function pairAngles(env: Vec3[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < env.length; i++)
    for (let j = i + 1; j < env.length; j++) {
      const a = env[i]!;
      const b = env[j]!;
      const c = dot(a, b) / (Math.hypot(...a) * Math.hypot(...b));
      out.push((Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI);
    }
  return out.sort((x, y) => x - y);
}
