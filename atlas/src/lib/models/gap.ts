// A Gaussian approximation potential (GAP) fitted from scratch, small enough to train in a browser.
//
// The total energy is a sum of local energies, ε(i) = Σ_s α_s K(p̂_i, p̂_s), over M representative
// ("sparse") environments s, with the SOAP kernel K = δ² (p̂·p̂')^ζ (src/lib/models/soap.ts).
// Only total energies are observed. With L the operator that sums atoms into their configuration,
// the regularised least-squares (sparse GP) solution is (Bartók & Csányi, IJQC 2015, Eq. 32;
// Bartók et al., PRX 8, 041048, Eq. 9)
//   α = (K_MM + K_MN L Λ⁻¹ Lᵀ K_NM)⁻¹ K_MN L Λ⁻¹ y,
// with Λ diagonal: the expected error of each total energy (σ_E² per atom, times the number of atoms).
// The uncertainty is the simplified per-atom variance of the Si GAP (PRX 8, 041048, Eq. 10),
//   σ²(i) = K(p̂_i, p̂_i) − kᵀ (K_MM + σ_e I)⁻¹ k,   k_s = K(p̂_i, p̂_s),
// which depends only on how far an environment is from the representative set, not on the data.
//
// The "DFT" here is the Stillinger–Weber potential for Si, so every reference energy is exact and the
// training cells can be generated on the fly (8-atom cubic diamond cells, periodic).

import { mulberry32 } from '../chem/rng';
import { swPerAtom } from '../physics/stillinger-weber';
import { soapKernel, soapVector, type SoapOptions, type Vec3 } from './soap';

/** Stillinger–Weber's diamond lattice constant (Å): σ = 2.0951 Å was chosen to reproduce it. */
export const SI_A0 = 5.431;

export interface Cell {
  /** Cubic box edge (Å). */
  L: number;
  positions: Vec3[];
}

const DIAMOND: Vec3[] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]];

/** The 8-atom conventional diamond cell with edge a. */
export const diamondCell = (a = SI_A0): Cell => ({ L: a, positions: DIAMOND.map((f) => [f[0] * a, f[1] * a, f[2] * a]) });

/** Isotropic strain ε: every length × (1 + ε). */
export const strained = (c: Cell, eps: number): Cell => ({ L: c.L * (1 + eps), positions: c.positions.map((p) => p.map((x) => x * (1 + eps)) as Vec3) });

/** Every atom displaced by Gaussian noise of `amp` Å per component (seeded). */
export function rattled(c: Cell, amp: number, seed: number): Cell {
  const rand = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  return { L: c.L, positions: c.positions.map((p) => [p[0] + amp * gauss(), p[1] + amp * gauss(), p[2] + amp * gauss()] as Vec3) };
}

/** Per-atom neighbour vectors within rc, over all periodic images. */
export function neighbourVectors(c: Cell, rc: number): Vec3[][] {
  const n = Math.ceil(rc / c.L);
  return c.positions.map((pi, i) => {
    const out: Vec3[] = [];
    c.positions.forEach((pj, j) => {
      for (let a = -n; a <= n; a++)
        for (let b = -n; b <= n; b++)
          for (let k = -n; k <= n; k++) {
            if (i === j && a === 0 && b === 0 && k === 0) continue;
            const v: Vec3 = [pj[0] + a * c.L - pi[0], pj[1] + b * c.L - pi[1], pj[2] + k * c.L - pi[2]];
            if (Math.hypot(v[0], v[1], v[2]) < rc) out.push(v);
          }
    });
    return out;
  });
}

/** Stillinger–Weber total energy of a periodic cell (eV; the isolated atom is 0). */
export function swEnergy(c: Cell): number {
  const e = swPerAtom(neighbourVectors(c, 3.8));
  return (e.two + e.three) * c.positions.length;
}

/** Normalised SOAP vectors of every atom in a cell. */
export const cellDescriptors = (c: Cell, o: SoapOptions): Float64Array[] => neighbourVectors(c, o.rc).map((env) => soapVector(env, o));

// ---------------------------------------------------------------------------
// Dense linear algebra (small, symmetric positive definite)
// ---------------------------------------------------------------------------

/** Cholesky factor (lower, row-major n×n) of a symmetric positive-definite matrix. */
export function cholesky(A: Float64Array, n: number): Float64Array {
  const Lm = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j]!;
      for (let k = 0; k < j; k++) s -= Lm[i * n + k]! * Lm[j * n + k]!;
      if (i === j) {
        if (s <= 0) throw new Error('matrix is not positive definite');
        Lm[i * n + i] = Math.sqrt(s);
      } else Lm[i * n + j] = s / Lm[j * n + j]!;
    }
  return Lm;
}

/** Cholesky with escalating diagonal jitter for nearly singular kernel matrices (duplicate environments). */
export function choleskyJittered(A: Float64Array, n: number, scale: number): Float64Array {
  for (let j = 0, eps = 0; j < 8; j++, eps = eps ? eps * 100 : 1e-10 * scale) {
    const B = A.slice();
    for (let i = 0; i < n; i++) B[i * n + i]! += eps;
    try {
      return cholesky(B, n);
    } catch {
      /* add more jitter */
    }
  }
  throw new Error('matrix is not positive definite');
}

/** Least-squares solution of A x ≈ b (A is rows × cols, row-major, rows ≥ cols) by Householder QR. */
export function leastSquares(A0: Float64Array, rows: number, cols: number, b0: ArrayLike<number>): Float64Array {
  const A = A0.slice();
  const b = Float64Array.from(b0);
  for (let k = 0; k < cols; k++) {
    let norm = 0;
    for (let i = k; i < rows; i++) norm += A[i * cols + k]! ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = A[k * cols + k]! > 0 ? -norm : norm;
    const v = new Float64Array(rows - k);
    for (let i = k; i < rows; i++) v[i - k] = A[i * cols + k]!;
    v[0]! -= alpha;
    let vv = 0;
    for (let i = 0; i < v.length; i++) vv += v[i]! * v[i]!;
    if (vv === 0) continue;
    for (let j = k; j < cols; j++) {
      let d = 0;
      for (let i = k; i < rows; i++) d += v[i - k]! * A[i * cols + j]!;
      const f = (2 * d) / vv;
      for (let i = k; i < rows; i++) A[i * cols + j]! -= f * v[i - k]!;
    }
    let d = 0;
    for (let i = k; i < rows; i++) d += v[i - k]! * b[i]!;
    const f = (2 * d) / vv;
    for (let i = k; i < rows; i++) b[i]! -= f * v[i - k]!;
  }
  const x = new Float64Array(cols);
  for (let i = cols - 1; i >= 0; i--) {
    let s = b[i]!;
    for (let j = i + 1; j < cols; j++) s -= A[i * cols + j]! * x[j]!;
    x[i] = s / A[i * cols + i]!;
  }
  return x;
}

/** Solve (L Lᵀ) x = b. */
export function cholSolve(Lm: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= Lm[i * n + k]! * y[k]!;
    y[i] = s / Lm[i * n + i]!;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!;
    for (let k = i + 1; k < n; k++) s -= Lm[k * n + i]! * y[k]!;
    y[i] = s / Lm[i * n + i]!;
  }
  return y;
}

// ---------------------------------------------------------------------------
// Fitting and prediction
// ---------------------------------------------------------------------------

export interface TrainingConfig {
  /** Normalised SOAP vector of every atom. */
  envs: Float64Array[];
  /** Total energy (eV). */
  energy: number;
}

export interface GapSettings {
  zeta: number;
  /** Energy scale δ (eV). */
  delta: number;
  /** Expected error per atom of the total energies, σ_E (eV). */
  sigmaE: number;
  /** Regulariser of the variance formula, σ_e (eV², added to K_MM). */
  sigmaVar: number;
  /** Jitter on K_MM's diagonal (relative to δ²). */
  jitter: number;
}

export const DEFAULT_GAP: GapSettings = { zeta: 4, delta: 3.0, sigmaE: 0.001, sigmaVar: 1e-6, jitter: 1e-8 };

export interface GapModel {
  settings: GapSettings;
  sparse: Float64Array[];
  alpha: Float64Array;
  /** Cholesky factor of K_MM + σ_e I, for the variance. */
  varFactor: Float64Array;
}

/**
 * Choose M representative environments by farthest-point sampling in the kernel distance, starting
 * from the first environment. (The Si GAP used CUR decomposition of the descriptor matrix instead.)
 */
export function selectSparse(envs: Float64Array[], M: number, zeta: number): number[] {
  const n = envs.length;
  const chosen = [0];
  const dist = envs.map((e) => 2 - 2 * soapKernel(e, envs[0]!, zeta));
  while (chosen.length < Math.min(M, n)) {
    let best = -1;
    let far = -1;
    for (let i = 0; i < n; i++)
      if (dist[i]! > far) {
        far = dist[i]!;
        best = i;
      }
    if (far <= 1e-14) break; // everything left duplicates a chosen environment
    chosen.push(best);
    for (let i = 0; i < n; i++) dist[i] = Math.min(dist[i]!, 2 - 2 * soapKernel(envs[i]!, envs[best]!, zeta));
  }
  return chosen;
}

export function fitGap(configs: TrainingConfig[], sparse: Float64Array[], s: GapSettings = DEFAULT_GAP): GapModel {
  const M = sparse.length;
  const T = configs.length;
  const d2 = s.delta * s.delta;
  const K = (a: Float64Array, b: Float64Array) => soapKernel(a, b, s.zeta, s.delta);
  const Kmm = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) Kmm[i * M + j] = K(sparse[i]!, sparse[j]!);
  // (K_MN L)_{s t} = Σ_{i ∈ t} K(s, i)
  const KmT = new Float64Array(M * T);
  configs.forEach((c, t) => {
    for (let i = 0; i < M; i++) {
      let v = 0;
      for (const e of c.envs) v += K(sparse[i]!, e);
      KmT[i * T + t] = v;
    }
  });
  // Eq. (5) are the normal equations of a regularised least-squares problem,
  //   min ‖Λ^{-1/2} (y − (K_MN L)ᵀ α)‖² + αᵀ K_MM α,   with K_MM = U Uᵀ.
  // Solving the stacked system [Λ^{-1/2} (K_MN L)ᵀ ; Uᵀ] α ≈ [Λ^{-1/2} y ; 0] by QR squares-roots the
  // condition number, so the result no longer depends on last-bit differences between JS engines.
  const Kj = Kmm.slice();
  for (let i = 0; i < M; i++) Kj[i * M + i]! += s.jitter * d2;
  const U = choleskyJittered(Kj, M, d2);
  const rows = T + M;
  const S = new Float64Array(rows * M);
  const rhs = new Float64Array(rows);
  configs.forEach((c, t) => {
    const w = 1 / (s.sigmaE * Math.sqrt(c.envs.length));
    for (let i = 0; i < M; i++) S[t * M + i] = KmT[i * T + t]! * w;
    rhs[t] = c.energy * w;
  });
  for (let i = 0; i < M; i++) for (let j = i; j < M; j++) S[(T + i) * M + j] = U[j * M + i]!; // row i of Uᵀ
  const alpha = leastSquares(S, rows, M, rhs);
  const V = Kmm.slice();
  for (let i = 0; i < M; i++) V[i * M + i]! += s.sigmaVar + s.jitter * d2;
  return { settings: s, sparse, alpha, varFactor: choleskyJittered(V, M, d2) };
}

export interface Prediction {
  energy: number;
  local: number[];
  /** Per-atom σ from the simplified variance (eV). */
  sigma: number[];
}

export function predict(model: GapModel, envs: Float64Array[]): Prediction {
  const s = model.settings;
  const M = model.sparse.length;
  const local: number[] = [];
  const sigma: number[] = [];
  for (const e of envs) {
    const k = new Float64Array(M);
    for (let i = 0; i < M; i++) k[i] = soapKernel(e, model.sparse[i]!, s.zeta, s.delta);
    let eps = 0;
    for (let i = 0; i < M; i++) eps += model.alpha[i]! * k[i]!;
    local.push(eps);
    const x = cholSolve(model.varFactor, M, k);
    let q = 0;
    for (let i = 0; i < M; i++) q += k[i]! * x[i]!;
    sigma.push(Math.sqrt(Math.max(0, soapKernel(e, e, s.zeta, s.delta) - q)));
  }
  return { energy: local.reduce((a, b) => a + b, 0), local, sigma };
}

// ---------------------------------------------------------------------------
// The lab's training set: strained and rattled diamond cells
// ---------------------------------------------------------------------------

export interface TrainingSpec {
  count: number;
  /** Strains drawn uniformly from [−maxStrain, maxStrain]. */
  maxStrain: number;
  /** Rattle amplitudes drawn uniformly from [0, maxRattle] Å. */
  maxRattle: number;
  seed: number;
}

export function trainingCells(spec: TrainingSpec): Cell[] {
  const rand = mulberry32(spec.seed);
  return Array.from({ length: spec.count }, (_, k) => {
    const eps = (2 * rand() - 1) * spec.maxStrain;
    const amp = rand() * spec.maxRattle;
    return rattled(strained(diamondCell(), eps), amp, spec.seed * 1000 + k + 1);
  });
}
