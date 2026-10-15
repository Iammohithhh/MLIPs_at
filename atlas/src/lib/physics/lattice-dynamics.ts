// Rigid-ion lattice dynamics of rock salt (Electrochemistry Lab, LO–TO splitting).
// TOY MODEL of Kellermann type: point ions ±Z on the NaCl lattice, a nearest-neighbour
// short-range pair term, and the Coulomb interaction either summed exactly (Ewald) or in a
// standard local form (damped shifted force, zero beyond r_c). Phonon frequencies come
// from the 6×6 dynamical matrix; in rock salt both ions sit on inversion centres, so with
// the phase convention e^{iq·(R+τ'−τ)} the matrix is real and symmetric.

import { COULOMB_K } from '../chem/elements';
import { MADELUNG_NACL, erfc } from './madelung';

type V3 = [number, number, number];

/** Standard atomic weights (IUPAC 2021 abridged values), amu. */
export const MASS = { Na: 22.98977, Cl: 35.45 } as const;
/** √(eV / (Å² amu)) expressed as an ordinary frequency ω/2π in THz. */
export const THZ = 15.633302;

export interface RockSaltModel {
  /** Nearest-neighbour distance (Å). */
  r0: number;
  /** Ionic charge magnitude (e). */
  Z: number;
  /** Range ρ of a Born–Mayer repulsion (Å); sets the radial stiffness A = −B r0/ρ. */
  rho: number;
  /** Exact long-range Coulomb (Ewald), or a local model: damped shifted force Coulomb that vanishes
   *  beyond rc, with the short-range stiffness refitted to the exact TO frequency at Γ. */
  coulomb: 'ewald' | 'cutoff';
  rc?: number;
  /** Damping of the local (DSF) Coulomb, Å⁻¹. */
  alpha?: number;
}

export interface Lattice {
  a: [V3, V3, V3];
  b: [V3, V3, V3];
  omega: number;
  tau: [V3, V3];
  charges: [number, number];
  masses: [number, number];
}

export function rockSalt(r0: number, Z: number): Lattice {
  const a: [V3, V3, V3] = [
    [0, r0, r0],
    [r0, 0, r0],
    [r0, r0, 0],
  ];
  const omega = 2 * r0 ** 3;
  const k = Math.PI / r0; // 2π/a with a = 2 r0
  const b: [V3, V3, V3] = [
    [-k, k, k],
    [k, -k, k],
    [k, k, -k],
  ];
  return { a, b, omega, tau: [[0, 0, 0], [r0, 0, 0]], charges: [Z, -Z], masses: [MASS.Na, MASS.Cl] };
}

/** Short-range constants: B = φ′(r0)/r0 fixed by equilibrium with the Madelung energy, A = φ″(r0). */
export function shortRange(m: RockSaltModel) {
  const B = -(MADELUNG_NACL * COULOMB_K * m.Z * m.Z) / (6 * m.r0 ** 3);
  return { A: (-B * m.r0) / m.rho, B };
}

const dot = (u: V3, v: V3) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
type M3 = number[][];
const zero3 = (): M3 => [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];

/** Lattice vectors R_l + d with |R_l + d| < rmax (excluding the zero vector). */
function shell(L: Lattice, d: V3, rmax: number): V3[] {
  const n = Math.ceil(rmax / L.a[0][1]) + 2;
  const out: V3[] = [];
  for (let i = -n; i <= n; i++)
    for (let j = -n; j <= n; j++)
      for (let l = -n; l <= n; l++) {
        const r: V3 = [
          i * L.a[0][0] + j * L.a[1][0] + l * L.a[2][0] + d[0],
          i * L.a[0][1] + j * L.a[1][1] + l * L.a[2][1] + d[1],
          i * L.a[0][2] + j * L.a[1][2] + l * L.a[2][2] + d[2],
        ];
        const rr = Math.hypot(...r);
        if (rr > 1e-9 && rr < rmax) out.push(r);
      }
  return out;
}

const shellCache = new Map<string, V3[]>();
function cachedShell(L: Lattice, d: V3, rmax: number) {
  const key = `${L.a[0][1]}|${d.join(',')}|${rmax}`;
  let s = shellCache.get(key);
  if (!s) {
    s = shell(L, d, rmax);
    shellCache.set(key, s);
  }
  return s;
}

/** Add c·[g″ r̂r̂ + (g′/r)(1 − r̂r̂)] for a radial function g at vector r. */
function addRadial(T: M3, r: V3, g1: number, g2: number, c: number) {
  const rr = Math.hypot(...r);
  for (let x = 0; x < 3; x++)
    for (let y = 0; y < 3; y++) {
      const rr2 = (r[x]! * r[y]!) / (rr * rr);
      T[x]![y]! += c * (g2 * rr2 + (g1 / rr) * ((x === y ? 1 : 0) - rr2));
    }
}

/**
 * T_αβ(q, d) = Σ′_l ∂α∂β(1/r) cos(q·r), r = R_l + d, by Ewald summation (η in Å⁻¹).
 * qhat gives the direction of approach when q = 0 (the non-analytic G = 0 term);
 * pass null to drop the G = 0 term (allowed when the result is multiplied by a zero total charge).
 */
export function coulombT(L: Lattice, q: V3, d: V3, qhat: V3 | null, eta = 0.55, rCut = 11, gCut = 9): M3 {
  const T = zero3();
  const sp = 2 / Math.sqrt(Math.PI);
  for (const r of cachedShell(L, d, rCut)) {
    const rr = Math.hypot(...r);
    const ec = erfc(eta * rr);
    const ex = sp * eta * Math.exp(-eta * eta * rr * rr);
    const g1 = -ec / (rr * rr) - ex / rr;
    const g2 = (2 * ec) / rr ** 3 + ex * (2 / (rr * rr) + 2 * eta * eta);
    addRadial(T, r, g1, g2, Math.cos(dot(q, r)));
  }
  const n = Math.ceil(gCut / Math.hypot(...L.b[0])) + 1;
  for (let i = -n; i <= n; i++)
    for (let j = -n; j <= n; j++)
      for (let l = -n; l <= n; l++) {
        const G: V3 = [
          i * L.b[0][0] + j * L.b[1][0] + l * L.b[2][0],
          i * L.b[0][1] + j * L.b[1][1] + l * L.b[2][1],
          i * L.b[0][2] + j * L.b[1][2] + l * L.b[2][2],
        ];
        const k: V3 = [q[0] + G[0], q[1] + G[1], q[2] + G[2]];
        const k2 = dot(k, k);
        if (k2 > gCut * gCut) continue;
        let kk: V3 = k;
        let w: number;
        if (k2 < 1e-14) {
          if (!qhat) continue;
          const l2 = Math.hypot(...qhat);
          kk = [qhat[0] / l2, qhat[1] / l2, qhat[2] / l2];
          w = (4 * Math.PI) / L.omega;
        } else {
          w = ((4 * Math.PI) / (L.omega * k2)) * Math.exp(-k2 / (4 * eta * eta));
        }
        const c = -w * Math.cos(dot(G, d));
        for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) T[x]![y]! += c * kk[x]! * kk[y]!;
      }
  if (Math.hypot(...d) < 1e-12) {
    const self = (4 * eta ** 3) / (3 * Math.sqrt(Math.PI));
    for (let x = 0; x < 3; x++) T[x]![x]! += self;
  }
  return T;
}

/**
 * The lattice sum for a local Coulomb kernel: the damped shifted force form
 * g(r) = erfc(αr)/r + C r + const for r < rc (zero beyond), with C chosen so g′(rc) = 0.
 */
export function coulombTCut(L: Lattice, q: V3, d: V3, rc: number, alpha = 0.2): M3 {
  const T = zero3();
  const sp = 2 / Math.sqrt(Math.PI);
  const C = erfc(alpha * rc) / (rc * rc) + (sp * alpha * Math.exp(-alpha * alpha * rc * rc)) / rc;
  for (const r of cachedShell(L, d, rc)) {
    const rr = Math.hypot(...r);
    const ec = erfc(alpha * rr);
    const ex = sp * alpha * Math.exp(-alpha * alpha * rr * rr);
    const g1 = -ec / (rr * rr) - ex / rr + C;
    const g2 = (2 * ec) / rr ** 3 + ex * (2 / (rr * rr) + 2 * alpha * alpha);
    addRadial(T, r, g1, g2, Math.cos(dot(q, r)));
  }
  return T;
}

const refitCache = new Map<string, number>();

/** Mass-weighted dynamical matrix (6×6, eV/(Å² amu)) at wavevector q (Å⁻¹). */
export function dynamicalMatrix(m: RockSaltModel, q: V3, qhat?: V3): number[][] {
  const sr = shortRange(m);
  if (m.coulomb === 'ewald') return rawMatrix(m, sr.A, q, qhat);
  // The local model's radial stiffness is refitted so that its Γ optical frequency equals the exact
  // TO one, as a model trained on periodic Γ-point data (zero macroscopic field) would learn. The
  // exact TO satisfies μω² = 2A + 4B − 4πkZ²/(3Ω) (Lorentz local field; see the tests).
  const key = `${m.r0}|${m.Z}|${m.rho}|${m.rc}|${m.alpha}`;
  let A = refitCache.get(key);
  if (A === undefined) {
    const L = rockSalt(m.r0, m.Z);
    const mu = (L.masses[0] * L.masses[1]) / (L.masses[0] + L.masses[1]);
    const target = 2 * sr.A + 4 * sr.B - (4 * Math.PI * COULOMB_K * m.Z * m.Z) / (3 * L.omega);
    const lam = jacobiEigen(rawMatrix(m, sr.A, [0, 0, 0], [1, 0, 0])).values.at(-1)!;
    A = sr.A + (target - mu * lam) / 2;
    refitCache.set(key, A);
  }
  return rawMatrix(m, A, q, qhat);
}

function rawMatrix(m: RockSaltModel, A: number, q: V3, qhat?: V3): number[][] {
  const L = rockSalt(m.r0, m.Z);
  const B = shortRange(m).B;
  const D = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
  const dir = qhat ?? (Math.hypot(...q) > 1e-12 ? q : ([1, 0, 0] as V3));
  const T = (qq: V3, d: V3, hat: V3 | null) =>
    m.coulomb === 'ewald' ? coulombT(L, qq, d, hat) : coulombTCut(L, qq, d, m.rc ?? 8, m.alpha);
  for (let s = 0; s < 2; s++)
    for (let t = 0; t < 2; t++) {
      const d: V3 = [L.tau[t]![0] - L.tau[s]![0], L.tau[t]![1] - L.tau[s]![1], L.tau[t]![2] - L.tau[s]![2]];
      const blk = zero3();
      // Coulomb: −Z_s Z_t k T(q, d) ...
      const Tq = T(q, d, dir);
      for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) blk[x]![y]! -= L.charges[s]! * L.charges[t]! * COULOMB_K * Tq[x]![y]!;
      if (s === t) {
        // ... + δ_st Σ_u Z_s Z_u k T(0, τ_u − τ_s) (the G = 0 term cancels by neutrality).
        for (let u = 0; u < 2; u++) {
          const du: V3 = [L.tau[u]![0] - L.tau[s]![0], L.tau[u]![1] - L.tau[s]![1], L.tau[u]![2] - L.tau[s]![2]];
          const T0 = m.coulomb === 'ewald' ? coulombT(L, [0, 0, 0], du, null) : coulombTCut(L, [0, 0, 0], du, m.rc ?? 8, m.alpha);
          for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) blk[x]![y]! += L.charges[s]! * L.charges[u]! * COULOMB_K * T0[x]![y]!;
        }
      }
      // Short range, Na–Cl nearest neighbours: Φ = A r̂r̂ + B(1 − r̂r̂).
      const nn: V3[] = [
        [m.r0, 0, 0], [-m.r0, 0, 0], [0, m.r0, 0], [0, -m.r0, 0], [0, 0, m.r0], [0, 0, -m.r0],
      ];
      for (const r of nn) {
        const phi = (x: number, y: number) => {
          const rr = (r[x]! * r[y]!) / (m.r0 * m.r0);
          return A * rr + B * ((x === y ? 1 : 0) - rr);
        };
        for (let x = 0; x < 3; x++)
          for (let y = 0; y < 3; y++) {
            if (s === t) blk[x]![y]! += phi(x, y);
            else blk[x]![y]! -= phi(x, y) * Math.cos(dot(q, r));
          }
      }
      const w = 1 / Math.sqrt(L.masses[s]! * L.masses[t]!);
      for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) D[3 * s + x]![3 * t + y] = blk[x]![y]! * w;
    }
  // Symmetrise away round-off.
  for (let i = 0; i < 6; i++) for (let j = 0; j < i; j++) D[i]![j] = D[j]![i] = (D[i]![j]! + D[j]![i]!) / 2;
  return D;
}

/** Cyclic Jacobi eigen-decomposition of a real symmetric matrix; eigenvalues ascending, vectors as columns. */
export function jacobiEigen(Ain: number[][]): { values: number[]; vectors: number[][] } {
  const n = Ain.length;
  const A = Ain.map((r) => [...r]);
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p]![q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = A[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k]![p]!;
          const akq = A[k]![q]!;
          A[k]![p] = c * akp - s * akq;
          A[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p]![k]!;
          const aqk = A[q]![k]!;
          A[p]![k] = c * apk - s * aqk;
          A[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k]![p]!;
          const vkq = V[k]![q]!;
          V[k]![p] = c * vkp - s * vkq;
          V[k]![q] = s * vkp + c * vkq;
        }
      }
  }
  const order = A.map((_, i) => i).sort((i, j) => A[i]![i]! - A[j]![j]!);
  return { values: order.map((i) => A[i]![i]!), vectors: V.map((row) => order.map((i) => row[i]!)) };
}

export interface Mode {
  /** Frequency ω/2π in THz; negative means imaginary (unstable). */
  nu: number;
  /** Mass-weighted eigenvector components [Na x,y,z, Cl x,y,z]. */
  e: number[];
  /** Fraction of the optical (Na against Cl) motion along q̂: 1 = longitudinal, 0 = transverse. */
  longitudinal: number;
  optical: boolean;
}

export function modes(m: RockSaltModel, q: V3, qhat?: V3): Mode[] {
  const D = dynamicalMatrix(m, q, qhat);
  const { values, vectors } = jacobiEigen(D);
  const dir = qhat ?? (Math.hypot(...q) > 1e-12 ? q : ([1, 0, 0] as V3));
  const dl = Math.hypot(...dir);
  const u: V3 = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
  return values.map((lam, k) => {
    const e = vectors.map((row) => row[k]!);
    // Real displacements u = e / √M; the optical character compares the two ions' motion.
    const na: V3 = [e[0]! / Math.sqrt(MASS.Na), e[1]! / Math.sqrt(MASS.Na), e[2]! / Math.sqrt(MASS.Na)];
    const cl: V3 = [e[3]! / Math.sqrt(MASS.Cl), e[4]! / Math.sqrt(MASS.Cl), e[5]! / Math.sqrt(MASS.Cl)];
    const rel: V3 = [na[0] - cl[0], na[1] - cl[1], na[2] - cl[2]];
    const Mt = MASS.Na + MASS.Cl;
    const sum: V3 = [(MASS.Na * na[0] + MASS.Cl * cl[0]) / Mt, (MASS.Na * na[1] + MASS.Cl * cl[1]) / Mt, (MASS.Na * na[2] + MASS.Cl * cl[2]) / Mt];
    const optical = dot(rel, rel) > 4 * dot(sum, sum);
    const v = optical ? rel : sum;
    const vv = dot(v, v) || 1;
    return { nu: Math.sign(lam) * Math.sqrt(Math.abs(lam)) * THZ, e, longitudinal: dot(v, u) ** 2 / vv, optical };
  });
}

/** High-symmetry points of the fcc Brillouin zone (a = 2 r0). */
export function zonePoints(r0: number) {
  const k = Math.PI / r0;
  return { X: [k, 0, 0] as V3, L: [k / 2, k / 2, k / 2] as V3, K: [(3 * k) / 4, (3 * k) / 4, 0] as V3 };
}
