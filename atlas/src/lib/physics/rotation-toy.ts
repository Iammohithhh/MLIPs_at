// Physics Lab P4 — what changes under rotation? Two TOY MODELs on a molecule:
//
// (A) Distance-only spring model (invariant by construction):
//       E_A = Σ_bonds ½ k (r_ij − r⁰_ij)²
//     depends only on interatomic distances, so E_A(QR) = E_A(R) for any rotation Q,
//     and its forces F = −∇E co-rotate: F(QR) = Q F(R).
//
// (B) "Raw-coordinate" model: the same springs plus a term that looks at the
//     x-component of every bond vector in the fixed lab frame,
//       E_B = E_A + λ Σ_bonds (d_ij · x̂)²,   d_ij = r_j − r_i,
//     the kind of dependence a network fed raw Cartesian differences can learn.
//     It is translation-invariant but not rotation-invariant.
//
// Units: Å and eV (toy parameters; not a fitted force field).

import type { Vec3 } from '../structure/types';
import { covalentRadius } from '../structure/elements';

export type Mat3x3 = [Vec3, Vec3, Vec3];

export const SPRING_K = 30; // eV/Å²
export const LAMBDA = 0.6; // eV/Å²

/** Rotation by θ (radians) about a unit axis (Rodrigues). Rows act on column vectors. */
export function rotation(axis: Vec3, theta: number): Mat3x3 {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n];
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

export function apply(Q: Mat3x3, v: Vec3): Vec3 {
  return [
    Q[0][0] * v[0] + Q[0][1] * v[1] + Q[0][2] * v[2],
    Q[1][0] * v[0] + Q[1][1] * v[1] + Q[1][2] * v[2],
    Q[2][0] * v[0] + Q[2][1] * v[1] + Q[2][2] * v[2],
  ];
}

export function centroid(ps: Vec3[]): Vec3 {
  const c: Vec3 = [0, 0, 0];
  for (const p of ps) for (let k = 0; k < 3; k++) c[k as 0 | 1 | 2] += p[k as 0 | 1 | 2] / ps.length;
  return c;
}

/** Rotate positions by Q about their centroid. */
export function rotateAbout(ps: Vec3[], Q: Mat3x3): Vec3[] {
  const c = centroid(ps);
  return ps.map((p) => {
    const r = apply(Q, [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
    return [r[0] + c[0], r[1] + c[1], r[2] + c[2]];
  });
}

export interface SpringModel {
  bonds: Array<[number, number]>;
  /** Rest lengths (Å), from the reference geometry. */
  r0: number[];
}

/** Bonds where d < 1.15 (rᵢ + rⱼ) with covalent radii; rest lengths from `ref`. */
export function springModel(symbols: string[], ref: Vec3[]): SpringModel {
  const bonds: Array<[number, number]> = [];
  const r0: number[] = [];
  for (let i = 0; i < ref.length; i++)
    for (let j = i + 1; j < ref.length; j++) {
      const d = dist(ref[i]!, ref[j]!);
      if (d < 1.15 * (covalentRadius(symbols[i]!).r + covalentRadius(symbols[j]!).r)) {
        bonds.push([i, j]);
        r0.push(d);
      }
    }
  return { bonds, r0 };
}

const dist = (a: Vec3, b: Vec3) => Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);

export type ModelId = 'invariant' | 'coordinate';

export function energy(model: ModelId, m: SpringModel, R: Vec3[]): number {
  let e = 0;
  m.bonds.forEach(([i, j], b) => {
    const d: Vec3 = [R[j]![0] - R[i]![0], R[j]![1] - R[i]![1], R[j]![2] - R[i]![2]];
    const r = Math.hypot(d[0], d[1], d[2]);
    e += 0.5 * SPRING_K * (r - m.r0[b]!) ** 2;
    if (model === 'coordinate') e += LAMBDA * d[0] * d[0];
  });
  return e;
}

/** F = −∇E, analytically. */
export function forces(model: ModelId, m: SpringModel, R: Vec3[]): Vec3[] {
  const f: Vec3[] = R.map(() => [0, 0, 0]);
  m.bonds.forEach(([i, j], b) => {
    const d: Vec3 = [R[j]![0] - R[i]![0], R[j]![1] - R[i]![1], R[j]![2] - R[i]![2]];
    const r = Math.hypot(d[0], d[1], d[2]);
    const g = (SPRING_K * (r - m.r0[b]!)) / r; // ∂E/∂d = g·d
    for (let k = 0; k < 3; k++) {
      let dEdd = g * d[k as 0 | 1 | 2];
      if (model === 'coordinate' && k === 0) dEdd += 2 * LAMBDA * d[0];
      f[j]![k as 0 | 1 | 2] -= dEdd;
      f[i]![k as 0 | 1 | 2] += dEdd;
    }
  });
  return f;
}

/** max_i |F(QR)_i − Q F(R)_i| (eV/Å): zero for an equivariant force field. */
export function equivarianceError(model: ModelId, m: SpringModel, R: Vec3[], Q: Mat3x3): number {
  const fRot = forces(model, m, rotateAbout(R, Q));
  const fRef = forces(model, m, R).map((v) => apply(Q, v));
  return Math.max(...fRot.map((v, i) => dist(v, fRef[i]!)));
}

/** Deterministic small distortion (Å) so the molecule is mid-vibration and forces are non-zero. */
export function distort(ps: Vec3[], amplitude = 0.09): Vec3[] {
  return ps.map((p, i) => [
    p[0] + amplitude * Math.sin(1.7 * i + 0.3),
    p[1] + amplitude * Math.cos(2.3 * i + 1.1),
    p[2] + amplitude * Math.sin(3.1 * i + 2.0),
  ]);
}
