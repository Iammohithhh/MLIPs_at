// Lennard-Jones pair potential in reduced units (TOY MODEL):
//   V(r) = 4ε[(σ/r)¹² − (σ/r)⁶],  F_i = −∂E/∂r_i = Σ_j (24ε/r²)[2(σ/r)¹² − (σ/r)⁶] (r_i − r_j)
// Open boundaries (a finite cluster), no cutoff.
import type { Vec3 } from '../structure/types';

export interface LJParams {
  epsilon: number;
  sigma: number;
}

export function ljEnergy(positions: Vec3[], { epsilon, sigma }: LJParams): number {
  let e = 0;
  for (let i = 0; i < positions.length; i++)
    for (let j = i + 1; j < positions.length; j++) {
      const a = positions[i]!;
      const b = positions[j]!;
      const r2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
      const s6 = (sigma * sigma / r2) ** 3;
      e += 4 * epsilon * (s6 * s6 - s6);
    }
  return e;
}

export function ljForces(positions: Vec3[], { epsilon, sigma }: LJParams): Vec3[] {
  const f: Vec3[] = positions.map(() => [0, 0, 0]);
  for (let i = 0; i < positions.length; i++)
    for (let j = i + 1; j < positions.length; j++) {
      const a = positions[i]!;
      const b = positions[j]!;
      const d: Vec3 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
      const r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
      const s6 = (sigma * sigma / r2) ** 3;
      const k = (24 * epsilon / r2) * (2 * s6 * s6 - s6);
      for (let c = 0; c < 3; c++) {
        f[i]![c]! += k * d[c]!;
        f[j]![c]! -= k * d[c]!;
      }
    }
  return f;
}
