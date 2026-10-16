// Thermally activated hopping and what a barrier error does to diffusion (Failure Atlas: ion migration).
// THEORY + TOY MODEL: an ion hops between nearest-neighbour sites of a simple cubic lattice
// (spacing a) at rate ν·exp(−E_m/k_BT) per direction. The tracer diffusion coefficient is then
// D = a² ν exp(−E_m/k_BT), so a barrier error δ multiplies D by exp(δ/k_BT).

import { mulberry32 } from '../chem/rng';

/** Boltzmann constant, eV/K (CODATA 2018). */
export const KB = 8.617333262e-5;

/** Hop rate per direction (1/s). */
export const hopRate = (Em: number, T: number, nu = 1e13) => nu * Math.exp(-Em / (KB * T));

/** Tracer diffusion coefficient (cm²/s) on a simple cubic lattice; a in Å. */
export const diffusivity = (Em: number, T: number, a = 3, nu = 1e13) => a * a * 1e-16 * hopRate(Em, T, nu);

/** Factor by which D changes when the barrier is lowered by δ (eV). */
export const diffusivityRatio = (delta: number, T: number) => Math.exp(delta / (KB * T));

/** Barrier error that changes D by a factor f at temperature T. */
export const deltaForFactor = (f: number, T: number) => KB * T * Math.log(f);

type V3 = [number, number, number];
const DIRS: V3[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/** A lattice random walk of n hops (lattice units), reproducible from the seed. */
export function walk(n: number, seed: number): V3[] {
  const rand = mulberry32(seed);
  const out: V3[] = [[0, 0, 0]];
  let p: V3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const d = DIRS[Math.floor(rand() * 6)]!;
    p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    out.push(p);
  }
  return out;
}

/** Mean squared displacement per hop of many independent walks (lattice units²); → 1 for this lattice. */
export function msdPerHop(walks: number, hops: number, seed: number): number {
  let s = 0;
  for (let w = 0; w < walks; w++) {
    const p = walk(hops, seed + w).at(-1)!;
    s += p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
  }
  return s / walks / hops;
}
