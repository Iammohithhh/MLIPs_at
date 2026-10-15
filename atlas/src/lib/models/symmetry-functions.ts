// Behler–Parrinello atom-centred symmetry functions (as defined in Behler & Parrinello,
// PRL 98, 146401 (2007) and Behler, JCP 134, 074106 (2011); forms as implemented in n2p2):
//
//   G²_i = Σ_{j≠i} exp(−η (r_ij − r_s)²) f_c(r_ij)
//   G³_i = 2^{1−ζ} Σ_{j<k} (1 + λ cos θ_ijk)^ζ exp(−η (r_ij² + r_ik² + r_jk²)) f_c(r_ij) f_c(r_ik) f_c(r_jk)
//
// with the cosine cutoff f_c(r) = ½[cos(πr/r_c) + 1] for r < r_c. Inputs are neighbour
// displacement vectors from atom i.

import { cosineCutoff } from '../cutoffs';
import type { Vec3 } from '../structure/types';

export interface G2Params {
  kind: 'G2';
  eta: number;
  rs: number;
}
export interface G3Params {
  kind: 'G3';
  eta: number;
  zeta: number;
  lambda: 1 | -1;
}
export type SFParams = G2Params | G3Params;

const len = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
const fc = (r: number, rc: number) => cosineCutoff(r, rc).f;

export function g2(env: Vec3[], p: G2Params, rc: number): number {
  return g2Contributions(env, p, rc).reduce((a, b) => a + b, 0);
}

/** Per-neighbour contributions to a G² function (same order as `env`). */
export function g2Contributions(env: Vec3[], p: G2Params, rc: number): number[] {
  return env.map((d) => {
    const r = len(d);
    return r < rc ? Math.exp(-p.eta * (r - p.rs) ** 2) * fc(r, rc) : 0;
  });
}

/** G³ value and each neighbour's share (half of every triplet term it takes part in, as j or k). */
export function g3WithShares(env: Vec3[], p: G3Params, rc: number): { value: number; shares: number[] } {
  const shares = new Array<number>(env.length).fill(0);
  let value = 0;
  const pref = 2 ** (1 - p.zeta);
  for (let j = 0; j < env.length; j++) {
    const rij = len(env[j]!);
    if (rij >= rc) continue;
    for (let k = j + 1; k < env.length; k++) {
      const rik = len(env[k]!);
      if (rik >= rc) continue;
      const a = env[j]!;
      const b = env[k]!;
      const djk: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const rjk = len(djk);
      if (rjk >= rc) continue;
      const cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (rij * rik);
      const base = 1 + p.lambda * cos;
      const t = pref * (base > 0 ? base ** p.zeta : 0) * Math.exp(-p.eta * (rij * rij + rik * rik + rjk * rjk)) * fc(rij, rc) * fc(rik, rc) * fc(rjk, rc);
      value += t;
      shares[j]! += t / 2;
      shares[k]! += t / 2;
    }
  }
  return { value, shares };
}

export function evaluate(env: Vec3[], p: SFParams, rc: number): number {
  return p.kind === 'G2' ? g2(env, p, rc) : g3WithShares(env, p, rc).value;
}

/** A small, typical set: 6 radial (r_s = 0, widening) + 6 angular (ζ ∈ {1, 2, 4}, λ = ±1). */
export const DEFAULT_SET: SFParams[] = [
  ...[0.003, 0.01, 0.03, 0.07, 0.15, 0.4].map((eta) => ({ kind: 'G2' as const, eta, rs: 0 })),
  ...[1, 2, 4].flatMap((zeta) => ([1, -1] as const).map((lambda) => ({ kind: 'G3' as const, eta: 0.005, zeta, lambda }))),
];

export function label(p: SFParams): string {
  return p.kind === 'G2' ? `G² η=${p.eta}` : `G³ ζ=${p.zeta} λ=${p.lambda > 0 ? '+1' : '−1'}`;
}
