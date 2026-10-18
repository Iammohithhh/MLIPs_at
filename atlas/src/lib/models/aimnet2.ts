// AIMNet2 (Anstine, Zubatyuk & Isayev, Chem. Sci. 2025): the first-pass environment features, neural
// charge equilibration (NQE) and the energy terms, plus typed access to the CPU precompute of the
// released ωB97M-D3 checkpoint (pipeline/atlas_pipeline/aimnet2.py → src/data/models/aimnet2.json).
//
// The environment of atom i in pass 0 is built from Gaussian shells g_s(r) = exp(−η (r − r_s)²) f_c(r)
// with a cosine cutoff f_c at 5 Å, an l = 0 part Σ_j a_j g_s(r_ij) and an l = 1 part Σ_j a_j g_s(r_ij) r̂_ij
// whose shells are mixed by learned weights and then squared (aimnet/modules/aev.py, AEVSV and ConvSV).
// a_j is the embedding of neighbour j's element in pass 0. So for each pair of neighbour species
// (Z, Z′) every first-pass invariant of atom i is a linear function of
//   S_Z[s]       = Σ_{j∈Z} g_s(r_ij)
//   P_ZZ′[s, t]  = (Σ_{j∈Z} g_s(r_ij) r̂_ij) · (Σ_{k∈Z′} g_t(r_ik) r̂_ik),
// which is what this module computes.

import raw from '~/data/models/aimnet2.json';

export type Vec3 = [number, number, number];

export interface PassFrag {
  q_tilde: [number, number];
  f: [number, number];
  q: [number, number];
}
export interface DimerRow {
  d: number;
  E: number;
  U_local: number;
  U_coul: number;
  U_disp: number;
  fragQ: [number, number];
  passes: PassFrag[];
  q: number[];
}
export interface Dimer {
  id: string;
  label: string;
  Q: number;
  numbers: number[];
  fragment: number[];
  A: Vec3[];
  B: Vec3[];
  isolated: Record<string, number>;
  rows: DimerRow[];
}
export interface MethaneRow {
  theta: number;
  E: number;
  qC: number;
  conv: Array<{ pass: number; C: number; H: number }>;
}
export interface Aimnet2Data {
  status: string;
  model: {
    checkpoint: string;
    file: string;
    member: number;
    url: string;
    sha256: string;
    licence: string;
    parameters: number;
    hyperparameters: {
      rc: number;
      eta: number;
      shifts: number[];
      nfeature: number;
      ncomb_v: number;
      passes: number;
      aim_size: number;
      coulomb_sr_rc: number;
      coulomb_sr_envelope: string;
      d3: { s6: number; s8: number; a1: number; a2: number };
      species: number[];
    };
  };
  provenance: { aimnet: string; torch: string; ase: string; kE: number; methaneRefTheta: number; dtype: string };
  benzeneEQ: Array<{ Q: number; E: number; qC: number; qH: number }>;
  dimers: Dimer[];
  methane: MethaneRow[];
}

export const DATA = raw as unknown as Aimnet2Data;
export const HP = DATA.model.hyperparameters;
export const K_E = DATA.provenance.kE;
export const TD_THETA = (Math.acos(1 / Math.sqrt(3)) * 180) / Math.PI;
export const CH_BOND = 1.087;

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

/** CH4 on the D2d path: H1,2 = r(±sin θ, 0, cos θ), H3,4 = r(0, ±sin θ, −cos θ). Td at θ = 54.74°, D4h at 90°. */
export function methaneD2d(thetaDeg: number, r = CH_BOND): { numbers: number[]; positions: Vec3[] } {
  const t = (thetaDeg * Math.PI) / 180;
  const s = r * Math.sin(t);
  const c = r * Math.cos(t);
  return {
    numbers: [6, 1, 1, 1, 1],
    positions: [
      [0, 0, 0],
      [s, 0, c],
      [-s, 0, c],
      [0, s, -c],
      [0, -s, -c],
    ],
  };
}

/** aimnet ops.cosine_cutoff. */
export const cosineCutoff = (r: number, rc = HP.rc) => (r >= rc ? 0 : 0.5 * (Math.cos((Math.PI * r) / rc) + 1));

/** The Gaussian shells of one distance (AEVSV._calc_aev, scalar part). */
export function shells(r: number, shifts = HP.shifts, eta = HP.eta, rc = HP.rc): number[] {
  const fc = cosineCutoff(r, rc);
  return shifts.map((s) => Math.exp(-eta * (r - s) ** 2) * fc);
}

export interface Invariants {
  /** species → S[s] */
  S: Map<number, number[]>;
  /** "Z,Z′" → P[s][t] */
  P: Map<string, number[][]>;
  /** species → Σ_j g_s(r_ij) r̂_ij, per shell */
  V: Map<number, Vec3[]>;
}

/** First-pass invariants of atom `i` (see the header). Neighbours are all other atoms within r_c. */
export function firstPassInvariants(numbers: number[], positions: Vec3[], i: number): Invariants {
  const ns = HP.shifts.length;
  const S = new Map<number, number[]>();
  const V = new Map<number, Vec3[]>();
  positions.forEach((p, j) => {
    if (j === i) return;
    const d = sub(p, positions[i]!);
    const r = norm(d);
    if (r >= HP.rc) return;
    const u: Vec3 = [d[0] / r, d[1] / r, d[2] / r];
    const g = shells(r);
    const z = numbers[j]!;
    if (!S.has(z)) {
      S.set(z, new Array<number>(ns).fill(0));
      V.set(
        z,
        Array.from({ length: ns }, () => [0, 0, 0] as Vec3),
      );
    }
    const s = S.get(z)!;
    const v = V.get(z)!;
    for (let k = 0; k < ns; k++) {
      s[k] = s[k]! + g[k]!;
      v[k]![0] += g[k]! * u[0];
      v[k]![1] += g[k]! * u[1];
      v[k]![2] += g[k]! * u[2];
    }
  });
  const P = new Map<string, number[][]>();
  for (const [za, va] of V)
    for (const [zb, vb] of V) {
      P.set(
        `${za},${zb}`,
        va.map((a) => vb.map((b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2])),
      );
    }
  return { S, P, V };
}

/** Largest absolute difference between two invariant sets (species present in either). */
export function invariantDistance(a: Invariants, b: Invariants): number {
  let m = 0;
  const keysS = new Set([...a.S.keys(), ...b.S.keys()]);
  for (const k of keysS) {
    const x = a.S.get(k) ?? [];
    const y = b.S.get(k) ?? [];
    for (let s = 0; s < Math.max(x.length, y.length); s++) m = Math.max(m, Math.abs((x[s] ?? 0) - (y[s] ?? 0)));
  }
  const keysP = new Set([...a.P.keys(), ...b.P.keys()]);
  for (const k of keysP) {
    const x = a.P.get(k) ?? [];
    const y = b.P.get(k) ?? [];
    for (let s = 0; s < Math.max(x.length, y.length); s++)
      for (let t = 0; t < Math.max(x[s]?.length ?? 0, y[s]?.length ?? 0); t++) m = Math.max(m, Math.abs((x[s]?.[t] ?? 0) - (y[s]?.[t] ?? 0)));
  }
  return m;
}

/** Largest |P| entry: zero means every l = 1 feature vanishes, whatever the learned shell mixing. */
export function maxVector(inv: Invariants): number {
  let m = 0;
  for (const p of inv.P.values()) for (const row of p) for (const v of row) m = Math.max(m, Math.abs(v));
  return m;
}

/** Sorted distances from atom i to the others (what the l = 0 part sees). */
export function distancesFrom(positions: Vec3[], i: number): number[] {
  return positions
    .filter((_, j) => j !== i)
    .map((p) => norm(sub(p, positions[i]!)))
    .sort((a, b) => a - b);
}

/** Neural charge equilibration: q = q̃ + f / (Σf + ε) · (Q − Σq̃)  (aimnet ops.nse; Zubatyuk et al. 2021). */
export function nqe(qTilde: number[], f: number[], Q: number, eps = 1e-6): number[] {
  const F = f.reduce((a, b) => a + b, 0) + eps;
  const dQ = Q - qTilde.reduce((a, b) => a + b, 0);
  return qTilde.map((q, i) => q + (f[i]! / F) * dQ);
}

/** aimnet ops.exp_cutoff: exp(−1/(1 − (r/r_c)²)) / e⁻¹; 1 at r = 0, 0 at r ≥ r_c. */
export function expCutoff(r: number, rc = HP.coulomb_sr_rc): number {
  const x = Math.min(Math.max(r / rc, 0), 1 - 1e-6);
  return Math.exp(-1 / (1 - x * x)) / Math.exp(-1);
}

/** The Coulomb energy AIMNet2 adds: k Σ_{i<j} q_i q_j (1 − f_c(r_ij)) / r_ij (eV, Å, e). */
export function coulombLR(q: number[], positions: Vec3[]): number {
  let e = 0;
  for (let i = 0; i < q.length; i++)
    for (let j = i + 1; j < q.length; j++) {
      const r = norm(sub(positions[i]!, positions[j]!));
      e += (q[i]! * q[j]! * (1 - expCutoff(r))) / r;
    }
  return K_E * e;
}

/** Positions of a dimer at separation d (fragment B shifted along z), as in the pipeline. */
export function dimerPositions(dm: Dimer, d: number): Vec3[] {
  return [...dm.A, ...dm.B.map((p) => [p[0], p[1], p[2] + d] as Vec3)];
}

/** The two integer-charge dissociation limits of a Q = +1 dimer in the model's own energies. */
export function integerLimits(dm: Dimer): { plusA: number; plusB: number; half?: number } {
  const iso = dm.isolated;
  if (dm.Q === 0) return { plusA: iso.A0! + iso.B0!, plusB: iso.A0! + iso.B0! };
  return { plusA: iso.A1! + iso.B0!, plusB: iso.A0! + iso.B1!, half: iso['A0.5']! + iso['B0.5']! };
}

/** E(Q) on the straight line between the integer charges either side (exact theory: Perdew et al. 1982). */
export function chord(eq: Aimnet2Data['benzeneEQ'], Q: number): number {
  const lo = Math.floor(Q + 1e-9);
  const hi = Math.ceil(Q - 1e-9);
  const at = (x: number) => eq.find((r) => Math.abs(r.Q - x) < 1e-6)!.E;
  if (lo === hi) return at(lo);
  return at(lo) + (Q - lo) * (at(hi) - at(lo));
}

/** Fragment charges of two separated fragments from their summed q̃ and f (NQE on fragment sums). */
export function fragmentSplit(qtA: number, qtB: number, fA: number, fB: number, Q: number, eps = 1e-6): [number, number] {
  const dQ = Q - qtA - qtB;
  const F = fA + fB + eps;
  return [qtA + (fA / F) * dQ, qtB + (fB / F) * dQ];
}
