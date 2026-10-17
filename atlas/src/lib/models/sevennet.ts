// SevenNet (Park et al., JCTC 2024; Kim et al., JACS 2024) as small, exact calculations.
//
// SevenNet is NequIP's architecture with a spatial-decomposition algorithm that lets a
// message-passing network run on many GPUs inside LAMMPS. Each rank owns the atoms in its
// subdomain and keeps "ghost" copies of atoms within r_c of it. A strictly local potential needs
// nothing more; a T-layer network sees T·r_c, so the naive fix is a ghost shell T·r_c thick and a
// lot of repeated work. SevenNet keeps the shell at r_c and instead exchanges node features of the
// ghosts after every layer (forward communication) and, on the way back, the energy gradients
// ∂E/∂h of ghosts and their partial forces (reverse communication) (Park 2024, Sec. 3.1, Fig. 3).
//
// Three things on the SevenNet page are computed here:
//  1. Ghost-shell bookkeeping in a real crystal (α-quartz, the paper's benchmark): owned atoms,
//     ghosts at r_c and at T·r_c, node updates and floats communicated per MD step.
//  2. A toy invariant message-passing network in 2D, evaluated serially, with SevenNet's
//     communication pattern, and with the naive T·r_c ghost shell, so the three can be compared
//     to machine precision, and with any of the three communications switched off.
//  3. Multi-fidelity (SevenNet-MF): a one-hot fidelity concatenated to the input of a linear layer
//     is a fidelity-dependent bias. In the readout's energy shift it moves every structure's energy
//     by a constant (forces identical); inside the message-passing layers it changes the forces.
//
// Weights are random and untrained: energies are in arbitrary units, but agreement between serial
// and parallel evaluation, and which communication matters for what, are exact properties.

import { mulberry32 } from '../chem/rng';
import type { Structure } from '../structure/types';

// ---------------------------------------------------------------------------
// 1. Ghost shells in a crystal
// ---------------------------------------------------------------------------

/** SevenNet-0 (11Jul2024) settings from its pre_train.yaml: cutoff, layers and node irreps. */
export const SEVENNET0 = {
  rc: 5.0,
  layers: 5,
  /** 128x0e + 64x1e + 32x2e: 128 + 64·3 + 32·5 floats per atom between layers. */
  featureFloats: 128 + 64 * 3 + 32 * 5,
} as const;

/** The paper's scaling benchmarks: α-quartz, 4,608 atoms per GPU, r_c = 4 Å (Park 2024, Sec. 4). */
export const PAPER_BENCH = { atomsPerGpu: 4608, rc: 4.0 } as const;

type V3 = [number, number, number];

function inv3(m: number[][]): number[][] {
  const [a, b, c] = m as [number[], number[], number[]];
  const det =
    a[0]! * (b[1]! * c[2]! - b[2]! * c[1]!) - a[1]! * (b[0]! * c[2]! - b[2]! * c[0]!) + a[2]! * (b[0]! * c[1]! - b[1]! * c[0]!);
  const cof = (r1: number[], r2: number[]): number[] => [r1[1]! * r2[2]! - r1[2]! * r2[1]!, r1[2]! * r2[0]! - r1[0]! * r2[2]!, r1[0]! * r2[1]! - r1[1]! * r2[0]!];
  // Rows of the inverse transpose are the reciprocal vectors.
  const ra = cof(b, c).map((x) => x / det);
  const rb = cof(c, a).map((x) => x / det);
  const rc = cof(a, b).map((x) => x / det);
  // inverse of the row-lattice matrix M (r = f·M) is columns ra, rb, rc.
  return [0, 1, 2].map((i) => [ra[i]!, rb[i]!, rc[i]!]);
}

/** Number density of a periodic structure, atoms per Å³. */
export function numberDensity(s: Structure): number {
  const m = s.cell!;
  const [a, b, c] = m;
  const vol = Math.abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]));
  return s.symbols.length / vol;
}

/**
 * Visit every atom of the infinite crystal inside the axis-aligned box [lo, hi]. The callback gets
 * the Cartesian position and the basis index.
 */
export function forEachAtomInBox(s: Structure, lo: V3, hi: V3, f: (p: V3, k: number) => void): void {
  const m = s.cell!;
  const minv = inv3(m);
  // Fractional ranges over the box corners, padded by the basis extent.
  const fr = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (const x of [lo[0], hi[0]])
    for (const y of [lo[1], hi[1]])
      for (const z of [lo[2], hi[2]])
        for (let j = 0; j < 3; j++) {
          const fj = x * minv[0]![j]! + y * minv[1]![j]! + z * minv[2]![j]!;
          fr[2 * j] = Math.min(fr[2 * j]!, fj);
          fr[2 * j + 1] = Math.max(fr[2 * j + 1]!, fj);
        }
  const pos = s.positions;
  for (let i = Math.floor(fr[0]!) - 1; i <= Math.ceil(fr[1]!) + 1; i++)
    for (let j = Math.floor(fr[2]!) - 1; j <= Math.ceil(fr[3]!) + 1; j++)
      for (let l = Math.floor(fr[4]!) - 1; l <= Math.ceil(fr[5]!) + 1; l++) {
        const tx = i * m[0][0] + j * m[1][0] + l * m[2][0];
        const ty = i * m[0][1] + j * m[1][1] + l * m[2][1];
        const tz = i * m[0][2] + j * m[1][2] + l * m[2][2];
        for (let k = 0; k < pos.length; k++) {
          const p: V3 = [pos[k]![0] + tx, pos[k]![1] + ty, pos[k]![2] + tz];
          if (p[0] >= lo[0] && p[0] < hi[0] && p[1] >= lo[1] && p[1] < hi[1] && p[2] >= lo[2] && p[2] < hi[2]) f(p, k);
        }
      }
}

/** Distance outside the box [0, L]³ measured as LAMMPS extends a subdomain by its ghost cutoff: per axis, the largest overhang. */
export function overhang(p: V3, L: number): number {
  let d = 0;
  for (const x of p) d = Math.max(d, -x, x - L);
  return d;
}

export interface ShellCounts {
  /** Domain edge, Å. */
  L: number;
  owned: number;
  /** cum[k] = owned + ghosts within k·r_c, k = 0 … T. */
  cum: number[];
}

/**
 * Count the atoms in a cubic subdomain of edge L placed in the infinite crystal at `origin`, and
 * in the boxes extended by k·r_c (k = 1 … T), as LAMMPS sets up ghosts with a communication cutoff.
 */
export function shellCounts(s: Structure, L: number, rc: number, T: number, origin: V3 = [0.1, 0.1, 0.1]): ShellCounts {
  const D = T * rc;
  const cum = new Array<number>(T + 1).fill(0);
  forEachAtomInBox(
    s,
    [origin[0] - D, origin[1] - D, origin[2] - D],
    [origin[0] + L + D, origin[1] + L + D, origin[2] + L + D],
    (p) => {
      const d = overhang([p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]], L);
      // Atoms on a face belong to the box that starts there (half-open, like the enumeration).
      const k = d <= 0 ? 0 : Math.ceil(d / rc - 1e-12);
      for (let j = Math.max(0, k); j <= T; j++) cum[j]!++;
    },
  );
  return { L, owned: cum[0]!, cum };
}

/** Edge of the cube that holds N atoms on average. */
export const domainEdge = (s: Structure, atoms: number) => Math.cbrt(atoms / numberDensity(s));

export interface DecompositionCost {
  owned: number;
  /** Ghosts with SevenNet's r_c shell and with the naive T·r_c shell. */
  ghostsSeven: number;
  ghostsNaive: number;
  /** Node updates per force call: T × owned for SevenNet; Σ_t atoms within (T − t)·r_c for the naive shell. */
  updatesSeven: number;
  updatesNaive: number;
  /** Floats received per rank per force call (positions + species, features, gradients, forces). */
  commSeven: { positions: number; features: number; gradients: number; forces: number; total: number; messages: number };
  commNaive: { positions: number; forces: number; total: number; messages: number };
}

/**
 * Per-rank cost of one energy-and-force evaluation with T layers, from the shell counts.
 * Communication is counted as floats arriving for ghost atoms: 4 per ghost for position and species,
 * F per ghost for each of the T − 1 feature exchanges and gradient exchanges, and 3 per ghost for
 * the partial forces sent back (Park 2024, Fig. 3). A message is one exchange step.
 */
export function decompositionCost(c: ShellCounts, T: number, F: number): DecompositionCost {
  const owned = c.owned;
  const g1 = c.cum[1]! - owned;
  const gT = c.cum[T]! - owned;
  let updatesNaive = 0;
  for (let t = 1; t <= T; t++) updatesNaive += c.cum[T - t]!;
  const positions = 4 * g1;
  const features = (T - 1) * F * g1;
  const gradients = (T - 1) * F * g1;
  const forces = 3 * g1;
  return {
    owned,
    ghostsSeven: g1,
    ghostsNaive: gT,
    updatesSeven: T * owned,
    updatesNaive,
    commSeven: { positions, features, gradients, forces, total: positions + features + gradients + forces, messages: 2 * (T - 1) + 2 },
    commNaive: { positions: 4 * gT, forces: 3 * gT, total: 7 * gT, messages: 2 },
  };
}

// ---------------------------------------------------------------------------
// 2. A toy message-passing network in 2D, serial and in parallel
// ---------------------------------------------------------------------------

export interface System2D {
  /** x, y per atom (toy units: the nearest-neighbour spacing is 1). */
  pos: Float64Array;
  species: Int32Array;
  n: number;
  width: number;
  height: number;
}

/** A jittered triangular lattice of two species in a free (non-periodic) rectangle. */
export function makeSystem(cols = 30, rows = 10, jitter = 0.12, seed = 3): System2D {
  const rand = mulberry32(seed);
  const xs: number[] = [];
  const sp: number[] = [];
  const dy = Math.sqrt(3) / 2;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      xs.push(0.5 + c + (r % 2 ? 0.5 : 0) + jitter * (2 * rand() - 1), 0.5 + r * dy + jitter * (2 * rand() - 1));
      sp.push(rand() < 0.5 ? 0 : 1);
    }
  return { pos: Float64Array.from(xs), species: Int32Array.from(sp), n: sp.length, width: cols + 0.5, height: (rows - 1) * dy + 1 };
}

export const TOY_RC = 1.8;
const NB = 4; // radial basis functions
const P_ENV = 6; // NequIP's polynomial cutoff order

/** NequIP's polynomial envelope u(x) = 1 − (p+1)(p+2)/2 x^p + p(p+2) x^{p+1} − p(p+1)/2 x^{p+2}, and du/dr. */
export function envelope(r: number, rc = TOY_RC): [number, number] {
  if (r >= rc) return [0, 0];
  const p = P_ENV;
  const x = r / rc;
  const u = 1 - ((p + 1) * (p + 2) / 2) * x ** p + p * (p + 2) * x ** (p + 1) - (p * (p + 1) / 2) * x ** (p + 2);
  const du = (-(p * (p + 1) * (p + 2)) / 2) * x ** (p - 1) + p * (p + 1) * (p + 2) * x ** p - ((p * (p + 1) * (p + 2)) / 2) * x ** (p + 1);
  return [u, du / rc];
}

/** Bessel basis sin(kπr/r_c)/r times the envelope, k = 1 … NB, and radial derivatives. */
function basis(r: number, rc: number, out: Float64Array, dout: Float64Array) {
  const [u, du] = envelope(r, rc);
  for (let k = 0; k < NB; k++) {
    const w = ((k + 1) * Math.PI) / rc;
    const b = Math.sin(w * r) / r;
    const db = (w * Math.cos(w * r)) / r - Math.sin(w * r) / (r * r);
    out[k] = b * u;
    dout[k] = db * u + b * du;
  }
}

export interface ToyModel {
  C: number;
  T: number;
  rc: number;
  /** emb[z][c] */
  emb: number[][];
  /** Per layer: filter weights A[c][k], message weights W[c][c'], update weights U[c][c']. */
  A: number[][][];
  W: number[][][];
  U: number[][][];
  q: number[];
  /** Per-species energy shift (fidelity 0). */
  shift: number[];
  /** Fidelity-dependent parts: biases from the one-hot in the embedding and in each update layer, and in the shift. */
  fid: { emb: number[]; upd: number[][]; shift: number[] };
}

export function makeToyModel(T = 3, C = 4, seed = 11, rc = TOY_RC): ToyModel {
  const rand = mulberry32(seed);
  const g = (s: number) => s * (2 * rand() - 1);
  const mat = (r: number, c: number, s: number) => Array.from({ length: r }, () => Array.from({ length: c }, () => g(s)));
  return {
    C,
    T,
    rc,
    emb: mat(2, C, 1),
    A: Array.from({ length: T }, () => mat(C, NB, 0.6)),
    W: Array.from({ length: T }, () => mat(C, C, 0.8)),
    U: Array.from({ length: T }, () => mat(C, C, 0.8)),
    q: Array.from({ length: C }, () => g(1)),
    shift: [-1.3, -2.1],
    fid: { emb: Array.from({ length: C }, () => g(0.15)), upd: Array.from({ length: T }, () => Array.from({ length: C }, () => g(0.15))), shift: [-0.4, 0.25] },
  };
}

/** Where a one-hot fidelity enters: nowhere, only the energy shift, or every linear layer (SevenNet-MF). */
export type FidelityMode = 'none' | 'shift' | 'all';

interface Edge {
  v: number;
  w: number;
  r: number;
  /** unit vector from v to w */
  ux: number;
  uy: number;
}

/** All directed edges v ← w with r < r_c, grouped by receiver. */
export function neighbourList(sys: System2D, rc: number): Edge[][] {
  const out: Edge[][] = Array.from({ length: sys.n }, () => []);
  const p = sys.pos;
  for (let v = 0; v < sys.n; v++)
    for (let w = 0; w < sys.n; w++) {
      if (v === w) continue;
      const dx = p[2 * w]! - p[2 * v]!;
      const dy = p[2 * w + 1]! - p[2 * v + 1]!;
      const r = Math.hypot(dx, dy);
      if (r < rc) out[v]!.push({ v, w, r, ux: dx / r, uy: dy / r });
    }
  return out;
}

/** One rank's working memory (global-sized arrays; only local entries are meaningful). */
interface RankState {
  /** h[t] for t = 0 … T (h[0] is the embedding = paper's h^(1)). */
  h: Float64Array[];
  /** pre-activations a[t] of layer t (t = 0 … T−1). */
  a: Float64Array[];
  /** ∂E/∂positions accumulated on this rank (x, y per atom). */
  grad: Float64Array;
}

const newRank = (n: number, C: number, T: number): RankState => ({
  h: Array.from({ length: T + 1 }, () => new Float64Array(n * C)),
  a: Array.from({ length: T }, () => new Float64Array(n * C)),
  grad: new Float64Array(2 * n),
});

const fidBit = (fidelity: number, mode: FidelityMode, where: 'shift' | 'layer') => (mode === 'all' || (mode === 'shift' && where === 'shift') ? fidelity : 0);

function embed(m: ToyModel, sys: System2D, atoms: Iterable<number>, h0: Float64Array, fidelity: number, mode: FidelityMode) {
  const f = fidBit(fidelity, mode, 'layer');
  for (const v of atoms) for (let c = 0; c < m.C; c++) h0[v * m.C + c] = m.emb[sys.species[v]!]![c]! + f * m.fid.emb[c]!;
}

/**
 * Layer t (0-based) for the receivers given: m_v = Σ_w f_t(r) ⊙ (W_t h_w), a_v = U_t m_v (+ fidelity bias),
 * h^{t+1}_v = h^t_v + tanh(a_v). Neighbours outside `local` are skipped (they are not on this rank).
 */
function forwardLayer(m: ToyModel, t: number, receivers: number[], nbrs: Edge[][], local: Uint8Array, st: RankState, fidelity: number, mode: FidelityMode) {
  const C = m.C;
  const hin = st.h[t]!;
  const hout = st.h[t + 1]!;
  const a = st.a[t]!;
  const b = new Float64Array(NB);
  const db = new Float64Array(NB);
  const msg = new Float64Array(C);
  const f = fidBit(fidelity, mode, 'layer');
  for (const v of receivers) {
    msg.fill(0);
    for (const e of nbrs[v]!) {
      if (!local[e.w]) continue;
      basis(e.r, m.rc, b, db);
      for (let c = 0; c < C; c++) {
        let filt = 0;
        for (let k = 0; k < NB; k++) filt += m.A[t]![c]![k]! * b[k]!;
        let wh = 0;
        for (let c2 = 0; c2 < C; c2++) wh += m.W[t]![c]![c2]! * hin[e.w * C + c2]!;
        msg[c]! += filt * wh;
      }
    }
    for (let c = 0; c < C; c++) {
      let s = f * m.fid.upd[t]![c]!;
      for (let c2 = 0; c2 < C; c2++) s += m.U[t]![c]![c2]! * msg[c2]!;
      a[v * C + c] = s;
      hout[v * C + c] = hin[v * C + c]! + Math.tanh(s);
    }
  }
}

/**
 * Backward through layer t: given g = ∂E/∂h^{t+1} (local entries), accumulate ∂E/∂h^t into gNext
 * and ∂E/∂positions into st.grad for every edge of the receivers.
 */
function backwardLayer(m: ToyModel, t: number, receivers: number[], nbrs: Edge[][], local: Uint8Array, st: RankState, g: Float64Array, gNext: Float64Array) {
  const C = m.C;
  const hin = st.h[t]!;
  const a = st.a[t]!;
  const b = new Float64Array(NB);
  const db = new Float64Array(NB);
  const mu = new Float64Array(C);
  for (const v of receivers) {
    // Residual path.
    for (let c = 0; c < C; c++) gNext[v * C + c]! += g[v * C + c]!;
    // ∂E/∂m_v = U_tᵀ (g ⊙ (1 − tanh² a)).
    mu.fill(0);
    for (let c = 0; c < C; c++) {
      const th = Math.tanh(a[v * C + c]!);
      const d = g[v * C + c]! * (1 - th * th);
      for (let c2 = 0; c2 < C; c2++) mu[c2]! += m.U[t]![c]![c2]! * d;
    }
    for (const e of nbrs[v]!) {
      if (!local[e.w]) continue;
      basis(e.r, m.rc, b, db);
      let dEdr = 0;
      for (let c = 0; c < C; c++) {
        let filt = 0;
        let dfilt = 0;
        for (let k = 0; k < NB; k++) {
          filt += m.A[t]![c]![k]! * b[k]!;
          dfilt += m.A[t]![c]![k]! * db[k]!;
        }
        let wh = 0;
        for (let c2 = 0; c2 < C; c2++) wh += m.W[t]![c]![c2]! * hin[e.w * C + c2]!;
        dEdr += mu[c]! * dfilt * wh;
        // ∂E/∂h_w += W_tᵀ (f ⊙ μ).
        const fm = filt * mu[c]!;
        for (let c2 = 0; c2 < C; c2++) gNext[e.w * C + c2]! += m.W[t]![c]![c2]! * fm;
      }
      // r = |r_w − r_v|: ∂r/∂r_w = û, ∂r/∂r_v = −û.
      st.grad[2 * e.w]! += dEdr * e.ux;
      st.grad[2 * e.w + 1]! += dEdr * e.uy;
      st.grad[2 * v]! -= dEdr * e.ux;
      st.grad[2 * v + 1]! -= dEdr * e.uy;
    }
  }
}

function atomEnergy(m: ToyModel, sys: System2D, v: number, h: Float64Array, fidelity: number, mode: FidelityMode): number {
  let e = m.shift[sys.species[v]!]! + fidBit(fidelity, mode, 'shift') * m.fid.shift[sys.species[v]!]!;
  for (let c = 0; c < m.C; c++) e += m.q[c]! * h[v * m.C + c]!;
  return e;
}

export interface Evaluation {
  energy: number;
  /** Per-atom energies (owner's value). */
  atomic: Float64Array;
  /** Forces, x and y per atom. */
  forces: Float64Array;
}

/** The reference: one process, every atom local. */
export function evaluateSerial(m: ToyModel, sys: System2D, fidelity = 0, mode: FidelityMode = 'none', nbrs = neighbourList(sys, m.rc)): Evaluation {
  const all = Array.from({ length: sys.n }, (_, i) => i);
  const local = new Uint8Array(sys.n).fill(1);
  const st = newRank(sys.n, m.C, m.T);
  embed(m, sys, all, st.h[0]!, fidelity, mode);
  for (let t = 0; t < m.T; t++) forwardLayer(m, t, all, nbrs, local, st, fidelity, mode);
  const atomic = new Float64Array(sys.n);
  let energy = 0;
  for (const v of all) energy += atomic[v] = atomEnergy(m, sys, v, st.h[m.T]!, fidelity, mode);
  let g = new Float64Array(sys.n * m.C);
  for (const v of all) for (let c = 0; c < m.C; c++) g[v * m.C + c] = m.q[c]!;
  for (let t = m.T - 1; t >= 0; t--) {
    const gNext = new Float64Array(sys.n * m.C);
    backwardLayer(m, t, all, nbrs, local, st, g, gNext);
    g = gNext;
  }
  return { energy, atomic, forces: st.grad.map((x) => -x) };
}

export interface Domain {
  lo: number;
  hi: number;
  owned: number[];
  /** Ghost atoms within the shell (by x distance from the strip). */
  ghosts: number[];
}

/** Split the system into P vertical strips and attach ghosts within `shell` of each strip in x. */
export function decompose(sys: System2D, P: number, shell: number): Domain[] {
  const w = sys.width / P;
  return Array.from({ length: P }, (_, d) => {
    const lo = d * w;
    const hi = d === P - 1 ? Infinity : (d + 1) * w;
    const owned: number[] = [];
    const ghosts: number[] = [];
    for (let i = 0; i < sys.n; i++) {
      const x = sys.pos[2 * i]!;
      if (x >= lo && x < hi) owned.push(i);
      else if (x >= lo - shell && x < hi + shell) ghosts.push(i);
    }
    return { lo, hi: d === P - 1 ? sys.width : hi, owned, ghosts };
  });
}

export interface CommSwitches {
  /** Node features of ghosts after each layer but the last (Fig. 3d). */
  features: boolean;
  /** ∂E/∂h of ghosts back to their owners before each layer but the first (Fig. 3e). */
  gradients: boolean;
  /** Partial forces on ghosts back to their owners (Fig. 3f). */
  forces: boolean;
}

export interface ParallelEvaluation extends Evaluation {
  domains: Domain[];
  /** Floats received, summed over ranks, by kind. */
  comm: { positions: number; features: number; gradients: number; forces: number };
  /** Exchange steps per force call. */
  messages: number;
  /** Node updates (layer evaluations of one atom), summed over ranks. */
  updates: number;
}

/**
 * SevenNet's scheme: ghosts within r_c; forward exchange of ghost features between layers, reverse
 * exchange of ghost gradients and of ghost partial forces. With every switch on this reproduces the
 * serial result to rounding; switching one off shows what it carries.
 */
export function evaluateSevenNet(
  m: ToyModel,
  sys: System2D,
  P: number,
  sw: CommSwitches = { features: true, gradients: true, forces: true },
  fidelity = 0,
  mode: FidelityMode = 'none',
  nbrs = neighbourList(sys, m.rc),
): ParallelEvaluation {
  const C = m.C;
  const domains = decompose(sys, P, m.rc);
  const owner = new Int32Array(sys.n);
  domains.forEach((d, r) => d.owned.forEach((i) => (owner[i] = r)));
  const ranks = domains.map(() => newRank(sys.n, C, m.T));
  const locals = domains.map((d) => {
    const l = new Uint8Array(sys.n);
    for (const i of d.owned) l[i] = 1;
    for (const i of d.ghosts) l[i] = 1;
    return l;
  });
  const comm = { positions: 0, features: 0, gradients: 0, forces: 0 };
  let messages = 1;
  // Positions and species of ghosts arrive first (Fig. 3c); h^(1) of a ghost is its embedding, no exchange needed.
  domains.forEach((d, r) => {
    comm.positions += 3 * d.ghosts.length + d.ghosts.length;
    embed(m, sys, [...d.owned, ...d.ghosts], ranks[r]!.h[0]!, fidelity, mode);
  });
  let updates = 0;
  for (let t = 0; t < m.T; t++) {
    domains.forEach((d, r) => {
      forwardLayer(m, t, d.owned, nbrs, locals[r]!, ranks[r]!, fidelity, mode);
      updates += d.owned.length;
    });
    if (t < m.T - 1) {
      messages++;
      domains.forEach((d, r) => {
        const st = ranks[r]!;
        for (const gi of d.ghosts) {
          for (let c = 0; c < C; c++)
            // Without the exchange the ghost keeps the stale feature from the layer before.
            st.h[t + 1]![gi * C + c] = sw.features ? ranks[owner[gi]!]!.h[t + 1]![gi * C + c]! : st.h[t]![gi * C + c]!;
          if (sw.features) comm.features += C;
        }
      });
    }
  }
  const atomic = new Float64Array(sys.n);
  let energy = 0;
  domains.forEach((d, r) => {
    for (const v of d.owned) energy += atomic[v] = atomEnergy(m, sys, v, ranks[r]!.h[m.T]!, fidelity, mode);
  });
  // Reverse path: ∂E/∂h^(T+1) = q on owned atoms.
  let gs = domains.map((d) => {
    const g = new Float64Array(sys.n * C);
    for (const v of d.owned) for (let c = 0; c < C; c++) g[v * C + c] = m.q[c]!;
    return g;
  });
  for (let t = m.T - 1; t >= 0; t--) {
    const next = domains.map(() => new Float64Array(sys.n * C));
    domains.forEach((d, r) => backwardLayer(m, t, d.owned, nbrs, locals[r]!, ranks[r]!, gs[r]!, next[r]!));
    if (t > 0) {
      // Ghost partials of ∂E/∂h^t go home and are summed by their owners (Eq. 8).
      messages++;
      domains.forEach((d, r) => {
        for (const gi of d.ghosts) {
          if (sw.gradients) {
            for (let c = 0; c < C; c++) next[owner[gi]!]![gi * C + c]! += next[r]![gi * C + c]!;
            comm.gradients += C;
          }
          for (let c = 0; c < C; c++) next[r]![gi * C + c] = 0;
        }
      });
    }
    gs = next;
  }
  // Partial forces on ghosts go home (the F_wv terms of Eq. 6).
  messages++;
  const grad = new Float64Array(2 * sys.n);
  domains.forEach((d, r) => {
    const st = ranks[r]!;
    for (const v of d.owned) {
      grad[2 * v]! += st.grad[2 * v]!;
      grad[2 * v + 1]! += st.grad[2 * v + 1]!;
    }
    for (const gi of d.ghosts) {
      if (!sw.forces) continue;
      grad[2 * gi]! += st.grad[2 * gi]!;
      grad[2 * gi + 1]! += st.grad[2 * gi + 1]!;
      comm.forces += 2;
    }
  });
  return { energy, atomic, forces: grad.map((x) => -x), domains, comm, messages, updates };
}

/**
 * The naive scheme: ghosts within T·r_c and no exchange between layers. Layer t is evaluated for every
 * local atom within (T − t)·r_c of the strip, which is redundant work shared with neighbouring ranks.
 */
export function evaluateNaive(m: ToyModel, sys: System2D, P: number, fidelity = 0, mode: FidelityMode = 'none', nbrs = neighbourList(sys, m.rc)): ParallelEvaluation {
  const C = m.C;
  const domains = decompose(sys, P, m.T * m.rc);
  const comm = { positions: 0, features: 0, gradients: 0, forces: 0 };
  const grad = new Float64Array(2 * sys.n);
  const atomic = new Float64Array(sys.n);
  let energy = 0;
  let updates = 0;
  domains.forEach((d) => {
    const local = new Uint8Array(sys.n);
    const all = [...d.owned, ...d.ghosts];
    for (const i of all) local[i] = 1;
    comm.positions += 4 * d.ghosts.length;
    const dist = (i: number) => {
      const x = sys.pos[2 * i]!;
      return Math.max(0, d.lo - x, x - d.hi);
    };
    const st = newRank(sys.n, C, m.T);
    embed(m, sys, all, st.h[0]!, fidelity, mode);
    const own = new Set(d.owned);
    const recv = Array.from({ length: m.T }, (_, t) => all.filter((i) => own.has(i) || dist(i) < (m.T - 1 - t) * m.rc));
    for (let t = 0; t < m.T; t++) {
      forwardLayer(m, t, recv[t]!, nbrs, local, st, fidelity, mode);
      updates += recv[t]!.length;
    }
    for (const v of d.owned) energy += atomic[v] = atomEnergy(m, sys, v, st.h[m.T]!, fidelity, mode);
    let g = new Float64Array(sys.n * C);
    for (const v of d.owned) for (let c = 0; c < C; c++) g[v * C + c] = m.q[c]!;
    for (let t = m.T - 1; t >= 0; t--) {
      const next = new Float64Array(sys.n * C);
      backwardLayer(m, t, recv[t]!, nbrs, local, st, g, next);
      g = next;
    }
    for (const i of all) {
      grad[2 * i]! += st.grad[2 * i]!;
      grad[2 * i + 1]! += st.grad[2 * i + 1]!;
    }
    comm.forces += 2 * d.ghosts.length;
  });
  return { energy, atomic, forces: grad.map((x) => -x), domains, comm, messages: 2, updates };
}

/** Largest force-component difference and its atom. */
export function maxForceError(a: Float64Array, b: Float64Array): { max: number; atom: number } {
  let max = 0;
  let atom = -1;
  for (let i = 0; i < a.length / 2; i++) {
    const d = Math.hypot(a[2 * i]! - b[2 * i]!, a[2 * i + 1]! - b[2 * i + 1]!);
    if (d > max) {
      max = d;
      atom = i;
    }
  }
  return { max, atom };
}

// ---------------------------------------------------------------------------
// 3. Multi-fidelity: a dimer scan
// ---------------------------------------------------------------------------

/** Total energy E(r) of a two-atom system for both fidelities. */
export function dimerCurves(m: ToyModel, species: [number, number], rs: number[], mode: FidelityMode): { lo: number[]; hi: number[] } {
  const curve = (fid: number) => {
    const at = (r: number) => {
      const sys: System2D = { pos: Float64Array.from([0, 0, r, 0]), species: Int32Array.from(species), n: 2, width: r + 1, height: 1 };
      return evaluateSerial(m, sys, fid, mode).energy;
    };
    return rs.map(at);
  };
  return { lo: curve(0), hi: curve(1) };
}
