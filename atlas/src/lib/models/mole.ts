// UMA's Mixture of Linear Experts (Wood et al. 2025, Sec. 2.3) as a small, exact toy.
//
// Each MoLE layer holds E expert matrices W_k. For a system with global features g, a routing MLP
// gives coefficients α(g) and the layer computes y = Σ_k α_k (W_k x) = (Σ_k α_k W_k) x (Eqs. 1–2), so
// the experts can be merged into one matrix W* once per system and inference costs as much as one
// dense layer. The routing input follows fairchem (models/uma/escn_moe.py, nn/mole_utils.py at
// commit d4e10d0):
//   composition = mean over atoms of a learned per-element embedding (so it depends only on the
//                 reduced composition),
//   csd         = SiLU(Linear([charge, spin, task] embeddings)),
//   α           = softmax(SiLU(Linear(SiLU(Linear(SiLU(Linear([composition, csd]))))))) + 0.005.
// Positions never enter α, so forces −∂E/∂r flow only through the merged weights.
//
// fairchem's code comment records that UMA-S-1.2 included a zero-initialised value in the mean,
// i.e. divided the summed embeddings by N + 1: α then depends on the number of atoms, not only on
// the reduced composition, and the energy per atom of a perfect supercell changes with its size.
// The `includeSelf` option reproduces that.
//
// Weights are random and untrained. The energy is a stand-in: per-atom readout of W* applied to a
// fixed per-element environment feature, so a perfect crystal has E/N independent of N whenever α is.

import { mulberry32 } from '../chem/rng';

export const UMA_TASKS = ['omat', 'omol', 'oc20', 'odac', 'omc'] as const;
export type UmaTask = (typeof UMA_TASKS)[number];
/** Experts per MoLE layer in UMA-S and UMA-M (UMA paper, App. Table 6). */
export const UMA_EXPERTS = 32;
/** Constant added after the softmax in fairchem's `_softmax`. */
export const ALPHA_FLOOR = 0.005;

export type Composition = Record<string, number>;

const silu = (x: number) => x / (1 + Math.exp(-x));

interface Lin {
  w: Float64Array;
  b: Float64Array;
  nIn: number;
  nOut: number;
}
function lin(rand: () => number, nIn: number, nOut: number, gain = 1): Lin {
  const s = gain / Math.sqrt(nIn);
  return { w: new Float64Array(nIn * nOut).map(() => (2 * rand() - 1) * s * Math.sqrt(3)), b: new Float64Array(nOut).map(() => (2 * rand() - 1) * 0.1), nIn, nOut };
}
function apply(l: Lin, x: ArrayLike<number>, act = false): Float64Array {
  const y = new Float64Array(l.nOut);
  for (let o = 0; o < l.nOut; o++) {
    let s = l.b[o]!;
    for (let i = 0; i < l.nIn; i++) s += l.w[o * l.nIn + i]! * x[i]!;
    y[o] = act ? silu(s) : s;
  }
  return y;
}

export interface ToyMole {
  d: number;
  experts: number;
  element: Map<string, Float64Array>;
  /** Per-element environment feature fed to the MoLE layer (stands in for the backbone). */
  env: Map<string, Float64Array>;
  task: Map<UmaTask, Float64Array>;
  charge: (q: number) => Float64Array;
  spin: (s: number) => Float64Array;
  mixCsd: Lin;
  routing: [Lin, Lin, Lin];
  /** E expert matrices d×d, row-major, concatenated. */
  W: Float64Array;
  bias: Float64Array;
  readout: Float64Array;
}

export const TOY_ELEMENTS = ['H', 'C', 'N', 'O', 'Na', 'Cl', 'Si', 'Cu', 'Zn', 'Pt'] as const;

export function makeToyMole(o: { d?: number; experts?: number; seed?: number } = {}): ToyMole {
  const d = o.d ?? 8;
  const E = o.experts ?? UMA_EXPERTS;
  const rand = mulberry32(o.seed ?? 31);
  const vec = (n: number, s = 1) => new Float64Array(n).map(() => (2 * rand() - 1) * s);
  const element = new Map(TOY_ELEMENTS.map((z) => [z, vec(d, 1.5)] as const));
  const env = new Map(TOY_ELEMENTS.map((z) => [z, vec(d)] as const));
  const task = new Map(UMA_TASKS.map((t) => [t, vec(d, 1.5)] as const));
  const qTable = new Map<number, Float64Array>();
  const sTable = new Map<number, Float64Array>();
  const qSeed = vec(d);
  const sSeed = vec(d);
  // Each integer charge or spin has its own embedding (UMA Sec. 2.2.1); seeded deterministically.
  const table = (t: Map<number, Float64Array>, base: Float64Array, salt: number) => (k: number) => {
    if (!t.has(k)) {
      const r = mulberry32(1000 * salt + k + 500);
      t.set(k, base.map((b) => b + (2 * r() - 1)));
    }
    return t.get(k)!;
  };
  return {
    d,
    experts: E,
    element,
    env,
    task,
    charge: table(qTable, qSeed, 1),
    spin: table(sTable, sSeed, 2),
    mixCsd: lin(rand, 3 * d, d),
    routing: [lin(rand, 2 * d, 2 * E, 2), lin(rand, 2 * E, 2 * E, 2), lin(rand, 2 * E, E, 3)],
    W: vec(E * d * d, 1 / Math.sqrt(d)),
    bias: vec(d, 0.1),
    readout: vec(d),
  };
}

export interface SystemSpec {
  composition: Composition;
  task: UmaTask;
  charge?: number;
  spin?: number;
}

export const atomCount = (c: Composition) => Object.values(c).reduce((a, n) => a + n, 0);
export const scaled = (c: Composition, n: number): Composition => Object.fromEntries(Object.entries(c).map(([z, k]) => [z, k * n]));

/** Mean of per-element embeddings over atoms; with includeSelf the sum is divided by N + 1 (UMA-S-1.2). */
export function compositionEmbedding(m: ToyMole, c: Composition, includeSelf = false): Float64Array {
  const out = new Float64Array(m.d);
  const N = atomCount(c);
  for (const [z, k] of Object.entries(c)) {
    const e = m.element.get(z);
    if (!e) throw new Error(`no toy embedding for ${z}`);
    for (let i = 0; i < m.d; i++) out[i]! += k * e[i]!;
  }
  const denom = includeSelf ? N + 1 : N;
  return out.map((v) => v / denom);
}

/** Expert coefficients α (softmax + 0.005, so Σα = 1 + 0.005 E). */
export function routingCoefficients(m: ToyMole, s: SystemSpec, includeSelf = false): Float64Array {
  const csd = apply(m.mixCsd, [...m.charge(s.charge ?? 0), ...m.spin(s.spin ?? 0), ...m.task.get(s.task)!], true);
  const x = [...compositionEmbedding(m, s.composition, includeSelf), ...csd];
  const h1 = apply(m.routing[0], x, true);
  const h2 = apply(m.routing[1], h1, true);
  const z = apply(m.routing[2], h2, true);
  const mx = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, v) => a + v, 0);
  return ex.map((v) => v / sum + ALPHA_FLOOR);
}

/** W* = Σ_k α_k W_k (UMA Eq. 2). */
export function mergedWeights(m: ToyMole, alpha: ArrayLike<number>): Float64Array {
  const dd = m.d * m.d;
  const out = new Float64Array(dd);
  for (let k = 0; k < m.experts; k++) for (let i = 0; i < dd; i++) out[i]! += alpha[k]! * m.W[k * dd + i]!;
  return out;
}

/** y = Σ_k α_k (W_k x) + b, expert by expert (UMA Eq. 1). */
export function mixtureApply(m: ToyMole, alpha: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const y = Float64Array.from(m.bias);
  const dd = m.d * m.d;
  for (let k = 0; k < m.experts; k++)
    for (let o = 0; o < m.d; o++) {
      let s = 0;
      for (let i = 0; i < m.d; i++) s += m.W[k * dd + o * m.d + i]! * x[i]!;
      y[o]! += alpha[k]! * s;
    }
  return y;
}

/** y = W* x + b with the merged matrix. */
export function mergedApply(m: ToyMole, Wstar: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const y = Float64Array.from(m.bias);
  for (let o = 0; o < m.d; o++) for (let i = 0; i < m.d; i++) y[o]! += Wstar[o * m.d + i]! * x[i]!;
  return y;
}

/** Stand-in energy: Σ_atoms readout · SiLU(W* env(z) + b). Units arbitrary. */
export function toyEnergy(m: ToyMole, s: SystemSpec, includeSelf = false): number {
  const Ws = mergedWeights(m, routingCoefficients(m, s, includeSelf));
  let e = 0;
  for (const [z, k] of Object.entries(s.composition)) {
    const y = mergedApply(m, Ws, m.env.get(z)!);
    let t = 0;
    for (let i = 0; i < m.d; i++) t += m.readout[i]! * silu(y[i]!);
    e += k * t;
  }
  return e;
}

/** Multiply–adds for one MoLE layer applied to n inputs: merged once, or expert by expert. */
export function moleCost(d: number, experts: number, n: number): { merged: number; mixture: number } {
  return { merged: experts * d * d + n * d * d, mixture: n * experts * d * d };
}

export const l1 = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i]! - b[i]!);
  return s;
};
export const relFrob = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let d = 0;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    d += (a[i]! - b[i]!) ** 2;
    n += b[i]! ** 2;
  }
  return Math.sqrt(d / n);
};
