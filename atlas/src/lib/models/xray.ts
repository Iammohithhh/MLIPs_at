// Architecture X-ray: typed access to the intermediate tensors of a real MACE forward pass,
// precomputed by pipeline/atlas_pipeline/xray.py (MACE-MP-0a medium, CPU, float64).
//
// Conventions of the data file (atlas-xray v1):
//   edges: src → dst, vec = r_dst − r_src + shift (Å); the message of an edge lands on dst.
//   radial[e][n]: the 10 Bessel × envelope features of edge e;  sh[e][k]: Y_lm, k = l² + l + m
//   (e3nn order, component normalisation; l = 1 is (x, y, z)).
//   layers[t].A.norms[l][i][c] = |A_{i,c,l·}|;  A.l1[i][c] = Cartesian l = 1 block.
//   layers[t].h.l0[i][c], h.l1[i][c][xyz]: node features after the product + update.
//   layers[t].siteEnergy[i]: readout t before scale and shift.
//   nodeEnergy = e0 + scale · Σ_t siteEnergy_t + shift;  energy = Σ_i nodeEnergy_i.

export type Vec3 = [number, number, number];

export interface XrayLayer {
  A: { shape: number[]; peak: number[]; norms: (number[][] | null)[]; l1: Vec3[][] | null };
  h: { shape: number[]; irreps: string; l0: number[][]; l1: Vec3[][] | null };
  siteEnergy: number[];
}

export interface XraySystem {
  id: string;
  label: string;
  symbols: string[];
  positions: Vec3[];
  cell: Vec3[] | null;
  pbc: boolean | boolean[];
  n: number;
  edges: { count: number; src: number[]; dst: number[]; vec: Vec3[]; radial: number[][]; sh: number[][] };
  embed: { shape: number[]; h0: number[][] };
  layers: XrayLayer[];
  energy: number;
  nodeEnergy: number[];
  e0: number[];
  forces: Vec3[];
  rotated?: {
    R: [Vec3, Vec3, Vec3];
    positions: Vec3[];
    energy: number;
    forces: Vec3[];
    nodeEnergy: number[];
    layers: { h: { l0: number[][]; l1: Vec3[][] | null }; A: { norms: (number[][] | null)[]; l1: Vec3[][] | null }; siteEnergy: number[] }[];
  };
}

export interface XrayDoc {
  format: 'atlas-xray';
  version: 1;
  status: 'PRECOMPUTED';
  model: {
    family: string;
    checkpoint: string;
    file: string;
    url: string;
    sha256: string;
    licence: string;
    hyperparameters: {
      r_max: number;
      num_bessel: number;
      num_polynomial_cutoff: number;
      max_ell: number;
      num_interactions: number;
      hidden_irreps: string;
      MLP_irreps: string;
      correlation: number;
      avg_num_neighbors: number;
      radial_MLP: number[];
      atomic_inter_scale: number;
      atomic_inter_shift: number;
    };
  };
  provenance: Record<string, unknown> & { rotation: { axis: number[]; angleDeg: number; system: string } };
  systems: XraySystem[];
}

/**
 * The rotated forward pass as a system of its own (positions, features, energies and forces
 * all from the second, rotated run). Edge tensors are not stored for it: they are dropped.
 */
export function rotatedView(sys: XraySystem): XraySystem | null {
  const r = sys.rotated;
  if (!r) return null;
  return {
    ...sys,
    positions: r.positions,
    edges: { count: 0, src: [], dst: [], vec: [], radial: [], sh: [] },
    layers: sys.layers.map((L, t) => {
      const Lr = r.layers[t]!;
      return { A: { ...L.A, norms: Lr.A.norms, l1: Lr.A.l1 }, h: { ...L.h, l0: Lr.h.l0, l1: Lr.h.l1 }, siteEnergy: Lr.siteEnergy };
    }),
    energy: r.energy,
    nodeEnergy: r.nodeEnergy,
    forces: r.forces,
    rotated: undefined,
  };
}

/** The tensors a reader can select, in dataflow order. */
export type TensorId = 'embed' | 'radial' | 'sh' | 'A1' | 'h1' | 'A2' | 'h2' | 'energy';

export const CHANNELS = 128;

export function matVec(R: readonly Vec3[], v: readonly number[]): Vec3 {
  return [
    R[0]![0] * v[0]! + R[0]![1] * v[1]! + R[0]![2] * v[2]!,
    R[1]![0] * v[0]! + R[1]![1] * v[1]! + R[1]![2] * v[2]!,
    R[2]![0] * v[0]! + R[2]![1] * v[1]! + R[2]![2] * v[2]!,
  ];
}

const norm3 = (v: readonly number[]) => Math.hypot(v[0]!, v[1]!, v[2]!);

/** Which irreps (l) a node tensor carries in this checkpoint, and whether each is non-vanishing. */
export function tensorLs(sys: XraySystem, id: TensorId): { l: number; live: boolean }[] {
  switch (id) {
    case 'embed':
      return [{ l: 0, live: true }];
    case 'A1':
    case 'A2': {
      const L = sys.layers[id === 'A1' ? 0 : 1]!;
      return L.A.norms.map((b, l) => ({ l, live: b !== null }));
    }
    case 'h1':
    case 'h2': {
      const L = sys.layers[id === 'h1' ? 0 : 1]!;
      return L.h.l1 ? [{ l: 0, live: true }, { l: 1, live: true }] : [{ l: 0, live: true }];
    }
    default:
      return [];
  }
}

/**
 * Per-atom value of one channel of one irrep: the signed scalar for l = 0 features, the
 * norm |x_{c,l·}| for l ≥ 1 (and for A, whose l = 0 block is reported as a norm too).
 * Returns null when the block is absent or vanishes in this checkpoint.
 */
export function channelValues(sys: XraySystem, id: TensorId, l: number, c: number): number[] | null {
  if (id === 'embed') return l === 0 ? sys.embed.h0.map((row) => row[c]!) : null;
  if (id === 'A1' || id === 'A2') {
    const b = sys.layers[id === 'A1' ? 0 : 1]!.A.norms[l];
    return b ? b.map((row) => row[c]!) : null;
  }
  if (id === 'h1' || id === 'h2') {
    const h = sys.layers[id === 'h1' ? 0 : 1]!.h;
    if (l === 0) return h.l0.map((row) => row[c]!);
    if (l === 1 && h.l1) return h.l1.map((row) => norm3(row[c]!));
  }
  return null;
}

/** l = 1 vector of channel c at every atom (Cartesian, Å-free feature units), or null. */
export function channelVectors(sys: XraySystem, id: TensorId, c: number): Vec3[] | null {
  const pick = (blk: Vec3[][] | null) => (blk ? blk.map((row) => row[c]!) : null);
  if (id === 'A1') return pick(sys.layers[0]!.A.l1);
  if (id === 'A2') return pick(sys.layers[1]!.A.l1);
  if (id === 'h1') return pick(sys.layers[0]!.h.l1);
  if (id === 'h2') return pick(sys.layers[1]!.h.l1);
  return null;
}

/** Atoms × channels matrix shown as the tensor strip (same convention as channelValues). */
export function channelMatrix(sys: XraySystem, id: TensorId, l: number): number[][] | null {
  const rows: number[][] = [];
  for (let c = 0; c < CHANNELS; c++) {
    const v = channelValues(sys, id, l, c);
    if (!v) return null;
    rows.push(v);
  }
  return sys.symbols.map((_, i) => rows.map((col) => col[i]!));
}

/** Channel whose values differ most between atoms (largest variance): a telling default. */
export function mostVariedChannel(sys: XraySystem, id: TensorId, l: number): number {
  let best = 0;
  let bestVar = -1;
  for (let c = 0; c < CHANNELS; c++) {
    const v = channelValues(sys, id, l, c);
    if (!v) return 0;
    const mean = v.reduce((s, x) => s + x, 0) / v.length;
    const variance = v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length;
    if (variance > bestVar) {
      bestVar = variance;
      best = c;
    }
  }
  return best;
}

/** Incoming edges of atom i (messages that land on i), with the neighbour's image position. */
export function incomingEdges(sys: XraySystem, i: number): { e: number; j: number; r: number; from: Vec3 }[] {
  const out: { e: number; j: number; r: number; from: Vec3 }[] = [];
  const p = sys.positions[i]!;
  for (let e = 0; e < sys.edges.count; e++) {
    if (sys.edges.dst[e] !== i) continue;
    const v = sys.edges.vec[e]!;
    out.push({ e, j: sys.edges.src[e]!, r: norm3(v), from: [p[0] - v[0], p[1] - v[1], p[2] - v[2]] });
  }
  return out.sort((a, b) => a.r - b.r);
}

/** Per-atom energy decomposition: E0, the two scaled readouts and the shift (eV). */
export function energyParts(doc: XrayDoc, sys: XraySystem) {
  const { atomic_inter_scale: s, atomic_inter_shift: b } = doc.model.hyperparameters;
  return sys.symbols.map((sym, i) => {
    const r = sys.layers.map((L) => s * L.siteEnergy[i]!);
    const total = sys.e0[i]! + r.reduce((a, x) => a + x, 0) + b;
    return { sym, e0: sys.e0[i]!, readouts: r, shift: b, interaction: total - sys.e0[i]!, total };
  });
}

/**
 * The rotation test on the stored rotated copy: l = 0 features must be unchanged and
 * l = 1 features must equal R·v. Returns the largest deviations relative to the largest
 * feature magnitude, for each layer that carries the block.
 */
export function rotationCheck(sys: XraySystem) {
  const rot = sys.rotated;
  if (!rot) return null;
  const R = rot.R;
  const layers = sys.layers.map((L, t) => {
    const Lr = rot.layers[t]!;
    let d0 = 0;
    let m0 = 0;
    L.h.l0.forEach((row, i) =>
      row.forEach((x, c) => {
        d0 = Math.max(d0, Math.abs(x - Lr.h.l0[i]![c]!));
        m0 = Math.max(m0, Math.abs(x));
      }),
    );
    const vec = (a: Vec3[][] | null, b: Vec3[][] | null) => {
      if (!a || !b) return null;
      let d = 0;
      let m = 0;
      a.forEach((row, i) =>
        row.forEach((v, c) => {
          const Rv = matVec(R, v);
          const w = b[i]![c]!;
          d = Math.max(d, Math.hypot(Rv[0] - w[0], Rv[1] - w[1], Rv[2] - w[2]));
          m = Math.max(m, norm3(v));
        }),
      );
      return d / m;
    };
    return { l0: m0 > 0 ? d0 / m0 : 0, hl1: vec(L.h.l1, Lr.h.l1), Al1: vec(L.A.l1, Lr.A.l1) };
  });
  let dF = 0;
  let mF = 0;
  sys.forces.forEach((f, i) => {
    const Rf = matVec(R, f);
    const g = rot.forces[i]!;
    dF = Math.max(dF, Math.hypot(Rf[0] - g[0], Rf[1] - g[1], Rf[2] - g[2]));
    mF = Math.max(mF, norm3(f));
  });
  return { dE: Math.abs(rot.energy - sys.energy), forces: dF / mF, layers };
}

/** Tensor shapes along the dataflow, read from the data (not typed by hand). */
export function dataflowShapes(doc: XrayDoc, sys: XraySystem): Record<TensorId, string> {
  const hp = doc.model.hyperparameters;
  const [L1, L2] = sys.layers;
  const n = sys.n;
  const E = sys.edges.count;
  return {
    embed: `[${sys.embed.shape.join(', ')}]`,
    radial: `[${E}, ${hp.num_bessel}]`,
    sh: `[${E}, ${(hp.max_ell + 1) ** 2}]`,
    A1: `[${L1!.A.shape.join(', ')}]`,
    h1: `[${L1!.h.shape.join(', ')}]`,
    A2: `[${L2!.A.shape.join(', ')}]`,
    h2: `[${L2!.h.shape.join(', ')}]`,
    energy: `[${n}] → E`,
  };
}
