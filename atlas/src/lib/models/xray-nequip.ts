// Architecture X-ray, NequIP: typed access to the intermediate tensors of a real NequIP-OAM-S 0.1
// forward pass, precomputed by pipeline/atlas_pipeline/xray_nequip.py (CPU, float64), plus the
// closed forms the page plots (Bessel basis with the trained frequencies, the polynomial cutoff,
// the ZBL pair term), each tested against the stored network outputs.
//
// Conventions of the data file (atlas-xray-nequip v1):
//   edges: src → dst; the message of an edge lands on dst (edge_index[0] in NequIP);
//   vec = r_dst − r_src (+ shift), as in the MACE file; sh[e] = Y(r̂) of the bond from dst to src
//   (NequIP's r_ij = r_j − r_i, i.e. −vec), e3nn order (l = 1 is (x, y, z)), component normalisation.
//   radial[e][n]: Bessel × cutoff × 2π/r_c², the input of the radial MLPs.
//   layers[t].message.norms[b][i][c] = |m_{i,c}| of tensor-product path block b (before linear_2);
//   message.l1[b][i][c][xyz] for vector blocks. h.l0/h.l1: features after the gate.
//   nodeEnergy_i = scale(z_i)·readout_i + shift(z_i) + zbl_i;  energy = Σ_i nodeEnergy_i.

import type { Vec3 } from './xray';

export type { Vec3 };

export interface NequipLayer {
  tp: { irreps_in1: string; irreps_in2: string; irreps_out: string; paths: [number, number, number][]; weightNumel: number };
  message: { ls: number[]; norms: number[][][]; l1: (Vec3[][] | null)[] };
  edgeWeightNorm: number[];
  h: { irreps: string; l0: number[][]; l1: Vec3[][] | null };
  preGate: { l0: number[][] };
  gates: number[][] | null;
}

interface Pass {
  positions: Vec3[];
  energy: number;
  forces: Vec3[];
  nodeEnergy: number[];
  layers: { h: { l0: number[][]; l1: Vec3[][] | null }; message: { norms: number[][][]; l1: (Vec3[][] | null)[] } }[];
}

export interface NequipSystem {
  id: string;
  label: string;
  symbols: string[];
  positions: Vec3[];
  cell: Vec3[] | null;
  pbc: boolean | boolean[];
  n: number;
  edges: { count: number; src: number[]; dst: number[]; vec: Vec3[]; radial: number[][]; cutoff: number[]; sh: number[][] };
  embed: { shape: number[]; h0: number[][] };
  layers: NequipLayer[];
  readout: number[];
  scale: number[];
  shift: number[];
  zbl: number[];
  nodeEnergy: number[];
  energy: number;
  forces: Vec3[];
  rotated?: Pass & { R: [Vec3, Vec3, Vec3] };
  mirrored?: Pass & { M: [Vec3, Vec3, Vec3] };
}

export interface DimerScan {
  pair: string;
  r: number[];
  total: number[];
  learned: number[];
  zbl: number[];
  e0: number[];
}

export interface NequipDoc {
  format: 'atlas-xray-nequip';
  version: 1;
  status: 'PRECOMPUTED';
  model: {
    family: string;
    checkpoint: string;
    file: string;
    url: string;
    doi: string;
    sha256: string;
    licence: string;
    packagedWith: Record<string, string>;
    numTypes: number;
    parameters: Record<string, number>;
    hyperparameters: {
      r_max: number;
      num_bessels: number;
      bessel_trainable: boolean;
      polynomial_cutoff_p: number;
      edge_factor: number;
      type_embed_num_features: number;
      irreps_edge_sh: string;
      feature_irreps_hidden: string[];
      radial_mlp_dims: number[][];
      scatter_norm_factor: number;
      pair_potential: string | null;
      num_layers: number;
      model_dtype: string;
      avg_num_neighbors: number;
      bessel_weights: number[];
    };
  };
  provenance: Record<string, unknown> & {
    rotation: { axis: number[]; angleDeg: number; system: string };
    mirror: { normal: number[]; system: string };
  };
  systems: NequipSystem[];
  dimers: DimerScan[];
}

// ---------------------------------------------------------------------------
// Closed forms

/** DimeNet polynomial envelope as nequip implements it, x = r / r_c. */
export function polynomialCutoff(x: number, p: number): number {
  if (x >= 1) return 0;
  return 1 - ((p + 1) * (p + 2) / 2) * x ** p + p * (p + 2) * x ** (p + 1) - (p * (p + 1) / 2) * x ** (p + 2);
}

/**
 * Radial feature n (0-based) on an edge of length r: the network's input to the radial MLP,
 * (2/r_c)·sin(w_n π r/r_c)/r · f_cut(r/r_c), i.e. Batzner et al. Eq. (6) with the trained w_n
 * (initialised to n + 1). nequip computes it as sinc(x w)·w · f_cut(x) · 2π/r_c².
 */
export function besselFeature(r: number, n: number, w: readonly number[], rc: number, p: number): number {
  const x = r / rc;
  if (x >= 1) return 0;
  const wn = w[n]!;
  const arg = Math.PI * x * wn;
  const sinc = arg === 0 ? 1 : Math.sin(arg) / arg;
  return sinc * wn * polynomialCutoff(x, p) * ((2 * Math.PI) / (rc * rc));
}

/** e²/4πε₀ in eV·Å as nequip and LAMMPS (metal units) use it. */
export const QQR2E = 14.399645;

/**
 * ZBL universal screened Coulomb repulsion between nuclei Zi, Zj at r (Å), in eV, without cutoff:
 * E = (Zi Zj e²/4πε₀ r) φ(r/a), a = 0.46850/(Zi^0.23 + Zj^0.23), φ a sum of four exponentials.
 */
export function zblPair(Zi: number, Zj: number, r: number): number {
  const x = ((Zi ** 0.23 + Zj ** 0.23) * r) / 0.4685;
  const phi = 0.02817 * Math.exp(-0.20162 * x) + 0.28022 * Math.exp(-0.4029 * x) + 0.50986 * Math.exp(-0.94229 * x) + 0.18175 * Math.exp(-3.1998 * x);
  return ((QQR2E * Zi * Zj) / r) * phi;
}

/** The model's ZBL energy of one pair: the bare term times the radial cutoff (split ½ to each atom). */
export function zblWithCutoff(Zi: number, Zj: number, r: number, rc: number, p: number): number {
  return zblPair(Zi, Zj, r) * polynomialCutoff(r / rc, p);
}

// ---------------------------------------------------------------------------
// Bookkeeping

const norm3 = (v: readonly number[]) => Math.hypot(v[0]!, v[1]!, v[2]!);

/** Per-atom energy split: learned (scale × readout), shift (isolated-atom reference) and ZBL. */
export function energyParts(sys: NequipSystem) {
  return sys.symbols.map((sym, i) => {
    const learned = sys.scale[i]! * sys.readout[i]!;
    return { sym, learned, shift: sys.shift[i]!, zbl: sys.zbl[i]!, total: learned + sys.shift[i]! + sys.zbl[i]! };
  });
}

/** Irreps string → blocks with multiplicity and l, e.g. '128x0e+64x1e'. */
export function parseIrreps(irreps: string): { mul: number; l: number; p: 'e' | 'o' }[] {
  return irreps.split('+').map((part) => {
    const [mul, ir] = part.split('x') as [string, string];
    return { mul: Number(mul), l: Number(ir.slice(0, -1)), p: ir.slice(-1) as 'e' | 'o' };
  });
}

/** Tensor-product paths of a layer as 'l₁ ⊗ l₂ → l_out' with their channel counts. */
export function tpPaths(layer: NequipLayer) {
  const a = parseIrreps(layer.tp.irreps_in1);
  const b = parseIrreps(layer.tp.irreps_in2);
  const o = parseIrreps(layer.tp.irreps_out);
  return layer.tp.paths.map(([i, j, k]) => ({ in1: a[i]!, in2: b[j]!, out: o[k]!, label: `${a[i]!.l} ⊗ ${b[j]!.l} → ${o[k]!.l}` }));
}

/** True when some path couples two l = 1 inputs into l = 1: the cross product a pseudoscalar needs. */
export function hasCrossProductPath(doc: NequipDoc): boolean {
  return doc.systems[0]!.layers.some((L) => tpPaths(L).some((p) => p.in1.l === 1 && p.in2.l === 1 && p.out.l === 1));
}

const matVec = (R: readonly Vec3[], v: readonly number[]): Vec3 => [
  R[0]![0] * v[0]! + R[0]![1] * v[1]! + R[0]![2] * v[2]!,
  R[1]![0] * v[0]! + R[1]![1] * v[1]! + R[1]![2] * v[2]!,
  R[2]![0] * v[0]! + R[2]![1] * v[1]! + R[2]![2] * v[2]!,
];

/**
 * Compare the stored pass with its rotated or mirrored copy (matrix Q): scalars must be equal,
 * l = 1 features and forces must equal Q·v. Deviations are relative to the largest magnitude.
 */
export function symmetryCheck(sys: NequipSystem, which: 'rotated' | 'mirrored') {
  const t = sys[which];
  if (!t) return null;
  const Q = which === 'rotated' ? (t as NonNullable<NequipSystem['rotated']>).R : (t as NonNullable<NequipSystem['mirrored']>).M;
  const rel = (pairs: [number, number][]) => {
    let d = 0;
    let m = 0;
    for (const [a, b] of pairs) {
      d = Math.max(d, Math.abs(a - b));
      m = Math.max(m, Math.abs(a));
    }
    return m > 0 ? d / m : 0;
  };
  const vecRel = (a: Vec3[][] | null, b: Vec3[][] | null) => {
    if (!a || !b) return null;
    let d = 0;
    let m = 0;
    a.forEach((row, i) =>
      row.forEach((v, c) => {
        const Qv = matVec(Q, v);
        const w = b[i]![c]!;
        d = Math.max(d, Math.hypot(Qv[0] - w[0], Qv[1] - w[1], Qv[2] - w[2]));
        m = Math.max(m, norm3(v));
      }),
    );
    return m > 0 ? d / m : 0;
  };
  const layers = sys.layers.map((L, k) => {
    const Lt = t.layers[k]!;
    return {
      l0: rel(L.h.l0.flatMap((row, i) => row.map((x, c) => [x, Lt.h.l0[i]![c]!] as [number, number]))),
      hl1: vecRel(L.h.l1, Lt.h.l1),
      ml1: L.message.l1.map((blk, b) => vecRel(blk, Lt.message.l1[b] ?? null)).find((x) => x !== null) ?? null,
    };
  });
  let dF = 0;
  let mF = 0;
  sys.forces.forEach((f, i) => {
    const Qf = matVec(Q, f);
    const g = t.forces[i]!;
    dF = Math.max(dF, Math.hypot(Qf[0] - g[0], Qf[1] - g[1], Qf[2] - g[2]));
    mF = Math.max(mF, norm3(f));
  });
  return { dE: Math.abs(t.energy - sys.energy), forces: dF / mF, layers, det: det3(Q) };
}

export function det3(Q: readonly Vec3[]): number {
  const [a, b, c] = Q as [Vec3, Vec3, Vec3];
  return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

/** The transformed pass as a system of its own (no edge tensors are stored for it). */
export function transformedView(sys: NequipSystem, which: 'rotated' | 'mirrored'): NequipSystem | null {
  const t = sys[which];
  if (!t) return null;
  return {
    ...sys,
    positions: t.positions,
    edges: { count: 0, src: [], dst: [], vec: [], radial: [], cutoff: [], sh: [] },
    layers: sys.layers.map((L, k) => {
      const Lt = t.layers[k]!;
      return { ...L, message: { ...L.message, norms: Lt.message.norms, l1: Lt.message.l1 }, h: { ...L.h, l0: Lt.h.l0, l1: Lt.h.l1 } };
    }),
    nodeEnergy: t.nodeEnergy,
    energy: t.energy,
    forces: t.forces,
    rotated: undefined,
    mirrored: undefined,
  };
}

// ---------------------------------------------------------------------------
// Tensors the explorer can select

export type NequipTensorId = 'embed' | 'radial' | 'sh' | 'm1' | 'h1' | 'm2' | 'h2' | 'energy';

export interface Block {
  /** Index of the block within the tensor (a path for messages, an irrep for features). */
  key: number;
  l: number;
  channels: number;
  label: string;
}

/** Selectable blocks of a node tensor. */
export function tensorBlocks(sys: NequipSystem, id: NequipTensorId): Block[] {
  if (id === 'embed') return [{ key: 0, l: 0, channels: sys.embed.shape[1]!, label: `${sys.embed.shape[1]}×0` }];
  if (id === 'm1' || id === 'm2') {
    const L = sys.layers[id === 'm1' ? 0 : 1]!;
    return tpPaths(L).map((p, k) => ({ key: k, l: p.out.l, channels: L.message.norms[k]![0]!.length, label: p.label }));
  }
  if (id === 'h1' || id === 'h2') {
    const L = sys.layers[id === 'h1' ? 0 : 1]!;
    return parseIrreps(L.h.irreps).map((b, k) => ({ key: k, l: b.l, channels: b.mul, label: `${b.mul}×${b.l}` }));
  }
  return [];
}

/** Signed value (l = 0 features) or norm (l = 1 blocks and messages) of one channel at every atom. */
export function channelValues(sys: NequipSystem, id: NequipTensorId, key: number, c: number): number[] | null {
  if (id === 'embed') return sys.embed.h0.map((row) => row[c]!);
  if (id === 'm1' || id === 'm2') {
    const blk = sys.layers[id === 'm1' ? 0 : 1]!.message.norms[key];
    return blk ? blk.map((row) => row[c]!) : null;
  }
  if (id === 'h1' || id === 'h2') {
    const h = sys.layers[id === 'h1' ? 0 : 1]!.h;
    if (key === 0) return h.l0.map((row) => row[c]!);
    if (key === 1 && h.l1) return h.l1.map((row) => norm3(row[c]!));
  }
  return null;
}

export function isSigned(id: NequipTensorId, key: number): boolean {
  return key === 0 && (id === 'embed' || id === 'h1' || id === 'h2');
}

/** Cartesian l = 1 vector of channel c at every atom, or null. */
export function channelVectors(sys: NequipSystem, id: NequipTensorId, key: number, c: number): Vec3[] | null {
  const pick = (blk: Vec3[][] | null | undefined) => (blk ? blk.map((row) => row[c]!) : null);
  if (id === 'm1' || id === 'm2') return pick(sys.layers[id === 'm1' ? 0 : 1]!.message.l1[key]);
  if ((id === 'h1' || id === 'h2') && key === 1) return pick(sys.layers[id === 'h1' ? 0 : 1]!.h.l1);
  return null;
}

/** Atoms × channels matrix of one block. */
export function channelMatrix(sys: NequipSystem, id: NequipTensorId, key: number): number[][] | null {
  const blk = tensorBlocks(sys, id).find((b) => b.key === key);
  if (!blk) return null;
  const cols: number[][] = [];
  for (let c = 0; c < blk.channels; c++) {
    const v = channelValues(sys, id, key, c);
    if (!v) return null;
    cols.push(v);
  }
  return sys.symbols.map((_, i) => cols.map((col) => col[i]!));
}

export function mostVariedChannel(sys: NequipSystem, id: NequipTensorId, key: number): number {
  const m = channelMatrix(sys, id, key);
  if (!m) return 0;
  let best = 0;
  let bestVar = -1;
  for (let c = 0; c < m[0]!.length; c++) {
    const v = m.map((row) => row[c]!);
    const mean = v.reduce((s, x) => s + x, 0) / v.length;
    const variance = v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length;
    if (variance > bestVar) {
      bestVar = variance;
      best = c;
    }
  }
  return best;
}

/** Tensor shapes along the dataflow, read from the data. */
export function dataflowShapes(doc: NequipDoc, sys: NequipSystem): Record<NequipTensorId, string> {
  const hp = doc.model.hyperparameters;
  const E = sys.edges.count;
  const n = sys.n;
  const width = (irreps: string) => parseIrreps(irreps).reduce((s, b) => s + b.mul * (2 * b.l + 1), 0);
  return {
    embed: `[${sys.embed.shape.join(', ')}]`,
    radial: `[${E}, ${hp.num_bessels}]`,
    sh: `[${E}, ${width(hp.irreps_edge_sh)}]`,
    m1: `[${n}, ${width(sys.layers[0]!.tp.irreps_out)}]`,
    h1: `[${n}, ${width(sys.layers[0]!.h.irreps)}]`,
    m2: `[${n}, ${width(sys.layers[1]!.tp.irreps_out)}]`,
    h2: `[${n}, ${width(sys.layers[1]!.h.irreps)}]`,
    energy: `[${n}] → E`,
  };
}

/** Share of the trainable parameters in each part of the network. */
export function parameterShares(doc: NequipDoc) {
  const p = doc.model.parameters;
  const total = Object.values(p).reduce((s, x) => s + x, 0);
  return { total, parts: Object.entries(p).map(([name, count]) => ({ name, count, share: count / total })) };
}
