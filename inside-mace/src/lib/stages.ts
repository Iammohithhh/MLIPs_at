// The seven walkthrough stages, mapped onto the actual computation of the MACE-MP-0a medium checkpoint
// (ScaleShiftMACE, 2 interaction layers, hidden irreps 128x0e+128x1o, correlation order 3, l_max = 3,
// r_max = 6 Å, 10 Bessel functions, polynomial cutoff p = 5). Implementation locators are file:line in the installed
// mace-torch 0.3.16 package (checked against site-packages in the pipeline venv, 2026-10-17).
// Equations follow Batatia et al., NeurIPS 2022 (MACE), §3; the l = 1 ordering is e3nn's (x, y, z).
//
// `data` names the recorded tensor in the precomputed files; `subset` says when only some channels
// or atoms are stored. The walkthrough island (next sessions) renders these; this module is the
// single source of the stage text so beginner and technical views never diverge.

import { mic, norm, type Manifest } from './experiment';
import type { Stage } from './urlState';

export interface StageInfo {
  id: Stage;
  n: number;
  title: string;
  /** One plain-language sentence. */
  plain: string;
  /** What is new relative to the previous stage. */
  changed: string;
  /** KaTeX source. */
  tex: string;
  code: { file: string; line: number; symbol: string }[];
  data: string;
  shape: string;
  subset?: string;
}

const PKG = 'mace-torch 0.3.16';
export const CODE_PACKAGE = PKG;

export const STAGE_INFO: StageInfo[] = [
  {
    id: 'species',
    n: 1,
    title: 'Species and coordinates',
    plain: 'The model receives only each atom’s element and position, plus the periodic cell.',
    changed: 'Starting point: %N% atoms (%COMP%), positions in Å, periodic in x, y and z.',
    tex: String.raw`h_i^{(0)} = W_{\text{embed}}\, \mathrm{onehot}(Z_i)`,
    code: [{ file: 'mace/modules/models.py', line: 501, symbol: 'ScaleShiftMACE.forward: node_embedding' }],
    data: 'baseline.embed (moved atom)',
    shape: '[128] scalars (128x0e)',
  },
  {
    id: 'neighbours',
    n: 2,
    title: 'Neighbour graph',
    plain: 'Every pair of atoms closer than 6 Å becomes a directed graph edge; this is a cutoff rule, not a chemical bond.',
    changed: 'Atoms become a graph: the moved %EL% atom has %NB% neighbours within r_max at the baseline (periodic images included).',
    tex: String.raw`\mathcal{N}(i) = \{\, j : \lVert \mathbf r_j - \mathbf r_i + \mathbf S_{ij} \rVert < r_{\max} = 6\ \text{Å} \,\}`,
    code: [{ file: 'mace/data/neighborhood.py', line: 7, symbol: 'get_neighborhood' }],
    data: 'frames/*.json edgesIntoMoved.src, .shift, .vec',
    shape: '[edges into the moved atom]',
  },
  {
    id: 'radial',
    n: 3,
    title: 'Distances and directions',
    plain: 'Each edge is described by a smooth function of its length and by spherical harmonics of its direction.',
    changed: 'Edges gain features: 10 radial Bessel values (with a smooth cutoff) and 16 spherical-harmonic values (l = 0…3).',
    tex: String.raw`R_n(r_{ij}) = \sqrt{\tfrac{2}{r_{\max}}}\,\frac{\sin(n\pi r_{ij}/r_{\max})}{r_{ij}}\, f_{\text{cut}}(r_{ij}),\qquad Y_l^m(\hat{\mathbf r}_{ij})`,
    code: [
      { file: 'mace/modules/blocks.py', line: 390, symbol: 'RadialEmbeddingBlock' },
      { file: 'mace/modules/models.py', line: 502, symbol: 'spherical_harmonics(vectors)' },
    ],
    data: 'frames/*.json edgesIntoMoved.radial, .sh',
    shape: 'radial [edges, 10]; sh [edges, 16]',
  },
  {
    id: 'messages',
    n: 4,
    title: 'Neighbour aggregation',
    plain: 'Each atom sums contributions from its neighbours, weighted by learned functions of distance and direction, divided by a fixed average neighbour count (n̄ = 61.96 for this checkpoint).',
    changed: 'Per-atom tensors A appear: 128 channels for every l = 0…3, built from the edges of stage 3.',
    tex: String.raw`A_{i,k l m}^{(t)} = \frac{1}{\bar n}\sum_{j\in\mathcal N(i)} \sum_{l_1 l_2} C^{lm}_{l_1 m_1 l_2 m_2}\, R^{(t)}_{k l_1 l_2 l}(r_{ij})\, Y_{l_1}^{m_1}(\hat{\mathbf r}_{ij})\, W h^{(t)}_{j,k l_2 m_2}`,
    code: [
      { file: 'mace/modules/blocks.py', line: 705, symbol: 'RealAgnosticResidualInteractionBlock' },
      { file: 'mace/modules/blocks.py', line: 794, symbol: 'scatter_sum over edges; ÷ avg_num_neighbors (line 800)' },
    ],
    data: 'frames/*.json layers[t].Anorm',
    shape: '[local atoms, 4 (l), 128] block norms',
    subset: 'Stored as the norm over m of each (channel, l) block, for the moved atom and its first two neighbour shells.',
  },
  {
    id: 'features',
    n: 5,
    title: 'Higher-order features',
    plain: 'The aggregated tensors are multiplied with themselves to build features that depend on up to three neighbours at once (four-body).',
    changed: 'New features h: 128 scalar and 128 vector channels after layer 1; 128 scalar channels after layer 2.',
    tex: String.raw`B^{(t)}_{i,\eta \kappa L M} = \sum_{\mathbf{lm}} \mathcal C^{LM}_{\eta,\mathbf{lm}} \prod_{\xi=1}^{\nu} A^{(t)}_{i,\kappa l_\xi m_\xi},\ \ \nu \le 3;\qquad h^{(t+1)}_{i} = \text{Linear}(B^{(t)}_i) + \text{residual}`,
    code: [
      { file: 'mace/modules/blocks.py', line: 435, symbol: 'EquivariantProductBasisBlock' },
      { file: 'mace/modules/symmetric_contraction.py', line: 26, symbol: 'SymmetricContraction' },
    ],
    data: 'frames/*.json layers[t].h0, .h1; manifest frames[].change',
    shape: 'layer 1: [local, 128] + [local, 128, 3]; layer 2: [local, 128]',
    subset: 'All 128 channels, for the moved atom and its first two neighbour shells; change norms for all %N% atoms.',
  },
  {
    id: 'energy',
    n: 6,
    title: 'Atomic energies and the total',
    plain: 'A readout turns each atom’s scalar features into an energy contribution; the contributions add up to the total energy.',
    changed: 'Numbers with units: a site energy per atom (eV) and their sum, the total energy.',
    tex: String.raw`E_i = E_0(Z_i) + s\sum_{t=1}^{2} \mathcal R^{(t)}\big(h^{(t)}_i\big) + \mu,\qquad E = \sum_i E_i`,
    code: [
      { file: 'mace/modules/models.py', line: 570, symbol: 'readout per layer' },
      { file: 'mace/modules/models.py', line: 577, symbol: 'scale_shift; total_energy = e0 + inter_e' },
    ],
    data: 'manifest frames[].nodeEnergy, .site, .energy',
    shape: '[%N%] eV per frame; total in eV',
  },
  {
    id: 'forces',
    n: 7,
    title: 'Forces from the energy gradient',
    plain: 'The force on each atom is how fast the total energy changes when that atom moves — not the slope of its own site energy.',
    changed: 'Vectors with units: one force (eV/Å) per atom, from automatic differentiation of the total energy.',
    tex: String.raw`\mathbf F_i = -\frac{\partial E}{\partial \mathbf r_i} = -\sum_{j} \frac{\partial E_j}{\partial \mathbf r_i}\ \ne\ -\frac{\partial E_i}{\partial \mathbf r_i}`,
    code: [{ file: 'mace/modules/utils.py', line: 22, symbol: 'compute_forces: −autograd.grad(E, positions)' }],
    data: 'manifest frames[].forces; checks.finiteDifference',
    shape: '[%N%, 3] eV/Å per frame',
  },
];

/** Graph neighbours of `atom` within r_max at the baseline (minimum image; the cells are wider than 2·r_max). */
export function baselineNeighbourCount(m: Manifest, atom: number): number {
  const P = m.structure.positions;
  const ri = P[atom]!;
  let n = 0;
  for (let j = 0; j < P.length; j++) if (j !== atom && norm(mic(m, ri as [number, number, number], P[j] as [number, number, number])) < m.rMax) n++;
  return n;
}

/**
 * The stage texts for one experiment: counts, element and shapes come from the manifest, so no stage
 * sentence assumes the silicon run.
 */
export function stagesFor(m: Manifest): StageInfo[] {
  const counts = new Map<string, number>();
  for (const sym of m.structure.symbols) counts.set(sym, (counts.get(sym) ?? 0) + 1);
  const comp = [...counts].map(([sym, c]) => `${c} ${sym}`).join(', ');
  const atom = m.displacement.atom;
  const fill = (t: string) =>
    t
      .replaceAll('%N%', String(m.structure.n))
      .replaceAll('%COMP%', comp)
      .replaceAll('%EL%', m.structure.symbols[atom] ?? '?')
      .replaceAll('%NB%', String(baselineNeighbourCount(m, atom)));
  return STAGE_INFO.map((s) => ({ ...s, changed: fill(s.changed), shape: fill(s.shape), subset: s.subset === undefined ? undefined : fill(s.subset) }));
}
