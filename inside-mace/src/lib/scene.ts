// Scene data for the Intervention Lab: the moved atom's neighbourhood cut from the periodic supercell
// by minimum image, centred on the moved atom's baseline position. Pure and tested.

import { framePositions, mic, norm, okFrameAt, type Manifest, type Vec3 } from './experiment';

export interface SceneAtom {
  /** Index in the supercell (stable across frames). */
  index: number;
  /** Position relative to the moved atom's baseline position (Å), minimum image. */
  pos: Vec3;
  /** Baseline distance to the moved atom (Å). */
  r0: number;
}

/** Atoms within `radius` of the moved atom at baseline, as minimum images, for frame k. */
export function neighbourhood(m: Manifest, k: number, radius: number): SceneAtom[] | null {
  const pos = framePositions(m, k);
  const base = framePositions(m, m.displacement.baselineIndex);
  if (!pos || !base) return null;
  const centre = base[m.displacement.atom]!;
  const out: SceneAtom[] = [];
  base.forEach((p, i) => {
    const v0 = mic(m, centre, p);
    const r0 = norm(v0);
    if (r0 > radius + 1e-6) return;
    // Same image in every frame: offset the frame position by the baseline image shift.
    const shift: Vec3 = [v0[0] - (p[0] - centre[0]), v0[1] - (p[1] - centre[1]), v0[2] - (p[2] - centre[2])];
    const q = pos[i]!;
    out.push({ index: i, r0, pos: [q[0] - centre[0] + shift[0], q[1] - centre[1] + shift[1], q[2] - centre[2] + shift[2]] });
  });
  return out.sort((a, b) => a.r0 - b.r0);
}

/** Display set of indices within `radius` (frame-independent). */
export function neighbourhoodIndices(m: Manifest, radius: number): number[] {
  return (neighbourhood(m, m.displacement.baselineIndex, radius) ?? []).map((a) => a.index);
}

/** Graph edges into atom `i` in frame k: neighbours within the model cutoff r_max (not chemical bonds). */
export function graphEdges(m: Manifest, k: number, atoms: SceneAtom[], i: number): { j: number; r: number }[] {
  const ai = atoms.find((a) => a.index === i);
  if (!ai || !okFrameAt(m, k)) return [];
  return atoms
    .filter((a) => a.index !== i)
    .map((a) => ({ j: a.index, r: norm([a.pos[0] - ai.pos[0], a.pos[1] - ai.pos[1], a.pos[2] - ai.pos[2]]) }))
    .filter((e) => e.r < m.rMax)
    .sort((a, b) => a.r - b.r);
}

/** Chemical-bond sticks drawn from a documented distance criterion (default: the Si diamond lattice; per experiment in experimentInfo.ts). */
export const BOND_CUTOFF = 2.6; // Å; Si–Si nearest neighbour 2.35 Å, second shell 3.84 Å

export function bonds(atoms: SceneAtom[], cutoff = BOND_CUTOFF): [number, number][] {
  const out: [number, number][] = [];
  for (let a = 0; a < atoms.length; a++)
    for (let b = a + 1; b < atoms.length; b++) {
      const p = atoms[a]!.pos;
      const q = atoms[b]!.pos;
      if (norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]]) < cutoff) out.push([a, b]);
    }
  return out;
}
