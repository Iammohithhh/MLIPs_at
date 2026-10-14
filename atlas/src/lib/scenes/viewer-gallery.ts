// Structures for the atomic-viewer page that are built in TypeScript (the rest
// come from the Python structure pipeline). Parameters are stated, not measured.
import { ljForces } from '~/lib/physics/lennard-jones';
import { mulberry32 } from '~/lib/toy-lattice';
import type { Structure, Vec3 } from '~/lib/structure/types';

/** Argon-like σ used only to give the reduced-unit cluster a length scale in Å. */
export const LJ_SIGMA_A = 3.4;

/**
 * 13-atom icosahedral Lennard-Jones cluster (centre + 12 vertices) with every atom
 * displaced by a small seeded random amount, so the toy forces are non-zero.
 * Forces are in reduced units ε/σ; positions in Å with σ = 3.4 Å.
 */
export function ljCluster(seed = 3, jitter = 0.03): Structure {
  const phi = (1 + Math.sqrt(5)) / 2;
  const raw: Vec3[] = [
    [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
  ];
  // Centre–vertex distance 2^(1/6) σ (the pair minimum).
  const scale = 2 ** (1 / 6) / Math.hypot(0, 1, phi);
  const rand = mulberry32(seed);
  const reduced: Vec3[] = [[0, 0, 0], ...raw.map((v) => v.map((c) => c * scale) as Vec3)].map(
    (p) => p.map((c) => c + (rand() * 2 - 1) * jitter) as Vec3,
  );
  const forces = ljForces(reduced, { epsilon: 1, sigma: 1 });
  return {
    id: 'lj13',
    title: 'Lennard-Jones 13-atom cluster (toy forces)',
    symbols: reduced.map(() => 'Ar'),
    positions: reduced.map((p) => p.map((c) => c * LJ_SIGMA_A) as Vec3),
    cell: null,
    pbc: [false, false, false],
    display: { radius: 0.32 * LJ_SIGMA_A },
    vectors: {
      forces: { values: forces, unit: 'ε/σ', label: 'Lennard-Jones force', status: 'TOY_MODEL' },
    },
    provenance: {
      source: 'Icosahedron with centre–vertex distance 2^(1/6)σ, seeded random displacements of up to 0.03σ',
      notes: 'Forces F = −∂E/∂r from V(r) = 4ε[(σ/r)¹² − (σ/r)⁶], open boundaries, no cutoff.',
    },
  };
}
