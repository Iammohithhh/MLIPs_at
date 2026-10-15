// Interfacial water and the potential drop it creates (Electrochemistry Lab).
// TOY MODEL built from the illustrative Au(111)/water/NaCl interface and SPC/E partial
// charges: the first water layer is re-oriented by a tilt angle θ of its dipole from
// the surface normal; the planar-averaged charge density ρ̄(z) of all point charges
// (Gaussian-smeared) gives φ̄(z) from d²φ̄/dz² = −ρ̄/ε₀ with zero field below the film.
// The metal's electronic response is not modelled (Au carries no charge).

import { auWaterInterface, minImage, waterHydrogens, type Atom, type Vec3 } from '../chem/structures';
import { EPS0 } from '../physics/slab-dipole';

export interface InterfaceProfile {
  z: number[];
  rho: number[];
  phi: number[];
  /** Potential step across the film (V), φ above minus φ at the metal surface. */
  drop: number;
  /** Dipole per area of the first water layer along +z (e/Å). */
  firstLayerDipole: number;
  /** Contributions to the step (V) from each neutral group: Δφ = Σ q z / (A ε₀). */
  parts: { ions: number; firstLayer: number; otherWater: number };
  firstLayer: number[];
  atoms: Atom[];
  zTop: number;
  area: number;
}

const BASE = auWaterInterface();
const AREA = BASE.structure.cell[0] * BASE.structure.cell[1];

/** Indices (O) of the water molecules whose oxygen is within 3.6 Å of the top Au layer. */
export const FIRST_LAYER = BASE.oxygens.filter((i) => BASE.structure.atoms[i]!.p[2] - BASE.zTop < 3.6);

export function interfaceProfile(thetaDeg: number, width = 0.45, n = 900): InterfaceProfile {
  const s = BASE.structure;
  const atoms: Atom[] = s.atoms.map((a) => ({ ...a, p: [...a.p] as Vec3 }));
  const t = (thetaDeg * Math.PI) / 180;
  FIRST_LAYER.forEach((iO, k) => {
    const o = atoms[iO]!.p;
    const phi = 2.399963 * k; // golden-angle azimuths: even spread, deterministic
    const u: Vec3 = [Math.sin(t) * Math.cos(phi), Math.sin(t) * Math.sin(phi), Math.cos(t)];
    const [h1, h2] = waterHydrogens(o, u, 1.1 * k);
    atoms[iO + 1]!.p = h1;
    atoms[iO + 2]!.p = h2;
  });
  const z0 = BASE.zTop;
  const z1 = s.cell[2];
  const dz = (z1 - z0) / n;
  const z = Array.from({ length: n }, (_, i) => z0 + (i + 0.5) * dz);
  const rho = new Array<number>(n).fill(0);
  const norm = 1 / (width * Math.sqrt(2 * Math.PI) * AREA);
  for (const a of atoms) {
    if (!a.q) continue;
    const zi = a.p[2];
    const lo = Math.max(0, Math.floor((zi - 4 * width - z0) / dz));
    const hi = Math.min(n - 1, Math.ceil((zi + 4 * width - z0) / dz));
    for (let i = lo; i <= hi; i++) rho[i]! += a.q * norm * Math.exp(-((z[i]! - zi) ** 2) / (2 * width * width));
  }
  const phi: number[] = [];
  let E = 0;
  let V = 0;
  for (let i = 0; i < n; i++) {
    E += (rho[i]! * dz) / EPS0;
    V -= E * dz;
    phi.push(V);
  }
  const p = FIRST_LAYER.reduce((acc, iO) => {
    const o = atoms[iO]!.p;
    const h1 = minImage(o, atoms[iO + 1]!.p, s.cell, s.pbc);
    const h2 = minImage(o, atoms[iO + 2]!.p, s.cell, s.pbc);
    return acc + atoms[iO + 1]!.q * (h1[2] + h2[2]);
  }, 0);
  const first = new Set(FIRST_LAYER.flatMap((i) => [i, i + 1, i + 2]));
  const group = (pred: (a: Atom, i: number) => boolean) =>
    atoms.reduce((acc, a, i) => (a.q && pred(a, i) ? acc + a.q * a.p[2] : acc), 0) / (AREA * EPS0);
  const parts = {
    ions: group((a) => a.mol < 0),
    firstLayer: group((a, i) => a.mol >= 0 && first.has(i)),
    otherWater: group((a, i) => a.mol >= 0 && !first.has(i)),
  };
  return { z, rho, phi, drop: phi.at(-1)!, firstLayerDipole: p / AREA, parts, firstLayer: FIRST_LAYER, atoms, zTop: z0, area: AREA };
}
