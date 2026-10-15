// Physics Lab P9 — a polar slab in a periodic cell (planar averages along z).
//
// A dipole layer of surface density p (e/Å) sits on one face of a slab: a sheet −σ in
// the surface plane and a sheet +σ a distance d above it, σ = p/d, each smoothed by a
// Gaussian. The planar-averaged Poisson equation  d²φ/dz² = −ρ̄(z)/ε₀  is solved on one
// cell of length L_z in two ways:
//
// • Periodic (plain 3D Ewald / periodic DFT): φ must be periodic, so the mean field over
//   the cell vanishes. The dipole then forces an artificial uniform field
//       E_vac = p / (ε₀ L_z)
//   through the vacuum, which decays only as 1/L_z.
// • Dipole-corrected (Bengtsson, PRB 59, 12301, 1999; Yeh & Berkowitz, JCP 111, 3155,
//   1999): the field is zero in the vacuum and φ jumps by p/ε₀ at the cell boundary
//   (placed in the vacuum). The spurious energy removed per cell area A is
//       U/A = p² / (2 ε₀ L_z)        (= M_z²/(2ε₀V) with M_z = pA, V = A L_z).
//
// Units: Å, e, V. ε₀ = 5.526349e-3 e/(V·Å).

export const EPS0 = 5.526349e-3; // e / (V·Å)

export interface SlabParams {
  /** Surface dipole density (e/Å). */
  p: number;
  /** Cell length along z (Å). */
  Lz: number;
  /** z of the lower (−σ) sheet, measured from the cell bottom (Å). */
  zSheet: number;
  /** Sheet separation (Å). */
  d?: number;
  /** Gaussian smoothing width of each sheet (Å). */
  width?: number;
  /** Grid points. */
  n?: number;
}

export interface SlabProfile {
  z: number[];
  rho: number[];
  /** Field (V/Å) and potential (V), periodic boundary conditions. */
  Eper: number[];
  phiPer: number[];
  /** Field and potential with the dipole correction (φ not periodic: jumps at L_z). */
  Ecor: number[];
  phiCor: number[];
}

function gauss(x: number, w: number): number {
  return Math.exp(-(x * x) / (2 * w * w)) / (w * Math.sqrt(2 * Math.PI));
}

export function slabProfile({ p, Lz, zSheet, d = 1.6, width = 0.35, n = 1200 }: SlabParams): SlabProfile {
  const sigma = p / d;
  const dz = Lz / n;
  const z = Array.from({ length: n }, (_, i) => (i + 0.5) * dz);
  const rho = z.map((x) => sigma * (gauss(x - (zSheet + d), width) - gauss(x - zSheet, width)));
  // E0(z) = (1/ε₀) ∫₀^z ρ — starts at zero in the vacuum at the cell bottom.
  const E0: number[] = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += rho[i]! * dz;
    E0.push(acc / EPS0);
  }
  const mean = E0.reduce((a, b) => a + b, 0) / n;
  const Eper = E0.map((e) => e - mean);
  const integrate = (E: number[]) => {
    const phi: number[] = [];
    let s = 0;
    for (let i = 0; i < n; i++) {
      s -= E[i]! * dz;
      phi.push(s);
    }
    return phi;
  };
  return { z, rho, Eper, phiPer: integrate(Eper), Ecor: E0, phiCor: integrate(E0) };
}

/** Artificial vacuum field under periodic boundary conditions (V/Å). */
export const vacuumField = (p: number, Lz: number) => p / (EPS0 * Lz);
/** Potential step across the dipole layer (V). */
export const potentialStep = (p: number) => p / EPS0;
/** Spurious electrostatic energy per area (eV/Å²) removed by the slab correction. */
export const slabEnergyPerArea = (p: number, Lz: number) => (p * p) / (2 * EPS0 * Lz);
