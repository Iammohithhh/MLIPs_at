// Rigid cleavage of an fcc(111) crystal with a Morse pair potential (Failure Atlas: surfaces).
// TOY MODEL: the top half of a periodic-in-plane fcc(111) block is pulled away from the bottom
// half by a gap d without relaxation. Only pairs that straddle the cleavage plane change, so
// E(d) = Σ_{i below, j above, images} [φ(r_ij(d)) − φ(r_ij(0))], per in-plane cell area.
// As d → ∞ this is the (unrelaxed) cleavage energy, twice the surface energy.

type V3 = [number, number, number];

/** Morse parameters for Cu as tabulated by Girifalco & Weizer (1959): D (eV), α (1/Å), r0 (Å). */
export const MORSE_CU = { D: 0.3429, alpha: 1.3588, r0: 2.866 } as const;
/** Cu lattice constant used for the toy geometry (Å). */
export const A_CU = 3.615;

export interface Morse {
  D: number;
  alpha: number;
  r0: number;
}

/** Morse pair energy shifted to zero at rc. */
export function morse(r: number, p: Morse, rc: number): number {
  if (r >= rc) return 0;
  const f = (x: number) => p.D * (Math.exp(-2 * p.alpha * (x - p.r0)) - 2 * Math.exp(-p.alpha * (x - p.r0)));
  return f(r) - f(rc);
}

export interface Fcc111 {
  /** Positions with layer index; z of layer k is k·dz. */
  atoms: Array<{ p: V3; layer: number }>;
  cell: [number, number];
  dz: number;
  nn: number;
  layers: number;
}

/** Rectangular fcc(111) block: nx × ny rectangular units (2 atoms each) per layer, ABC stacking. */
export function fcc111(a: number, nx: number, ny: number, layers: number): Fcc111 {
  const as = a / Math.SQRT2;
  const h = (as * Math.sqrt(3)) / 2;
  const dz = a / Math.sqrt(3);
  const shifts: Array<[number, number]> = [
    [0, 0],
    [as / 2, h / 3],
    [as, (2 * h) / 3],
  ];
  const atoms: Fcc111['atoms'] = [];
  for (let k = 0; k < layers; k++) {
    const [sx, sy] = shifts[k % 3]!;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++)
        for (const [bx, by] of [
          [0, 0],
          [as / 2, h],
        ] as const) {
          const x = (((i * as + bx + sx) % (nx * as)) + nx * as) % (nx * as);
          const y = (((j * 2 * h + by + sy) % (ny * 2 * h)) + ny * 2 * h) % (ny * 2 * h);
          atoms.push({ p: [x, y, k * dz], layer: k });
        }
  }
  return { atoms, cell: [nx * as, ny * 2 * h], dz, nn: as, layers };
}

/** Pairs (i below, j above) with their in-plane image offsets within rc at zero gap; opening
 *  the gap only moves them apart, so no other pair ever enters the cutoff. */
export function crossPairs(s: Fcc111, rc: number) {
  const mid = s.layers / 2;
  const pairs: Array<{ i: number; j: number; dx: number; dy: number; dz0: number }> = [];
  const [Lx, Ly] = s.cell;
  const nix = Math.ceil(rc / Lx);
  const niy = Math.ceil(rc / Ly);
  s.atoms.forEach((ai, i) => {
    if (ai.layer >= mid) return;
    s.atoms.forEach((aj, j) => {
      if (aj.layer < mid) return;
      for (let u = -nix; u <= nix; u++)
        for (let v = -niy; v <= niy; v++) {
          const dx = aj.p[0] + u * Lx - ai.p[0];
          const dy = aj.p[1] + v * Ly - ai.p[1];
          const dz0 = aj.p[2] - ai.p[2];
          if (dx * dx + dy * dy + dz0 * dz0 < rc * rc) pairs.push({ i, j, dx, dy, dz0 });
        }
    });
  });
  return pairs;
}

/** Energy per area (meV/Å²) to open a gap d across the middle of the block, rigidly. */
export function cleavageEnergy(s: Fcc111, pairs: ReturnType<typeof crossPairs>, d: number, p: Morse, rc: number): number {
  let e = 0;
  for (const q of pairs) {
    const r2 = q.dx * q.dx + q.dy * q.dy;
    e += morse(Math.sqrt(r2 + (q.dz0 + d) ** 2), p, rc) - morse(Math.sqrt(r2 + q.dz0 * q.dz0), p, rc);
  }
  return (1000 * e) / (s.cell[0] * s.cell[1]);
}

/** Number of neighbours of each atom within rcut, with the top half shifted up by d (periodic in-plane). */
export function coordination(s: Fcc111, d: number, rcut: number): number[] {
  const mid = s.layers / 2;
  const [Lx, Ly] = s.cell;
  const z = (k: number) => s.atoms[k]!.p[2] + (s.atoms[k]!.layer >= mid ? d : 0);
  return s.atoms.map((ai, i) => {
    let n = 0;
    s.atoms.forEach((aj, j) => {
      if (i === j) return;
      let dx = aj.p[0] - ai.p[0];
      let dy = aj.p[1] - ai.p[1];
      dx -= Lx * Math.round(dx / Lx);
      dy -= Ly * Math.round(dy / Ly);
      const dz = z(j) - z(i);
      if (dx * dx + dy * dy + dz * dz < rcut * rcut) n++;
    });
    return n;
  });
}
