import { describe, expect, it } from 'vitest';
import {
  besselRoots,
  dihedral,
  envelope,
  forward,
  gemnetOcBounds,
  graph,
  interactionCounts,
  hexagonAndTriangles,
  hexagonSecondNeighbour,
  legendre,
  makeToyDimeNet,
  makeToyGemNetQ,
  PEROXIDE,
  peroxide,
  peroxideDistances,
  quadruplets,
  realY,
  rbfTilde,
  relDiff,
  sbfTilde,
  sphericalBessel,
  torsionScan,
  y0,
  type Vec3,
} from './dimenet';
import { forward as schnetForward, makeToySchNet } from './schnet';

const D = 1.5; // ring bond length, Å

function rotate(pos: readonly Vec3[], axis: Vec3, angle: number, shift: Vec3 = [0, 0, 0]): Vec3[] {
  const n = Math.hypot(...axis);
  const [x, y, z] = axis.map((a) => a / n) as [number, number, number];
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const R = [
    [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
  ];
  return pos.map((p) => R.map((row) => row[0]! * p[0] + row[1]! * p[1] + row[2]! * p[2]).map((v, k) => v + shift[k]!) as unknown as Vec3);
}

describe('special functions', () => {
  it('j_l matches closed forms for l = 0, 1, 2 on both sides of the series/recurrence switch', () => {
    for (const x of [0.01, 0.4, 0.99, 1.01, 3.7, 12.5]) {
      expect(sphericalBessel(0, x)).toBeCloseTo(Math.sin(x) / x, 12);
      expect(sphericalBessel(1, x)).toBeCloseTo(Math.sin(x) / x ** 2 - Math.cos(x) / x, 11);
      expect(sphericalBessel(2, x)).toBeCloseTo((3 / x ** 2 - 1) * (Math.sin(x) / x) - (3 * Math.cos(x)) / x ** 2, 10);
    }
    expect(sphericalBessel(3, 0)).toBe(0);
    expect(sphericalBessel(0, 0)).toBe(1);
  });

  it('series and recurrence agree for higher l at the switch', () => {
    for (const l of [3, 4, 5, 6]) {
      const x = 0.5 * l + 1;
      expect(sphericalBessel(l, x - 1e-9)).toBeCloseTo(sphericalBessel(l, x + 1e-9), 9);
    }
  });

  it('roots: z_0n = nπ, and tabulated first roots of j_1, j_2', () => {
    besselRoots(0, 6).forEach((z, k) => expect(z).toBeCloseTo((k + 1) * Math.PI, 10));
    expect(besselRoots(1, 1)[0]!).toBeCloseTo(4.493409457909, 9);
    expect(besselRoots(2, 1)[0]!).toBeCloseTo(5.763459196895, 9);
    for (let l = 0; l < 7; l++) for (const z of besselRoots(l, 6)) expect(Math.abs(sphericalBessel(l, z))).toBeLessThan(1e-12);
  });

  it('Legendre polynomials: P_2, P_3 closed forms', () => {
    for (const x of [-0.8, 0.1, 0.6]) {
      expect(legendre(2, x)).toBeCloseTo((3 * x * x - 1) / 2, 14);
      expect(legendre(3, x)).toBeCloseTo((5 * x ** 3 - 3 * x) / 2, 14);
    }
  });
});

describe('the paper’s bases (Eqs. 6–8)', () => {
  it('envelope (p = 6): u(0) = 1 and u, u′, u″ vanish at x = 1 (root of multiplicity 3)', () => {
    expect(envelope(0)).toBe(1);
    const u = (x: number) => 1 - 28 * x ** 6 + 48 * x ** 7 - 21 * x ** 8; // Eq. 8 with p = 6, unclipped
    const du = (x: number) => -168 * x ** 5 + 336 * x ** 6 - 168 * x ** 7;
    const d2u = (x: number) => -840 * x ** 4 + 2016 * x ** 5 - 1176 * x ** 6;
    expect(u(1)).toBe(0);
    expect(du(1)).toBe(0);
    expect(d2u(1)).toBe(0);
    expect(d2u(0.9)).not.toBe(0);
    expect(envelope(0.5)).toBeCloseTo(u(0.5), 14);
    expect(envelope(1.2)).toBe(0);
  });

  it('radial basis is orthonormal on [0, c] with weight d² (Eq. 7)', () => {
    const c = 5;
    const N = 4000;
    for (const [a, b] of [[1, 1], [2, 2], [1, 3], [2, 5]] as const) {
      let s = 0;
      for (let k = 0; k < N; k++) {
        const d = ((k + 0.5) / N) * c;
        s += rbfTilde(a, d, c) * rbfTilde(b, d, c) * d * d * (c / N);
      }
      expect(s).toBeCloseTo(a === b ? 1 : 0, 5);
    }
  });

  it('2D basis is orthonormal over the ball of radius c (Eq. 6; ∫ dφ gives 2π)', () => {
    const c = 5;
    const Nd = 600;
    const Na = 300;
    const inner = (l1: number, n1: number, l2: number, n2: number) => {
      let s = 0;
      for (let a = 0; a < Nd; a++) {
        const d = ((a + 0.5) / Nd) * c;
        for (let b = 0; b < Na; b++) {
          const al = ((b + 0.5) / Na) * Math.PI;
          s += sbfTilde(l1, n1, d, al, c) * sbfTilde(l2, n2, d, al, c) * d * d * Math.sin(al) * (c / Nd) * (Math.PI / Na);
        }
      }
      return 2 * Math.PI * s;
    };
    expect(inner(0, 1, 0, 1)).toBeCloseTo(1, 3);
    expect(inner(2, 3, 2, 3)).toBeCloseTo(1, 3);
    expect(inner(1, 2, 1, 3)).toBeCloseTo(0, 3);
    expect(inner(1, 2, 3, 2)).toBeCloseTo(0, 3);
  });

  it('every 2D basis function vanishes at the cutoff (the boundary condition Ψ(c) = 0)', () => {
    for (let l = 0; l < 7; l++) for (let n = 1; n <= 6; n++) expect(Math.abs(sbfTilde(l, n, 5 - 1e-9, 0.7, 5))).toBeLessThan(1e-6);
  });
});

describe('graph construction (Eq. 4)', () => {
  it('a bent triatomic: 4 directed edges, 2 triplets through the centre, both with the bond angle', () => {
    const t = (104.5 * Math.PI) / 180;
    const pos: Vec3[] = [
      [0, 0, 0],
      [1, 0, 0],
      [Math.cos(t), Math.sin(t), 0],
    ];
    const { edges, triplets } = graph(pos, 1.5); // outer atoms are 1.58 Å apart: no edge
    expect(edges).toHaveLength(4);
    expect(triplets).toHaveLength(2);
    for (const tr of triplets) {
      expect(edges[tr.kj]!.i).toBe(0);
      expect(tr.alpha).toBeCloseTo(t, 12);
    }
  });
});

describe('toy DimeNet', () => {
  const model = makeToyDimeNet({ rc: 5, elements: [1, 6, 8] });
  const Z = [6, 8, 1, 1, 6];
  const pos: Vec3[] = [
    [0, 0, 0],
    [1.2, 0.1, 0],
    [-0.6, 0.9, 0.2],
    [-0.5, -0.9, -0.3],
    [0.3, 0.2, 1.5],
  ];

  it('the energy is invariant under rotation, translation, inversion and permutation', () => {
    const e0 = forward(model, Z, pos).energy;
    expect(Math.abs(e0)).toBeGreaterThan(1e-3);
    expect(forward(model, Z, rotate(pos, [0.3, -1, 0.5], 1.1, [2, -3, 7])).energy).toBeCloseTo(e0, 10);
    expect(forward(model, Z, pos.map((p) => [-p[0], -p[1], -p[2]] as Vec3)).energy).toBeCloseTo(e0, 10);
    const perm = [3, 0, 4, 1, 2];
    expect(forward(model, perm.map((k) => Z[k]!), perm.map((k) => pos[k]!)).energy).toBeCloseTo(e0, 10);
  });

  it('the energy is smooth as an atom crosses the cutoff', () => {
    const at = (x: number) => forward(model, [6, 6], [[0, 0, 0], [x, 0, 0]]).energy;
    const far = at(6);
    expect(at(5 - 1e-6)).toBeCloseTo(far, 8);
    expect(at(5 + 1e-6)).toBeCloseTo(far, 8);
  });
});

describe('App. A, Fig. 6: a hexagon vs two triangles with the same bond length', () => {
  const { hexagon, triangles } = hexagonAndTriangles(D);
  const Z = Array(6).fill(6) as number[];

  it('below d√3 every atom has the same in-cutoff neighbourhood in both: SchNet features are identical at every depth', () => {
    for (const rc of [1.8, 2.2, hexagonSecondNeighbour(D) - 0.01]) {
      const m = makeToySchNet({ rc, elements: [6], blocks: 4 });
      const a = schnetForward(m, Z, hexagon);
      const b = schnetForward(m, Z, triangles);
      for (let t = 1; t <= 4; t++) for (let i = 0; i < 6; i++) expect(relDiff(a[t]![i]!, b[t]![i]!)).toBeLessThan(1e-12);
    }
  });

  it('above d√3 SchNet tells them apart', () => {
    const m = makeToySchNet({ rc: 2.8, elements: [6] });
    expect(relDiff(schnetForward(m, Z, hexagon)[1]![0]!, schnetForward(m, Z, triangles)[1]![0]!)).toBeGreaterThan(1e-3);
  });

  it('DimeNet separates them from its first interaction block at any cutoff above d (120° vs 60° angles)', () => {
    for (const rc of [1.8, 2.2, 2.5]) {
      const m = makeToyDimeNet({ rc, elements: [6] });
      const a = forward(m, Z, hexagon);
      const b = forward(m, Z, triangles);
      // The embedding block sees only distances: identical.
      expect(relDiff(a.atomFeat[0]![0]!, b.atomFeat[0]![0]!)).toBeLessThan(1e-12);
      expect(relDiff(a.atomFeat[1]![0]!, b.atomFeat[1]![0]!)).toBeGreaterThan(1e-5);
      expect(Math.abs(a.energy - b.energy)).toBeGreaterThan(1e-6);
    }
  });
});

describe('GemNet two-hop term on a torsion (H–O–O–H)', () => {
  it('real spherical harmonics are orthonormal on the sphere', () => {
    const Np = 200;
    const Na = 200;
    const inner = (l1: number, m1: number, l2: number, m2: number) => {
      let s = 0;
      for (let a = 0; a < Np; a++) {
        const th = ((a + 0.5) / Np) * Math.PI;
        for (let b = 0; b < Na; b++) {
          const ph = ((b + 0.5) / Na) * 2 * Math.PI;
          s += realY(l1, m1, th, ph) * realY(l2, m2, th, ph) * Math.sin(th) * (Math.PI / Np) * ((2 * Math.PI) / Na);
        }
      }
      return s;
    };
    expect(inner(2, 1, 2, 1)).toBeCloseTo(1, 3);
    expect(inner(3, -2, 3, -2)).toBeCloseTo(1, 3);
    expect(inner(3, 2, 3, -2)).toBeCloseTo(0, 3);
    expect(inner(2, 0, 4, 0)).toBeCloseTo(0, 3);
    expect(realY(4, 0, 0.3, 1)).toBeCloseTo(y0(4, 0.3), 13);
  });

  it('the dihedral of the peroxide chain is the angle that was set', () => {
    for (const t of [0, 37, 90, 111.5, 180]) {
      const p = peroxide(t);
      expect((dihedral(sub3(p[0]!, p[1]!), sub3(p[3]!, p[2]!), sub3(p[2]!, p[1]!)) * 180) / Math.PI).toBeCloseTo(t, 9);
    }
    const d = peroxideDistances(90);
    expect(d.hOfar).toBeGreaterThan(PEROXIDE.rOO);
  });

  const scanAt = (rc: number) => {
    const opts = { rc, elements: [1, 8], blocks: 3 };
    return torsionScan(makeToyDimeNet(opts), makeToyGemNetQ(opts), [0, 45, 90, 135, 180], 111);
  };

  it('graph of bonds only (1.45 Å < r_c < H···O): DimeNet is blind to the torsion at every depth, the quadruplet term is not', () => {
    const rc = 1.7;
    expect(peroxideDistances(0).hOfar).toBeGreaterThan(rc);
    const s = scanAt(rc);
    for (const v of s.dime) expect(v).toBeLessThan(1e-12);
    expect(Math.max(...s.gem)).toBeGreaterThan(1e-6);
  });

  it('once the 1–3 (H···O) edges are in the graph, DimeNet sees the torsion through the angle at O', () => {
    const s = scanAt(2.2);
    expect(Math.max(...s.dime)).toBeGreaterThan(1e-6);
  });

  it('below the O–O distance the molecule is two OH fragments: neither sees the torsion', () => {
    const s = scanAt(1.3);
    for (const v of [...s.dime, ...s.gem]) expect(v).toBeLessThan(1e-12);
  });

  it('the toy GemNet-Q energy stays invariant under rotation and mirror (unsigned dihedral)', () => {
    const m = makeToyGemNetQ({ rc: 3, elements: [1, 8] });
    const p = peroxide(70);
    const e0 = forward(m, [1, 8, 8, 1], p).energy;
    expect(forward(m, [1, 8, 8, 1], rotate(p, [1, 2, 3], 0.8, [1, 1, 1])).energy).toBeCloseTo(e0, 10);
    expect(forward(m, [1, 8, 8, 1], p.map((v) => [v[0], v[1], -v[2]] as Vec3)).energy).toBeCloseTo(e0, 10);
    expect(forward(m, [1, 8, 8, 1], p).quads.length).toBeGreaterThan(0);
  });
});

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

describe('interaction counts in crystals', () => {
  // fcc lattice, a = 3.615 Å (Cu), as a finite cluster for brute force and as a neighbour function.
  const a = 3.615;
  const basis: Vec3[] = [
    [0, 0, 0],
    [0, a / 2, a / 2],
    [a / 2, 0, a / 2],
    [a / 2, a / 2, 0],
  ];
  const lattice = (R: number): Vec3[] => {
    const out: Vec3[] = [];
    const n = Math.ceil(R / a) + 1;
    for (let x = -n; x <= n; x++)
      for (let y = -n; y <= n; y++)
        for (let z = -n; z <= n; z++)
          for (const b of basis) {
            const p: Vec3 = [x * a + b[0], y * a + b[1], z * a + b[2]];
            if (Math.hypot(...p) < R) out.push(p);
          }
    return out;
  };
  const neighbours = (rc: number) => lattice(rc).filter((p) => Math.hypot(...p) > 1e-9);

  it('fcc: 12 nearest neighbours at a/√2, 132 triplets per atom at r_c just above it', () => {
    const c = interactionCounts(neighbours, 2.7);
    expect(c.edges).toBe(12);
    expect(c.triplets).toBe(132);
  });

  it('the quadruplet count matches brute force over GemNet’s rule (c_int = c_emb)', () => {
    for (const rc of [2.7, 3.0]) {
      const cluster = lattice(3 * rc + 0.5);
      const centre = cluster.findIndex((p) => Math.hypot(...p) < 1e-9);
      const { edges } = graph(cluster, rc);
      const quads = quadruplets(cluster, edges).filter((q) => edges[q.ca]!.i === centre);
      expect(interactionCounts(neighbours, rc).quadruplets).toBe(quads.length);
    }
  });

  it('Cu at 5 Å: 42 neighbours, 1722 triplets; a 10 Å quadruplet cutoff gives about 9× the quadruplets (the page caption)', () => {
    const c5 = interactionCounts(neighbours, 5);
    const c510 = interactionCounts(neighbours, 5, 10);
    expect(c5.edges).toBe(42);
    expect(c5.triplets).toBe(1722);
    expect(c5.quadruplets).toBeGreaterThan(1e4);
    expect(c510.quadruplets / c5.quadruplets).toBeGreaterThan(8.5);
    expect(c510.quadruplets / c5.quadruplets).toBeLessThan(9.5);
  });

  it('GemNet-OC bounds: 30·29 triplets and 30·8·29 quadruplets per atom at most', () => {
    expect(gemnetOcBounds()).toEqual({ edges: 30, triplets: 870, quadruplets: 6960 });
  });
});
