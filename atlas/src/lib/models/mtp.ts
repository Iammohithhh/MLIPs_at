// Moment tensor potentials (Shapeev 2016; MLIP package, Novikov et al. 2021).
//
// Moments M_{μ,ν} = Σ_j f_μ(r_j) r̂_j ⊗ … ⊗ r̂_j (ν copies) of an atom's neighbourhood, contracted to
// scalars B_α. A basis function is a loopless multigraph: one node per moment, one edge per pair of
// contracted indices, so a node's degree is its ν. Disconnected graphs are products. The level of a
// moment is 2 + 4μ + ν and of a contraction the sum over its moments.
//
// The unit vectors follow mlip-2's mtp.cpp (it divides x^a y^b z^c by r^ν); the papers write r^⊗ν,
// which differs only by r^ν in the radial function.

export type Vec3 = [number, number, number];

export const level = (mu: number, nu: number) => 2 + 4 * mu + nu;

/** mlip-2's RBChebyshev: Q^β(r) = T_β(ξ)(r − R_cut)², ξ = (2r − R_min − R_cut)/(R_cut − R_min); 0 beyond R_cut. */
export function chebyshevRadial(r: number, rmin: number, rcut: number, size: number): Float64Array {
  const q = new Float64Array(size);
  if (r >= rcut) return q;
  const ksi = (2 * r - (rmin + rcut)) / (rcut - rmin);
  const env = (r - rcut) * (r - rcut);
  q[0] = env;
  if (size > 1) q[1] = ksi * env;
  for (let b = 2; b < size; b++) q[b] = 2 * ksi * q[b - 1]! - q[b - 2]!;
  return q;
}

export interface Radial {
  rmin: number;
  rcut: number;
}

/** Linear (2016-style) MTP: radial function μ is Q^μ itself, not a learned combination. */
export const radialValue = (r: number, mu: number, o: Radial) => chebyshevRadial(r, o.rmin, o.rcut, mu + 1)[mu]!;

/** Full symmetric tensor Σ_j f_μ(r_j) r̂_j^⊗ν, flattened with index Σ_k i_k 3^k (3^ν entries). */
export function momentTensor(env: Vec3[], mu: number, nu: number, o: Radial): Float64Array {
  const n = 3 ** nu;
  const out = new Float64Array(n);
  for (const v of env) {
    const r = Math.hypot(v[0], v[1], v[2]);
    if (r >= o.rcut || r === 0) continue;
    const f = radialValue(r, mu, o);
    const u = [v[0] / r, v[1] / r, v[2] / r];
    for (let k = 0; k < n; k++) {
      let p = f;
      let t = k;
      for (let s = 0; s < nu; s++) {
        p *= u[t % 3]!;
        t = (t / 3) | 0;
      }
      out[k]! += p;
    }
  }
  return out;
}

/** The tensor read as a polynomial on the sphere: M[n̂, …, n̂] = Σ_j f_μ(r_j)(r̂_j · n̂)^ν. */
export function momentOnSphere(env: Vec3[], mu: number, nu: number, o: Radial, n: Vec3): number {
  let s = 0;
  for (const v of env) {
    const r = Math.hypot(v[0], v[1], v[2]);
    if (r >= o.rcut || r === 0) continue;
    s += radialValue(r, mu, o) * ((v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) / r) ** nu;
  }
  return s;
}

// ---------------------------------------------------------------- basis enumeration

export interface BasisFunction {
  /** μ of each moment (node). */
  mu: number[];
  /** ν of each moment = its degree in the graph. */
  nu: number[];
  /** Upper triangle of the edge-multiplicity matrix, row by row: (0,1), (0,2), …, (k−2,k−1). */
  adj: number[];
  level: number;
  /** Canonical label, identical to the Python pipeline's. */
  key: string;
}

export interface BasisLimits {
  /** Most moments in one contraction (mlip-2's templates: 4). */
  maxMoments: number;
  muMax: number;
  nuMax: number;
}

/** Limits that reproduce the number of basis functions in every untrained template shipped with mlip-2. */
export const MLIP2_LIMITS: BasisLimits = { maxMoments: 4, muMax: 5, nuMax: 10 };

const pairsOf = (k: number) => {
  const p: [number, number][] = [];
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) p.push([i, j]);
  return p;
};

function permutations(k: number): number[][] {
  if (k === 0) return [[]];
  const out: number[][] = [];
  for (const p of permutations(k - 1)) for (let s = 0; s <= p.length; s++) out.push([...p.slice(0, s), k - 1, ...p.slice(s)]);
  return out;
}

const cmp = (a: number[], b: number[]) => {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i]! - b[i]!;
  return 0;
};

/** Lexicographically smallest (μ list, adjacency list) over relabellings of the nodes. */
export function canonical(mu: number[], adj: number[]): { mu: number[]; adj: number[] } {
  const k = mu.length;
  const pairs = pairsOf(k);
  const A = (i: number, j: number) => (i === j ? 0 : adj[pairs.findIndex(([a, b]) => a === Math.min(i, j) && b === Math.max(i, j))]!);
  let best: { mu: number[]; adj: number[] } | null = null;
  for (const p of permutations(k)) {
    const m = p.map((q) => mu[q]!);
    const a = pairs.map(([i, j]) => A(p[i]!, p[j]!));
    if (!best || cmp(m, best.mu) < 0 || (cmp(m, best.mu) === 0 && cmp(a, best.adj) < 0)) best = { mu: m, adj: a };
  }
  return best!;
}

export const basisKey = (mu: number[], adj: number[]) => `${mu.length}|${mu.join(',')}|${adj.join(',')}`;

function compositions(total: number, parts: number, out: number[][], cur: number[] = []): void {
  if (cur.length === parts - 1) {
    out.push([...cur, total]);
    return;
  }
  for (let c = 0; c <= total; c++) compositions(total - c, parts, out, [...cur, c]);
}

/** Every distinct scalar contraction with level ≤ levMax, sorted by level then key. */
export function enumerateBasis(levMax: number, lim: BasisLimits = MLIP2_LIMITS): BasisFunction[] {
  const seen = new Map<string, BasisFunction>();
  for (let k = 1; k <= lim.maxMoments; k++) {
    const pairs = pairsOf(k);
    for (let E = 0; 2 * k + 2 * E <= levMax; E++) {
      const adjs: number[][] = [];
      if (pairs.length === 0) {
        if (E > 0) continue;
        adjs.push([]);
      } else compositions(E, pairs.length, adjs);
      const muBudget = Math.floor((levMax - 2 * k - 2 * E) / 4);
      for (const adj of adjs) {
        const nu = Array.from({ length: k }, (_, i) => pairs.reduce((s, [a, b], e) => s + (a === i || b === i ? adj[e]! : 0), 0));
        if (Math.max(...nu) > lim.nuMax) continue;
        const mus: number[][] = [];
        const rec = (cur: number[], left: number) => {
          if (cur.length === k) return void mus.push(cur);
          for (let m = 0; m <= Math.min(left, lim.muMax); m++) rec([...cur, m], left - m);
        };
        rec([], muBudget);
        for (const mu of mus) {
          const c = canonical(mu, adj);
          const key = basisKey(c.mu, c.adj);
          if (seen.has(key)) continue;
          const cnu = Array.from({ length: k }, (_, i) => pairs.reduce((s, [a, b], e) => s + (a === i || b === i ? c.adj[e]! : 0), 0));
          const lev = c.mu.reduce((s, m, i) => s + level(m, cnu[i]!), 0);
          seen.set(key, { mu: c.mu, nu: cnu, adj: c.adj, level: lev, key });
        }
      }
    }
  }
  return [...seen.values()].sort((a, b) => a.level - b.level || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

/** Distinct moments used by a basis set, as "μ,ν" strings. */
export function momentsUsed(basis: BasisFunction[]): Array<{ mu: number; nu: number }> {
  const s = new Map<string, { mu: number; nu: number }>();
  for (const b of basis) b.mu.forEach((m, i) => s.set(`${m},${b.nu[i]}`, { mu: m, nu: b.nu[i]! }));
  return [...s.values()].sort((a, b) => a.mu - b.mu || a.nu - b.nu);
}

/** Independent components of the moments a basis uses, (ν+1)(ν+2)/2 each: mlip-2's alpha_index_basic_count. */
export const momentComponents = (basis: BasisFunction[]) => momentsUsed(basis).reduce((s, m) => s + ((m.nu + 1) * (m.nu + 2)) / 2, 0);

/** Number of distinct radial functions a basis needs: mlip-2's radial_funcs_count. */
export const radialFunctionsUsed = (basis: BasisFunction[]) => Math.max(...basis.flatMap((b) => b.mu)) + 1;

/** Largest number of moments in one contraction, so body order − 1. */
export const maxMomentsUsed = (basis: BasisFunction[]) => Math.max(...basis.map((b) => b.mu.length));

// ---------------------------------------------------------------- evaluation

export type MomentCache = Map<string, Float64Array>;

export function moments(env: Vec3[], basis: BasisFunction[], o: Radial): MomentCache {
  const c: MomentCache = new Map();
  for (const { mu, nu } of momentsUsed(basis)) c.set(`${mu},${nu}`, momentTensor(env, mu, nu, o));
  return c;
}

/** Full contraction of the graph's tensors: a sum over 3^E index assignments, one index per edge copy. */
export function contract(b: BasisFunction, M: MomentCache): number {
  const k = b.mu.length;
  const pairs = pairsOf(k);
  const slots: number[][] = Array.from({ length: k }, () => []);
  let E = 0;
  pairs.forEach(([i, j], e) => {
    for (let c = 0; c < b.adj[e]!; c++) {
      slots[i]!.push(E);
      slots[j]!.push(E);
      E++;
    }
  });
  const T = b.mu.map((m, i) => M.get(`${m},${b.nu[i]}`)!);
  const idx = new Int32Array(E);
  let sum = 0;
  const total = 3 ** E;
  for (let a = 0; a < total; a++) {
    let t = a;
    for (let e = 0; e < E; e++) {
      idx[e] = t % 3;
      t = (t / 3) | 0;
    }
    let p = 1;
    for (let i = 0; i < k; i++) {
      let f = 0;
      let w = 1;
      for (const s of slots[i]!) {
        f += idx[s]! * w;
        w *= 3;
      }
      p *= T[i]![f]!;
      if (p === 0) break;
    }
    sum += p;
  }
  return sum;
}

export function basisValues(env: Vec3[], basis: BasisFunction[], o: Radial): Float64Array {
  const M = moments(env, basis, o);
  return Float64Array.from(basis, (b) => contract(b, M));
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => String(n).split('').map((d) => SUB[Number(d)]).join('');
export const momentName = (mu: number, nu: number) => `M${sub(mu)},${sub(nu)}`;

const SUP_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const op = (m: number) => (m === 1 ? '·' : m === 2 ? ':' : `·${String(m).split('').map((d) => SUP_DIGITS[Number(d)]).join('')}`);

/** Readable form, e.g. "M₀,₂ : M₀,₂", "M₀,₀ (M₀,₁ · M₀,₁)" or "M₀,₁ · M₀,₂ · M₀,₁"; other graphs list their edges. */
export function describe(b: BasisFunction): string {
  const k = b.mu.length;
  const pairs = pairsOf(k);
  const name = (i: number) => momentName(b.mu[i]!, b.nu[i]!);
  // connected components
  const comp = Array.from({ length: k }, (_, i) => i);
  const find = (i: number): number => (comp[i] === i ? i : (comp[i] = find(comp[i]!)));
  pairs.forEach(([i, j], e) => {
    if (b.adj[e]) comp[find(i)] = find(j);
  });
  const groups = new Map<number, number[]>();
  for (let i = 0; i < k; i++) groups.set(find(i), [...(groups.get(find(i)) ?? []), i]);
  const parts = [...groups.values()].map((g) => {
    if (g.length === 1) return name(g[0]!);
    const edges = pairs.map((p, e) => ({ p, m: b.adj[e]! })).filter((x) => x.m && g.includes(x.p[0]));
    // A chain (path graph) reads left to right with one operator per link: · one index, : two, ·ⁿ more.
    const links = (i: number) => edges.filter((x) => x.p.includes(i));
    const isChain = edges.length === g.length - 1 && g.every((i) => links(i).length <= 2);
    if (isChain) {
      let cur = g.find((i) => links(i).length === 1)!;
      let out = name(cur);
      const used = new Set<number>([cur]);
      for (let s = 1; s < g.length; s++) {
        const e = links(cur).find((x) => !used.has(x.p[0] === cur ? x.p[1] : x.p[0]))!;
        const next = e.p[0] === cur ? e.p[1] : e.p[0];
        out += ` ${op(e.m)} ${name(next)}`;
        used.add(next);
        cur = next;
      }
      return out;
    }
    // A ring of matrices joined by single indices is the trace of their product.
    if (edges.length === g.length && edges.every((x) => x.m === 1) && g.every((i) => b.nu[i] === 2)) return `tr(${g.map(name).join(' ')})`;
    const at = (i: number) => g.indexOf(i) + 1;
    return `{${g.map(name).join(', ')}: ${edges.map((x) => `${at(x.p[0])}–${at(x.p[1])}${x.m > 1 ? ` ×${x.m}` : ''}`).join(', ')}}`;
  });
  return parts.map((p) => (parts.length > 1 && p.includes(' ') && !p.startsWith('{') ? `(${p})` : p)).join(' ');
}

// ---------------------------------------------------------------- scenes

export interface MomentMesh {
  positions: Float32Array;
  /** +1 / −1 per vertex: sign of M[n̂, …, n̂]. */
  signs: Int8Array;
  indices: Uint32Array;
  /** Largest |M[n̂, …, n̂]| on the grid. */
  maxAbs: number;
}

/**
 * Polar plot of a moment tensor: the vertex in direction n̂ sits at radius |M[n̂,…,n̂]| · radius / max. A symmetric
 * tensor of rank ν is fixed by this degree-ν polynomial on the sphere, so the surface shows the whole tensor.
 */
export function momentMesh(env: Vec3[], mu: number, nu: number, o: Radial, radius = 1.4, nTheta = 48, nPhi = 96): MomentMesh {
  const nv = (nTheta + 1) * (nPhi + 1);
  const vals = new Float64Array(nv);
  const dirs: Vec3[] = [];
  let maxAbs = 0;
  for (let i = 0; i <= nTheta; i++) {
    const th = (Math.PI * i) / nTheta;
    for (let j = 0; j <= nPhi; j++) {
      const ph = (2 * Math.PI * j) / nPhi;
      const d: Vec3 = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const v = momentOnSphere(env, mu, nu, o, d);
      vals[dirs.length] = v;
      dirs.push(d);
      maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  const s = maxAbs > 0 ? radius / maxAbs : 0;
  const positions = new Float32Array(nv * 3);
  const signs = new Int8Array(nv);
  dirs.forEach((d, k) => {
    const r = Math.abs(vals[k]!) * s;
    positions.set([d[0] * r, d[1] * r, d[2] * r], 3 * k);
    signs[k] = vals[k]! >= 0 ? 1 : -1;
  });
  const indices = new Uint32Array(nTheta * nPhi * 6);
  let t = 0;
  for (let i = 0; i < nTheta; i++)
    for (let j = 0; j < nPhi; j++) {
      const a = i * (nPhi + 1) + j;
      const b = a + nPhi + 1;
      indices.set([a, b, a + 1, a + 1, b, b + 1], t);
      t += 6;
    }
  return { positions, signs, indices, maxAbs };
}

/** Frobenius norm of a moment tensor. */
export const tensorNorm = (t: Float64Array) => Math.sqrt(t.reduce((s, x) => s + x * x, 0));
