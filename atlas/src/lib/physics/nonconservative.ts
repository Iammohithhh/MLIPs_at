// Failure Atlas — non-conservative forces. A TOY MODEL in two dimensions, reduced
// units (mass 1, energy and length dimensionless): one particle in a smooth
// anharmonic well, integrated with velocity Verlet under three force "models":
//
//   exact         F = −∇V
//   gradient      F = −∇(V + δV)          an error that is still a gradient
//   direct        F = −∇V + δF, ∇×δF ≠ 0  an error with a curl (a direct-force head)
//
// Both errors are scaled to the same RMS force error over the disc r ≤ R_DISC, so
// a force-error table cannot tell them apart. Pure and tested (nonconservative.test.ts).

export type Vec2 = [number, number];
export type ForceModel = 'exact' | 'gradient' | 'direct';

export const MODELS: { id: ForceModel; label: string; short: string }[] = [
  { id: 'exact', label: 'Exact gradient', short: 'F = −∇V' },
  { id: 'gradient', label: 'Error with no curl', short: 'F = −∇(V + δV)' },
  { id: 'direct', label: 'Error with a curl', short: 'F = −∇V + δF' },
];

/** Well parameters: V = ½(x² + κy²) + β(x² + y²)². */
export const WELL = { kappa: 1, beta: 0.05 } as const;
/** Radius of the disc over which force errors are measured. */
export const R_DISC = 2;
/** Wavenumber of the gradient error δV ∝ cos(qx)cos(qy). */
const Q = 2;

export function potential([x, y]: Vec2): number {
  const r2 = x * x + y * y;
  return 0.5 * (x * x + WELL.kappa * y * y) + WELL.beta * r2 * r2;
}

export function gradV([x, y]: Vec2): Vec2 {
  const r2 = x * x + y * y;
  return [x + 4 * WELL.beta * r2 * x, WELL.kappa * y + 4 * WELL.beta * r2 * y];
}

/** Unit-amplitude conservative error: δV₁ = cos(qx)cos(qy)/q. */
export function deltaV1([x, y]: Vec2): number {
  return (Math.cos(Q * x) * Math.cos(Q * y)) / Q;
}
/** Its force −∇δV₁. */
export function deltaF1Gradient([x, y]: Vec2): Vec2 {
  return [Math.sin(Q * x) * Math.cos(Q * y), Math.cos(Q * x) * Math.sin(Q * y)];
}
/** Unit-amplitude rotational error δF₁ = (−y, x): curl 2 everywhere, divergence 0. */
export function deltaF1Curl([x, y]: Vec2): Vec2 {
  return [-y, x];
}

/** RMS of |δF₁| over the disc r ≤ R (midpoint polar quadrature). */
export function rmsOverDisc(f: (p: Vec2) => Vec2, R = R_DISC, nr = 120, nt = 240): number {
  let s = 0;
  let w = 0;
  for (let i = 0; i < nr; i++) {
    const r = ((i + 0.5) * R) / nr;
    for (let j = 0; j < nt; j++) {
      const t = ((j + 0.5) * 2 * Math.PI) / nt;
      const v = f([r * Math.cos(t), r * Math.sin(t)]);
      s += r * (v[0] * v[0] + v[1] * v[1]);
      w += r;
    }
  }
  return Math.sqrt(s / w);
}

/** Exact RMS of the curl error over the disc: √(∫r²·r dr / ∫r dr) = R/√2. */
export const RMS_CURL_UNIT = R_DISC / Math.SQRT2;
let rmsGradUnit: number | undefined;
export function rmsGradientUnit(): number {
  rmsGradUnit ??= rmsOverDisc(deltaF1Gradient);
  return rmsGradUnit;
}

/** Amplitude that gives each erroneous model an RMS force error σ on the disc. */
export function amplitudeFor(model: ForceModel, sigma: number): number {
  if (model === 'exact') return 0;
  return model === 'gradient' ? sigma / rmsGradientUnit() : sigma / RMS_CURL_UNIT;
}

export function force(model: ForceModel, p: Vec2, amp: number): Vec2 {
  const g = gradV(p);
  if (model === 'exact') return [-g[0], -g[1]];
  const d = model === 'gradient' ? deltaF1Gradient(p) : deltaF1Curl(p);
  return [-g[0] + amp * d[0], -g[1] + amp * d[1]];
}

/**
 * The energy a model conserves, if it has one: V for the exact model, V + δV for the
 * gradient-error model. The direct model has none; its trajectory is scored on V.
 */
export function modelPotential(model: ForceModel, p: Vec2, amp: number): number {
  return model === 'gradient' ? potential(p) + amp * deltaV1(p) : potential(p);
}

/** Circulation ∮F·dl counter-clockwise round the circle of radius R at the origin. */
export function circulation(model: ForceModel, amp: number, R: number, n = 512): number {
  let s = 0;
  for (let k = 0; k < n; k++) {
    const t = (2 * Math.PI * k) / n;
    const p: Vec2 = [R * Math.cos(t), R * Math.sin(t)];
    const f = force(model, p, amp);
    s += f[0] * -R * Math.sin(t) + f[1] * R * Math.cos(t);
  }
  return (s * 2 * Math.PI) / n;
}

/** ∮δF·dl for the curl error round a circle of radius R: amp · curl · area = 2πR²·amp. */
export const circulationExact = (amp: number, R: number) => 2 * Math.PI * R * R * amp;

export interface Trajectory {
  t: Float64Array;
  x: Float64Array;
  y: Float64Array;
  /** Kinetic + reference potential V (what a reference calculation would assign). */
  eRef: Float64Array;
  /** Kinetic + the model's own potential (equal to eRef for exact and direct). */
  eModel: Float64Array;
  /** Angular momentum x·v_y − y·v_x. */
  L: Float64Array;
}

export interface RunOptions {
  model: ForceModel;
  sigma: number;
  dt: number;
  tEnd: number;
  x0: Vec2;
  v0: Vec2;
  /** Store every k-th step. */
  every?: number;
}

/** Velocity Verlet (Störmer–Verlet) with unit mass. */
export function run({ model, sigma, dt, tEnd, x0, v0, every = 1 }: RunOptions): Trajectory {
  const amp = amplitudeFor(model, sigma);
  const steps = Math.round(tEnd / dt);
  const n = Math.floor(steps / every) + 1;
  const out: Trajectory = {
    t: new Float64Array(n),
    x: new Float64Array(n),
    y: new Float64Array(n),
    eRef: new Float64Array(n),
    eModel: new Float64Array(n),
    L: new Float64Array(n),
  };
  let p: Vec2 = [x0[0], x0[1]];
  let v: Vec2 = [v0[0], v0[1]];
  let f = force(model, p, amp);
  const store = (k: number, i: number) => {
    const ke = 0.5 * (v[0] * v[0] + v[1] * v[1]);
    out.t[i] = k * dt;
    out.x[i] = p[0];
    out.y[i] = p[1];
    out.eRef[i] = ke + potential(p);
    out.eModel[i] = ke + modelPotential(model, p, amp);
    out.L[i] = p[0] * v[1] - p[1] * v[0];
  };
  store(0, 0);
  for (let k = 1; k <= steps; k++) {
    v = [v[0] + 0.5 * dt * f[0], v[1] + 0.5 * dt * f[1]];
    p = [p[0] + dt * v[0], p[1] + dt * v[1]];
    f = force(model, p, amp);
    v = [v[0] + 0.5 * dt * f[0], v[1] + 0.5 * dt * f[1]];
    if (k % every === 0) store(k, k / every);
  }
  return out;
}

/** Default launch: from (1.2, 0) with speed 0.9 counter-clockwise (+1) or clockwise (−1). */
export function launch(direction: 1 | -1): { x0: Vec2; v0: Vec2 } {
  return { x0: [1.2, 0], v0: [0.25, 0.9 * direction] };
}

/** Drift of a stored energy series: last value minus first, and the largest excursion. */
export function drift(e: Float64Array): { net: number; maxAbs: number } {
  let maxAbs = 0;
  for (const v of e) maxAbs = Math.max(maxAbs, Math.abs(v - e[0]!));
  return { net: e[e.length - 1]! - e[0]!, maxAbs };
}

/** The well z = V(x, y) on a polar grid out to rMax (a circular rim), for the 3D scene. */
export function wellMesh(rMax: number, nr: number, nt: number): { positions: Float32Array; heights: Float32Array; indices: Uint32Array } {
  // Vertex 0 is the centre; ring i (1..nr) at radius i·rMax/nr has nt vertices.
  const count = 1 + nr * nt;
  const positions = new Float32Array(count * 3);
  const heights = new Float32Array(count);
  heights[0] = potential([0, 0]);
  positions.set([0, 0, heights[0]], 0);
  for (let i = 1; i <= nr; i++)
    for (let j = 0; j < nt; j++) {
      const r = (i * rMax) / nr;
      const t = (2 * Math.PI * j) / nt;
      const p: Vec2 = [r * Math.cos(t), r * Math.sin(t)];
      const k = 1 + (i - 1) * nt + j;
      heights[k] = potential(p);
      positions.set([p[0], p[1], heights[k]!], 3 * k);
    }
  const idx: number[] = [];
  const v = (i: number, j: number) => 1 + (i - 1) * nt + (j % nt);
  for (let j = 0; j < nt; j++) idx.push(0, v(1, j), v(1, j + 1));
  for (let i = 1; i < nr; i++)
    for (let j = 0; j < nt; j++) idx.push(v(i, j), v(i + 1, j), v(i, j + 1), v(i, j + 1), v(i + 1, j), v(i + 1, j + 1));
  return { positions, heights, indices: new Uint32Array(idx) };
}
