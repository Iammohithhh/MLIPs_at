// Polar-plot surface of a real spherical harmonic: the point at direction r̂ is
// drawn at radius |Y_lm(r̂)|·scale, coloured by the sign of Y_lm.

import { realSH, type Vec3 } from '../physics/spherical-harmonics';

export interface LobeMesh {
  positions: Float32Array;
  /** +1 / −1 per vertex (sign of Y). */
  signs: Int8Array;
  indices: Uint32Array;
  /** Largest |Y| on the grid. */
  maxAbs: number;
}

export function lobeMesh(l: number, m: number, radius = 1.7, nTheta = 72, nPhi = 144): LobeMesh {
  const nv = (nTheta + 1) * (nPhi + 1);
  const values = new Float64Array(nv);
  const dirs: Vec3[] = [];
  let maxAbs = 0;
  for (let i = 0; i <= nTheta; i++) {
    const th = (Math.PI * i) / nTheta;
    for (let j = 0; j <= nPhi; j++) {
      const ph = (2 * Math.PI * j) / nPhi;
      const d: Vec3 = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const y = realSH(l, m, d);
      values[dirs.length] = y;
      dirs.push(d);
      maxAbs = Math.max(maxAbs, Math.abs(y));
    }
  }
  const s = radius / (maxAbs || 1);
  const positions = new Float32Array(nv * 3);
  const signs = new Int8Array(nv);
  dirs.forEach((d, k) => {
    const r = Math.abs(values[k]!) * s;
    positions[3 * k] = d[0] * r;
    positions[3 * k + 1] = d[1] * r;
    positions[3 * k + 2] = d[2] * r;
    signs[k] = values[k]! >= 0 ? 1 : -1;
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

// Polar plot of a band-limited signal f(r̂) = Σ x_lm Y_lm(r̂), l ≤ L, centred at `center`: the point
// at direction r̂ is drawn at radius |f(r̂)|·radius/max|f|. The harmonics on the grid are cached per L.

const gridCache = new Map<string, { dirs: Vec3[]; Y: Float64Array; indices: Uint32Array }>();

function shGrid(L: number, nTheta: number, nPhi: number) {
  const key = `${L}:${nTheta}:${nPhi}`;
  const hit = gridCache.get(key);
  if (hit) return hit;
  const nc = (L + 1) ** 2;
  const dirs: Vec3[] = [];
  for (let i = 0; i <= nTheta; i++) {
    const th = (Math.PI * i) / nTheta;
    for (let j = 0; j <= nPhi; j++) {
      const ph = (2 * Math.PI * j) / nPhi;
      dirs.push([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]);
    }
  }
  const Y = new Float64Array(dirs.length * nc);
  dirs.forEach((d, k) => {
    for (let l = 0; l <= L; l++) for (let m = -l; m <= l; m++) Y[k * nc + l * l + l + m] = realSH(l, m, d);
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
  const g = { dirs, Y, indices };
  gridCache.set(key, g);
  return g;
}

export function signalMesh(x: ArrayLike<number>, L: number, center: Vec3, radius: number, nTheta = 40, nPhi = 80): LobeMesh {
  const { dirs, Y, indices } = shGrid(L, nTheta, nPhi);
  const nc = (L + 1) ** 2;
  const values = new Float64Array(dirs.length);
  let maxAbs = 0;
  for (let k = 0; k < dirs.length; k++) {
    let s = 0;
    for (let i = 0; i < nc; i++) s += x[i]! * Y[k * nc + i]!;
    values[k] = s;
    maxAbs = Math.max(maxAbs, Math.abs(s));
  }
  const sc = radius / (maxAbs || 1);
  const positions = new Float32Array(dirs.length * 3);
  const signs = new Int8Array(dirs.length);
  dirs.forEach((d, k) => {
    const r = Math.abs(values[k]!) * sc;
    positions[3 * k] = center[0] + d[0] * r;
    positions[3 * k + 1] = center[1] + d[1] * r;
    positions[3 * k + 2] = center[2] + d[2] * r;
    signs[k] = values[k]! >= 0 ? 1 : -1;
  });
  return { positions, signs, indices, maxAbs };
}
