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
