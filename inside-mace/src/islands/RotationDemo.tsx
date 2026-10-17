// "What changes when the coordinate frame changes?" — a real comparison: the same structure (positions
// AND cell) rotated by R and evaluated again by the model. Camera rotation, by contrast, changes only the view.
import { fmtD } from '../lib/format';
import { useEffect, useState } from 'react';
import type { Manifest, Vec3 } from '../lib/experiment';
import { loadChunk, type FrameChunk } from './FeatureInspector';

/** True minus sign for displayed numbers (U+2212), as in the rest of the instrument. */
const mn = (t: string) => t.replace(/^-/, '−');
const f3 = (v: readonly number[]) => `(${v.map((x) => (Math.abs(x) < 5e-5 ? '0.0000' : mn(x.toFixed(4)))).join(', ')})`;
const apply = (R: readonly (readonly number[])[], v: readonly number[]): Vec3 =>
  [0, 1, 2].map((i) => R[i]![0]! * v[0]! + R[i]![1]! * v[1]! + R[i]![2]! * v[2]!) as Vec3;

/**
 * A vector seen looking down the rotation axis n: a rotation by θ about n turns the perpendicular part of
 * every vector by exactly θ in this plane and leaves the part along n unchanged, so the glyph shows the
 * transformation without any arbitrary projection. Teal: original run; dashed: R applied to it;
 * amber: the separately evaluated rotated run.
 */
function RotGlyph({ title, unit, v, Rv, vRot, axis, angle }: { title: string; unit: string; v: Vec3; Rv: Vec3; vRot: Vec3; axis: Vec3; angle: number }) {
  const n = Math.hypot(...axis);
  const a: Vec3 = [axis[0] / n, axis[1] / n, axis[2] / n];
  // e1 ⊥ a (from a × z, or a × x if a ∥ z), e2 = a × e1
  const cross = (p: Vec3, q: Vec3): Vec3 => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  let e1 = cross(a, [0, 0, 1]);
  if (Math.hypot(...e1) < 1e-6) e1 = cross(a, [1, 0, 0]);
  const l1 = Math.hypot(...e1);
  e1 = [e1[0] / l1, e1[1] / l1, e1[2] / l1];
  const e2 = cross(a, e1);
  const dot = (p: Vec3, q: Vec3) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const proj = (u: Vec3) => [dot(u, e1), dot(u, e2)] as const;
  const P = [proj(v), proj(Rv), proj(vRot)];
  const r = Math.max(...P.map(([x, y]) => Math.hypot(x, y))) || 1;
  const S = 120;
  const c = S / 2;
  const k = (S / 2 - 14) / r;
  const pt = ([x, y]: readonly [number, number]) => [c + x * k, c - y * k] as const;
  const [o, rv, vr] = P.map(pt) as [readonly [number, number], readonly [number, number], readonly [number, number]];
  const ang0 = Math.atan2(P[0]![1], P[0]![0]);
  const arcR = 22;
  const arc = (t: number) => `${c + arcR * Math.cos(t)} ${c - arcR * Math.sin(t)}`;
  const ang1 = ang0 + (angle * Math.PI) / 180;
  return (
    <figure className="rot__glyph">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`${title}: seen along the rotation axis, the vector turns by ${angle} degrees`}>
        <circle cx={c} cy={c} r={S / 2 - 10} className="rot__ring" />
        <path d={`M ${arc(ang0)} A ${arcR} ${arcR} 0 0 0 ${arc(ang1)}`} className="rot__arc" />
        <line x1={c} y1={c} x2={o[0]} y2={o[1]} className="rot__v0" />
        <line x1={c} y1={c} x2={vr[0]} y2={vr[1]} className="rot__v1" />
        <line x1={c} y1={c} x2={rv[0]} y2={rv[1]} className="rot__vR" />
        <circle cx={c} cy={c} r={2} className="rot__axis" />
      </svg>
      <figcaption>
        {title}: turns by {angle}° about the axis; the part along the axis stays {mn(dot(v, a).toFixed(3))} → {mn(dot(vRot, a).toFixed(3))} {unit}
      </figcaption>
    </figure>
  );
}

export function RotationDemo({ m, channel }: { m: Manifest; channel: number }) {
  const rot = m.checks.rotation;
  const run = rot.runs.find((r) => r.frame !== m.displacement.baselineIndex) ?? rot.runs[0]!;
  const [chunk, setChunk] = useState<FrameChunk | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    loadChunk(m.id, run.frame).then(setChunk, (e: unknown) => setErr(String(e)));
  }, [m.id, run.frame]);
  const atom = m.displacement.atom;
  const f = m.frames[run.frame];
  if (!f || f.status !== 'ok') return null;
  const F = f.forces[atom]!;
  const RF = apply(rot.R, F);
  const Frot = run.forces[atom]!;
  const row = m.local.indexOf(atom);
  const h1 = chunk?.layers[0]?.h1?.[row]?.[channel] ?? null;
  const h1rot = run.local.layers[0]?.h1?.[row]?.[channel] ?? null;
  const h0 = chunk?.layers[0]?.h0[row]?.[channel] ?? null;
  const h0rot = run.local.layers[0]?.h0[row]?.[channel] ?? null;
  return (
    <section className="rot" aria-labelledby="rot-title">
      <h3 id="rot-title">What changes when the coordinate frame changes?</h3>
      <p className="rot__lede">
        Turning the camera only changes your view. Here the <em>model input</em> itself was rotated — every position and the cell, by{' '}
        {rot.angleDeg}° about ({rot.axis.join(', ')}) — and the checkpoint was run again (frame d = {fmtD(f.d)} Å).{' '}
        {rot.runs.length > 1 &&
          `Rotated runs are stored for ${rot.runs.map((r) => `d = ${mn(m.displacement.values[r.frame]!.toFixed(3))} Å`).join(' and ')}; the displaced one is shown because its force and vector features are non-zero, so their rotation is visible.`}{' '}
        <span className="label">Precomputed</span>
      </p>
      <dl className="rot__grid mono">
        <dt>Total energy</dt>
        <dd>
          {mn(f.energy.toFixed(6))} eV → {mn(run.energy.toFixed(6))} eV; |ΔE| = {run.energyAbsDiff.toExponential(1)} eV <b>invariant</b>
        </dd>
        <dt>Force on moved atom</dt>
        <dd>
          F = {f3(F)} · R F = {f3(RF)} · rotated run {f3(Frot)} eV/Å; largest deviation over all atoms {run.forceEquivarianceMaxAbs.toExponential(1)}{' '}
          eV/Å <b>rotates with the input</b>
        </dd>
        <dt>Layer 1 scalar, ch {channel}</dt>
        <dd>
          {h0 === null || h0rot === null ? (err ? 'not available' : '…') : `${mn(h0.toPrecision(5))} → ${mn(h0rot.toPrecision(5))}`}; max over all atoms and channels{' '}
          {Math.max(...run.h0InvarianceMaxAbs).toExponential(1)} <b>invariant</b>
        </dd>
        <dt>Layer 1 vector, ch {channel}</dt>
        <dd>
          {h1 && h1rot ? `h = ${f3(h1)} · R h = ${f3(apply(rot.R, h1))} · rotated run ${f3(h1rot)}` : err ? 'not available' : '…'}; max deviation{' '}
          {(run.h1EquivarianceMaxAbs[0] ?? NaN).toExponential(1)} <b>rotates like a vector</b>
        </dd>
      </dl>
      <div className="rot__glyphs">
        <RotGlyph title="Force on the moved atom" unit="eV/Å" v={[F[0], F[1], F[2]]} Rv={RF} vRot={[Frot[0], Frot[1], Frot[2]]} axis={rot.axis} angle={rot.angleDeg} />
        {h1 && h1rot && <RotGlyph title={`Layer 1 vector, ch ${channel}`} unit="model units" v={[h1[0]!, h1[1]!, h1[2]!]} Rv={apply(rot.R, h1)} vRot={[h1rot[0]!, h1rot[1]!, h1rot[2]!]} axis={rot.axis} angle={rot.angleDeg} />}
        <p className="rot__key">
          Seen down the rotation axis (dot). <span className="k0">teal</span>: original run · <span className="k1">amber</span>: rotated-input run ·{' '}
          <span className="kR">dashed</span>: R applied to the original. The amber and dashed lines coincide because the model is equivariant.
        </p>
      </div>
      <p className="rot__note">
        Components are compared after applying R to the original — comparing raw x, y, z components across the two frames would be meaningless.
        Deviations are float64 round-off. The l = 1 features use e3nn’s (x, y, z) component order.
      </p>
    </section>
  );
}
