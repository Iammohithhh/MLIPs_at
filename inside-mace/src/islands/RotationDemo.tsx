// "What changes when the coordinate frame changes?" — a real comparison: the same structure (positions
// AND cell) rotated by R and evaluated again by the model. Camera rotation, by contrast, changes only the view.
import { useEffect, useState } from 'react';
import type { Manifest, Vec3 } from '../lib/experiment';
import { loadChunk, type FrameChunk } from './FeatureInspector';

const f3 = (v: readonly number[]) => `(${v.map((x) => (Math.abs(x) < 5e-5 ? '0.0000' : x.toFixed(4))).join(', ')})`;
const apply = (R: readonly (readonly number[])[], v: readonly number[]): Vec3 =>
  [0, 1, 2].map((i) => R[i]![0]! * v[0]! + R[i]![1]! * v[1]! + R[i]![2]! * v[2]!) as Vec3;

export function RotationDemo({ m, channel }: { m: Manifest; channel: number }) {
  const rot = m.checks.rotation;
  const run = rot.runs.find((r) => r.frame !== m.displacement.baselineIndex) ?? rot.runs[0]!;
  const [chunk, setChunk] = useState<FrameChunk | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    loadChunk(run.frame).then(setChunk, (e: unknown) => setErr(String(e)));
  }, [run.frame]);
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
        {rot.angleDeg}° about ({rot.axis.join(', ')}) — and the checkpoint was run again (frame d = {f.d.toFixed(3)} Å).{' '}
        <span className="label">Precomputed</span>
      </p>
      <dl className="rot__grid mono">
        <dt>Total energy</dt>
        <dd>
          {f.energy.toFixed(6)} eV → {run.energy.toFixed(6)} eV; |ΔE| = {run.energyAbsDiff.toExponential(1)} eV <b>invariant</b>
        </dd>
        <dt>Force on moved atom</dt>
        <dd>
          F = {f3(F)} · R F = {f3(RF)} · rotated run {f3(Frot)} eV/Å; largest deviation over all atoms {run.forceEquivarianceMaxAbs.toExponential(1)}{' '}
          eV/Å <b>rotates with the input</b>
        </dd>
        <dt>Layer 1 scalar, ch {channel}</dt>
        <dd>
          {h0 === null || h0rot === null ? (err ? 'not available' : '…') : `${h0.toPrecision(5)} → ${h0rot.toPrecision(5)}`}; max over all atoms and channels{' '}
          {Math.max(...run.h0InvarianceMaxAbs).toExponential(1)} <b>invariant</b>
        </dd>
        <dt>Layer 1 vector, ch {channel}</dt>
        <dd>
          {h1 && h1rot ? `h = ${f3(h1)} · R h = ${f3(apply(rot.R, h1))} · rotated run ${f3(h1rot)}` : err ? 'not available' : '…'}; max deviation{' '}
          {(run.h1EquivarianceMaxAbs[0] ?? NaN).toExponential(1)} <b>rotates like a vector</b>
        </dd>
      </dl>
      <p className="rot__note">
        Components are compared after applying R to the original — comparing raw x, y, z components across the two frames would be meaningless.
        Deviations are float64 round-off. The l = 1 features use e3nn’s (x, y, z) component order.
      </p>
    </section>
  );
}
