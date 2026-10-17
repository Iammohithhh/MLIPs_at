// Recorded evidence shown with each stage of the walkthrough (all values from the precomputed frame chunks).
import { useEffect, useState } from 'react';
import { norm, okFrameAt, type Manifest } from '../lib/experiment';
import { baselineFacts, experimentInfo } from '../lib/experimentInfo';
import type { Stage } from '../lib/urlState';
import { loadChunk, type FrameChunk } from './FeatureInspector';

type EdgeChunk = FrameChunk & { edgesIntoMoved: { src: number[]; vec: number[][]; radial: number[][]; sh: number[][] } };

function EdgeEvidence({ m, frame }: { m: Manifest; frame: number }) {
  const [c, setC] = useState<EdgeChunk | null>(null);
  useEffect(() => {
    let live = true;
    loadChunk(m.id, frame).then((x) => live && setC(x as EdgeChunk), () => live && setC(null));
    return () => {
      live = false;
    };
  }, [frame]);
  if (!c) return <p className="ev__note">Loading edge data…</p>;
  const e = c.edgesIntoMoved;
  const rows = e.src
    .map((j, i) => ({ j, r: norm(e.vec[i]!), radial: e.radial[i]!, sh: e.sh[i]! }))
    .sort((a, b) => a.r - b.r)
    .slice(0, 6);
  const max = Math.max(...e.radial.flat().map(Math.abs));
  return (
    <div className="ev">
      <p className="ev__title">
        {e.src.length} edges (r &lt; {m.rMax} Å) end on the moved atom (d = {c.d.toFixed(3)} Å); the 6 shortest, with their {rows[0]?.radial.length} radial basis values
      </p>
      <div className="ev__scroll">
        <table className="mono">
          <thead>
            <tr>
              <th>
                from atom
                <br />r (Å)
              </th>
              <th>
                R₁…R₁₀ (colour: ±{max.toPrecision(2)})
                <br />Y₁ (x, y, z)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.j}>
                <td>
                  {row.j}
                  <br />
                  {row.r.toFixed(4)}
                </td>
                <td className="ev__edge">
                  <span className="ev__strip">
                    {row.radial.map((v, n) => (
                      <span key={n} title={`R${n + 1} = ${v.toPrecision(4)}`} style={{ background: v >= 0 ? `rgb(240 169 69 / ${Math.abs(v) / max})` : `rgb(58 159 140 / ${Math.abs(v) / max})` }} />
                    ))}
                  </span>
                  <br />
                  {row.sh.slice(1, 4).map((v) => v.toFixed(3).replace('-', '−')).join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="ev__note">
        Radial values include the smooth cutoff envelope; Y₁ are the l = 1 spherical harmonics (e3nn component normalisation, order x, y, z), proportional to
        the edge direction. Edge vectors follow MACE’s convention r_receiver − r_sender + shift.
      </p>
    </div>
  );
}

function MessageEvidence({ m, frame, atom }: { m: Manifest; frame: number; atom: number }) {
  const [c, setC] = useState<FrameChunk | null>(null);
  const [b, setB] = useState<FrameChunk | null>(null);
  useEffect(() => {
    let live = true;
    loadChunk(m.id, frame).then((x) => live && setC(x), () => live && setC(null));
    loadChunk(m.id, m.displacement.baselineIndex).then((x) => live && setB(x), () => live && setB(null));
    return () => {
      live = false;
    };
  }, [frame, m]);
  const row = m.local.indexOf(atom);
  if (row < 0) return <p className="ev__note">Message tensors are stored for the moved atom and its first two shells only.</p>;
  if (!c || !b) return <p className="ev__note">Loading message data…</p>;
  const l2 = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  return (
    <div className="ev">
      <p className="ev__title">
        Aggregated A of atom {atom}, layer 1: block norm over 128 channels for each l (d = {c.d.toFixed(3)} Å vs baseline)
      </p>
      <dl className="mono">
        {[0, 1, 2, 3].map((l) => {
          const now = l2(c.layers[0]!.Anorm[row]![l]!);
          const base = l2(b.layers[0]!.Anorm[row]![l]!);
          return (
            <div key={l}>
              <dt>l = {l}</dt>
              <dd>
                {base.toPrecision(4)} → {now.toPrecision(4)}
              </dd>
            </div>
          );
        })}
      </dl>
      <p className="ev__note">
        Norms of the stored per-channel norms (|A| over m, then over channels).
        {symmetricBaseline(m) &&
          ' In the perfect crystal the l = 1 and l = 2 blocks vanish by the site’s tetrahedral symmetry, while l = 3 does not; displacing the atom switches the lower ones on.'}
      </p>
    </div>
  );
}

type FeatureChunk = FrameChunk & { layers: { h0: number[][]; h1: number[][][] | null }[] };

/** One row of 128 cells, diverging colour (teal < 0 < amber) or sequential for non-negative norms. */
function Cells({ v, max, label, signed }: { v: number[]; max: number; label: string; signed: boolean }) {
  return (
    <span className="ev__embed" role="img" aria-label={label}>
      {v.map((x, c) => (
        <span
          key={c}
          title={`channel ${c}: ${x.toPrecision(4)}`}
          style={{ background: !signed || x >= 0 ? `rgb(240 169 69 / ${max > 0 ? Math.abs(x) / max : 0})` : `rgb(58 159 140 / ${Math.abs(x) / max})` }}
        />
      ))}
    </span>
  );
}

function FeatureEvidence({ m, frame, atom }: { m: Manifest; frame: number; atom: number }) {
  const [c, setC] = useState<FeatureChunk | null>(null);
  const [b, setB] = useState<FeatureChunk | null>(null);
  useEffect(() => {
    let live = true;
    loadChunk(m.id, frame).then((x) => live && setC(x as FeatureChunk), () => live && setC(null));
    loadChunk(m.id, m.displacement.baselineIndex).then((x) => live && setB(x as FeatureChunk), () => live && setB(null));
    return () => {
      live = false;
    };
  }, [frame, m]);
  const row = m.local.indexOf(atom);
  if (row < 0) return <p className="ev__note">Full feature vectors are stored for the moved atom and its first two shells only; pick one of those atoms.</p>;
  if (!c || !b) return <p className="ev__note">Loading feature data…</p>;
  const blocks: { key: string; title: string; base: number[]; cur: number[]; signed: boolean }[] = [];
  c.layers.forEach((L, t) => {
    const B0 = b.layers[t]!;
    blocks.push({ key: `${t}s`, title: `Layer ${t + 1}, 128 scalars h⁽${t + 1}⁾ (l = 0)`, base: B0.h0[row]!, cur: L.h0[row]!, signed: true });
    if (L.h1 && B0.h1)
      blocks.push({ key: `${t}v`, title: `Layer ${t + 1}, 128 vectors (l = 1): length |h| per channel`, base: B0.h1[row]!.map(norm), cur: L.h1[row]!.map(norm), signed: false });
  });
  return (
    <div className="ev">
      <p className="ev__title">
        Atom {atom}: baseline vs this frame (d = {c.d.toFixed(3)} Å)
      </p>
      {blocks.map((bl) => {
        const max = Math.max(...bl.base.map(Math.abs), ...bl.cur.map(Math.abs));
        return (
          <div key={bl.key} className="ev__feat">
            <p className="ev__note">
              {bl.title}; colour {bl.signed ? '±' : '0 –'} {max === 0 ? '0 (all below 10⁻¹², stored as 0)' : max.toPrecision(3)}, shared by both rows
            </p>
            <span className="ev__rowlabel mono">baseline</span>
            <Cells v={bl.base} max={max} signed={bl.signed} label={`${bl.title}, baseline`} />
            <span className="ev__rowlabel mono">d = {c.d.toFixed(3)} Å</span>
            <Cells v={bl.cur} max={max} signed={bl.signed} label={`${bl.title}, this frame`} />
          </div>
        );
      })}
      <p className="ev__note">
        Each new feature is a learned linear mix of products of up to three aggregated tensors A (correlation order ν = 3), so one
        feature can depend on up to three neighbours jointly (four-body terms) in a single layer. The vector channels are rotation-equivariant, so only
        their lengths are drawn here, not their directions.
        {symmetricBaseline(m) && ' In the perfect crystal the layer-1 vectors vanish by symmetry (stored values below 10⁻¹² are written as 0).'}
      </p>
    </div>
  );
}

/** Symmetry sentences are shown only for an experiment that documents the reason and whose stored baseline vectors are zero. */
function symmetricBaseline(m: Manifest): boolean {
  return Boolean(experimentInfo(m.id)?.vectorsVanishReason) && baselineFacts(m).vectorsVanish === true;
}

export function StageEvidence({ m, stage, frame, atom }: { m: Manifest; stage: Stage; frame: number; atom: number }) {
  const f = okFrameAt(m, frame);
  if (!f) return null;
  if (stage === 'species') {
    const v = m.baseline.embed;
    const el = m.structure.symbols[m.displacement.atom] ?? '';
    const sym = m.structure.symbols[atom] ?? '';
    const max = Math.max(...v.map(Math.abs));
    const p = m.structure.positions[atom]!;
    return (
      <div className="ev">
        <p className="ev__title">
          Input for atom {atom}: {m.structure.symbols[atom]} at ({(atom === m.displacement.atom ? f.movedPosition : p).map((x) => x.toFixed(3)).join(', ')}) Å
        </p>
        <p className="ev__note">
          Initial features h⁽⁰⁾ of a {el} atom (the moved atom’s element): {v.length} learned numbers selected by the element alone (identical for
          every {el} atom, independent of position). Colour ±{max.toPrecision(2)}.
          {sym !== el && ` Atom ${atom} is ${sym}; its embedding is not stored, so it is not shown.`}
        </p>
        {sym === el && (
          <span className="ev__embed" role="img" aria-label={`${v.length} embedding values of ${el}`}>
            {v.map((x, c) => (
              <span key={c} title={`channel ${c}: ${x.toPrecision(4)}`} style={{ background: x >= 0 ? `rgb(240 169 69 / ${Math.abs(x) / max})` : `rgb(58 159 140 / ${Math.abs(x) / max})` }} />
            ))}
          </span>
        )}
      </div>
    );
  }
  if (stage === 'features') return <FeatureEvidence m={m} frame={frame} atom={atom} />;
  if (stage === 'messages') return <MessageEvidence m={m} frame={frame} atom={atom} />;
  if (stage === 'radial' || stage === 'neighbours') return <EdgeEvidence m={m} frame={frame} />;
  if (stage === 'energy') {
    const sum = f.nodeEnergy.reduce((a, b) => a + b, 0);
    const e0 = m.baseline.e0[atom];
    const ei = f.nodeEnergy[atom];
    return (
      <div className="ev">
        <p className="ev__title">This frame (d = {f.d.toFixed(3)} Å)</p>
        <dl className="mono">
          <dt>E_{atom} (site)</dt>
          <dd>{eV5(ei)} eV, of which E₀({m.structure.symbols[atom]}) = {eV5(e0)} eV is a fixed per-element reference</dd>
          <dt>Σᵢ Eᵢ, {m.structure.n} atoms</dt>
          <dd>{eV5(sum)} eV</dd>
          <dt>E (model total)</dt>
          <dd>{eV5(f.energy)} eV</dd>
        </dl>
        <p className="ev__note">Site energies are a model-internal partition of E, not separately measurable quantities.</p>
      </div>
    );
  }
  if (stage === 'forces') {
    return (
      <div className="ev">
        <p className="ev__title">Check: autograd force vs central finite difference of the total energy</p>
        {/* One card per stored check, so every column stays readable in the narrow side panel (a wide table hid
            the finite-difference columns behind a horizontal scroll). */}
        <ol className="ev__fd">
          {m.checks.finiteDifference.map((r) => {
            const d = m.displacement.values[r.frame]!;
            return (
              <li key={`${r.frame}-${r.atom}`}>
                <p className="ev__fd-head">
                  d = {d > 0 ? '+' : d < 0 ? '−' : ''}
                  {Math.abs(d).toFixed(3)} Å · {r.label} (atom {r.atom})
                </p>
                <p className="ev__fd-val mono">
                  autograd F = {Math.abs(r.analytic) < 1e-12 ? '0' : r.analytic.toFixed(6).replace('-', '−')} eV/Å
                </p>
                <dl className="ev__fd-steps mono" aria-label="absolute difference between finite difference and autograd, by step h">
                  {r.steps.map((st) => (
                    <div key={st.h}>
                      <dt>{st.h}</dt>
                      <dd>{st.absDiff === 0 ? '0' : st.absDiff.toExponential(1).replace('e-', 'e−')}</dd>
                    </div>
                  ))}
                </dl>
              </li>
            );
          })}
        </ol>
        <p className="ev__note">In each card: top row the step h in Å, below it |F_FD − F_autograd| in eV/Å.</p>
        <p className="ev__note">
          F = −[E(r + h) − E(r − h)] / 2h, every energy a full model evaluation in float64. The error falls about tenfold for each
          3.3× smaller h, the h² behaviour of central differences, so truncation error dominates at these step sizes.{' '}
          {baselineFacts(m).forceFree
            ? 'At d = 0 the moved atom’s predicted force is zero, yet its finite difference is not exactly zero at large h because the energy well is asymmetric (cubic term).'
            : `At d = 0 this unrelaxed geometry already carries a predicted force (largest ${baselineFacts(m).maxForce.toFixed(3)} eV/Å), and the finite difference reproduces it.`}
        </p>
      </div>
    );
  }
  return null;
}

/** Energy to 5 decimals with a true minus sign; missing values are shown as missing, never as 0. */
function eV5(x: number | null | undefined): string {
  return x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(5).replace('-', '−');
}
