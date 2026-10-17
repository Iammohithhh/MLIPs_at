// Recorded evidence shown with the energy and forces stages of the walkthrough.
import { useEffect, useState } from 'react';
import { norm, okFrameAt, type Manifest } from '../lib/experiment';
import type { Stage } from '../lib/urlState';
import { loadChunk, type FrameChunk } from './FeatureInspector';

type EdgeChunk = FrameChunk & { edgesIntoMoved: { src: number[]; vec: number[][]; radial: number[][]; sh: number[][] } };

function EdgeEvidence({ m, frame }: { m: Manifest; frame: number }) {
  const [c, setC] = useState<EdgeChunk | null>(null);
  useEffect(() => {
    let live = true;
    loadChunk(frame).then((x) => live && setC(x as EdgeChunk), () => live && setC(null));
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
              <th>from atom</th>
              <th>r (Å)</th>
              <th>R₁…R₁₀ (colour: ±{max.toPrecision(2)})</th>
              <th>Y₁ (x, y, z)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.j}>
                <td>{row.j}</td>
                <td>{row.r.toFixed(4)}</td>
                <td>
                  <span className="ev__strip">
                    {row.radial.map((v, n) => (
                      <span key={n} title={`R${n + 1} = ${v.toPrecision(4)}`} style={{ background: v >= 0 ? `rgb(240 169 69 / ${Math.abs(v) / max})` : `rgb(58 159 140 / ${Math.abs(v) / max})` }} />
                    ))}
                  </span>
                </td>
                <td>{row.sh.slice(1, 4).map((v) => v.toFixed(3)).join(', ')}</td>
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

export function StageEvidence({ m, stage, frame, atom }: { m: Manifest; stage: Stage; frame: number; atom: number }) {
  const f = okFrameAt(m, frame);
  if (!f) return null;
  if (stage === 'species') {
    const v = m.baseline.embed;
    const max = Math.max(...v.map(Math.abs));
    const p = m.structure.positions[atom]!;
    return (
      <div className="ev">
        <p className="ev__title">
          Input for atom {atom}: {m.structure.symbols[atom]} at ({(atom === m.displacement.atom ? f.movedPosition : p).map((x) => x.toFixed(3)).join(', ')}) Å
        </p>
        <p className="ev__note">
          Initial features h⁽⁰⁾ of a Si atom: {v.length} learned numbers selected by the element alone (identical for every Si atom, independent of
          position). Colour ±{max.toPrecision(2)}.
        </p>
        <span className="ev__embed" role="img" aria-label={`${v.length} embedding values of Si`}>
          {v.map((x, c) => (
            <span key={c} title={`channel ${c}: ${x.toPrecision(4)}`} style={{ background: x >= 0 ? `rgb(240 169 69 / ${Math.abs(x) / max})` : `rgb(58 159 140 / ${Math.abs(x) / max})` }} />
          ))}
        </span>
      </div>
    );
  }
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
          <dd>{ei?.toFixed(5)} eV, of which E₀(Si) = {e0?.toFixed(5)} eV is a fixed per-element reference</dd>
          <dt>Σᵢ Eᵢ, 216 atoms</dt>
          <dd>{sum.toFixed(5)} eV</dd>
          <dt>E (model total)</dt>
          <dd>{f.energy.toFixed(5)} eV</dd>
        </dl>
        <p className="ev__note">Site energies are a model-internal partition of E, not separately measurable quantities.</p>
      </div>
    );
  }
  if (stage === 'forces') {
    return (
      <div className="ev">
        <p className="ev__title">Check: autograd force vs central finite difference of the total energy</p>
        <div className="ev__scroll">
          <table className="mono">
            <thead>
              <tr>
                <th>frame d</th>
                <th>component</th>
                <th>autograd (eV/Å)</th>
                {m.checks.finiteDifference[0]!.steps.map((s) => (
                  <th key={s.h}>|Δ| at h = {s.h} Å</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {m.checks.finiteDifference.map((r) => (
                <tr key={`${r.frame}-${r.atom}`}>
                  <td>{m.displacement.values[r.frame]!.toFixed(3)}</td>
                  <td>{r.label}</td>
                  <td>{Math.abs(r.analytic) < 1e-12 ? '0' : r.analytic.toFixed(6)}</td>
                  {r.steps.map((s) => (
                    <td key={s.h}>{s.absDiff === 0 ? '0' : s.absDiff.toExponential(1)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="ev__note">
          F = −[E(r + h) − E(r − h)] / 2h, every energy a full model evaluation in float64. The error falls about tenfold for each
          3.3× smaller h, the h² behaviour of central differences, so truncation error dominates at these step sizes. At d = 0 the moved atom’s force is zero by
          symmetry, yet its finite difference is not exactly zero at large h because the energy well is asymmetric (cubic term).
        </p>
      </div>
    );
  }
  return null;
}
