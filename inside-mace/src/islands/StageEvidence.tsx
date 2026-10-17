// Recorded evidence shown with the energy and forces stages of the walkthrough.
import { okFrameAt, type Manifest } from '../lib/experiment';
import type { Stage } from '../lib/urlState';

export function StageEvidence({ m, stage, frame, atom }: { m: Manifest; stage: Stage; frame: number; atom: number }) {
  const f = okFrameAt(m, frame);
  if (!f) return null;
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
