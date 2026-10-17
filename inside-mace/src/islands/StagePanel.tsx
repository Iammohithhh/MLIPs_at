// Compact computation-stage navigator and the selected stage's explanation: plain sentence, what is new,
// equation (KaTeX, on demand), recorded tensor and the mace-torch implementation location.
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { useMemo, useState } from 'react';
import { CODE_PACKAGE, STAGE_INFO } from '../lib/stages';
import type { Stage } from '../lib/urlState';

export function StageNav({ stage, onStage }: { stage: Stage; onStage: (s: Stage) => void }) {
  return (
    <nav className="sn" aria-label="Computation stages">
      <ol>
        {STAGE_INFO.map((s) => (
          <li key={s.id}>
            <button aria-current={s.id === stage ? 'step' : undefined} className={s.id === stage ? 'is-on' : ''} onClick={() => onStage(s.id)}>
              <span className="sn__n mono">{s.n}</span>
              <span className="sn__t">{s.title}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export function StageDetail({ stage, onStage }: { stage: Stage; onStage: (s: Stage) => void }) {
  const [technical, setTechnical] = useState(false);
  const i = STAGE_INFO.findIndex((s) => s.id === stage);
  const s = STAGE_INFO[i]!;
  const html = useMemo(() => katex.renderToString(s.tex, { displayMode: true, throwOnError: false, output: 'htmlAndMathml' }), [s.tex]);
  return (
    <article className="sd" aria-live="polite">
      <p className="sd__step mono">
        Stage {s.n} of {STAGE_INFO.length}
      </p>
      <h3>{s.title}</h3>
      <p className="sd__plain">{s.plain}</p>
      <p className="sd__changed">
        <strong>New at this stage:</strong> {s.changed}
      </p>
      <button className="sd__more" aria-expanded={technical} onClick={() => setTechnical((x) => !x)}>
        {technical ? 'Hide equation and code' : 'Show equation and code'}
      </button>
      {technical && (
        <div className="sd__tech">
          <div className="sd__eq" dangerouslySetInnerHTML={{ __html: html }} />
          <dl className="mono">
            <dt>Recorded tensor</dt>
            <dd>
              {s.data} · {s.shape}
            </dd>
            {s.subset && (
              <>
                <dt>Subset</dt>
                <dd>{s.subset}</dd>
              </>
            )}
            <dt>Code ({CODE_PACKAGE})</dt>
            {s.code.map((c) => (
              <dd key={c.file + c.line}>
                {c.file}:{c.line} — {c.symbol}
              </dd>
            ))}
          </dl>
        </div>
      )}
      <div className="sd__nav">
        <button disabled={i === 0} onClick={() => onStage(STAGE_INFO[i - 1]!.id)}>
          ← Previous
        </button>
        <button disabled={i === STAGE_INFO.length - 1} onClick={() => onStage(STAGE_INFO[i + 1]!.id)}>
          Next stage →
        </button>
      </div>
    </article>
  );
}
