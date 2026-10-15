import { useState } from 'react';
import './PhysicsLab.css';
import { tensorProduct } from '~/lib/physics/spherical-harmonics';

// Physics Lab P5, part 2: which outputs can two feature blocks l₁ and l₂ produce?
// Each block is drawn as 2l+1 cells, so the dimension count is visible.

const NAMES: Record<number, string> = { 0: 'scalar', 1: 'vector', 2: 'rank-2', 3: 'rank-3' };

function Cells({ n, tone }: { n: number; tone: string }) {
  return (
    <span className={`tp-cells tone-${tone}`} aria-hidden="true">
      {Array.from({ length: n }, (_, k) => (
        <span key={k} />
      ))}
    </span>
  );
}

const EXAMPLES: Record<string, string> = {
  '1x1': 'Two vectors a, b: l = 0 is the dot product a·b, l = 1 the cross product a × b (a pseudovector: it does not flip under inversion), l = 2 the traceless symmetric part of the outer product.',
  '0x1': 'A scalar times a vector is just a scaled vector: the only output is l = 1.',
  '1x2': 'A vector and a rank-2 block give l = 1, 2 and 3; the l = 2 path has the “wrong” parity and is a pseudotensor.',
};

export default function TensorProductLab() {
  const [l1, setL1] = useState(1);
  const [l2, setL2] = useState(1);
  const paths = tensorProduct(l1, l2);
  const inDim = (2 * l1 + 1) * (2 * l2 + 1);
  const key = `${Math.min(l1, l2)}x${Math.max(l1, l2)}`;

  return (
    <div className="lab tp">
      <div className="tp-inputs">
        {(
          [
            ['l₁', l1, setL1],
            ['l₂', l2, setL2],
          ] as const
        ).map(([label, val, set]) => (
          <div key={label} className="tp-input">
            <div className="seg" role="group" aria-label={`Input ${label}`}>
              <span className="seg-label">{label}</span>
              {[0, 1, 2, 3].map((k) => (
                <button key={k} type="button" aria-pressed={val === k} onClick={() => set(k)}>
                  {k}
                </button>
              ))}
            </div>
            <div className="tp-block">
              <Cells n={2 * val + 1} tone="arch" />
              <span className="tp-name">
                {NAMES[val]} · {2 * val + 1} component{val ? 's' : ''}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="tp-arrow" aria-hidden="true">
        ⊗ → ⊕
      </div>

      <ol className="tp-outputs" aria-live="polite">
        {paths.map((p) => (
          <li key={p.l} className={p.natural ? 'natural' : 'pseudo'}>
            <span className="tp-l">
              l = {p.l}
              <small>{p.natural ? 'polar' : 'pseudo'}</small>
            </span>
            <Cells n={p.dim} tone={p.natural ? 'phys' : 'caveat'} />
          </li>
        ))}
      </ol>

      <p className="tp-count">
        <span className="num">
          ({2 * l1 + 1}) × ({2 * l2 + 1}) = {inDim}
        </span>{' '}
        ={' '}
        <span className="num">
          {paths.map((p) => p.dim).join(' + ')} = {paths.reduce((s, p) => s + p.dim, 0)}
        </span>
        <span className="tp-check">✓ nothing lost, nothing created</span>
      </p>
      <p className="lab-note tp-note">
        Outputs run from |l₁ − l₂| to l₁ + l₂. Each carries parity (−1)<sup>l₁+l₂</sup>; when that differs from (−1)
        <sup>l</sup> the output is a pseudo-quantity that does not flip sign under inversion.{' '}
        {EXAMPLES[key] ?? 'Equivariant networks choose which of these paths to keep, with learned weights on each.'}
      </p>
    </div>
  );
}
