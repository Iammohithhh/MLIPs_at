// The computation graph of the MACE-MP-0a medium checkpoint as one compact diagram (brief §5/§6): the
// actual blocks, shapes and hyperparameters (from the checkpoint metadata recorded in the Atlas X-ray,
// atlas/src/data/xray/mace-mp-0a-medium.json), not a generic network cartoon. Selecting a walkthrough
// stage emphasises the blocks it covers; clicking a block selects its stage. Built from HTML boxes so it
// reflows to a vertical stack on narrow screens instead of shrinking.
import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import type { Stage } from '../lib/urlState';
import { usePrefersReducedMotion } from '../three/theme';

interface Block {
  id: string;
  stage: Stage;
  title: string;
  shape: string;
  note?: string;
}

// Rows top → bottom in dependency order. Two blocks in a row are computed side by side.
const ROWS: { label: string; blocks: Block[] }[] = [
  {
    label: 'Input',
    blocks: [
      { id: 'z', stage: 'species', title: 'Species Zᵢ → embedding h⁽⁰⁾', shape: '128×0e', note: 'one-hot × learned matrix' },
      { id: 'graph', stage: 'neighbours', title: 'Positions, cell → edges j ∈ N(i)', shape: 'rᵢⱼ < r_max = 6 Å', note: 'periodic images included' },
    ],
  },
  {
    label: 'Edges',
    blocks: [
      { id: 'radial', stage: 'radial', title: 'Bessel basis × cutoff → radial MLP', shape: '10 → 64 → 64 → 64 → weights', note: 'polynomial cutoff p = 5' },
      { id: 'sh', stage: 'radial', title: 'Spherical harmonics Y(r̂ᵢⱼ)', shape: 'l = 0…3, 16 values' },
    ],
  },
  {
    label: 'Layer 1',
    blocks: [
      { id: 'A1', stage: 'messages', title: 'Interaction: Aᵢ = Σⱼ R ⊗ Y ⊗ W hⱼ / n̄', shape: '128 ch × l = 0…3', note: 'n̄ = 61.96' },
      { id: 'B1', stage: 'features', title: 'Product basis (ν ≤ 3) + update → h⁽¹⁾', shape: '128×0e + 128×1o' },
      { id: 'R1', stage: 'energy', title: 'Readout 1 (linear)', shape: 'h⁽¹⁾ scalars → 1' },
    ],
  },
  {
    label: 'Layer 2',
    blocks: [
      { id: 'A2', stage: 'messages', title: 'Interaction on h⁽¹⁾ of neighbours', shape: '128 ch × l = 0…3' },
      { id: 'B2', stage: 'features', title: 'Product basis (ν ≤ 3) + update → h⁽²⁾', shape: '128×0e (scalars only)' },
      { id: 'R2', stage: 'energy', title: 'Readout 2 (MLP)', shape: '128×0e → 16×0e → 1' },
    ],
  },
  {
    label: 'Energy',
    blocks: [{ id: 'E', stage: 'energy', title: 'Eᵢ = E₀(Zᵢ) + s·(readout₁ + readout₂) + μ;  E = Σᵢ Eᵢ', shape: 'eV', note: 's = 0.804, μ = 0.164 eV' }],
  },
  {
    label: 'Forces',
    blocks: [{ id: 'F', stage: 'forces', title: 'Fᵢ = −∂E/∂rᵢ by automatic differentiation through every block above', shape: 'eV/Å, [N atoms, 3]' }],
  },
];

export function ArchitectureDiagram({ stage, onStage }: { stage: Stage | null; onStage: (s: Stage) => void }) {
  const ref = useRef<HTMLElement | null>(null);
  const reduced = usePrefersReducedMotion();
  // GSAP settles the blocks of the newly selected stage into their emphasised state together (no stagger: the
  // blocks of one stage are not computed one after another in the order of an animation).
  useEffect(() => {
    const fig = ref.current;
    if (!fig || reduced || stage === null) return;
    const on = fig.querySelectorAll('.arch__block.is-on');
    if (on.length === 0) return;
    const tween = gsap.fromTo(on, { scale: 0.97 }, { scale: 1, duration: 0.3, ease: 'back.out(2)', clearProps: 'transform' });
    return () => {
      tween.kill();
      gsap.set(on, { clearProps: 'transform' });
    };
  }, [stage, reduced]);
  return (
    <figure ref={ref} className="arch" aria-label="MACE-MP-0a medium computation graph">
      <ol className="arch__rows">
        {ROWS.map((row) => (
          <li key={row.label} className="arch__row">
            <span className="arch__label">{row.label}</span>
            <div className="arch__blocks">
              {row.blocks.map((b) => (
                <button key={b.id} className={`arch__block${stage === b.stage ? ' is-on' : ''}`} aria-pressed={stage === b.stage} onClick={() => onStage(b.stage)}>
                  <span className="arch__title">{b.title}</span>
                  <span className="arch__shape mono">{b.shape}</span>
                  {b.note && <span className="arch__note">{b.note}</span>}
                </button>
              ))}
            </div>
          </li>
        ))}
      </ol>
      <figcaption>
        Blocks and shapes of the checkpoint used here (ScaleShiftMACE, hidden irreps 128×0e + 128×1o, correlation order 3, l_max = 3). Layer 1 feeds
        layer 2 through each neighbour’s h⁽¹⁾; both layers contribute a readout. Click a block to open its walkthrough stage.
      </figcaption>
    </figure>
  );
}
