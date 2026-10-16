## What this changes

<!-- Pages, records, pipeline modules. -->

## Evidence checklist

- [ ] Every new claim has a source with a locator (figure/table/section) or an on-page derivation.
- [ ] `verified_against` says how each source was read (full text, abstract, secondary); only full-text sources can back published records.
- [ ] Benchmark numbers come from a tested pipeline snapshot, not typed by hand; different protocols are never merged into one ranking.
- [ ] Interactives are labelled (TOY MODEL, LIVE CALCULATION, PRECOMPUTED, PUBLISHED RESULT, REPRODUCED) with assumptions, units and sign conventions in view.
- [ ] Architecture, checkpoint, dataset and benchmark result stay separate records.
- [ ] No DFT-only quantity (Fermi level, DOS, band alignment) is shown as an MLIP prediction.
- [ ] New papers were added to the bibliography pipeline (`atlas_pipeline.bibliography --only <ids>`).

## Checks

- [ ] `cd atlas && npm ci && npm run check && npm test && npm run build && npm run check:links`
- [ ] `cd pipeline && python -m pytest -q tests`
- [ ] Light and dark themes and a phone-width screenshot look right.
