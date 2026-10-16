# Contributing to MLIP Atlas

MLIP Atlas is evidence-first: every statement on the site is a record that says where it comes from and how strong
that evidence is. Corrections and new sources are welcome — the rules below are what makes them checkable.

## The quickest way to help

- **Something is wrong or overstated:** open a [correction](https://github.com/Iammohithhh/MLIPs_at/issues/new?template=correction.yml). Every claim on the site has a “Suggest a correction” link that fills in the record id.
- **A paper, dataset, checkpoint or result is missing:** open a [new source](https://github.com/Iammohithhh/MLIPs_at/issues/new?template=new-source.yml) issue.
- **A failure mode should be documented:** open a [failure case proposal](https://github.com/Iammohithhh/MLIPs_at/issues/new?template=failure-case.yml).

## Evidence rules

1. **One checkable statement per claim**, with a source and a locator (figure, table, section, equation), or an on-page derivation.
2. **Say how you read the source.** `verified_against` is `full-text`, `abstract` or `secondary`. Only full-text sources can back a published record.
3. **Grade the evidence:** `PUBLISHED_RESULT` (a paper reports it), `REPRODUCED` (recomputed by the Atlas; needs a matching reproduction experiment), `THEORY` (derivation), `TOY_DEMO`, `HYPOTHESIS`. A reproduction that does not match is recorded with `matches: false`, never hidden.
4. **Never type benchmark numbers.** Add or extend a pipeline module in `pipeline/atlas_pipeline/` that copies them from the source with provenance, with tests.
5. **Keep architecture, checkpoint, dataset and benchmark result separate.** Results from different protocols are never combined into one ranking.
6. **Label every interactive** (TOY MODEL, LIVE CALCULATION, PRECOMPUTED, PUBLISHED RESULT, REPRODUCED) and show assumptions, units and sign conventions next to it.
7. **Electronic-structure quantities** (Fermi level, density of states, band alignment) are never shown as MLIP output.

## Review lifecycle

Records move from `draft` (written, checked by tooling) to `in-review` (a reviewer is reading the sources) to
`published` (a named reviewer has checked every source against its locator). The build refuses a published record
that rests on a draft, or on a source read only as an abstract.

## Working on the code

- Site: `cd atlas && npm ci && npm run dev`. Before a pull request: `npm run check && npm test && npm run build && npm run check:links`.
- Pipeline: `python -m venv .venv && .venv/bin/pip install -r pipeline/requirements.txt`, then `cd pipeline && python -m pytest -q tests`.
- Project conventions (3D engine, maths, content schemas) are in [CLAUDE.md](CLAUDE.md); the product plan is in [MLIP_Atlas_Vision.md](MLIP_Atlas_Vision.md).
