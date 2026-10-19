---
slug: strip-company-intro
epic: label-noise-and-text
kind: feature
depends_on: [class-centroid-cosine]
status: draft
version: 1
updated: 2026-10-17
---
# Strip company-intro boilerplate from the CafeBERT input

As a **thesis author**, I want **to know whether removing company-introduction text from what CafeBERT reads raises category macro-F1**, so that **the 512-token budget goes to the job and not to the employer's self-description**.

Board row: T14.5 (US5). Gated: if the ceiling or a symmetric pair shows label noise dominates, this story stays `todo`.

## Acceptance criteria
- AC1 — Given the CafeBERT text builder, When the boilerplate rule is defined, Then the share of `train` postings it changes is measured and written down.
- AC2 — Given seeds 42–44 on `device=cpu`, When the runs finish, Then three runs and one ablation row in `04-results.md` compare against the CafeBERT+RNN baseline and the probe 0.5297.
- AC3 — Given the three-seed mean macro-F1 on `dev`, When it does not beat the baseline, Then the step is recorded as "dropped" in `docs/02-vietnamese-nlp.md` and removed (Rule 2 item 4).
- AC4 — Given the step is kept, When it is merged, Then it has a test in `tests/test_vitext.py` or `tests/test_dl_text.py`.
- AC5 — Given the runs are final, When the story is closed, Then Rules 1, 7, 10 and 13 are followed, `scripts/export_runs.py` is re-run and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Other text steps (`cafebert-preprocess-ablation`, `field-order-512`).

## Open questions
- How is "company intro" detected without a classifier? Plan decides; the measured share in AC1 comes first.

## Decisions
