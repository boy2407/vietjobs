---
slug: field-order-512
epic: label-noise-and-text
kind: feature
depends_on: [class-centroid-cosine]
status: draft
version: 1
updated: 2026-10-17
---
# Choose the field order under the 512-token limit

As a **thesis author**, I want **to know which field the 512-token limit cuts first and which order loses the least**, so that **truncation removes the least informative text**.

Board row: T14.6 (US6). Gated like `strip-company-intro`.

## Acceptance criteria
- AC1 — Given the CafeBERT input at 512 tokens, When truncation is measured on `train`, Then the share of postings truncated, and tokens cut, per field is reported.
- AC2 — Given each field order under test, When seeds 42–44 run on `device=cpu`, Then three runs per order exist with one ablation row each in `04-results.md`.
- AC3 — Given the three-seed means on `dev`, When the best order is chosen, Then it becomes the default only if it beats the current order; otherwise the result is recorded as "dropped" in `docs/02-vietnamese-nlp.md`.
- AC4 — Given the runs are final, When the story is closed, Then Rules 1, 7, 10 and 13 are followed, `scripts/export_runs.py` is re-run and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Changing `--max-len` for PhoBERT.

## Open questions
- Which orders are tested? Recommended: current order, the reverse, and the field with the most truncation moved last.

## Decisions
