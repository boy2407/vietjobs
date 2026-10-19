---
slug: salary-model-comparison-docs
epic: label-noise-and-text
kind: feature
depends_on: []
status: draft
version: 1
updated: 2026-10-17
---
# Record the CafeBERT+RNN salary comparisons in the docs

As a **thesis author**, I want **the salary comparison of CafeBERT+RNN against Dense and against PhoBERT+RNN written down with its p-values**, so that **the report states what is a win and what is unclear**.

Board row: T14.9 (US9). Documentation only: no training.

## Acceptance criteria
- AC1 — Given the three-seed results, When documented, Then `docs/06-baseline-dl.md` states CafeBERT+RNN beats Dense with p = 0.000 on 3 seeds.
- AC2 — Given the PhoBERT+RNN comparison, When documented, Then it states p = 0.076 / 0.246 / 0.000 and calls the result unclear.
- AC3 — Given every p-value, When it is written, Then it names the source file it came from.
- AC4 — Given the same facts, When the report is updated, Then the matching chapter 4 in `docs/bao-cao/` says the same, and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- New runs or new significance tests.

## Open questions
- Which file holds each p-value? To locate before writing; one with no file behind it is deleted, not quoted (Rule 12 item 4).

## Decisions
