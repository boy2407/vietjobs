---
slug: cafebert-rnn-mean-reconcile
epic: label-noise-and-text
kind: feature
depends_on: []
status: done
version: 1
updated: 2026-10-17
---
# Reconcile the two CafeBERT+RNN category means

As a **thesis author**, I want **one sourced mean for CafeBERT+RNN on `category`**, so that **the report does not quote 0.6152 in one place and 0.6097 in another**.

Board row: T14.10 (US10).

## Acceptance criteria
- AC1 — Given `artifacts/dl-cat-cafe-rnn-ce-s4{2,3,4}/metrics.json`, When the mean macro-F1 is computed, Then one number is written with its source.
- AC2 — Given the other value, When its origin is traced, Then the cause of the difference is written down.
- AC3 — Given every document that quotes a CafeBERT+RNN `category` mean, When the story closes, Then each quotes the one number.
- AC4 — Given `docs/04-results.md` is append-only, When the story closes, Then no existing row of it was edited, and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- New runs.

## Open questions
- The `04-results.md` rows give 0.6172 / 0.6223 / 0.6060. Does the other value come from a different seed set or a different eval? Trace it first.

## Decisions
