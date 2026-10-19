---
slug: top3-product-result
epic: label-noise-and-text
kind: feature
depends_on: []
status: draft
version: 1
updated: 2026-10-17
---
# Report top-3 accuracy as a product result

As a **job seeker using the classifier**, I want **the top three suggested occupations to contain the right one 0.9373 of the time**, so that **I can trust a short list even when the top pick is wrong**.

Board row: T14.11 (US11). Waits for T5.1.

## Acceptance criteria
- AC1 — Given the inference entry point from T5.1 and `dev`, When top-3 accuracy is computed, Then it reproduces 0.9373.
- AC2 — Given the reproduced number, When documented, Then a sourced row exists in the docs and in `docs/bao-cao/` chapter 5.
- AC3 — Given the measurement, When it runs, Then `test` is not read.
- AC4 — Given the number is final, When the story is closed, Then Rules 1, 7 and 10 are followed and `pytest -q` is green.

**INVEST** — I ❌ (needs the inference entry point of T5.1, which has no story yet) · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Building the inference entry point (T5.1).

## Open questions
- T5.1 has no story and no Contract. Write T5.1 as a `foundation` story first, then list it in `depends_on`? Recommended: yes. Do not approve this story before.

## Decisions
