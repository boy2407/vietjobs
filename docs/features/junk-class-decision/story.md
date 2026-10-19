---
slug: junk-class-decision
epic: label-noise-and-text
kind: feature
depends_on: [hand-label-dev-errors, class-centroid-cosine]
status: draft
version: 1
updated: 2026-10-17
---
# Decide how to handle the junk class

As a **thesis author**, I want **a recorded decision on `nhom_nghe_khac` (F1 0.000, n = 25 on `dev`)**, so that **the zero is explained and every later number says how the class was treated**.

Board row: T14.4 (US4). Human decision: the author decides, an agent only records it.

## Acceptance criteria
- AC1 — Given the hand-label counts and the centroid cosines, When the author chooses between merge, relabel, drop, or only reporting `f1_macro_no_junk`, Then the choice and its reasons are written in the log of `tasks/T14-nhieu-nhan-va-van-ban.md`.
- AC2 — Given a choice that changes a label or a split, When it is recorded, Then the author's explicit approval is written with it, together with the note that every row of `04-results.md` becomes incomparable.
- AC3 — Given any choice, When it is applied, Then `test.csv` is untouched (Rule 5).
- AC4 — Given the decision is recorded, When the story is closed, Then `docs/` and `docs/bao-cao/` are updated per Rules 1, 7 and 10 and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Applying a rebuild of the splits, unless the decision itself says so and is approved.

## Open questions
- Is a rebuild that invalidates `04-results.md` ever acceptable? Author's call.

## Decisions
