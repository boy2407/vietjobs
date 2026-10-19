---
slug: hand-label-dev-errors
epic: label-noise-and-text
kind: feature
depends_on: []
status: draft
version: 1
updated: 2026-10-17
---
# Hand-label 150 dev errors

As a **thesis author**, I want **to label 150 of the model's `dev` errors by hand as model error, wrong original label or ambiguous**, so that **the noise ceiling rests on a human verdict and not only on a formula**.

Board row: T14.2 (US2). Human only: an agent prepares the file and never fills the verdict.

## Acceptance criteria
- AC1 — Given the `dev` predictions of `dl-cat-cafe-rnn-ce-s42`, When the sample is drawn with a fixed seed, Then `artifacts/eda/hand-label-150.csv` holds 150 rows with `pred != true`, unique row ids, and columns id, title, true, pred and an empty `verdict`.
- AC2 — Given the author has filled `verdict` with one of `model_error`, `wrong_label`, `ambiguous`, When the rows are counted, Then the count per verdict is written next to the sampling seed.
- AC3 — Given the agent prepared the file, When it hands it over, Then no `verdict` cell is filled by the agent.
- AC4 — Given the counts are final, When the story is closed, Then `docs/` and `docs/bao-cao/` are updated per Rules 1, 7 and 10 and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Changing any label from the verdicts (see `junk-class-decision`).

## Open questions
- Sampling seed: reuse `SPLIT_SEED = 20260826`? Recommended: yes.

## Decisions
