---
slug: label-noise-ceiling
epic: label-noise-and-text
kind: feature
depends_on: []
status: dropped
version: 1
updated: 2026-10-17
---
# Estimate the label-noise ceiling on dev

As a **thesis author**, I want **an estimate of how much of the category macro-F1 gap is label noise**, so that **I know whether a better model or better text can still buy it back**.

Board row: T14.1 (US1) in `tasks/T14-nhieu-nhan-va-van-ban.md`.

## Acceptance criteria
- AC1 — Given `dev` (3,812 rows) grouped by `group_id`, When the majority label per group is taken (dataset-diagnosis block A3), Then the `oracle == category` rate and the oracle macro-F1 are reported with n.
- AC2 — Given `f1_macro` from `artifacts/dl-cat-cafe-rnn-ce-s42/metrics.json`, When `ceiling = f1_macro / (1 − wrong_label_fraction)` is computed, Then every input, its source file and the result are written to `artifacts/eda/label-ceiling.json`; a ceiling above 1 is reported capped and flagged as outside the formula's validity.
- AC3 — Given the result, When it is documented, Then `docs/05-phan-tich-du-lieu.md` carries one sourced line with scope "fitted on `train`, scored on `dev`" that says the ceiling is an upper bound.
- AC4 — Given the work is read-only, When it finishes, Then `git diff` shows no change under `src/`, no run row in `04-results.md`, and `test.csv` was not read.
- AC5 — Given the numbers are final, When the story is closed, Then `docs/` and `docs/bao-cao/` are updated per Rules 1, 7 and 10 and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Any label, split or class change (Rules 2 and 5).
- Hand labelling (`hand-label-dev-errors`).

## Open questions
- `dev` has 4.20 % rows in multi-label groups against 56.04 % in `train` (TASKS.md §5, 2026-10-17), so A3 on `dev` may understate the noise. Report that next to the number? Recommended: yes.

## Decisions
- Dropped 2026-10-17 at the author's request: `group_id` is removed from analysis, training and documentation; the stored `category` is the ground truth. `scripts/measure_label_ceiling.py` and `artifacts/eda/label-ceiling.json` were deleted.
