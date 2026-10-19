---
slug: salary-province-field
epic: label-noise-and-text
kind: feature
depends_on: []
status: draft
version: 1
updated: 2026-10-17
---
# Add a province field to the salary input

As a **thesis author**, I want **the salary model to read the posting's province**, so that **high-pay postings, where the residual is −8.97 on the >20M quintile, are no longer all pulled toward the mean**.

Board row: T14.8 (US8).

## Acceptance criteria
- AC1 — Given the `salary` task, When the province column is read, Then it goes through `features.resolve_column` and only a `*_masked` column is used (Rule 3).
- AC2 — Given the new column, When the leak tests run, Then it is listed in `_MASKABLE` and `UNMASKED_COLUMNS` and `tests/test_no_leak.py` and `tests/test_dl_text.py` are green.
- AC3 — Given seeds 42–44 on `device=cpu`, When the runs finish, Then a row in `04-results.md` reports `r2_log` and MAE on the >20M quintile against the baseline.
- AC4 — Given the runs are final, When the story is closed, Then Rules 1, 7, 10 and 13 are followed, `scripts/export_runs.py` is re-run and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Other new salary features; changing the loss.

## Open questions
- Re-encoding the salary vectors takes a GPU or MPS run; is that acceptable for one field? Recommended: yes, `encode.py` may stay on `mps`.

## Decisions
