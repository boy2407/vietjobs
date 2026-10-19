---
slug: class-centroid-cosine
epic: label-noise-and-text
kind: feature
depends_on: []
status: done
version: 1
updated: 2026-10-17
---
# Measure cosine between confused class centroids

As a **thesis author**, I want **the cosine between the class centroids of three confused pairs**, so that **each confusion is read as a taxonomy error or a model error instead of a guess**.

Board row: T14.3 (US3). Pairs: du_lich→kinh_doanh, thiet_ke→xay_dung, du_lich→san_xuat.

## Acceptance criteria
- AC1 — Given the `train` vectors (34,354 rows) of the chosen encoder, When the centroid of each class in the three pairs is taken (block A6), Then the row count of the vectors equals the `train` rows and the three cosines lie in [−1, 1].
- AC2 — Given the `dev` predictions of `dl-cat-cafe-rnn-ce-s42`, When the confusions are counted, Then each pair shows its confusion count next to its cosine.
- AC3 — Given a control pair of unrelated classes, When its cosine is measured, Then it is reported beside the three, so a high cosine shared by every pair is visible.
- AC4 — Given both indicators, When the A6 rule is applied, Then each pair gets a verdict: cosine > 0.80 and in the confusion list → label finding; 0.65–0.80 → feature candidate; < 0.50 → model error. The table is written to `artifacts/eda/class-centroid-cosine.json`.
- AC5 — Given the numbers are final, When the story is closed, Then `docs/05-phan-tich-du-lieu.md`, `docs/bao-cao/` (Rules 1, 7, 10) are updated and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Any label or taxonomy change.

## Open questions
- Which vectors: CafeBERT `raw-len256` (the run under diagnosis) or also PhoBERT? Recommended: CafeBERT, plus PhoBERT if the file exists.

## Decisions
