---
slug: cafebert-preprocess-ablation
epic: label-noise-and-text
kind: feature
depends_on: [class-centroid-cosine]
status: draft
version: 1
updated: 2026-10-17
---
# Ablate NFC, tone marks and abbreviations on the CafeBERT path

As a **thesis author**, I want **an on/off ablation row for each of three preprocessing steps on the CafeBERT path**, so that **each step on that path is backed by a measurement or removed (Rule 2 item 4)**.

Board row: T14.7 (US7). `docs/02-vietnamese-nlp.md` has no ablation numbers for CafeBERT today. Gated like `strip-company-intro`.

## Acceptance criteria
- AC1 — Given NFC, tone marks and abbreviations, When each is switched off alone, Then three-seed (42–44) runs on `device=cpu` exist for each, plus the all-on baseline.
- AC2 — Given the runs, When they are logged, Then each step has one ablation row (on vs off) in `04-results.md`.
- AC3 — Given the rows, When a step does not win, Then it is marked "dropped" and removed from the path (Rule 2 item 4).
- AC4 — Given the rows, When documented, Then the ablation table in `docs/02-vietnamese-nlp.md` is filled with the CafeBERT numbers.
- AC5 — Given the runs are final, When the story is closed, Then Rules 1, 7, 10 and 13 are followed, `scripts/export_runs.py` is re-run and `pytest -q` is green.

**INVEST** — I ✅ · N ✅ · V ✅ · E ✅ · S ✅ · T ✅

## Out of scope
- Word segmentation and the salary mask (already on the PhoBERT path).

## Open questions
- None.

## Decisions
