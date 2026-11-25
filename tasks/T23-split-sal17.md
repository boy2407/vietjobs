# T23 — split `splits-sal17`: salary-disclosed only, 17-column conflicts removed

**Goal.** A new split set for the salary task: from the raw file (47,707 rows after exact
de-duplication), drop every row whose 17 non-label columns are identical to another row's
but whose `category` differs (5,220 rows), keep salary-disclosed rows only (30,279), then
re-split train/dev/test 8:2 then 9:1 with seed 20260826.

**Author exception, 2026-11-24.** This re-splits `test`, which Rule 2 and Rule 5 forbid.
The author accepted that after being told the consequence: nothing in `docs/04-results.md`
is comparable to runs on this set. The original `data/processed/splits/` is untouched.

- [x] `scripts/build_sal17_split.py`
- [x] build `data/processed/splits-sal17/` and check counts, 0 groups straddling splits
- [x] mask salary in the 7 extra input columns (`scripts/mask_sal17_extra.py`)
- [x] join 11 columns into the salary input (`scripts/build_sal17_11c_split.py` → `splits-sal17-11c`)
- [x] decide with the author: exception recorded in AGENTS.md Rule 5 item 7; runs planned in [T23-train-lai](T23-train-lai.md)

**Done when** the three CSVs exist, `build_meta.json` counts match, and the author has
decided the next runs.

## Log

- 2026-11-24 — script written; build started (segmentation).
- 2026-11-24 — built: train 21,801 / dev 2,422 / test 6,056 (72.0 / 8.0 / 20.0 %), 0 groups straddle. 10,104 train rows still sit in `group_id` groups with ≥ 2 labels (loose definition, not removed).
- 2026-11-24 — author chose 11 input columns. Added `<col>_masked` for location_raw, qualifications_text, technical_skills_text, languages_text, experience_raw, working_hours, contract_type in all three CSVs; only 2 train `working_hours` cells changed. Regex re-check on the 11 masked columns: the 8 remaining hits in `description_masked` are user counts ("40 triệu người dùng"), not pay.
- 2026-11-24 — built `splits-sal17-11c`: 8 labelled extra fields appended to `requirements_masked` (+ `_seg`), 100% of rows got extra text. Salary input (segmented) median 264 words, p90 389–400; 0 rows with đồng/USD figures.
- 2026-11-24 — grilling: both tasks read `splits-sal17-11c`; AGENTS.md Rule 5 item 7 added; retraining planned in `T23-train-lai.md`.
