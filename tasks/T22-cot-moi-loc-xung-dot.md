# T22 — salary-disclosed train, conflicting-label filter, extra columns, oversampling

**Goal.** Measure, on one shared row set for both tasks, whether (a) dropping `train`
rows whose three input columns are identical but whose `category` differs, (b)
adding the structured and short-text columns, and (c) repeating rows of small classes
instead of class weights, change `dev` scores for CafeBERT 512 + Dense. Agreed with the
author in the 2026-11-23 grilling session.

**Fixed before any number is read.**

- Encoder CafeBERT 512, `masked_mean`, last layer; Dense head; `cpu`; seeds 42–44.
- `category`: CE, class weight effective number β = 0.999, label smoothing 0.05.
  `salary`: SmoothL1 β = 0.1 with effective-density sample weights (`--class-weight`).
- `train` keeps only salary-disclosed rows (24,622). `dev` keeps only salary-disclosed
  rows (2,698) for both tasks; `dev` is not conflict-filtered. `test` is not read.
- Conflict = same `job_title`, `description`, `requirements_text` (exact) with ≥ 2
  distinct `category` values, computed on the filtered `train`. 12,580 rows remain.
- A difference below 0.003 macro-F1 is noise (Rule 13). Effects are read as paired Δ
  per seed against R0.

| ID | Run | `train` | Extra columns | Imbalance | Status |
|---|---|---|---|---|---|
| T22.0 | code + tests (`dl/extra.py`, flags in `train_dl.py`) | — | — | — | done |
| T22.1 | encode `technical_skills_text`, `benefits_masked` with CafeBERT len 128 (`train`, `dev`) | — | — | — | done |
| T22.2 | R0 `dl-{cat,sal}-t22-r0-s{42..44}` | disclosed | none | as above | done |
| T22.3 | R1 `dl-{cat,sal}-t22-r1-s{42..44}` | disclosed, no conflict | none | as above | done |
| T22.4 | R2 `dl-{cat,sal}-t22-r2-s{42..44}` | disclosed | all | as above | done |
| T22.5 | R3 `dl-{cat,sal}-t22-r3-s{42..44}` | disclosed, no conflict | all | as above | done |
| T22.6 | R4 `dl-cat-t22-r4-s{42..44}` (category only) | best of R0–R3 | best of R0–R3 | class weight off, small classes repeated up to N | done |
| T22.7 | log, docs, thesis, workbook (Rules 1, 7, 10, 13) | — | — | — | todo |
| T22.8 | R6-RNN `dl-sal-t22-r6rnn-{phobert,cafebert,xlmr,visobert,bamibert}-s42` (rental 27) | — | 6 fields as text (`splits-r6text`) | BiGRU-LSTM-CNN, seed 42 | dropped |

**Extra columns** (fitted on `train` only, then applied to `dev`):

| Column | Encoding |
|---|---|
| `experience_months` | `log1p` |
| `contract_type` | 6 flags (multi-valued strings set several) |
| `province` | one-hot of provinces with ≥ 100 `train` rows, plus `other` |
| education, from `qualifications_text` | lowest level named (0–5) + `unknown` flag |
| `working_hours` | days per week, daily span in hours, Saturday, Sunday, shift, unparsed |
| `languages_text` | flags English, Chinese, Japanese, Korean, other |
| `technical_skills_text` | CafeBERT vector, len 128 |
| `benefits_masked` | CafeBERT vector, len 128 (masked for both tasks, Rule 3) |

Earlier conflict-free split sets (`splits-dedup-norm-1label`, `v7-1label26146`)
re-split the whole corpus and therefore changed `test`; no run in `04-results.md`
uses them. This task filters `train` only and keeps the frozen splits.

## Log

| Date | Who | What |
|---|---|---|
| 2026-11-23 | Claude | Sheet created after the author's "ok"; nothing run yet |
| 2026-11-23 | Claude | R0–R4 run on Mac `cpu` (27 runs, `04-results.md`). PhoBERT xf caches not built (rental 25 too slow). Test scored once from saved `best.pt` by the author's override (`scripts/score_t22_test.py`, run by the author; `test.csv` sha256 70ba98d5… unchanged), 27 rows tagged test-informed. Results in sheet tabs ` Dashboard-v2` and `Score_v2`. T22.7 docs/thesis still open |
| 2026-11-23 | Claude | R6-L4 `dl-sal-t22-r6l4-s{42..46}` (rental 26): dev MAE 4,075 ± 0,027, better than Last4 on 5/5 seeds. Logged in 04-results, 06 §5.10, bao-cao §4.4.6, 00, workbook, sheet ` Dashboard-v2` row 13. Test not scored |
| 2026-11-23 | Claude | R6-L4 for PhoBERT, ViSoBERT, XLM-R, BamiBERT (rental 26); only CafeBERT passes the rule. Test scored once by the author on vast (`score_r6l4_test.py`), all 25 runs; sheets ` Dashboard-v2` rows 13–17 and `Score_v2` written |
| 2026-11-24 | Claude | Added T22.8 (R6-RNN, 1 seed, author's choice). Rental 27 file, `scripts/vast_run_t22_r6rnn.sh`, and an `r6text` branch in `eval_sal_rnn.py` (test text built in memory, no ext37k) prepared before renting |
| 2026-11-24 | Claude | Author dropped R6 (R6-L4 and R6-RNN). R6-RNN stopped during BamiBERT; dev MAE before stop: phobert 3.839, cafebert 3.770, xlmr 3.940, visobert 3.850 (old RNN 4.047/3.840/3.984/3.972). R6-L4 removed from docs/06 §5.10, bao-cao §4.4.6, 00-tong-quan and the two sheet tabs; its 04-results rows stay (append-only). Artifacts deleted by the author |
