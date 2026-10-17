# T14 — label noise, Vietnamese text and salary follow-ups (user stories)

Detail sheet for board section **T14** of [TASKS.md](../TASKS.md). Same rules as
[T10-salary-label-compare.md](T10-salary-label-compare.md): the author (Nghĩa) assigns
tasks, an agent fills *Evidence* before `done` and appends to the log here and in
TASKS.md §5. Status lives **only here** (TASKS.md §3 Rule 7). Written in English.
Created 2026-10-17. All rows `todo`. `test` is never read or transformed (Rule 5);
no label or split change without the author's explicit approval (Rule 2, Rule 5).

## 1. Board

### A — labels (category)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T14.1 (US1) | Estimate the label-noise ceiling with dataset-diagnosis block A3 on `dev` | todo | — | — | `ceiling = f1_macro / (1 − wrong_label_fraction)` written with its source: the A3 output (`oracle == category` rate and oracle macro-F1 on `dev`) and `f1_macro` from `artifacts/dl-cat-cafe-rnn-ce-s42/metrics.json`; result in `artifacts/eda/label-ceiling.json` and one line in `docs/05` |
| T14.2 (US2) | Hand-label 150 `dev` errors of `dl-cat-cafe-rnn-ce-s42` as model error / wrong original label / ambiguous | todo — **human only, never dispatchable** | Nghĩa | T14.1 | `artifacts/eda/hand-label-150.csv` (row id, true, predicted, verdict) filled by the author; counts per verdict with the sampling seed written next to them |
| T14.3 (US3) | Cosine between class centroids (block A6) for du_lich→kinh_doanh, thiet_ke→xay_dung, du_lich→san_xuat | todo | — | — | table of the three pairs, symmetric cosine, and each pair's confusion count in the run's `dev` predictions; a pair with cosine > 0.80 gets a label verdict (taxonomy error vs model error, A6 rule) |
| T14.4 (US4) | Decide handling of the junk class `nhom_nghe_khac` (F1 0.000, n = 25 on `dev`) | todo — **human decision** | Nghĩa | T14.2, T14.3 | decision recorded in this sheet's log. Any label/split change needs explicit author approval, would invalidate every row of `04-results.md`, and leaves `test` untouched (Rule 5) |

### B — Vietnamese text, CafeBERT path

Each row: ablation row in `04-results.md`, 3 seeds (42–44), compared against the probe baseline macroF1 0.5297 and the CafeBERT dense/RNN baselines. **Gated on the outcome of T14.1 / T14.3**: if the ceiling or a symmetric pair shows label noise dominates, a text step cannot buy back the gap and these stay `todo`.

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T14.5 (US5) | Strip company-intro boilerplate from the text the CafeBERT path reads | todo | — | T14.1, T14.3 | 3 runs + one ablation row in `04-results.md` vs baseline; step kept only if it wins, else recorded "dropped" in `docs/02` (Rule 2 item 4) |
| T14.6 (US6) | Field order under the 512-token limit (which field is truncated first) | todo | — | T14.1, T14.3 | 3 runs per order + ablation rows; truncation share per field measured |
| T14.7 (US7) | Ablation rows for NFC / tone marks / abbreviations on the CafeBERT path (`docs/02` carries no ablation numbers for it) | todo | — | T14.1, T14.3 | one ablation row per step (on vs off), 3 seeds; `docs/02` table filled |

### C — salary

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T14.8 (US8) | Add a province field to the `salary` input through `features.resolve_column` (Rule 3, `*_masked` only) | todo | — | — | 3 seeds; `r2_log` and MAE on the >20M quintile vs the baseline (residual −8.97) in a `04-results.md` row; `tests/test_no_leak.py` + `tests/test_dl_text.py` green with the new column listed in `_MASKABLE` and `UNMASKED_COLUMNS` |
| T14.9 (US9) | Record CafeBERT+RNN vs Dense and vs PhoBERT+RNN on `salary` in docs | todo | — | — | docs/06 and bao-cao ch.4 state: win vs Dense p = 0.000 on 3 seeds; vs PhoBERT-RNN p = 0.076 / 0.246 / 0.000 = unclear; each p-value traced to its source file |

### D — consistency and product

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T14.10 (US10) | Reconcile the two CafeBERT+RNN `category` means (0.6152 vs 0.6097) into one sourced number | todo | — | — | one number, computed from `artifacts/dl-cat-cafe-rnn-ce-s4{2,3,4}/metrics.json` (the `04-results.md` rows give 0.6172 / 0.6223 / 0.6060); the other value's origin explained; every doc quotes the one number |
| T14.11 (US11) | Report top-3 accuracy 0.9373 as a product result | todo | — | T5.1 | number reproduced by the inference entry point from T5.1 on `dev` (not `test`); sourced row in docs and bao-cao ch.5 |
| T14.12 (US12) | Upkeep after each story: docs, bao-cao, `export_runs.py` | todo | — | each story, when it finishes | Rules 1, 7, 10, 13 tables followed; `.venv/bin/python scripts/export_runs.py` re-run; `pytest -q` green |

## 2. Plan for group A (T14.1 and T14.3 executable; T14.2 and T14.4 human)

Read-only measurement. No code in `src/`, no training, no `04-results.md` rows (no model is trained; these are descriptive artefacts). Scripts live in the scratchpad, not in the repo.

**Spec gaps found (resolve before step 1):**
1. SKILL.md snippets read `splits/*.parquet`; the repo has `data/processed/splits/{train,dev,test}.csv`. Use `D.load_split("dev")` / `pd.read_csv` on train and dev only.
2. A6's snippet loads `train-raw-len256.npy` (PhoBERT, per the skill) while the run under diagnosis is CafeBERT. Available: `cafebert-train-raw-len256.npy` (and `len512`). Decide which vectors; default CafeBERT `raw-len256`, and also PhoBERT if the file exists. These are encoder-level vectors per posting; an A6 centroid is a mean of those. State the choice in the output.
3. `metrics.json` has no `top_confusions`; derive them from `artifacts/dl-cat-cafe-rnn-ce-s42/predictions_dev.parquet`.
4. Rule 2 of the skill says A3 may read `dev`; A6 uses `train` only. Neither touches `test`.

| # | Step | Command / file | Verify |
|---|---|---|---|
| 1 | Write `<scratchpad>/a3.py` from SKILL.md block A3: majority label per `group_id` on `dev`, `oracle == category` rate, oracle macro-F1; `wrong_label_fraction = 1 − that rate` | `.venv/bin/python <scratchpad>/a3.py` (`PYTHONPATH=src`) | prints 3 numbers; dev n = 3,812; oracle rate is in (0, 1] |
| 2 | Compute `ceiling = f1_macro / (1 − wrong_label_fraction)` with `f1_macro` = `metrics.metrics.f1_macro` of `dl-cat-cafe-rnn-ce-s42` (0.6172) | same script | every input printed with its file; if ceiling > 1, report it capped and say the formula is not valid there |
| 3 | Write `artifacts/eda/label-ceiling.json` (inputs, outputs, sources) and one sourced line in `docs/05-phan-tich-du-lieu.md` (Vietnamese, comma decimals, scope "fitted on `train`, scored on `dev`") | edit | `git diff --stat` touches only those; `pytest -q` green |
| 4 | Write `<scratchpad>/a6.py`: centroids of the three pairs from train vectors (block A6), cosine, plus the run's confusion counts for the same pairs from `predictions_dev.parquet` | `PYTHONPATH=src .venv-dl/bin/python <scratchpad>/a6.py` (needs only numpy/pandas/sklearn; `.venv` works too) | row count of vectors == train rows (34,354); 3 cosines in [−1, 1] |
| 5 | Apply the A6 rule per pair: > 0.80 **and** pair in the confusion list (two indicators, SKILL Rule 3) → label finding; 0.65–0.80 → feature candidate; < 0.50 → model error. Write the table to `artifacts/eda/class-centroid-cosine.json` and into `docs/05` | edit | each verdict names both indicators; no `⛔` left for these numbers |
| 6 | Sample for the human: write `<scratchpad>/hand-label-150-template.csv`, 150 `dev` rows where `pred != true`, fixed seed (e.g. 20260826), columns id, title, true, pred, empty `verdict`. Hand to Nghĩa | script | 150 rows, all errors, ids unique. Do not fill it |
| 7 | Update this sheet (rows, log), TASKS.md §5 line, `docs/00` status cell and `docs/09` if cited; bao-cao ch.3 §3.6 (Rules 1, 7) | edit | Rule 10 five contents present or `⛔` one-liners |

**Disjoint files:** T14.1 writes `label-ceiling.json`; T14.3 writes `class-centroid-cosine.json`. Both then edit `docs/05` and this sheet, so run them in sequence or merge those two edits last (one writer per file). No overlap with `train_dl.py`, `tests/test_dl_heads.py`, or `TASKS.md` working-tree edits already pending.

**Risks:** A3 on `dev` can overstate noise because `dev` shares only some groups with `train` (a prior diagnosis measured 56.04 % of `train` rows in multi-label groups vs 4.20 % of `dev`; see TASKS.md §5, 2026-10-17); the ceiling formula assumes errors are only label noise (an upper-bound estimate, say so). A6 cosine on mean-pooled vectors may sit high for every pair (check a control pair). Any numbers that cannot be re-measured get `⛔` rather than a stale value.

**Human must decide:** T14.2 labelling of 150 rows; T14.4 whether to merge/relabel/drop `nhom_nghe_khac` or only report `f1_macro_no_junk`; whether a rebuild (invalidating `04-results.md`) is ever acceptable; which vectors A6 uses (gap 2).

## 3. Log (append-only)

| Date | Who | What happened |
|---|---|---|
| 2026-10-17 | Claude | Created the sheet: T14.1–T14.12 from the author's user stories US1–US12, all `todo`; wrote the group A plan. No code, no runs |
