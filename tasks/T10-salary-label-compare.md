# T10 — is the 37K salary label the same quantity as the training label?

Detail sheet for board section **T10** of [TASKS.md](../TASKS.md). Same rules as
[T9-salary-37k.md](T9-salary-37k.md): the author (Nghĩa) assigns tasks, an agent fills
*Evidence* before `done` and appends to the log here and in TASKS.md §5. Written in English.

## 1. Goal

Decide whether `data/external/vietjobs37k/ext37k-sal.csv` can score the `salary` model.
That needs the 37K label (regex over posting text, `external.extract_salary`) to be the same
quantity as the training label (platform-supplied `salary_min`/`salary_max` in `VietJobs.csv`).

## 2. What the code says (read 2026-10-07)

| | Training (`VietJobs.csv`) | `ext37k-sal.csv` |
|---|---|---|
| Source of bounds | platform structured columns | regex over title + description + requirements (`external.py:412`) |
| Filter | none (93 disclosed rows < 2 tr, 13 > 200 tr kept) | refuses USD, non-monthly, outside [2, 200] |
| Ladders | as given | merged into min–max |
| Arithmetic | `dataset.derive_salary` | same function |
| Text | `*_masked` for the model | unmasked (answer leaks) |

## 3. Board

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T10.1 | Run the 37K extractor on `dev` postings, compare with the platform salary | done | Claude | — | below §4 |
| T10.2 | Distribution check: train disclosed `salary_mid` in [2, 200] vs `ext37k-sal.csv` (median, quantiles, `is_range`, per source) | todo (only if the author wants to go on) | — | T10.1 | — |
| T10.3 | Evaluation recipe: `mask_salary` on 37K text, drop the ~2 % overlap, encode `masked` family, score `dl-sal-*` | blocked (T10.1 gate failed) | — | T10.1, T10.2 | — |
| T10.4 | Extractor: when a posting states both base pay (`lương cứng/cơ bản/cố định`) and income (`thu nhập`), keep the income range (platform convention). Target ≥ 90 % exact on rows read where the platform salary is in the text | done — target **not met**; `ext37k*.csv` not rebuilt | Claude | T10.1 | see §4b |
| T10.5 | Score MTL `dl-mtl-cafe512-a5-e80` s42–44 on `ext37k-sal.csv` (masked text, crosswalk draft), both heads; artifacts only, no `04-results.md` row (author) | done — provisional | Claude | T10.4 | §4c |

Gate: ≥ ~90 % of overlapping rows within ±10 % → 37K label is a fair stand-in. Otherwise stop.

## 4. Evidence — T10.1 (2026-10-12)

Read-only scratchpad script; text = title + description + requirements + benefits (unmasked); `D.load_split("dev")`.

- Dev 3,812 rows; platform disclosed 2,698; the regex finds a figure in 269.
- 91.4 % of platform-disclosed rows state no figure in the text.
- Rows with both (n = 232; random 200 in brackets): exact min & max 24.1 % [22 %] · mid ±10 % 31 % [28.5 %] · ±20 % 43.5 % [41 %] · MAE 6.52 tr [6.68 tr].
- 37 rows where the regex reads a figure but the platform says 0.
- Pattern in the disagreements: text says base pay ("lương cứng/cơ bản 7 triệu"), platform holds the advertised total (e.g. 10–15 tr).
- Re-run on `train` too (2026-10-12, same script): train+dev 38,166 rows, platform disclosed 27,320, regex reads 2,709, both 2,273 → exact min & max 20.5 % · ±10 % 27.4 % · ±20 % 38.1 % · MAE 10.60 tr; 436 rows regex has a figure where the platform says 0. `test` not read.
- **Gate failed** (31 % ≪ 90 %): the two labels are not the same quantity. 
- Not verified: the sanity check from the plan (reproduce 6,328 disclosed rows on `ext37k.csv`).

## 4b. Evidence — T10.4 (2026-10-12)

Subset: 379 train+dev postings whose platform min/max literally appear in the text.
- Before: script reads 293 (77.3 %); exact min & max 222 = 75.8 % of rows read; ±10 % 81.9 %.
- After (`_BASE_PAY` / `_INCOME_WORD` in `external._scan`, 4 new tests, `pytest -q` 175 passed): exact 234 = **79.9 %**; ±10 % 83.3 %.
- On the 48 rows stating both base and income: platform = income range 25, = base-min..income-max 13, other 10. The platform has no single convention.
- Remaining 59 mismatches: platform keeps only the top of a base range ("lương cứng 8-10 tr" → 10–10), mixes base and income, or holds a figure the text does not give.
- All train+dev overlap (n = 2,273): exact 20.5 % → 21.3 %, ±10 % 27.4 % → 27.9 %.
- Not done: `scripts/build_ext37k.py` rebuild and the 199-row review re-check — the code now reads differently from the files on disk.

## 4c. Evidence — T10.5 (2026-10-12)

- `scripts/build_ext37k.py` now writes `job_title_masked / description_masked / requirements_masked` into `ext37k-sal.csv` (`V.preprocess(..., mask=True)`, same as `dataset.clean`). Files rebuilt: 6,161 rows, 86 rows' bounds changed by T10.4.
- Mask leak (extractor still reads a figure after masking, title+desc+req): 48k dev 6/36 (16.7 %), 37K 1,338/6,161 (21.7 %). Same `mask_salary` as training; not changed.
- `scripts/eval_external_mtl.py`. Checks: re-encoding 32 dev rows matches the training cache (cosine 1.00000); loading s42 checkpoints reproduces dev macroF1 0.6175 / MAE 4.16 from `metrics.json`.
- 6,161 → 50 overlapping with our corpus dropped → 6,111 scored for salary; category scorable 6,096 (strict 2,812).

| s42–44, mean ± sd | strict macroF1 | strict acc | lenient hit | MAE (tr) | R²log | Spearman |
|---|---|---|---|---|---|---|
| 48k `dev` | 0.6103 ± 0.0085 | 0.6414 ± 0.0038 | — | 4.146 ± 0.027 | 0.5245 ± 0.0027 | 0.6950 ± 0.0009 |
| 37K `ext37k-sal` | 0.5089 ± 0.0182 | 0.5514 ± 0.0111 | 0.6816 ± 0.0126 | 5.496 ± 0.145 | 0.3147 ± 0.0261 | 0.5765 ± 0.0242 |

Provisional: crosswalk is `draft` (T7.1); 37K salary is silver (T10.1/T10.4), so the salary drop mixes model error and label mismatch. Artifacts: `artifacts/dl-mtl-cafe512-a5-e80-s4{2,3,4}/external_37k/sal/`.

## 4d. Evidence — T10.6 (2026-10-12): strict 37K mask + demo model

- Author asked to mask **all** pay information in 37K. New `external.mask_salary_strict` (37K only, applied after the training mask in `build_ext37k.py`): masks every money amount (M / tr / triệu / k / `8tr5` / dong-written / de-id remnant), then loops on `extract_salary` until nothing is readable. 16 tests; `pytest -q` 191 passed.
- After rebuild: extractor reads 0/6,161 masked rows (was 1,338); all 6,161 contain `<SALARY>`; a sweep of money-shaped strings left 70, all heights / addresses / counts plus one date and "190K khách".
- Stricter than the training mask (48k dev still leaks ~17 %), so 37K input is slightly off the training distribution.
- `vietjobs-demo/model/` = `dl-mtl-cafe-rnn-a5-s42` (same `best.pt` checksum). RNN path added to `scripts/eval_external_mtl.py`; check: 48 dev rows reproduce `predictions_dev.parquet` (category 100 %, |Δ salary| ≤ 0.0031 tr).

| Model | Set | strict macroF1 | strict acc | lenient hit | MAE (tr) | R²log | Spearman |
|---|---|---|---|---|---|---|---|
| dense s42–44 | 48k dev | 0.6103 ± 0.0085 | 0.6414 | — | 4.146 | 0.5245 | 0.6950 |
| dense s42–44 | 37K | 0.5058 ± 0.0157 | 0.5494 | 0.6800 | 5.463 ± 0.147 | 0.3176 | 0.5793 |
| RNN s42 (demo) | 48k dev | 0.6063 | 0.6577 | — | 4.021 | 0.5735 | 0.7274 |
| RNN s42 (demo) | 37K | **0.5957** | 0.6401 | 0.7544 | **4.546** | 0.4696 | **0.7120** |

Provisional (draft crosswalk, silver salary, one RNN seed). Workbook: `data/eda_xlsx/danh_gia_37k_mtl.xlsx`.

## 5. Log

| Date | Who | Entry |
|---|---|---|
| 2026-10-12 | Claude | Opened T10 at the author's request; ran T10.1 (above). No `src/`, `data/` change. |
| 2026-10-12 | Claude | T10.1 extended to `train` (author's request): 27.4 % within ±10 % on 2,273 rows — same conclusion. |
| 2026-10-12 | Claude | T10.4: income-over-base rule in `external._scan`; exact 75.8 % → 79.9 % on the 379-row subset, target 90 % not met; data files not rebuilt. |
| 2026-10-12 | Claude | T10.5: masked columns in `ext37k-sal.csv`, `scripts/eval_external_mtl.py`, MTL s42–44 scored on 37K (provisional). |
| 2026-10-12 | Claude | T10.6: `mask_salary_strict` for 37K, rebuilt files, re-scored dense s42–44 and the demo RNN model; workbook regenerated. |
