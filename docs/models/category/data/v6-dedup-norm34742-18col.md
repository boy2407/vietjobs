---
model: category
dataset_version: v6-dedup-norm34742-18col (source sha256 85862b06…, `scripts/build_dedup_splits.py --norm`, seed 20260826)
status: not-ready
grade: F
updated: 2026-10-17
---
# Data — category @ v6-dedup-norm34742-18col

Same rows as `v5-dedup-norm34742` (34,742 texts, same key and split), now carrying the 18 original columns beside the cleaned ones (prefix `raw_`, 67 columns per file). Scope here: the three text columns, the label, and all 18 raw columns for PII, encoding and empties. The procedure is in `cach-xet-trung.md`.

Scope of reading: all three splits read for counts and cross-split duplicate checks only; manual look on `train`. `datasketch` is not installed, so near-duplicates use the punctuation/case fallback, not MinHash.

## Numbers
| Item | Value | Limit |
|---|---|---|
| train / dev / test | 25,014 / 2,778 / 6,950 | — |
| imbalance (max/min) | train 22.0 · dev 22.0 · test 21.8 | ≤ 10, or a plan |
| classes with < 30 samples | dev: `ngôn_ngữ_dịch_thuật` 28, `nông_nghiệp_năng_lượng_môi_trường` 26, `nhóm_nghề_khác` 21 · test min 53 | none |
| exact duplicates across splits | 0 | 0 |
| keys in > 1 split (lowercase, no punctuation, no whitespace) | 0 | 0 |
| chi-square p · PSI, dev and test vs train | 1 · 0.000 | p ≥ 0.01 · PSI < 0.25 |
| texts with ≥ 2 labels (label picked by "first row") | 8,596 / 34,742 = 24.74 % | — |
| Cohen's κ · audited noise | ⛔ no second annotator, no audit | κ ≥ 0.6 · upper < 10 % |
| test MDE, accuracy at p = 0.65 | test (n = 6,950) 0.0227 | ≤ δ (δ not set) |
| length outliers · HTML residue | 186 · 1 | note |
| email · URL · phone, over all 18 raw columns | 0 · 0 · 0 | 0 |
| mis-encoded text, over all 18 raw columns | 0 (20 pattern hits were `ĐÃ` in capitals, false positives) | 0 |
| empty cells: `languages_required` · `technical_skills` · `working_hours` · `soft_skills` | 73.4 % · 12.2 % · 10.3 % · 5.5 % | note |
| test rows per `contract_type` | Toàn thời gian 6,644 · Bán thời gian 124 · Thực tập 138 · 6 other values < 30 | ≥ 30 |
| test rows per `country` | Việt Nam 6,928 · 9 other values < 30 | ≥ 30 |

Samples behind every row: `v6-dedup-norm34742-18col.issues.csv` (id, check, detail).

## Manual look
- 20 random `train` titles per class (seed 20260826), seen on the same rows as v5. No content-free cue: no URLs, emails or phones in any raw column; text length is spread alike across classes.
- The class follows the employer's field more than the job title ("Nhân Viên Kinh Doanh" sits in `công_nghệ_thông_tin…`, `du_lịch…`, `kỹ_thuật_điện…`, `y_tế…`, ids 12210, 20562, 23700, 22902).
- `raw_requirements_text` and `raw_description` hold `ĐÃ` in capitals that looks like mojibake to a naive pattern (ids `train-3`, `train-1159`, `train-1873`); it is real Vietnamese. `train-16497` has a case typo, "xÃ hội", not an encoding fault.
- D9 outliers are long postings (ids `train-83`, `train-376`, `train-607`); the HTML residue is `<Yêu cầu bắt buộc>` (`train-12653`).

## Checks
D1 ❌ F1 · D2 ❌ F2 · D3 ✅ · D4 ⚠ unjudged (no δ) · D5 ❌ F3 · D6 ❌ F4 · D7 ✅ · D8 ❌ F5 · D9 ❌ F6 · D10 ❌ F7 · D11 ⚠ not run · D12 ⚠ not run · D13 ❌ F8

## Findings
| ID | Check | Severity | Finding → fix |
|---|---|---|---|
| F1 | D1 | critical | `problem.md` is `draft`, not approved → the author answers its open questions and approves it |
| F2 | D2 | critical | licence and personal-data handling of `VietJobs.csv` are recorded nowhere. The scan of all 18 raw columns found 0 emails, URLs and phones → **skipped by the author 2026-10-17** (recorded in `problem.md` Decisions); stays critical, so the grade stays F |
| F3 | D5 | major | `dev` has three classes under 30 rows (21, 26, 28) → read `f1_macro_no_junk` and the larger classes, or confirm those classes on `test`. Imbalance 22.0 > 10 has a plan (macro-F1 headline, focal and class-weight runs) |
| F4 | D6 | major | 8,596 of 34,742 texts (24.74 %) carry a label chosen by "first row" among conflicting labels; a conflict rate, not an audited noise rate → audit ≥ 200 random rows, then `stats.py wilson`; or choose labels another way (a new version) |
| F5 | D8 | minor | no second annotator and no audit of ≥ 200 random labels |
| F6 | D9 | minor | 186 length outliers and one HTML residue (`train-12653`), in `issues.csv` `D9-outlier`, `D9-html` → keep or clean |
| F7 | D10 | minor | the fallback found 0 cross-split keys, but MinHash (≥ 0.9) was not run → `pip install datasketch` and re-run |
| F8 | D13 | major | slices with < 30 test rows: 6 of 9 `contract_type` values and 9 of 10 `country` values; `Toàn thời gian` is 95.6 % of test and `Việt Nam` 99.7 %. The task does not evaluate by these columns → **accepted by the author 2026-10-17** (scope is job postings in Vietnam; recorded in `problem.md` Decisions); the finding stays, the grade rule is unchanged |

Not run: MinHash near-duplicates (full D10), suspect labels (D11), production gap (D12).
