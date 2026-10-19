---
model: category
dataset_version: v5-dedup-norm34742 (source sha256 85862b06…, built by `scripts/build_dedup_splits.py --norm`, seed 20260826)
status: not-ready
grade: F
updated: 2026-10-17
---
# Data — category @ v5-dedup-norm34742

One row per distinct key. Key = the BERT input (title . description . requirements after NFC, tone marks, abbreviations) lowercased, with all punctuation, special characters and whitespace removed; Vietnamese diacritics kept. 47,707 rows after exact dedup → **34,742**. A text with several labels keeps the label of the first row in the raw file. Split: same two-stage algorithm and seed as v2, each text its own unit → `data/processed/splits-dedup-norm/`. The procedure is in `cach-xet-trung.md`.

Scope: only the three text columns (`job_title`, `description`, `requirements_text`) and the label. Other columns, such as province, are out of scope at the author's request. All three splits read for counts and cross-split duplicate checks only; manual look on `train`. `datasketch` is not installed: near-duplicates use the punctuation/case fallback, not MinHash.

## Numbers
| Item | Value | Limit |
|---|---|---|
| train / dev / test | 25,014 / 2,778 / 6,950 | — |
| imbalance (max/min) | train 22.0 · dev 22.0 · test 21.8 | ≤ 10, or a plan |
| classes with < 30 samples | dev: `ngôn_ngữ_dịch_thuật` 28, `nông_nghiệp_năng_lượng_môi_trường` 26, `nhóm_nghề_khác` 21 · test min 53 | none |
| exact duplicates across splits | 0 | 0 |
| keys in > 1 split (lowercase, no punctuation, no whitespace) | 0 | 0 |
| groups in > 1 split | no group column (each text is its own unit) | 0 |
| chi-square p · PSI, dev and test vs train | 1 · 0.000 | p ≥ 0.01 · PSI < 0.25 |
| texts with ≥ 2 labels (label picked by "first row") | 8,596 / 34,742 = 24.74 % | — |
| Cohen's κ · audited noise | ⛔ no second annotator, no audit | κ ≥ 0.6 · upper < 10 % |
| test MDE, accuracy at p = 0.65 | test (n = 6,950) 0.0227 · dev (n = 2,778) 0.0358 | ≤ δ (δ not set) |
| length outliers · HTML residue | 186 · 1 | note |
| PII in text (email · URL · phone) | 0 · 0 · 0 | 0 |

Samples behind every row: `v5-dedup-norm34742.issues.csv` (id, check, detail).

## Manual look
- 20 random `train` titles per class (seed 20260826). No content-free cue: no URLs, emails or phones; text length is spread alike across classes (median 966 characters over all of `train`, 99th percentile 2,756).
- The class follows the employer's field more than the job title. "Nhân Viên Kinh Doanh" sits in `công_nghệ_thông_tin…`, `du_lịch…`, `kỹ_thuật_điện…`, `y_tế…` (ids 12210, 20562, 23700, 22902). `nhóm_nghề_khác` mixes legal, HR, mechanical and customer-service titles (ids 23518, 23635, 23561, 23599). Accounting titles appear in `tài_chính…` and a few unrelated titles too (ids 24406, 24459, 24603).
- D9 outliers are long postings, 3,306 to 3,745+ characters against a normal range of 1–3,105 (ids `train-83`, `train-376`, `train-607`, `train-688`, `train-823`). The HTML residue is `<Yêu cầu bắt buộc>` (`train-12653`).
- `train.csv` rows are ordered by class. Harmless while the training loop shuffles.

## Checks
D1 ❌ F1 · D2 ❌ F2 · D3 ✅ · D4 ⚠ unjudged (no δ) · D5 ❌ F3 · D6 ❌ F4 · D7 ✅ · D8 ❌ F5 · D9 ❌ F6 · D10 ❌ F7 · D11 ⚠ not run (no out-of-fold predictions) · D12 ⚠ not run (no production sample) · D13 ✅ (three text columns: no cue found)

## Findings
| ID | Check | Severity | Finding → fix |
|---|---|---|---|
| F1 | D1 | critical | `problem.md` is `draft`, not approved → the author answers its open questions and approves it |
| F2 | D2 | critical | licence and personal-data handling of `VietJobs.csv` are recorded nowhere. The text scan found 0 emails, URLs and phones → the author writes source and licence into `problem.md` |
| F3 | D5 | major | `dev` has three classes under 30 rows (21, 26, 28), so their F1 on `dev` is noise → read `f1_macro_no_junk` and the larger classes, or confirm those classes on `test`. Imbalance 22.0 > 10 has a plan (macro-F1 headline, focal and class-weight runs) |
| F4 | D6 | major | 8,596 of 34,742 texts (24.74 %) carry a label chosen by "first row" among conflicting labels. This is a conflict rate, not an audited noise rate → audit ≥ 200 random rows, then `stats.py wilson`; or choose labels another way (a new version) |
| F5 | D8 | minor | no second annotator and no audit of ≥ 200 random labels |
| F6 | D9 | minor | 186 length outliers and one HTML residue (`train-12653`), in `issues.csv` `D9-outlier`, `D9-html` → keep or clean |
| F7 | D10 | minor | the punctuation/case fallback found 0 cross-split keys, but MinHash (≥ 0.9) was not run → `pip install datasketch` and re-run |

Not run: MinHash near-duplicates (D10 full), suspect labels from a baseline (D11), production gap (D12).
