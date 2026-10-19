---
model: category
dataset_version: v8-minhash-1label (source sha256 85862b06…, `scripts/build_dedup_splits.py --minhash --drop-multi`, seed 20260826)
status: not-ready
grade: F
updated: 2026-10-17
---
# Data — category @ v8-minhash-1label

Rough filter first (rows identical in all 18 columns, label included: 48,092 → 47,707), then MinHash on the BERT input (title . description . requirements after NFC, tone marks, abbreviations; lowercased; 5-character shingles, 128 permutations, Jaccard ≥ 0.9) over all 47,707 rows. Rows joined by MinHash form clusters: 33,052. Clusters with two or more labels (8,960) are **dropped**; each remaining cluster keeps its first row → **24,092** rows. Split with the same two-stage algorithm and seed (8:2, then 9:1), each cluster its own unit → `data/processed/splits-minhash-1label/`, with the 18 raw columns kept as `raw_*`. Procedure in `cach-xet-trung.md` (to be updated for this version).

Scope: three text columns, the label, and the contract-type and country columns as slices. All splits read for counts and cross-split duplicate checks only; no manual look at `test`.

## Numbers
| Item | Value | Limit |
|---|---|---|
| train / dev / test | 17,350 / 1,927 / 4,815 | — |
| imbalance (max/min) | train 24.9 · dev 25.6 · test 24.8 | ≤ 10, or a plan |
| classes with < 30 samples | dev: `nông_nghiệp_năng_lượng_môi_trường` 12, `nhóm_nghề_khác` 19, `ngôn_ngữ_dịch_thuật` 20 · test min **31** | test ≥ 30 |
| exact duplicates across splits | 0 | 0 |
| MinHash near-duplicate pairs across splits (Jaccard ≥ 0.9) | **0** | 0 |
| chi-square p · PSI, dev and test vs train | 1 · 0.000 | p ≥ 0.01 · PSI < 0.25 |
| clusters with ≥ 2 labels | 0 (dropped: 8,960 of 33,052) | — |
| Cohen's κ | ⛔ no second annotator | ≥ 0.6 |
| noise rate | not re-measured on this version; on `v7-1label26146`: wrong 23/200 = 11.5 % [7.79, 16.66] | upper < 10 % |
| test MDE, accuracy at p = 0.65 | test (n = 4,815) 0.0272 | ≤ δ (δ not set) |
| length outliers · HTML residue | 120 · 1 | note |
| email · URL · phone in the three text columns | 0 · 0 · 0 | 0 |
| `Toàn thời gian` · `Việt Nam` share of test | 95.7 % · 99.3 % | accepted |

Samples behind every row: `v8-minhash-1label.issues.csv` (id, check, detail).

## Notes — what dropping the multi-label clusters changed
Rows lost per class against the 34,742-row version (`splits-dedup-norm`): `du_lịch…` 57.6 %, `nông_nghiệp…` 51.9 %, `giáo_dục…` 46.1 %, `xây_dựng…` 41.1 %, `marketing…` 36.9 %. The kept rows are postings whose near-copies all agree on one label. The ambiguous postings are gone, so scores on this version describe unambiguous postings only and are not comparable to `docs/04-results.md` or to earlier versions. `nông_nghiệp…` has 31 rows in `test`, one above the limit of 30.

## Manual look
Not repeated: the rows are a subset of `v6-dedup-norm34742-18col` and `v7-1label26146`, whose manual look found no content-free cue. The four largest MinHash clusters (100, 27, 23, 21 rows) were read: each is one repeated posting (same description and employer, titles differing by a few words), not an over-merged chain.

## Checks
D1 ✅ · D2 ❌ F1 · D3 ✅ · D4 ⚠ unjudged (no δ) · D5 ✅ (test min 31; imbalance 24.8 has a plan) · D6 ❌ F2 (accepted) · D7 ✅ · D8 ❌ F3 (waived) · D9 ❌ F4 (noted) · D10 ✅ · D11 ⚠ not run · D12 ⚠ not run · D13 ❌ F5 (accepted)

## Findings
| ID | Check | Severity | Finding → fix |
|---|---|---|---|
| F1 | D2 | critical | the paper reference and the licence of `dinhieufam/VietJobs` are not recorded; the personal-data handling is not stated; 0 emails, URLs, phones in the text → **deferred by the author 2026-10-17** (`problem.md` Decisions); stays critical until written, so the grade stays F |
| F2 | D6 | major | noise rate on `v7` was 11.5 % [7.79, 16.66], upper bound above 10 %; not re-measured on this version → **accepted by the author 2026-10-17** (`problem.md` Decisions); re-measure on this version if a number for the thesis is needed |
| F3 | D8 | minor | no second annotator → **waived by the author** |
| F4 | D9 | minor | 120 length outliers and one HTML-like heading, in `issues.csv` `D9-outlier`, `D9-html` → **resolved by a note, data kept** (`problem.md` Decisions) |
| F5 | D13 | major | slices with < 30 test rows in `contract_type` and `country` → **accepted by the author 2026-10-17** (scope is job postings in Vietnam) |

Not run: suspect labels from a baseline (D11), production gap (D12).

## Grade
- `nm-data` rule (accepted findings still count): **F**, from F1 (critical, deferred).
- Author's rule, 2026-10-17 (accepted, deferred or waived findings do not count): F1, F2, F3, F4, F5 are all waived, nothing remains: grade **A**.
