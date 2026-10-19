---
model: category
task: nlp
status: approved
version: 3
updated: 2026-10-17
---
# Classify a Vietnamese job posting into one of 16 occupation classes

## Business goal
Who uses it, what decision it drives, what it replaces: **not stated anywhere in the repo** (see Open questions). Today the class is the posting's own `category` label, taken from the source site. The project's stated purpose is occupation classification and salary estimation from Vietnamese postings (AGENTS.md §0). Current way + its cost: ⛔ no cost measured.

## ML task
- Type: single-label classification, 16 classes.
- Input → output: `job_title` + `description` + `requirements_text` (raw family, Rule 3) → one of 16 `category` values; the model also returns class probabilities (top-3 accuracy is already logged).
- Costlier error: ⛔ not decided (open question). Until then no asymmetric metric is justified, so macro-F1 is the headline (the classes are skewed 25.5 : 1).
- Constraints: frozen encoder, head trained on `cpu` (Rule 13 item 5); encoder may use `mps`/`cuda`. Python 3.9 for `dl/`. `test` read once (Rule 5). Latency, hardware and privacy limits ⛔ not stated.

## Data exploration
Scope per Rule 8: descriptive numbers on the original file (47,707 rows); text and samples from `train` / `dev` only. `test` not read.

| Item | Finding |
|---|---|
| Sources + licence | Hugging Face dataset `dinhieufam/VietJobs`, file `VietJobs.csv`, 48,092 rows, 18 columns, sha256 `85862b06…` (`data/README.md`); the author says it comes from a published paper with about 48k rows. Paper reference ⛔ not recorded in the repo; licence ⛔ not recorded |
| Size | 47,707 rows, all labelled. Splits: train 34,354 / dev 3,812 / test 9,541 (seed 20260826) |
| Labels | 16 classes, from `artifacts/eda/summary.json`: largest `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` (5,466 in `train`), smallest `nông_nghiệp_năng_lượng_môi_trường` (198 in `train`). Imbalance 25.5 : 1. Junk class `nhóm_nghề_khác` = 0.72 % of rows. Who labels and any guideline ⛔ unknown |
| Shape | `train`, characters: title p50 25, p95 48; description p50 521, p95 1,308; requirements p50 358, p95 753. 0 % empty in all three. CafeBERT 512-token truncation: 4.45 % of `train` rows |
| Groups | Reposts of identical text are bound to one split by a hash key built when the splits were made; training does not read it |
| Manual look | 12 random `dev` titles: all plausible occupations (e.g. "Kế Toán Tổng Hợp" → tài chính kế toán, "Kiến trúc sư web" → công nghệ thông tin). The stored `category` is taken as ground truth |
| Gaps | `nhóm_nghề_khác` has F1 0.000 on `dev` (n = 25). `nông_nghiệp…` has 198 train rows. `du_lịch…` recall 0.4628 (n = 443) and `kinh_doanh…` recall 0.5840 (n = 786) are the two largest-support weak classes |

Verdict: **enough rows; the stored `category` is the ground truth (no relabelling).** Raw centroid cosines are 0.992–0.999 for every class pair, so they cannot separate overlapping classes from model error; after centring, the three confused pairs are 0.469 / 0.242 / −0.055 (`artifacts/eda/class-centroid-cosine.json`).

## Approaches
| # | Approach | Data needed | Cost (GPU / API) | Latency | Risk | Reference |
|---|---|---|---|---|---|---|
| A1 | No-model floor: always predict the largest class | none | ~0 | ms | useless as a product, only a floor | `artifacts/eda/summary.json` key `floors`: macro-F1 0.0214, accuracy 0.2062 |
| A2 | Linear probe (logistic regression, C = 1.0) on frozen PhoBERT vectors | existing PhoBERT cache | CPU, 34.7 s | ms | needs the 768-d vectors | `probe-cat-s2`: macro-F1 0.5898 · acc 0.6388 · top3 0.9258 |
| A3 | Frozen PhoBERT → Dense head | existing cache | CPU, 6.2 s | ms | head noise: differences < 0.003 are noise (Rule 13) | `dl-cat-ce-cpu-0920`: macro-F1 0.6030 · acc 0.6501; `dl-cat-s2` top3 0.9318 |
| A4 | Frozen CafeBERT 512 → Bi-GRU‖Bi-LSTM → CNN head | CafeBERT token cache (large) | cuda for the head (vast.ai) | ms at inference | not reproducible on `cpu` as run; sd 0.0084 over 3 seeds | `dl-cat-cafe-rnn-ce-s42/43/44`: mean macro-F1 0.6152 (sd 0.0084); s42 acc 0.6679, `f1_macro_no_junk` 0.6608 |

## Choice
- Baseline: **A2 (probe, 0.5898)** is the bar a model must clear on the same vectors, with A1 (0.0214) as the floor — both already the project's bars (Rule 12 item 2). Champion to beat: A4 (0.6152 ± 0.0084), ahead of A3 (0.6030) by 0.0122 (within about 1.5 sd of A4).
- Try first: A4 text-input variants (`strip-company-intro`, `field-order-512`, `cafebert-preprocess-ablation`), because the stored label is the ground truth and every model family so far sits in 0.59–0.62.
- Primary metric + first threshold: **macro-F1 on `dev`, three seeds, `device=cpu`**, with `f1_macro_no_junk` beside it. Threshold: gain over 0.6152 larger than the seed noise (sd 0.0084). The 0.003 figure is the noise limit for two single runs. A product threshold ⛔ needs the business goal.
- Dropped: none yet.

## Contract
Confirmed by the author 2026-10-17. There is no inference entry point in the repo today (`TASKS.md` T5.1), so this is what product code may stub.
- Interface: a Python call `classify_posting(job_title: str, description: str, requirements_text: str)` returning `{"label": str, "confidence": float, "top3": [{"label": str, "prob": float}, ...]}`. `label` is one of the 16 `category` values verbatim, `confidence` is the top-1 probability in 0–1, `top3` is sorted by `prob`. Input goes through the same text path as training (`dl/text.py`, `features.resolve_column`, Rule 4).
- Limits: input is the three raw text fields, cut at 512 tokens by the CafeBERT path. No p95 latency target (not measured, no serving yet). Low confidence: the caller shows `top3` instead of a single answer when `confidence < 0.5`.
- Owner: the author (model side); product code reads `top3` and `confidence` only.

## Open questions
- None.

## Decisions
- Contract-type and country skew in the data (`Toàn thời gian` 95.6 % and `Việt Nam` 99.7 % of `test`): accepted. The scope is classifying and estimating salary for job postings in Vietnam, so results are not reported per contract type or per country (2026-10-17)
- Source of `data/raw/VietJobs.csv`: Hugging Face `dinhieufam/VietJobs`, and the author says it is from a published paper with about 48k rows (2026-10-17). The paper reference and the licence are not recorded; the author chose to skip them. The scan of all 18 raw columns found 0 emails, URLs and phones
- Who uses the classifier and what a wrong class costs: not stated by the author. Working answer, taken from the recommendation: the product may show the top 3, and the primary metric stays macro-F1 until the author says otherwise (2026-10-17)
- Which `dev` runs count against the baseline for a "go" decision: the 3-seed mean of A4 (0.6152 macro-F1), the best number reproducible by seed (2026-10-17)
- Dataset version this problem uses: `v7-1label26146` (`data/processed/splits-dedup-norm-1label/`, 26,146 texts, texts with two or more labels dropped). Scores on it are not comparable to `docs/04-results.md` or to the earlier versions (2026-10-17)
- `problem.md` approved by the author, version 2 (2026-10-17)
- Paper reference and licence of the dataset: the author will write them later (2026-10-17). Until then they stay marked ⛔
- `problem.md` set back to `draft`, version 3: the new `nm-problem` requires a Contract before approval; the earlier approval (version 2) predates it (2026-10-17)
- Length outliers in the three text columns (125 postings of 3,220–5,928 characters, 0.48 % of 26,146; corpus median 986): kept. They are real long postings, mostly `tài_chính…` (30), `marketing…` (16), `xây_dựng…` (16), and the CafeBERT path cuts every text at 512 tokens anyway. One text contains the literal headings `<Yêu cầu bắt buộc>` and `<Yêu cầu về b…>` (`train-9092`); it is a Vietnamese heading, not HTML, and is kept (2026-10-17)
- Label noise: the audit of 200 random rows (judged by Claude, one annotator) gives wrong 23/200 = 11.5 % [7.79, 16.66] and wrong or ambiguous 52/200 = 26.0 % [20.41, 32.49]. The author accepts this noise level for now (2026-10-17)
- Author's grading rule: findings the author has accepted or deferred are not counted against the project grade. `nm-data` keeps its own grade, which still counts them (2026-10-17)
- Contract confirmed by the author (2026-10-17): the interface is a Python call, there is no p95 latency target, and the caller shows `top3` when `confidence < 0.5`
- `problem.md` approved by the author, version 3 (2026-10-17)
