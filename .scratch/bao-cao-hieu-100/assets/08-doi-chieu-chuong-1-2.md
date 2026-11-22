# Audit: Doc Chương 1–2 (bao-cao.txt lines 292–429) against measured sources

Scope: Doc lines 292–429. Repo manuscript: `docs/bao-cao/02-chuong-1-gioi-thieu.md` (cited below as `ch1.md`) and `docs/bao-cao/03-chuong-2-tong-quan.md` (`ch2.md`). Final model: `dl-mtl-caf-rnn-b01-cw-s42-v2` (`R` below = `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2`).

Note on the repo manuscript: `ch1.md` and `ch2.md` were last changed in commit febae91 and are **stale against the Doc**. They still describe a PhoBERT-only, two-separate-networks thesis with three sub-tasks and old headline numbers. Most Doc sentences in these chapters have no counterpart in the repo md (column "repo md" = `—`).

Paper checks: the two PDFs in `/Users/admin/Documents/TrNghia/VietJob/` (`2603.05262.pdf` = VietJobs paper [19], `2112.11052.pdf` = Tran–Vo–Luu [20]) plus `tasks/T6-lit-survey.md`. No other cited paper ([13]–[18], [21]–[25]) is in the repo. `docs/bao-cao/08-tai-lieu-tham-khao.md` only lists [1]–[12].

| Doc line | Doc section | claim (short) | number in Doc | number in repo md | source value | source file + key | verdict |
|---|---|---|---|---|---|---|---|
| 328 | 1.8 Đóng góp | CafeBERT chosen as the encoder "cho cả hai bài toán" | (claim) | — | Same RNN-MTL `-v2` loss: PhoBERT dev MAE **3.6528** < CafeBERT **3.7001**. CafeBERT wins only on macro-F1 (0.6322 vs 0.6261) | `artifacts/dl-mtl-pho-rnn-b01-cw-s42-v2/metrics.json → metrics_sal.mae_trieu`; `R/metrics.json → metrics_sal.mae_trieu`; table in `docs/06-baseline-dl.md` l.527–532 | nói quá |
| 328 | 1.8 Đóng góp | "so sánh có hệ thống với bốn bộ mã hóa khác" | 4 | — | In the final configuration (RNN-MTL, `-v2` loss), only PhoBERT was compared. XLM-R `-v2` stopped at epoch 24 with no checkpoint, BamiBERT `-v2` not run: `⛔ CHƯA CÓ SỐ LIỆU (T17)`. The four-encoder sweep exists only under the older loss / Dense setups | `docs/06-baseline-dl.md` ⛔ line after the `-v2` table; `docs/04-results.md` rows 182–202 | nói quá |
| 328 | 1.8 Đóng góp | One MTL model reaches F1 0,6322 **and** MAE 3,70 | 0,6322 / 3,70 | — | These come from two checkpoints of the run: F1 from `best.pt` (epoch 21), MAE from `best_sal.pt` (epoch 30) | `R/metrics.json → best_epoch = 21`, `best_epoch_sal = 30`; `src/vietjobs/dl/train_mtl.py` l.335–359 | nói quá |
| 328 | 1.8 Đóng góp | Test is quoted for F1 only, not MAE | (0,6278 test) | — | Test MAE = 3.9234 (not in Doc). Quoting test F1 and dev MAE side by side is asymmetric | `R/metrics_test.json → metrics_sal.mae_trieu` | nói quá |
| 326 | 1.7 Phương pháp | "tập test chỉ dùng để báo cáo kết quả cuối" | (claim) | — | 71 runs have a `metrics_test.json`. AGENTS.md Rule 5 item 6 (2026-10-15 exception) labels those numbers *test-informed* | `ls artifacts/*/metrics_test.json` = 71 files | nói quá |
| 331 | 1.8 Đóng góp | Split "theo nhóm tin trùng lặp gần thay vì theo dòng" | (claim) | ch1.md §1.5.1 same claim | `group_key` = SHA-1 of the case-, accent- and punctuation-folded **exact** text of title+desc+req. Only exact-after-normalisation reposts are grouped. After the split, 443 / 3,812 dev rows (11.6 %) and 1,089 / 9,541 test rows still have a Jaccard ≥ 0.9 near-duplicate in train | `src/vietjobs/vitext.py → group_key` l.540–549; `manifest.json → groups_straddling_splits = 0`; `R/metrics_dev_clean.json → near_dup.n_cat = 443`; `R/metrics_test_clean.json → near_dup.n_cat = 1089` | nói quá |
| 349 | 2.2 Trong nước | Paper's best salary result RMSE 10,24 / R² 0,23 | 10,24 / 0,23 | — | 10.24 / 0.23 is "Fine-tuned on Both" (VietJobs + Kaggle Vietnam Jobs Dataset). Best on VietJobs alone: RMSE 10.60 / R² 0.17. The Doc sentence does not say extra data was used | `2603.05262.pdf` salary table, "Fine-tuned on Both", Llama-SEA-LION-v3-8B-IT; `tasks/T6-lit-survey.md` §1.2 | nói quá |
| 313 | 1.4 Thách thức | thiết kế ↔ marketing confused "theo cả hai chiều" | (claim) | — | Final model, dev: thiết kế→marketing 18, marketing→thiết kế 29. The largest thiết kế error is one-way: →xây_dựng 71 vs 4 back. Bảng 4.14 (old run `dl-cat-ce-cpu-0920`) lists only thiết kế→marketing (57). The repo states "cả hai chiều" only for du_lịch↔kinh_doanh (final model 122 / 76) | `R/predictions_dev.parquet` (`y_true_cat`, `y_pred_cat`); `docs/bao-cao/05-chuong-4-thuc-nghiem.md` Bảng 4.14 | nói quá |
| 415 | 2.3.3 Đối chiếu | XLM-R = "mô hình đa ngữ mà CafeBERT được xây dựng trên đó" | (claim) | — | The compared XLM-R is `FacebookAI/xlm-roberta-base` (768-d). CafeBERT is built on XLM-R **large** (1024-d). Different size, not the same base | `src/vietjobs/dl/encode.py → ENCODERS["xlmr"]`, `ENCODERS["cafebert"].hidden = 1024`; `docs/06-baseline-dl.md` l.118 "(XLM-R-large)" | nói quá |
| 415 | 2.3.3 Đối chiếu | BamiBERT supports up to 2048 tokens | 2048 | — | not found | searched `src/`, `docs/bao-cao/`, `docs/06-baseline-dl.md`, `artifacts/embeddings/*.json` | không có nguồn |
| 345 | 2.2 Thế giới | Safikhani et al.: 52.807 answers, KldB 2010, kappa BERT 66,01 % / GPT-3 42,60 % | 52.807; 2010; 66,01 %; 42,60 % | — (ch2.md §2.3.1 is `✍️ CẦN VIẾT TAY`) | no copy of [13] in repo | searched repo + `/Users/admin/Documents/TrNghia/VietJob/` (only 2 PDFs); `08-tai-lieu-tham-khao.md` stops at [12] | không có nguồn |
| 345 | 2.2 Thế giới | Beręsewicz et al.: hierarchy adds 1–2 pp | 1–2 | — | no copy of [14] | same | không có nguồn |
| 345 | 2.2 Thế giới | Jalilian et al.: 321 titles, F1 0,816 → 0,894 | 321; 0,816; 0,894 | — | no copy of [15] | same | không có nguồn |
| 346 | 2.2 Thế giới | Wang et al.: ~8 triệu Japanese postings | 8 triệu | — | no copy of [16] | same | không có nguồn |
| 346 | 2.2 Thế giới | Sun et al.: 215.308 IT postings | 215.308 | — | no copy of [17] | same | không có nguồn |
| 346 | 2.2 Thế giới | Bana: BERT explains 73 % variance, +12 pp | 73 %; 12 | — | no copy of [18] | same | không có nguồn |
| 346 | 2.2 Thế giới | [15] MiniLM + XGBoost MAE 1.587 euro | 1.587 | — | no copy of [15] | same | không có nguồn |
| 350 | 2.2 Trong nước | Huynh et al.: 10.000 IT descriptions, 25 titles, ensemble F1 72,71 % | 10.000; 25; 72,71 % | — | no copy of [21] | same | không có nguồn |
| 352–404 | 2.2 Bảng 2.1 | Table of related work | (empty: only "STT" + ~50 blank lines) | ch2.md: empty template + `⛔ CHƯA CÓ SỐ LIỆU (T6.1)` | — | Doc export l.353–404 | không có nguồn |
| 298 | 1.2 Vấn đề | Share of postings with no salary: "hơn một phần tư" | 28,5 % | "gần một phần ba" | 1 − 0.71463 = 0.2854 (Doc wording is right, repo wording is wrong) | `summary.json → disclosure_rate` | repo≠Doc |
| 302 | 1.3 Bài toán | Number of tasks | 2 | 3 sub-tasks (adds `disclosed`) | — | ch1.md §1.3 | repo≠Doc |
| 318 | 1.5 Mục tiêu | General objective | "mô hình học sâu" | "hệ thống web" | — | ch1.md §1.2.1 | repo≠Doc |
| 323 | 1.6 Phạm vi | Frozen encoders | "các bộ mã hóa" | "PhoBERT đóng băng" | `R/metrics.json → model = cafebert-frozen+rnn-mtl` | ch1.md §1.4.2 | repo≠Doc |
| 328 | 1.8 Đóng góp | Headline results | F1 0,6322; MAE 3,70; −35,0 % | ch1.md §1.5.2: MAE 4,13; −27,5 %; F1 0,6030; "lệch 27:1"; F1w 0,6413 | repo values are stale. "27:1" also contradicts `imbalance_ratio` = 25.51 | ch1.md §1.5.2 | repo≠Doc |
| 426 | 2.4.3 Đa nhiệm | Both tasks share one MTL trunk | (claim) | ch2.md §2.2.3: "Hai bài toán được giải bằng hai mạng riêng" | the final model is MTL (`R/metrics.json → task = mtl`) | ch2.md §2.2.3 | repo≠Doc |
| 298 | 1.2 Vấn đề | Rows after exact dedup | 47.707 | 47.707 | 47707 | `summary.json → source.rows_after_exact_dedup` | khớp |
| 298 | 1.2 Vấn đề | Disclosure rate | 71,5 % | 71,5 % | 0.714633 | `summary.json → disclosure_rate` | khớp |
| 298 | 1.2 Vấn đề | Undisclosed share | 28,5 % | 28,5 % | 0.285367 | `1 − summary.json → disclosure_rate` | khớp |
| 299 | 1.2 Vấn đề | Lowest per-class disclosure | 58,6 % | 58,6 % | 0.58551 (nhóm_nghề_khác) | `summary.json → per_category[14].disclosure_rate` | khớp |
| 299 | 1.2 Vấn đề | Highest per-class disclosure | 76,5 % | 76,5 % | 0.76489 (du_lịch) | `summary.json → per_category[4].disclosure_rate` | khớp |
| 299 | 1.2 Vấn đề | Pearson(disclosure, log class size) | 0,610 | 0,610 (p = 0,0122) | 0.60972 (p = 0.01215) | `summary.json → distribution.size_vs_disclosure_pearson_logn` / `_p` | khớp |
| 302 | 1.3 Bài toán | Number of classes | 16 | 16 | 16 | `summary.json → n_classes` | khớp |
| 306 | 1.3.2 Đầu ra | Target = log1p(salary_mid) | log1p | — | `salary_mid_log = log1p(mid)` | `src/vietjobs/dataset.py` l.72 | khớp |
| 312 | 1.4 Thách thức | Largest class size | 8.213 | — | 8213 | `summary.json → class_counts.kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` | khớp |
| 312 | 1.4 Thách thức | Largest / smallest ratio | 25,5 | — | 25.5062 | `summary.json → imbalance_ratio` | khớp |
| 312 | 1.4 Thách thức | Smallest class = nông nghiệp – năng lượng – môi trường | (322) | — | nông_nghiệp_năng_lượng_môi_trường | `summary.json → smallest_class` | khớp |
| 314 | 1.4 Thách thức | Salary labels on 71,5 % of postings | 71,5 % | 71,5 % | 0.714633 | `summary.json → disclosure_rate` | khớp |
| 314 | 1.4 Thách thức | Strong right skew | (claim) | — | skew 11.90 | `summary.json → salary.skew` | khớp |
| 315 | 1.4 Thách thức | Category explains 3,2 % of log-salary variance | 3,2 % | 3,2 % | 0.031744 | `summary.json → variance.eta_squared_log` | khớp |
| 319 | 1.5 Mục tiêu | Final model scored on test and on an external set | (claim) | — | `R/metrics_test.json` and `R/external_37k/` exist | `R/` | khớp |
| 322 | 1.6 Phạm vi | Raw rows | 48.092 | 48.092 | 48092 | `manifest.json → source_rows` | khớp |
| 322 | 1.6 Phạm vi | Rows after exact dedup | 47.707 | 47.707 | 47707 | `manifest.json → rows_after_exact_dedup` | khớp |
| 322 | 1.6 Phạm vi | Classes | 16 | 16 | 16 | `manifest.json → splits.*.categories` | khớp |
| 322 | 1.6 Phạm vi | Split 8:2 then 9:1 | 8:2; 9:1 | — | test 0.2, dev 0.1 | `manifest.json → split_test_fraction`, `split_dev_fraction`; `config.py` l.38–39 | khớp |
| 322 | 1.6 Phạm vi | Train / dev / test rows | 34.354 / 3.812 / 9.541 | — | 34354 / 3812 / 9541 | `manifest.json → splits.{train,dev,test}.rows` | khớp |
| 322 | 1.6 Phạm vi | Salary unit triệu/tháng | (unit) | triệu VND/tháng | triệu VND/month | `manifest.json → salary_unit` | khớp |
| 325 | 1.7 Phương pháp | Encoders compared: PhoBERT, CafeBERT, ViSoBERT, XLM-R, BamiBERT; RNN = BiGRU‖BiLSTM→CNN | 5 encoders | — | all five appear in `ENCODERS` and in Dense / RNN rows of the log | `src/vietjobs/dl/encode.py → ENCODERS`; `docs/04-results.md` rows 182–202 | khớp |
| 328 | 1.8 Đóng góp | Dev macro-F1 of the final model | 0,6322 | — | 0.632231 | `R/metrics.json → metrics_cat.f1_macro` | khớp |
| 328 | 1.8 Đóng góp | Test macro-F1 | 0,6278 | — | 0.627821 | `R/metrics_test.json → metrics_cat.f1_macro` | khớp |
| 328 | 1.8 Đóng góp | Dev MAE | 3,70 triệu | — | 3.70006 | `R/metrics.json → metrics_sal.mae_trieu` | khớp |
| 328 | 1.8 Đóng góp | Median-guess floor MAE | 5,70 triệu | ch2.md 5,70 | 5.69552 | `summary.json → floors.salary_global_median.mae` | khớp |
| 328 | 1.8 Đóng góp | Reduction vs floor | 35,0 % | — | (5.69552 − 3.70006) / 5.69552 = 0.3504 (same 2,698 dev rows: `floors.n_dev_disclosed` = `metrics_sal.n` = 2698) | derived from the two keys above | khớp |
| 330 | 1.8 Đóng góp | Category explains 3,2 % of log-salary variance | 3,2 % | 3,2 % | 0.031744 | `summary.json → variance.eta_squared_log` | khớp |
| 333 | 1.9 Cấu trúc | Six chapters | 6 | 6 | — | ch1.md §1.6 (chapter titles differ; expected) | khớp |
| 339 | 2.1.4 Phân loại | Largest:smallest class ratio | 25,5:1 | 25,5 : 1 | 25.5062 | `summary.json → imbalance_ratio` | khớp |
| 341 | 2.1.5 Lương | Disclosure rate | 71,5 % | 71,5 % | 0.714633 | `summary.json → disclosure_rate` | khớp |
| 341 | 2.1.5 Lương | Raw salary skewness | 11,90 | 11,90 | 11.9001 | `summary.json → salary.skew` | khớp |
| 341 | 2.1.5 Lương | Share of disclosed postings giving a range | 92,9 % | 92,9 % | 0.929252 | `summary.json → salary_is_range_share` | khớp |
| 349 | 2.2 Trong nước | VietJobs paper: 48.092 postings, TopCV, 16 groups, 34 provinces | 48.092; 16; 34 | — | "48,092 postings … all 34 provinces"; 16 domains | `2603.05262.pdf` abstract; `tasks/T6-lit-survey.md` §1.1 | khớp |
| 349 | 2.2 Trong nước | Best classification macro-F1 | 0,42 | — | 0.42 (Qwen2.5-7B few-shot) | `2603.05262.pdf` classification table; `tasks/T6-lit-survey.md` §1.2 | khớp |
| 349 | 2.2 Trong nước | Paper split 80/10/10 | 80/10/10 | — | "80%, 10%, 10%" | `2603.05262.pdf` "Data Split" | khớp |
| 349 | 2.2 Trong nước | Salary input = 5 structured fields, no description | 5 fields | — | `job_title, contract_type, location, country, experience_required` | `tasks/T6-lit-survey.md` §1.1 | khớp |
| 350 | 2.2 Trong nước | Tran–Vo–Luu: 68 job titles, multi-label | 68 | — | "68 job titles" | `2112.11052.pdf` §III | khớp |
| 350 | 2.2 Trong nước | Best = mBERT, F1 62,20 % dev / 47,44 % test | 62,20 %; 47,44 % | — | 62.20 / 47.44 (BERT row, Table V) | `2112.11052.pdf` abstract + Table V | khớp |
| 410 | 2.3.1 PhoBERT | Segmenter = underthesea | (tool) | underthesea | underthesea | `manifest.json → segmenter`; `vitext.py` l.169 | khớp |
| 410 | 2.3.1 PhoBERT | Max 256 tokens; phobert-base-v2; 768-d | 256; 768 | 768 (ch1.md §1.3.2) | `vinai/phobert-base-v2`, hidden 768, default `max_len` 256 | `encode.py → ENCODERS["phobert"]`; `summary.json → tokens.tokenizer` | khớp |
| 412 | 2.3.2 CafeBERT | uitnlp/CafeBERT, built on XLM-R large | (large) | — | `uitnlp/CafeBERT`, hidden 1024 | `encode.py → ENCODERS["cafebert"]` | khớp |
| 413 | 2.3.2 CafeBERT | Reads unsegmented text | (claim) | — | `segmented: False` | `encode.py → ENCODERS["cafebert"].segmented` | khớp |
| 413 | 2.3.2 CafeBERT | 512 tokens; 5,7 % of dev truncated | 512; 5,7 % | — | `max_len` 512; truncated_frac 0.057188 | `R/config.json → max_len`; `artifacts/embeddings/cafebert-dev-masked-len512.json → truncated_frac` | khớp |
| 415 | 2.3.3 Đối chiếu | ViSoBERT and XLM-R at 512, BamiBERT at 1024, all unsegmented | 512; 512; 1024 | — | runs `dl-*-visobert-*-512`, `dl-*-xlmr-*-512`, `dl-*-bamibert-*-1024`; `segmented: False` | `docs/04-results.md` rows 182–202; `encode.py → ENCODERS` | khớp |
| 419 | 2.4.1 Dense | Mean-pool → LayerNorm → 256 → 128, GELU, dropout 0,3 → 16 or 1 | 256; 128; 0,3; 16; 1 | — | `DenseHead`: LN → Linear(·,hidden) → GELU → Dropout → Linear(hidden, hidden//2) → … → out; hidden 256, dropout 0.3; mean pooling in `encode_texts` | `src/vietjobs/dl/heads.py → DenseHead`; `encode.py` l.169–187; `R/config.json → hidden, dropout` | khớp |
| 423 | 2.4.2 RNN | SpatialDropout → BiGRU ‖ BiLSTM → Conv1d per branch → masked avg+max → concat → LayerNorm → FC | (structure) | — | matches `BiGruLstmCnn.features` / `out` | `heads.py → BiGruLstmCnn` | khớp |
| 426 | 2.4.3 Đa nhiệm | Two heads, 16 classes + salary | 16 | — | `MultiTaskRnn.head_cat`, `head_sal` | `heads.py → MultiTaskRnn` | khớp |
| 426 | 2.4.3 Đa nhiệm | L = CE + α·SmoothL1, α = 5 | 5 | — | `loss = l_cat + alpha_sal * l_sal`; alpha_sal 5.0 | `train_mtl.py` l.283; `R/config.json → alpha_sal` | khớp |
| 426 | 2.4.3 Đa nhiệm | CE with effective class weights, label smoothing 0,05 | 0,05 | — | cw_mode effective, label_smoothing 0.05 | `R/config.json → cw_mode, label_smoothing` | khớp |
| 426 | 2.4.3 Đa nhiệm | SmoothL1 β = 0,1; per-row weight by salary bin; salary loss only on disclosed rows | 0,1 | — | beta_sal 0.1; `compute_salary_bin_weights` (7 bins, β_eff 0.999); `l_sal` masked by `keep_sal` | `R/config.json → beta_sal`; `train_dl.py` l.106–124; `train_mtl.py` l.210–213, 280–281 | khớp |
| 428 | 2.5 Hàm mất mát | Effective-number class weights β = 0,999 | 0,999 | — | 0.999 | `R/config.json → cw_beta` | khớp |
| 428 | 2.5 Hàm mất mát | Label smoothing | 0,05 | — | 0.05 | `R/config.json → label_smoothing` | khớp |
| 428 | 2.5 Hàm mất mát | Focal Loss tried with γ = 2 | 2 | — | runs `dl-cat-focal-s2` etc. "focal(g=2)"; `--focal-gamma` default 2.0 | `docs/04-results.md` rows 30, 33, 38, 44, 65; `train_dl.py` l.270 | khớp |
| 429 | 2.5 Hàm mất mát | Skew 11,90 → 0,10 after log1p | 11,90; 0,10 | 11,90; 0,10 | 11.9001 → 0.10253 | `summary.json → salary.skew`, `salary.skew_log1p` | khớp |

## Non-numeric issues found during the audit (not counted in the table)

- **Doc l.349, author names.** The Doc writes "Dinh, Huy và El-Haj". The PDF authors are Hieu **Pham Dinh**, Hung **Nguyen Huy**, Mo El-Haj, i.e. family names Phạm and Nguyễn. Compare l.350, where "Tran, Vo và Luu" uses family names correctly.
- **Doc ch2 heading numbers (l.337–426).** Sub-headings are numbered 4, 5, 6, …, 13 straight through instead of 2.1.1, 2.1.2, 2.2.1, …. This looks like a Google Doc list-numbering problem.
- **Doc l.422–423.** The sentence "Đầu RNN của đề tài theo kiến trúc của Tran, Vo và Luu [20]" is repeated: it ends l.422 and the next paragraph restates it.
- **Doc l.304 vs code.** The Doc says salary figures are masked "Với bài toán lương". In the MTL model the shared trunk reads the `masked` cache for **both** heads (`train_mtl.py` l.70–75, `load_mtl` uses task `"salary"`). So the classifier also sees masked text. This is not wrong, but it is not stated.
- **Repo md ch1.md §1.4.2.** It still carries `✍️ CẦN VIẾT TAY` for data source and collection date. The Doc gives the source (TopCV, via [19]) only in Chương 2, and does not give the collection date (the paper says July 2025, per `tasks/T6-lit-survey.md` §1.1).
