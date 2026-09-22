# T6.1 — literature survey: reading notes and candidate entries

Detail sheet for board row **T6.1** of [TASKS.md](../TASKS.md). Split out under TASKS.md §3 Rule 7. Written in English (AGENTS.md Rule 6); the two draft blocks meant for `docs/bao-cao/` are in Vietnamese.

**Status of everything here: candidate.** T6.1 is *Human only* (Rule 7 item 2: never cite what was not read). An agent may prepare a reading note; only the author, after reading the paper, copies the draft reference and the draft Table 2.1 row into `docs/bao-cao/`. Nothing on this sheet is cited anywhere yet.

---

## 1. arXiv 2603.05262 — "VietJobs: A Vietnamese Job Advertisement Dataset"

H. Pham Dinh, H. Nguyen Huy, M. El-Haj (VinUniversity). arXiv:2603.05262 [cs.CL], 5 Mar 2026. Code: https://github.com/VinNLP/VietJobs. PDF kept at the repo root: `../../2603.05262.pdf`. Read by Claude 2026-09-19 (full text + the public evaluation code); **not yet read by the author**.

**Why it matters.** This is the paper that introduces the dataset this thesis trains on (HF `dinhieufam/VietJobs`, 48,092 postings, 16 categories — same file, same counts as `data/README.md`). It also benchmarks the same two tasks: job-category classification and salary estimation. It is therefore the most direct point of comparison the thesis can have.

### 1.1 What the paper does

| | Paper |
|---|---|
| Data | TopCV, crawled July 2025 with Crawl4AI; fields extracted by GPT-4o / Gemini 2.5. 48,092 postings, 15.4 M words, 78,002 vocabulary, 0.32 % English tokens. 24 platform labels → 16 categories (ISCO-08 / O*NET / ESCO). 34 provinces, mostly Hanoi + HCMC. 34,365 (71.5 %) state a salary, 28.5 % "thoả thuận". Medians min / avg / max = 10 / 13 / 15 triệu; range 1–500; right-skewed |
| Models | 10 generative LLMs, chat/instruct variants: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410, Granite-3.3-8B-Instruct, Llama-SEA-LION-v3-8B-IT, Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, PhoGPT-4B-Chat, BloomVN-8B-Chat, Vistral-7B-Chat. Three settings: zero-shot, few-shot, LoRA fine-tune. **No classical-ML or encoder-based baseline** — the paper's own future-work section asks for one |
| Fine-tune | LoRA r=8, α=16, dropout 0.2 on q/k/v/o_proj; AdamW lr 5e-5; effective batch 64; 2 epochs; max_len 512; BF16; one A40. ~5 h per model for classification, 1–2 h for salary |
| Split | 80 / 10 / 10 random. No de-duplication, no repost grouping |
| Classification input | the `description` field |
| Salary input | five structured fields only: `job_title, contract_type, location, country, experience_required` — the description is **not** read |
| Salary label | column `salary_avg` of the raw CSV, kept as the string `"X triệu"`; rows with "thoả thuận" or any empty input field dropped; the first number in the string is the gold value (`format_prompt_salary.py`, `evaluation_salary.py`) |
| Scoring, classification | generated text `strip().lower()` must **exactly match** one of the 16 category names; anything else counts as wrong. sklearn `accuracy_score` + `f1_score(average="macro")` (`evaluation_category.py`) |
| Scoring, salary | RMSE and R² on the raw triệu scale (no log) |

### 1.2 Published results (their test split)

| Task | Setting | Best model | Result |
|---|---|---|---|
| Classification | zero-shot | Qwen2.5-7B-Instruct | Acc 0.31 · Macro-F1 0.32 |
| Classification | few-shot | Qwen2.5-7B-Instruct | **Acc 0.47 · Macro-F1 0.42** |
| Classification | fine-tuned | Qwen2.5-7B-Instruct | Acc 0.34 · Macro-F1 0.33 |
| Salary | zero-shot | Llama-SEA-LION-v3-8B-IT | RMSE 11.72 · R² 0.07 |
| Salary | few-shot | Llama-SEA-LION-v3-8B-IT | RMSE 10.65 · R² 0.16 |
| Salary | fine-tuned on VietJobs | Llama-SEA-LION-v3-8B-IT | **RMSE 10.60 · R² 0.17** |
| Salary | fine-tuned on VietJobs + Vietnam Jobs Dataset, scored on VietJobs | Llama-SEA-LION-v3-8B-IT | RMSE 10.24 · R² 0.23 |

Their own reading: few-shot beats LoRA fine-tuning for classification (fine-tuned accuracy plateaus around 0.3x); multilingual instruct models beat the Vietnamese-specific ones (PhoGPT, BloomVN, Vistral produce malformed outputs); salary R² stays low everywhere.

### 1.3 Same metrics, different conditions — what a comparison must say

`src/vietjobs/evaluate.py` already computes the four numbers the paper reports: `accuracy`, `f1_macro` (classification) and `rmse_trieu`, `r2_raw` (salary, back-transformed from log1p with `expm1`). Since 2026-09-19 the salary headline written to `04-results.md` also carries `RMSE` and `R2raw`, so every new row can be read against the paper without opening `metrics.json`.

Current thesis baselines, scored on **`dev`** (`artifacts/dl-cat-s2/metrics.json`, `artifacts/dl-sal-s2/metrics.json`, rows `dl-cat-s2` / `dl-sal-s2` in `docs/04-results.md`):

| Task | Thesis (`dev`) | Paper best (their `test`) |
|---|---|---|
| Classification | Macro-F1 **0.6025** · Acc **0.6511** (n = 3,812) | Macro-F1 0.42 · Acc 0.47 |
| Salary | RMSE **8.29** · R² raw **0.342** · MAE 4.15 (n = 2,698) | RMSE 10.60 · R² 0.17 |

The numbers are on the same scale and the same formulas, but **not on the same evaluation set or inputs**. Five differences to state next to any comparison:

1. **Different test rows.** Paper: random 80/10/10 on the raw file, reposts not grouped — a reposted ad can sit in both train and test (26.8 % of rows are reposts, `docs/05`). Thesis: exact dedup + group-disjoint 72/8/20 (`SPLIT_SEED` 20260826), which is the harder condition. Thesis numbers above are `dev`; `test` is scored once at the end (Rule 5).
2. **Different salary inputs.** Paper reads five short structured fields, never the description. Thesis reads `job_title + description + requirements` with salary figures masked (Rule 3). More text is available to the thesis model.
3. **Different salary label, by ≤ 0.5 triệu.** Paper gold = `salary_avg` = `(min+max)/2` rounded down to an integer (measured on the raw file: 53.9 % of the 34,365 disclosed rows equal the exact midpoint, 46.1 % differ by 0.5). Thesis target = exact `salary_mid`, trained on `log1p`. Same quantity, rounding differs.
4. **Different model class.** Paper: 7–8 B generative LLMs producing text. Thesis: frozen PhoBERT-base (135 M) + dense head, reading vectors. Paper's classification failure mode — malformed output that matches no label — does not exist for a softmax head.
5. **Different row filter for salary.** Paper also drops rows with any empty input field; thesis drops only undisclosed rows.

So the honest sentence is: *on the same metrics, a frozen-encoder head reaches Macro-F1 0.60 and RMSE 8.29 where the best 8 B LLM reports 0.42 and 10.60 — under a stricter split but with richer salary inputs.* Not a leaderboard claim.

### 1.4 How the paper's data analysis compares with `docs/05-phan-tich-du-lieu.md`

Agrees: 48,092 postings, 16 classes, 71.5 % disclosed, largest class Business/Sales (paper 8,276; thesis 8,213 after exact dedup), right-skewed salary with a tail to 500 triệu.

Thesis goes further: exact dedup (−385 rows) and repost groups (12,808 rows, 26.8 %); Gini 0.439 / Zipf −1.18 on class sizes; salary skew 11.90 → 0.10 after `log1p`; IQR fence at 28.0 with rows flagged, never dropped; disclosure rate per class (58.6–76.5 %) and its correlation with class size; eta² = 0.032 (category explains 3.2 % of log-salary variance); no-model floors. The paper does none of these; it removes outliers only for the figure. Text length is not comparable: paper 321 words over the whole posting, thesis 225 words over `title + description + requirements`.

### 1.5 Draft entries for `docs/bao-cao/` — copy only after the author has read the paper

Reference (IEEE, next free number is [13]; renumber by first appearance when inserted):

> [13] H. Pham Dinh, H. Nguyen Huy, and M. El-Haj, "VietJobs: A Vietnamese job advertisement dataset," *arXiv preprint arXiv:2603.05262*, 2026.

Row for Bảng 2.1 (`03-chuong-2-tong-quan.md` §2.3.3):

> | 1 | Pham Dinh và cộng sự (2026) [13] | Bộ VietJobs, 48.092 tin TopCV, 16 nhóm nghề; 10 LLM 7–8B (Qwen2.5, SEA-LION, Llama-3.1…) ở ba chế độ zero-shot, few-shot, LoRA; đầu vào lương chỉ 5 trường cấu trúc | Phân loại: Macro-F1 0,42 / Acc 0,47 (Qwen2.5-7B few-shot). Lương: RMSE 10,60 / R² 0,17 (SEA-LION fine-tune) | Chỉ có LLM, chưa có baseline encoder hay ML; chia ngẫu nhiên 80/10/10 không tách tin đăng lại; không dùng mô tả cho bài toán lương; R² lương thấp |

---

## 2. Log

| Date | Who | What happened |
|---|---|---|
| 2026-09-19 | Claude | Sheet opened at the author's request. Read arXiv 2603.05262 in full plus `evaluation_category.py`, `evaluation_salary.py`, `format_prompt_salary.py` in `VinNLP/VietJobs` to pin down the label and scoring; measured on `data/raw/VietJobs.csv` that `salary_avg` is the floored midpoint. Wrote §1 as a candidate; nothing copied into `docs/bao-cao/` |
