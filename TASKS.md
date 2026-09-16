# TASKS.md — the work board

The one file that says **what is being worked on right now, what comes next, and
what is already done**. [docs/09-lo-trinh.md](docs/09-lo-trinh.md) is the
*strategy* (priorities and why); this file is the *operations* (concrete tasks,
who holds them, what "done" looks like). Written in English (AGENTS.md Rule 6).

**How to use it**

- **Human (any group member):** pick the first `todo` task whose dependencies are `done`,
  set it to `doing`, put your name in *Owner*. When finished, set `done`, fill
  *Evidence*, and add one line to §5. Ask the agent "giao task tiếp" to be
  assigned the next one.
- **AI agent (Claude / Cursor / Codex…):** read this file **first** in every
  session, before touching code. §1 is the context you would otherwise have to
  rediscover. Never mark a task `done` without the evidence column filled in;
  never skip a task whose dependency is still open. After any real change,
  update §2 and §5 in the same session.

Status vocabulary: `todo` · `doing` · `blocked (why)` · `done` · `dropped (why)`.

---

## 1. Context snapshot — 2026-09-10

What an agent needs to know before doing anything:

| Fact | State |
|---|---|
| Main track | Deep learning: frozen PhoBERT → dense → one head per task; multi-task later. The ML track is closed and archived (bar: test macro-F1 0.6112) |
| Split scheme | **v2** since 2026-09-09: `train:test` 8:2 then `train:dev` 9:1 → 34,354 / 3,812 / 9,541. `val` is now called `dev` |
| Every DL result so far | **Stale.** All 10 rows in `docs/04-results.md` were measured on the v1 dev (7,159 rows). They stay in the log (append-only) but are labelled `(lược đồ v1)` in the report and must be re-run |
| `data/` and `artifacts/` on this machine | `data/raw/VietJobs.csv` downloaded and built 2026-09-10: `data/processed/manifest.json` (47,707 rows → 34,354 / 3,812 / 9,541), `pytest -q` green (108 passed). T0.1 + T0.2 done on this machine |
| Working tree | 19 uncommitted files (AGENTS.md Rule 8, docs/05 rewrite, bao-cao ch.3, two figures). Last commits are named `add` — history needs a real message |
| Team | **Group project** — several members, each on their own machine. Who takes which task is recorded in the *Owner* column |
| Hardware | **Depends on each member's machine — to be agreed by the group later.** Reference point so far: this Intel macOS has no GPU/MPS, frozen encoding ≈ 30 postings/s (~20 min for `train`); fine-tuning (T3) will run on whichever member has a GPU, or on Colab/Kaggle. Splits + `manifest.json` must be identical on every machine |
| Thesis | `docs/bao-cao/` (Vietnamese). Ch.1, ch.3 drafted; ch.4 holds stale numbers; ch.2 needs a real literature survey; ch.5 not started. `⛔` and `✍️` markers are the to-do list |

---

## 2. Board

**Currently active:** `T0.3` (commit the dirty docs tree) and `T6.1` (literature survey). `T0.1`–`T1.5` are all done on this machine. `T1.6` and `T1.7` can start; `T2.1`–`T2.3` can start (T1.3/T1.4 done).

### T0 — get the machine back to a runnable state

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T0.1 | Pull `data/raw/VietJobs.csv` once it is pushed (see `data/README.md`) and rebuild the splits with `python -m vietjobs.dataset build` | done | Claude | — | `data/processed/manifest.json`: 47,707 rows, 34,354 / 3,812 / 9,541, seed 20260826, sha256 verified; `pytest -q` green (108 passed). See §5 for a reproducibility bug found and fixed along the way |
| T0.2 | Set up both venvs (`.venv` py≥3.10, `.venv-dl` py3.9) and confirm `pytest -q` passes in `.venv` | done | Claude | T0.1 | `.venv` on Python 3.14.5: `pytest -q` → 108 passed (needed `brew install libomp` for LightGBM). `.venv-dl` (py3.9): confirmed working 2026-09-12 while running T1.1 — torch 2.2.2, transformers 4.46.3 installed |
| T0.3 | Commit the current dirty docs tree with a real message (Rule 8, docs/05 on the original file, ch.3) | todo | — | — | commit hash in §5 |

### T1 — re-measure everything on split scheme v2 (Priority 1, redo)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T1.1 | Re-run `python scripts/analyze_data.py` on the original file **with `--tokens`** so `artifacts/eda/summary.json` matches the figures and figure 07 (token length, truncation cost at 256) exists | done | Claude | T0.1 | `summary.json`: `n_rows = 47707`, floors re-measured on v2 `dev` (3,812 rows). All 7 figures regenerated incl. `eda-07-do-dai-token.png`. Every `⛔` in docs/05 and bao-cao ch.3 §3.6 replaced by a measured number; added new §3.6.8 (token length) to bao-cao ch.3; status rows updated in docs/00-tong-quan.md and bao-cao/00-index.md. `pytest -q` green (108 passed) |
| T1.2 | Re-encode PhoBERT embeddings for `train dev` on v2 splits, both `raw` (category) and `masked` (salary) families | done | Claude | T0.2 | `artifacts/embeddings/{train,dev}-{raw,masked}-len256.npy`: shapes (34354, 768) / (3812, 768) each, matching manifest row counts; `pytest -q` 108 passed |
| T1.3 | Re-run the classification baseline (`--task category`, with and without `--class-weight`) | done | Claude | T1.2 | `dl-cat-s2` macroF1=0.6025 · top3=0.9318 (best 16/24); `dl-cat-s2-cw` macroF1=0.5710 · balAcc=0.6931 (best 15/23) — both `eval=dev`, n=3812. Rows in `04-results.md`; tables in docs/06 §5.1 + bao-cao ch.4 Bảng 4.5/4.5b/4.12 updated. `pytest -q` 108 passed |
| T1.4 | Re-run the salary baseline (`--task salary`) | done | Claude | T1.2 | `dl-sal-s2` MAE=4.15tr · MedAE=2.50tr · R2log=0.512 · ±20%=51.6% (best 26/34, `eval=dev`, n=2698). Row in `04-results.md`; docs/06 §5.2, docs/07, bao-cao ch.4 Bảng 4.6/4.6b/4.12 updated. `pytest -q` 108 passed |
| T1.5 | Re-run the two linear probes (`scripts/probe_embeddings.py`) as the diagnostic bar | done | Claude | T1.2 | `probe-cat-s2` macroF1=0.5898 · top3=0.9258; `probe-sal-s2` MAE=4.42tr · R2log=0.466 — both `eval=dev`. Two rows in `04-results.md`; docs/06 §5.1/§5.2 + bao-cao ch.4 Bảng 4.5/4.6 updated. `pytest -q` 108 passed |
| T1.6 | Re-measure the no-model floors on v2 `train`→`dev` (median-salary MAE, "always yes" accuracy) so the *bar to beat* stops carrying `(lược đồ v1)` | todo | — | T0.1 | docs/05 §7 and docs/09 bar table updated with a source |
| T1.7 | Flip every "stale" cell in `docs/00-tong-quan.md` status table and the warning in `bao-cao/00-index.md` §3 | todo | — | T1.1–T1.6 | no cell says "stale"; ch.4 status ≠ "số liệu đã cũ" |

### T2 — the three findings the first baseline produced

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T2.1 | Salary right tail: the head never predicts above ~80M while data reaches 275M (MAE 25.58 above 30M vs 3.70 below). Try an asymmetric / quantile loss on `dev` | todo | — | T1.4 | a `04-results.md` row + the tail-error table in docs/07 |
| T2.2 | Decide what to do with `nhóm_nghề_khác` (F1 = 0.000, 48 samples): merge, drop from train/dev, or keep — a labelling decision, measured with `f1_macro_no_junk` | todo | — | T1.3 | decision written in docs/01 + docs/06, test **untouched** (Rule 5) |
| T2.3 | Label-noise ceiling: hand-label 100 `dev` postings from the top confusions (`kinh_doanh` ↔ `du_lịch_nhà_hàng`), measure agreement | todo | — | T1.3 | agreement number in docs/05 with the sample file path |

### T3 — fine-tune PhoBERT (Priority 2)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T3.1 | Prepare the Colab/Kaggle package: `splits/` + `manifest.json` + a notebook that runs `vietjobs.dl` end to end, 256 tokens | todo | — | T1.3, T1.4 | notebook committed under `scripts/`, sha256 of splits matches manifest |
| T3.2 | Fine-tune for `category`, score on `dev` | todo | — | T3.1 | `04-results.md` row; docs/06 architecture diagram + "five decisions" table |
| T3.3 | Fine-tune for `salary`, score on `dev` | todo | — | T3.1 | same |

### T4 — multi-task (Priority 3)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T4.1 | Shared trunk → two heads, `loss_B` masked for the 28.5 % without salary, trunk reads the `masked` family only | todo | — | T3.2, T3.3 (or T1.3, T1.4 if fine-tuning is unreachable) | `04-results.md` row for both heads; cost of masking on the classification head measured and written in docs/06 |

### T5 — the system (Priority 5)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T5.1 | Wire the DL path into `predict.py` so Rule 4 (one code path) holds for the DL branch, with a test | todo | — | T1.3 | `tests/` has a DL round-trip test; docs/08 updated |
| T5.2 | `artifacts/PRODUCTION.json` mapping task → run_id, replacing `_latest_run` by mtime | todo | — | T5.1 | file + docs/08 |
| T5.3 | Measure inference latency (PhoBERT ~30/s CPU vs TF-IDF instant) and report it next to the score | todo | — | T5.1 | number with its source in docs/06 + bao-cao ch.5 |
| T5.4 | Write bao-cao ch.5 (`06-chuong-5-he-thong.md`) | todo | — | T5.1–T5.3 | status in bao-cao index = "xong bản thảo" |

### T6 — thesis manuscript (runs in parallel with everything)

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T6.1 | Literature survey for ch.2 — read the papers, then fill Table 2.1 and `08-tai-lieu-tham-khao.md`. **Human only** (Rule 7 item 2: never cite what was not read) | todo | Nghĩa | — | no `✍️` left in ch.2 |
| T6.2 | Sweep every `⛔` in ch.4 once T1 is done | todo | — | T1.7 | `grep -c "⛔" docs/bao-cao/*.md` = 0 for ch.4 |
| T6.3 | Final checklist — Phụ lục F in `09-phu-luc.md`; assemble Word per `bao-cao/00-index.md` §5 | todo | — | everything | markers = 0 across `docs/bao-cao/` |

### Backlog (not scheduled)

- Salary edge cases: 15 postings > 200M, plus the sub-2M group (count pending T1.1)
- Selection bias: disclosure rate spans 58.6 % → 76.5 % by sector — measure it
- `UNMASKED_COLUMNS` is hand-written; `soft_skills_text`, `qualifications_text` slipped once. Adding a text field means updating `_MASKABLE` **and** `UNMASKED_COLUMNS`

---

## 3. Rules that bind every task

1. `SPLIT_SEED = 20260826` and the split fractions never change.
2. `test` is read once, at the very end, behind `--confirm-test`. Never transformed.
3. `04-results.md` is append-only.
4. Salary tasks read `*_masked` columns only; `raw` and `masked` embedding files are never mixed.
5. A task is not `done` until `pytest -q` is green and the docs in the Rule 1 / Rule 7 tables are updated.

---

## 4. Session opener (copy to the agent)

> Read `TASKS.md`. Tell me which task is active, what its dependencies say, and
> what the next concrete command is. Then do it, and update §2 and §5.

---

## 5. Log (append-only, newest last)

| Date | Who | What happened |
|---|---|---|
| 2026-09-10 | Claude | Created this board. Found `data/` and `artifacts/` absent on this machine; T0.1 is the gate for everything in T1. 19 files uncommitted since Rule 8 + docs/05 rewrite |
| 2026-09-10 | Nghĩa | Group project: GPU depends on each member's machine, to be agreed later. Data will be pushed later → T0.1 marked blocked, T0.3 + T6.1 active |
| 2026-09-10 | Claude | Ran T0.1: downloaded `data/raw/VietJobs.csv` (sha256 verified), built v2 splits. First build's group fingerprints did not match the (until-then-unverified) values printed in `data/README.md`, though row/group counts already matched exactly — found the cause: `group_stratified_split` (`src/vietjobs/dataset.py`) sorted same-size groups with the default unstable `quicksort`, whose tie order is not guaranteed identical across numpy versions, so two machines could silently draw different group assignments from the same seed. Fixed with `sort_values(..., kind="stable")` and rebuilt; `data/README.md`'s fingerprint table now holds the corrected values. No experiment had touched the old fingerprints, so nothing else needed re-measuring. Also hit and fixed two environment issues: macOS was auto-hiding the editable-install `.pth` file (Python 3.13+ site.py skips hidden `.pth` files) — worked around with `PYTHONPATH=src`; and LightGBM needed `brew install libomp`. `pytest -q` now 108 passed. `data/external/vietjobs37k/` — Zenodo's own endpoint stayed down (504 Gateway Time-out across several retries); Nghĩa downloaded it manually (`~/Downloads/vietjobs37k.zip`, already the pruned 16-file package) and it was extracted into place, 50 MB as `data/README.md` states. `shasum -a 256 -c SHA256SUMS` confirms all 16 kept files OK; the 47 dropped files report missing, as expected |
| 2026-09-11 | Claude | Added `hoc-tap/` — five personal study notes in Vietnamese explaining both deep-learning pipelines from zero, for a reader with no DL background: the two task shapes (16-class `category` vs `log1p` salary regression), the shared `dl/text.py` path and the `raw`/`masked` leak rule, then architecture A (frozen PhoBERT + dense) mechanism by mechanism — tokenizer, attention mask, masked mean pooling, the `.npy` cache, per-dimension standardisation, LayerNorm/Linear/GELU/Dropout, softmax + cross-entropy, class weighting, log1p/Huber/expm1, AdamW, clipping, the linear probe — and architecture B, a basic TextCNN (vocabulary fitted on `train`, `nn.Embedding` learned from scratch, Conv1d 3/4/5, max-over-time pooling) marked throughout as **proposed, never trained**: no code under `src/vietjobs/dl/`, no row in `04-results.md`, written in the conditional with no comparative claim. The notes carry construction numbers only (768, 16, 256, split sizes); every measured result is a link into `docs/06-baseline-dl.md` or `docs/04-results.md`, so nothing there needs re-editing after T1.3/T1.4. Deliberately outside the documentation tree at the author's request: `docs/` untouched (no TOC, no `nen-tang/00-index.md`, no `SOURCES` entry in `render_figures.sh`, no `bao-cao/` chapter), so Rule 1 and Rule 7 do not fire. Two lines added to `AGENTS.md` to keep the rules honest: `hoc-tap/` declared in the §2 layout table, and named in Rule 6 item 1 as the second deliberate Vietnamese exception alongside `docs/bao-cao/`. No `src/`, no `data/`, no `artifacts/` changes; `pytest -q` still 108 passed |
| 2026-09-12 | Claude | Ran T1.1: `PYTHONPATH=src .venv-dl/bin/python scripts/analyze_data.py --tokens` (found `.venv-dl` already has torch 2.2.2 + transformers 4.46.3 — corrects the T0.2 note above, no member had reported setting it up) then `.venv/bin/python scripts/analyze_data.py`. Key finding: `max_len=256` truncates 19.8 % of postings but keeps 91.5 % of all tokens (median 178, p95 370) — the first real measurement behind the `--max-len` default, previously an assumption. `summary.json` fully regenerated (`n_rows=47707`); no-model floors re-fitted on the v2 splits (`train` 34,354 / `dev` 3,812): salary median floor MAE 5.70M (was 5.86M under v1), category-oracle floor MAE 5.57M (was 5.75M), majority-class macro-F1 0.0214, disclosed-majority accuracy 0.7078. Closed every `⛔` in `docs/05-phan-tich-du-lieu.md` and `docs/bao-cao/04-chuong-3-phuong-phap.md` §3.6 with these numbers; added new §3.6.8 + Hình 3.10 to the bao-cao chapter for figure 07 (did not exist before). Updated the "Data analysis" status row in `docs/00-tong-quan.md` and the corresponding row + stale-figure blockquote in `docs/bao-cao/00-index.md`. Did **not** touch the DL-baseline stale markers (T1.2–T1.7 still own those). `pytest -q` 108 passed throughout |
| 2026-09-13 | Claude | Follow-up fix (user reported a broken figure while reviewing T1.1): `docs/05-phan-tich-du-lieu.md` and `docs/bao-cao/04-chuong-3-phuong-phap.md` still linked the pre-Rule-9 filenames `eda-02-lech-lop.png`, `eda-03-tan-xa-co-lop.png`, `eda-05-phan-bo-luong.png`, none of which exist any more — Rule 9 split each into `-02a-`/`-02b-`, `-03a-`/`-03b-`, `-05a-`/`-05b-`/`-05c-` but the links were never updated. Verified each replacement image's actual content (viewed the PNGs) before repointing: `02a` = class-size bars, `02b` = Lorenz/Gini curve, `03a` = class-size scatter, `03b` = disclosure-vs-size scatter, `05a` = raw-scale histogram, `05b` = post-log1p histogram, `05c` = IQR-tail histogram. Fixed 9 links across the two files (image embeds + text links + one "Nguồn số liệu" block), re-pointing each to the sub-figure that actually matches the surrounding text. Confirmed every `eda-*.png` reference in both files plus `docs/bao-cao/00-index.md` now resolves to a real file. Not part of T1.1's original scope but touches the same files T1.1 edited; `pytest -q` still 108 passed |
| 2026-09-13 | Claude | Ran T1.2: `PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode --task category --splits train dev` then `--task salary --splits train dev` (`.venv-dl`, CPU, ~63-64 postings/s, no GPU/MPS on this machine). Produced `artifacts/embeddings/{train,dev}-{raw,masked}-len256.npy`: `train-raw` (34354, 768), `dev-raw` (3812, 768), `train-masked` (34354, 768), `dev-masked` (3812, 768) — row counts match `manifest.json`. `--task salary` covers `disclosed` too since the cache is keyed by column family, not task name (`encode.family`). `artifacts/embeddings/` did not exist before this run. `pytest -q` (main `.venv`) 108 passed |
| 2026-09-15 | Claude | Ran T1.3: `PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --run-id dl-cat-s2` then `--class-weight --run-id dl-cat-s2-cw` (`.venv-dl`, MPS this time — earlier T1.2 log said "no GPU/MPS on this machine", `env.json` now reports `device=mps`, so that note was wrong or the two runs picked different backends; not investigated further since the numbers still land 15-16 s/run either way). Loaded the cached `artifacts/embeddings/{train,dev}-raw-len256.npy` from T1.2, no re-encoding needed. `dl-cat-s2`: macroF1=0.6025 · acc=0.6511 · balAcc=0.6199 · top3=0.9318 · `f1_macro_no_junk`=0.6454, best epoch 16/24, n=3812. `dl-cat-s2-cw`: macroF1=0.5710 · acc=0.5976 · balAcc=0.6931 · top3=0.9208 · `f1_macro_no_junk`=0.5958, best epoch 15/23. Both `eval=dev`; `train_dl.py` auto-appended both rows to `04-results.md`. Updated `docs/06-baseline-dl.md` §5/§5.1 (added the scheme-v2 table + floor row, relabelled the old table "v1 — history", rewrote the class-weight/top-3 prose from the new numbers, left the TF-IDF bar as an unre-measured v1 reference), `docs/bao-cao/05-chuong-4-thuc-nghiem.md` (new Bảng 4.5 on v2, old table renamed Bảng 4.5b, §4.3.1 rewritten, one cell of Bảng 4.12 updated with a `(lược đồ v2)` note, top-of-chapter blockquote amended to say §4.3 is now v2 while §4.4+ stay v1), and `docs/00-tong-quan.md` rows 78-79 (PhoBERT embeddings and classification baseline both flipped from **stale** to **done** — the embeddings row had been missed by T1.2's own log entry). Did **not** touch `docs/bao-cao/00-index.md` §3 or the salary/error-analysis tables (T1.4-T1.7, T2.x own those). `pytest -q` 108 passed |
| 2026-09-16 | Claude | Ran T1.4: `PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary --run-id dl-sal-s2` (`.venv-dl`, MPS, defaults unchanged — lr 3e-4, standardised, h=256, patience 8). Loaded the cached `artifacts/embeddings/{train,dev}-masked-len256.npy` from T1.2, no re-encoding needed. Result: MAE=4.15tr · MedAE=2.50tr · R2log=0.512 · ±20%=51.6%, best epoch 26/34, `eval=dev`, n=2698 (dev rows with a disclosed salary) — clearly better than the v1 result (MAE 4.83/4.88), a bigger margin over the (also re-measured) floor: −27.2 % vs v1's −17.6 %. `train_dl.py` auto-appended the row to `04-results.md`. Updated `docs/06-baseline-dl.md` §5.2 (new scheme-v2 table with the T1.1-measured v2 floors 5.70/5.57, old table relabelled "v1 — history", prose rewritten from the new numbers, noted `probe-sal` not yet re-run on v2 — T1.5), `docs/07-bai-toan-luong.md` (top blockquote's floor numbers corrected from the stale v1 5.86/5.75 to the v2 5.70/5.57 already measured by T1.1 but never propagated here — an omission from that task, not new measurement), `docs/bao-cao/05-chuong-4-thuc-nghiem.md` (new Bảng 4.6 on v2, old table renamed Bảng 4.6b, §4.4.1 rewritten, two Bảng 4.12 cells updated with `(lược đồ v2)`, top-of-chapter blockquote amended to say §4.4 is now v2 too). Did **not** touch the error-analysis tables (§4.6-§4.8, T2.x) or `docs/bao-cao/00-index.md` §3 (T1.7). `pytest -q` 108 passed |
| 2026-09-16 | Claude | Ran T1.5: found two small bugs in `scripts/probe_embeddings.py` before running it — the logged `Scope` column was hard-coded `val` even though Rule 5 renamed the split to `dev` on 2026-09-09 (the script itself always reads `D.load_split("dev")`, so the label was simply wrong post-rename), and the default `run_id` (`probe-cat`/`probe-sal`) is identical to the existing v1 rows, which would make new and old rows indistinguishable by id alone. Fixed both: added an optional `--run-id` argument and replaced the hard-coded `val` literal with `dev`. `pytest -q` 108 passed before running anything (change touches only this script, not `src/`). Then ran `PYTHONPATH=src .venv-dl/bin/python scripts/probe_embeddings.py --task category --run-id probe-cat-s2` and `--task salary --run-id probe-sal-s2` (`.venv-dl`, scikit-learn, ~35s and <1s). `probe-cat-s2`: macroF1=0.5898 · acc=0.6388 · balAcc=0.5969 · top3=0.9258, n=3812 — only 0.0127 below `dl-cat-s2`'s macro-F1, similar to the v1 gap (0.0120), so the dense head's extra capacity buys a small, consistent margin over the linear probe on this scheme too. `probe-sal-s2`: MAE=4.42tr · MedAE=2.77tr · R2log=0.466 · ±20%=48.0%, n=2698 — **notably reverses the v1 finding**: on v1, `probe-sal` (MAE 6.60) was worse than predicting the median; on v2, the same linear method reaches R² 0.466, most of the way to `dl-sal-s2`'s 0.512. Flagged as an open question (not resolved here) whether this is a property of the new split or an artefact of the old one — left for T1.7/T2.x error analysis. Both rows auto-appended to `04-results.md`. Updated `docs/06-baseline-dl.md` §5.1 and §5.2 (added the probe row to each v2 table, added the "dense vs. probe gap" and "reverses the v1 finding" prose) and `docs/bao-cao/05-chuong-4-thuc-nghiem.md` Bảng 4.5 and Bảng 4.6 (probe rows + matching Vietnamese prose, updated `Nguồn số liệu` blocks). Did **not** touch `docs/00-tong-quan.md` (no status row exists specifically for the probes) or T1.6/T1.7. `pytest -q` 108 passed after the doc updates too |
