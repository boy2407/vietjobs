# T23 — retrain the six `last4` structures × 5 encoders on `splits-sal17-11c`

**Goal.** Re-run every structure in the sheet tab `Dashboard_last4` with exactly the same
configuration and change one thing only: the data, which is now `splits-sal17-11c`
([split sheet](T23-split-sal17.md)). Decided with the author in the 2026-11-24 grilling session.

## Decisions (fixed before any number is read)

| # | Decision |
|---|---|
| Data | One split set for **both** tasks: `splits-sal17-11c` (train 21,801 / dev 2,422 / test 6,056, all salary-disclosed, 17-column conflicts removed before splitting). No `conflict_cat` / `conflict_sal` flags are needed: conflicts were removed before the split |
| Input | `masked` family for every structure, `category` included (T4.1 B1/B2: masking costs nothing on dense). Text = 3 base columns + 8 labelled extra fields appended to `requirements_masked` (11 columns) |
| Encoders | PhoBERT-v2 256, CafeBERT 512, XLM-R 512, ViSoBERT 512, BamiBERT 1024 — the old lengths; `truncated_frac` measured on 11c and recorded here |
| Configuration | Losses as in `Dashboard_last4`: `last4` pooling; `category` CE + class weight effective number β 0.999 + label smoothing 0.05; `salary` SmoothL1 β 0.1 + `--class-weight`; MTL α = 5. **One shared hyper-parameter set for all six structures** (author, 2026-11-24, so structures compare): lr 3e-4, weight decay 0.01, dropout 0.3, **epochs 80, patience 12**; Dense hidden 256; RNN h=100 / c=50 for both single-task RNN and MTL RNN. This deviates from the sheet in epochs/patience (single task 40/8, MTL Dense 80/15) and MTL RNN hidden (256). ViSoBERT kept as is even though Dense + `last4` collapsed there (0.2412); a collapse again is recorded as a failure with its cause |
| MTL salary loss | Masked on undisclosed rows and averaged over disclosed rows in the batch (`keep_sal`). Has no effect on this set (every row is disclosed); kept so the code path is unchanged |
| Seeds | Dense structures (Dense `category`, Dense `salary`, MTL Dense): 5 seeds 42–46; RNN structures (RNN `category`, RNN `salary`, MTL RNN): seed 42 → 75 + 15 = **90 runs** (author, 2026-11-24) |
| Device | Every Dense structure (MTL Dense included) on `cpu` (Rule 13); every RNN structure on `cuda` |
| Where | vast.ai, **one rental** (rental 28), ceiling **15 USD**. If the estimate exceeds it, report to the author before renting |
| Order | encode pooled caches → Dense, MTL Dense → score `test` + ext37k → encode token caches → RNN, MTL RNN → score `test` + ext37k |
| External set | Every trained model (all 90) is scored on `data/external/vietjobs37k/ext37k-sal.csv`. The model only takes text, so ext37k's text (its 3 masked columns, joined as `dl/text.py` does) goes in as it is (author, 2026-11-24). Columns and metrics as tab `Score_last4`: Test F1-macro, Accuracy, MAE, RMSE, R² log; Ext37K F1-macro (strict), Accuracy (strict), MAE, RMSE, R² log. Written to tab `Score_sal17` |
| `test` | Selection on `dev` only. Each run is scored once on `test` from its saved `best.pt` (Rule 5 item 6), command typed by the author, rows tagged test-informed |
| Results | New sheet tabs `Dashboard_sal17` and `Score_sal17`; `Dashboard_last4` untouched. Not comparable to any earlier row in `04-results.md` (different `test`) |

## Tasks

| ID | What | Status | Owner | Depends | Evidence | Cost |
|---|---|---|---|---|---|---|
| T23.0 | Code: `train_dl` reads the `masked` family for `category` (flag + test); one shared hyper-parameter set in `scripts/vast_run_t23.sh` | done | Claude | — | `pytest -q` green; a smoke run's `config.json` shows `masked` + shared params | Mac, ~1 h |
| T23.1 | Rental file `vast/dot-28-t23-train-lai.md` + `vast/README.md` row | done | Claude | T23.0 | file exists, status `CHƯA THUÊ` | Mac |
| T23.2 | Encode pooled `last4` `masked` caches on 11c, 5 encoders, train/dev — `test` cache belongs to the scoring step (Rule 5); record `truncated_frac` | done | Claude | T23.1 | 15 `.npy` + sidecars on the Mac | vast GPU ~1 h, ~5 GB |
| T23.3 | Dense `category` × 5 encoders × seeds 42–46 `dl-cat-{enc}-dense-sal17-s{42..46}` | done | Claude | T23.2 | 25 rows; mean ± sd macro-F1 per encoder | cpu, ~25 × 15 s |
| T23.4 | Dense `salary` × 5 × seeds 42–46 `dl-sal-{enc}-dense-sal17-s{42..46}` | done | Claude | T23.2 | 25 rows; mean ± sd MAE | cpu, ~25 × 15 s |
| T23.5 | MTL Dense × 5 × seeds 42–46 `dl-mtl-{enc}-dense-sal17-s{42..46}` | done | Claude | T23.2 | 25 rows; macro-F1 and MAE mean ± sd | cpu, ~25 × a few min |
| T23.6 | Score T23.3–T23.5 on `test` (author types the command) | todo | — | T23.3–T23.5 | 75 test rows, test-informed; `Score_sal17` | vast ~10 min |
| T23.6b | Encode ext37k pooled `last4` `masked` caches (3 columns), 5 encoders; score T23.3–T23.5 on ext37k | todo | — | T23.6 | 75 ext37k rows in `Score_sal17` | vast GPU ~20 min |
| T23.7 | Encode token caches, 5 encoders (11c train/dev — `test` cache belongs to the scoring step (Rule 5)) | todo | — | T23.6b | caches present; disk measured | vast GPU ~2 h, ~150 GB disk |
| T23.8 | RNN `category` × 5, seed 42 `dl-cat-{enc}-rnn-sal17-s42` | todo | — | T23.7 | 5 rows | cuda ~30 min each |
| T23.9 | RNN `salary` × 5, seed 42 `dl-sal-{enc}-rnn-sal17-s42` | todo | — | T23.7 | 5 rows | cuda ~30 min each |
| T23.10 | MTL RNN × 5, seed 42 `dl-mtl-{enc}-rnn-sal17-s42` | todo | — | T23.7 | 5 rows | cuda ~1 h each |
| T23.11 | Score T23.8–T23.10 on `test`; log, docs, thesis, workbook, sheet `Dashboard_sal17` (Rules 1, 7, 10, 13) | todo | — | T23.8–T23.10 | `04-results.md` 90 rows; `vast/README.md` checklist passes | Mac + vast |
| T23.11b | Encode ext37k token caches; score T23.8–T23.10 on ext37k | todo | — | T23.7, T23.10 | 15 ext37k rows; all 90 runs have an ext37k score | vast GPU ~40 min, ~60 GB disk |

Cost check: if the estimate for T23.7–T23.11b would push the rental past 15 USD, report to the author before T23.7.

**Done when** 90 rows are in `04-results.md`, **every final model has a `test` and an ext37k score**, the two sheet tabs are filled, the
`vast/README.md` checklist passes before the instance is destroyed, and `pytest -q` is green.

## Log

| Date | Who | What |
|---|---|---|
| 2026-11-24 | Claude | Sheet created from the grilling session; AGENTS.md Rule 5 item 7 records the split exception. Nothing rented or run yet |
| 2026-11-24 | Claude | Author approved the split (tach-task): Dense structures 5 seeds, one shared hyper-parameter set, every model scored on ext37k; rows T23.0–T23.11b |
| 2026-11-24 | Claude | T23.0: `--family masked` in `encode.py`/`train_dl.py` + `tests/test_dl_family.py`; `pytest -q` 229 passed; smoke run `config.json` showed `family: masked`, `last4`, shared params. T23.1: `vast/dot-28-t23-train-lai.md` (CHƯA THUÊ) + `scripts/vast_run_t23.sh`. Estimate ~25 h, ~5–9 USD (unmeasured). Waiting for the author to rent |
| 2026-11-25 | Claude | Author moved the training to the Windows desktop (ssh `desktop`, RTX 5060 8 GB, Ryzen 7 7800X3D, RAM 31 GB, no WSL): repo `C:\Users\trnghia\vietjobs`, `.venv-dl` Python 3.12 + torch 2.11.0+cu128, runner `scripts/run_t23_windows.ps1` (same flags and run_ids as `vast_run_t23.sh`). `load_split` failed on the 7 new `*_masked` columns → `splits-sal17-11c/manifest.json` (written by `build_sal17_11c_split.py --manifest-only`), CSV md5 unchanged. Smoke: BamiBERT 1024 `last4` pooled on dev 58 s, peak VRAM 1.59 GB, `truncated_frac` 0.0029 |
| 2026-11-25 | Claude | T23.2–T23.5 done on the desktop (`run_t23_windows.ps1 pooled dense`, started 09:18 detached from ssh via `Win32_Process.Create`; a scheduled task could not write the repo without elevation). Encode 09:18–09:59 (PhoBERT 3 min, CafeBERT 20, XLM-R 6, ViSoBERT 5, BamiBERT 7). 75 Dense runs 09:59–10:22, 6–22 s each, all stopped by patience; 100 `best*.pt` pulled and `torch.load` checked. Truncation on train: PhoBERT 256 cuts 67.4 % of postings / 21.4 % of tokens; 512 encoders 17–19 % / 5 %; BamiBERT 0.24 % / 0.10 % (`logs/t23-truncation.txt`). Dev, mean ± sd over 5 seeds: best is CafeBERT (Dense cat 0.6231 ± 0.0047, Dense sal MAE 4.122 ± 0.014, MTL 0.6248 / 4.092); ViSoBERT lowest (0.5795), no collapse. Results in `docs/06-baseline-dl.md` §5.10 |
