# T12 — dashboard configuration map (T12.1a)

Sheet for [TASKS.md](../TASKS.md) section T12. Authority: AGENTS.md Rule 5 item 6 (author, 2026-10-15); scope widened by the author on 2026-10-16 to **all 34 dashboard rows**. Dashboard values are `dev` numbers; each mapping was checked against `artifacts/<run_id>/metrics.json` (a config's mean F1 over its runs matches the dashboard to 3 decimals; ambiguous rows were decided by accuracy or by the mean of the runs). The dashboard sd is the population sd (ddof=0): that is the only convention that reproduces the 2-seed row `dl-mtl-h256-a1.0`.

Counts: 32 rows map to existing checkpoints, 2 rows are retrained (below). Runs to score: 65 = 59 with an existing checkpoint (40 Dense-head, 19 RNN-head) + 6 retrained (`-t12`, RNN).

## Map

| # | Dashboard configuration | Task | Head | Seeds (dashboard) | Run directories with checkpoint | Status | Note |
|---|---|---|---|---|---|---|---|
| 1 | `dl-cat-cafe-dense-512-p20` | category | dense | 3 | `dl-cat-cafe-dense-512-p20-s42`, `dl-cat-cafe-dense-512-p20-s43`, `dl-cat-cafe-dense-512-p20-s44` | ok (author) | own run directories scored as asked; suspected duplicate: dev metrics identical to dl-cat-cafe-dense-512 s42-s44 (0.6077 / 0.6069 / 0.6034) |
| 2 | `dl-cat-cafe-dense-512` | category | dense | 5 | `dl-cat-cafe-dense-512`, `dl-cat-cafe-dense-512-s43`, `dl-cat-cafe-dense-512-s44`, `dl-cat-cafe-dense-512-s45`, `dl-cat-cafe-dense-512-s46` | ok |  |
| 3 | `dl-cat-cafe-dense` | category | dense | 1 | `dl-cat-cafe-dense` | ok |  |
| 4 | `dl-cat-cafe-focal-512` | category | dense | 5 | `dl-cat-cafe-focal-512-s42`, `dl-cat-cafe-focal-512-s43`, `dl-cat-cafe-focal-512-s44`, `dl-cat-cafe-focal-512-s45`, `dl-cat-cafe-focal-512-s46` | ok |  |
| 5 | `dl-cat-cafe-rnn-ce` | category | rnn | 3 | `dl-cat-cafe-rnn-ce-s42-t12`, `dl-cat-cafe-rnn-ce-s43-t12`, `dl-cat-cafe-rnn-ce-s44-t12` | retrain (T12.1b) | original checkpoints unusable: s42 best.pt is not a valid zip (6,641,333 B, truncated), s43 and s44 have none. Retrained with the original config (vast_run_cafe_rnn.sh) under new run_ids; dev numbers may differ from the dashboard |
| 6 | `dl-cat-ce` | category | dense | 1 | `dl-cat-ce-0920` | ok | author-approved mapping; dev accuracy 0.6511 -> 0.651 matches the dashboard; -cpu-0920 (0.6501) and -fix-t23 (F1 0.5983) do not |
| 7 | `dl-cat-focal` | category | dense | 2 | `dl-cat-focal-0920`, `dl-cat-focal-s2` | ok (author) | author-approved mapping: mean F1 0.5959 -> 0.596, acc 0.6433 -> 0.643, population sd 0.0003 -> 0.000; the other pairs fail on acc or F1 |
| 8 | `dl-cat-rnn-ce` | category | rnn | 4 | `dl-cat-rnn-ce`, `dl-cat-rnn-ce-s42`, `dl-cat-rnn-ce-s43`, `dl-cat-rnn-ce-s44` | ok |  |
| 9 | `dl-cat-rnn-focal` | category | rnn | 1 | `dl-cat-rnn-focal` | ok |  |
| 10 | `dl-cat-bamibert-dense-1024` | category | dense | 1 | `dl-cat-bamibert-dense-1024` | ok |  |
| 11 | `dl-cat-bamibert-rnn-1024` | category | rnn | 1 | `dl-cat-bamibert-rnn-1024` | ok |  |
| 12 | `dl-cat-visobert-dense-512` | category | dense | 1 | `dl-cat-visobert-dense-512` | ok |  |
| 13 | `dl-cat-visobert-rnn-512` | category | rnn | 1 | `dl-cat-visobert-rnn-512` | ok |  |
| 14 | `dl-cat-xlmr-dense-512` | category | dense | 1 | `dl-cat-xlmr-dense-512` | ok |  |
| 15 | `dl-cat-xlmr-rnn-512` | category | rnn | 1 | `dl-cat-xlmr-rnn-512` | ok |  |
| 16 | `dl-mtl-cafe-rnn-a5` | mtl | rnn | 1 | `dl-mtl-cafe-rnn-a5-s42` | ok |  |
| 17 | `dl-mtl-h256-a1.0` | mtl | dense | 2 | `dl-mtl-h256-a1.0-1001-0735`, `dl-mtl-h256-a1.0-1001-0746` | ok (author) | author-approved mapping: the mean of the two runs reproduces all five dashboard numbers (F1 0.5725 -> 0.573, acc 0.6367 -> 0.637, RMSE 8.378 -> 8.38, MAE 4.398 -> 4.40, R2 0.4535 -> 0.453; population sd 0.0315 -> 0.031); -0735 stopped at epoch 2; no best_sal.pt, both heads read best.pt |
| 18 | `dl-mtl-bamibert1024-a5-e80` | mtl | dense | 1 | `dl-mtl-bamibert1024-a5-e80-s42` | ok |  |
| 19 | `dl-mtl-bamibert-rnn1024-a5` | mtl | rnn | 1 | `dl-mtl-bamibert-rnn1024-a5-s42` | ok |  |
| 20 | `dl-mtl-visobert512-a5-e80` | mtl | dense | 1 | `dl-mtl-visobert512-a5-e80-s42` | ok |  |
| 21 | `dl-mtl-visobert-rnn512-a5` | mtl | rnn | 1 | `dl-mtl-visobert-rnn512-a5-s42` | ok |  |
| 22 | `dl-mtl-xlmr512-a5-e80` | mtl | dense | 1 | `dl-mtl-xlmr512-a5-e80-s42` | ok |  |
| 23 | `dl-mtl-xlmr-rnn512-a5` | mtl | rnn | 1 | `dl-mtl-xlmr-rnn512-a5-s42` | ok |  |
| 24 | `dl-sal` | salary | dense | 2 | `dl-sal-0920`, `dl-sal-s2` | ok (author) | author-approved mapping: metrics identical to 4 decimals (MAE 4.1487, RMSE 8.2917, R2 0.5117), probably a duplicate rerun, so the sd is 0 |
| 25 | `dl-sal-cafe-dense-512` | salary | dense | 5 | `dl-sal-cafe-dense-512`, `dl-sal-cafe-dense-512-s43`, `dl-sal-cafe-dense-512-s44`, `dl-sal-cafe-dense-512-s45`, `dl-sal-cafe-dense-512-s46` | ok |  |
| 26 | `dl-sal-cafe-rnn` | salary | rnn | 3 | `dl-sal-cafe-rnn-s42-t12`, `dl-sal-cafe-rnn-s43-t12`, `dl-sal-cafe-rnn-s44-t12` | retrain (T12.1b) | original s42-s44 have no best.pt. Retrained with the original config under new run_ids; dev numbers may differ from the dashboard |
| 27 | `dl-sal-cpu` | salary | dense | 5 | `dl-sal-cpu-0920`, `dl-sal-cpu-s43`, `dl-sal-cpu-s44`, `dl-sal-cpu-s45`, `dl-sal-cpu-s46` | ok | 5 seeds = -0920 + s43..s46 |
| 28 | `dl-sal-rnn` | salary | rnn | 4 | `dl-sal-rnn`, `dl-sal-rnn-s42`, `dl-sal-rnn-s43`, `dl-sal-rnn-s44` | ok |  |
| 29 | `dl-sal-bamibert-dense-1024` | salary | dense | 1 | `dl-sal-bamibert-dense-1024` | ok |  |
| 30 | `dl-sal-bamibert-rnn-1024` | salary | rnn | 1 | `dl-sal-bamibert-rnn-1024` | ok |  |
| 31 | `dl-sal-visobert-dense-512` | salary | dense | 1 | `dl-sal-visobert-dense-512` | ok |  |
| 32 | `dl-sal-visobert-rnn-512` | salary | rnn | 1 | `dl-sal-visobert-rnn-512` | ok |  |
| 33 | `dl-sal-xlmr-dense-512` | salary | dense | 1 | `dl-sal-xlmr-dense-512` | ok |  |
| 34 | `dl-sal-xlmr-rnn-512` | salary | rnn | 1 | `dl-sal-xlmr-rnn-512` | ok |  |

## Missing checkpoints: retrained inside T12.1 (author, 2026-10-16)

Six CafeBERT 512 + BiGRU-LSTM-CNN runs have no usable checkpoint and are retrained on the rental ([vast/dot-5-t12-score-test.md](../vast/dot-5-t12-score-test.md), stage `train`), with the original config (`scripts/vast_run_cafe_rnn.sh`: `--head rnn --encoder cafebert --max-len 512 --hidden 100 --conv-channels 50`, `category` with `--loss ce`, same seeds, defaults otherwise) and **new run_ids** with the suffix `-t12`. The original directories are never overwritten.

| Original run | Why unusable | Retrained as |
|---|---|---|
| `dl-cat-cafe-rnn-ce-s42` | `best.pt` is not a valid zip archive (6,641,333 B, truncated) | `dl-cat-cafe-rnn-ce-s42-t12` |
| `dl-cat-cafe-rnn-ce-s43`, `-s44` | no `best.pt` | `dl-cat-cafe-rnn-ce-s43-t12`, `-s44-t12` |
| `dl-sal-cafe-rnn-s42`, `-s43`, `-s44` | no `best.pt` | `dl-sal-cafe-rnn-s42-t12` … `-s44-t12` |

The retrained dev numbers get ordinary `04-results.md` rows and may differ from the dashboard's (same seed and config, but a different machine and library state). The `dev` numbers in the dashboard stay as they are. Checkpoint integrity was scanned over all 65 planned runs (`zipfile` + `torch.load`): `dl-cat-cafe-rnn-ce-s42` is the only corrupt one.

## Mappings the author approved (2026-10-16)

| Dashboard row | Mapped to | Evidence |
|---|---|---|
| `dl-cat-ce` | `dl-cat-ce-0920` | dev accuracy 0.6511 → 0.651; `-cpu-0920` has 0.6501, `-fix-t23` has F1 0.5983 |
| `dl-mtl-h256-a1.0` | `-1001-0735` + `-1001-0746` | the mean of the two reproduces F1, acc, RMSE, MAE and R²; neither run alone does; `-0735` stopped at epoch 2 |
| `dl-cat-focal` | `dl-cat-focal-0920` + `dl-cat-focal-s2` | mean F1 0.5959, acc 0.6433, population sd 0.0003; the other pairs fail on acc or F1 |
| `dl-sal` | `dl-sal-0920` + `dl-sal-s2` | identical metrics to 4 decimals (probably a duplicate rerun, sd 0) |
| `dl-cat-cafe-dense-512-p20` | its own `-s42` … `-s44` | scored as asked; dev metrics are identical to `dl-cat-cafe-dense-512` s42-s44 (suspected duplicate, kept as a note) |

## Caches needed (pooled float32; RNN heads encode per batch and need none)

| Encoder | max_len | `test` raw | `test` masked | `ext37k-sal` masked | `ext37k-sal` raw |
|---|---|---|---|---|---|
| `phobert` | 256 | yes | yes | yes | yes |
| `cafebert` | 256 | yes | no | no | yes |
| `cafebert` | 512 | yes | yes | yes | yes |
| `visobert` | 512 | yes | yes | yes | yes |
| `xlmr` | 512 | yes | yes | yes | yes |
| `bamibert` | 1024 | yes | yes | yes | yes |

`ext37k-sal` raw text is the file's raw columns through `eval_external.prepare_frame` (the same preprocessing and word segmentation used for the 37K gold/test scoring). The retraining also needs token caches for CafeBERT 512 (`category` raw and `salary` masked, train + dev), which are built on the rental and not copied back.

Vast plan: [vast/dot-5-t12-score-test.md](../vast/dot-5-t12-score-test.md).
