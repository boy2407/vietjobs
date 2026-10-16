# T12 — dashboard configuration map (T12.1a)

Sheet for [TASKS.md](../TASKS.md) section T12. Authority: AGENTS.md Rule 5 item 6 (author, 2026-10-15). Scope: the 34 rows of the author's results dashboard (tab `Dashboard`, 11 columns), mapped to run directories under `artifacts/`. Dashboard values are `dev` numbers; each mapping was checked against `artifacts/<run_id>/metrics.json` (a config's mean F1 over its runs matches the dashboard to 3 decimals; ambiguous rows were decided by accuracy). Dashboard sd is the population sd (ddof=0): that is the only convention that reproduces the 2-seed row `dl-mtl-h256-a1.0`.

Counts: 28 rows ok, 1 partial, 4 blocked pending the author, 1 excluded. 51 run directories are scoreable (20 RNN-head, 31 Dense-head).

## Map

| # | Dashboard configuration | Task | Head | Seeds (dashboard) | Run directories with checkpoint | Status | Note |
|---|---|---|---|---|---|---|---|
| 1 | `dl-cat-cafe-dense-512-p20` | category | dense | 3 | `dl-cat-cafe-dense-512-p20-s42`, `dl-cat-cafe-dense-512-p20-s43`, `dl-cat-cafe-dense-512-p20-s44` | blocked | blocked (author): metrics identical to dl-cat-cafe-dense-512 s42-s44 (0.6077 / 0.6069 / 0.6034), suspected duplicate |
| 2 | `dl-cat-cafe-dense-512` | category | dense | 5 | `dl-cat-cafe-dense-512`, `dl-cat-cafe-dense-512-s43`, `dl-cat-cafe-dense-512-s44`, `dl-cat-cafe-dense-512-s45`, `dl-cat-cafe-dense-512-s46` | ok |  |
| 3 | `dl-cat-cafe-dense` | category | dense | 1 | `dl-cat-cafe-dense` | ok |  |
| 4 | `dl-cat-cafe-focal-512` | category | dense | 5 | `dl-cat-cafe-focal-512-s42`, `dl-cat-cafe-focal-512-s43`, `dl-cat-cafe-focal-512-s44`, `dl-cat-cafe-focal-512-s45`, `dl-cat-cafe-focal-512-s46` | ok |  |
| 5 | `dl-cat-cafe-rnn-ce` | category | rnn | 3 | `dl-cat-cafe-rnn-ce-s42` | partial 1/3 | missing best.pt: dl-cat-cafe-rnn-ce-s43, dl-cat-cafe-rnn-ce-s44 |
| 6 | `dl-cat-ce` | category | dense | 1 | `dl-cat-ce-0920` | ok | dev accuracy 0.6511 -> 0.651 matches; -cpu-0920 (0.6501) and -fix-t23 (F1 0.5983) do not |
| 7 | `dl-cat-focal` | category | dense | 2 | `dl-cat-focal-s2` | blocked | blocked (author): likely pair dl-cat-focal-0920 + dl-cat-focal-s2 (mean F1 0.5959, acc 0.6433, pop sd 0.0003), unconfirmed |
| 8 | `dl-cat-rnn-ce` | category | rnn | 4 | `dl-cat-rnn-ce`, `dl-cat-rnn-ce-s42`, `dl-cat-rnn-ce-s43`, `dl-cat-rnn-ce-s44` | ok |  |
| 9 | `dl-cat-rnn-focal` | category | rnn | 1 | `dl-cat-rnn-focal` | ok |  |
| 10 | `dl-cat-bamibert-dense-1024` | category | dense | 1 | `dl-cat-bamibert-dense-1024` | ok |  |
| 11 | `dl-cat-bamibert-rnn-1024` | category | rnn | 1 | `dl-cat-bamibert-rnn-1024` | ok |  |
| 12 | `dl-cat-visobert-dense-512` | category | dense | 1 | `dl-cat-visobert-dense-512` | ok |  |
| 13 | `dl-cat-visobert-rnn-512` | category | rnn | 1 | `dl-cat-visobert-rnn-512` | ok |  |
| 14 | `dl-cat-xlmr-dense-512` | category | dense | 1 | `dl-cat-xlmr-dense-512` | ok |  |
| 15 | `dl-cat-xlmr-rnn-512` | category | rnn | 1 | `dl-cat-xlmr-rnn-512` | ok |  |
| 16 | `dl-mtl-cafe-rnn-a5` | mtl | rnn | 1 | `dl-mtl-cafe-rnn-a5-s42` | ok |  |
| 17 | `dl-mtl-h256-a1.0` | mtl | dense | 2 | — | blocked | blocked (author): neither run alone matches; the mean of -1001-0735 (best_epoch 2 of 2) and -1001-0746 reproduces all five dashboard numbers (population sd 0.031) |
| 18 | `dl-mtl-bamibert1024-a5-e80` | mtl | dense | 1 | `dl-mtl-bamibert1024-a5-e80-s42` | ok |  |
| 19 | `dl-mtl-bamibert-rnn1024-a5` | mtl | rnn | 1 | `dl-mtl-bamibert-rnn1024-a5-s42` | ok |  |
| 20 | `dl-mtl-visobert512-a5-e80` | mtl | dense | 1 | `dl-mtl-visobert512-a5-e80-s42` | ok |  |
| 21 | `dl-mtl-visobert-rnn512-a5` | mtl | rnn | 1 | `dl-mtl-visobert-rnn512-a5-s42` | ok |  |
| 22 | `dl-mtl-xlmr512-a5-e80` | mtl | dense | 1 | `dl-mtl-xlmr512-a5-e80-s42` | ok |  |
| 23 | `dl-mtl-xlmr-rnn512-a5` | mtl | rnn | 1 | `dl-mtl-xlmr-rnn512-a5-s42` | ok |  |
| 24 | `dl-sal` | salary | dense | 2 | `dl-sal-s2` | blocked | blocked (author): dl-sal-0920 and dl-sal-s2 are identical to 4 decimals (MAE 4.1487, RMSE 8.2917, R2 0.5117), unconfirmed |
| 25 | `dl-sal-cafe-dense-512` | salary | dense | 5 | `dl-sal-cafe-dense-512`, `dl-sal-cafe-dense-512-s43`, `dl-sal-cafe-dense-512-s44`, `dl-sal-cafe-dense-512-s45`, `dl-sal-cafe-dense-512-s46` | ok |  |
| 26 | `dl-sal-cafe-rnn` | salary | rnn | 3 | — | excluded: no checkpoint | no best.pt in any of dl-sal-cafe-rnn-s42, dl-sal-cafe-rnn-s43, dl-sal-cafe-rnn-s44 |
| 27 | `dl-sal-cpu` | salary | dense | 5 | `dl-sal-cpu-0920`, `dl-sal-cpu-s43`, `dl-sal-cpu-s44`, `dl-sal-cpu-s45`, `dl-sal-cpu-s46` | ok | 5 seeds = -0920 + s43..s46 |
| 28 | `dl-sal-rnn` | salary | rnn | 4 | `dl-sal-rnn`, `dl-sal-rnn-s42`, `dl-sal-rnn-s43`, `dl-sal-rnn-s44` | ok |  |
| 29 | `dl-sal-bamibert-dense-1024` | salary | dense | 1 | `dl-sal-bamibert-dense-1024` | ok |  |
| 30 | `dl-sal-bamibert-rnn-1024` | salary | rnn | 1 | `dl-sal-bamibert-rnn-1024` | ok |  |
| 31 | `dl-sal-visobert-dense-512` | salary | dense | 1 | `dl-sal-visobert-dense-512` | ok |  |
| 32 | `dl-sal-visobert-rnn-512` | salary | rnn | 1 | `dl-sal-visobert-rnn-512` | ok |  |
| 33 | `dl-sal-xlmr-dense-512` | salary | dense | 1 | `dl-sal-xlmr-dense-512` | ok |  |
| 34 | `dl-sal-xlmr-rnn-512` | salary | rnn | 1 | `dl-sal-xlmr-rnn-512` | ok |  |

## Missing checkpoints (listed, not retrained)

- `dl-cat-cafe-rnn-ce-s43`, `dl-cat-cafe-rnn-ce-s44`: no `best.pt`. The row reports 1 of 3 seeds, so no sd.
- `dl-sal-cafe-rnn-s42`, `-s43`, `-s44`: no `best.pt`. The whole row is excluded.
- Every other scoreable run has `best.pt` (MTL runs also `best_sal.pt`).

## Blocked pending the author (nothing scored, nothing uploaded)

1. `dl-mtl-h256-a1.0`: confirm that the config means the pair `dl-mtl-h256-a1.0-1001-0735` (2 epochs) + `-1001-0746`, or drop it.
2. `dl-cat-focal`: confirm the pair `dl-cat-focal-0920` + `dl-cat-focal-s2`.
3. `dl-sal`: confirm the pair `dl-sal-0920` + `dl-sal-s2` (identical metrics).
4. `dl-cat-cafe-dense-512-p20`: s42-s44 have the same dev metrics as `dl-cat-cafe-dense-512` s42-s44; confirm they are distinct runs, or drop.

## Caches needed (pooled float32; RNN heads encode per batch and need none)

| Encoder | max_len | `test` raw (category) | `test` masked (salary, mtl) | `ext37k-sal` masked | Status on the Mac |
|---|---|---|---|---|---|
| `phobert` | 256 | yes | yes | yes | ext37k-sal masked done; needs `artifacts/t12/ext37k-sal-masked-seg.parquet` for the rest |
| `cafebert` | 256 | yes | no | no | none |
| `cafebert` | 512 | yes | yes | yes | ext37k-sal masked exists |
| `visobert` | 512 | yes | yes | yes | none |
| `xlmr` | 512 | yes | yes | yes | none |
| `bamibert` | 1024 | yes | yes | yes | none |

Open question for the author: the Dense `category` configs are trained on raw text, so scoring them on `ext37k-sal` needs `ext37k-sal` **raw** caches (6 more files) and the crosswalk path of `eval_external.py`; without that, `category` is not scored on `ext37k-sal` and its cells stay empty. Decide before T12.1d.

Vast plan: [vast/dot-5-t12-score-test.md](../vast/dot-5-t12-score-test.md).
