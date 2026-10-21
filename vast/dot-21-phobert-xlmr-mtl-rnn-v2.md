# Round 21: PhoBERT + XLM-R MTL RNN `-v2` (combined loss) on one RTX 3090 Ti — RUNNING

Task: T17 (remaining MTL RNN runs). Script: `scripts/vast_run_dot20.sh rnn:phobert` and `rnn:xlmr` (one tmux session each, run in parallel).
Instance: 1x RTX 3090 Ti 24 GB, AMD Ryzen 9 5950X, RAM 125 GB, 100 GB disk (`/` and `/workspace` on the same volume), `171.227.216.98:48739`. torch 2.14.1+cu126, transformers 4.46.3.

| # | Step | Status |
|---|---|---|
| 1 | Set up, upload splits + ext37k (`test.csv` checksum matches the Mac) | done |
| 2 | PhoBERT `dl-mtl-pho-rnn-b01-cw-s42-v2`: 14:45–15:55 UTC, early stop at epoch 58, dev gate matched (F1 0.6261, MAE 3.653), scored on test + ext37k, pulled to the Mac and verified 1/1 (`torch.load`, checksum) | done |
| 3 | XLM-R `dl-mtl-xlm-rnn-b01-cw-s42-v2`: started 15:14 UTC in parallel; paused (SIGSTOP) 15:31 while PhoBERT finished training, resumed 15:53 (watcher script) | running |
| 4 | Pull XLM-R to the Mac and verify | todo |
| 5 | Destroy the instance after the checklist in `vast/README.md` | todo |

Measured on this machine: PhoBERT 60 s/epoch alone, 64–103 s/epoch while XLM-R was encoding; XLM-R 176 s/epoch in parallel.
