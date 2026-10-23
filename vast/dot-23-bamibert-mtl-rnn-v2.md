# Đợt 23: BamiBERT MTL RNN `-v2` (loss mới) — XONG (chờ destroy)

Task: T18 — run MTL RNN `-v2` cuối cùng còn thiếu. Script: `scripts/vast_run_dot20.sh rnn:bamibert` (có `--resume`).

## Vì sao có đợt này

- Đợt 22 đã có XLM-R và ViSoBERT `-v2` (đã về Mac); BamiBERT chưa chạy và máy Đợt 22 đã tắt (ssh 2026-10-23: connection refused).
- Đây là encoder duy nhất còn thiếu để đủ 5/5 MTL RNN `-v2`.

## Run

| Run | Cấu hình | Trạng thái | dev F1 | dev MAE | test F1 | ext37k F1 |
|---|---|---|---|---|---|---|
| `dl-mtl-bam-rnn-b01-cw-s42-v2` | bamibert, len 1024, hidden 256, conv 64 (giống 4 run `-v2` kia) | xong, tác giả dừng sớm sau epoch 62 | 0,6267 (ep 21) | 3,660 (ep 62) | 0,6344 | 0,6107 |

Chung: `--head rnn --last4`, cw effective 0,999 + ls 0,05 + β 0,1, α 5, patience 12, epochs 80, seed 42, `--device cuda`.

## Instance

Tiêu chí: 24 GB VRAM (3090 / 3090 Ti / 4090 / RTX PRO 4000), CPU mạnh từng lõi, RAM ≥ 64 GB.
**Đĩa ≥ 150 GB trên ổ chứa `/workspace`** (ước, chưa đo): cache token float16 train+dev ở 1024 = 38.166 × 1024 × 768 × 2 B ≈ 60 GB; lúc chấm thêm test + ext37k ≈ 25 GB.
Thời gian: chưa đo; ước 2,5–3,5 giờ (T18 §6).

Thực tế: 1x RTX 4090 24 GB, 256 vCPU, RAM 1 TB, đĩa 100 GB (`/` và `/workspace` cùng ổ), `38.117.87.59:45480`. `pip install -r requirements-dl.txt` hạ torch về 2.2.2 → cài lại torch 2.14.0+cu130 (bản của image, cùng dòng 2.14 với các run `-v2` trước) + nvidia-nccl-cu13 2.30.7; numpy 1.26.4, transformers 4.46.3. Script `vast_run_dot20.sh` copy thẳng từ Mac (commit a967011, chưa push).

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Thuê; `df -h / /workspace`, `free -g`, `nvidia-smi`; torch chạy được một phép tính trên GPU | done |
| 2 | `git clone -b t8-rnn-colab`, `pip install`, đẩy `data/processed/splits`, `manifest.json`, `data/external/vietjobs37k`; checksum `test.csv` khớp Mac | done (70ba98d5…) |
| 3 | tmux: `bash scripts/vast_run_dot20.sh rnn:bamibert` | done 02:20–07:58 UTC; dừng tay sau epoch 62 (tác giả), ~305 s/epoch |
| 4 ✅ | Watcher kéo `best.pt`, `best_sal.pt`, `scaler.npz`, `config.json`, `history.jsonl`, log khi `metrics.json` xuất hiện; `torch.load` + checksum | todo |
| 5 ✅ | `check_mtl_dev.py` khớp → kéo `metrics_test.json`, `external_37k/metrics.json`, log chấm; lấy `git diff docs/04-results.md` | todo |
| 6 | Destroy sau checklist `vast/README.md` | todo |
| 7 ✅ | Sheet `Dashboard_last4` / `Score_last4` (xem trước với tác giả), tài liệu theo T18 §7 | todo |

## Lệnh

```bash
ssh -p PORT root@IP
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/dl-mtl-bam-rnn-b01-cw-s42-v2 logs" | tar xzf - -C .
```

## Nhật ký

- 07:59 UTC: chốt bằng `--resume --epochs 62` → crash khi ghi `predictions_dev.parquet` (máy thiếu `pyarrow`), sau khi đã ghi `best.pt`/`best_sal.pt` và xoá `last.pt`.
- 08:09: cài `pyarrow`; `metrics.json` dựng lại từ `history.jsonl` (F1 ep 21, MAE/R²log ep 62; có trường `note`). Accuracy/RMSE `dev` không có.
- 08:17: `check_mtl_dev.py` KHỚP (F1 0,6267, MAE 3,660) → chấm: test F1 0,6344, MAE 3,805; ext37k F1 0,6107, MAE 5,018, Spearman 0,667. `best.pt`/`best_sal.pt` về Mac, sha256 khớp, `torch.load` OK. Sheet: Dashboard_last4 dòng 33, Score_last4 dòng 33.
