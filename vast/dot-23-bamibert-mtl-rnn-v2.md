# Đợt 23: BamiBERT MTL RNN `-v2` (loss mới) — CHƯA THUÊ

Task: T18 — run MTL RNN `-v2` cuối cùng còn thiếu. Script: `scripts/vast_run_dot20.sh rnn:bamibert` (có `--resume`).

## Vì sao có đợt này

- Đợt 22 đã có XLM-R và ViSoBERT `-v2` (đã về Mac); BamiBERT chưa chạy và máy Đợt 22 đã tắt (ssh 2026-10-23: connection refused).
- Đây là encoder duy nhất còn thiếu để đủ 5/5 MTL RNN `-v2`.

## Run

| Run | Cấu hình | Trạng thái | dev F1 | dev MAE | test F1 | ext37k F1 |
|---|---|---|---|---|---|---|
| `dl-mtl-bam-rnn-b01-cw-s42-v2` | bamibert, len 1024, hidden 256, conv 64 (giống 4 run `-v2` kia) | todo | | | | |

Chung: `--head rnn --last4`, cw effective 0,999 + ls 0,05 + β 0,1, α 5, patience 12, epochs 80, seed 42, `--device cuda`.

## Instance

Tiêu chí: 24 GB VRAM (3090 / 3090 Ti / 4090 / RTX PRO 4000), CPU mạnh từng lõi, RAM ≥ 64 GB.
**Đĩa ≥ 150 GB trên ổ chứa `/workspace`** (ước, chưa đo): cache token float16 train+dev ở 1024 = 38.166 × 1024 × 768 × 2 B ≈ 60 GB; lúc chấm thêm test + ext37k ≈ 25 GB.
Thời gian: chưa đo; ước 2,5–3,5 giờ (T18 §6).

Thực tế: —

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Thuê; `df -h / /workspace`, `free -g`, `nvidia-smi`; torch chạy được một phép tính trên GPU | todo |
| 2 | `git clone -b t8-rnn-colab`, `pip install`, đẩy `data/processed/splits`, `manifest.json`, `data/external/vietjobs37k`; checksum `test.csv` khớp Mac | todo |
| 3 | tmux: `bash scripts/vast_run_dot20.sh rnn:bamibert` | todo |
| 4 | Watcher kéo `best.pt`, `best_sal.pt`, `scaler.npz`, `config.json`, `history.jsonl`, log khi `metrics.json` xuất hiện; `torch.load` + checksum | todo |
| 5 | `check_mtl_dev.py` khớp → kéo `metrics_test.json`, `external_37k/metrics.json`, log chấm; lấy `git diff docs/04-results.md` | todo |
| 6 | Destroy sau checklist `vast/README.md` | todo |
| 7 | Sheet `Dashboard_last4` / `Score_last4` (xem trước với tác giả), tài liệu theo T18 §7 | todo |

## Lệnh

```bash
ssh -p PORT root@IP
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/dl-mtl-bam-rnn-b01-cw-s42-v2 logs" | tar xzf - -C .
```
