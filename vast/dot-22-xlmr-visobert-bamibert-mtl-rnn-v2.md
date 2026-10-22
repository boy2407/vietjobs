# Đợt 22: XLM-R + ViSoBERT + BamiBERT MTL RNN `-v2` (loss mới) — ĐANG CHẠY

Task: T17 — 3 MTL RNN `-v2` cuối cùng. Script: `scripts/vast_run_dot20.sh` (đã thêm `--resume`).

## Vì sao có đợt này

- XLM-R `-v2` mất hai lần: Đợt 19 hủy trước epoch đầu, Đợt 21 dừng ở epoch 24 khi chưa ghi checkpoint.
- ViSoBERT và BamiBERT chưa có bản `-v2` (loss mới của Đợt 20).
- `train_mtl.py` giờ ghi `best.pt` mỗi lần cải thiện, có `last.pt` và `--resume` (commit 5b13830): máy bị ngắt thì chạy lại đúng lệnh, train tiếp.

## Run

| Run | Cấu hình | Trạng thái | dev F1 | dev MAE |
|---|---|---|---|---|
| `dl-mtl-xlm-rnn-b01-cw-s42-v2` | xlmr, len 512, conv 64 | todo | | |
| `dl-mtl-vis-rnn-b01-cw-s42-v2` | visobert, len 512, conv 64 | todo | | |
| `dl-mtl-bam-rnn-b01-cw-s42-v2` | bamibert, len 1024, hidden 100, conv 50 | todo | | |

Chung: `--head rnn --last4`, cw effective 0,999 + ls 0,05 + β 0,1, α 5, patience 12, seed 42, `--device cuda`.

## Instance

Dự kiến: 1x RTX PRO 4000 24 GB (Blackwell), Core Ultra 9 285K, Austria, $0,295/giờ (m:58964). Dự phòng: Slovakia m:150278.
Thực tế: RTX PRO 4000 Blackwell 24 GB (driver 615.71.09), Core Ultra 9 285K (24 vCPU), RAM 125 GB, 100 GB (`/` và `/workspace` cùng ổ), `77.224.107.236:30225`. torch 2.14.1+cu132, capability (12, 0), matmul trên GPU chạy được.

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Push code (`--resume` trong script) | todo |
| 2 | Thuê; `df -h / /workspace`, `free -g` (BamiBERT cần RAM lớn), `nvidia-smi`; torch chạy được một phép tính trên GPU (Blackwell cần bản cu128 trở lên) | done |
| 3 | `git clone -b t8-rnn-colab`, `pip install`, đẩy `data/processed/splits`, `manifest.json`, `data/external/vietjobs37k`; checksum `test.csv` khớp Mac | todo |
| 4 | Chạy trong tmux: `bash scripts/vast_run_dot20.sh rnn:xlmr rnn:visobert rnn:bamibert` (tuần tự) | todo |
| 5 | Sau mỗi run: kéo về theo tên run (không glob), `torch.load` + checksum `best.pt` | todo |
| 6 | Ghi sheet `Score_last4` / `Dashboard_last4` (xem trước với tác giả) | todo |
| 7 | Destroy sau checklist `vast/README.md` | todo |

## Lệnh

```bash
ssh -p PORT root@IP
# kéo một run về Mac
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/<run_id> logs" | tar xzf - -C .
```
