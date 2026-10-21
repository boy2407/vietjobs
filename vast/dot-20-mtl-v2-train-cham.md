# Đợt 20: Train lại MTL với loss mới, train xong chấm ngay trên cùng máy — CHƯA THUÊ

Task: T17 (phần còn lại). Script: `scripts/vast_run_dot20.sh` (+ `scripts/check_mtl_dev.py`).

## Vì sao có đợt này

1. **Loss mới cho mọi MTL** (yêu cầu tác giả, 2026-10-21): đầu ngành **last4 + cw effective 0,999 + label smoothing 0,05**; đầu lương **last4 + cw + β = 0,1**. `train_mtl.py` đã có `--cw-mode/--cw-beta/--label-smoothing`, và RNN giờ đọc cache token last4.
2. **Train và chấm phải cùng môi trường** (Rule 4). Đợt 17 chứng minh: `best.pt` train ở môi trường cũ, chấm ở máy mới thì số phồng (PhoBERT MTL Dense dev 0,5809 lúc train → 0,6215 khi encode lại); còn CafeBERT/ViSoBERT train và chấm cùng máy thì `dev` khớp tuyệt đối.
3. BamiBERT và XLM-R của Đợt 17/18 chưa xong (tác giả tắt máy lúc ~10:20 UTC) và dùng loss cũ.

## Danh sách run (đuôi `-v2`, run cũ giữ nguyên)

| Nhóm | Run | Cấu hình |
|---|---|---|
| MTL RNN ×5 | `dl-mtl-{pho,caf,vis,xlm,bam}-rnn-b01-cw-s42-v2` | `--head rnn --last4`, loss mới, α 5, patience 12, seed 42; BamiBERT hidden 100 / conv 50, còn lại conv 64 |
| MTL Dense ×5 | `dl-mtl-{pho,caf,vis,xlm,bam}-dense-last4-cw-s42-v2` | `--head dense --last4`, loss mới, α 5, patience 15, seed 42 |
| Salary Dense ×25 | `dl-sal-{pho,cafe,visobert,xlmr,bamibert}-dense-last4-b01-cw-s{42..46}-v2` | lệnh gốc Đợt 5 (last4 + cw + β 0,1), `--device cpu` |

## Quy trình mỗi job (tự động trong script)

`rnn:<enc>`: train → `check_mtl_dev.py` (dev chấm lại phải khớp `metrics.json` ±0,001, lệch thì **dừng, không chấm test**) → `eval_mtl_all.py` chỉ run đó (`ONLY`, `RUN_SUFFIX=-v2`) → xoá cache token.
`dense`: encode pooled last4 train/dev 5 encoder → 5 MTL Dense + kiểm dev → 25 Salary Dense → `eval_dense_all.py` (`RUN_SUFFIX=-v2`).
Script có `pipefail` và tự dừng nếu ổ chứa `artifacts/` còn < 70 GB.

## Chia máy (đề xuất, ~3,5–4,5 giờ wall-clock — ước tính từ ~2,5 phút/epoch đo ở Đợt 17)

| Máy | Lệnh | Yêu cầu |
|---|---|---|
| A: RTX 5090 / 4090, RAM ≥ 128 GB | `bash scripts/vast_run_dot20.sh rnn:bamibert dense` | đĩa ≥ 100 GB **trên ổ `/workspace`** |
| B: RTX 3090 | `bash scripts/vast_run_dot20.sh rnn:cafebert rnn:phobert` | đĩa ≥ 100 GB, CPU mạnh từng lõi (train bị chặn ở CPU) |
| C: RTX 3090 | `bash scripts/vast_run_dot20.sh rnn:visobert rnn:xlmr` | như B |

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Commit + push code (`train_mtl.py`, `eval_*`, `check_mtl_dev.py`, script này) | todo |
| 2 | Thuê máy; `df -h / /workspace`, `nvidia-smi`, torch thấy GPU | todo |
| 3 | `git clone -b t8-rnn-colab`, `pip install` (như Đợt 17), đẩy `data/processed/splits`, `manifest.json`, `data/external/vietjobs37k` bằng `cd /workspace/vietjobs && tar xzf -` | todo |
| 4 | Chạy lệnh của máy trong tmux | todo |
| 5 | Kéo về theo tên run (không glob), `torch.load` + checksum từng `best.pt` | todo |
| 6 | Ghi sheet `Score_last4` / `Dashboard_last4` (xem trước với tác giả) | todo |
| 7 | Destroy sau checklist `vast/README.md` | todo |
