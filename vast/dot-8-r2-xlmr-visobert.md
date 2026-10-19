# Đợt 8: train lại XLM-R 512 (6 run) + ViSoBERT 512 RNN (3 run) trên đúng bộ chia của Mac (`-r2`) — ĐANG CHẠY

Task: train lại 9/15 run T11 có dev lệch Mac ([tasks/T16-train-lai-r2.md](../tasks/T16-train-lai-r2.md)); máy A (BamiBERT) ở [dot-9](dot-9-r2-bamibert.md).
Script: `scripts/vast_run_r2.sh xlmr visobert` | Instance: RTX 5060 Ti 16 GB, 112 nhân, RAM 503 GB, **disk 120 GB**, `ssh -p 43372 root@158.181.52.18`, thư mục `/workspace/vietjobs`, tmux `r2`, log `logs/r2-all.txt`.

Lý do: `predictions_dev` của các run Đợt 4 khớp nhãn dev.csv của Mac chỉ 93,1 % (mtl/sal: 2.705 tin có lương, Mac 2.698) → Đợt 4 train trên bộ chia khác. Lần này script kiểm sha256 `train/dev/test.csv` + `manifest.json` trước train, sau train và sau chấm; lệch là dừng.

Đĩa 120 GB < 200 GB ước tính → token cache mỗi encoder (≈ 60 GB, raw + masked) bị xoá ngay sau 3 run RNN của nó.

## Run

| Run | Thời gian Đợt 4 (đo) | Trạng thái | Dev | Test | Ext37K |
|---|---|---|---|---|---|
| `dl-cat-xlmr-dense-512-r2` | < 1 phút | todo | | | |
| `dl-sal-xlmr-dense-512-r2` | 1 phút | todo | | | |
| `dl-mtl-xlmr512-a5-e80-s42-r2` | 2 phút | todo | | | |
| `dl-cat-xlmr-rnn-512-r2` | 29 phút | todo | | | |
| `dl-sal-xlmr-rnn-512-r2` | 19 phút | todo | | | |
| `dl-mtl-xlmr-rnn512-a5-s42-r2` | 68 phút | todo | | | |
| `dl-cat-visobert-rnn-512-r2` | 75 phút | todo | | | |
| `dl-sal-visobert-rnn-512-r2` | 53 phút | todo | | | |
| `dl-mtl-visobert-rnn512-a5-s42-r2` | 71 phút | todo | | | |

Ước tính tổng ≈ 7 giờ (train 5,3 giờ đo ở Đợt 4 + nhúng, chấm chưa đo).

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy | done | tác giả, 2026-10-19 |
| 2 | SSH | done | key `id_ed25519` |
| 3 | Đẩy code + splits + manifest + `ext37k-sal.csv` | done | sha256 trên máy thuê khớp Mac (4 file) |
| 4 | Cài thư viện, chạy script trong tmux | doing | khởi chạy 2026-10-19; `sha256 ok (before)`, pytest 25 passed; XLM-R encode ≈ 104 tin/s |
| 5 | Kéo `artifacts/*-r2/` (kèm `best.pt`) + log về Mac | todo | |
| 6 | Kiểm `predictions_dev` khớp 100 % dev.csv Mac | todo | |
| 7 | Checklist `README.md`, destroy | todo | |

## Lệnh

```bash
# Mac
rsync -avzR -e "ssh -p 43372" src scripts tests pyproject.toml data/processed/splits data/processed/manifest.json \
  data/external/vietjobs37k/ext37k-sal.csv data/external/vietjobs37k/crosswalk_60_to_16.json root@158.181.52.18:/workspace/vietjobs/
# máy thuê
cd /workspace/vietjobs && source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn pyarrow underthesea pyvi pytest
tmux new -d -s r2 "bash scripts/vast_run_r2.sh xlmr visobert 2>&1 | tee logs/r2-all.txt"
```

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
