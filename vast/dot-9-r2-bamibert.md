# Đợt 9: train lại BamiBERT 1024 (6 run) trên đúng bộ chia của Mac (`-r2`) — ĐANG CHẠY

Task: train lại 6/15 run T11 có dev lệch Mac ([tasks/T16-train-lai-r2.md](../tasks/T16-train-lai-r2.md)); máy B (XLM-R + ViSoBERT) ở [dot-8](dot-8-r2-xlmr-visobert.md).
Script: `scripts/vast_run_r2.sh bamibert` | Instance: RTX 5060 Ti 16 GB, 32 nhân, RAM 377 GB, **disk 120 GB**, `ssh -p 45000 root@85.10.218.46`, thư mục `/workspace/vietjobs`, tmux `r2`, log `logs/r2-all.txt`.

Lý do: `predictions_dev` của các run Đợt 4 khớp nhãn dev.csv của Mac chỉ 93,1 % (mtl/sal: 2.705 tin có lương, Mac 2.698) → Đợt 4 train trên bộ chia khác. Lần này script kiểm sha256 `train/dev/test.csv` + `manifest.json` trước train, sau train và sau chấm; lệch là dừng.

Đĩa 120 GB < 200 GB ước tính → token cache BamiBERT 1024 ≈ 60 GB mỗi họ → nhúng từng họ (raw, rồi masked), xoá ngay sau các run dùng nó.

## Run

| Run | Thời gian Đợt 4 (đo) | Trạng thái | Dev | Test | Ext37K |
|---|---|---|---|---|---|
| `dl-cat-bamibert-dense-1024-r2` | < 1 phút | todo | | | |
| `dl-sal-bamibert-dense-1024-r2` | 1 phút | todo | | | |
| `dl-mtl-bamibert1024-a5-e80-s42-r2` | 2 phút | todo | | | |
| `dl-cat-bamibert-rnn-1024-r2` | 100 phút | todo | | | |
| `dl-sal-bamibert-rnn-1024-r2` | 84 phút | todo | | | |
| `dl-mtl-bamibert-rnn1024-a5-s42-r2` | 116 phút | todo | | | |

Ước tính tổng ≈ 7,5 giờ (train 5,1 giờ đo ở Đợt 4 + nhúng, chấm chưa đo).

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy | done | tác giả, 2026-10-19 |
| 2 | SSH | done | key `id_ed25519` |
| 3 | Đẩy code + splits + manifest + `ext37k-sal.csv` | done | sha256 trên máy thuê khớp Mac (4 file) |
| 4 | Cài thư viện, chạy script trong tmux | doing | khởi chạy 2026-10-19; `sha256 ok (before)`, pytest 25 passed; BamiBERT encode ≈ 82 tin/s |
| 5 | Kéo `artifacts/*-r2/` (kèm `best.pt`) + log về Mac | todo | |
| 6 | Kiểm `predictions_dev` khớp 100 % dev.csv Mac | todo | |
| 7 | Checklist `README.md`, destroy | todo | |

## Lệnh

```bash
# Mac
rsync -avzR -e "ssh -p 45000" src scripts tests pyproject.toml data/processed/splits data/processed/manifest.json \
  data/external/vietjobs37k/ext37k-sal.csv data/external/vietjobs37k/crosswalk_60_to_16.json root@85.10.218.46:/workspace/vietjobs/
# máy thuê
cd /workspace/vietjobs && source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn pyarrow underthesea pyvi pytest
tmux new -d -s r2 "bash scripts/vast_run_r2.sh bamibert 2>&1 | tee logs/r2-all.txt"
```

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
