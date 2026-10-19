# Đợt 11: BamiBERT 1024, RNN category với last4 + cw effective 0,999 + ls 0,05 (seed 42) và cache pooled last4 — ĐANG CHẠY

Task: chưa có ID. Nằm trong kế hoạch "last4 + cw + ls cho 5 encoder × Dense và RNN" của bài `category`; máy này làm BamiBERT, XLM-R + ViSoBERT sẽ làm trên máy khác khi rảnh.
Script: `scripts/vast_run_last4_rnn.sh bamibert 1024 32` | Instance: RTX 5060 Ti 16 GB, 64 nhân, RAM 503 GB, disk 120 GB, `ssh -p 45000 root@85.10.218.46`, thư mục `/workspace/vietjobs`, tmux `last4`, log `logs/last4-encode-bamibert.txt`.

Máy này là máy của Đợt 9 (`dot-9`) sau khi xong: cả 5 `best.pt` BamiBERT `-r2` đã nằm trên Mac và md5 khớp máy (kiểm 2026-10-19); `dl-mtl-bamibert-rnn1024-a5-s42-r2` chạy ở máy khác (`46062`).

Phạm vi: nhúng pooled last4 (`train`, `dev`) để train Dense trên Mac; nhúng token last4 rồi train **một** run RNN (h=100, c=50, seed 42, `cuda`); xoá cache token sau run. Không đụng `test`.

Bảng run:

| Run | Trạng thái | macro-F1 | Mốc (BamiBERT RNN CE, `dl-cat-bamibert-rnn-1024`) |
|---|---|---|---|
| `dl-cat-bamibert-rnn-last4-cweff999-ls05-s42` | doing | | 0,6056 |

Ước tính (từ tốc độ đo ở Đợt 9: nhúng ≈ 82 tin/s; RNN BamiBERT CE cũ 99,9 phút): nhúng pooled ≈ 8 phút + token ≈ 8 phút + train ≈ 1,5–2 giờ. Giá máy tác giả cho: $0,25/giờ.

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Đẩy `src scripts tests` mới (có `--layers` cho token) | doing | rsync |
| 2 | `bash scripts/vast_run_last4_rnn.sh bamibert 1024 32` trong tmux | todo | |
| 3 | Kéo 4 tệp `bamibert-{train,dev}-raw-len1024-last4.{npy,json}` + `artifacts/dl-cat-bamibert-rnn-last4-cweff999-ls05-s42/` (kèm `best.pt`) + log | todo | `scp`, so md5 |
| 4 | `torch.load` `best.pt` trên Mac | todo | |
| 5 | Mac: Dense last4 + cw + ls và CE lớp cuối, 5 seed | todo | |
| 6 | Checklist `README.md`, destroy | todo | tác giả bấm trên web |

## Lệnh

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p 45000" src scripts tests pyproject.toml root@85.10.218.46:/workspace/vietjobs/
ssh -p 45000 root@85.10.218.46 "cd /workspace/vietjobs && tmux new -d -s last4 'source /venv/main/bin/activate && bash scripts/vast_run_last4_rnn.sh bamibert 1024 32 2>&1 | tee logs/last4-all.txt'"
```

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
