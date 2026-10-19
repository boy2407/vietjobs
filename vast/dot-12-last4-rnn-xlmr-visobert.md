# Đợt 12: XLM-R 512, RNN last4 + cw + ls (seed 42) và cache pooled last4 — ĐANG CHẠY (ViSoBERT chuyển sang [dot-14](dot-14-last4-rnn-visobert.md))

Task: chưa có ID. Nằm trong kế hoạch last4 + cw effective 0,999 + ls 0,05 cho 5 encoder × Dense và RNN của bài `category` (máy A đã làm BamiBERT ở [dot-11](dot-11-last4-rnn-bamibert.md)).
Script: `scripts/vast_run_last4_rnn.sh <encoder> <max_len> <batch>` | Instance: RTX 5060 Ti 16 GB, disk 120 GB, `ssh -p 45000 root@85.10.218.46 (máy A, tmux `last4b`, log `logs/last4b-all.txt`)`, thư mục `/workspace/vietjobs`, giá $0,25/giờ.

Máy này là máy A của dot-11 sau khi xong (kết quả BamiBERT đã ở Mac, md5 khớp). Các máy B/C-T16 và phiên T16 không liên quan.
Phạm vi: nhúng token last4 (và pooled last4 nếu không đặt `SKIP_POOLED=1`), train **một** run RNN (h=100, c=50, seed 42, `cuda`) cho mỗi encoder, xoá cache token sau run. Không đụng `test`.

| Run | Trạng thái | macro-F1 | Mốc CE lớp cuối |
|---|---|---|---|
| `dl-cat-xlmr-rnn-last4-cweff999-ls05-s42` | doing | | 0,6134 (`dl-cat-xlmr-rnn-512`) |
| `dl-cat-visobert-rnn-last4-cweff999-ls05-s42` | chuyển sang máy B ([dot-14](dot-14-last4-rnn-visobert.md)) | | 0,6046 (`dl-cat-visobert-rnn-512`) |

Ước tính: nhúng XLM-R ≈ 104 tin/s, RNN XLM-R cũ 29 phút; ViSoBERT nhúng ≈ 90 tin/s, RNN cũ 75 phút → tổng khoảng 2–2,5 giờ, ≈ $0,5–0,6.

## Tiến độ

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Đẩy `src scripts tests` có `--layers` cho token | done |
| 2 | Chạy script trong tmux, log `logs/last4*-all.txt` | doing |
| 3 | Kéo cache pooled last4 (nếu có) + thư mục run (kèm `best.pt`) + log về Mac, so md5, `torch.load` | todo |
| 4 | Mac: Dense last4 + cw + ls và CE lớp cuối, 5 seed; ghi `04-results.md`, `export_runs.py` | todo |
| 5 | Checklist `README.md`, destroy | todo (tác giả bấm trên web) |

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| 2026-10-19 | `rsync` lần đầu dùng sai user (`admin@`) nên code mới chưa lên máy; trên máy C script chưa có nên tmux chết ngay | chạy lại `rsync -l root`; máy A vẫn dùng bản script đã có từ dot-11 (đủ cho XLM-R/ViSoBERT vì cần pooled) |
| 2026-10-19 | Tác giả cho dùng máy B | ViSoBERT chuyển sang [dot-14](dot-14-last4-rnn-visobert.md); máy A chỉ chạy XLM-R (cờ `logs/SKIP_visobert` trên máy này) |
