# Đợt 13: CafeBERT 512 rồi PhoBERT 256, RNN last4 + cw + ls (seed 42) — ĐANG CHẠY

Task: chưa có ID. Nằm trong kế hoạch last4 + cw effective 0,999 + ls 0,05 cho 5 encoder × Dense và RNN của bài `category` (máy A đã làm BamiBERT ở [dot-11](dot-11-last4-rnn-bamibert.md)).
Script: `scripts/vast_run_last4_rnn.sh <encoder> <max_len> <batch>` | Instance: RTX 5060 Ti 16 GB, disk 120 GB, `ssh -p 43372 root@158.181.52.18 (máy C, tmux `last4c`, log `logs/last4c-all.txt`)`, thư mục `/workspace/vietjobs`, giá $0,25/giờ.

Máy này trước là máy C của phiên T16 ([dot-8](dot-8-r2-xlmr-visobert.md)), tác giả xác nhận dùng được; 9 run `-r2` XLM-R + ViSoBERT đã ở Mac. Script không xoá `artifacts/*-r2` của phiên đó.
Phạm vi: nhúng token last4 (và pooled last4 nếu không đặt `SKIP_POOLED=1`), train **một** run RNN (h=100, c=50, seed 42, `cuda`) cho mỗi encoder, xoá cache token sau run. Không đụng `test`.

| Run | Trạng thái | macro-F1 | Mốc CE lớp cuối |
|---|---|---|---|
| `dl-cat-cafebert-rnn-last4-cweff999-ls05-s42` | doing | | 0,6152 (3 seed `dl-cat-cafe-rnn-ce-s42..44`) |
| `dl-cat-phobert-rnn-last4-cweff999-ls05-s42` | doing | | 0,6084 (4 seed `dl-cat-rnn-ce*`) |

Ước tính: CafeBERT nhúng token ≈ 30 tin/s (≈ 21 phút), RNN cũ 27–49 phút; PhoBERT nhúng + RNN cũ 16,5 phút → khoảng 1,5–2 giờ, ≈ $0,4–0,5. `SKIP_POOLED=1` vì cache pooled last4 của hai encoder này đã có trên Mac.

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
