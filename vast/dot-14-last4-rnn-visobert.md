# Đợt 14: ViSoBERT 512, RNN last4 + cw + ls (seed 42) và cache pooled last4 — ĐANG CHẠY

Task: chưa có ID. Nằm trong kế hoạch last4 + cw effective 0,999 + ls 0,05 cho 5 encoder × Dense và RNN của bài `category`. ViSoBERT được chuyển từ máy A ([dot-12](dot-12-last4-rnn-xlmr-visobert.md)) sang máy B để chạy song song với XLM-R (tác giả cho phép dùng máy B, 2026-10-19).
Script: `scripts/vast_run_last4_rnn.sh visobert 512 32` | Instance: RTX 5060 Ti 16 GB, 64 nhân, RAM 503 GB, disk 120 GB, `ssh -p 46062 root@85.10.218.46`, thư mục `/workspace/vietjobs`, tmux `last4d`, log `logs/last4d-all.txt`, giá $0,25/giờ.

Máy này trước là máy của phiên T16: 3 run `-r2` còn trên đĩa (`dl-mtl-bamibert-rnn1024-a5-s42-r2`, `dl-mtl-visobert-rnn512-a5-s42-r2`, `dl-sal-visobert-rnn-512-r2`) và cả ba đã có trên Mac; script không xoá `artifacts/*-r2`.

Phạm vi: nhúng pooled last4 (`train`, `dev`) để train Dense trên Mac; nhúng token last4 rồi train **một** run RNN (h=100, c=50, seed 42, `cuda`); xoá cache token sau run. Không đụng `test`.

| Run | Trạng thái | macro-F1 | Mốc CE lớp cuối |
|---|---|---|---|
| `dl-cat-visobert-rnn-last4-cweff999-ls05-s42` | doing | | 0,6046 (`dl-cat-visobert-rnn-512`) |

Ước tính: nhúng ≈ 90–100 tin/s (2 lượt ≈ 15 phút), RNN ViSoBERT CE cũ 75 phút → khoảng 1,5 giờ, ≈ $0,4.

## Tiến độ

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Đẩy `src scripts tests` có `--layers` cho token; kiểm sha256 splits | done |
| 2 | `bash scripts/vast_run_last4_rnn.sh visobert 512 32` trong tmux `last4d` | doing |
| 3 | Kéo `visobert-{train,dev}-raw-len512-last4.{npy,json}` + thư mục run (kèm `best.pt`) + log về Mac, so md5, `torch.load` | todo |
| 4 | Mac: Dense last4 + cw + ls và CE lớp cuối, 5 seed; ghi `04-results.md`, `export_runs.py` | todo |
| 5 | Checklist `README.md`, destroy | todo (tác giả bấm trên web) |

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| 2026-10-19 | Chuỗi trên máy A đã lập sẵn `XLM-R; ViSoBERT` nên sẽ chạy trùng ViSoBERT | thêm cờ `logs/SKIP_<encoder>` vào script, `rsync` (ghi file mới nên XLM-R đang chạy giữ bản cũ) rồi tạo `logs/SKIP_visobert` trên máy A; máy A thoát ngay khi tới lượt ViSoBERT |
