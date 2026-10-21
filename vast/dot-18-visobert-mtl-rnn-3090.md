# Đợt 18: ViSoBERT MTL RNN (tách từ Đợt 17) — ĐÃ TẮT

Task: T17 — train `dl-mtl-vis-rnn-b01-cw-s42` song song với Đợt 17, rồi rồi `dl-mtl-xlm-rnn-b01-cw-s42` (chuyển từ Đợt 19), sau đó chấm `test` + `ext37k-sal` cho hai mô hình này.
Script: `/workspace/run_one.sh vis` (trên máy, không vào repo) — cùng lệnh `train_mtl` với `scripts/vast_run_mtl_rnn.sh`, thêm `set -o pipefail` và xóa cache token khi xong.
Instance: 1x RTX 3090 24 GB, AMD Threadripper PRO (32 vCPU), RAM 188 GB, Samsung 980 PRO 100 GB, `38.145.234.214:60880`, ~$0,271/giờ. torch 2.14.1+cu132, transformers 4.46.3.

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Cài môi trường, đẩy splits | done (lần đẩy đầu rơi vào `/root/data` do `cd` trong subshell; đã chuyển) |
| 2 | Train ViSoBERT (bắt đầu 2026-10-21 07:55 UTC, ~2,5 phút/epoch) | doing |
| 2b | Train XLM-R `dl-mtl-xlm-rnn-b01-cw-s42` — từng xếp hàng sau ViSoBERT; 08:50 UTC tác giả quyết định **hoãn sang hôm sau**, đã hủy tmux `queue` | hoãn |
| 3 | Đẩy ext37k (done 08:50), chạy `scripts/eval_mtl_all.py` | todo |
| 4 | Kéo `artifacts/dl-mtl-vis-rnn-b01-cw-s42` + log về Mac, `torch.load` | todo |
| 5 | Destroy | todo |

## Kết thúc 2026-10-21 ~10:20 UTC (tác giả tắt máy)

- ViSoBERT MTL RNN train xong, chấm cùng máy → hợp lệ (dev khớp tuyệt đối). Đã kéo về Mac, kiểm đạt.
- XLM-R MTL RNN (loss cũ) bị tắt ở bước thống kê chuẩn hoá (máy bị tranh CPU, load ~10) — không có kết quả. Làm lại ở Đợt 20.
