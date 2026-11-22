# Đợt 24 — chấm test phần sạch của mô hình chốt

**Trạng thái: XONG, ĐÃ DESTROY (2026-11-22)** · Task: T21, ticket wayfinder 06 (`.scratch/bao-cao-hieu-100/`) · Script: `scripts/score_test_clean.py test`

## Mục đích

Chấm lại `dl-mtl-caf-rnn-b01-cw-s42-v2` trên `test` từ `best.pt`/`best_sal.pt` có sẵn, **lưu dự đoán từng tin**, rồi tách điểm phần sạch (bỏ tin gần trùng Jaccard ≥ 0,9 với train). Không train, không chọn mô hình, không sửa `test.csv` (Rule 5).

**Cổng:** điểm trên toàn bộ test phải khớp `metrics_test.json` cũ (macro-F1 0,6278, MAE 3,9234). Lệch thì không dùng số mới.

## Máy

| GPU | Đĩa | RAM | CPU | SSH |
|---|---|---|---|---|
| RTX 5080 16 GB | 46 GB `/` | 251 GB | 192 luồng | `ssh -p 40705 root@95.169.14.15` |

Torch có sẵn 2.14.1+cu132 (khác bản 4090 của đợt 20 → so số kỹ ở cổng).

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Kiểm máy (GPU, đĩa, torch) | xong |
| 2 | Cài transformers, sentencepiece, pandas, pyarrow, scikit-learn | xong |
| 3 | Đẩy mã, splits train/test, manifest, run dir, `near_dup_test_best.npy` | xong |
| 4 | Chạy `score_test_clean.py test` | xong, ~3,5 phút cho 9.541 tin |
| 5 | Kiểm cổng | đạt: macro-F1 0,627821 khớp, MAE 3,923432 so với 3,923431 |
| 6 | Kéo `predictions_test.parquet`, `metrics_test_clean.json`, log về Mac, so md5 | xong, md5 khớp |
| 7 | Checklist `vast/README.md` → báo tác giả destroy | xong, tác giả đã destroy |

## Kết quả

| Phần | n phân loại | macro-F1 | n lương | MAE (tr) |
|---|---|---|---|---|
| toàn bộ | 9.541 | 0,6278 | 6.773 | 3,923 |
| sạch | 8.452 | 0,6488 | 5.965 | 4,109 |
| gần trùng | 1.089 | 0,4511 | 808 | 2,552 |

Nguồn: `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/metrics_test_clean.json`.

## Lệnh

```bash
# đẩy lên
rsync -az -e "ssh -p 40705" src scripts requirements-dl.txt root@95.169.14.15:/root/vj/
# chạy
ssh -p 40705 root@95.169.14.15 'cd /root/vj && PYTHONPATH=src /venv/main/bin/python scripts/score_test_clean.py test'
# kéo về
scp -P 40705 root@95.169.14.15:/root/vj/artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/{predictions_test.parquet,metrics_test_clean.json} artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/
```
