# Đợt vast dot-24: chấm test phần sạch

Type: task
Status: resolved

## Question

HITL: tác giả thuê vast (GPU 24 GB, ~30 GB đĩa); Claude tạo vast/dot-24 (Rule 14), chạy scripts/score_test_clean.py test cho mô hình chốt, cổng: toàn bộ test khớp 0,6278 / 3,92. Kết quả: metrics_test_clean.json và predictions_test.parquet về Mac.

## Answer

Chấm lại trên RTX 5080 (vast/dot-24-cham-test-sach.md). Cổng đạt: toàn bộ test macro-F1 0,6278 và MAE 3,923 khớp lần chấm cũ.
Phần sạch (8.452 tin, 5.965 có lương): macro-F1 0,6488, MAE 4,109. Phần gần trùng (1.089 tin): macro-F1 0,4511, MAE 2,552.
Kết quả: `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/metrics_test_clean.json`, dự đoán từng tin ở `predictions_test.parquet`.
