# T19 — Tuần 9: ba yêu cầu phân tích (Weekly Report, tab "Tuần 9")

**Mục tiêu:** trả lời ba ô B7–B9 của tab Tuần 9.

- [x] B7 — rút 1.000 tin từ ext37k-sal (6.111 tin), chấm `dl-mtl-caf-rnn-b01-cw-s42-v2`, vẽ ma trận nhầm lẫn + biểu đồ lương, so với dev (`scripts/analyze_ext37k_1k.py`)
- [x] B8 — vì sao `nhóm_nghề_khác` bị nhầm nhiều (dev, cùng run)
- [x] B9 — tứ phân vị các tin lương ≥ 100 triệu (file gốc 47.707 dòng, Rule 8)

**Xong khi:** ba mục có số đo và nguồn; kết quả B7 nằm ở `artifacts/<run>/external_37k/sample1k/`.

## Nhật ký
- 2026-10-30: B8, B9 đo xong; B7 đang chạy suy luận CafeBERT trên CPU.
- 2026-10-30: B7 xong; hình `docs/figures/t19-*.png`, số ở `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/external_37k/sample1k/metrics.json`.
- 2026-11-04: B7 đo lại — 1.000 tin rút từ 2.812 tin ext37k có cả lương lẫn `strict_target` (trước: 454/1.000 có nhãn). Phân lớp n=1.000: macro-F1 0,6129 (dev 0,6322); lương MAE 5,57 tr (dev 3,70), R² log 0,274. Vẽ lại 4 hình `t19-cm-*`, `t19-sal-line-*`.
