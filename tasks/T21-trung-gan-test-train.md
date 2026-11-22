# T21 — Trùng gần giữa test và train mà `group_key` bỏ sót

**Mục tiêu:** đo, ghi lại và (nếu tác giả quyết) xử lý các tin test gần như sao chép một
tin train. Phát hiện trong buổi hỏi vặn báo cáo ngày 2026-11-22.

## Tóm tắt cho tác giả (đọc phần này là đủ)

- Mô hình **học theo từng tin riêng lẻ**. `group_key` không dùng lúc huấn luyện; nó chỉ
  quyết định mỗi tin rơi vào tập nào (train, dev hay test) khi chia dữ liệu.
- `group_key` chỉ gom các tin giống hệt nhau sau khi bỏ dấu, dấu câu và chữ hoa
  ([vitext.py:540](../src/vietjobs/vitext.py)). Khác một từ là khác nhóm.
- Đo được: 11,4 % tin test (1.089 / 9.541) có bản giống ≥ 90 % trong train.
  - Nhãn ngành trùng chỉ 30,6 %, nên macro-F1 test **không** bị thổi phồng. Đây còn là
    bằng chứng nhãn nhiễu: cùng nội dung, người đăng chọn ngành khác.
  - Lương trùng 82,8 % (773 cặp có lương), nên MAE test **có thể** thấp hơn thực tế.
- Mô hình chốt `dl-mtl-caf-rnn-b01-cw-s42-v2` dùng bộ chia mặc định `data/processed/splits`
  (dev n = 3.812 trong `metrics.json`), tức là chịu ảnh hưởng này.
- Repo đã có các bộ dữ liệu lọc trùng/lọc nhãn mâu thuẫn từ trước
  (`splits-dedup-norm-1label-mh`, `splits-minhash-1label`, T14). Chưa rõ kết quả của chúng
  đã được ghi vào `04-results.md` chưa.

## Checklist

- [x] Script chỉ đọc `scripts/near_dup_test_train.py` → `artifacts/eda/near_dup_test.json`
- [x] Ghi vào bao-cao §3.2 (lỗ hổng `group_key` + số đo)
- [ ] Ghi bằng chứng nhãn nhiễu vào bao-cao §3.5.1 và docs/01-data-audit.md
- [ ] Ghi hạn chế MAE test vào §6.2
- [x] Tác giả quyết (b): báo thêm điểm test trên phần sạch — đã ghi bảng vào bao-cao §3.2
- [x] Đo trên dev từ dự đoán có sẵn (`scripts/score_test_clean.py dev` → `metrics_dev_clean.json`):
      MAE toàn bộ 3,700 (khớp Dashboard_last4), phần sạch 3,774, phần gần trùng 3,141;
      macro-F1 toàn bộ 0,6322, phần sạch 0,6506, phần gần trùng 0,4596
- [x] Chấm test phần sạch trên vast dot-24: MAE toàn bộ 3,923 (khớp), sạch 4,109, gần trùng 2,552; macro-F1 toàn bộ 0,6278, sạch 0,6488, gần trùng 0,4511
- [x] Đo thêm phần test không gần trùng với **train + dev** (`clean_train_dev` trong
      `metrics_test_clean.json`; 1.312 tin bị loại, `vs_train_dev` trong `near_dup_test.json`):
      macro-F1 0,6511 (8.229 tin), MAE 4,116 (5.796 tin có lương)
- [ ] Không bỏ tin khỏi `test` (Rule 5). Bỏ tin khỏi `train` là một thí nghiệm mới, cần train lại

## Tiêu chí xong

Báo cáo nói rõ lỗ hổng, có số đo và nguồn, và tác giả trả lời được câu hỏi của hội đồng
"test có trùng train không".

## Nhật ký

| Ngày | Ai | Việc |
|---|---|---|
| 2026-11-22 | Claude | Đo trùng gần, viết script và json, sửa §3.2 và Bảng 3.8 (số tin có lương theo manifest) |
| 2026-11-22 | Claude | Đo trùng gần trên dev và chấm phần sạch từ `predictions_dev.parquet` |
| 2026-11-22 | Claude | Chấm test phần sạch trên vast dot-24, cổng đạt |
| 2026-11-22 | Claude | Thêm phép so với train + dev vào hai script, chạy lại từ dự đoán đã lưu (không chạy mô hình) |
| 2026-11-22 | Claude | Thêm bảng điểm phần sạch (train; train + dev) vào bao-cao §3.2 |
