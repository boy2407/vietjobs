# Đợt 19: XLM-R MTL RNN trên 3090 — ĐÃ HỦY (không có kết quả)

Task: T17 — định train `dl-mtl-xlm-rnn-b01-cw-s42`.
Instance: 1x RTX 3090 24 GB, Core i5-10500 (12 vCPU), RAM 102 GB, Kingston SNV2S, `178.61.243.26:21372`, ~$0,228/giờ.

- 2026-10-21 07:57 UTC: lần chạy đầu chết với `Bus error` — trên máy này `/workspace` là ổ riêng 10 GB (đầy 100%), còn ổ 100 GB nằm ở `/`; cache token 27 GB không ghi được. Đã symlink `artifacts/embeddings` → `/root/embeddings` và chạy lại 08:03.
- Bước thống kê chuẩn hoá chậm (~16 phút, so với ~5,5 phút trên Đợt 18), nhiều khả năng do CPU/đĩa yếu.
- ~08:25 UTC: tác giả destroy máy, chưa train epoch nào. XLM-R chuyển sang máy khác (đợt sau).
