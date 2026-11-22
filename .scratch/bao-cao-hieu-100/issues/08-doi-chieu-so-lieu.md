# Đối chiếu số liệu Chương 1–3 với nguồn

Type: task
Status: closed

## Question

AFK: đối chiếu mọi con số trong Chương 1–3 của Doc với summary.json, manifest.json, metrics.json, 04-results.md. Ra danh sách lệch (số Doc, số nguồn, file nguồn).

## Answer

Đối chiếu ngày 2026-11-22 trên bản Doc xuất cùng ngày (Doc chưa đổi từ 21/11). Bảng chi tiết: [Chương 1–2](../assets/08-doi-chieu-chuong-1-2.md) (80 số), [Chương 3 + Phụ lục A](../assets/08-doi-chieu-chuong-3.md) (67 số). Claude đã tự kiểm lại các điểm chính bằng `metrics.json`, `manifest.json`, `encode.py`, `dataset.py`.

**Số đo: không có số nào lệch nguồn trên Doc.** 108/147 khớp. Các số "repo≠Doc" ở Chương 3 là phần repo đã sửa hôm nay (Bảng 3.4/3.8, đoạn tin gần trùng), chờ chép sang Doc.

**Câu nói quá (cần tác giả quyết cách viết lại):**
1. "CafeBERT cho cả hai bài toán": cùng cấu hình MTL RNN `-v2`, PhoBERT có MAE dev thấp hơn (3,653 so với 3,700); CafeBERT chỉ hơn macro-F1 (0,6322 so với 0,6261).
2. "So sánh có hệ thống với bốn bộ mã hóa khác": ở cấu hình chốt mới so đủ với PhoBERT.
3. F1 0,6322 và MAE 3,70 là của hai checkpoint (`best.pt` epoch 21, `best_sal.pt` epoch 30), Doc viết như một mô hình.
4. "test chỉ dùng để báo cáo kết quả cuối": nhiều run đã được chấm test theo ngoại lệ Rule 5 mục 6.
5. "Chia theo nhóm tin trùng lặp gần" và "tuần sau với một câu sửa khác vẫn cùng nhóm": `group_key` chỉ gom tin giống hệt sau khi bỏ dấu (repo đã sửa, Doc chưa).
6. "Hai ràng buộc cùng lúc" khi chia: code chỉ bắt buộc không nhóm nào nằm ở hai tập; tỷ lệ từng lớp lệch xa 72/8/20 (8.507 nhóm mang hơn một nhãn).
7. Bài [19]: RMSE 10,24 / R² 0,23 là khi tinh chỉnh trên hai bộ dữ liệu; chỉ VietJobs là 10,60 / 0,17. Tên tác giả: Phạm Dinh, Nguyễn Huy (không phải "Dinh, Huy").
8. "thiết kế ↔ marketing nhầm cả hai chiều": lỗi lớn nhất của thiết kế là sang xây dựng (71 lần, chiều ngược 4).
9. XLM-R trong code là bản base; CafeBERT dựng trên XLM-R large. "BamiBERT tới 2048 token" không có nguồn.
10. "Đuôi lương cao là thật" viết tuyệt đối; có 15 tin > 200 triệu bị coi là lỗi đơn vị.

**Không có nguồn trong repo:** số liệu trích từ bài [13]–[18], [21] (repo chỉ có PDF [19], [20]); Bảng 2.1 trống trên Doc; 10,65 % và "nhóm lớn nhất 33" chỉ có trong docs, không có json (đo lại đúng).

**Lỗi chỉ có trong repo (Doc đúng hoặc không có):** Chương 1–2 repo còn tả bản chỉ PhoBERT với số cũ (MAE 4,13, macro-F1 0,6030); "gần một phần ba" phải là "hơn một phần tư"; §3.5.2 "198 trong train" phải là 257; §3.1 "780,3 giây" khác `manifest.json` `segment_seconds` 108,9 (cần xác định số nào đúng); Bảng 3.3 repo còn ghi cột đếm/one-hot dù mô hình chỉ đọc 3 trường văn bản; "lớp có 196 dòng" không có nguồn.

**Trình bày:** đánh số mục Doc hỏng (Chương 2 ra 4–13, §3.5 ra 14–17); câu về [20] lặp hai lần.
