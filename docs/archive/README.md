# Kho lưu — track học máy (2026-08-26 → 09-07)

Năm note trong thư mục này mô tả **cụm mô hình học máy** đã chạy xong trước khi đề
tài chuyển trục sang học sâu ngày 2026-09-08. Chúng không bị xoá vì một lý do:
mọi con số của mạng học sâu chỉ có nghĩa khi đặt cạnh chúng.

| Note | Chứa gì | Con số phải nhớ |
|---|---|---|
| [04-results-ml.md](04-results-ml.md) | 101 dòng thí nghiệm, append-only | `cat-FINAL-svm-C0.02-test` — **test macro-F1 0,6112** |
| [05-dac-trung-tfidf.md](05-dac-trung-tfidf.md) | TF-IDF 236.596 chiều, 10 khối đặc trưng | mốc so sánh cho biểu diễn PhoBERT 768 chiều |
| [06-mo-hinh-phan-lop.md](06-mo-hinh-phan-lop.md) | Bậc thang phân lớp, vì sao macro-F1 | baseline đa số **0,0210** · từ khoá **0,4321** |
| [09-lo-trinh-ml.md](09-lo-trinh-ml.md) | Lộ trình cũ, 7 ưu tiên | ưu tiên 0 (train/serve skew) vẫn chưa đóng |
| [10-so-sanh-mo-hinh.md](10-so-sanh-mo-hinh.md) | 11 thuật toán × 6 cấu hình, paired bootstrap | LinearSVC thắng; khoảng cách tới hạng nhì hẹp |

Mã nguồn tương ứng ở [`scripts/archive/`](../../scripts/archive/); artifact của
các lần chạy ở `artifacts/archive-ml/` (không nằm trong git).

**Không sửa các file này.** Chúng là bản ghi của một giai đoạn đã đóng.
