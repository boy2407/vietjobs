[← Tổng quan](../00-tong-quan.md)

# Nền tảng — dành cho người mới vào ML/DL

Tám note giải thích **khái niệm**, không chứa kết quả thí nghiệm. Chúng không phải
cập nhật sau mỗi lần chạy mô hình — đó là lý do chúng nằm riêng ở đây.

Mọi ví dụ lấy từ chính kho dữ liệu VietJobs, không lấy ví dụ sách giáo khoa.

---

## Đọc theo thứ tự này

| # | Note | Đọc xong hiểu được gì | Ghép với |
|---|---|---|---|
| 1 | [Vì sao phải làm sạch dữ liệu](01-vi-sao-phai-lam-sach.md) | Lý do thật không phải "rác vào rác ra". Là **xẻ chiều** và **rò rỉ** | [01-data-audit](../01-data-audit.md) |
| 2 | [TF-IDF là gì](02-tf-idf-la-gi.md) | Chữ biến thành số thế nào. Vì sao phải chuẩn hoá L2. Có ma trận in đầy đủ và ví dụ tính tay | [05-dac-trung-tfidf](../05-dac-trung-tfidf.md) |
| 3 | [n-gram và ranh giới từ](03-ngram-va-ranh-gioi-tu.md) | Vì sao tiếng Việt làm hỏng giả định "khoảng trắng tách từ" | [02-vietnamese-nlp](../02-vietnamese-nlp.md) |
| 4 | [Ma trận thưa và số chiều](04-ma-tran-thua-va-so-chieu.md) | 236.596 chiều nghĩa là gì. Vì sao KNN sụp mà SVM thì không | [06-mo-hinh-phan-lop](../06-mo-hinh-phan-lop.md) |
| 5 | [Rò rỉ dữ liệu](05-ro-ri-du-lieu.md) | Ba loại rò rỉ, và vì sao **không loại nào báo lỗi** | [03-protocol](../03-protocol.md) |
| 6 | [Đo lường và baseline](06-do-luong-va-baseline.md) | Vì sao accuracy nói dối khi lớp lệch 27:1 | [04-results](../04-results.md) |
| 7 | [Chính quy hoá](07-chinh-quy-hoa.md) | `C` là gì. Vì sao `C` tối ưu = 0,02 là mô hình đang kêu cứu | [07-bai-toan-luong](../07-bai-toan-luong.md) |
| 8 | [Đọc một bảng so sánh mô hình](08-doc-mot-bang-so-sanh.md) | Siêu tham số · "ô" là gì · vì sao cùng số ô vẫn chưa công bằng · lời nguyền của người thắng · σ so với paired bootstrap | [10-so-sanh-mo-hinh](../10-so-sanh-mo-hinh.md) |

---

## Một câu tóm tắt cho bảy note đầu

> Học máy trên văn bản là bài toán **đếm chuỗi ký tự rồi tìm trọng số**.
> Mọi bước tiền xử lý chỉ có một việc: làm cho phép đếm đó đếm đúng thứ ta muốn đếm.

Bốn hệ quả, mỗi hệ quả là một note ở trên:

1. Cùng một khái niệm viết hai cách → bị đếm thành hai thứ → tín hiệu yếu đi (note 1, 3).
2. Đếm rồi phải cân trọng số, vì "xuất hiện nhiều" ≠ "quan trọng" (note 2).
3. Đếm ra rất nhiều chiều thì hình học không gian đổi tính chất (note 4).
4. Nếu vô tình đếm cả đáp án vào đầu vào thì mọi con số đo được đều vô nghĩa (note 5, 6, 7).

---

## Ba điều nên gạt bỏ ngay từ đầu

**"Nhiều đặc trưng hơn thì tốt hơn."** Sai. Trong dự án này, KNN dùng toàn văn
(0,4059) **tệ hơn** KNN chỉ dùng tiêu đề (0,5307), và tệ hơn cả baseline không dùng
học máy (0,4321). Xem [note 4](04-ma-tran-thua-va-so-chieu.md).

**"Tiền xử lý nhiều thì mô hình tốt hơn."** Sai. Chín bước xử lý tiếng Việt trong
dự án này cộng lại đóng góp **+0,0017** macro-F1. Một dòng chỉnh siêu tham số đóng
góp **+0,0287** — gấp gần 17 lần. Xem
[02-vietnamese-nlp §4](../02-vietnamese-nlp.md#4-ablation--bước-nào-thật-sự-đáng-giữ).

**"Điểm cao là mô hình tốt."** Không nhất thiết. Điểm cao thường là dấu hiệu của
rò rỉ trước khi nó là dấu hiệu của mô hình tốt. Xem [note 5](05-ro-ri-du-lieu.md).

Ba điều này không phải ý kiến. Chúng là kết luận **đo được** trong chính dự án
này, và mỗi kết luận có một dòng trong [04-results.md](../04-results.md) đứng sau.
