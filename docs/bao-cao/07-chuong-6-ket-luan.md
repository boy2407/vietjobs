[← Chương 5](06-chuong-5-he-thong.md) · [Bảng điều khiển](00-index.md) · [Tài liệu tham khảo →](08-tai-lieu-tham-khao.md)

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

---

## 6.1. Kết luận

Đề tài đặt ra bốn mục tiêu cụ thể. Bảng dưới đối chiếu từng mục tiêu với trạng thái
thực tế, không làm tròn lên.

**Bảng 6.1: Đối chiếu mục tiêu và kết quả**

| # | Mục tiêu cụ thể | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | Biểu diễn tin tuyển dụng bằng mô hình ngôn ngữ tiếng Việt tiền huấn luyện | **Đạt** | PhoBERT-base-v2 đóng băng → vectơ 768 chiều, lưu đệm theo hai họ cột |
| 2 | Xây dựng mạng học sâu **đa nhiệm** | **Đạt một phần** | Hai mạng riêng đã chạy và có số (§4.3, §4.4) |
| 3 | So sánh với học máy truyền thống, cả độ chính xác **lẫn tốc độ** | **Chưa đạt** | Mạng dense mới được đối chiếu với mốc ngây thơ và mốc dò tuyến tính (§4.2, §4.7) |
| 4 | Triển khai hệ thống hoàn chỉnh cho người dùng | **Chưa đạt** | — |

### 6.1.1. Ba kết quả có giá trị nhất

**Thứ nhất — mạng ước lượng lương đọc được tín hiệu thật từ văn bản.** MAE giảm từ
mốc **5,70** xuống **4,15 triệu** (−27,2 %), và R² trên thang logarit đi từ **−0,059**
lên **0,512** (`dl-sal-s2`, `dev` 2.698 tin). Con số R² mới là điều đáng nói: nó chuyển từ âm sang dương, nghĩa là mô
hình giải thích được biến thiên thật chứ không chỉ đoán quanh trung vị.

**Thứ hai — phép đo cho thấy ngành nghề gần như không nói gì về lương.** eta² = **0,032**.
Biết trước 100 % nhãn ngành nghề chỉ cải thiện MAE **2,3 %** (5,70 → 5,57 triệu).

**Thứ ba — quy trình chống rò rỉ hai tầng chạy được và được kiểm thử canh giữ.**
Che số lương trong văn bản đầu vào, chia dữ liệu theo nhóm tin đăng lại với
`groups_straddling_splits = 0`, tất cả đi qua một cánh cửa duy nhất trong mã nguồn,
được 30 kiểm thử bảo vệ. Đây là loại kết quả không hiện ra trong bảng điểm số nhưng
quyết định việc bảng điểm số đó có đáng tin hay không.

### 6.1.2. Mạng dense chỉ hơn mốc dò tuyến tính một khoảng nhỏ

Mạng phân loại đạt macro-F1 **0,6025** (`dl-cat-s2`), còn hồi quy logistic trên cùng
bộ vectơ (`probe-cat-s2`) đạt **0,5898** — khoảng cách chỉ **0,0127**. Top-3 là 0,9318
so với 0,9258.

Kết quả này được báo cáo nguyên vẹn vì nó mang một thông tin quan trọng: với PhoBERT
**đóng băng**, phần lớn tín hiệu phân loại đã khai thác được bằng một mô hình tuyến
tính; lớp dense chỉ thêm một khoảng nhỏ.

---

## 6.2. Ưu điểm và hạn chế

### 6.2.1. Ưu điểm

1. **Mọi con số đều truy được về nguồn đo.** Không có ước lượng, không có "khoảng".
   Mỗi con số trong báo cáo tìm được trong `manifest.json`, `summary.json`,
   `metrics.json` hoặc một dòng của nhật ký thí nghiệm.
2. **Hệ thống mốc cơ sở hai tầng.** Mốc ngây thơ cho biết sàn, mốc dò tuyến tính phân
   biệt "đặc trưng tồi" với "đầu mô hình hỏng".
3. **Thất bại được ghi lại như bằng chứng.** Lần chạy phân kỳ ở macro-F1 0,042 được
   giữ nguyên trong nhật ký kèm bốn dòng `history.jsonl` chỉ ra nguyên nhân.
4. **Bước tiền xử lý nào không chứng minh được đóng góp thì bị gỡ**, nên đường vào
   PhoBERT chỉ còn các bước ở §3.3.
5. **Giao thức tập kiểm tra chặt chẽ.** `test` chưa từng bị chạm và chưa từng bị biến
   đổi.

### 6.2.2. Hạn chế

**Bảng 6.2: Hạn chế còn tồn tại**

| # | Hạn chế | Mức nghiêm trọng |
|---|---|---|
| 1 | Chưa có phân tích lỗi theo lớp và theo dải lương (§4.5) | Trung bình |
| 2 | Lớp `nhóm_nghề_khác` kéo macro-F1 xuống 0,043 (`f1_macro_no_junk` 0,6454, §4.3) | Trung bình — là vấn đề về nhãn, không phải về mô hình |
| 3 | Các tin lương ở hai biên chưa xử lý: 15 tin > 200 triệu, 132 tin < 2 triệu (Bảng 3.14) | Thấp về số lượng, cao về vị trí — nằm đúng chỗ hại MAE nhất |
| 4 | Danh sách cột được bảo vệ khỏi rò rỉ là **danh sách viết tay** | Thấp hiện tại, cao nếu thêm trường văn bản mới |
| 5 | Mẫu hồi quy đã bị **chọn lọc**: chỉ học trên 71,5 % tin có nhãn, và tỷ lệ này phụ thuộc ngành | Chưa đo được ảnh hưởng |
| 6 | Đánh giá chéo trên VietJobs-37K dùng bảng ánh xạ nhãn còn ở trạng thái nháp (§4.8) | Trung bình |

---

## 6.3. Hướng phát triển

Các việc còn lại, theo thứ tự ưu tiên của [09-lo-trinh.md](../09-lo-trinh.md):

1. **Tinh chỉnh PhoBERT** cho cả hai bài toán, trên GPU, giữ nguyên phép chia
   (`data/processed/splits/` + `manifest.json`).
2. **Hợp nhất đa nhiệm** (mục tiêu 2): một thân chung đọc họ cột `masked`, `loss_B`
   được che cho 28,5 % tin không có nhãn lương, và đo cái giá của việc che lên nhánh
   phân loại.
3. **Hệ thống minh hoạ** (mục tiêu 4): nối đường học sâu vào `predict.py`, tạo tệp
   ánh xạ bài toán → mô hình, đo độ trễ suy luận (mục tiêu 3).
4. **Chất lượng nhãn và dữ liệu:** đo trần nhiễu nhãn, quyết định lớp
   `nhóm_nghề_khác`, xử lý tin lương ở hai biên, đo thiên lệch chọn mẫu, duyệt bảng
   ánh xạ 60 → 16 rồi chấm lại §4.8 bằng mô hình cuối.
5. **Đuôi phải của mạng lương:** thử hàm mất mát bất đối xứng hoặc dự đoán phân vị.

> **Nguồn số liệu:** [09-lo-trinh.md](../09-lo-trinh.md) ·
> [06-baseline-dl.md](../06-baseline-dl.md) ·
> [05-phan-tich-du-lieu.md](../05-phan-tich-du-lieu.md) §7.

---

[← Chương 5](06-chuong-5-he-thong.md) · [Tài liệu tham khảo →](08-tai-lieu-tham-khao.md)
