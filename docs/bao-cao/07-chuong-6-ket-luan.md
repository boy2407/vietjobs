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
| 2 | Xây dựng mạng học sâu **đa nhiệm** | **Đạt một phần** | Hai nhánh đã chạy **riêng biệt** và có số. Bản hợp nhất đa nhiệm **chưa chạy** |
| 3 | So sánh với học máy truyền thống, cả độ chính xác **lẫn tốc độ** | **Đạt một phần** | Có bảng 11 thuật toán và bảng đánh đổi chi phí huấn luyện. **Thiếu bảng độ trễ suy luận** |
| 4 | Triển khai hệ thống hoàn chỉnh cho người dùng | **Chưa đạt** | Hệ thống web chưa xây; `predict.py` chưa nối đường học sâu |

### 6.1.1. Ba kết quả có giá trị nhất

**Thứ nhất — nhánh ước lượng lương đọc được tín hiệu thật từ văn bản.** MAE giảm từ
mốc **5,86** xuống **4,83 triệu** (−17,6 %), và R² trên thang logarit đi từ **−0,063**
lên **0,381**. Con số R² mới là điều đáng nói: nó chuyển từ âm sang dương, nghĩa là mô
hình giải thích được biến thiên thật chứ không chỉ đoán quanh trung vị.

**Thứ hai — phép đo cho thấy ngành nghề gần như không nói gì về lương.** eta² = **0,032**.
Biết trước 100 % nhãn ngành nghề chỉ cải thiện MAE **1,8 %**. Đây là kết quả định
hình toàn bộ lộ trình: nó hạ kỳ vọng đặt vào mô hình đa nhiệm xuống mức thực tế
**trước khi** tốn công xây nó, và nó nói rằng giá trị của bản đa nhiệm nhiều khả năng
nằm ở chỗ "một mô hình thay vì hai", không nằm ở điểm số.

**Thứ ba — quy trình chống rò rỉ hai tầng chạy được và được kiểm thử canh giữ.**
Che số lương trong văn bản đầu vào, chia dữ liệu theo nhóm tin đăng lại với
`groups_straddling_splits = 0`, tất cả đi qua một cánh cửa duy nhất trong mã nguồn,
được 28 kiểm thử bảo vệ. Đây là loại kết quả không hiện ra trong bảng điểm số nhưng
quyết định việc bảng điểm số đó có đáng tin hay không.

### 6.1.2. Một kết quả âm, và vì sao nó được giữ lại

Nhánh phân loại đạt macro-F1 **0,5987**, còn mốc học máy TF-IDF + LinearSVC là
**0,6050 ± 0,0071**. Khoảng cách 0,0063 nhỏ hơn một nửa khoảng tin cậy của chính mốc
đó, nên kết luận đúng là **hai mô hình chưa phân biệt được**, không phải "học sâu
thua".

Kết quả này được báo cáo nguyên vẹn thay vì bị giấu đi, vì nó mang một thông tin quan
trọng: **PhoBERT ở trạng thái đóng băng đã ngang mốc TF-IDF mà chưa dùng đến khả năng
thích nghi của encoder.** Đó chính là lý do việc tinh chỉnh encoder là hạng mục có kỳ
vọng cao nhất còn lại.

---

## 6.2. Ưu điểm và hạn chế

### 6.2.1. Ưu điểm

1. **Mọi con số đều truy được về nguồn đo.** Không có ước lượng, không có "khoảng".
   Mỗi con số trong báo cáo tìm được trong `manifest.json`, `summary.json`,
   `metrics.json` hoặc một dòng của nhật ký thí nghiệm.
2. **Hệ thống mốc cơ sở ba tầng.** Mốc ngây thơ cho biết sàn, mốc dò tuyến tính phân
   biệt "đặc trưng tồi" với "đầu mô hình hỏng", mốc học máy cho biết cách làm cũ đạt
   bao nhiêu.
3. **Thất bại được ghi lại như bằng chứng.** Lần chạy phân kỳ ở macro-F1 0,042 được
   giữ nguyên trong nhật ký kèm bốn dòng `history.jsonl` chỉ ra nguyên nhân.
4. **Bước tiền xử lý nào không chứng minh được đóng góp thì bị gỡ.** Hai bước đã bị
   gỡ khỏi quy trình chín bước.
5. **Giao thức tập kiểm tra chặt chẽ.** `test` chưa từng bị chạm và chưa từng bị biến
   đổi.

### 6.2.2. Hạn chế

**Bảng 6.2: Hạn chế còn tồn tại**

| # | Hạn chế | Mức nghiêm trọng |
|---|---|---|
| 1 | **Toàn bộ kết quả đo trên lược đồ chia v1**, phải chạy lại trên lược đồ hiện tại | Cao — chặn mọi kết luận cuối cùng |
| 2 | Hệ thống web chưa xây; `predict.py` chưa nối đường học sâu | Cao — là mục tiêu cụ thể số 4 |
| 3 | Chưa có bảng độ trễ suy luận | Trung bình — là một phần mục tiêu số 3 |
| 4 | Mô hình đa nhiệm chưa chạy | Trung bình — là mục tiêu số 2 |
| 5 | Nhánh hồi quy **không dám ra đuôi phải**: MAE 25,58 ở dải > 30 triệu so với 3,70 ở dải dưới | Trung bình |
| 6 | Lớp `nhóm_nghề_khác` có F1 = 0,000 | Trung bình — là quyết định về nhãn, không phải về mô hình |
| 7 | **Trần nhiễu nhãn chưa được đo** — không biết mô hình còn cách trần bao xa | Trung bình |
| 8 | Các tin lương ở hai biên chưa xử lý (15 tin > 200 triệu; nhóm < 2 triệu ⛔ chưa đo lại trên tệp gốc) | Thấp về số lượng, cao về vị trí — nằm đúng chỗ hại MAE nhất |
| 9 | Danh sách cột được bảo vệ khỏi rò rỉ vẫn là **danh sách viết tay** | Thấp hiện tại, cao nếu thêm trường văn bản mới |
| 10 | Thiếu kiểm thử cho `predict.py`, `dataset.clean`, `group_stratified_split` | Trung bình |
| 11 | Mẫu hồi quy đã bị **chọn lọc**: chỉ học trên 71,5 % tin có nhãn, và tỷ lệ này phụ thuộc ngành | Chưa đo được ảnh hưởng |

---

## 6.3. Hướng phát triển

Sắp theo **giá trị kỳ vọng trên mỗi giờ công**, không theo thứ tự dễ làm.

### 6.3.1. Ưu tiên 1 — chạy lại toàn bộ trên lược đồ chia hiện tại

Không có việc nào khác có ý nghĩa trước khi việc này xong, vì mọi kết luận hiện tại
đều treo trên số liệu của lược đồ cũ. Bao gồm: chạy lại `scripts/analyze_data.py`,
nhúng lại vectơ PhoBERT, chạy lại hai nhánh, chạy lại mốc học máy.

### 6.3.2. Ưu tiên 2 — tinh chỉnh PhoBERT

Đây là hạng mục còn lại có kỳ vọng cao nhất: bản đóng băng đã ngang mốc TF-IDF **mà
chưa dùng đến khả năng thích nghi của encoder**. Ràng buộc đã đo được: máy hiện tại
không có GPU/MPS, nên việc này cần một GPU trên Colab hoặc Kaggle, và phải chép kèm
`data/processed/splits/` cùng `manifest.json` sang đó để giữ nguyên phép chia.

### 6.3.3. Ưu tiên 3 — hoàn thiện hệ thống

Ba việc ở Bảng 5.5: nối đường học sâu vào `predict.py`, tạo tệp ánh xạ mô hình sản
phẩm, nạp đúng bộ chuẩn hoá. Kèm theo là bảng đo độ trễ suy luận cho mục tiêu số 3.

### 6.3.4. Ưu tiên 4 — hợp nhất đa nhiệm

Hai ràng buộc không được vi phạm:

1. **`loss_B` phải được che.** 28,5 % tin không có nhãn lương; với chúng `loss_B` bằng
   0, chứ không phải nhãn bằng 0.
2. **Nhánh lương vẫn phải đọc cột đã che số lương.** Khi hai nhánh dùng chung một
   thân, cái thân đó buộc phải đọc bản đã che — nghĩa là bản đa nhiệm chạy trên họ
   cột `masked`, và **cái giá của việc đó** (nhánh phân loại mất phần văn bản chứa
   con số lương) là một con số phải đo, không phải một giả định.

### 6.3.5. Ưu tiên 5 — chất lượng nhãn và dữ liệu

- Đo trần nhiễu nhãn: lấy mẫu 100 tin, gán nhãn tay, đo độ đồng thuận.
- Quyết định số phận lớp `nhóm_nghề_khác`: gộp hay bỏ, không phải cố học nó.
- Xử lý các tin lương ở hai biên (15 tin > 200 triệu, cộng nhóm < 2 triệu).
- Đo thiên lệch chọn mẫu của nhánh hồi quy.

### 6.3.6. Ưu tiên 6 — cải thiện đuôi phải của nhánh lương

Thử hàm mất mát bất đối xứng, hoặc chuyển sang dự đoán phân vị thay vì một điểm, để
mô hình dám đi ra dải lương cao.

> **Nguồn số liệu:** [09-lo-trinh.md](../09-lo-trinh.md) ·
> [06-baseline-dl.md](../06-baseline-dl.md) ·
> [05-phan-tich-du-lieu.md](../05-phan-tich-du-lieu.md) §7.

---

[← Chương 5](06-chuong-5-he-thong.md) · [Tài liệu tham khảo →](08-tai-lieu-tham-khao.md)
