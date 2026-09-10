[← Phần đầu](01-phan-dau.md) · [Bảng điều khiển](00-index.md) · [Chương 2 →](03-chuong-2-tong-quan.md)

# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

---

## 1.1. Lý do chọn đề tài

Thị trường tuyển dụng trực tuyến tại Việt Nam vận hành trên một cơ chế đăng tin gần
như tự do. Nhà tuyển dụng tự nhập tiêu đề, tự nhập mô tả công việc, tự chọn ngành
nghề trong một danh sách thả xuống, và tự quyết định có ghi mức lương hay không. Hệ
quả là ba vấn đề xuất hiện đồng thời trên cùng một tin đăng.

**Thứ nhất, nhãn ngành nghề không đáng tin.** Trong bộ dữ liệu của đề tài, lớp
`nhóm_nghề_khác` chứa những tin có tiêu đề "Nhân Viên Seo Web" hay "Nhân Viên Quản
Trị Website" — đây là công việc thuộc marketing và công nghệ thông tin, không phải
một ngành nghề riêng. Người đăng tin chọn "khác" vì nhanh hơn là tìm đúng mục. Đây
là **nhiễu nhãn**, và nó không tự bộc lộ: mô hình học từ dữ liệu này sẽ học luôn cả
thói quen chọn nhầm của người đăng.

**Thứ hai, gần một phần ba tin không công bố mức lương.** Đo trên tập huấn luyện,
tỷ lệ tin có công bố lương là **71,8 %**, tức **28,2 %** số tin để trống. Người tìm
việc vì vậy không có mặt bằng tham chiếu, còn về mặt kỹ thuật thì nhãn của bài toán
hồi quy chỉ tồn tại trên bảy phần mười dữ liệu.

**Thứ ba, việc thiếu nhãn lương không xảy ra ngẫu nhiên.** Tỷ lệ công bố lương trải
từ **56,8 %** đến **76,4 %** tuỳ ngành, và tương quan Pearson giữa tỷ lệ này với
logarit cỡ lớp là **0,670** (p = 0,0045). Nói cách khác, ngành càng ít tin thì càng
ít công bố lương. Những ngành nhỏ chịu phạt hai lần: ít mẫu để học ra lớp, lại càng
ít nhãn hơn nữa cho nhánh hồi quy.

Tự động hoá việc phân loại ngành nghề và ước lượng mức lương từ chính nội dung văn
bản của tin đăng vì vậy có ý nghĩa thực tiễn rõ ràng: nó trả lại cho người tìm việc
một điểm tham chiếu mà thị trường không tự cung cấp. Đồng thời đây là một bài toán
xử lý ngôn ngữ tự nhiên tiếng Việt còn nhiều thách thức — văn bản tin tuyển dụng
lẫn lộn tiếng Việt có dấu và không dấu, đầy từ viết tắt chuyên ngành, và ranh giới
từ trong tiếng Việt không trùng với ranh giới khoảng trắng.

> **Nguồn số liệu:** `artifacts/eda/summary.json` (`disclosure_rate_train` = 0,7176) ·
> [05-phan-tich-du-lieu.md §5](../05-phan-tich-du-lieu.md) · nhận xét về
> `nhóm_nghề_khác` ở [03-protocol.md §4](../03-protocol.md).

---

## 1.2. Mục tiêu đề tài

### 1.2.1. Mục tiêu tổng quát

Xây dựng một hệ thống web tự động phân loại ngành nghề và ước lượng mức lương từ
nội dung một tin tuyển dụng tiếng Việt.

### 1.2.2. Mục tiêu cụ thể

1. Biểu diễn nội dung tin tuyển dụng thành vectơ ngữ nghĩa bằng một mô hình ngôn
   ngữ tiếng Việt đã được tiền huấn luyện.
2. Xây dựng một mạng nơ-ron học sâu đa nhiệm: vectơ biểu diễn đi qua các lớp kết
   nối đầy đủ dùng chung rồi tách thành hai nhánh đầu ra — một nhánh phân loại
   ngành nghề, một nhánh ước lượng mức lương xử lý được cả những tin không công bố
   lương — toàn mạng tối ưu bằng một hàm mất mát hợp nhất.
3. So sánh mô hình học sâu với các mô hình học máy truyền thống dùng làm mốc cơ sở,
   đánh giá cả độ chính xác lẫn tốc độ xử lý để chọn phương án triển khai.
4. Triển khai hệ thống hoàn chỉnh cho phép người dùng nhập tin tuyển dụng và nhận
   kết quả dự đoán.

---

## 1.3. Phát biểu bài toán

Đề tài giải quyết ba bài toán con trên cùng một đầu vào. Tách ra ba bài toán chứ
không gộp thành một, vì ba bài toán này có ba loại nhãn khác nhau và ba mốc cơ sở
khác nhau.

### 1.3.1. Đầu vào

Một tin tuyển dụng tiếng Việt gồm ba trường văn bản tự do:

| Trường | Nội dung | Vai trò |
|---|---|---|
| `job_title` | Tiêu đề tin, ví dụ "Nhân Viên Kinh Doanh Bất Động Sản" | Tín hiệu ngành nghề mạnh nhất |
| `description` | Mô tả công việc | Nguồn tín hiệu chính về cấp bậc, quy mô |
| `requirements_text` | Yêu cầu ứng viên | Chứa kinh nghiệm, bằng cấp, ngoại ngữ |

### 1.3.2. Quá trình xử lý

Ba trường trên được chuẩn hoá theo chín bước xử lý tiếng Việt, nối lại thành một
chuỗi, đưa qua PhoBERT để thu vectơ 768 chiều, rồi qua khối kết nối đầy đủ và tách
nhánh (chi tiết ở Chương 3).

### 1.3.3. Đầu ra

| Bài toán | Ký hiệu trong mã nguồn | Kiểu đầu ra | Độ đo chính |
|---|---|---|---|
| Phân loại ngành nghề | `category` | 1 trong 16 lớp | macro-F1 |
| Tin có công bố lương hay không | `disclosed` | nhị phân 0/1 | accuracy |
| Ước lượng mức lương | `salary` | số thực, triệu VND/tháng | MAE |

### 1.3.4. Ví dụ minh hoạ

Đầu vào (trích nguyên văn dạng dữ liệu thô):

```
Tiêu đề    : "Nhân Viên Kinh Doanh Bất Động Sản"
Mô tả      : "Tìm kiếm khách hàng tiềm năng, tư vấn sản phẩm căn hộ..."
Yêu cầu    : "Tốt nghiệp Cao đẳng trở lên, có 2 năm kinh nghiệm..."
```

Đầu ra mong đợi:

```
Ngành nghề     : kinh_doanh_bán_hàng_chăm_sóc_khách_hàng
Có công bố lương: có
Mức lương      : ~15 triệu VND/tháng
```

Ba chi tiết trong ví dụ này giải thích vì sao bài toán không tầm thường:

- Cụm "Bất Động Sản" kéo tin về phía lớp `xây_dựng_kiến_trúc_bất_động_sản`, trong
  khi nhãn thật là `kinh_doanh…`. Đây chính là cặp lớp bị nhầm nhiều thứ ba trong
  thực nghiệm.
- "2 năm kinh nghiệm" là tín hiệu lương mạnh, nhưng chỉ đọc được nếu hệ thống hiểu
  "2 năm" và "24 tháng" là cùng một đại lượng.
- Chữ "Nhân Viên" viết hoa từng chữ là quy ước trình bày của trang tuyển dụng, không
  mang nghĩa. Nếu chuẩn hoá sai, nó trở thành một đặc trưng giả.

---

## 1.4. Đối tượng và phạm vi nghiên cứu

### 1.4.1. Đối tượng nghiên cứu

Nội dung văn bản của tin tuyển dụng tiếng Việt, và mối liên hệ giữa nội dung đó với
ngành nghề cùng mức lương của vị trí tuyển dụng.

### 1.4.2. Phạm vi nghiên cứu

**Về dữ liệu.** Bộ dữ liệu gồm **48.092** tin thô, sau khi loại trùng khớp hoàn toàn
còn **47.707** tin, thuộc **16 nhóm ngành nghề**, trong đó **71,8 %** tin có công bố
mức lương. Đơn vị lương là **triệu VND/tháng**.

> ✍️ **CẦN VIẾT TAY** — mục phạm vi còn thiếu **nguồn thu thập dữ liệu** và **mốc
> thời gian thu thập**. Hai thông tin này không suy ra được từ mã nguồn; chúng phải
> do tác giả ghi lại. Ô "Thời gian thực hiện" trong đề cương cũng còn để trống.

**Về mô hình.** Đề tài dùng PhoBERT ở trạng thái **đóng băng trọng số** làm mức cơ
sở. Việc tinh chỉnh (fine-tune) toàn bộ encoder nằm ngoài phạm vi thực nghiệm trên
máy hiện tại vì một ràng buộc phần cứng đã đo được: máy chạy kiến trúc Intel
x86_64, **không có MPS hay CUDA**, và thư viện `torch` không còn phát hành bản dựng
cho macOS Intel sau phiên bản 2.2.2.

**Về hệ thống.** Hệ thống hoạt động trên một máy chủ đơn, phục vụ dự đoán cho từng
tin do người dùng nhập vào, không xử lý theo lô lớn và không có cơ chế huấn luyện
lại trực tuyến.

> **Nguồn số liệu:** `data/processed/manifest.json` (`source_rows` = 48.092,
> `rows_after_exact_dedup` = 47.707) · ràng buộc phần cứng ở
> [09-lo-trinh.md, Ưu tiên 2](../09-lo-trinh.md).

---

## 1.5. Ý nghĩa của đề tài

### 1.5.1. Ý nghĩa học thuật

Đề tài đóng góp ba điểm có thể kiểm chứng lại được:

1. **Một quy trình xử lý tiếng Việt có bằng chứng cho từng bước.** Chín bước tiền
   xử lý đều đi kèm một dòng đo tách biệt (ablation). Bước nào không cải thiện được
   con số thì bị gỡ bỏ, không giữ lại vì "sách viết thế". Trong đề tài này đã có hai
   bước đảo ngược kết luận khi trục mô hình chuyển từ TF-IDF sang PhoBERT.
2. **Một phép đo giới hạn thực tế của bài toán lương.** Phân rã phương sai cho thấy
   nhãn ngành nghề chỉ giải thích **3,2 %** biến thiên của log-lương. Đây là con số
   ít được báo cáo trong các công trình cùng dạng, nhưng nó quyết định toàn bộ kỳ
   vọng đặt vào mô hình đa nhiệm.
3. **Một giao thức thực nghiệm chống rò rỉ dữ liệu ở hai tầng** — che số lương
   trong văn bản đầu vào của bài toán lương, và chia dữ liệu theo nhóm tin trùng
   lặp gần thay vì theo dòng.

### 1.5.2. Ý nghĩa thực tiễn

Với người tìm việc, hệ thống trả lại một điểm tham chiếu về mặt bằng lương cho
những tin không công bố. Với trang tuyển dụng, hệ thống gợi ý ngành nghề ngay lúc
nhà tuyển dụng đang soạn tin, giúp giảm tỷ lệ gán nhãn sai ngay từ nguồn — điều mà
mọi khâu xử lý phía sau đều được hưởng lợi.

Độ chính xác top-3 đo được (**0,9257**) đáng chú ý hơn độ chính xác top-1 đối với
ứng dụng thực tế: một giao diện gợi ý ba ngành nghề để người đăng tin chọn thì đúng
hơn 92 lần trong 100.

---

## 1.6. Cấu trúc của báo cáo

Báo cáo gồm sáu chương:

- **Chương 1 – Giới thiệu đề tài.** Trình bày lý do chọn đề tài, mục tiêu nghiên
  cứu, phát biểu ba bài toán con kèm đầu vào/đầu ra và ví dụ minh hoạ, đối tượng và
  phạm vi nghiên cứu, ý nghĩa học thuật và thực tiễn.
- **Chương 2 – Tổng quan nghiên cứu.** Khảo sát cơ sở lý thuyết về biểu diễn văn
  bản, mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt và học đa nhiệm; khảo sát các
  công trình liên quan trong và ngoài nước; chỉ ra khoảng trống nghiên cứu và đề
  xuất hướng tiếp cận của đề tài.
- **Chương 3 – Phương pháp thực hiện.** Trình bày toàn bộ quy trình: kiểm toán và
  làm sạch dữ liệu, chín bước xử lý tiếng Việt, cơ chế chống rò rỉ dữ liệu, phân
  chia tập dữ liệu theo nhóm, phân tích khám phá dữ liệu, kiến trúc mạng học sâu và
  các độ đo đánh giá.
- **Chương 4 – Thực nghiệm và đánh giá.** Trình bày thiết lập thực nghiệm, hệ thống
  mốc cơ sở ba tầng, kết quả của từng nhánh, so sánh với học máy truyền thống, phân
  tích lỗi và bài học rút ra từ những lần chạy thất bại.
- **Chương 5 – Cài đặt hệ thống minh hoạ.** Trình bày kiến trúc phần mềm, mã nguồn,
  giao diện người dùng và kết quả kiểm thử hệ thống.
- **Chương 6 – Kết luận và hướng phát triển.** Tổng kết kết quả đạt được, nêu ưu
  điểm và hạn chế còn tồn tại, đề xuất hướng phát triển tiếp theo.

---

[← Phần đầu](01-phan-dau.md) · [Chương 2 →](03-chuong-2-tong-quan.md)
