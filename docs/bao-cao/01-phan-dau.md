[← Bảng điều khiển](00-index.md) · [Chương 1 →](02-chuong-1-gioi-thieu.md)

# Phần đầu

---

## Trang bìa

<div align="center">

ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH
**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN**

**NGUYỄN TRỌNG NGHĨA**

**KHOÁ LUẬN TỐT NGHIỆP**

**XÂY DỰNG HỆ THỐNG PHÂN LOẠI NGÀNH NGHỀ VÀ ƯỚC LƯỢNG MỨC LƯƠNG
TỪ TIN TUYỂN DỤNG TIẾNG VIỆT**

*Building a System for Job Category Classification and Salary Estimation
from Vietnamese Job Postings*

CỬ NHÂN NGÀNH …

TP. HỒ CHÍ MINH, 2026

</div>

> ✍️ **CẦN VIẾT TAY** — điền đúng tên ngành/chuyên ngành và tháng bảo vệ theo mẫu
> bìa chính thức của khoa.

---

## Trang bìa phụ — cán bộ hướng dẫn

<div align="center">

**CÁN BỘ HƯỚNG DẪN**

**TS. LƯU THANH SƠN**

</div>

---

## Lời cảm ơn

> ✍️ **CẦN VIẾT TAY** — đây là phần duy nhất trong quyển báo cáo mang giọng cá
> nhân, không nên để người khác viết hộ. Gợi ý mạch viết, mỗi ý một đoạn ngắn:
>
> 1. Cảm ơn TS. Lưu Thanh Sơn — nói cụ thể một điều thầy đã chỉnh, chứ không nói
>    chung chung. Ví dụ: yêu cầu báo cáo tiến độ hằng tuần đã buộc mỗi quyết định
>    kỹ thuật phải có số đo đi kèm.
> 2. Cảm ơn quý thầy cô Trường Đại học Công nghệ Thông tin.
> 3. Cảm ơn gia đình và bạn bè.
> 4. Một câu nhận trách nhiệm về những thiếu sót còn lại.

---

## Mục lục

> Chèn mục lục tự động trong Word (References → Table of Contents), lấy tới cấp
> Heading 3. Không gõ tay — mục lục gõ tay sẽ sai số trang ngay lần sửa đầu tiên.

Cấu trúc:

```
TÓM TẮT NỘI DUNG KHOÁ LUẬN
CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI
CHƯƠNG 2. TỔNG QUAN NGHIÊN CỨU
CHƯƠNG 3. PHƯƠNG PHÁP THỰC HIỆN
CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ
CHƯƠNG 5. CÀI ĐẶT MÃ NGUỒN THỰC NGHIỆM
CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
TÀI LIỆU THAM KHẢO
PHỤ LỤC
```

---

## Danh mục hình

> Chèn tự động trong Word (Insert Table of Figures, nhãn `Hình`). Danh sách nguồn
> ảnh nằm ở [00-index.md §6](00-index.md#6-danh-mục-hình-dùng-trong-báo-cáo).

---

## Danh mục bảng

> Chèn tự động trong Word (Insert Table of Figures, nhãn `Bảng`).

---

## Danh mục từ viết tắt

| Viết tắt | Dạng đầy đủ | Nghĩa tiếng Việt |
|---|---|---|
| BPE | Byte-Pair Encoding | Mã hoá cặp byte — cách cắt từ thành đơn vị con |
| CPU | Central Processing Unit | Bộ xử lý trung tâm |
| CUDA | Compute Unified Device Architecture | Nền tảng tính toán trên GPU của NVIDIA |
| EDA | Exploratory Data Analysis | Phân tích khám phá dữ liệu |
| GELU | Gaussian Error Linear Unit | Một hàm kích hoạt phi tuyến |
| GPU | Graphics Processing Unit | Bộ xử lý đồ hoạ |
| IQR | Interquartile Range | Khoảng tứ phân vị |
| MAE | Mean Absolute Error | Sai số tuyệt đối trung bình |
| MPS | Metal Performance Shaders | Nền tảng tăng tốc GPU trên macOS |
| MSE | Mean Squared Error | Sai số bình phương trung bình |
| RMSE | Root Mean Squared Error | Căn bậc hai sai số bình phương trung bình |
| NFC | Normalization Form C | Một dạng chuẩn hoá Unicode |
| NLP | Natural Language Processing | Xử lý ngôn ngữ tự nhiên |
| PhoBERT | — | Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt |
| R² | Coefficient of Determination | Hệ số xác định |
| VND | Vietnam Dong | Đồng Việt Nam |

---

## Tóm tắt nội dung khoá luận

Tin tuyển dụng trực tuyến tại Việt Nam được đăng tự do, không theo một chuẩn chung.
Ngành nghề bị gán sai hoặc để trống, mức lương nhiều khi không được công bố. Khoá
luận này xây dựng hai mô hình đọc nội dung một tin tuyển dụng tiếng Việt: một mô hình
dự đoán **nhóm ngành nghề**, một mô hình ước lượng **mức lương**.

Dữ liệu gồm 48.092 tin thô, sau khi loại trùng còn **47.707 tin** thuộc **16 nhóm
ngành nghề**, trong đó **71,5 %** tin có công bố mức lương. Dữ liệu được chia theo
**nhóm tin trùng lặp gần** chứ không theo dòng, thành ba tập 34.354 / 3.812 / 9.541
tin, với **0 nhóm nào bị tách qua hai tập**.

Phương pháp: nội dung tin được đưa qua **PhoBERT** đã tiền huấn luyện để thu vectơ
ngữ nghĩa 768 chiều, rồi qua hai mạng kết nối đầy đủ riêng — một mạng phân loại 16
lớp, một mạng hồi quy mức lương trên thang `log1p` chỉ đọc văn bản đã che số lương. Toàn bộ
được đối chiếu với hai loại mốc cơ sở: mốc ngây thơ không dùng mô hình và mốc dò
tuyến tính trên chính vectơ đó.

Trên tập `dev`, mạng hồi quy lương giảm sai số tuyệt đối trung bình từ mốc **5,70
triệu** xuống **4,13 triệu** đồng/tháng (R² trên thang log 0,515), còn mạng phân loại
đạt macro-F1 **0,6030** và F1 có trọng số **0,6413**, so với **0,5898** và **0,6271**
của mốc dò tuyến tính trên cùng bộ vectơ.

Một phát hiện quan trọng: **nhãn ngành nghề gần như không nói gì về mức lương** —
phân rã phương sai cho eta² = **0,032**, tức ngành nghề chỉ giải thích 3,2 % biến
thiên của log-lương.

> **Nguồn số liệu:** `data/processed/manifest.json` · `artifacts/eda/summary.json` ·
> [04-results.md](../04-results.md) dòng `dl-cat-s2`, `dl-sal-s2`, `probe-cat-s2` ·
> [05-phan-tich-du-lieu.md §8](../05-phan-tich-du-lieu.md).

**Từ khoá:** xử lý ngôn ngữ tự nhiên tiếng Việt, phân loại văn bản, hồi quy mức
lương, PhoBERT, tin tuyển dụng.

---

[← Bảng điều khiển](00-index.md) · [Chương 1 →](02-chuong-1-gioi-thieu.md)
