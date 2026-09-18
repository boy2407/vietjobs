[← Tổng quan](00-tong-quan.md) · [← Mô hình phân lớp](archive/06-mo-hinh-phan-lop.md) · [Mã nguồn →](08-ma-nguon.md)

# Bài toán 2 — ước lượng lương

Đầu ra lương là một mạng dense trên vector PhoBERT họ `masked`
([06-baseline-dl.md](06-baseline-dl.md)). Kết quả trên lược đồ chia v2 (T1.4,
`dl-sal-s2`, `dev` 2.698 tin có nhãn lương): MAE **4,15** triệu, RMSE **8,29** triệu, R² log **0,512**,
so với mốc đoán trung vị 5,70 triệu — chi tiết ở
[06 §5.2](06-baseline-dl.md#52-ước-lượng-lương).

Hai con số cần nhớ, đo trong [05](05-phan-tich-du-lieu.md): chỉ **71,5 %** tin
tuyển dụng có nhãn lương (trên tệp gốc, 34.093 trên 47.707), và biết đúng ngành
nghề chỉ giúp MAE giảm từ **5,70** xuống **5,57** triệu (`train` → `dev` v2).

---

## Vì sao bài toán này khó hơn phân lớp

Ba lý do, và không lý do nào có thể khắc phục bằng một mô hình tốt hơn:

**1. Đầu vào bị cố ý làm nghèo đi.** Phân lớp ngành nghề đọc văn bản gốc; còn
bài toán lương chỉ được đọc bản đã **che lương**. Đây là bắt buộc — `salary_*`
chính là nhãn, và **10,65 %** số dòng có nhắc lại con số lương trong trường
phúc lợi
([02-vietnamese-nlp.md](02-vietnamese-nlp.md#3-chín-bước--làm-gì-vì-sao-và-điều-gì-đo-được),
bước 4). Nếu không che, mô hình sẽ đọc thấy đáp án của chính nó.

**2. Nhãn chỉ tồn tại trên một phần dữ liệu, và phần đó không ngẫu nhiên.**
Xem phần thiên lệch chọn mẫu bên dưới.

**3. Mục tiêu là một số thực có đuôi dài**, không phải nhãn rời rạc. Đó là lý
do mục tiêu hồi quy là `log1p(salary_mid)` chứ không phải `salary_mid`
([01-data-audit.md §5](01-data-audit.md#5-nhãn--xây-dựng-và-sửa-lỗi)).

---

## Cạm bẫy đã đo: hồi quy tuyến tính không chính quy hoá

Đo ở nhánh học máy (TF-IDF trên tiêu đề, p/n = 0,22, lược đồ v1):

| Mô hình | R² train | R² dev | MAE dev |
|---|---|---|---|
| LinearRegression thuần | 0,718 | **0,081** | **6,00 M** |
| Ridge alpha=1 | 0,657 | **0,507** | **4,25 M** |

MAE 6,00 triệu còn tệ hơn cả mốc chuẩn "trung vị theo nhóm ngành nghề"
(5,54 M, lược đồ v1) — nghĩa là tệ hơn cả việc không dùng mô hình nào.

Đọc kỹ hai dòng này: R² trên train giảm (0,718 → 0,657) trong khi R² trên dev
**tăng gấp sáu lần** (0,081 → 0,507). Đó chính là định nghĩa của hiện tượng
overfitting (quá khớp), và cũng là toàn bộ lý do vì sao chính quy hoá tồn tại
— [nền tảng: chính quy hoá](nen-tang/07-chinh-quy-hoa.md).

---

## Thiên lệch chọn mẫu — một giới hạn của dữ liệu

Tỷ lệ công khai lương khác nhau theo ngành nghề (tệp gốc, [05 §3](05-phan-tich-du-lieu.md)):

| Ngành nghề | Tỷ lệ công khai lương | Trung vị lương (triệu) |
|---|---|---|
| `nhóm_nghề_khác` | **58,6 %** — thấp nhất | 11,0 |
| `công_nghệ_thông_tin_kỹ_thuật_số` | 60,1 % | **16,0** — cao nhất |
| `du_lịch_nhà_hàng_khách_sạn_dịch_vụ` | **76,5 %** — cao nhất | 12,5 |

Mô hình lương chỉ học trên các tin **có** công khai lương, nên nó học trên một
mẫu đã bị chọn lọc theo ngành. Ảnh hưởng của việc này lên sai số chưa được đo
(xem [09-lo-trinh.md](09-lo-trinh.md) Ưu tiên 4).

---

## Đọc tiếp

- [Xử lý tiếng Việt §4](02-vietnamese-nlp.md#4-luồng-phobert-thực-sự-chạy-những-bước-nào) — vì sao bài toán này đọc các cột `*_masked`
- [Làm sạch dữ liệu](01-data-audit.md#5-nhãn--xây-dựng-và-sửa-lỗi) — nhãn lương được xây dựng ra sao
- Nền tảng: [chính quy hoá](nen-tang/07-chinh-quy-hoa.md) · [rò rỉ dữ liệu](nen-tang/05-ro-ri-du-lieu.md)
