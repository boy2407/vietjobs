[← Tổng quan](00-tong-quan.md) · [← Baseline học sâu](06-baseline-dl.md) · [Mã nguồn →](08-ma-nguon.md)

# Bài toán 2 — ước lượng lương

Đầu ra lương là một mạng dense trên vector PhoBERT họ `masked`
([06-baseline-dl.md](06-baseline-dl.md)). Kết quả trên lược đồ chia v2 (T1.4,
`dl-sal-cpu-0920`, `dev` 2.698 tin có nhãn lương): MAE **4,13** triệu, RMSE **8,13** triệu, R² log **0,515**,
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

## Đuôi phải: thử hàm mất mát bất đối xứng (T2.1, 2026-10-17)

*Phạm vi: huấn luyện trên `train`, chấm trên `dev`, 2.698 tin có công khai lương, trong đó 127 tin ≥ 30 triệu.* Mô hình: CafeBERT 512 token đóng băng, họ `masked`, head Dense h = 256, `cpu`. Chỉ hàm mất mát thay đổi; mỗi cấu hình chạy 3 seed (42, 43, 44); bảng là trung bình 3 seed.

Quantile (pinball) phạt đoán thấp với trọng số τ và đoán cao với 1 − τ. Huber bất đối xứng nhân Huber với 2τ khi đoán thấp và 2(1 − τ) khi đoán cao; τ = 0,5 trùng Huber gốc. τ > 0,5 đẩy dự đoán lên.

| Hàm mất mát | MAE toàn bộ (± sd) | MAE < 30 tr | RMSE < 30 tr | MAE ≥ 30 tr | RMSE ≥ 30 tr | Dự đoán TB nhóm ≥ 30 tr | Dự đoán lớn nhất |
|---|---|---|---|---|---|---|---|
| Huber (gốc) | **4,118** ± 0,021 | 3,22 | 4,45 | 22,25 | 32,16 | 24,7 | 44,9 |
| quantile τ = 0,5 | **4,082** ± 0,007 | 3,16 | 4,42 | 22,66 | 32,53 | 24,4 | 43,5 |
| quantile τ = 0,7 | 4,627 ± 0,151 | 3,92 | 5,33 | 18,97 | 29,30 | 28,9 | 59,8 |
| quantile τ = 0,9 | 8,534 ± 0,560 | 8,15 | 10,70 | **16,38** | **25,17** | 41,7 | 136,1 |
| Huber bất đối xứng τ = 0,7 | 4,353 ± 0,073 | 3,56 | 4,93 | 20,44 | 30,27 | 27,2 | 62,1 |
| Huber bất đối xứng τ = 0,9 | 5,920 ± 0,297 | 5,38 | 7,56 | 16,86 | 26,08 | 35,0 | 106,2 |

Luật chọn được đặt trước khi đọc bảng: chỉ giữ cấu hình làm MAE ≥ 30 tr giảm **và** MAE toàn bộ tăng không quá 0,05 tr so với Dense CafeBERT (4,130). **Không cấu hình nào đạt.** τ = 0,5 giữ MAE toàn bộ nhưng đuôi không giảm (22,66 so 22,25). Mọi τ > 0,5 kéo đuôi xuống (tốt nhất 16,38 ở quantile τ = 0,9) nhưng làm hỏng phần giữa nhiều hơn: MAE < 30 tr tăng từ 3,22 lên 3,56–8,15. Có 2.571 tin < 30 tr và 127 tin ≥ 30 tr, nên đổi phần giữa lấy đuôi làm MAE toàn bộ tệ đi. Đây là đánh đổi, không phải cải thiện. Huber gốc được giữ.

Một giới hạn của phép đo: epoch tốt nhất được chọn theo MAE toàn bộ trên `dev`, tức là theo đúng tiêu chí mà các τ cao bị phạt.

> **Nguồn số liệu:** `artifacts/dl-sal-cafe-dense-512{,-s43,-s44}/` và `artifacts/dl-sal-cafe-dense-512-{q50,q70,q90,ah70,ah90}-s{42,43,44}/` — `predictions_dev.parquet` (bảng tách ngưỡng) và `metrics.json`; 15 dòng tương ứng trong `04-results.md`.

---

## Đọc tiếp

- [Xử lý tiếng Việt §4](02-vietnamese-nlp.md#4-luồng-phobert-thực-sự-chạy-những-bước-nào) — vì sao bài toán này đọc các cột `*_masked`
- [Làm sạch dữ liệu](01-data-audit.md#5-nhãn--xây-dựng-và-sửa-lỗi) — nhãn lương được xây dựng ra sao
- Nền tảng: [chính quy hoá](nen-tang/07-chinh-quy-hoa.md) · [rò rỉ dữ liệu](nen-tang/05-ro-ri-du-lieu.md)
