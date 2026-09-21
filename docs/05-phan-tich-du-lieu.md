[← Tổng quan](00-tong-quan.md) · [← Xử lý tiếng Việt](02-vietnamese-nlp.md) · [Baseline học sâu →](06-baseline-dl.md)

# Phân tích dữ liệu trước khi xây dựng bất kỳ mô hình nào

**Phạm vi — tệp gốc.** Mọi con số mô tả dưới đây được đo trên
`data/raw/VietJobs.csv` sau khi loại trùng lặp chính xác: **47.707 dòng**, toàn
bộ kho dữ liệu như đã thu thập, không phải một tập con ([AGENTS.md Rule 8](../AGENTS.md#rule-8--data-analysis-is-measured-on-the-original-file)).
Các con số **quyết định** — ngưỡng sàn không mô hình, lựa chọn `--max-len` — là
những con số duy nhất được đo trên một tập con, chúng được khớp (fit) trên
`train` và chấm điểm trên `dev`, và được đánh dấu như vậy. `test` không bao giờ
được đọc ở đây.

```bash
python scripts/analyze_data.py                                   # đo + vẽ biểu đồ
PYTHONPATH=src .venv-dl/bin/python scripts/analyze_data.py --tokens   # hình 07
```

Đầu ra đầy đủ: [`artifacts/eda/summary.json`](../artifacts/eda/summary.json).

> **Đã đo lại ngày 2026-09-12** (T1.1). `artifacts/eda/summary.json` đã được
> tạo lại bằng lệnh ở trên, bao gồm cả `--tokens` bên trong `.venv-dl`, và các
> ngưỡng sàn trong [mục 7](#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev)
> đã được khớp và chấm điểm trên các tập chia v2 hiện tại (`train` 34.354 /
> `dev` 3.812). Không còn `⛔` nào bên dưới.

---

## 1. Bảy câu hỏi, bảy hình vẽ

| Câu hỏi | Hình | Câu trả lời một câu |
|---|---|---|
| Những trường nào thực sự được điền đầy đủ? | [eda-01](figures/eda/eda-01-do-day-truong.png) | Mười một trong mười tám cột được điền đầy đủ **100%**; `languages_required` chỉ được điền trong **25,6%** số dòng |
| Các lớp lệch nhau đến mức nào? | [eda-02a](figures/eda/eda-02a-lech-lop.png) · [eda-02b](figures/eda/eda-02b-lorenz.png) | **25,5 : 1** giữa lớp lớn nhất và lớp nhỏ nhất, Gini **0,439** |
| 16 ngành nghề phân bố theo quy mô như thế nào? | [eda-03a](figures/eda/eda-03a-co-lop.png) · [eda-03b](figures/eda/eda-03b-cong-bo-luong.png) | Một dải liên tục, độ dốc Zipf **−1,18**; lớp càng nhỏ thì tần suất công bố lương càng thấp (r = **0,610**) |
| Lương phân tán như thế nào trong từng ngành? | [eda-04](figures/eda/eda-04-luong-theo-nganh.png) | Trung vị của cả 16 ngành chỉ trải từ **11,0 → 16,0**, trong khi mỗi ngành lại trải từ 1 đến vài trăm |
| Lương được phân phối như thế nào? | [eda-05a](figures/eda/eda-05a-thang-tho.png) · [eda-05b](figures/eda/eda-05b-sau-log1p.png) | Độ lệch (skew) **11,90** trên thang gốc, **0,10** sau `log1p` |
| Nhãn lương tồn tại ở đâu? | [eda-06](figures/eda/eda-06-cong-bo-luong.png) | **71,5%** tin tuyển dụng có công bố lương; ngành thấp nhất là 58,6% |
| PhoBERT mất bao nhiêu văn bản ở ngưỡng 256 token? | [eda-07](figures/eda/eda-07-do-dai-token.png) | Trung vị **178** token, p95 **370**; ngưỡng 256 cắt bớt **19,8%** số tin nhưng giữ lại **91,5%** tổng số token. 256 là **trần cứng** của `phobert-base-v2` (`max_position_embeddings = 258`), không phải một siêu tham số được chọn — chuỗi dài hơn làm mô hình báo `IndexError` |

![Lương phân tán theo ngành](figures/eda/eda-04-luong-theo-nganh.png)

---

## 2. Độ đầy đủ của trường dữ liệu — cột nào thực sự là đặc trưng

![Độ đầy đủ trường dữ liệu](figures/eda/eda-01-do-day-truong.png)

Nguồn: [eda-01](figures/eda/eda-01-do-day-truong.png), 47.707 dòng.

| Cột | Đã điền | Tỷ lệ điền |
|---|---|---|
| `salary_avg` · `salary_max` · `salary_min` · `category` · `description` · `salary` · `contract_type` · `location` · `job_title` · `experience_required` · `country` | 47.707 | **100%** |
| `requirements_text` | 47.698 | 100,0% |
| `benefits` | 47.668 | 99,9% |
| `qualifications` | 47.338 | 99,2% |
| `soft_skills` | 45.087 | 94,5% |
| `working_hours` | 42.760 | **89,6%** |
| `technical_skills` | 41.331 | **86,6%** |
| `languages_required` | 12.223 | **25,6%** |

Ba cột dưới 90% là những cột cần đọc kỹ. Một trường trống ở ba trong bốn dòng
thì **không phải là đặc trưng** — một mô hình huấn luyện trên trường đó sẽ học
được rằng "tin này không điền trường đó", đây là một đặc điểm của quy trình
nhập liệu, không phải của công việc. `languages_required` ở mức 25,6% chính
xác là trường hợp đó, và cần nhớ rằng cùng cột này cũng chịu ảnh hưởng nặng
nhất bởi sự nhập nhằng giữa "rỗng" và "thiếu" trong các tệp CSV đã chia tách
([01-data-audit.md §8](01-data-audit.md#8-tách-từ-diễn-ra-ở-đây--đúng-một-lần)).

Ba cột văn bản tự do được đưa vào PhoBERT — `job_title`, `description`,
`requirements_text` — được điền gần như ở mọi dòng. Đầu vào của hướng chính
không bị ảnh hưởng bởi bất kỳ điều nào ở trên.

---

## 3. Mất cân bằng lớp — 25,5 : 1

![Mất cân bằng lớp](figures/eda/eda-02a-lech-lop.png)

Nguồn: [eda-02a](figures/eda/eda-02a-lech-lop.png) và
[eda-03a](figures/eda/eda-03a-co-lop.png), 47.707 dòng.

| Ngành | Số tin | Tỷ trọng | Số tin công bố lương | Tỷ lệ | Lương trung vị |
|---|---|---|---|---|---|
| `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` | **8.213** | 17,2% | 6.141 | 74,8% | 14,0 |
| `sản_xuất_lao_động_phổ_thông_cơ_khí` | 6.341 | 13,3% | 4.371 | 68,9% | 12,5 |
| `marketing_truyền_thông_quảng_cáo_nội_dung` | 5.947 | 12,5% | 4.388 | 73,8% | 12,5 |
| `tài_chính_kế_toán_ngân_hàng_bảo_hiểm` | 5.454 | 11,4% | 3.798 | 69,6% | 13,0 |
| `du_lịch_nhà_hàng_khách_sạn_dịch_vụ` | 4.198 | 8,8% | 3.211 | **76,5%** | 12,5 |
| `thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí` | 3.406 | 7,1% | 2.524 | 74,1% | 13,5 |
| `nhân_sự_hành_chính_pháp_chế_tư_vấn` | 3.210 | 6,7% | 2.166 | 67,5% | 12,5 |
| `xây_dựng_kiến_trúc_bất_động_sản` | 2.809 | 5,9% | 1.987 | 70,7% | **16,0** |
| `công_nghệ_thông_tin_kỹ_thuật_số` | 1.903 | 4,0% | 1.143 | 60,1% | **16,0** |
| `logistics_vận_tải_chuỗi_cung_ứng` | 1.811 | 3,8% | 1.249 | 69,0% | 12,5 |
| `kỹ_thuật_điện_điện_tử_viễn_thông` | 1.236 | 2,6% | 894 | 72,3% | 13,5 |
| `giáo_dục_đào_tạo_nghiên_cứu` | 1.165 | 2,4% | 882 | 75,7% | 13,5 |
| `y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học` | 963 | 2,0% | 693 | 72,0% | 14,0 |
| `ngôn_ngữ_dịch_thuật` | 384 | 0,8% | 238 | 62,0% | 15,0 |
| `nhóm_nghề_khác` | 345 | 0,7% | 202 | **58,6%** | **11,0** |
| `nông_nghiệp_năng_lượng_môi_trường` | **322** | 0,7% | 206 | 64,0% | 14,0 |
| **Tổng** | **47.707** | 100% | **34.093** | **71,5%** | 13,5 |

| Đại lượng | Giá trị | Cách đọc |
|---|---|---|
| Tỷ lệ lệch | **25,5 : 1** | lớn nhất 8.213 so với nhỏ nhất 322 |
| Chia đều cho 16 lớp | 2.982 tin/lớp | chỉ có **7** lớp đạt mức đó, 9 lớp còn lại nằm dưới |
| Kích thước lớp trung vị | 2.356 tin | thấp hơn trung bình 2.982 — một dải lệch phải |
| Bốn lớp lớn nhất | **54,4%** dữ liệu | bốn ngành chiếm hơn một nửa kho dữ liệu |
| Bốn lớp nhỏ nhất | **4,2%** dữ liệu | cả bốn cộng lại vẫn chưa bằng một phần tư lớp lớn nhất |
| Gini của kích thước lớp | **0,439** | |
| Độ dốc Zipf (log n theo log hạng) | **−1,18** | một đuôi dài liên tục, không phải vài lớp hiếm rời rạc |

![Biểu đồ phân tán kích thước lớp](figures/eda/eda-03a-co-lop.png)

Độ dốc −1,18 là con số quan trọng nhất trong bảng: nó cho thấy mất cân bằng
lớp ở đây là **một dải mượt**, không phải hai cụm "lớp bình thường" và "lớp
hiếm". Vì vậy không có ngưỡng tự nhiên nào để tách một vài lớp hiếm và gộp
chúng vào `nhóm_nghề_khác` — bất kỳ điểm cắt nào cũng là tùy tiện, và mỗi lớp
bị gộp sẽ mất đi nhãn thật của nó.

Hệ quả cho việc chấm điểm: luôn dự đoán lớp lớn nhất cho độ chính xác
(accuracy) gần với tỷ trọng của nó (17,2% trên toàn kho dữ liệu) và macro-F1
gần 1/16 giá trị đó. Con số chính xác trên `dev` là con số quyết định và nằm ở
[mục 7](#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev).
Đó là lý do chỉ số chính là macro-F1 — xem
[nen-tang/06-do-luong-va-baseline.md](nen-tang/06-do-luong-va-baseline.md).

Hệ quả cho việc huấn luyện: một hàm mất mát cross-entropy trần trụi sẽ thiên
vị bốn lớp lớn (54% dữ liệu). Cờ `--class-weight` trong
[`dl/train_dl.py`](../src/vietjobs/dl/train_dl.py) tồn tại để **đo** xem việc
cân bằng lớp có giúp ích hay không, chứ không phải để bật mặc định.

## 4. Phân phối lương — đuôi dài, và đó là lý do dùng `log1p`

Chỉ tính trên **34.093** tin có công bố lương, đơn vị triệu VNĐ/tháng. Nguồn:
[eda-05a](figures/eda/eda-05a-thang-tho.png) ·
[eda-05b](figures/eda/eda-05b-sau-log1p.png).

| Đại lượng | Giá trị |
|---|---|
| Trung vị | **13,5** |
| Trung bình | **15,6** |
| Giá trị lớn nhất | **500,0** |
| Độ lệch (skew) — thang gốc | **11,90** |
| Độ lệch (skew) — sau `log1p` | **0,10** |
| Độ nhọn (kurtosis) — thang gốc | **282** |
| p01 · p05 · p25 · p75 · p95 · p99 | **2,5 · 6,5 · 10,5 · 17,5 · 30,0 · 50,0** |

Giá trị trung bình 15,6 nằm **cao hơn trung vị 13,5** — dấu hiệu kinh điển
của đuôi phải. Huấn luyện hồi quy trên thang gốc nghĩa là để vài chục tin
đăng trên 200 triệu kéo lệch toàn bộ gradient; huấn luyện trên `log1p` đưa độ
lệch về 0,10, gần như đối xứng. Vì vậy đầu ra hồi quy học `log1p(salary_mid)`
và mọi báo cáo chuyển ngược lại về triệu VNĐ
([`evaluate.regression_metrics`](../src/vietjobs/evaluate.py)).

## 5. Phần rìa — đâu là đuôi phân phối thật, đâu là rác

Nguồn: [eda-05c](figures/eda/eda-05c-duoi-iqr.png).

| Ngưỡng | Số tin | Cách đọc |
|---|---|---|
| Trên hàng rào IQR (> 28,00) | **2.204** (6,5% số nhãn đã công bố) | Phần lớn là lương quản lý thật — **không bị cắt bỏ** |
| > 200 triệu/tháng | **15** | Gần như chắc chắn là lương năm, hoặc sai đơn vị |
| < 2 triệu/tháng | **132** | Gần như chắc chắn là lương theo giờ hoặc theo ca bị nhập nhầm trường |
| Dưới hàng rào IQR | **0** | Cận dưới của hàng rào là 0,0, nên không có gì rơi xuống dưới |
| Đã được `dataset.clean` gắn cờ `salary_extreme` (≥ 100) | **99** | Thấp hơn 2.204 tin trên hàng rào IQR — bộ lọc **không** bao phủ mọi tin đáng ngờ |
| Tỷ lệ tin đã công bố cho một *khoảng*, không phải một con số | **92,9%** | `salary_mid` là điểm giữa của khoảng đó, nên bản thân nhãn cũng chỉ là một giá trị xấp xỉ |

## 6. Nhãn lương chỉ tồn tại cho 71,5% dữ liệu

![Tỷ lệ công bố lương theo ngành](figures/eda/eda-06-cong-bo-luong.png)

**34.093 trong số 47.707** tin đăng có công bố lương — **71,5%**. Theo từng
ngành, tỷ lệ này trải từ **58,6%** (`nhóm_nghề_khác`) đến **76,5%**
(`du_lịch…`), cột theo từng lớp trong [mục 3](#3-mất-cân-bằng-lớp--255--1), và nó
**không phân bố ngẫu nhiên**:
[eda-03b](figures/eda/eda-03b-cong-bo-luong.png) cho thấy lớp càng nhỏ thì
tần suất công bố lương càng thấp.

| Kích thước lớp kéo theo điều gì | Hệ số | p |
|---|---|---|
| Tỷ lệ công bố — Pearson(log n) | **0,610** | 0,0122 |
| Lương trung vị — Spearman | **−0,227** | 0,398 (không có ý nghĩa thống kê) |

Kích thước lớp gắn chặt với **việc nhãn lương có tồn tại hay không**. Vì vậy
năm ngành nhỏ nhất chịu hình phạt kép: ít tin đăng hơn để học lớp đó, và càng
ít nhãn hơn cho đầu ra hồi quy. Ba hệ quả:

1. Đầu ra hồi quy chỉ có nhãn cho 7 trong 10 tin đăng. Mạng lương chỉ học và
   chấm trên các tin có công bố lương; **28,5%** còn lại bị loại khỏi bài này,
   không bị gán nhãn 0.
2. Nhiệm vụ `disclosed` có một ngưỡng sàn theo đa số (majority baseline) gần
   bằng chính tỷ lệ công bố. Bất kỳ mô hình nào không vượt qua được ngưỡng đó
   đều vô dụng; con số chính xác trên `dev` nằm ở
   [mục 7](#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev).
3. Việc công bố lương **phụ thuộc vào ngành**, nên việc bỏ qua 28,5% còn lại
   không phải là một sự thiếu sót ngẫu nhiên: mô hình hồi quy học trên một
   mẫu đã bị chọn lọc từ trước. Cùng sự chọn lọc đó cũng nằm bên dưới eta² ở
   [mục 7](#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev), vì
   con số đó cũng chỉ được tính trên tập con đã công bố lương.

## 7. Ngưỡng sàn — các con số quyết định, khớp trên `train`, chấm điểm trên `dev`

Đây là mục duy nhất **không** đọc tệp gốc. Một ngưỡng sàn là thứ mà một mô
hình phải vượt qua, nên nó phải được đo ở nơi các mô hình được đo: khớp trên
`train`, chấm điểm trên `dev`, không bao giờ trên `test` (Rule 5, Rule 8).

> **Đã đo lại theo lược đồ chia tập v2** (`train` 34.354 / `dev` 3.812 /
> `test` 9.541), ngày 2026-09-12. Các dòng dưới đây thay thế các con số v1
> (train 33.396 / dev 7.159).

| Ngưỡng sàn | Giá trị | R² |
|---|---|---|
| Lương — trung vị `train` cho mọi tin đăng (13,0) | MAE **5,70** triệu | −0,059 |
| Lương — **biết ngành thật**, dự đoán trung vị của ngành đó | MAE **5,57** triệu | −0,022 |
| Nghề nghiệp — luôn dự đoán lớp lớn nhất | accuracy 0,2062 · macro-F1 **0,0214** | — |
| Công bố lương — luôn dự đoán "có" | accuracy **0,7078** | — |

## 8. Phát hiện quan trọng nhất: ngành nghề hầu như không nói lên điều gì về mức lương

Phân rã phương sai của `log1p(salary_mid)` theo 16 ngành, trên 34.093 nhãn đã
công bố của tệp gốc. Nguồn: [eda-04](figures/eda/eda-04-luong-theo-nganh.png).

| Đại lượng | Giá trị |
|---|---|
| eta² (phương sai giữa các ngành / tổng phương sai) | **0,032** — ngành nghề giải thích **3,2%** |
| Trung vị thấp nhất | `nhóm_nghề_khác` — 11,0 |
| Trung vị cao nhất | `xây_dựng_kiến_trúc_bất_động_sản` và `công_nghệ_thông_tin_kỹ_thuật_số` — 16,0 |
| Trung vị toàn kho dữ liệu | 13,5 |

Các ngưỡng sàn trong [mục 7](#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev)
đi đến cùng kết luận nhưng từ phía ngược lại: biết 100% nhãn ngành chỉ làm
MAE giảm từ 5,70 xuống 5,57, tức giảm **2,3%**. Hai kết luận:

1. **Ngưỡng cần vượt qua cho đầu ra hồi quy xấp xỉ ngưỡng sàn dự đoán-bằng-
   trung-vị**, chứ không phải 0. Một mô hình chỉ đạt đúng ngưỡng sàn đó coi
   như chưa học được gì.
2. Tín hiệu về lương nằm trong **các chi tiết của tin đăng** — cấp bậc, số
   năm kinh nghiệm, ngôn ngữ, địa điểm — chứ không nằm trong nhãn ngành. Đó
   chính xác là điều PhoBERT có cơ hội đọc được còn nhãn ngành thì
   không.

Hãy đọc con số 3,2% cùng với thiên lệch chọn mẫu (selection bias) ở
[mục 6](#6-nhãn-lương-chỉ-tồn-tại-cho-715-dữ-liệu): nó được đo trên
tập con đã công bố lương, và bản thân việc công bố lại tương quan với kích
thước lớp (r = 0,610).

---

[← Tổng quan](00-tong-quan.md) · [Baseline học sâu →](06-baseline-dl.md)
