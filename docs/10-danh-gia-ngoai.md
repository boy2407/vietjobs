[← Tổng quan](00-tong-quan.md) · [← Lộ trình](09-lo-trinh.md)

# Đánh giá chéo trên tập ngoài VietJobs-37K

Câu hỏi của ghi chú này: mô hình phân loại ngành học được *nghề* hay chỉ học
*giọng* của `VietJobs.csv`? Cách trả lời: chấm mô hình đã huấn luyện trên một
bộ tin tuyển dụng do người khác thu thập, tiền xử lý khác, gán nhãn khác, và
**không đưa một dòng nào của bộ đó vào huấn luyện**.

> **Trạng thái 2026-09-17:** đã chạy với bảng ánh xạ 60 → 16 ở trạng thái `draft`
> — mọi con số dưới đây là **số tham khảo** (T7.1).

---

## 1. Bộ dữ liệu ngoài là gì

| | VietJobs (của dự án) | VietJobs-37K (ngoài) |
|---|---|---|
| Nguồn | topcv, careerviet, job3s | topcv, careerviet, job3s, vieclam24h · 1/2025 → 4/2026 |
| Số tin | 47.707 | 37.274 (train 29.795 · dev 3.701 · test 3.778) + Gold-1000 lấy từ test |
| Nhãn | 16 lớp, **đơn nhãn** | 60 nhãn, **đa nhãn** (trung bình 1,67 nhãn/tin) |
| Ai gán nhãn | nhóm dự án | máy (từ khóa + LLM audit), gọi là *silver*; chỉ Gold-1000 do người soát (Krippendorff α = 0,79) |
| Lương | có, 71,5 % tin công khai | **không có trường lương** · một cột *bạc* đọc từ văn bản phủ 16,5 % tin (§3) |
| Văn bản | ba cột riêng | một chuỗi `[TITLE] … [REQ] … [DESC] …`, đã che tên công ty, email, điện thoại, URL |

Nguồn gốc, giấy phép và những gì đã lược bỏ khỏi gói gốc:
[data/external/vietjobs37k/PROVENANCE.md](../data/external/vietjobs37k/PROVENANCE.md).

Vì không có lương, bộ này **chỉ phục vụ bài `category`**.

## 2. Ba việc phải làm trước khi chấm

### 2.1. Khử tin trùng với dữ liệu của ta

Cùng nguồn, cùng khoảng thời gian, nên một phần tin của 37K cũng nằm trong
`VietJobs.csv`. Chấm trên những tin đó là chấm trên tập huấn luyện. Hai đường
bắt trùng, trúng một trong hai là loại (`external.overlap_mask`):

- **Khớp băm chính xác** — `group_key` (băm sau khi gập hoa/thường, dấu, ký tự
  đặc biệt; chính là khóa `dataset.build` dùng để giữ tin đăng lại trong cùng
  một tập chia) có mặt trong `train`, `dev` hoặc `test` của ta.
- **Khớp mờ** — cùng tiêu đề (đã gập) **và** Jaccard tập từ của mô tả ≥ 0,5,
  sau khi bỏ các token che `[COMPANY]`, `[EMAIL]`…

Vì sao cần đường mờ: văn bản 37K đã bị che định danh và định dạng lại, nên
đường băm chính xác **bắt được 0 tin** trên cả Gold lẫn test. Ngưỡng 0,5 thấp
có chủ ý (gói gốc tự kiểm dùng 0,85): bỏ oan một tin ngoài chỉ mất một chút
mẫu, giữ nhầm một tin trong tập huấn luyện thì thổi cả con số.

| Tập | Số tin | Cùng tiêu đề | Trùng (Jaccard ≥ 0,5) | Trùng (Jaccard ≥ 0,85) |
|---|---|---|---|---|
| Gold-1000 | 1.000 | 287 | **19 (1,9 %)** | 3 |
| test | 3.778 | 1.043 | **69 (1,83 %)** | 12 |

Con số **13,6 %** ghi trong PROVENANCE.md trước đây không tái lập được bằng
định nghĩa nào ở trên; nó gần với tỉ lệ *cùng tiêu đề* (28,7 % / 27,6 %) chia
đôi hơn là với trùng nội dung. Mức trùng thật, theo định nghĩa của dự án, là
**dưới 2 %** — đủ nhỏ để bộ này xứng đáng gọi là tập ngoài.

### 2.2. Ánh xạ 60 nhãn → 16 lớp

Bảng ánh xạ nằm ở
[data/external/vietjobs37k/crosswalk_60_to_16.json](../data/external/vietjobs37k/crosswalk_60_to_16.json).
Mỗi nhãn 37K trỏ tới đúng một lớp của ta hoặc `null` (không chấm). Bốn nhãn
là `null`: *Quản lý điều hành*, *Quản lý dự án* (chức danh, không phải ngành,
rải khắp 16 lớp), *Thư viện / Lưu trữ* (8 mẫu), *Phi chính phủ / NGO* (16 mẫu,
là loại tổ chức). Tám mục được đánh `confidence: low`, trong đó có *Tư vấn* (1.565 tin
train) và *Truyền thông / Báo chí* (704 tin), vì tên lớp của ta có thể nhận cả
hai phía.

Sau ánh xạ, trên Gold-1000 sau khử trùng (981 tin): 4 tin mất hết nhãn, 323
tin có **hơn một** lớp đích, 529 tin có đúng một nhãn 37K và ánh xạ được.

### 2.3. Hai quy ước chấm

Vì 37K đa nhãn còn mô hình của ta chỉ trả một lớp, một con số không đủ:

- **strict** — chỉ những tin có **đúng một** nhãn 37K và nhãn đó ánh xạ được.
  Chấm như bình thường bằng `evaluate.classification_metrics`, kèm khoảng tin
  cậy bootstrap 1.000 lần. Đây là **con số chính**.
- **lenient** — mọi tin chấm được; tính đúng nếu lớp dự đoán nằm trong tập lớp
  đích. Để tính macro-F1 bằng hàm thường, nhãn "thật" của tin được quy là lớp
  dự đoán nếu trúng, ngược lại là lớp đích đầu tiên. Rộng tay theo cấu tạo, nên
  **không bao giờ báo một mình**.

## 3. Cột lương bạc đọc từ văn bản (T9)

Bộ ngoài không có trường lương, nhưng một phần tin tự nói mức lương trong văn
bản. `external.extract_salary` đọc con số đó (**bạc**, regex, chưa ai soát) và
`dataset.derive_salary` quy về đúng năm cột `VietJobs.csv` mang. Kết quả nằm ở
`data/external/vietjobs37k/ext37k.csv` (38.274 dòng) và `ext37k-sal.csv` (chỉ
tin có lương, mỗi văn bản một lần). Đơn vị và ý nghĩa giữ nguyên như
`VietJobs.csv`: **khoảng lương tin rao, triệu VNĐ/tháng**. Lương giờ, ngày, năm
và USD bị từ chối chứ không quy đổi.

### 3.1. Mỗi tin nằm trong đúng một nhóm

`scripts/build_ext37k.py --audit` xếp cả 38.274 dòng vào một nhóm duy nhất, nên
câu "16,5 % tin có lương" đọc được tiếp thành "trong phần còn lại, bao nhiêu tin
có nêu con số mà ta từ chối, bao nhiêu tin không nêu gì".

| Nhóm | Vòng 1 | Vòng 2 | Nghĩa |
|---|---|---|---|
| `read` | 5.289 | **6.328** | Đọc được lương |
| `payword_unit` | 2.261 | 1.846 | Có từ lương + số, đơn vị không đọc được — gần hết là "lương tháng 13" |
| `payword_gap` | 909 | 1.723 | Có từ lương, con số ở quá xa (> 40 ký tự) |
| `payword_break_word` | 435 | 676 | Giữa từ lương và con số có từ chặn (thưởng, phụ cấp…) |
| `payword_not_monthly` | 0 | 179 | Lương giờ/ngày/năm |
| `payword_far` | 101 | 73 | Có từ lương và có số, nhưng không cùng câu |
| `payword_loose_range` | 88 | 53 | "9 15 triệu" không dấu gạch, không đủ căn cứ |
| `payword_out_of_range` | 29 | 18 | Ngoài khoảng [2, 200] triệu |
| `header_figure` | 717 | 0 | Số đứng sau tiêu đề mục — vòng 2 đọc hết thành `read` |
| `figure_only` | 376 | 282 | Có số tiền, không có mỏ neo nào |
| `negotiable` | 3.431 | 3.246 | "Thỏa thuận", "cạnh tranh", không con số |
| `no_figure` | 24.638 | 23.850 | Không nêu con số nào |
| **Tổng** | 38.274 | 38.274 | |

Trần của việc đọc lương từ văn bản là khoảng 9.200 dòng (24 %), không phải
38.274: gần 24.000 tin không có con số nào trong văn bản để mà đọc.

### 3.2. Bốn luật vòng 2

Mỗi luật có một test nhận và một test từ chối trong `tests/test_external.py`.

| Luật | Ví dụ đọc được | Ví dụ vẫn từ chối | Dòng luật đọc |
|---|---|---|---|
| `payword` — từ lương + con số (vòng 1, không đổi) | `Lương: 10 - 15 triệu` | `Lương: thỏa thuận` | 5.078 |
| `header` — tiêu đề mục làm mỏ neo thay từ "lương" | `Quyền lợi:- 3.000.000 - 16.000.000 VNĐ` | `Quyền lợi- Phụ cấp ăn trưa: 40.000 vnđ` | 717 |
| `dong_range` — hai số viết đủ chữ số là một khoảng | `Lương cơ bản 10.000.000 13.000.000` | `Lương tháng 13 14` | 223 |
| `m_unit` — "M" là triệu trong lối viết tin | `Thu nhập 30-40M` | `Gói Benefit 18M năm` | 194 |
| `bare_range` — không mỏ neo thì chỉ khoảng có đơn vị | `Kế Toán Trưởng (45 - 60 Triệu)` | `ngân sách trong khoảng 100 - 200 triệu` | 116 |

Cột cuối là **số dòng cột `salary_rule` gán cho luật đó trong tệp hiện tại**,
không phải số dòng luật đó lấy thêm: thuộc tính luật suy ngược từ hình dạng
đoạn trích, nên một tin vòng 1 đã đọc được mà đoạn trích có dạng "M" thì tính
vào `m_unit`. Con số so vòng 1 là **+1.041 dòng đọc thêm, −2 dòng mất**
(một tin lương năm `130-160tr năm`, một tin "lương doanh thu" — cả hai đều
đúng ra phải từ chối), tức 5.289 → 6.328.

Hai từ khóa chặn được thêm vì đọc mẫu thấy sai: `ngân sách`, `trị giá`,
`hạn mức`, `doanh thu` (tiền nhưng không phải lương), và từ chỉ chu kỳ đứng
ngay sau con số (`100 triệu năm`) — vì khâu che định danh của gói gốc đã xóa
dấu `/`, nên `100 triệu/năm` đến tay ta thành `100 triệu năm`.

### 3.3. Ba cột mới: vì sao trống, luật nào đọc, tin nào đăng lại

- `salary_note` — tin không có lương luôn nói vì sao: `no_figure` 23.850 ·
  `figure_unread` 4.850 · `negotiable` 3.246. Không dòng nào im lặng.
- `salary_rule` — luật nào đọc ra dòng đó, để soát được precision **từng luật**.
- `template_id` — một mã cho mỗi thân tin giống hệt (mô tả + yêu cầu). Ai chia
  train/dev trên tệp này phải chia theo nhóm, nếu không cùng một tin rơi vào cả
  hai bên.

> **Giới hạn đã đo, chưa xử lý:** `template_id` chỉ bắt thân tin **giống hệt**.
> Mẫu tin lớn nhất — một ngân hàng đăng lại theo từng chi nhánh, `Quyền lợi-
> 3.000.000 - 16.000.000 VNĐ` — có 711 dòng (694 dòng trong `ext37k-sal.csv`,
> **11,3 %** tệp đó) nhưng phần yêu cầu đổi chút ít theo chi nhánh nên rơi vào
> **34 nhóm** khác nhau. Chia theo `template_id` vẫn rò mẫu tin này. Cần một
> khóa mờ (Jaccard) mới gom hết — chưa làm.

### 3.4. Người soát: precision 97,0 %

199 tin được soát tay (soát bởi tác giả, 2026-09-19), lấy mẫu phân tầng theo
luật, ghi ở `data/external/vietjobs37k/review-salary.csv` với ba giá trị
`correct` / `wrong` / `ambiguous`.

Con số dưới đây tính trên **165 dòng mang bằng chứng khác nhau** (gộp các dòng
có `salary_text` trùng nhau về một), và tính `ambiguous` là **sai**:

| Luật | Bằng chứng đã soát | Đúng | Precision | Độ phủ bằng chứng của luật |
|---|---|---|---|---|
| `payword` | 40 | 40 | 100 % | 40 / 3.434 (1,2 %) |
| `header` | 8 | 8 | 100 % | **8 / 8 — soát hết** |
| `bare_range` | 38 | 37 | 97,4 % | 38 / 95 (40 %) |
| `m_unit` | 40 | 38 | 95,0 % | 40 / 175 (23 %) |
| `dong_range` | 39 | 37 | 94,9 % | 39 / 206 (19 %) |
| **Tổng** | **165** | **160** | **97,0 %** | 165 / 3.918 (4,2 %) |

Ngưỡng đặt trước khi soát là 95 % chung và 90 % mỗi luật; **không luật nào bị
gỡ**. Hai dòng `wrong`: một tin đọc ra `80-100 triệu` vốn là tiền thuế hoàn khi
về nước (`bare_range`), một tin đọc cận dưới `2` vốn là phụ cấp KPI
(`dong_range`). Ba dòng `ambiguous` đều cùng một dạng: tin nêu cả *lương cứng*
lẫn *thu nhập* và luật trộn hai mức.

> **Cảnh báo về mẫu, đã đo:** mẫu ban đầu lọc trùng theo `template_id`, nhưng
> 33/39 dòng `header` vẫn rơi vào cùng tin ngân hàng — tin đó đổi chút phần yêu
> cầu theo chi nhánh nên giữ 33 mã `template_id` khác nhau. Bộ lọc mẫu nay lọc
> trùng theo **cả `template_id` lẫn `salary_text`**. Hệ quả cho luật `header`:
> nó lấy thêm 717 dòng nhưng chỉ dựa trên **7 quảng cáo khác nhau** (710 dòng là
> một tin ngân hàng). Đã soát hết cả 7, nên precision 100 % là đầy đủ chứ không
> phải mẫu mỏng — nhưng đóng góp của luật này gần như là *một nhà tuyển dụng*.

> **Nguồn số liệu:** `PYTHONPATH=src .venv/bin/python scripts/build_ext37k.py`
> và `--audit`, chạy 2026-09-18 · `data/external/vietjobs37k/ext37k.csv`,
> `ext37k-sal.csv`, `review-salary.csv` (199 dòng, soát xong 2026-09-19) ·
> `tests/test_external.py` (46 test xanh).

## 4. Kết quả

Mô hình: `dl-cat-s2` (PhoBERT đông cứng → dense h=256, huấn luyện trên `train`
của ta, chọn epoch trên `dev`). Lệnh:

```bash
PYTHONPATH=src .venv-dl/bin/python scripts/eval_external.py --run-id dl-cat-s2 --subset gold,test
```

**Bảng 10.1: Mô hình `dl-cat-s2` chấm trên VietJobs-37K, sau khử trùng và ánh xạ**

| Tập | n chấm | strict n | strict macro-F1 [KTC 95 %] | strict F1 | strict acc | lenient trúng | lenient macro-F1 |
|---|---|---|---|---|---|---|---|
| **Gold-1000** (người soát) | 977 | 529 | **0,3977** [0,353; 0,443] | 0,5094 | 0,5104 | 0,6080 | 0,4775 |
| test 37K (silver) | 3.687 | 2.053 | 0,4525 [0,421; 0,479] | 0,5393 | 0,5329 | 0,6138 | 0,4935 |
| *`dev` của ta, để đối chiếu* | 3.812 | 3.812 | *0,6025* | *0,6423* | *0,6511* | — | — |

Đọc bảng:

1. **Có tổng quát hóa, nhưng mất nhiều.** Trên Gold, macro-F1 strict rơi từ
   0,6025 (trong miền) xuống 0,3977 — mất **0,205**, gấp gần năm lần bề rộng
   khoảng tin cậy. F1 có trọng số giảm nhẹ hơn (0,642 → 0,509): mô hình giữ
   được các lớp lớn, phần mất tập trung ở các lớp nhỏ.
2. **Gold thấp hơn silver** (0,398 so với 0,453). Nhãn silver được gán bằng
   từ khóa, và mô hình của ta cũng dựa nhiều vào từ vựng, nên hai bên "đồng ý"
   với nhau nhiều hơn là với người soát. Đây là lý do Gold là con số chính.
3. **Khoảng tin cậy rộng** trên Gold (±0,045) vì chỉ 529 tin strict chia cho
   16 lớp; vài lớp có dưới 10 tin.

**Bảng 10.2: Phân bố lớp dự đoán trên Gold-1000 (977 tin) — dấu hiệu lệch miền**

| Lớp dự đoán | Số tin | Tỉ lệ |
|---|---|---|
| nhân_sự_hành_chính_pháp_chế_tư_vấn | 276 | 28,3 % |
| kinh_doanh_bán_hàng_chăm_sóc_khách_hàng | 208 | 21,3 % |
| xây_dựng_kiến_trúc_bất_động_sản | 138 | 14,1 % |
| tài_chính_kế_toán_ngân_hàng_bảo_hiểm | 93 | 9,5 % |
| sản_xuất_lao_động_phổ_thông_cơ_khí | 90 | 9,2 % |
| … | | |
| công_nghệ_thông_tin_kỹ_thuật_số | **5** | 0,5 % |
| du_lịch_nhà_hàng_khách_sạn_dịch_vụ | 3 | 0,3 % |
| ngôn_ngữ_dịch_thuật | 1 | 0,1 % |
| nhóm_nghề_khác | 0 | 0 % |

Hai điều bất thường trong phân bố dự đoán:

- Mô hình đổ **28 %** tin vào lớp nhân sự – hành chính; bốn trong năm cặp nhầm
  lớn nhất trên Gold strict có đích là lớp này (tài chính → nhân sự 27 tin, sản
  xuất → nhân sự 15, CNTT → nhân sự 13, logistics → nhân sự 12).
- Lớp CNTT gần như **không được dự đoán** (5 tin) dù Gold có ít nhất 67 tin
  CNTT phần mềm + hạ tầng.

Tệp chi tiết: `artifacts/dl-cat-s2/external_37k/{gold,test}/` gồm
`metrics.json`, `dedup_report.json`, `per_class_strict.csv`,
`confusions_strict.csv`, `predictions.parquet`.

## 5. Vì sao không so trực tiếp với Qwen-14B zero-shot

Gói gốc kèm mốc Qwen2.5-14B-Instruct zero-shot: macro-F1 0,459 trên test 37K,
**trên 60 nhãn đa nhãn**. Con số của ta là trên 16 lớp đơn nhãn sau ánh xạ.
Hai không gian nhãn khác nhau, hai bài toán khác nhau; đặt cạnh nhau chỉ để
nhắc rằng bài này khó ngay cả với mô hình lớn, không phải để xếp hạng.

## 6. Giới hạn cần viết vào luận văn

- Nhãn 37K là silver (trừ Gold); sai số nhãn của họ lẫn vào sai số của ta.
- Bảng ánh xạ do nhóm tự xây; đổi tám mục `low` có thể dịch con số vài phần trăm.
  Vì thế script in cảnh báo và ghi trạng thái bảng vào cột `Prep` của
  [04-results.md](04-results.md).
- Khử trùng theo tiêu đề + Jaccard là một định nghĩa; tin viết lại hoàn toàn
  không bị bắt. Số trùng là **cận dưới**.
- Văn bản 37K đã che định danh; mô hình của ta chưa từng thấy token `[COMPANY]`.

> **Nguồn số liệu:** hai dòng `dl-cat-s2-ext37k-gold` và `dl-cat-s2-ext37k-test`
> trong [04-results.md](04-results.md) ·
> `artifacts/dl-cat-s2/external_37k/gold/metrics.json` và `…/dedup_report.json` ·
> `artifacts/dl-cat-s2/external_37k/test/metrics.json` ·
> `data/external/vietjobs37k/taxonomy/per_label_stats.csv` (số tin Gold theo nhãn) ·
> `data/external/vietjobs37k/baselines/qwen14b_zeroshot/metrics.json`.
