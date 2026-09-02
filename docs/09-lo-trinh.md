[← Tổng quan](00-tong-quan.md) · [← Mã nguồn](08-ma-nguon.md)

# Lộ trình còn lại — xếp theo thứ tự phải làm

Thứ tự dưới đây không phải danh sách mong muốn. Mỗi bước đứng trước là điều kiện
để bước sau có nghĩa.

---

## Ưu tiên 0 — khoá train/serve skew (chặn mọi thứ khác)

`predict.build_frame` phải cho ra **đúng** bố cục cột mà `dataset.clean` đã tạo.
Docstring của `predict.py` nói `tests/test_predict.py` giữ điều này — **file đó chưa tồn tại**.

Cách làm: lấy 20 dòng bất kỳ từ `VietJobs.csv`, cho qua `build_frame`, so từng ô với
đúng 20 dòng đó trong `train.parquet`. Lệch một cột là mọi con số hệ thống mất giá trị.
Đây là loại lỗi **không báo lỗi** — nó chỉ làm dự đoán kém đi một cách lặng lẽ.

## Ưu tiên 0b — đóng lỗ hổng che lương chưa được test phủ

Hai cột đi thẳng vào mô hình lương ở dạng **chưa che**:

| Cột | Vào mô hình lương ở đâu | Vì sao lọt |
|---|---|---|
| `soft_skills_text` | [features.py:223](../src/vietjobs/features.py#L223) — truyền thẳng, không qua `col()` | Không có bản `_masked` |
| `qualifications_text` | [features.py:222](../src/vietjobs/features.py#L222) — qua `col()` nhưng không nằm trong `_MASKABLE` | Không có bản `_masked` |

`UNMASKED_COLUMNS` ([features.py:50](../src/vietjobs/features.py#L50)) không liệt kê
hai cột này, nên `tests/test_no_leak.py` **không bắt được**.
`scripts/measure_vitext.py` chỉ đo `job_title`, `description`, `requirements_text`
và `benefits` — nên **chưa biết** hai cột đó có nhắc lại con số lương hay không.

**Kiểm tra rẻ nhất, làm trước khi chạy bất kỳ dòng nào của bài toán 2:** đếm số
dòng khớp `_PAT_MILLIONS` / `_PAT_DONG` / `_PAT_USD` trong hai cột đó.

- Bằng 0 → đóng lại, chỉ cần thêm hai cột vào `UNMASKED_COLUMNS` cho chắc.
- Khác 0 → **là rò rỉ thật**. Phải dựng bản `_masked` cho hai cột, thêm vào
  `_MASKABLE`, và mọi con số bài toán 2 chạy trước đó phải bỏ.

Đây là ví dụ sống cho một điều đáng nhớ: **test chống rò rỉ chỉ bảo vệ được những
cột mà người viết test nghĩ tới.** Ghi thêm ở
[nền tảng: rò rỉ dữ liệu](nen-tang/05-ro-ri-du-lieu.md).

## Ưu tiên 1 — chạy bài toán lương

Code đã đủ, chỉ còn chạy và ghi kết quả. Bốn bước, đúng thứ tự. Bối cảnh ở
[07-bai-toan-luong.md](07-bai-toan-luong.md).

**1a. Tầng 1 — `disclosed`.** Sàn phải vượt: 71,8% tin có công bố lương, nên
"đoán luôn là có" đạt accuracy **0,7176** và macro-F1 **0,4179**. Mô hình không vượt
được hai số này là chưa học gì.

Đã kiểm tra trước dấu hiệu rò rỉ, kết quả **an toàn**:

| Dấu hiệu trong văn bản đã che | Tần suất | P(công bố \| có) | Nền = 0,718 |
|---|---|---|---|
| còn token `<SALARY>` | 11,1% | 0,809 | lệch nhẹ |
| chữ "thoả thuận" | 3,9% | 0,605 | lệch nhẹ |
| chữ "cạnh tranh" | 11,5% | 0,544 | lệch nhẹ |

Không có từ khoá nào tra ra đáp án. Tầng 1 là bài toán thật, không phải bài tra từ điển.
Báo cáo kèm ROC-AUC và đường hiệu chuẩn.

**1b. `LinearRegression` — chạy để cho thấy nó hỏng.** Bốn cấu hình: chỉ tiêu đề ·
toàn văn · chỉ đặc trưng cấu trúc · + SVD. Đây là mục có giá trị nhất trong báo cáo,
vì nó giải thích *tại sao* cần chính quy hoá, bằng số đo chứ không bằng lời.

**1c. `Ridge` → `LightGBM` → `LightGBM` phân vị.** Phải vượt baseline
`category_median` (MAE 5,54 triệu). Không vượt được nghĩa là văn bản không đóng góp
gì ngoài việc chỉ ra nhóm nghề.

**1d. Kiểm tra độ phủ khoảng.** Khoảng q10–q90 phải chứa khoảng **80%** giá trị thật
trên val. Không đo độ phủ thì khoảng dự đoán chỉ là hai con số trang trí.

**1e. Ghi rõ thiên lệch chọn mẫu.** Bảng tỷ lệ công bố theo ngành ở
[07-bai-toan-luong.md](07-bai-toan-luong.md#thiên-lệch-chọn-mẫu--hạn-chế-không-sửa-được).
Mô hình sẽ ước lượng thấp cho CNTT. Đây là hạn chế cấu trúc, không sửa được bằng
mô hình tốt hơn — phải ghi vào báo cáo.

## Ưu tiên 2 — đo trần nhiễu nhãn

macro-F1 0,6112 nghe thấp, nhưng chưa biết trần thật là bao nhiêu. Cách đo: lấy
150–200 tin bị dự đoán sai trên val, gán tay từng tin vào một trong ba nhóm —
**mô hình sai** · **nhãn gốc sai** · **thật sự nhập nhằng (đa nhãn)**.

Nếu 40% lỗi là nhãn gốc sai thì 0,6112 tương đương khoảng 0,75 trên nhãn sạch.
Đây là con số quý nhất trong toàn bộ báo cáo và không mô hình nào thay thế được nó.

## Ưu tiên 3 — hai đòn bẩy rẻ cho phân lớp (làm trước khi nghĩ đến DL)

**3a. Trọng số theo khối.** SVM chỉ dùng tiêu đề đã đạt 0,5547; thêm cả 6 khối văn bản
còn lại chỉ lên 0,5719 — **mua thêm 0,017**. Đó là dấu hiệu pha loãng: mô tả dài lấn át
tiêu đề ngắn nhưng đậm tín hiệu. Thử nhân trọng số khối tiêu đề ×2, ×3.
Bảng trọng số theo khối ở
[05-dac-trung-tfidf.md §9](05-dac-trung-tfidf.md#9-khối-nào-thật-sự-được-dùng).

**3b. Cắt bớt chiều.** 236.596 chiều trên 33.396 mẫu, tỷ lệ p/n = 7,1. Việc `C` tối ưu
rơi xuống tận 0,02 chính là mô hình đang kêu cứu vì quá nhiều chiều. Quét `min_df`,
`max_features`, `sublinear_tf` — rẻ hơn và nhiều khả năng ăn hơn một vòng quét `C` nữa.

## Ưu tiên 4 — một mốc học sâu

Đây là trục còn thiếu để dự án đủ cả **ML và DL**. Hai lựa chọn theo phần cứng:

| Cách | Chi phí | macro-F1 kỳ vọng |
|---|---|---|
| Fine-tune PhoBERT-base, tiêu đề + mô tả, 256 token | 1–2 giờ có GPU/MPS | 0,66–0,70 |
| Embedding đóng băng + head tuyến tính | ~5 phút | 0,63–0,65 |

> Hai con số cột cuối là **kỳ vọng, chưa đo**. Chúng là ước lượng để lên kế hoạch,
> không phải kết quả, và không được đưa vào [04-results.md](04-results.md) cho tới
> khi chạy thật.

Bắt buộc đo kèm **độ trễ suy luận**, và báo cáo cả hai trục: điểm số *và* chi phí.
Một mô hình hơn 0,05 macro-F1 nhưng chậm gấp 200 lần là một đánh đổi, không phải
một chiến thắng — và trình bày được đánh đổi đó mới là điều làm báo cáo có trọng lượng.

## Ưu tiên 5 — hoàn thiện hệ thống

1. **Hiệu chuẩn xác suất.** `LinearSVC` chỉ cho biên; `predict.py` đang softmax nó và
   gọi là "score". Chạy `svm_calibrated` một lần ở `C` chốt để có confidence thật,
   dùng cho ngưỡng từ chối trả lời và cho gợi ý top-3 (đang đúng 93%).
2. **Bỏ tách từ lúc phục vụ.** Cấu hình chốt không bật `segment`, nhưng `build_frame`
   vẫn gọi `V.segment` **9 lần** cho mỗi tin. Đó là vài giây lãng phí cho mỗi yêu cầu.
   Cho tách từ chạy có điều kiện theo đúng thứ pipeline đã nạp cần.
3. **Chốt mô hình bằng file, không bằng thời gian sửa file.** `_latest_run` đang chọn
   thư mục artifact mới sửa gần nhất — chạm nhầm một file là đổi mô hình đang chạy.
   Thay bằng `artifacts/PRODUCTION.json` ánh xạ task → run_id, chọn có chủ ý.
4. **Sửa cột lệch trong `04-results.md`.** Header khai 11 cột nhưng `HEADLINE`
   ([train.py:89](../src/vietjobs/train.py#L89)) nhét nhiều metric ngăn bằng `|` vào
   một ô, nên mỗi dòng ra 12–15 ô và trình render cắt lặng lẽ hai cột cuối. Sửa ở
   `train.py` cho **dòng mới**; dòng cũ giữ nguyên vì log là append-only.
5. **Một mặt demo mỏng** (FastAPI hoặc Streamlit, ~50 dòng): biến "mấy mô hình" thành
   "một hệ thống" trong mắt người đọc báo cáo.

## Ưu tiên 6 — tài liệu

- [x] `docs/01-data-audit.md` — **xong**
- [x] `docs/02-vietnamese-nlp.md` — **xong**
- [x] `docs/05-dac-trung-tfidf.md` — **xong**
- [x] `docs/nen-tang/` — track nền tảng cho người mới, **xong**
- [ ] Notebook trình bày kết quả. Viết sau cùng, khi mọi con số đã đứng yên.

---

**Quy tắc không đổi:** test chỉ chạm một lần ở cuối · [04-results.md](04-results.md)
chỉ ghi thêm · bước tiền xử lý nào không cải thiện thì gỡ bỏ · **mỗi thay đổi thật
đều phải cập nhật lại tài liệu**. Chi tiết ở [03-protocol.md](03-protocol.md) và
[../CLAUDE.md](../CLAUDE.md).
