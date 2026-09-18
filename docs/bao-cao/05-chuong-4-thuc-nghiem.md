[← Chương 3](04-chuong-3-phuong-phap.md) · [Bảng điều khiển](00-index.md) · [Chương 5 →](06-chuong-5-he-thong.md)

# CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ

> Mọi số trong chương đo trên lược đồ chia hai tầng 8:2 rồi 9:1 (tập `dev` 3.812
> dòng); `test` chưa được đọc.

---

## 4.1. Thiết lập thực nghiệm

### 4.1.1. Cấu hình phần cứng và phần mềm

**Bảng 4.1: Môi trường thực nghiệm**

| Thành phần | Cấu hình |
|---|---|
| Máy chạy các lần `*-s2` | macOS arm64, `device=mps` (ghi trong `artifacts/<run_id>/env.json`) |
| Tăng tốc GPU | Không CUDA |
| Hệ điều hành | macOS |
| Môi trường 1 | Python ≥ 3.10 — dữ liệu, xử lý tiếng Việt, học máy, kiểm thử |
| Môi trường 2 | Python 3.9 — gói `dl/`, do `torch` không còn bản dựng cho macOS Intel sau 2.2.2 |
| Mô hình nền | PhoBERT-base-v2, 135 triệu tham số, đóng băng |

Việc tách **hai môi trường ảo riêng biệt** không phải một lựa chọn thẩm mỹ mà là một
ràng buộc bắt buộc: thư viện `torch` không phát hành bản dựng cho macOS x86_64 sau
phiên bản 2.2.2 và không có bản nào cho Python 3.14. Toàn bộ đường dữ liệu (`dl/text.py`)
được viết để **không phụ thuộc torch**, nhờ đó bộ kiểm thử đường dữ liệu vẫn chạy được
ở môi trường chính, nơi không có torch.

### 4.1.2. Chi phí tính toán

**Bảng 4.2: Chi phí từng khâu**

| Khâu | Chi phí đo được |
|---|---|
| Tách từ 47.707 tin (một lần, ghi vào đĩa) | **780,3 giây**, 0 lỗi |
| Nhúng PhoBERT `train` + `dev` (T1.2, 38.166 tin × 2 họ cột) | khoảng **63–64 tin/giây** |
| Một epoch trên vectơ đã lưu đệm | khoảng 10–15 giây trên máy rảnh |

> **Lưu ý khi đọc cột thời gian trong [04-results.md](../04-results.md):** các lần
> chạy này chồng lấn nhau trên cùng một máy 6 nhân, nên **số giây không so sánh được
> giữa các dòng**.

### 4.1.3. Những gì mỗi lần chạy phải ghi lại

Một lần chạy học sâu kéo dài nhiều
epoch, có thể hỏng giữa chừng, và **lý do hỏng nằm ở đường cong học chứ không nằm ở
con số cuối**. Vì vậy mỗi lần chạy ghi thêm:

**Bảng 4.3: Các tệp bắt buộc của một lần chạy**

| Tệp | Nội dung | Vì sao bắt buộc |
|---|---|---|
| `history.jsonl` | một dòng JSON mỗi epoch: mất mát, độ đo trên dev, số giây | Không có nó thì không phân biệt được thiếu khớp / quá khớp / sai tốc độ học |
| `config.json` | mọi siêu tham số + hạt giống | Một lần chạy không tái lập được thì không có giá trị làm số liệu |
| `env.json` | thiết bị, phiên bản thư viện, encoder | Hai lần chạy trên hai máy có thời gian không so sánh được |
| `best.pt` | trọng số tại epoch tốt nhất trên dev | Sập ở giờ thứ hai thì mất trắng |
| `predictions_<eval>.parquet` | dự đoán từng dòng | Để đọc lỗi thật sau khi chạy xong |

Ngoài ra mỗi lần chạy ghi **đúng một dòng** vào `04-results.md`, và tệp này là
**chỉ-thêm**: một dòng đã viết không bao giờ được sửa. Một thí nghiệm thất bại vẫn là
dữ liệu — nó ghi lại thứ đã thử và không chạy được, và đó chính là thứ ngăn ta thử
lại nó ba tuần sau.

---

## 4.2. Hệ thống mốc cơ sở hai tầng

Đây là đóng góp phương pháp luận chính của chương. Một điểm số không có mốc so sánh
là một con số không đọc được. Đề tài dựng **hai tầng mốc**, mỗi tầng trả lời một câu
hỏi khác nhau.

**Bảng 4.4: Hai tầng mốc cơ sở (dev 3.812 tin)**

| Tầng | Mốc | Trả lời câu hỏi |
|---|---|---|
| 1. Ngây thơ, không mô hình | phân loại: macro-F1 **0,0214** · lương: MAE **5,70 triệu** · RMSE **10,52** · disclosed: acc **0,7078** | "Không cần học gì thì đạt bao nhiêu?" |
| 2. Dò tuyến tính trên chính vectơ | `probe-cat-s2` macro-F1 **0,5898** · `probe-sal-s2` MAE **4,42** · RMSE **8,36** | "Đặc trưng tồi, hay đầu mô hình hỏng?" |

Tầng 2 là tầng ít gặp nhất trong các báo cáo cùng dạng nhưng lại hữu ích nhất khi
gỡ lỗi. Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai —
nếu nó đạt 0,59 trên chính bộ vectơ đó, thì vấn đề chắc chắn **không** nằm ở đặc
trưng. §4.6 cho thấy tầng mốc này đã cứu một lần chạy hỏng như thế nào.

Ngoài ra còn một mốc trần đáng chú ý: **biết trước 100 % nhãn ngành nghề** rồi đoán
trung vị của ngành chỉ đạt MAE **5,57 triệu**, tức chỉ hơn mốc ngây thơ **2,1 %**.
Nếu nhánh hồi quy chỉ về được quanh 5,6 triệu thì nó chưa đọc được gì từ văn bản —
nó chỉ đang đoán trung vị theo một đường vòng.

> **Nguồn số liệu:** tầng 1 và mốc trần — `artifacts/eda/summary.json` khoá `floors`
> (khớp trên `train`, chấm trên `dev`); tầng 2 — dòng `probe-cat-s2`, `probe-sal-s2`
> trong [04-results.md](../04-results.md).

---

## 4.3. Kết quả bài toán phân loại ngành nghề

**Bảng 4.5: Kết quả phân loại ngành nghề (dev 3.812 tin)**

| Lần chạy | Cấu hình | macro-F1 | acc | epoch tốt nhất |
|---|---|---|---|---|
| *mốc ngây thơ* | luôn đoán lớp đa số | *0,0214* | *0,2062* | — |
| `probe-cat-s2` | LogReg trên **cùng** bộ vectơ | 0,5898 | 0,6388 | — |
| **`dl-cat-s2`** | lr 3e-4 + cắt gradient + chuẩn hoá | **0,6025** | 0,6511 | 16/24 |
| `dl-cat-s2-cw` | như trên + cân bằng lớp | 0,5710 | 0,5976 | 15/23 |

> **Nguồn số liệu:** `artifacts/dl-cat-s2/metrics.json`, `artifacts/dl-cat-s2-cw/metrics.json`,
> ba dòng `dl-cat-s2` / `dl-cat-s2-cw` / `probe-cat-s2` trong [04-results.md](../04-results.md),
> và mốc ngây thơ từ `artifacts/eda/summary.json` (T1.1).

`f1_macro_no_junk` — macro-F1 khi loại lớp `nhóm_nghề_khác` (25 dòng) — là **0,6454**
cho `dl-cat-s2` và 0,5958 cho `dl-cat-s2-cw`. Riêng lớp này đã kéo con số tổng xuống
0,043.

Mạng dense chỉ hơn tầng dò tuyến tính **0,0127** macro-F1 trên cùng bộ vectơ (0,6025
so với 0,5898). Phần lớn tín hiệu
mà mô hình tuyến tính khai thác được, mạng phi tuyến cũng khai thác được; năng lực
tính toán thêm chỉ mua được một khoảng cải thiện nhỏ và ổn định, không phải một mức
khác biệt về chất.

### 4.3.1. Nhận xét

**Độ chính xác top-3 là 0,9318** — mô hình đưa đúng ngành vào top ba đáng tin cậy hơn
nhiều so với đoán đúng top-1.

**`--class-weight` là một phép đánh đổi, không phải một cải thiện:** macro-F1 giảm 0,0315 (0,6025 → 0,5710) trong khi balanced accuracy tăng 0,0733
(0,6199 → 0,6931). Nó kéo mô hình về phía các lớp nhỏ, đúng như thiết kế. Chọn cấu
hình nào phụ thuộc vào ứng dụng thật; mặc định để tắt.

Lần chạy phân kỳ `dl-cat-h256` (macro-F1 0,0420) được giữ lại ở §4.6: thiếu chuẩn
hoá và cắt gradient, và chính bài học đó đã dẫn tới cấu hình `dl-cat-s2` ở trên.

---

## 4.4. Kết quả bài toán ước lượng mức lương

**Bảng 4.6: Kết quả ước lượng mức lương (dev 2.698 tin có nhãn lương)**

| Lần chạy | Phương pháp | MAE (triệu) | RMSE (triệu) | R²(log) |
|---|---|---|---|---|
| *mốc* | đoán trung vị 13,0 cho mọi tin | *5,70* | *10,52* | *−0,059* |
| *trần "biết ngành"* | trung vị của ngành, dùng nhãn thật | *5,57* | *10,33* | *−0,022* |
| `probe-sal-s2` | Ridge trên vectơ PhoBERT | 4,42 | 8,36 | 0,466 |
| **`dl-sal-s2`** | dense(256), lr 3e-4, chuẩn hoá | **4,15** | **8,29** | **0,512** |

> **Nguồn số liệu:** `artifacts/dl-sal-s2/metrics.json` và hai dòng `dl-sal-s2` /
> `probe-sal-s2` trong [04-results.md](../04-results.md); hai mốc lấy từ
> `artifacts/eda/summary.json` (T1.1). RMSE của `dl-sal-s2` lấy từ khoá `rmse_trieu`
> trong `metrics.json`; RMSE hai mốc lấy từ khoá `rmse` trong `summary.json`; RMSE của
> `probe-sal-s2` tính lại bằng đúng cấu hình của lần chạy đó (Ridge alpha 1,0, cùng bộ
> vectơ), cho lại đúng MAE 4,42 và R² 0,466.

### 4.4.1. Nhận xét

Nhánh hồi quy **vượt mốc 1,55 triệu (−27,2 %)**: R² trên thang logarit đi từ **âm**
lên **0,512**, nghĩa là mô hình đọc được tín hiệu lương từ văn bản mà nhãn ngành nghề
không cung cấp (eta² = 0,032, đo trên tệp gốc).

Tầng dò tuyến tính `probe-sal-s2` đạt R² **0,466**, đã gần bằng mạng dense (0,512) và
rõ ràng vượt mốc: phần lớn tín hiệu lương nằm trong vectơ ở dạng tuyến tính.

RMSE của `dl-sal-s2` là **8,29 triệu**, gấp đôi MAE (4,15). RMSE bình phương sai số,
nên một số ít lỗi rất lớn kéo nó lên. Tách theo mức lương thật trên
`predictions_dev.parquet`: **2.571** tin dưới 30 triệu có RMSE **4,49** (MAE 3,21);
**127** tin từ 30 triệu trở lên (4,7 %) có RMSE **32,43** (MAE 23,13).

---

## 4.5. Phân tích lỗi

### 4.5.1. Lỗi của bài toán phân loại

> ⛔ **CHƯA CÓ SỐ LIỆU** (T2.2, T2.3)

### 4.5.2. Lỗi của bài toán hồi quy

> ⛔ **CHƯA CÓ SỐ LIỆU** (T2.1)

---

## 4.6. Lần chạy thất bại đầu tiên — giữ lại làm bằng chứng

Hai lần chạy đầu (`dl-cat-h256`, `dl-cat-h256-cw`, đo trên lược đồ chia cũ trước
2026-09-09) cho macro-F1 **0,042** — thấp hơn
cả việc thay vectơ PhoBERT bằng số ngẫu nhiên. Tệp `history.jsonl` cho biết lý do
trong bốn dòng:

```
epoch 1  loss 2.63     val_macro_f1 0.028
epoch 2  loss 2.57     val_macro_f1 0.033
epoch 3  loss 2.65     val_macro_f1 0.042
epoch 4  loss 321.6    val_macro_f1 0.011   ← nổ
epoch 5  loss 1941.8   val_macro_f1 0.019
```

Vòng huấn luyện **phân kỳ ở epoch 4**. Hai nguyên nhân, cả hai đều đo được:

1. **Vectơ PhoBERT dị hướng.** Cosin giữa hai tin bất kỳ có trung vị **0,896**
   (p05 = 0,838 · p95 = 0,943) — mọi tin nằm trong một hình nón hẹp. `LayerNorm`
   chuẩn hoá theo từng mẫu nên không gỡ được hướng chung đó.
2. **lr 1e-3 quá lớn** với loại đầu vào như vậy, lại không có cắt gradient.

Cách sửa: chuẩn hoá theo từng chiều bằng thống kê của `train` (lưu thành `scaler.npz`
cạnh mô hình, vì đường suy luận phải dùng đúng những con số đó), cắt chuẩn gradient ở
1,0, hạ lr xuống 3e-4 — cấu hình của `dl-cat-s2` (macro-F1 **0,6025**, §4.3).

**Bài học đáng giá hơn con số:** chính tầng dò tuyến tính — hồi quy logistic trên
cùng bộ vectơ — là thứ phân biệt được "đặc trưng tồi" với "đầu mô hình
hỏng". Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai; nếu
nó đạt gần 0,59 thì vấn đề chắc chắn không nằm ở đặc trưng. Đây là lý do tầng mốc thứ hai
ở §4.2 tồn tại.

---

## 4.7. Tổng hợp: đối chiếu kết quả với mốc

**Bảng 4.7: Tổng hợp đối chiếu (dev)**

| Bài toán | Mốc ngây thơ | Mốc dò tuyến tính | Kết quả học sâu | Kết luận |
|---|---|---|---|---|
| Phân loại (macro-F1) | 0,0214 | 0,5898 | **0,6025** | Hơn dò tuyến tính 0,0127 |
| Phân loại (top-3) | — | 0,9258 | **0,9318** | Hơn dò tuyến tính 0,0060 |
| Lương (MAE, triệu) | 5,70 | 4,42 | **4,15** | **Vượt mốc ngây thơ 27,2 %** |
| Lương (RMSE, triệu) | 10,52 | 8,36 | **8,29** | Vượt mốc ngây thơ 21,2 %; chỉ hơn dò tuyến tính 0,07 |
| Lương (R² log) | −0,059 | 0,466 | **0,512** | Từ âm lên dương — đọc được tín hiệu thật |

> **Nguồn số liệu:** các dòng `*-s2` trong [04-results.md](../04-results.md) ·
> [06-baseline-dl.md](../06-baseline-dl.md) §5 · `artifacts/eda/summary.json` khoá `floors`.

---

## 4.8. Đánh giá chéo trên tập dữ liệu ngoài

> ⛔ **Số tham khảo** — bảng ánh xạ 60 → 16 còn ở trạng thái `draft` (T7.1, T7.4).

Mọi kết quả ở §4.3 đều đo trên tập `dev` chia ra từ cùng một bộ dữ liệu với tập
huấn luyện. Chúng chưa trả lời được câu hỏi: mô hình học được *nghề*, hay học
*cách viết* của các tin trong `VietJobs.csv`? Để trả lời, mô hình `dl-cat-s2` được
chấm trên **VietJobs-37K** — bộ 37.274 tin tuyển dụng tiếng Việt do một nhóm khác
thu thập (1/2025 – 4/2026), gán nhãn theo hệ 60 nhãn đa nhãn, kèm 1.000 tin
(Gold-1000) do người soát lại. Không có tin nào của bộ này được dùng để huấn luyện
hay chọn mô hình.

Ba bước chuẩn bị, chi tiết ở [10-danh-gia-ngoai.md](../10-danh-gia-ngoai.md):

1. **Khử tin trùng** với ba tập chia của ta: cùng tiêu đề (đã gập dấu) và Jaccard
   tập từ của mô tả ≥ 0,5. Loại 19 / 1.000 tin Gold (1,9 %) và 69 / 3.778 tin test
   (1,83 %).
2. **Ánh xạ 60 nhãn về 16 lớp**; bốn nhãn không ánh xạ được (chức danh quản lý,
   thư viện, NGO) bị loại.
3. **Hai quy ước chấm**: *strict* chỉ trên tin có đúng một nhãn gốc (chấm như
   bình thường, kèm khoảng tin cậy bootstrap 1.000 lần); *lenient* trên mọi tin,
   tính đúng nếu lớp dự đoán thuộc tập lớp đích.

**Bảng 4.8: Mô hình `dl-cat-s2` trên VietJobs-37K, sau khử trùng và ánh xạ**

| Tập | n chấm | strict n | strict macro-F1 [KTC 95 %] | strict acc | strict top-3 | lenient trúng |
|---|---|---|---|---|---|---|
| Gold-1000 (người soát) | 977 | 529 | **0,3977** [0,353; 0,443] | 0,5104 | 0,8204 | 0,6080 |
| test 37K (nhãn máy) | 3.687 | 2.053 | 0,4525 [0,421; 0,479] | 0,5329 | 0,8315 | 0,6138 |
| *`dev` nội bộ (§4.3, đối chiếu)* | 3.812 | 3.812 | *0,6025* | *0,6511* | *0,9318* | — |

Nhận xét:

- Mô hình **có tổng quát hóa** sang dữ liệu của người khác, nhưng mất **0,205**
  macro-F1 trên Gold — gấp gần năm lần bề rộng khoảng tin cậy. Top-3 giữ được tốt
  hơn (0,93 → 0,82): nghề đúng vẫn thường nằm trong ba lựa chọn đầu.
- Điểm trên nhãn máy (0,453) **cao hơn** trên nhãn người (0,398): nhãn máy của bộ
  ngoài được gán bằng từ khóa, mô hình của ta cũng dựa nhiều vào từ vựng, nên hai
  bên đồng ý với nhau hơn là với người soát. Đó là lý do lấy Gold làm số chính.
- Phân bố lớp dự đoán lệch rõ: 28,3 % tin Gold bị gán vào lớp nhân sự – hành chính,
  còn lớp công nghệ thông tin chỉ được dự đoán 5 lần dù Gold có ít nhất 67 tin
  CNTT.

Giới hạn phải nói rõ: nhãn của bộ ngoài (trừ Gold) là nhãn máy; bảng ánh xạ do
nhóm tự xây; bộ ngoài **không có trường lương**, nên mục này chỉ đánh giá bài phân
loại.

> **Nguồn số liệu:** hai dòng `dl-cat-s2-ext37k-gold` và `dl-cat-s2-ext37k-test`
> trong [04-results.md](../04-results.md) ·
> `artifacts/dl-cat-s2/external_37k/{gold,test}/metrics.json` và `dedup_report.json` ·
> [10-danh-gia-ngoai.md](../10-danh-gia-ngoai.md).

---

[← Chương 3](04-chuong-3-phuong-phap.md) · [Chương 5 →](06-chuong-5-he-thong.md)
