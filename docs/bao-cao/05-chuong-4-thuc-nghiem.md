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
| Thiết bị huấn luyện | macOS arm64, **`device=cpu`** cho mọi lần chạy `*-cpu-0920` — mặc định từ 2026-09-20 vì `mps` không tái lập được (§4.3.2). Các lần chạy `*-s2` cũ dùng `device=mps` (ghi trong `artifacts/<run_id>/env.json`) |
| Tăng tốc GPU | Không CUDA |
| Hệ điều hành | macOS |
| Môi trường 1 | Python ≥ 3.10 — dữ liệu, xử lý tiếng Việt, độ đo, kiểm thử |
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
| 2. Dò tuyến tính trên chính vectơ | `probe-cat-0920` macro-F1 **0,5898** · `probe-sal-0920` MAE **4,42** · RMSE **8,36** | "Đặc trưng tồi, hay đầu mô hình hỏng?" |

Tầng 2 là tầng ít gặp nhất trong các báo cáo cùng dạng nhưng lại hữu ích nhất khi
gỡ lỗi. Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai —
nếu nó đạt 0,59 trên chính bộ vectơ đó, thì vấn đề chắc chắn **không** nằm ở đặc
trưng. §4.6 cho thấy tầng mốc này đã cứu một lần chạy hỏng như thế nào.

Ngoài ra còn một mốc trần đáng chú ý: **biết trước 100 % nhãn ngành nghề** rồi đoán
trung vị của ngành chỉ đạt MAE **5,57 triệu**, tức chỉ hơn mốc ngây thơ **2,1 %**.
Nếu nhánh hồi quy chỉ về được quanh 5,6 triệu thì nó chưa đọc được gì từ văn bản —
nó chỉ đang đoán trung vị theo một đường vòng.

> **Nguồn số liệu:** tầng 1 và mốc trần — `artifacts/eda/summary.json` khoá `floors`
> (khớp trên `train`, chấm trên `dev`); tầng 2 — dòng `probe-cat-0920`, `probe-sal-0920`
> trong [04-results.md](../04-results.md).

---

## 4.3. Kết quả bài toán phân loại ngành nghề

Đây là **mô hình nền thứ nhất** của đề tài. Toàn bộ điều kiện của lần chạy được ghi
lại ngay dưới đây, để đọc chương này là đủ dựng lại lần chạy, không cần mở thêm tệp
nào khác.

**Bảng 4.5a: Hồ sơ lần chạy `dl-cat-ce-cpu-0920`**

| Hạng mục | Giá trị |
|---|---|
| Lệnh | `python -m vietjobs.dl.train_dl --task category --run-id dl-cat-ce-cpu-0920` (biến `PYTHONPATH=src`) |
| Môi trường | Môi trường 2 — Python 3.9.6, `torch` 2.2.2, `transformers` 4.46.3, `numpy` 1.26.4, `pandas` 2.3.3, `scikit-learn` 1.6.1, **`device=cpu`**, macOS arm64 |
| Ngày chạy | 2026-09-20 |
| Hàm mất mát | `CrossEntropyLoss`, không cân bằng lớp |
| Bộ mã hoá | `vinai/phobert-base-v2`, **đóng băng**; vectơ nạp lại từ đệm `train`/`dev` họ `raw`, `len256` |
| Cột đọc vào | `job_title_seg` · `description_seg` · `requirements_seg` |
| Kiến trúc đầu ra | LayerNorm → 768→256 → GELU → dropout 0,3 → 256→128 → GELU → dropout → 128→**16 lớp** (§3.7.1) |
| Siêu tham số | lr 3e-4 · batch 256 · weight decay 0,01 · cắt gradient 1,0 · chuẩn hoá đầu vào · tối đa 40 epoch, dừng sớm sau 8 epoch không cải thiện · hạt giống 42 |
| Dữ liệu | khớp trên `train` 34.354 tin; chấm trên `dev` 3.812 tin; `test` không đụng tới |
| Epoch tốt nhất | 11/19 |
| Tái lập | chạy ba lần cho cùng một kết quả tới chữ số cuối (§4.3.2) |

> **Nguồn số liệu:** `artifacts/dl-cat-ce-cpu-0920/config.json`,
> `artifacts/dl-cat-ce-cpu-0920/env.json`, `artifacts/dl-cat-ce-cpu-0920/metrics.json`,
> `data/processed/manifest.json`.

**Bảng 4.5: Kết quả phân loại ngành nghề (dev 3.812 tin)**

| Lần chạy | Hàm mất mát | macro-F1 | F1 | acc | epoch tốt nhất |
|---|---|---|---|---|---|
| *mốc ngây thơ* | luôn đoán lớp đa số | *0,0214* | *0,0705* | *0,2062* | — |
| `probe-cat-0920` | — (LogReg trên **cùng** bộ vectơ) | 0,5898 | 0,6271 | 0,6388 | — |
| **`dl-cat-ce-cpu-0920`** | **CrossEntropyLoss** (Dense) | **0,6030** | **0,6413** | 0,6501 | 11/19 |
| `dl-cat-focal-cpu-0920` | FocalLoss (γ=2), Dense | 0,5972 | 0,6343 | 0,6427 | 11/19 |
| `dl-cat-rnn-ce` | CrossEntropyLoss (Bi-GRU‖Bi-LSTM→CNN, T8) | 0,6046 | 0,6567 | 0,6655 | 8/16 |
| `dl-cat-rnn-focal` | FocalLoss (γ=2), cùng head | 0,6033 | 0,6451 | 0,6529 | 9/17 |

Hai dòng `*-cpu-0920` dùng cùng kiến trúc và cùng siêu tham số của Bảng 4.5a,
trên `device=cpu`, và **tái lập được từng chữ số**; §4.3.2 giải thích vì sao
điều đó phải được nói ra. Hai dòng `dl-cat-rnn-*` chạy trên GPU Colab
(`device=cuda`) và **không** tái lập được từng chữ số cùng lý do `mps` không
tái lập được ở §4.3.2 — xem §4.4.2 cho phần đối chiếu có bootstrap. `F1` là F1
trung bình có trọng số theo số mẫu mỗi lớp; `macro-F1` cho mọi lớp trọng số
bằng nhau và vẫn là chỉ số chọn mô hình.

> **Nguồn số liệu:** `artifacts/dl-cat-ce-cpu-0920/metrics.json`,
> `artifacts/dl-cat-focal-cpu-0920/metrics.json`, các dòng `dl-cat-ce-cpu-0920` /
> `dl-cat-focal-cpu-0920` / `probe-cat-0920` trong [04-results.md](../04-results.md),
> mốc ngây thơ từ `artifacts/eda/summary.json` (T1.1), và
> `data/eda_xlsx/ket_qua_chay.xlsx` sheet `01_Phan_Lop` (Rule 13).

`f1_macro_no_junk` — macro-F1 khi loại lớp `nhóm_nghề_khác` (25 dòng) — là **0,6458**
cho `dl-cat-ce-cpu-0920` và 0,6395 cho `dl-cat-focal-cpu-0920`. Riêng lớp này đã kéo
con số tổng xuống 0,043.

Mạng dense chỉ hơn tầng dò tuyến tính **0,0132** macro-F1 trên cùng bộ vectơ (0,6030
so với 0,5898). Phần lớn tín hiệu
mà mô hình tuyến tính khai thác được, mạng phi tuyến cũng khai thác được; năng lực
tính toán thêm chỉ mua được một khoảng cải thiện nhỏ và ổn định, không phải một mức
khác biệt về chất.

### 4.3.1. Nhận xét

**Focal loss thua CrossEntropy 0,0058 macro-F1** (0,5972 so với 0,6030).
`FocalLoss` (`dl/focal_loss.py`, bản itakurah/focal-loss-pytorch) nhân cross-entropy từng mẫu với `(1-p_t)^γ`, tức
cân theo *độ khó* của mẫu thay vì *độ hiếm* của lớp — đúng cơ chế được kỳ vọng
sẽ giúp trên bộ dữ liệu lệch 27:1. Nó không giúp. Khoảng cách 0,0058 lớn hơn
biên nhiễu 0,003 đo ở §4.3.2, nên đây là kết luận thật chứ không phải dao động
giữa hai lần chạy. Cả hai lần chạy dùng đúng cùng kiến trúc, cùng siêu tham số,
cùng hạt giống và cùng thiết bị; khác biệt duy nhất là hàm mất mát.

**`--class-weight` cũng là một phép đánh đổi, không phải một cải thiện:**
macro-F1 giảm 0,0315 (`dl-cat-s2` 0,6025 → `dl-cat-s2-cw` 0,5710). Nó kéo mô
hình về phía các lớp nhỏ, đúng như thiết kế; mặc định để tắt. Hai lần chạy này
đo trên `device=mps` ngày 15/09 nên từng con số riêng lẻ không tái lập được,
nhưng khoảng cách 0,0315 vượt xa biên nhiễu nên kết luận vẫn đứng.

Lần chạy phân kỳ `dl-cat-h256` (macro-F1 0,0420) được giữ lại ở §4.6: thiếu chuẩn
hoá và cắt gradient, và chính bài học đó đã dẫn tới cấu hình ở trên.

### 4.3.2. Tái lập: vì sao mọi lần chạy đều đặt trên `cpu`

Ngày 2026-09-20, khi chạy lại mô hình nền để đo bằng bộ chỉ số mới,
`dl-cat-s2` (chạy 15/09, macro-F1 0,6025) **không tái lập được**: cùng hạt
giống, cùng dữ liệu, cùng mã nguồn, kết quả ra 0,6013.

Truy nguyên bằng cách dựng một cây làm việc git tại đúng commit `12a847c1` của
lần chạy cũ rồi chạy **mã nguồn nguyên bản** ngày hôm đó — cũng ra 0,6013. Vậy
nguyên nhân không nằm ở thay đổi mã. Loại trừ tiếp: bộ đệm vectơ (không đổi từ
13/09), tệp chia và `manifest.json` (không đổi từ 10/09), kiến trúc, hạt giống,
`torch` 2.2.2, cùng chuỗi `platform`.

Nguyên nhân là **thiết bị**:

| Thiết bị | macro-F1 qua các lần chạy cùng cấu hình | Thời gian |
|---|---|---|
| `mps` | 0,6012 · 0,6013 · 0,6013 · 0,6013 · 0,6025 · 0,6041 · 0,6041 · 0,6041 | 13–15 giây |
| `cpu` | 0,6030 · 0,6030 · 0,6030 (giống tới chữ số cuối) | 12–16 giây |

Biên dao động của `mps` là **0,003 macro-F1**, lớn hơn phần lớn khác biệt mà
chương này muốn đo. `cpu` cho đúng một số mỗi lần và **không chậm hơn**: mạng
dense chỉ 768→256→128→16, quá nhỏ để GPU có lợi ích. Từ đó `train_dl.py` mặc
định `--device cpu`.

Ba hệ quả ràng buộc cách đọc mọi bảng trong chương:

1. Các lần chạy `*-cpu-0920` tái lập được; các lần chạy `-s2` (trên `mps`) thì
   không, và được giữ lại như dữ liệu lịch sử.
2. Trên các lần chạy `mps` cũ, **mọi chênh lệch dưới 0,003 macro-F1 là nhiễu**.
   Hai kết luận của §4.3.1 vẫn đứng vì vượt xa biên đó: `--class-weight`
   (−0,0315) và focal loss (−0,0058).
3. `env.json` từ nay ghi thêm `numpy`, `pandas`, `scikit-learn` — ba thư viện
   mà việc không ghi lại đã khiến lần truy nguyên này dài hơn cần thiết.

Nhánh hồi quy không lộ vấn đề theo cách đó: `dl-sal-s2` tái lập trên `mps` tới
chữ số cuối (MAE 4,14868613775178). Nhưng chuyển sang `cpu` vẫn đổi kết quả
(4,13 triệu, hội tụ ở epoch 19 thay vì 26) — hai thiết bị là hai đường số học
khác nhau, nên §4.4 dùng lần chạy `cpu` cho nhất quán.

> **Nguồn số liệu:** `artifacts/dl-cat-ce-cpu-0920/`, `artifacts/dl-cat-s2/`,
> `data/eda_xlsx/ket_qua_chay.xlsx` sheet `00_Runs` (cột `device`), và
> AGENTS.md Rule 13 điều 5.

---

## 4.4. Kết quả bài toán ước lượng mức lương

**Mô hình nền thứ hai.** Cùng kiến trúc, cùng siêu tham số với §4.3 — khác đúng ba
điểm: đầu ra một giá trị thay vì 16 lớp, mục tiêu là `log1p(salary_mid)`, và vectơ
lấy từ họ `masked` để mô hình không đọc được chính con số lương in trong tin
(quy tắc chống rò rỉ, §3.4).

**Bảng 4.6a: Hồ sơ lần chạy `dl-sal-cpu-0920`**

| Hạng mục | Giá trị |
|---|---|
| Lệnh | `python -m vietjobs.dl.train_dl --task salary --run-id dl-sal-cpu-0920` (biến `PYTHONPATH=src`) |
| Môi trường | Môi trường 2 — Python 3.9.6, `torch` 2.2.2, `transformers` 4.46.3, `numpy` 1.26.4, `pandas` 2.3.3, `scikit-learn` 1.6.1, **`device=cpu`**, macOS arm64 |
| Ngày chạy | 2026-09-20 |
| Hàm mất mát | `SmoothL1Loss` (Huber) trên `log1p(salary_mid)` |
| Bộ mã hoá | `vinai/phobert-base-v2`, **đóng băng**; vectơ nạp lại từ đệm `train`/`dev` họ `masked`, `len256` |
| Cột đọc vào | `job_title_masked_seg` · `description_masked_seg` · `requirements_masked_seg` |
| Kiến trúc đầu ra | LayerNorm → 768→256 → GELU → dropout 0,3 → 256→128 → GELU → dropout → 128→**1** (mất mát Huber trên `log1p`) |
| Siêu tham số | lr 3e-4 · batch 256 · weight decay 0,01 · cắt gradient 1,0 · chuẩn hoá đầu vào · tối đa 40 epoch, dừng sớm sau 8 epoch không cải thiện · hạt giống 42 |
| Dữ liệu | khớp trên `train` 34.354 tin; chấm trên **2.698** tin `dev` có công bố lương; `test` không đụng tới |
| Epoch tốt nhất | 19/27 |

> **Nguồn số liệu:** `artifacts/dl-sal-cpu-0920/config.json`, `artifacts/dl-sal-cpu-0920/env.json`,
> `artifacts/dl-sal-s2/metrics.json`, `data/processed/manifest.json`.

**Bảng 4.6: Kết quả ước lượng mức lương (dev 2.698 tin có nhãn lương)**

| Lần chạy | Phương pháp | MAE (triệu) | RMSE (triệu) | R²(log) |
|---|---|---|---|---|
| *mốc* | đoán trung vị 13,0 cho mọi tin | *5,70* | *10,52* | *−0,059* |
| *trần "biết ngành"* | trung vị của ngành, dùng nhãn thật | *5,57* | *10,33* | *−0,022* |
| `probe-sal-0920` | Ridge trên vectơ PhoBERT | 4,42 | 8,36 | 0,466 |
| **`dl-sal-cpu-0920`** | dense(256), lr 3e-4, chuẩn hoá, SmoothL1Loss | **4,13** | **8,13** | **0,515** |
| `dl-sal-rnn` | Bi-GRU‖Bi-LSTM→CNN (T8), cùng hàm mất mát | **3,92** | 8,14 | **0,562** |

> **Nguồn số liệu:** `artifacts/dl-sal-cpu-0920/metrics.json` và hai dòng
> `dl-sal-cpu-0920` / `probe-sal-0920` trong [04-results.md](../04-results.md); hai mốc
> lấy từ `artifacts/eda/summary.json` (T1.1). RMSE của lần chạy dense lấy từ khoá
> `rmse_trieu` trong `metrics.json`; RMSE hai mốc lấy từ khoá `rmse` trong
> `summary.json`; RMSE của `probe-sal-0920` lấy từ chính dòng log của lần chạy đó.
> Bảng tổng hợp: `data/eda_xlsx/ket_qua_chay.xlsx` sheet `02_Luong` (Rule 13).

### 4.4.1. Nhận xét

Nhánh hồi quy **vượt mốc 1,57 triệu (−27,5 %)**: R² trên thang logarit đi từ **âm**
lên **0,515**, nghĩa là mô hình đọc được tín hiệu lương từ văn bản mà nhãn ngành nghề
không cung cấp (eta² = 0,032, đo trên tệp gốc).

Tầng dò tuyến tính `probe-sal-0920` đạt R² **0,466**, đã gần bằng mạng dense (0,515) và
rõ ràng vượt mốc: phần lớn tín hiệu lương nằm trong vectơ ở dạng tuyến tính.

RMSE của `dl-sal-cpu-0920` là **8,13 triệu**, gần gấp đôi MAE (4,13). RMSE bình phương sai số,
nên một số ít lỗi rất lớn kéo nó lên. Tách theo mức lương thật trên
`predictions_dev.parquet`: **2.571** tin dưới 30 triệu có RMSE **4,49** (MAE 3,21);
**127** tin từ 30 triệu trở lên (4,7 %) có RMSE **32,43** (MAE 23,13).

### 4.4.2. T8 — head đọc token thay vì vectơ đã gộp

Hai mô hình nền ở Bảng 4.5/4.6 đọc **vectơ đã gộp trung bình**: PhoBERT xuất
`[256, 768]` mỗi tin, cache nén còn `[768]` trước khi vào Dense. T8 thay Dense
bằng `BiGruLstmCnn` (SpatialDropout → Bi-GRU ‖ Bi-LSTM → Conv1d → pooling có
mặt nạ → dense), đọc thẳng `[256, 768]`, giữ thứ tự và phủ định mà phép trung
bình xoá mất — kiến trúc theo Tran–Vo–Luu (2022), `--hidden 100
--conv-channels 50` (1,30 M tham số). Chạy trên GPU Colab T4 vì một epoch quá
chậm để vừa một phiên trên `cpu`/`mps` của máy chính (23 phút và 8,5 phút,
đo được, so với ~4,4 phút trên T4) — nghĩa là **không tái lập được từng chữ
số** như các lần chạy `cpu` ở §4.3.2, cùng lý do `mps` không tái lập được ở đó.

**So cặp trên cùng dòng `dev`, 1.000 mẫu bootstrap** (`evaluate.bootstrap_indices`
+ `paired_delta`, Δ = RNN trừ Dense):

| So sánh | Δ macro-F1 | KTC 95% | P(RNN thắng) | Kết luận |
|---|---|---|---|---|
| `dl-cat-rnn-ce` so `dl-cat-ce-cpu-0920` | +0,0017 | [−0,017, +0,019] | 0,595 | không phân biệt được với nhiễu |
| Cùng cặp, Δ accuracy | +0,0159 | [+0,004, +0,028] | **0,994** | **thật** — nhưng ở acc, không macro-F1 |
| `dl-sal-rnn` so `dl-sal-cpu-0920`, Δ MAE | −0,22 triệu | [+0,10, +0,34]* | **1,000** | **thật, −5,1 %** |

*KTC ghi theo chiều Dense − RNN; dương nghĩa RNN tốt hơn.

Phân lớp: macro-F1 gần như đứng yên trong khi accuracy nhích thật — RNN đúng
thêm ở các lớp lớn (8/16 ngành tốt hơn, lớn nhất +0,056 ở `kinh_doanh…`
n=786) mà không đều trên toàn bộ 16 lớp (thua nặng nhất −0,095 ở
`nông_nghiệp…`, n=37 — mẫu quá nhỏ để tin). Đúng như phép dò tuyến tính đã chỉ
ra ở §4.3: phần lớn tín hiệu đã nằm sẵn trong 768 chiều gộp, đọc thêm token
mua được rất ít trên bài phân lớp.

Lương: MAE giảm thật (3,92 so với 4,13 triệu) nhưng RMSE gần như không đổi
(8,14 so với 8,13) vì cải thiện tập trung ở phần thân phân phối — tách theo
mức lương thật: dưới 30 triệu MAE 3,09 so với 3,23 (n=2.571), từ 30 triệu trở
lên MAE 20,63 so với 22,45 nhưng RMSE gần như bằng nhau (31,29 so với 31,46,
n=127). Head đọc token không sửa được vấn đề đuôi phải nêu ở §4.5.2, chỉ khá
hơn ở phần thân.

⛔ **Chưa có số liệu** cho ablation nhánh T8.5 (`--branches gru` / `lstm`
riêng, so với `dl-cat-rnn-ce` 0,6046 cả hai nhánh) — xem `TASKS.md` T8.5.

> **Nguồn số liệu:** `artifacts/dl-cat-rnn-ce/`, `artifacts/dl-cat-rnn-focal/`,
> `artifacts/dl-sal-rnn/metrics.json`, `history.jsonl`, `predictions_dev.parquet`;
> bốn dòng `dl-cat-rnn-*` / `dl-sal-rnn` trong [04-results.md](../04-results.md);
> `data/eda_xlsx/ket_qua_chay.xlsx` sheet `01_Phan_Lop`, `02_Luong` (Rule 13).

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
1,0, hạ lr xuống 3e-4 — cấu hình của `dl-cat-ce-cpu-0920` (macro-F1 **0,6030**, §4.3).

**Bài học đáng giá hơn con số:** chính tầng dò tuyến tính — hồi quy logistic trên
cùng bộ vectơ — là thứ phân biệt được "đặc trưng tồi" với "đầu mô hình
hỏng". Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai; nếu
nó đạt gần 0,59 thì vấn đề chắc chắn không nằm ở đặc trưng. Đây là lý do tầng mốc thứ hai
ở §4.2 tồn tại.

---

## 4.7. Tổng hợp: đối chiếu kết quả với mốc

**Bảng 4.7: Tổng hợp đối chiếu (dev)**

| Bài toán | Mốc ngây thơ | Mốc dò tuyến tính | Dense | Bi-GRU‖Bi-LSTM→CNN (T8) | Kết luận |
|---|---|---|---|---|---|
| Phân loại (macro-F1) | 0,0214 | 0,5898 | 0,6030 | **0,6046** | RNN không phân biệt được với Dense (P=0,595, bootstrap ghép cặp) |
| Phân loại (F1 có trọng số) | 0,0705 | 0,6271 | 0,6413 | **0,6567** | |
| Phân loại (accuracy) | 0,2062 | 0,6388 | 0,6501 | **0,6655** | RNN hơn Dense thật (P=0,994) — ở các lớp lớn, không đều trên 16 lớp |
| Lương (MAE, triệu) | 5,70 | 4,42 | 4,13 | **3,92** | RNN hơn Dense thật (P=1,000), **−31,2 % so mốc ngây thơ** |
| Lương (RMSE, triệu) | 10,52 | 8,36 | 8,13 | 8,14 | RNN không phân biệt được với Dense — cải thiện chỉ ở phần thân, không ở đuôi |
| Lương (R² log) | −0,059 | 0,466 | 0,515 | **0,562** | |

> **Nguồn số liệu:** các dòng `*-cpu-0920`, `probe-*-0920`, `dl-cat-rnn-*`, `dl-sal-rnn`
> trong [04-results.md](../04-results.md) · [06-baseline-dl.md](../06-baseline-dl.md) §5.1,
> §5.2, §5.6 (số bootstrap) · `artifacts/eda/summary.json` khoá `floors`.

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

| Tập | n chấm | strict n | strict macro-F1 [KTC 95 %] | strict F1 | strict acc | lenient trúng |
|---|---|---|---|---|---|---|
| Gold-1000 (người soát) | 977 | 529 | **0,3977** [0,353; 0,443] | 0,5094 | 0,5104 | 0,6080 |
| test 37K (nhãn máy) | 3.687 | 2.053 | 0,4525 [0,421; 0,479] | 0,5393 | 0,5329 | 0,6138 |
| *`dev` nội bộ (§4.3, đối chiếu)* | 3.812 | 3.812 | *0,6025* | *0,6423* | *0,6511* | — |

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
