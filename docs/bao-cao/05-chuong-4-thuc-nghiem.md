[← Chương 3](04-chuong-3-phuong-phap.md) · [Bảng điều khiển](00-index.md) · [Chương 5 →](06-chuong-5-he-thong.md)

# CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ

> ⛔ **CHƯA CÓ SỐ LIỆU CUỐI CÙNG.** Toàn bộ kết quả trong chương này đo trên **lược
> đồ chia dữ liệu phiên bản 1** (70/15/15, tập dev 7.159 dòng). Ngày 2026-09-09 lược
> đồ đổi sang hai tầng 8:2 rồi 9:1 (tập dev còn 3.812 dòng), nên **không con số nào
> dưới đây so sánh được với con số đo từ nay về sau**. Chúng được giữ lại vì hai lý
> do: mạch lập luận và các bài học chẩn đoán vẫn đúng, và chúng là mốc để đối chiếu
> khi các lần chạy mới hoàn tất. Danh sách việc phải chạy lại ở
> [09-lo-trinh.md](../09-lo-trinh.md).
>
> Mỗi bảng bị ảnh hưởng đều được đánh dấu **(lược đồ v1)**.

---

## 4.1. Thiết lập thực nghiệm

### 4.1.1. Cấu hình phần cứng và phần mềm

**Bảng 4.1: Môi trường thực nghiệm**

| Thành phần | Cấu hình |
|---|---|
| Kiến trúc | Intel x86_64, 6 nhân |
| Tăng tốc GPU | **Không có** — không CUDA, không MPS |
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
| Nhúng PhoBERT ~33 nghìn tin trên CPU | ~20 phút, khoảng **30 tin/giây** |
| Một epoch trên vectơ đã lưu đệm | khoảng 10–15 giây trên máy rảnh |

> **Lưu ý khi đọc cột thời gian trong [04-results.md](../04-results.md):** các lần
> chạy này chồng lấn nhau trên cùng một máy 6 nhân, nên **số giây không so sánh được
> giữa các dòng**.

### 4.1.3. Những gì mỗi lần chạy phải ghi lại

Một lần chạy học máy kết thúc bằng một con số. Một lần chạy học sâu kéo dài nhiều
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

## 4.2. Hệ thống mốc cơ sở ba tầng

Đây là đóng góp phương pháp luận chính của chương. Một điểm số không có mốc so sánh
là một con số không đọc được. Đề tài dựng **ba tầng mốc**, mỗi tầng trả lời một câu
hỏi khác nhau.

**Bảng 4.4: Ba tầng mốc cơ sở**

| Tầng | Mốc | Trả lời câu hỏi |
|---|---|---|
| 1. Ngây thơ, không mô hình | phân loại: macro-F1 **0,0210** · lương: MAE **5,86 triệu** · disclosed: acc **0,7117** | "Không cần học gì thì đạt bao nhiêu?" |
| 2. Dò tuyến tính trên chính vectơ | `probe-cat` macro-F1 **0,5867** · `probe-sal` MAE **6,60** | "Đặc trưng tồi, hay đầu mô hình hỏng?" |
| 3. Học máy truyền thống | TF-IDF + LinearSVC: dev **0,6050 ± 0,0071**, test **0,6112** | "Mô hình mới có hơn cách làm cũ không?" |

Tầng 2 là tầng ít gặp nhất trong các báo cáo cùng dạng nhưng lại hữu ích nhất khi
gỡ lỗi. Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai —
nếu nó đạt 0,59 trên chính bộ vectơ đó, thì vấn đề chắc chắn **không** nằm ở đặc
trưng. §4.6 cho thấy tầng mốc này đã cứu một lần chạy hỏng như thế nào.

Ngoài ra còn một mốc trần đáng chú ý: **biết trước 100 % nhãn ngành nghề** rồi đoán
trung vị của ngành chỉ đạt MAE **5,75 triệu**, tức chỉ hơn mốc ngây thơ **1,8 %**.
Nếu nhánh hồi quy chỉ về được quanh 5,8 triệu thì nó chưa đọc được gì từ văn bản —
nó chỉ đang đoán trung vị theo một đường vòng.

---

## 4.3. Kết quả bài toán phân loại ngành nghề

**Bảng 4.5: Kết quả phân loại ngành nghề (lược đồ v1, dev 7.159 tin)**

| Lần chạy | Cấu hình | macro-F1 | acc | balAcc | top-3 | epoch tốt nhất |
|---|---|---|---|---|---|---|
| `dl-cat-h256` | lr 1e-3 · không chuẩn hoá · không cắt gradient | **0,0420** | 0,2127 | 0,0752 | 0,4577 | 3/11 — **phân kỳ** |
| `dl-cat-h256-cw` | như trên + cân bằng lớp | 0,0747 | 0,1904 | 0,1117 | 0,3519 | 3/11 — **phân kỳ** |
| `probe-cat` | LogReg trên **cùng** bộ vectơ | 0,5867 | 0,6423 | 0,5782 | 0,9225 | — |
| `dl-cat-v2-nostd` | lr 3e-4 + cắt gradient · không chuẩn hoá | 0,5934 | 0,6466 | 0,5917 | 0,9250 | 31/39 |
| **`dl-cat-v2`** | lr 3e-4 + cắt gradient + chuẩn hoá | **0,5987** | 0,6493 | 0,6048 | **0,9257** | 14/22 |
| `dl-cat-v2-cw` | như trên + cân bằng lớp | 0,5637 | 0,5913 | **0,6687** | 0,9012 | 6/14 |
| *mốc TF-IDF + LinearSVC* | `cat-SW-svm-1` | *0,6050 ± 0,0071* | *0,6445* | *0,6713* | — | — |

### 4.3.1. Nhận xét

**Khoảng cách tới mốc cũ là 0,0063 — nhỏ hơn một nửa khoảng tin cậy ±0,0071 của
chính mốc đó.** Hai mô hình **chưa phân biệt được** ở mức tin cậy này. Nói "PhoBERT
thua TF-IDF" là đọc sai bảng; nói đúng là "chưa đủ bằng chứng để phân biệt".

**Nhưng ở độ chính xác top-3 thì PhoBERT hơn rõ:** **0,9257** so với **0,8631** của
TF-IDF + LogReg (LinearSVC không cho xác suất nên không có top-3). Với một sản phẩm
gợi ý ba ngành nghề cho người đăng tin chọn, đây là khác biệt có ý nghĩa thực tế,
và nó không hiện ra trong độ đo top-1.

**`--class-weight` là một phép đánh đổi, không phải một cải thiện:** macro-F1 giảm
0,035 trong khi balanced accuracy tăng 0,064. Nó kéo mô hình về phía các lớp nhỏ,
đúng như thiết kế. Chọn cấu hình nào phụ thuộc vào ứng dụng thật; mặc định để tắt.

---

## 4.4. Kết quả bài toán ước lượng mức lương

**Bảng 4.6: Kết quả ước lượng mức lương (lược đồ v1, dev 5.095 tin có nhãn lương)**

| Lần chạy | Phương pháp | MAE (triệu) | MedAE | R²(log) | trong ±20 % |
|---|---|---|---|---|---|
| *mốc* | đoán trung vị 13,0 cho mọi tin | *5,86* | — | *−0,063* | — |
| *trần "biết ngành"* | trung vị của ngành, dùng nhãn thật | *5,75* | — | *−0,032* | — |
| `probe-sal` | Ridge trên vectơ PhoBERT | 6,60 | 3,26 | −0,004 | 42,1 % |
| **`dl-sal-v2`** | dense(256), 40 epoch | **4,83** | **2,86** | **0,381** | **47,1 %** |
| `dl-sal-v3-long` | cùng cấu hình, patience 15 | 4,88 | 2,86 | 0,357 | 45,9 % |

### 4.4.1. Nhận xét

Nhánh hồi quy **vượt mốc 1,03 triệu (−17,6 %)** và là kết quả rõ ràng nhất của cả
vòng thực nghiệm: R² trên thang logarit đi từ **âm** lên **0,381**, nghĩa là mô hình
thực sự đọc được tín hiệu lương từ văn bản — thứ mà nhãn ngành nghề không cung cấp
(eta² = 0,032).

Hai lần chạy cùng cấu hình cho 4,83 và 4,88: **dao động giữa các lần chạy vào khoảng
0,05 triệu**, nên không được đọc một chênh lệch nhỏ hơn thế thành cải thiện.

**Một quan sát đáng giá hơn cả con số:** Ridge trên **cùng bộ vectơ** cho **6,60** —
*tệ hơn cả việc đoán trung vị*. Cùng đặc trưng, cùng nhãn; khác biệt duy nhất là khối
dense có chuẩn hoá đầu vào và học được phi tuyến. Kết luận: **tín hiệu có nằm trong
vectơ, nhưng không nằm ở dạng tuyến tính.**

---

## 4.5. Phân tích lỗi

### 4.5.1. Lỗi của bài toán phân loại

Đọc từ `predictions_dev.parquet` của `dl-cat-v2`:

**Bảng 4.7: F1 theo lớp — bốn lớp đáng chú ý (lược đồ v1)**

| Lớp | F1 | n |
|---|---|---|
| `nhóm_nghề_khác` | **0,000** | 48 |
| `kỹ_thuật_điện_điện_tử_viễn_thông` | 0,411 | 207 |
| `thiết_kế_nghệ_thuật_giải_trí…` | 0,441 | 470 |
| `tài_chính_kế_toán_ngân_hàng_bảo_hiểm` | **0,851** | 713 |

Lớp gom tạp `nhóm_nghề_khác` bị **bỏ hoàn toàn** — đúng như phân tích dữ liệu ở
§3.6.1 đã dự báo: 48 mẫu, nội dung pha tạp. Đây là **nhiễu nhãn, không phải lỗi mô
hình**, và đó chính là lý do phải báo cáo `f1_macro_no_junk` song song.

**Bảng 4.8: Ba cặp lớp bị nhầm nhiều nhất (lược đồ v1)**

| Nhầm lẫn | Số tin |
|---|---|
| `kinh_doanh…` → `du_lịch_nhà_hàng…` | 202 |
| `du_lịch_nhà_hàng…` → `kinh_doanh…` | 180 |
| `thiết_kế_nghệ_thuật…` → `xây_dựng_kiến_trúc…` | 133 |

Cả ba cặp đều là **ranh giới nhãn mờ một cách hợp lệ**, không phải lỗi mô hình. Một
tin "Nhân viên kinh doanh cho khách sạn" thuộc về cặp thứ nhất, và cả hai nhãn đều
bảo vệ được. Cặp `kinh_doanh` ↔ `du_lịch_nhà_hàng` chiếm **382 tin** theo cả hai
chiều.

**Việc còn thiếu:** trần nhiễu nhãn **chưa bao giờ được đo**. Cách đo rẻ nhất là lấy
mẫu 100 tin, tự gán nhãn bằng tay, rồi đo độ đồng thuận với nhãn gốc. Không có con số
đó thì không biết mô hình còn cách trần bao xa.

### 4.5.2. Lỗi của bài toán hồi quy

**Bảng 4.9: Sai số theo dải lương (`dl-sal-v2`, lược đồ v1)**

| Dải lương | Số tin | MAE (triệu) |
|---|---|---|
| ≤ 30 triệu | 4.833 | **3,70** |
| > 30 triệu | 262 | **25,58** |

Mô hình dự đoán cao nhất chỉ **80,7** trong khi dữ liệu có tin ở **275**. Nó **không
dám đi ra cái đuôi phải** — đây là hệ quả trực tiếp của việc huấn luyện trên `log1p`
với hàm mất mát Huber: cả hai cùng làm giảm ảnh hưởng của giá trị lớn, và cái giá
phải trả nằm đúng ở đó.

Theo ngành: khó nhất là `nhóm_nghề_khác` (MAE 11,03), dễ nhất là
`nông_nghiệp_năng_lượng_môi_trường` (MAE 2,67).

**Hướng khắc phục đề xuất:** dùng hàm mất mát bất đối xứng, hoặc chuyển sang dự đoán
phân vị thay vì một điểm. Cả hai đều chưa được đo.

---

## 4.6. Lần chạy thất bại đầu tiên — giữ lại làm bằng chứng

Hai lần chạy đầu (`dl-cat-h256`, `dl-cat-h256-cw`) cho macro-F1 **0,042** — thấp hơn
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
1,0, hạ lr xuống 3e-4. Kết quả: **0,042 → 0,5987**.

**Bài học đáng giá hơn con số:** chính `probe-cat` — hồi quy logistic trên cùng bộ
vectơ, macro-F1 **0,5867** — là thứ phân biệt được "đặc trưng tồi" với "đầu mô hình
hỏng". Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học để chỉnh sai; nếu
nó đạt 0,59 thì vấn đề chắc chắn không nằm ở đặc trưng. Đây là lý do tầng mốc thứ hai
ở §4.2 tồn tại.

---

## 4.7. So sánh với học máy truyền thống

Trục học máy đã đóng ngày 2026-09-08 với **101 lần thí nghiệm**, và được giữ nguyên
làm mốc so sánh.

**Bảng 4.10: Xếp tầng 11 thuật toán học máy theo macro-F1 (lược đồ v1)**

| Tầng | Mô hình | macro-F1 | Trong tầng |
|---|---|---|---|
| **1** | `logreg` 0,6072 · **`svm` 0,6050** · `sgd` 0,6010 | 0,60–0,61 | hoà nhau |
| **2** | `svm_plain` 0,5927 · `xgb` 0,5912 · `lgbm` 0,5830 | 0,58–0,59 | hoà nhau |
| **3** | `extra` 0,5665 · `rf` 0,5644 · `nb` 0,5535 | 0,55–0,57 | hoà nhau |
| **4** | `knn` 0,4361 | | thua rõ tầng trên |
| **5** | `centroid` 0,3204 | | thua tất cả |

Việc xếp tầng dựa trên bootstrap ghép cặp chứ không dựa trên độ lệch chuẩn: hai mô
hình trong cùng một tầng có `P(A > B)` nằm quanh 0,5, tức không phân biệt được.

### 4.7.1. Đánh đổi giữa độ chính xác và tốc độ

Đây là bảng mà mục tiêu 3 của đề tài yêu cầu, và là bảng quyết định phương án triển
khai thật.

**Bảng 4.11: Đánh đổi độ chính xác — chi phí huấn luyện (lược đồ v1)**

| Mô hình | macro-F1 | thời gian fit | so với rẻ nhất |
|---|---|---|---|
| `knn` | 0,4361 | 17 s | 1,0× |
| `nb` | 0,5535 | 24 s | 1,5× |
| `sgd` | 0,6010 | 40 s | 2,4× |
| **`svm`** | **0,6050** | **42 s** | **2,5×** |
| `logreg` | 0,6072 | 4,8 phút | 17,1× |
| `rf` | 0,5644 | 5,3 phút | 18,9× |
| `lgbm` | 0,5830 | 12,9 phút | 46,2× |

`logreg` nhỉnh hơn `svm` đúng 0,0022 macro-F1 nhưng tốn gấp **7 lần** thời gian
huấn luyện — và khoảng chênh đó nhỏ hơn khoảng tin cậy của chính nó. Vì vậy
**LinearSVC được chọn làm mốc**, không phải mô hình có điểm cao nhất.

### 4.7.2. Đánh đổi ở khâu suy luận — bảng còn thiếu

> ⛔ **CHƯA CÓ SỐ LIỆU** — mục tiêu 3 của đề tài yêu cầu so sánh **cả tốc độ xử lý**
> giữa học sâu và học máy, nhưng **độ trễ suy luận chưa được đo**. Con số đã biết:
> PhoBERT chạy khoảng **30 tin/giây** trên CPU ở khâu nhúng, còn TF-IDF + LinearSVC
> gần như tức thời. Cần một bảng đo thật (mili-giây mỗi tin, cả hai đường) đặt cạnh
> bảng độ chính xác. Đây là việc bắt buộc trước khi bảo vệ, vì nó là một trong bốn
> mục tiêu cụ thể.

---

## 4.8. Tổng hợp: đối chiếu kết quả với mốc

**Bảng 4.12: Tổng hợp đối chiếu (lược đồ v1)**

| Bài toán | Mốc ngây thơ | Mốc học máy | Kết quả học sâu | Kết luận |
|---|---|---|---|---|
| Phân loại (macro-F1) | 0,0210 | **0,6050 ± 0,0071** | 0,5987 | Chưa phân biệt được với mốc học máy |
| Phân loại (top-3) | — | 0,8631 | **0,9257** | **Vượt rõ** |
| Lương (MAE, triệu) | 5,86 | — | **4,83** | **Vượt mốc 17,6 %** |
| Lương (R² log) | −0,063 | — | **0,381** | Từ âm lên dương — đọc được tín hiệu thật |
| `disclosed` (acc) | 0,7117 | — | *chưa chạy* | ⛔ |

### 4.8.1. Những gì chưa làm

**Bảng 4.13: Các hạng mục thực nghiệm còn thiếu**

| Hạng mục | Vì sao còn thiếu |
|---|---|
| Chạy lại toàn bộ trên lược đồ chia mới | Lược đồ đổi ngày 2026-09-09 |
| Chạy lại phân tích khám phá dữ liệu | Bảy hình đã sinh lại trên tệp gốc 47.707 dòng, nhưng `summary.json` còn ghi 33.396 dòng — các đại lượng chỉ có trong JSON đang mang dấu `⛔` ở §3.6 |
| Sinh hình 07 — độ dài token | Cần `--tokens` trong `.venv-dl`; cái giá của việc cắt ở 256 token chưa được đo |
| Bài toán `disclosed` | Chưa có lần chạy nào |
| Mô hình **đa nhiệm** | Theo lộ trình, chỉ hợp nhất sau khi hai nhánh có số riêng |
| Tinh chỉnh PhoBERT | Máy hiện tại không có GPU/MPS; cần Colab hoặc Kaggle |
| Bảng độ trễ suy luận | Xem §4.7.2 |
| Đo trần nhiễu nhãn | Xem §4.5.1 |
| Đánh giá cuối trên `test` | Chỉ được chạm **một lần**, ở bước cuối cùng |

> **Nguồn số liệu:** [04-results.md](../04-results.md) (10 dòng thí nghiệm) ·
> [06-baseline-dl.md](../06-baseline-dl.md) §5 ·
> [archive/10-so-sanh-mo-hinh.md](../archive/10-so-sanh-mo-hinh.md) ·
> [archive/04-results-ml.md](../archive/04-results-ml.md).

---

[← Chương 3](04-chuong-3-phuong-phap.md) · [Chương 5 →](06-chuong-5-he-thong.md)
