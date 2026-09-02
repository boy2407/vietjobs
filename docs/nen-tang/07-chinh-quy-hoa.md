[← Đo lường và baseline](06-do-luong-va-baseline.md) · [Nền tảng](00-index.md)

# 7. Chính quy hoá — `C`, `alpha`, và vì sao mô hình cần bị kìm lại

Note cuối, và là note giải thích con số kỳ lạ nhất trong dự án: **`C = 0,02`**,
thấp hơn mặc định 50 lần, mà lại cho kết quả tốt nhất.

---

## 1. Quá khớp — nhìn bằng số, không bằng lời

Hai dòng đo thật của dự án, cùng dữ liệu, cùng đặc trưng, chỉ khác một thứ:

| Mô hình | R² **train** | R² **val** | MAE val |
|---|---|---|---|
| `LinearRegression` thuần | 0,718 | **0,081** | **6,00 tr** |
| `Ridge` alpha=1 | 0,657 | **0,507** | **4,25 tr** |

Đọc kỹ hai dòng này — chúng chứa toàn bộ ý tưởng của chính quy hoá:

- Mô hình thứ nhất **giỏi hơn trên dữ liệu nó đã thấy** (0,718 > 0,657).
- Mô hình thứ nhất **tệ hơn rất nhiều trên dữ liệu mới** (0,081 so với 0,507 —
  kém hơn **sáu lần**).

Ridge **cố tình** làm mình tệ đi trên train, để đổi lấy tốt hơn nhiều trên val.

Thêm một chi tiết cay đắng: MAE 6,00 triệu của mô hình thuần còn **tệ hơn baseline
"lấy trung vị theo nhóm nghề"** (5,54 triệu). Tức là tệ hơn không dùng mô hình nào.

**Quá khớp** = mô hình học thuộc chi tiết vụn vặt và nhiễu của tập train, thay vì
học quy luật tổng quát.

---

## 2. Vì sao quá khớp xảy ra ở đây

Quay lại tỷ lệ p/n từ [note 4](04-ma-tran-thua-va-so-chieu.md):

```
p = 236.596 chiều      n = 33.396 mẫu      p/n = 7,1
```

Có nhiều "nút vặn" gấp 7 lần số ví dụ để học. Với `p > n`, luôn tồn tại một tổ
hợp trọng số khớp **hoàn hảo** dữ liệu train — kể cả khi nhãn hoàn toàn ngẫu nhiên.

Mô hình không cần *hiểu* gì. Nó chỉ cần *nhớ*. Và cái nó nhớ không dùng được cho
tin mới.

---

## 3. Chính quy hoá làm gì

Không chính quy hoá, mô hình chỉ tối ưu một thứ:

```
tối thiểu hoá:   sai số trên train
```

Chính quy hoá thêm một khoản phạt vào **độ lớn của trọng số**:

```
tối thiểu hoá:   sai số trên train  +  λ × (tổng bình phương trọng số)
```

Bây giờ mô hình phải trả giá cho mỗi trọng số nó đặt. Nó chỉ chịu trả giá khi
đặc trưng đó **thật sự** giúp giảm sai số nhiều hơn khoản phạt.

Kết quả: hàng trăm nghìn chiều nhiễu bị ép về gần 0, chỉ những chiều có tín hiệu
thật mới giữ được trọng số đáng kể. **Đó chính là cách SVM sống sót trong 236.596
chiều còn KNN thì không** — KNN không có cơ chế nào để bỏ qua chiều vô dụng.

---

## 4. `C` và `alpha` — cùng một nút vặn, ngược chiều nhau

Đây là chỗ hay nhầm nhất, nên nói rõ:

| Tham số | Ở đâu | Ý nghĩa | Giá trị **nhỏ** nghĩa là |
|---|---|---|---|
| `alpha` | `Ridge`, `Lasso` | **Là** λ, hệ số phạt | phạt **nhẹ** → mô hình tự do hơn |
| `C` | `LinearSVC`, `LogisticRegression` | Là **nghịch đảo** của λ | phạt **NẶNG** → mô hình bị kìm chặt |

```
alpha ↑   =   chính quy hoá mạnh hơn
C     ↓   =   chính quy hoá mạnh hơn      ← ngược chiều!
```

Nhớ bằng một câu: **`C` là "độ tự do" (Cost of misclassification), `alpha` là "độ kìm".**

---

## 5. Đường cong `C` của dự án — đọc như một triệu chứng

Quét `C` cho `LinearSVC`, cùng cấu hình `province`, cùng seed:

| `C` | macro-F1 (val) | Thời gian | Diễn giải |
|---|---|---|---|
| 4,0 | 0,5171 | 313,6 s | Quá tự do — quá khớp nặng |
| 1,0 *(mặc định)* | 0,5618 | 110,6 s | Vẫn quá tự do |
| 0,2 | 0,5873 | 90,9 s | |
| 0,1 | 0,5983 | 37,5 s | |
| 0,05 | 0,6030 | 33,5 s | |
| **0,02** | **0,6050** | **27,8 s** | **Điểm ngọt** |
| 0,01 | 0,5976 | 34,4 s | Bắt đầu quá kìm |
| 0,005 | 0,5865 | 44,5 s | Quá kìm — thiếu khớp |

Đường cong hình chữ **∩** kinh điển. Hai bên đều tệ, ở giữa có một đỉnh:

- **`C` quá lớn** → quá khớp. Học thuộc nhiễu của train.
- **`C` quá nhỏ** → thiếu khớp. Ép trọng số về 0 đến mức không học được cả tín hiệu thật.

### Ba điều đáng chú ý ngoài cái đỉnh

**a. `C` tối ưu thấp hơn mặc định 50 lần.** Đây không phải một con số ngẫu nhiên
đẹp — nó là **triệu chứng**. Nó nói rằng phần lớn 236.596 chiều đang là nhiễu.
Việc nên làm tiếp không phải quét `C` thêm vòng nữa mà là **cắt bớt chiều** —
quét `min_df`, `max_features`. Ghi ở
[09-lo-trinh.md — Ưu tiên 3b](../09-lo-trinh.md#ưu-tiên-3--hai-đòn-bẩy-rẻ-cho-phân-lớp-làm-trước-khi-nghĩ-đến-dl).

**b. Chính quy hoá mạnh chạy NHANH hơn.** `C=0,02` mất 27,8 s, `C=4,0` mất 313,6 s
— **nhanh gấp 11 lần** và điểm cao hơn. Ràng buộc chặt làm bài toán tối ưu dễ hội
tụ hơn. Bạn không phải đánh đổi gì cả ở đây.

**c. Một dòng siêu tham số ăn đứt chín bước xử lý ngôn ngữ.**

| Việc làm | Đóng góp macro-F1 |
|---|---|
| Toàn bộ 9 bước xử lý tiếng Việt | **+0,0017** |
| Chỉnh `C` từ 0,5 xuống 0,02 | **+0,0287** |

Gấp gần **17 lần**. Đây là bài học thực tế đắt giá nhất của dự án: nếu bạn còn
chưa quét siêu tham số, đừng bỏ thời gian viết thêm bước tiền xử lý.

---

## 6. `class_weight="balanced"` — một dạng chỉnh khác

Cấu hình chốt cũng bật `class_weight="balanced"`. Nó không phải chính quy hoá,
nhưng cùng tinh thần "sửa cho mô hình đừng đi theo đường dễ".

Với lớp lệch 27:1, đường dễ nhất là bỏ rơi lớp hiếm — accuracy vẫn đẹp
([note 6](06-do-luong-va-baseline.md)). `class_weight="balanced"` nhân trọng số
sai sót của mỗi lớp lên tỷ lệ nghịch với số mẫu của nó: đoán sai một tin thuộc
lớp 196 dòng bị phạt nặng hơn nhiều so với đoán sai một tin thuộc lớp 5.000 dòng.

Đây là lý do `balanced_accuracy` (0,6816) **cao hơn** macro-F1 (0,6112) trên test:
mô hình đang cố ý dành recall cho lớp hiếm.

---

## 7. Bảng chọn nhanh

| Triệu chứng | Nghĩa là | Làm gì |
|---|---|---|
| Điểm train cao, điểm val thấp | Quá khớp | Giảm `C` / tăng `alpha` · tăng `min_df` · giảm `max_features` |
| Điểm train thấp, điểm val cũng thấp | Thiếu khớp | Tăng `C` / giảm `alpha` · thêm đặc trưng · đổi mô hình |
| Hai điểm gần nhau, cả hai đều thấp | Đặc trưng không đủ tín hiệu | Đổi cách biểu diễn, đừng vặn siêu tham số |
| `C` tối ưu thấp bất thường | Quá nhiều chiều nhiễu | Cắt chiều, đừng quét `C` nữa |
| Điểm test **cao hơn** val | Không có dấu hiệu overfit vào val | Bình thường — dự án này 0,6112 > 0,6050 |

---

## Quay lại thực tế dự án

- [07-bai-toan-luong.md](../07-bai-toan-luong.md#bẫy-đã-biết-trước) — bảng `LinearRegression` vs `Ridge`
- [06-mo-hinh-phan-lop.md](../06-mo-hinh-phan-lop.md) — bậc thang kết quả đầy đủ
- [note 4 — ma trận thưa và số chiều](04-ma-tran-thua-va-so-chieu.md) — vì sao p/n = 7,1 là vấn đề
- [04-results.md](../04-results.md) — mọi dòng quét `C`
