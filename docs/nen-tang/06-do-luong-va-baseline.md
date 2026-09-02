[← Rò rỉ dữ liệu](05-ro-ri-du-lieu.md) · [Nền tảng](00-index.md) · [Chính quy hoá →](07-chinh-quy-hoa.md)

# 6. Đo lường và baseline

Chọn sai thước đo thì mọi việc sau đó đều vô nghĩa — bạn sẽ tối ưu chăm chỉ về
hướng sai. Note này giải thích vì sao dự án chọn macro-F1, và vì sao mọi mô hình
đều phải vượt qua một cái sàn trước khi được coi là "có học được gì".

---

## 1. Accuracy nói dối khi lớp lệch

Bài toán phân lớp nghề có 16 lớp, và chúng **rất lệch**: lớp lớn nhất gấp **27 lần**
lớp nhỏ nhất. Ba lớp nhỏ nhất chỉ có 196–258 dòng.

Xét mô hình ngu nhất có thể — luôn đoán lớp đông nhất, không nhìn dữ liệu:

| Thước đo | Điểm |
|---|---|
| **accuracy** | **0,2014** |
| macro-F1 | **0,0210** |
| balanced accuracy | 0,0625 |

Accuracy nói "đúng 20 %". Nghe không tệ lắm. Nhưng mô hình này **bỏ qua hoàn
toàn 15 trên 16 lớp** — nó chưa học gì cả. Macro-F1 nói đúng sự thật: 0,0210.

**Vì sao chênh nhau như vậy:**

- **Accuracy** = tỷ lệ dự đoán đúng trên **toàn bộ** mẫu. Lớp đông đóng góp nhiều
  mẫu nên nó chi phối con số. Đoán đúng lớp đông là đủ để có accuracy trông được.
- **Macro-F1** = tính F1 **cho từng lớp riêng**, rồi lấy trung bình **không trọng số**.
  Lớp 196 dòng có tiếng nói **ngang** lớp 5.000 dòng. Bỏ rơi một lớp là mất 1/16
  tổng điểm, bất kể lớp đó to hay nhỏ.

> **Quy tắc:** lớp càng lệch, accuracy càng dễ nói dối. Với 27:1 thì nó nói dối rất nhiều.

---

## 2. Precision, recall, F1 — nhắc lại nhanh

Cho một lớp cụ thể, ví dụ "kế toán":

|  | Mô hình nói "kế toán" | Mô hình nói lớp khác |
|---|---|---|
| **Thật sự là kế toán** | TP (đúng) | FN (bỏ sót) |
| **Không phải kế toán** | FP (báo nhầm) | TN (đúng) |

```
precision = TP / (TP + FP)    Trong những tin nó GỌI là kế toán, bao nhiêu đúng?
recall    = TP / (TP + FN)    Trong những tin THẬT SỰ là kế toán, nó bắt được bao nhiêu?
F1        = trung bình điều hoà của hai cái trên
```

**Vì sao là trung bình điều hoà, không phải trung bình cộng.** Trung bình điều hoà
phạt nặng sự mất cân bằng:

| precision | recall | TB cộng | **F1** |
|---|---|---|---|
| 1,00 | 0,10 | 0,55 | **0,18** |
| 0,55 | 0,55 | 0,55 | **0,55** |

Một mô hình chỉ dám gọi "kế toán" khi cực chắc chắn sẽ có precision 1,00 nhưng
bỏ sót 90 % — trung bình cộng cho nó 0,55, F1 cho nó 0,18. F1 nói đúng.

---

## 3. Bốn thước đo dự án dùng, và mỗi cái trả lời gì

| Thước đo | Trả lời câu hỏi | Vì sao có mặt |
|---|---|---|
| **macro-F1** | Mô hình làm tốt **đều** trên cả 16 lớp không? | Thước đo **chính** để chọn mô hình |
| `f1_macro_no_junk` | Bỏ lớp thùng rác `nhóm_nghề_khác` ra thì sao? | Tách **nhiễu nhãn** khỏi **lỗi mô hình** |
| `balanced_accuracy` | Recall trung bình trên các lớp | Nhạy với lớp bị bỏ rơi |
| `accuracy` | Tỷ lệ đúng thô | Chỉ để **so với sàn**, không để chọn mô hình |
| `top3_accuracy` | Đáp án đúng có nằm trong 3 gợi ý không? | Đúng cái người dùng thật cần |

**`f1_macro_no_junk` đáng chú ý.** Lớp `nhóm_nghề_khác` là thùng rác: nó chứa
"Nhân Viên Seo Web", "Nhân Viên Quản Trị Website" — những tin lẽ ra thuộc marketing
và IT. Mô hình đoán sai ở đó **không phải lỗi của mô hình**, đó là nhãn gốc sai.
Báo cáo cả hai con số giúp người đọc tách bạch hai chuyện.

**`top3_accuracy` = 93 %** trong khi macro-F1 chỉ 0,61. Chênh lệch đó không phải
mâu thuẫn — nó nói rằng mô hình gần như luôn đưa nhãn đúng vào top 3, chỉ hay
xếp nhầm thứ tự giữa những nghề vốn nhập nhằng. Với một hệ thống gợi ý cho người
dùng chọn, 93 % mới là con số đúng để báo cáo.

---

## 4. Baseline — cái sàn phải vượt

Một con số đứng một mình không nói lên điều gì. **0,6112 là tốt hay tệ?**
Không trả lời được, trừ khi biết so với cái gì.

Dự án dựng ba tầng sàn cho bài phân lớp:

| Tầng | Là gì | macro-F1 |
|---|---|---|
| **Sàn 1** — ngẫu nhiên | Luôn đoán lớp đông nhất | 0,0210 |
| **Sàn 2** — không dùng ML | Trùng từ khoá trong tiêu đề | **0,4321** |
| Mô hình rẻ nhất | SVM chỉ đọc tiêu đề | 0,5547 |
| Mô hình chốt | SVM C=0,02, toàn văn | 0,6050 |

**Sàn 2 mới là cái sàn thật.** Nó là câu hỏi: *"nếu không dùng học máy, chỉ viết
vài dòng if-else khớp từ khoá, thì được bao nhiêu?"* — 0,4321.

Bất kỳ mô hình học máy nào không vượt được 0,4321 là **không đáng tồn tại**: nó
tốn thời gian huấn luyện, khó giải thích, khó bảo trì, và thua một đoạn code
đơn giản hơn nhiều.

Trong dự án này, **hai mô hình KNN toàn văn thua sàn 2** (0,4059). Đó là một kết
quả có ích, và nó nằm trong log — xem [note 4](04-ma-tran-thua-va-so-chieu.md).

Cho bài toán lương, sàn tương ứng là:

| Bài toán | Sàn phải vượt |
|---|---|
| `disclosed` (có công bố lương không) | accuracy **0,7176** · macro-F1 **0,4179** — "đoán luôn là có" |
| `salary` (mức lương) | MAE **5,54 triệu** — trung vị theo nhóm nghề |

---

## 5. Nhiễu — khi nào chênh lệch là thật

Chạy hai cấu hình, một cái ra 0,6050 và một cái ra 0,6033. Cái đầu tốt hơn chứ?

**Chưa chắc.** Điểm số đo trên một tập val hữu hạn (7.159 tin) nên bản thân nó
có dao động ngẫu nhiên. Dự án đo dao động đó bằng **bootstrap 200 lần**:

```
độ lệch chuẩn macro-F1  ≈  0,009
```

Nghĩa là: **chênh lệch nhỏ hơn 0,009 không phân biệt được với nhiễu.**

Áp vào bảng ablation:

| So sánh | Chênh | Kết luận |
|---|---|---|
| `province` 0,6050 vs `raw` 0,6033 | +0,0017 | **Trong nhiễu.** "Có lẽ giúp", P(>0) = 0,97 |
| Bỏ từ dừng: 0,5756 vs 0,5754 | +0,0002 | **Trong nhiễu.** Không phân biệt được |
| `C` 0,02 vs `C` 0,5: 0,6050 vs 0,5763 | +0,0287 | **Thật.** Gấp hơn 3 lần σ |

Không có bước này thì rất dễ đi tối ưu những chênh lệch 0,001 suốt cả tuần, và
tin rằng mình đang tiến bộ.

> **Nguyên tắc:** trước khi mừng vì một cải thiện, hỏi *"nó có lớn hơn nhiễu không?"*
> Không biết nhiễu bằng bao nhiêu thì không được phép kết luận gì.

---

## 6. Trần — thứ chưa ai biết

macro-F1 0,6112 nghe thấp. Nhưng **trần thật là bao nhiêu?**

Nếu 40 % lỗi của mô hình thật ra là **nhãn gốc sai** (như những tin SEO bị dán
nhãn `nhóm_nghề_khác`), thì 0,6112 trên nhãn bẩn tương đương khoảng 0,75 trên
nhãn sạch — và mô hình đã gần chạm trần rồi.

Chưa đo. Cách đo ghi ở
[09-lo-trinh.md — Ưu tiên 2](../09-lo-trinh.md#ưu-tiên-2--đo-trần-nhiễu-nhãn):
lấy 150–200 tin bị đoán sai, gán tay từng tin vào ba nhóm — *mô hình sai* ·
*nhãn gốc sai* · *thật sự nhập nhằng*.

Đây là con số quý nhất mà một báo cáo có thể có, vì nó trả lời câu hỏi mà không
mô hình nào trả lời được: **còn bao nhiêu chỗ để cải thiện?**

---

## Quay lại thực tế dự án

- [03-protocol.md §4](../03-protocol.md) — định nghĩa metric chính thức của dự án
- [06-mo-hinh-phan-lop.md](../06-mo-hinh-phan-lop.md) — bậc thang kết quả và cách đọc nó
- [04-results.md](../04-results.md) — 27 dòng thí nghiệm
