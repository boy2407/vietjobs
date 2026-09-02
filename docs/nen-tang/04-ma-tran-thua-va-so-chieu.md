[← n-gram và ranh giới từ](03-ngram-va-ranh-gioi-tu.md) · [Nền tảng](00-index.md) · [Rò rỉ dữ liệu →](05-ro-ri-du-lieu.md)

# 4. Ma trận thưa và số chiều

Dự án này dựng một ma trận **236.596 chiều** trên **33.396 mẫu** huấn luyện.
Note này giải thích con số đó nghĩa là gì, vì sao nó lưu được trong bộ nhớ, và
vì sao nó làm KNN sụp đổ trong khi SVM vẫn sống.

---

## 1. "236.596 chiều" nghĩa là gì

Mỗi tin tuyển dụng biến thành một **dãy 236.596 số**. Mỗi vị trí trong dãy tương
ứng với một đặc trưng cụ thể: một từ, một cặp từ, một tỉnh, một cột đếm.

```
tin #1  =  [0, 0, 0.42, 0, 0, 0, 0.31, 0, ... , 0, 0.07, 0]
            ↑            ↑                          ↑
        "kế toán"    "kinh doanh"              experience_months
```

236.596 chiều đến từ đâu — cộng lại đúng bằng tổng:

| Khối | Chiều |
|---|---|
| mô tả | 103.459 |
| yêu cầu | 65.737 |
| phúc lợi | 29.936 |
| kỹ năng kỹ thuật | 14.163 |
| kỹ năng mềm | 10.747 |
| tiêu đề | 6.835 |
| bằng cấp | 5.626 |
| one-hot (tỉnh, hợp đồng, kinh nghiệm) | 57 |
| ngoại ngữ | 22 |
| số | 14 |
| **tổng** | **236.596** |

---

## 2. Thưa — chỉ 492 ô khác 0

Một tin tuyển dụng dùng vài trăm từ, không dùng hai trăm nghìn từ. Nên hầu hết
các ô trong dãy là **số 0**.

Trung bình mỗi dòng chỉ có **492 ô khác 0**:

```
492 / 236.596  =  0,21 %      →  99,79 % ma trận là số 0
```

Ví dụ nhỏ hơn, kiểm chứng được: tiêu đề "Nhân viên kinh doanh" chỉ kích hoạt
**7 ô** trên 6.835 chiều của khối tiêu đề — bảng ở
[note 2](02-tf-idf-la-gi.md#6-ví-dụ-tính-tay--trên-dữ-liệu-thật).

### Lưu thưa hay lưu đặc — con số

| Cách lưu | Cần bao nhiêu |
|---|---|
| **Đặc** — lưu đủ 236.596 × 33.396 số thực 8 byte | ~**63 GB** |
| **Thưa** — chỉ lưu (dòng, cột, giá trị) của ô khác 0 | ~**130 MB** |

Chênh nhau khoảng **480 lần**. Đó là lý do dự án ép `sparse_threshold=1.0` và
dùng `MaxAbsScaler` chứ không phải `StandardScaler` — `StandardScaler` trừ trung
bình, phép đó biến mọi số 0 thành khác 0 và **phá tính thưa** ngay lập tức.
Chi tiết ở [05-dac-trung-tfidf.md §6–7](../05-dac-trung-tfidf.md#6-khối-số--vì-sao-log1p-rồi-maxabsscaler).

---

## 3. Tỷ lệ p/n — chỉ số quan trọng nhất bạn nên nhớ

```
p = số chiều (đặc trưng)     = 236.596
n = số mẫu huấn luyện        =  33.396
p / n                        =       7,1
```

**Có nhiều đặc trưng gấp 7 lần số ví dụ để học.**

Vì sao điều đó nguy hiểm: với `p > n`, luôn tồn tại một tổ hợp trọng số khớp
**hoàn hảo** dữ liệu huấn luyện — kể cả khi nhãn hoàn toàn ngẫu nhiên. Mô hình
không cần *học quy luật*, nó chỉ cần *nhớ*. Và cái nó nhớ không tổng quát hoá
sang tin mới.

Bảng p/n trong dự án này:

| Cấu hình | p | n | p/n | Ghi chú |
|---|---|---|---|---|
| `scope=full` | 236.596 | 33.396 | **7,1** | Cấu hình thật |
| `scope=title` | 6.835 | 33.396 | 0,20 | |
| `scope=structured` | ~60 | 23.965 | 0,0025 | Sân chơi công bằng duy nhất cho `LinearRegression` thuần |

Đây là lý do `LinearRegression` không chính quy hoá **chắc chắn hỏng** ở
`scope=full`, và vì sao dự án tạo hẳn `scope=structured` để đo nó cho công bằng —
[note 7](07-chinh-quy-hoa.md).

---

## 4. Lời nguyền số chiều — vì sao KNN sụp

Đây là kết quả gây sốc nhất trong bảng của dự án:

| Mô hình | Đặc trưng | macro-F1 (val) |
|---|---|---|
| Baseline từ khoá, **không dùng ML** | tiêu đề | 0,4321 |
| KNN k=15 | **chỉ tiêu đề** | 0,5307 |
| KNN k=30 | **toàn văn** | **0,4059** ← thua cả baseline |
| SVM C=0,02 | toàn văn | 0,6050 |

**Cho KNN nhiều đặc trưng hơn làm nó tệ đi**, tệ đến mức thua một baseline không
dùng học máy. SVM trên đúng bộ đặc trưng đó lại tốt hơn hẳn.

### Vì sao

KNN dựa hoàn toàn vào **khoảng cách**: tìm k tin gần nhất, lấy nhãn đa số.
Trong không gian rất nhiều chiều, khoảng cách mất ý nghĩa phân biệt:

> Khi số chiều tăng, khoảng cách từ một điểm tới **hàng xóm gần nhất** và tới
> **hàng xóm xa nhất** hội tụ về nhau. "Gần nhất" không còn nghĩa là "giống nhất".

Cụ thể với văn bản: hai tin cùng ngành có thể **không dùng chung một từ nào** ngoài
từ dừng. Với TF-IDF chuẩn hoá, chúng gần như trực giao — khoảng cách bằng nhau
hết. KNN không có gì để bám vào.

### Vì sao SVM không sụp

SVM tuyến tính không đo khoảng cách giữa các điểm. Nó tìm một **siêu phẳng** —
một tổ hợp có trọng số của các chiều. Chiều nhiễu chỉ cần nhận trọng số gần 0
là bị vô hiệu hoá. Thêm chiều vô dụng làm bài toán khó hơn, nhưng **không phá**
cách nó làm việc.

Điều kiện để chuyện đó xảy ra: phải ép trọng số về gần 0 cho hầu hết các chiều.
Đó chính là việc của tham số `C` — và cũng là lý do `C` tối ưu rơi xuống tận 0,02.

---

## 5. Đọc `C = 0,02` như một triệu chứng

`C` mặc định của `LinearSVC` là 1,0. Dự án này quét và tìm ra:

| `C` | macro-F1 (val) |
|---|---|
| 4,0 | 0,5171 |
| 1,0 (mặc định) | 0,5618 |
| 0,1 | 0,5983 |
| **0,02** | **0,6050** |
| 0,01 | 0,5976 |
| 0,005 | 0,5865 |

`C` nhỏ = chính quy hoá mạnh = ép trọng số về gần 0. `C` tối ưu thấp hơn mặc định
**50 lần** không phải là một con số ngẫu nhiên đẹp. Nó là **mô hình đang kêu cứu
vì quá nhiều chiều**.

Cách đọc đúng: nếu phải chính quy hoá mạnh đến vậy mới chạy được, thì phần lớn
236.596 chiều đang là nhiễu. Việc nên làm tiếp không phải quét `C` thêm một vòng
nữa, mà là **cắt bớt chiều** — quét `min_df`, `max_features`. Ghi ở
[09-lo-trinh.md — Ưu tiên 3b](../09-lo-trinh.md#ưu-tiên-3--hai-đòn-bẩy-rẻ-cho-phân-lớp-làm-trước-khi-nghĩ-đến-dl).

---

## 6. Ba điều nên nhớ

1. **Nhiều đặc trưng hơn không tự động tốt hơn.** KNN toàn văn 0,4059 so với KNN
   chỉ tiêu đề 0,5307 là bằng chứng đo được, ngay trong dự án này.
2. **Thuật toán khác nhau chịu số chiều khác nhau.** Đừng kết luận "bộ đặc trưng
   này tệ" từ một thuật toán duy nhất — KNN nói tệ, SVM nói tốt, và SVM đúng.
3. **`C` tối ưu bất thường thấp là một thông điệp**, không phải một con số phải
   chép vào báo cáo rồi thôi.

---

## Quay lại thực tế dự án

- [06-mo-hinh-phan-lop.md](../06-mo-hinh-phan-lop.md) — bậc thang kết quả đầy đủ
- [05-dac-trung-tfidf.md](../05-dac-trung-tfidf.md) — 236.596 chiều được lắp thế nào
- [note 7 — chính quy hoá](07-chinh-quy-hoa.md) — `C` thật ra làm gì
