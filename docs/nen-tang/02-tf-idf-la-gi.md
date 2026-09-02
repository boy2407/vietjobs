[← Vì sao phải làm sạch](01-vi-sao-phai-lam-sach.md) · [Nền tảng](00-index.md) · [n-gram và ranh giới từ →](03-ngram-va-ranh-gioi-tu.md)

# 2. TF-IDF là gì

Mô hình học máy chỉ ăn được **số**. TF-IDF là cách biến một đoạn văn thành một
dãy số. Toàn bộ ý tưởng gói trong một câu:

> Một từ quan trọng với tài liệu này nếu nó **xuất hiện nhiều trong tài liệu này**
> nhưng **hiếm gặp ở các tài liệu khác**.

Hai vế đó chính là TF và IDF.

**Tên đầy đủ: Term Frequency – Inverse Document Frequency.** Bốn chữ, đọc từng chữ
thì ra luôn công thức:

| Chữ | Nghĩa | Trong công thức |
|---|---|---|
| **T**erm | *từ* (hoặc n-gram) — đơn vị được đếm | `t` |
| **F**requency | *tần suất* — đếm bao nhiêu **lần** | `tf(t, d)` |
| **I**nverse | *nghịch đảo* — càng nhiều thì càng nhỏ | dấu chia trong `N / df` |
| **D**ocument **F**requency | số **tài liệu** chứa từ đó | `df(t)` |

Hai chỗ dễ đọc sai:

- **"Inverse" gắn vào "Document Frequency", không gắn vào "Term Frequency".**
  Đọc là `TF × I(DF)`. Tần suất từ giữ chiều thuận — lặp nhiều thì điểm cao;
  chỉ tần suất tài liệu mới bị lật ngược.
- **"Document Frequency" không phải "tần suất trong tài liệu".** Nó đếm có bao
  nhiêu **tin** chứa từ đó, không phải từ đó xuất hiện bao nhiêu lần.

---

## 1. Túi từ — bước đầu tiên, và cái giá phải trả

Trước khi có TF-IDF, phải chấp nhận một sự đơn giản hoá thô bạo: **túi từ**
(bag of words). Văn bản bị coi là một cái túi đựng từ, **không có thứ tự**.

```
"Nhân viên kinh doanh"  →  {nhân: 1, viên: 1, kinh: 1, doanh: 1}
"Doanh kinh viên nhân"  →  {nhân: 1, viên: 1, kinh: 1, doanh: 1}   ← giống hệt!
```

Mô hình **không phân biệt được** hai câu trên. Đó là mất mát thật, và ta trả giá
đó một cách có ý thức, đổi lấy hai thứ:

- Ma trận thưa, tính rất nhanh — 27 thí nghiệm chạy trong một buổi.
- Mô hình tuyến tính, xem được trọng số từng từ, giải thích được vì sao nó đoán vậy.

Cách vá một phần mất mát này là **n-gram** — xem [note 3](03-ngram-va-ranh-gioi-tu.md).
Cách vá triệt để là mạng nơ-ron đọc theo thứ tự (PhoBERT), ghi ở
[09-lo-trinh.md — Ưu tiên 4](../09-lo-trinh.md#ưu-tiên-4--một-mốc-học-sâu).

---

## 2. TF — tần suất trong tài liệu

`tf(t, d)` = số lần từ `t` xuất hiện trong tài liệu `d`.

Nhưng đếm thô có vấn đề: một từ xuất hiện 10 lần **không** quan trọng gấp 10 lần
một từ xuất hiện 1 lần. Tin tuyển dụng hay lặp từ khoá ngành vì lý do SEO chứ
không phải vì tin đó "bán hàng hơn" tin khác.

Nên dự án bật `sublinear_tf=True`, đổi công thức thành:

```
tf = 1 + log(số lần xuất hiện)
```

| Số lần xuất hiện | `tf` thô | `1 + log(tf)` |
|---|---|---|
| 1 | 1 | 1,00 |
| 2 | 2 | 1,69 |
| 5 | 5 | 2,61 |
| 10 | 10 | 3,30 |
| 100 | 100 | 5,61 |

Lần thứ hai thêm nhiều thông tin; lần thứ một trăm gần như không thêm gì.
Thang log nói đúng điều đó.

---

## 3. IDF — độ hiếm trên toàn kho

Vế thứ hai sửa một lỗ hổng lớn của TF: từ *"công ty"* xuất hiện ở gần như **mọi**
tin tuyển dụng. TF của nó cao, nhưng nó chẳng phân biệt được nghề nào với nghề nào.

`idf` hạ trọng số những từ có mặt khắp nơi. Công thức sklearn dùng:

```
idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1
```

- `N` = tổng số tài liệu (ở đây: **33.396** tin trong tập train)
- `df(t)` = số tài liệu **có chứa** từ `t` (bao nhiêu tin, không phải bao nhiêu lần)

Từ càng phổ biến → `df` càng lớn → `idf` càng nhỏ.

| Nếu từ xuất hiện ở… | `df` | `idf` |
|---|---|---|
| 1 % số tin | 334 | 5,60 |
| 10 % số tin | 3.340 | 3,30 |
| 50 % số tin | 16.698 | 1,69 |
| 90 % số tin | 30.056 | 1,11 |

**TF-IDF = tf × idf**. Còn một bước cuối nữa — chuẩn hoá L2 — ở mục kế tiếp.

---

## 4. Chuẩn hoá L2 — vì sao mọi tin phải dài đúng 1

Nhân `tf × idf` xong vẫn chưa dùng được. Còn một bệnh chưa chữa: **tin dài tự
động thắng tin ngắn**.

### Kho ví dụ

Một kho duy nhất phục vụ cả mục này và [§5](#5-ma-trận-hiện-hình).

```
D1 = "tuyển nhân viên kinh doanh"                      kinh doanh · ngắn
D2 = "nhân viên kinh doanh phụ trách tìm kiếm khách     kinh doanh · DÀI
      hàng và chăm sóc khách hàng cũ"
D3 = "kế toán tổng hợp"                                 kế toán
D4 = "kế toán trưởng có kinh nghiệm"                    kế toán
D5 = "kỹ sư xây dựng công trình"                        kỹ thuật
```

Năm tin, ba nghề, `N = 5`. **D1 và D2 cùng nghề**, và D2 là tin dài.

### Bệnh

Một tin viết dài có nhiều từ khác 0 hơn, nên vector của nó **dài hơn** — theo
nghĩa hình học. Mà mọi phép tính phía sau đều đọc độ dài đó: khoảng cách của KNN,
điểm `w · x` của SVM. Mô hình học nhầm rằng "tin dài thì đặc biệt", trong khi độ
dài chỉ phản ánh công ty đó viết nhiều lời quảng cáo hay ít.

Đo khoảng cách từ **D1** tới các tin khác:

| | `‖D1‖` | `‖D2‖` | `‖D3‖` | `d(D1,D2)` | `d(D1,D3)` | Gần D1 nhất |
|---|---|---|---|---|---|---|
| **Không L2** | 3,870 | **8,430** | 3,813 | 8,055 | 5,433 | **D3 — sai nghề** ✗ |
| **Có L2** | 1,000 | 1,000 | 1,000 | 1,163 | 1,414 | **D2 — đúng nghề** ✓ |

Nhìn kỹ dòng đầu, vì nó phi lý đến mức đáng nhớ:

- D1 và D2 **chung bốn từ** (`nhân`, `viên`, `kinh`, `doanh`) → khoảng cách **8,055**
- D1 và D3 **không chung một từ nào** → khoảng cách **5,433**

Hai tin chẳng liên quan gì lại được xếp gần nhau hơn hai tin cùng nghề. Lý do rất
tầm thường: `‖D1‖ = 3,870` và `‖D3‖ = 3,813` gần bằng nhau — **cùng ngắn**.
Khoảng cách đang đo chênh lệch độ dài văn bản, không đo nội dung.

Mô hình dựa trên khoảng cách chịu đòn nặng nhất — xem
[note 4](04-ma-tran-thua-va-so-chieu.md) về vì sao KNN dễ sụp.

Tái lập bảng trên (kho khai báo ở [§5 · Tái lập](#tái-lập)):

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

for norm in (None, "l2"):
    X = TfidfVectorizer(sublinear_tf=True, norm=norm).fit_transform(kho.values()).toarray()
    print(norm, [round(np.linalg.norm(X[0] - X[i]), 3) for i in range(1, 5)])
```

### "Dài đúng 1" nghĩa là gì

Chia cả vector cho độ dài của chính nó. Nhưng "độ dài" ở đây là **định lý
Pythagore**, không phải phép cộng.

Vector 2 chiều `(3, 4)`. Độ dài của nó không phải 3 + 4 = 7, mà là:

```
√(3² + 4²) = √25 = 5      →      (3, 4) / 5 = (0,6 ; 0,8)
```

Kiểm tra kết quả:

- cộng **bình phương**: 0,6² + 0,8² = 0,36 + 0,64 = **1** ✓
- cộng thẳng: 0,6 + 0,8 = **1,4** ✗

> ⚠️ Chỗ này rất dễ nhớ nhầm. L2 **không** làm các giá trị cộng lại bằng 1 —
> nó làm chúng **cộng bình phương** lại bằng 1.

Vector D1 sau chuẩn hoá, kiểm lại bằng số thật:

| Từ | TF-IDF | Bình phương |
|---|---|---|
| tuyển | 0,5422 | 0,2940 |
| nhân | 0,4375 | 0,1914 |
| viên | 0,4375 | 0,1914 |
| doanh | 0,4375 | 0,1914 |
| kinh | 0,3631 | 0,1318 |
| **Cộng lại** | **2,2178** ✗ | **1,0000** ✓ |

Sau L2, mỗi tin là một **mũi tên dài đúng 1** xuất phát từ gốc. Mọi tin chỉ còn
khác nhau ở **hướng** — tức tỷ lệ các từ — chứ không còn khác nhau ở lượng chữ.

### Vì sao L2 chứ không phải L1

Chuẩn hoá sao cho **cộng thẳng** bằng 1 cũng có thật, tên là **L1**.

| | Ràng buộc | Đọc ra được gì |
|---|---|---|
| **L1** | Σ giá trị = 1 | Mỗi từ chiếm bao nhiêu **phần trăm** của tin |
| **L2** | Σ (giá trị)² = 1 | Mỗi tin là một mũi tên dài 1 |

L1 nghe trực giác hơn, và nó cũng chữa được bệnh tin dài. Dự án vẫn chọn L2 vì
một lý do hình học: **sau L2, tích vô hướng của hai tin chính bằng cosine góc
giữa chúng.** Quan hệ đó chỉ đúng với L2, vì Pythagore là hình học của khoảng
cách Euclid — mà `LinearSVC` (`w · x`) và KNN (khoảng cách) đều làm việc trong
đúng hình học đó. Chọn L1 thì các phép đo phía sau mất diễn giải gọn gàng này.

### Cái giá phải trả

**1. L2 xoá sạch thông tin độ dài.** Mà độ dài đôi khi có ích thật — tin tuyển vị
trí cấp cao thường mô tả dài hơn tin thời vụ. Dự án lấy lại nó bằng đường khác:
các cột số `desc_len`, `req_len`, `title_len` không đi qua TF-IDF nên không bị L2
chạm tới. Đây là một quyết định thiết kế đáng chú ý — **vứt thông tin ở kênh này
rồi đưa nó lại ở kênh khác dưới dạng sạch hơn**, thay vì để nó lẫn vào và làm
nhiễu phép đo khoảng cách. Xem [§7](#7-tf-idf-không-nhìn-thấy-gì).

**2. Tin dài bị loãng từ khoá.** D1 và D2 đều dài đúng 1 sau L2, nhưng chúng chia
ngân sách đó khác nhau:

| | Số ô khác 0 | `nhân` | `viên` | `doanh` | `kinh` |
|---|---|---|---|---|---|
| D1 — tin ngắn | 5 | **0,4375** | **0,4375** | **0,4375** | **0,3631** |
| D2 — tin dài, **cùng nghề** | 14 | 0,2009 | 0,2009 | 0,2009 | 0,1667 |

D2 phải chia độ dài 1 cho 14 chiều nên mỗi từ khoá nhạt đi hơn một nửa; D1 dồn hết
vào 5 chiều nên đậm. Phần lớn đây là hành vi **đúng** — trong một tin dài thì
`kinh doanh` thật sự chỉ chiếm một phần nhỏ nội dung. Nhưng nó cũng có mặt trái:
tin nào viết nhiều lời quảng cáo chung chung ("môi trường trẻ trung", "phúc lợi
hấp dẫn") sẽ tự làm loãng tín hiệu nghề của chính nó. Cách bù là **nhân trọng số
theo khối** để kéo lại phần đậm đã bị chia mỏng —
[09-lo-trinh.md Ưu tiên 3a](../09-lo-trinh.md#ưu-tiên-3--hai-đòn-bẩy-rẻ-cho-phân-lớp-làm-trước-khi-nghĩ-đến-dl).

---

## 5. Ma trận hiện hình

Bốn mục trên nói **từng bước**. Mục này in ra **vật thể cuối cùng** mà mô hình
thật sự nhận được — vẫn trên kho 5 tin ở [§4](#4-chuẩn-hoá-l2--vì-sao-mọi-tin-phải-dài-đúng-1),
đủ nhỏ để nhìn hết một lần.

### Bảng 1 · Từ điển — học một lần, dùng chung cho mọi tin

Đếm `df` = **số tin chứa** từ đó, rồi `idf = ln((1+5)/(1+df)) + 1`.

| `df` | `idf` | Gồm những từ nào |
|---|---|---|
| 3 | ln(6/4) + 1 = **1,4055** | `kinh` |
| 2 | ln(6/3) + 1 = **1,6931** | `doanh`, `kế`, `nhân`, `toán`, `viên` |
| 1 | ln(6/2) + 1 = **2,0986** | 22 từ còn lại |

Chỉ có **ba** giá trị `idf`, vì kho tí hon này chỉ có ba mức `df`. Từ càng hiếm,
`idf` càng lớn — `kinh` có mặt ở 3/5 tin nên bị dìm thấp nhất.

Đây là một **vector 28 số**, không phải ma trận. Nó không phụ thuộc vào tin nào.

### Bảng 2 · Dây chuyền tính, riêng tin D1

| Từ | 1 · `tf` thô | 2 · `1+ln(tf)` | 3 · `df` | 4 · `idf` | 5 · `tf × idf` | 6 · `÷ ‖D1‖` |
|---|---|---|---|---|---|---|
| tuyển | 1 | 1,0 | 1 | 2,0986 | 2,0986 | **0,5422** |
| nhân | 1 | 1,0 | 2 | 1,6931 | 1,6931 | **0,4375** |
| doanh | 1 | 1,0 | 2 | 1,6931 | 1,6931 | **0,4375** |
| viên | 1 | 1,0 | 2 | 1,6931 | 1,6931 | **0,4375** |
| kinh | 1 | 1,0 | 3 | 1,4055 | 1,4055 | **0,3631** |

```
‖D1‖ = √(2,0986² + 1,6931² + 1,6931² + 1,6931² + 1,4055²) = √14,9800 = 3,8704
```

Tổng bình phương hàng cuối = **1,000000** ✓

Cột 1 và cột 2 giống hệt nhau — **mọi `tf` của D1 đều bằng 1**, vì D1 không lặp
từ nào. Nghĩa là với riêng D1, TF-IDF thực chất chỉ còn là `idf` sau chuẩn hoá.
Bảng 4 lấy D2 để bật TF lên.

### Bảng 3 · Ma trận cuối — 5 hàng × 28 cột

Mỗi **hàng** là một tin, mỗi **cột** là một từ. Dấu `·` là số 0.

```
    chăm    có  công    cũ doanh  dựng  hàng   hợp khách  kinh  kiếm    kế    kỹ nghiệm
D1     ·     ·     ·     ·  0.44     ·     ·     ·     ·  0.36     ·     ·     ·      ·
D2  0.25     ·     ·  0.25  0.20     ·  0.42     ·  0.42  0.17  0.25     ·     ·      ·
D3     ·     ·     ·     ·     ·     ·     ·  0.55     ·     ·     ·  0.44     ·      ·
D4     ·  0.46     ·     ·     ·     ·     ·     ·     ·  0.31     ·  0.37     ·   0.46
D5     ·     ·  0.41     ·     ·  0.41     ·     ·     ·     ·     ·     ·  0.41      ·

    nhân   phụ   sóc    sư  toán trách trình trưởng tuyển   tìm  tổng  viên    và   xây
D1  0.44     ·     ·     ·     ·     ·     ·      ·  0.54     ·     ·  0.44     ·     ·
D2  0.20  0.25  0.25     ·     ·  0.25     ·      ·     ·  0.25     ·  0.20  0.25     ·
D3     ·     ·     ·     ·  0.44     ·     ·      ·     ·     ·  0.55     ·     ·     ·
D4     ·     ·     ·     ·  0.37     ·     ·   0.46     ·     ·     ·     ·     ·     ·
D5     ·     ·     ·  0.41     ·     ·  0.41      ·     ·     ·     ·     ·     ·  0.41
```

**Đọc theo hàng = một tin.** D1 có đúng 5 ô khác 0 — đúng 5 từ của nó. 23 cột còn
lại là 0: D1 "không chứa" mọi từ khác. Tổng bình phương mỗi hàng bằng 1.

**Đọc theo cột = một từ, xuyên qua cả kho.** Cột `kinh` khác 0 ở D1, D2, D4. Cột
`sư` chỉ khác 0 ở D5.

**Cùng một từ, giá trị khác nhau giữa các tin.** `nhân` được 0,44 ở D1 nhưng chỉ
0,20 ở D2 — cùng `idf`, cùng `tf`, chênh nhau **chỉ vì L2**. Đây là pha loãng ở
[§4](#4-chuẩn-hoá-l2--vì-sao-mọi-tin-phải-dài-đúng-1), nhìn thấy trực tiếp trên
ma trận: hàng D2 trải mỏng ra 14 cột, hàng D1 dồn vào 5 cột.

**Hàng D1 và hàng D3 không giao nhau ở một cột nào.** Đó chính là cặp mà [§4](#bệnh)
cho thấy bị xếp *gần nhau nhất* khi thiếu L2 — bằng chứng bằng mắt rằng khoảng
cách khi đó không hề đo nội dung.

**Ma trận không biết nghề.** D1 và D2 cùng nghề kinh doanh, và điều đó chỉ lộ ra
qua bốn cột chung. D3 và D4 cùng nghề kế toán, chung hai cột. Không có cột nào tên
là "nghề" cả — biết được nghề là việc của nhãn và của mô hình phân lớp, không phải
của ma trận này. Xem [§7](#7-tf-idf-không-nhìn-thấy-gì).

**35 ô khác 0 trên 140 ô — 25 %.** Ở dữ liệu thật tỷ lệ này nhỏ hơn hàng trăm lần;
đó là ý nghĩa của "thưa", xem [note 4](04-ma-tran-thua-va-so-chieu.md).

### Bảng 4 · TF thật sự hoạt động — khi từ lặp lại

D2 lặp `khách` và `hàng` hai lần ("tìm kiếm **khách hàng** mới và chăm sóc
**khách hàng** cũ"). Đây là chỗ duy nhất trong kho có `tf > 1`:

| Từ trong D2 | `tf` | `1+ln(tf)` | `idf` | `tf × idf` | Sau L2 |
|---|---|---|---|---|---|
| khách | **2** | **1,6931** | 2,0986 | 3,5533 | **0,4215** |
| hàng | **2** | **1,6931** | 2,0986 | 3,5533 | **0,4215** |
| nhân | 1 | 1,0000 | 1,6931 | 1,6931 | 0,2009 |
| kinh | 1 | 1,0000 | 1,4055 | 1,4055 | 0,1667 |

`khách` mạnh gấp **hơn hai lần** `nhân` trong cùng một tin. Hai lực cộng dồn:
nó lặp (TF cao) **và** nó hiếm trong kho (IDF cao). Đó đúng là định nghĩa của
"từ khoá đặc trưng cho tài liệu này".

Nếu để `tf` thô thay vì `1+ln(tf)`, `khách` sẽ được `2 × 2,0986 = 4,1972` thay vì
`3,5533` — đậm hơn 18 %. Và một tin nhồi từ khoá SEO hai mươi lần sẽ dìm mọi từ
khác của chính nó xuống gần 0, vì L2 buộc cả hàng phải dài đúng 1. Đó là lý do
dự án bật `sublinear_tf=True`.

### Tái lập

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

kho = {"D1": "tuyển nhân viên kinh doanh",
       "D2": "nhân viên kinh doanh phụ trách tìm kiếm khách hàng "
             "và chăm sóc khách hàng cũ",
       "D3": "kế toán tổng hợp",
       "D4": "kế toán trưởng có kinh nghiệm",
       "D5": "kỹ sư xây dựng công trình"}
V = TfidfVectorizer(sublinear_tf=True, norm="l2")
X = V.fit_transform(kho.values())
print(pd.DataFrame(X.toarray().round(2), index=kho,
                   columns=V.get_feature_names_out()).replace(0, "·").to_string())
```

---

## 6. Ví dụ tính tay — trên dữ liệu thật

Lấy tiêu đề phổ biến nhất trong kho: **"Nhân viên kinh doanh"** (754 tin trong
train). Với `ngram_range=(1,2)`, nó sinh ra **7 đặc trưng**:

| Đặc trưng | Loại | `df` (trên 33.396 tin) | `idf` | **TF-IDF** |
|---|---|---|---|---|
| `viên kinh` | bigram | 3.321 · 9,94 % | 3,308 | **0,491** |
| `kinh doanh` | bigram | 4.764 · 14,27 % | 2,947 | **0,437** |
| `doanh` | unigram | 4.899 · 14,67 % | 2,919 | **0,433** |
| `kinh` | unigram | 5.118 · 15,33 % | 2,876 | **0,427** |
| `nhân viên` | bigram | 14.494 · 43,40 % | 1,835 | **0,272** |
| `nhân` | unigram | 14.951 · 44,77 % | 1,804 | **0,268** |
| `viên` | unigram | 19.181 · **57,44 %** | 1,554 | **0,231** |

Tái lập bằng `_word_tfidf(PrepConfig(), max_features=60_000)` khớp trên cột
`job_title` của `train.parquet`. Cả 7 ô đều có `tf = 1` (mỗi từ xuất hiện một lần),
nên cột TF-IDF ở đây chính là `idf` sau chuẩn hoá L2.

### Bốn điều đọc ra từ bảng này

**a. Vector có 6.835 chiều nhưng chỉ 7 ô khác 0.** Từ điển tiêu đề đúng bằng
**6.835** từ — chính là con số trong sơ đồ ở
[06-mo-hinh-phan-lop.md](../06-mo-hinh-phan-lop.md). 6.828 chiều còn lại là số 0.
Đó là ý nghĩa của "thưa" — [note 4](04-ma-tran-thua-va-so-chieu.md).

**b. `viên` yếu nhất (0,231) dù nó là từ phổ biến nhất.** Đó chính là IDF làm việc:
`viên` có mặt ở 57 % số tin nên nó gần như không phân biệt được gì. Nó *có mặt* ở
nhân viên, chuyên viên, kỹ thuật viên, giáo viên — bốn nghề khác nhau.

**c. `viên kinh` mạnh nhất (0,491) — và đây là bigram vô nghĩa về mặt ngữ pháp.**
"viên kinh" không phải một từ tiếng Việt. Nhưng nó **hiếm** (9,94 %), nên nó là
dấu vân tay rất tốt cho đúng cụm "nhân viên kinh doanh". Máy không cần hiểu ngữ
pháp — nó chỉ cần một chuỗi ký tự phân biệt được.

**d. `viên` suýt bị loại.** Dự án đặt `max_df=0.6` — mọi từ có mặt ở hơn 60 % tài
liệu bị vứt thẳng. `viên` ở **57,44 %**, sát ngưỡng. Từ `công ty` phổ biến hơn
trong *mô tả* thì bị `max_df` cắt ở khối đó. Đây là **bộ lọc từ dừng tự động, học
từ chính dữ liệu** — và là lý do bước bỏ từ dừng thủ công đo ra chỉ **+0,0002**.

---

## 7. TF-IDF **không** nhìn thấy gì

Biết giới hạn của công cụ quan trọng ngang biết cách dùng nó.

| Nó không thấy | Hệ quả trong dự án này |
|---|---|
| **Thứ tự từ** | "không yêu cầu kinh nghiệm" và "yêu cầu kinh nghiệm" chỉ khác nhau ở một token `không`. Bigram vá được một phần |
| **Nghĩa / từ đồng nghĩa** | `NV` và `nhân viên` là hai chiều rời nhau cho tới khi bước mở viết tắt gộp chúng lại |
| **Phủ định, mỉa mai, ngữ cảnh xa** | Đây là lý do trần của mô hình túi từ nằm dưới trần của PhoBERT |
| **Độ dài văn bản** | [Chuẩn hoá L2](#4-chuẩn-hoá-l2--vì-sao-mọi-tin-phải-dài-đúng-1) xoá luôn thông tin này. Muốn giữ thì phải thêm cột riêng — đúng là điều `desc_len`, `req_len`, `title_len` làm |

Bảng này giải thích vì sao dự án **thêm** 14 cột số và 3 cột one-hot bên cạnh
TF-IDF: chúng mang đúng những thứ TF-IDF vứt đi. Và bảng trọng số ở
[05-dac-trung-tfidf.md §9](../05-dac-trung-tfidf.md#9-khối-nào-thật-sự-được-dùng)
cho thấy mô hình **rất coi trọng** những cột đó.

---

## Quay lại thực tế dự án

- [05-dac-trung-tfidf.md](../05-dac-trung-tfidf.md) — từng tham số TF-IDF của dự án, và bỏ đi thì sao
- [note 3 — n-gram và ranh giới từ](03-ngram-va-ranh-gioi-tu.md) — vì sao `(1,2)` mà không phải `(1,1)`
- [note 4 — ma trận thưa](04-ma-tran-thua-va-so-chieu.md) — 7 ô khác 0 trên 6.835 chiều nghĩa là gì
