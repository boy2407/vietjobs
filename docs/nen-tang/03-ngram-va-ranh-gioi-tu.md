[← TF-IDF là gì](02-tf-idf-la-gi.md) · [Nền tảng](00-index.md) · [Ma trận thưa →](04-ma-tran-thua-va-so-chieu.md)

# 3. n-gram và ranh giới từ

Note này giải thích một kết quả gây bất ngờ trong dự án: **bước tách từ tiếng Việt
— bước "đúng đắn" nhất về mặt ngôn ngữ học — đo ra không giúp gì và đã bị tắt.**

Lý do không phải vì tách từ vô nghĩa. Lý do là `ngram_range=(1,2)` đã giải sẵn
đúng bài toán đó.

---

## 1. n-gram là gì

n-gram = chuỗi `n` token liền nhau.

```
"Nhân viên kinh doanh"

unigram (1-gram):  nhân · viên · kinh · doanh
bigram  (2-gram):  nhân viên · viên kinh · kinh doanh
trigram (3-gram):  nhân viên kinh · viên kinh doanh
```

`ngram_range=(1,2)` nghĩa là: lấy **cả** unigram **và** bigram. Câu bốn chữ trên
sinh ra 4 + 3 = **7 đặc trưng**, đúng như bảng tính tay ở
[note 2](02-tf-idf-la-gi.md#6-ví-dụ-tính-tay--trên-dữ-liệu-thật).

Bigram là cách túi từ vá lại một phần thông tin thứ tự đã vứt đi: nó không biết
toàn bộ trật tự câu, nhưng nó biết **hai chữ nào đứng cạnh nhau**.

---

## 2. Vấn đề riêng của tiếng Việt

Hầu hết công cụ NLP mặc định một giả định lấy từ tiếng Anh:

> **Khoảng trắng tách từ.**

Với tiếng Anh, giả định đó gần đúng: `"salesperson"` là một từ, một token.
Với tiếng Việt, giả định đó **sai**. Tiếng Việt là ngôn ngữ **đơn lập**: khoảng
trắng tách **âm tiết**, không tách **từ**.

```
tiếng Anh:   salesperson              →  1 token,  đúng là 1 từ
tiếng Việt:  nhân viên kinh doanh     →  4 token,  thật ra là 2 từ
```

### Hậu quả: âm tiết "viên" nuốt bốn nghề

Đo được trong kho VietJobs: âm tiết `viên` xuất hiện **27.053 lần** trong tiêu đề,
đứng sau **70 âm tiết khác nhau**:

| Cụm | Số lần | Là nghề gì |
|---|---|---|
| nhân **viên** | 20.138 | nhân viên |
| chuyên **viên** | 4.858 | chuyên viên |
| kỹ thuật **viên** | 498 | kỹ thuật viên |
| giáo **viên** | … | giáo viên |

Ở mức **chỉ unigram**, cả bốn nghề đổ chung vào một chiều `viên`. Chiều đó xuất
hiện ở 57,44 % số tin nên gần như vô dụng — nó không phân biệt được gì.

---

## 3. Hai cách sửa — và chúng làm cùng một việc

### Cách A — tách từ (`underthesea`)

Chạy một mô hình phân đoạn tiếng Việt, nối các âm tiết thuộc cùng một từ bằng
gạch dưới:

```
"Nhân viên kinh doanh"  →  "Nhân_viên kinh_doanh"
```

Bây giờ `nhân_viên` là một token riêng, `chuyên_viên` là một token riêng.

### Cách B — bigram (`ngram_range=(1,2)`)

Không cần biết gì về tiếng Việt. Chỉ cần ghép mọi cặp chữ liền nhau:

```
"Nhân viên kinh doanh"  →  ... + "nhân viên" + "viên kinh" + "kinh doanh"
```

Bây giờ `nhân viên` cũng là một đặc trưng riêng, `chuyên viên` cũng vậy.

### So sánh

| | Tách từ | Bigram |
|---|---|---|
| Có `nhân_viên` / `nhân viên` là một đặc trưng riêng | ✅ | ✅ |
| Cần mô hình ngôn ngữ | ✅ underthesea | ❌ không cần gì |
| Chi phí | **1.533,7 giây** trên 48k tin | gần như bằng 0 |
| Sinh thêm đặc trưng rác (`viên kinh`) | ❌ | ✅ có |
| Sai thì sao | Cắt sai từ → token sai hoàn toàn | Không có khái niệm "sai" |

**Cả hai cách giải cùng một bài toán.** Mà bigram đã bật sẵn từ đầu.

---

## 4. Kết quả đo được

Cùng `C = 0,02`, chỉ bật/tắt bước tách từ:

| Cấu hình | macro-F1 (val) | Run |
|---|---|---|
| Chỉ chuẩn tỉnh | **0,6050** | `cat-T-svm-C0.02` |
| Chuẩn tỉnh + **tách từ** | 0,5998 | `cat-R-svm-C0.02-seg` |

Tách từ làm **giảm** 0,0052 — nằm trong nhiễu (σ ≈ 0,0077), nên kết luận đúng là
"không giúp gì", không phải "làm hại".

**Vì sao nó không giúp:** bigram đã bắt `nhân viên` rồi. Bước 5 giải một bài toán
mà bộ vectơ hoá đã giải sẵn — chỉ khác là nó tốn 25 phút và cần một thư viện ngoài.

**Vì sao nó hơi hại:** tách từ **giảm** số đặc trưng phân biệt. Sau khi tách,
`nhân_viên` là một token, và bigram của bản đã tách là `nhân_viên kinh_doanh` —
không còn `viên kinh` nữa. Mà theo bảng ở [note 2](02-tf-idf-la-gi.md), `viên kinh`
là đặc trưng **mạnh nhất** trong cả bảy (TF-IDF 0,491), chính vì nó hiếm.
Tách từ đã xoá mất đặc trưng mạnh nhất của tiêu đề phổ biến nhất.

---

## 5. Còn tin viết không dấu thì sao — kênh ký tự

**2.148 tiêu đề (4,50 %)** viết hoàn toàn không dấu:

```
"Nhan Vien Kinh Doanh"    ← kênh từ không cách nào khớp với "Nhân Viên Kinh Doanh"
```

Không thể sửa bằng cách bỏ dấu cả kho, vì **dấu tiếng Việt mang nghĩa** —
`má / mà / mả / mã / mạ` là năm từ khác nhau. Bỏ dấu toàn bộ là gộp năm chiều
có nghĩa thành một chiều vô nghĩa.

Giải pháp: một **kênh thứ hai** chạy song song, dùng n-gram **ký tự** trên bản đã
bỏ dấu:

```
analyzer="char_wb", ngram_range=(3, 5), preprocessor=fold_accents

"nhan vien"  →  nha · han · nhan · vie · ien · vien · ...
"nhân viên"  →  (sau khi bỏ dấu) nha · han · nhan · vie · ien · vien · ...
                                  ↑ khớp nhau
```

Kênh từ giữ nghĩa của dấu; kênh ký tự bắc cầu cho tin không dấu. **Bổ sung, không
thay thế.**

Kết quả đo: 0,6024 so với 0,6050 — cũng **không giúp**, và cũng bị tắt.
4,50 % số tin quá ít để bù cho 60.000 chiều nhiễu mà kênh này thêm vào.

---

## 6. Bài học rút ra

**Trước khi thêm một bước xử lý ngôn ngữ, hãy hỏi: bộ vectơ hoá đã làm việc đó chưa?**

Ba cặp trùng lặp có thật trong dự án này:

| Bước thủ công | Đã được làm sẵn bởi | Kết quả đo |
|---|---|---|
| Tách từ | `ngram_range=(1,2)` | −0,0052 |
| Bỏ từ dừng | `max_df=0.6` và `idf` | +0,0002 |
| Kênh gấp dấu | (không trùng, nhưng quá ít dữ liệu) | −0,0026 |

Bước tiền xử lý "đúng về mặt lý thuyết" không tự động là bước có ích. **Chỉ có
dòng ablation mới quyết định được** — đó là quy tắc số 6 trong
[03-protocol.md](../03-protocol.md).

---

## Quay lại thực tế dự án

- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — chín bước và bảng ablation đầy đủ
- [05-dac-trung-tfidf.md](../05-dac-trung-tfidf.md#4-từng-tham-số-tf-idf--và-bỏ-đi-thì-sao) — `ngram_range` và các tham số anh em
- [note 2 — TF-IDF là gì](02-tf-idf-la-gi.md) — bảng tính tay 7 đặc trưng
