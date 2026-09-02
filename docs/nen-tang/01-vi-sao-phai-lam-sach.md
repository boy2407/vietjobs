[← Nền tảng](00-index.md) · [TF-IDF là gì →](02-tf-idf-la-gi.md)

# 1. Vì sao phải làm sạch dữ liệu

Câu trả lời quen thuộc là "rác vào rác ra". Câu đó **không sai nhưng vô dụng** —
nó không cho biết bước nào đáng làm và bước nào là nghi lễ.

Có hai lý do thật, cụ thể, và đo được.

---

## Lý do 1 — máy đếm chữ, không đọc chữ

Đây là điều khó chấp nhận nhất với người mới, nên nói thẳng:

> Mô hình không hiểu tiếng Việt. Nó **đếm chuỗi ký tự**.

Với `TfidfVectorizer`, `"hoà"` và `"hòa"` là hai chuỗi ký tự khác nhau, nên chúng
là **hai đặc trưng riêng biệt**, giống hệt như `"hoà"` và `"máy xúc"` là hai đặc
trưng riêng biệt. Máy không có cách nào biết hai cái đầu là cùng một từ.

### Chuyện gì xảy ra khi một khái niệm bị xẻ đôi

Giả sử từ "hoà" xuất hiện trong 1.000 tin, nhưng viết theo hai kiểu:

| | Nếu **không** chuẩn hoá | Nếu **có** chuẩn hoá |
|---|---|---|
| Chiều `hoà` | 600 tin | **1.000 tin** |
| Chiều `hòa` | 400 tin | — |

Ba hậu quả, hậu quả sau nặng hơn hậu quả trước:

**a. Mỗi mảnh yếu đi.** Mô hình học trọng số cho từng chiều. Một chiều thấy 600
ví dụ học được ít hơn một chiều thấy 1.000 ví dụ.

**b. Hai mảnh học ra hai trọng số khác nhau** — thậm chí ngược dấu, nếu ngẫu nhiên
kiểu viết `hòa` hay xuất hiện trong ngành nào đó. Mô hình học một mối tương quan
**giả**, đến từ thói quen gõ phím chứ không từ nội dung.

**c. Lúc dự đoán, tin mới chỉ kích hoạt một trong hai chiều.** Nếu người dùng gõ
kiểu ít phổ biến hơn, mô hình dùng chiều yếu hơn — và trả lời kém hơn, một cách
hoàn toàn im lặng.

### Việc này to đến đâu trong kho VietJobs

Toàn số đo thật, từ `python scripts/measure_vitext.py`:

| Dạng "xẻ chiều" | Quy mô |
|---|---|
| Chữ tổ hợp Unicode chưa gộp (`ế` viết bằng 2–3 code point) | **529 dòng (1,11 %)** |
| Mô tả dùng cả hai lối đặt dấu `hoà`/`hòa` | 76,6 % dùng kiểu này, 30,3 % dùng kiểu kia — **phần lớn tin lẫn cả hai** |
| Mô tả chứa viết tắt chưa mở (`NV`, `BHXH`, `CSKH`) | **3.436 mô tả (7,2 %)** |
| Địa điểm bị vụn (`hà đông` không nối được với `hà nội`) | 984 chuỗi → chỉ còn **265** sau chuẩn hoá; **7.481 dòng (15,7 %)** đổi giá trị |
| Tiêu đề viết **không dấu** hoàn toàn | **2.148 tiêu đề (4,50 %)** |

Nhìn riêng thì mỗi con số nhỏ. Nhưng chúng chồng lên nhau, và mỗi cái đều xẻ đúng
những từ **phổ biến nhất** — tức những từ mô hình dựa vào nhiều nhất.

---

## Lý do 2 — chặn mô hình nhìn thấy đáp án

Lý do này quan trọng hơn lý do 1 rất nhiều, và hoàn toàn khác về bản chất.

Lý do 1 nói về **hiệu năng**: không làm thì mô hình kém hơn một chút.
Lý do 2 nói về **tính đúng đắn**: không làm thì mọi con số bạn báo cáo đều **sai**,
và sai theo hướng đẹp lên.

Trong dự án này có hai đường rò rỉ, cả hai đều bị chặn ở khâu làm sạch:

**a. Đáp án nằm trong chính đầu vào.** Bài toán 2 dự đoán mức lương. Nhưng
**10,65 %** dòng nhắc lại con số lương ngay trong ô *phúc lợi*. Không che thì mô
hình chỉ cần đọc lại con số đó — R² đẹp trong báo cáo, hỏng ngoài đời.

**b. Cùng một tin nằm ở cả train lẫn test.** Nhà tuyển dụng đăng lại tin nhiều
lần: **12.808 dòng (26,8 %)** là tin đăng lại. Chia tập theo dòng thì mô hình chỉ
cần **thuộc lòng** là có điểm cao trên test — và điểm đó không nói gì về khả năng
tổng quát hoá.

Chi tiết đầy đủ ở [note 5 — rò rỉ dữ liệu](05-ro-ri-du-lieu.md).

---

## Hệ quả: hai loại bước tiền xử lý

Từ hai lý do trên suy ra một cách phân loại rất hữu ích, và nó là xương sống
của cả dự án:

| Loại | Mục đích | Chứng minh bằng gì | Gỡ được không |
|---|---|---|---|
| **Bước hiệu năng** | Làm điểm cao lên | Một dòng ablation trong [04-results.md](../04-results.md) | **Có** — không cải thiện thì gỡ |
| **Bước đúng đắn** | Làm con số đo được có nghĩa | Không thể chứng minh bằng điểm số | **Không bao giờ** |

Trong chín bước xử lý tiếng Việt, bốn bước thuộc loại thứ hai (NFC, chuẩn dấu
thanh, che lương, khoá gộp nhóm) và **không** bước nào trong đó làm điểm cao lên
đáng kể. Chúng vẫn ở lại. Ba bước thuộc loại thứ nhất đo ra không giúp gì và
**đã bị tắt**.

Đây là điều phân biệt một pipeline có kỷ luật với một pipeline dài đầy nghi lễ:
không phải "làm nhiều bước", mà là **biết mỗi bước ở đó vì lý do nào**.

---

## Một cảnh báo ngược lại: làm sạch quá tay cũng phá

Làm sạch không phải càng nhiều càng tốt. Ba ví dụ có thật trong dự án này:

**Bỏ dấu tiếng Việt.** `strip_accents` là mặc định phổ biến khi làm TF-IDF cho
tiếng Anh. Ở tiếng Việt nó phá: `má` (mẹ), `mà` (liên từ), `mả` (mộ), `mã` (mã số),
`mạ` (mạ kim loại) là **năm từ khác nhau**. Bỏ dấu là gộp năm chiều có nghĩa
thành một chiều vô nghĩa.

**Bỏ từ dừng máy móc.** Bỏ chữ "không" sẽ biến *"không yêu cầu kinh nghiệm"*
thành *"yêu cầu kinh nghiệm"* — **đảo ngược nghĩa**. Danh sách từ dừng của dự án
cố ý chừa `không`, `chưa`, `trên/dưới`, `tối thiểu`.

**Mở viết tắt bừa.** Mở `TP` thành `trưởng phòng` trong `"TP HCM"` tạo ra một tín
hiệu **giả** — mô hình sẽ tưởng mọi tin ở Sài Gòn đều tuyển trưởng phòng.
**Mở sai tệ hơn không mở**, nên các viết tắt nhập nhằng chỉ được mở khi có ngữ
cảnh khớp.

Quy tắc rút ra: **mỗi phép làm sạch là một phép gộp thông tin, và gộp là mất mát
không lấy lại được.** Chỉ gộp khi chắc chắn hai thứ được gộp thật sự là một.

---

## Quay lại thực tế dự án

- [01-data-audit.md](../01-data-audit.md) — làm sạch được thực hiện thế nào
- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — chín bước, và bước nào sống sót
- [note 5 — rò rỉ dữ liệu](05-ro-ri-du-lieu.md) — lý do 2, nói kỹ
