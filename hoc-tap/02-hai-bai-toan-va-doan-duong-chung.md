[← Từ vựng tối thiểu](01-tu-vung-toi-thieu.md) · [Mục lục](00-index.md) · [Kiến trúc A →](03-kien-truc-a-phobert-dense.md)

# 2. Hai bài toán, và đoạn đường chúng đi chung

Dự án có hai bài toán học sâu. Nhìn thì tưởng khác nhau nhiều, thật ra **chỉ khác
nhau ở ba chỗ** — và cả ba chỗ đó đều nằm ở cuối đường ống.

---

## 1. Hai bài toán

**Bài toán 1 — phân loại ngành nghề (`category`).** Đọc một tin tuyển dụng, chọn
**một trong 16** nhãn ngành nghề. Đầu ra là một *lựa chọn*.

**Bài toán 2 — dự đoán lương (`salary`).** Đọc một tin tuyển dụng, trả về **một con
số**: mức lương giữa khoảng, đơn vị triệu đồng/tháng. Đầu ra là một *số thực*. Chỉ
những dòng có công khai lương (`salary_disclosed == 1`) mới được dùng để huấn luyện.

Bảng neo cho cả bộ ghi chú — mỗi dòng là một chỗ khác nhau, kèm bài sẽ giải thích nó:

| | phân loại ngành nghề | dự đoán lương |
|---|---|---|
| cái gì rời khỏi mạng | 16 điểm số | 1 con số |
| biến thành câu trả lời bằng | `softmax` + `argmax` ([bài 3 §10](03-kien-truc-a-phobert-dense.md#10-đầu-ra-a--16-điểm-số-thành-một-nhãn)) | `expm1` ([bài 3 §11](03-kien-truc-a-phobert-dense.md#11-đầu-ra-b--một-con-số-thành-một-mức-lương)) |
| huấn luyện chống lại | entropy chéo | Huber (`SmoothL1Loss`) |
| chấm điểm bằng | macro-F1 | MAE, đơn vị triệu đồng ([bài 5](05-vong-huan-luyen-va-do-luong.md)) |
| học trên dòng nào | mọi dòng có nhãn | chỉ dòng công khai lương |
| được đọc họ cột nào | `raw` | `masked` — xem §3 dưới đây |

**Điểm chốt cho người mới:** ngoài sáu dòng đó ra, **toàn bộ phần còn lại là y hệt
nhau**. Cùng ba trường văn bản, cùng cách ghép chuỗi, cùng bộ mã hoá, cùng khối
mạng, cùng vòng huấn luyện. Đổi bài toán = đổi lớp cuối và đổi hàm mất mát.

Đó cũng là lý do trong repo có **hai mạng riêng**, không phải một mạng hai đầu: ở bậc
mốc cơ sở, mỗi bài chạy riêng để biết từng bài đạt bao nhiêu **trước khi** gộp. Gộp mà
tệ đi thì còn biết là tệ đi so với cái gì.

---

## 2. Đoạn đường chung — từ tin tuyển dụng thành một chuỗi

Bước này nằm trọn trong một tệp: [`src/vietjobs/dl/text.py`](../src/vietjobs/dl/text.py),
40 dòng, và **cố tình không import torch** để phần kiểm thử chạy được trong môi
trường không có torch.

Ba trường văn bản, ghép theo đúng thứ tự này, ngăn bằng `" . "`:

```
job_title . description . requirements_text
```

**Vì sao tiêu đề đứng đầu?** Vì bước sau sẽ cắt chuỗi còn 256 token. Phần bị cắt bỏ
luôn là phần *đuôi*. Đặt tiêu đề lên đầu nghĩa là: thứ bị vứt đi là đoạn mô tả dài
dòng, không phải cái tên nghề — thứ nhận dạng nghề mạnh nhất.

Sau khi ghép, nhiều khoảng trắng liên tiếp được nén lại thành một. Kết quả: mỗi tin
tuyển dụng là **một chuỗi ký tự**.

---

## 3. Cái cổng duy nhất — và vì sao trộn cột là rò rỉ lương

Mỗi trường văn bản trong dữ liệu có **bốn bản sao**:

| Bản sao | Là gì | Ai được đọc |
|---|---|---|
| `description` | văn bản đã làm sạch | chỉ bài phân loại |
| `description_seg` | thêm bước tách từ | chỉ bài phân loại |
| `description_masked` | đã **che** mọi con số lương | chỉ bài lương |
| `description_masked_seg` | che rồi tách từ | chỉ bài lương |

Vì sao phải che? Vì mô tả công việc thường viết thẳng "lương 15 triệu". Nếu bài toán
lương được đọc câu đó, nó không học gì cả — nó **chép đáp án**. Điểm sẽ đẹp, và hoàn
toàn vô nghĩa: ra tin thật không ghi lương thì mô hình mù.

Chỗ quyết định đọc bản sao nào chỉ có **một**: hàm `resolve_column` trong
[`src/vietjobs/features.py`](../src/vietjobs/features.py). Cả đường học máy cũ lẫn
đường học sâu đều đi qua đúng cái cổng đó. Một cánh cửa thì kiểm thử được; mười nhánh
`if` rải khắp mã thì không.

> **Hệ quả cho học sâu:** bộ đệm vector PhoBERT trong `artifacts/embeddings/` được
> tách thành **hai họ tệp** — họ `raw` cho phân loại, họ `masked` cho lương. Tên họ
> nằm ngay trong tên tệp (`train-masked-len256.npy`) chính vì lý do này. Nạp nhầm tệp
> `raw` cho bài lương là một vụ rò rỉ **không báo lỗi**, không crash, chỉ làm điểm đẹp
> lên. Hai bộ kiểm thử canh chỗ này: `tests/test_no_leak.py` và `tests/test_dl_text.py`.

Bài 4 (TextCNN) chịu **đúng ràng buộc này**, không có ngoại lệ: nếu dựng từ điển cho
CNN thì cũng phải dựng riêng cho từng họ cột.

---

## 4. Ba tập dữ liệu, và vì sao `test` gần như không được chạm

| Tập | Số dòng | Trong đó có lương | Dùng để làm gì |
|---|---|---|---|
| `train` | 34.354 | 24.622 | Máy học tham số trên đây |
| `dev` | 3.812 | 2.698 | Người chọn cấu hình, chọn epoch dừng, so mô hình |
| `test` | 9.541 | 6.773 | Đo **một lần duy nhất**, ở cuối |

Lý do `test` phải nằm im: mỗi lần bạn nhìn một tập rồi sửa mô hình theo cái nhìn thấy,
bạn đã **dùng** tập đó để ra quyết định. Sau vài chục lần như thế, điểm trên tập đó
không còn là ước lượng cho dữ liệu mới nữa — nó thành điểm của bài thi đã có đáp án.
Trong mã, chạm `test` phải bật cờ `--confirm-test`; đó là một rào chắn cố ý.

Điểm mà người mới hay bỏ sót: **dừng sớm cũng là một lần dùng `dev`** — chi tiết ở
[bài 5](05-vong-huan-luyen-va-do-luong.md#3-dừng-sớm--và-vì-sao-nó-đã-tiêu-thụ-tập-dev).

Một chi tiết nữa: dữ liệu được chia **theo nhóm tin trùng lặp**, không chia theo dòng.
Hai tin gần giống nhau của cùng một công ty phải nằm cùng một tập — nếu một cái ở
`train`, một cái ở `dev` thì mô hình chỉ cần nhớ, không cần hiểu.

---

## 5. Tóm lại

> Đường ống nào cũng gồm bốn đoạn: **văn bản → số → mạng → câu trả lời**.
> Bài 3 và bài 4 khác nhau hoàn toàn ở đoạn "văn bản → số", giống nhau gần như
> hoàn toàn ở hai đoạn còn lại.

## Đọc tiếp

- [Bài 3 · kiến trúc A — PhoBERT đóng băng + dense](03-kien-truc-a-phobert-dense.md)
- [`docs/nen-tang/05-ro-ri-du-lieu.md`](../docs/nen-tang/05-ro-ri-du-lieu.md) — ba kiểu
  rò rỉ dữ liệu, và vì sao **không kiểu nào báo lỗi**
