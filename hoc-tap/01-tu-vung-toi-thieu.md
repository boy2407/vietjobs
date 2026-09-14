[← Mục lục](00-index.md) · [Hai bài toán →](02-hai-bai-toan-va-doan-duong-chung.md)

# 1. Từ vựng tối thiểu

Mười khái niệm. Đọc xong mười cái này là đủ để hiểu bốn bài còn lại. Chưa cần biết
đạo hàm, chưa cần biết đại số tuyến tính.

---

## 1. Mười khái niệm, mỗi cái một câu

| Khái niệm | Một câu | Tương tự đời thường |
|---|---|---|
| **Tensor** | Một mảng số nhiều chiều. Một tin tuyển dụng sau khi qua PhoBERT là một tensor 768 số | Một bảng Excel, nhưng có thể nhiều hơn hai chiều |
| **Tham số** (weight, bias) | Những con số **máy tự chỉnh**, khác với siêu tham số là con số **người đặt trước** | Núm vặn trong máy: tham số là núm máy tự vặn, siêu tham số là núm bạn vặn tay |
| **Lớp** (layer) | Một phép biến đổi có tham số, nhận một tensor và trả ra một tensor khác | Một công đoạn trong dây chuyền |
| **Mạng** (network) | Nhiều lớp xếp nối tiếp nhau | Cả dây chuyền |
| **Hàm mất mát** (loss) | **Một** con số duy nhất nói "đoán sai bao nhiêu". Đây là thứ duy nhất máy được phép làm nhỏ đi | Điểm trừ trong bài thi |
| **Gradient** | Hướng và độ dốc: nếu nhích tham số này lên một chút thì mất mát tăng hay giảm | Bạn đang đứng trên đồi sương mù, gradient là hướng dốc xuống dưới chân |
| **Học suất** (learning rate) | Mỗi bước đi dài bao nhiêu theo hướng đó | Bước chân dài hay ngắn. Dài quá thì nhảy qua đáy thung lũng |
| **Epoch** | Một lượt đi hết toàn bộ dữ liệu huấn luyện | Đọc hết quyển sách một lần |
| **Batch** | Một nhúm dòng được xử lý cùng lúc trước khi chỉnh tham số một lần | Chấm 256 bài rồi mới rút kinh nghiệm, thay vì chấm từng bài |
| **Quá khớp** (overfitting) | Mô hình thuộc lòng dữ liệu huấn luyện, ra dữ liệu mới thì tệ | Học tủ |

---

## 2. Một mạng nơ-ron thật ra là gì

Bỏ hết hình vẽ nơ-ron và bộ não đi. Một mạng nơ-ron là **một hàm số rất nhiều tham
số**, kiểu:

```
đầu_ra = f( đầu_vào , tham_số )
```

Huấn luyện là bài toán: tìm bộ `tham_số` sao cho trên dữ liệu `train`, `đầu_ra` gần
với đáp án nhất. "Gần" được đo bằng **hàm mất mát**. Máy tìm bằng cách lặp đi lặp lại
bốn bước:

1. Đưa một batch dữ liệu qua mạng → được dự đoán.
2. So dự đoán với đáp án → được **một** con số mất mát.
3. Hỏi ngược lại: mỗi tham số nên tăng hay giảm để con số đó nhỏ đi (**gradient**).
4. Nhích tất cả tham số theo hướng đó một đoạn bằng **học suất**. Quay lại bước 1.

Chỉ có thế. Toàn bộ những chữ khó ở các bài sau — `LayerNorm`, `GELU`, `Conv1d`, AdamW —
đều là chi tiết của bước 1 hoặc bước 4.

> **Điểm dễ hiểu nhầm:** máy **không** học "ý nghĩa" của tin tuyển dụng. Nó chỉ tìm
> bộ số làm mất mát nhỏ nhất. Nếu hàm mất mát đặt sai, hoặc dữ liệu có đường tắt (ví
> dụ văn bản chứa sẵn con số lương), máy sẽ vui vẻ đi đường tắt đó. Đấy là lý do
> [bài 2](02-hai-bai-toan-va-doan-duong-chung.md) nói kỹ về che lương.

---

## 3. Ba loại số đừng lẫn lộn

| Loại | Ai quyết định | Ví dụ trong dự án này |
|---|---|---|
| **Tham số** | Máy, trong lúc huấn luyện | Các trọng số trong `Linear(768, 256)` |
| **Siêu tham số** | Người, trước khi huấn luyện | `--lr 3e-4`, `--batch 256`, `--dropout 0.3`, `--hidden 256` |
| **Kết quả đo** | Không ai quyết định — đo xong mới biết | macro-F1, MAE. Nằm ở [`docs/04-results.md`](../docs/04-results.md), không nằm ở đây |

---

## 4. Bảng tra thuật ngữ Anh–Việt

Tài liệu chính trong `docs/` viết tiếng Anh. Bảng này để tra ngược.

| Tiếng Anh | Ở đây gọi là | Ghi chú |
|---|---|---|
| training / inference | huấn luyện / suy diễn | suy diễn = lúc dùng thật, không còn chỉnh tham số |
| frozen | đóng băng | trọng số bị khoá, không học |
| fine-tuning | tinh chỉnh | mở khoá, cho học tiếp trên dữ liệu của mình |
| embedding | vector nhúng | một từ hoặc một văn bản biến thành một dãy số |
| token | token | đơn vị nhỏ nhất mà mô hình đọc — có thể là từ, có thể là mảnh từ |
| subword | mảnh từ | `tuyển_dụng` có thể bị chẻ thành `tuyển@@` + `dụng` |
| pooling | gộp | nhiều vector → một vector |
| padding | đệm | thêm ô rỗng cho các câu dài bằng nhau |
| loss function | hàm mất mát | |
| gradient descent | hạ gradient | |
| learning rate | học suất | |
| gradient clipping | cắt gradient | chặn không cho một bước đi quá dài |
| weight decay | suy giảm trọng số | một kiểu chính quy hoá |
| dropout | dropout | tắt ngẫu nhiên một phần tín hiệu khi huấn luyện |
| overfitting | quá khớp | |
| early stopping | dừng sớm | |
| logits | logit | điểm số thô, chưa thành xác suất |
| softmax | softmax | biến điểm số thô thành xác suất cộng lại bằng 1 |
| cross-entropy | entropy chéo | hàm mất mát của bài phân loại |
| regression | hồi quy | đoán ra một số thực |
| classification | phân loại | chọn một nhãn |
| baseline | mốc cơ sở | mô hình đơn giản dùng để so |
| train / dev / test | tập huấn luyện / kiểm định / kiểm tra | trong repo này tên là `train`, `dev`, `test` |

---

## Đọc tiếp

- [Bài 2 · hai bài toán và đoạn đường chung](02-hai-bai-toan-va-doan-duong-chung.md)
- [`docs/nen-tang/09-vector-ngu-nghia.md`](../docs/nen-tang/09-vector-ngu-nghia.md) —
  vì sao dự án bỏ cách đếm từ để chuyển sang vector ngữ nghĩa
