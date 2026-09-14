[← Kiến trúc B](04-kien-truc-b-textcnn.md) · [Mục lục](00-index.md) · [Chạy T1.3 từng bước →](06-chay-t13-tung-buoc.md)

# 5. Vòng huấn luyện và đo lường

Bài này dùng cho **cả hai** kiến trúc: phần "học" của chúng giống nhau gần như hoàn
toàn. Mã nằm ở [`src/vietjobs/dl/train_dl.py`](../src/vietjobs/dl/train_dl.py).

---

## 1. "Học" là sáu việc lặp lại

Với mỗi batch dữ liệu:

1. **Tiến** — đẩy batch qua mạng, nhận dự đoán.
2. **Tính mất mát** — so dự đoán với đáp án, ra **một** con số.
3. **`loss.backward()`** — tính gradient: mỗi tham số nên tăng hay giảm.
4. **Cắt gradient** — nếu độ dài gradient vượt ngưỡng (1,0) thì rút lại.
5. **`opt.step()`** — nhích toàn bộ tham số theo gradient, bước dài bằng học suất.
6. **`opt.zero_grad()`** rồi quay lại bước 1 với batch tiếp theo.

Hết một lượt qua toàn bộ `train` là xong **một epoch**. Mặc định chạy tối đa 40 epoch,
nhưng gần như luôn dừng sớm hơn (§3).

---

## 2. Các núm vặn của vòng lặp

| Núm | Mặc định | Để làm gì |
|---|---|---|
| `--batch` | 256 | Bao nhiêu dòng được xử lý trước mỗi lần chỉnh tham số. Gradient từ 256 dòng ổn định hơn từ 1 dòng, mà rẻ hơn nhiều so với từ cả 24 nghìn dòng |
| xáo trộn | bật | Mỗi epoch đảo thứ tự dòng (`torch.randperm` với hạt giống cố định). Không đảo thì mô hình học luôn cả thứ tự tệp — một thứ hoàn toàn vô nghĩa |
| `--lr` | 3e-4 | Độ dài bước. Quá lớn → mất mát **tăng** thay vì giảm, mô hình phân kỳ |
| `--clip` | 1,0 | Chặn một batch dị thường hất mô hình xuống vực |
| `--epochs` | 40 | Trần trên. Thực tế dừng sớm quyết định |
| `--seed` | 42 | Cố định mọi thứ ngẫu nhiên để chạy lại ra đúng kết quả cũ |

**`model.train()` và `model.eval()`** — hai dòng nhỏ mà người mới rất hay vấp.
`Dropout` **chỉ** được bật lúc huấn luyện; lúc đánh giá phải tắt, nếu không mỗi lần
chấm lại ra một điểm khác nhau. Hai lời gọi đó chính là công tắc.

---

## 3. Dừng sớm — và vì sao nó đã tiêu thụ tập `dev`

**Làm gì.** Sau **mỗi** epoch, chấm điểm mô hình trên `dev`. Nếu điểm tốt hơn mọi epoch
trước thì lưu lại trọng số (`best.pt`) và reset bộ đếm kiên nhẫn. Nếu qua `--patience 8`
epoch liên tiếp mà không khá hơn thì dừng, và **trả mô hình về trọng số tốt nhất**, chứ
không phải trọng số cuối cùng.

**Vì sao cần.** Mất mát trên `train` thì gần như luôn giảm tiếp — mô hình ngày càng
thuộc lòng dữ liệu huấn luyện. Điểm trên `dev` thì đến một lúc nào đó **quay đầu**: đó
là điểm bắt đầu quá khớp. Dừng sớm là cách tìm ra chỗ quay đầu đó mà không cần người
ngồi canh.

> **Điểm phải nhớ:** dừng sớm **là một lần dùng `dev` để ra quyết định**. Cộng thêm
> việc chọn học suất, chọn `--hidden`, chọn có cân bằng lớp hay không — tất cả đều đọc
> `dev`. Nên điểm trên `dev` **đã bị lạc quan hoá**, nó không còn là ước lượng trung thực
> cho dữ liệu mới. Đó là lý do tồn tại cờ `--confirm-test`: tập `test` đứng ngoài toàn
> bộ quá trình này, và chỉ được chấm một lần ở cuối.

**Mỗi epoch ghi một dòng** vào `history.jsonl`. Đường học nằm ở tệp đó, không nằm ở
`metrics.json` — `metrics.json` chỉ có con số cuối. Khi mô hình hỏng, chỗ đầu tiên phải
mở là `history.jsonl`.

---

## 4. Mỗi lần chạy để lại những gì

Một thư mục `artifacts/<run_id>/`:

| Tệp | Nội dung |
|---|---|
| `config.json` | Mọi siêu tham số + hạt giống — đủ để chạy lại y hệt |
| `env.json` | Thiết bị, phiên bản torch/transformers, mã git |
| `history.jsonl` | Một dòng mỗi epoch |
| `metrics.json` | Con số cuối, cùng tên khoá với đường học máy cũ để so được |
| `best.pt` | Trọng số ở epoch tốt nhất |
| `scaler.npz` | μ và σ của `train` — suy diễn **phải** dùng đúng hai con số này |
| `predictions_dev.parquet` | Dự đoán từng dòng, để phân tích lỗi và để chạy bootstrap |

Cộng thêm **đúng một dòng** nối vào [`docs/04-results.md`](../docs/04-results.md). Không
bao giờ ghi từng epoch vào đó.

Vì sao phải lưu dự đoán từng dòng: muốn biết mô hình sai ở **lớp nào**, hoặc muốn so hai
mô hình bằng bootstrap bắt cặp, thì một con số tổng không đủ.

---

## 5. Đọc con số nào

Định nghĩa của mọi độ đo nằm ở **một chỗ duy nhất**:
[`src/vietjobs/evaluate.py`](../src/vietjobs/evaluate.py). Cả đường học máy cũ và đường
học sâu đều gọi vào đó, nên các con số so được với nhau.

### Bài phân loại

| Độ đo | Nghĩa | Vì sao có nó |
|---|---|---|
| **macro-F1** | Tính F1 cho từng lớp rồi lấy trung bình **không đánh trọng số** | Mỗi lớp đếm một phiếu như nhau. Lớp chỉ có vài chục dòng trong `dev` vẫn quan trọng y như lớp lớn nhất |
| accuracy | Tỷ lệ đoán đúng trên tổng số dòng | **Đừng đọc một mình nó.** Lớp lệch 27:1 nên một mô hình chỉ đoán mãi lớp lớn nhất vẫn có accuracy trông không tệ, trong khi nó vô dụng |
| balanced accuracy | Trung bình tỷ lệ đúng của từng lớp | Nhìn từ góc "lớp hiếm có được nhận ra không" |
| `f1_macro_no_junk` | macro-F1 sau khi bỏ lớp rác `nhóm_nghề_khác` | Lớp đó là thùng chứa những tin không phân loại được, điểm của nó nhiễu rất mạnh |
| top-3 | Đáp án đúng có nằm trong 3 gợi ý đầu không | Nếu sản phẩm gợi ý ba ngành nghề cho người dùng chọn thì đây mới là con số đúng với trải nghiệm thật |

Vì sao accuracy lừa người đọc khi lớp lệch: xem
[`docs/nen-tang/06-do-luong-va-baseline.md`](../docs/nen-tang/06-do-luong-va-baseline.md).

### Bài lương

| Độ đo | Nghĩa | Vì sao có nó |
|---|---|---|
| **MAE** (`mae_trieu`) | Sai trung bình, **đơn vị triệu đồng** | Con số duy nhất người ngoài ngành hiểu ngay |
| MedAE (`median_ae_trieu`) | Sai **trung vị** | Không bị dúm tin lương vài trăm triệu kéo lệch. Đọc kèm MAE: hai số lệch nhau nhiều nghĩa là lỗi tập trung ở cái đuôi |
| `r2_log` | Giải thích được bao nhiêu phần biến động, trên thang log | Thang log là thang mà mô hình thật sự học |
| `within_20pct` | Tỷ lệ dự đoán lệch không quá 20 % | Gần với câu hỏi thực tế: "con số này có dùng được không" |

---

## 6. Luôn so với mốc cơ sở, không so với 0

Một điểm số đứng một mình **không nói gì cả**. Phải so với mô hình ngu nhất có thể:

- Phân loại: mô hình luôn đoán lớp lớn nhất. Mô hình chỉ dùng quy tắc từ khoá trên tiêu
  đề. Và mốc thật sự phải vượt: đường TF-IDF + LinearSVC cũ — 101 thí nghiệm, giờ đã
  khoá lại trong `docs/archive/` **không phải vì nó rác, mà vì nó là thước đo**.
- Lương: mô hình đoán **trung vị** cho mọi tin. Nghe buồn cười, nhưng nó là cái sàn
  thật sự — một mô hình học sâu không vượt được nó thì nó chưa học được gì từ văn bản,
  chỉ đang đoán trung vị bằng đường vòng.

Mọi con số mốc nằm ở [`docs/06-baseline-dl.md` §3](../docs/06-baseline-dl.md#3-the-bars-to-beat).

**Một phát hiện của dự án cần biết trước khi đọc bất cứ MAE nào:** nhãn ngành nghề gần
như **không nói gì về lương**. Một mô hình *được cho biết sẵn* ngành nghề thật cũng chỉ
nhúc nhích hơn mô hình đoán trung vị. Nghĩa là nếu đầu B cho MAE xấp xỉ mốc trung vị,
đừng kết luận "mô hình học kém" trước khi nhớ rằng **trần trên của bài toán này vốn
thấp**.

---

## 7. Hai kiến trúc cạnh nhau

Chỉ so cơ chế. Không có điểm số ở đây — điểm số nằm ở `04-results.md`.

| | A · PhoBERT đóng băng + dense | B · TextCNN |
|---|---|---|
| nghĩa của từ đến từ đâu | tiền huấn luyện trên hàng chục GB tiếng Việt | tự học từ riêng 34.354 tin của `train` |
| đơn vị token | mảnh từ (BPE) | từ đã tách, nguyên vẹn |
| gặp từ lạ | chẻ thành mảnh, vẫn có vector | rơi vào `<unk>`, mất trắng |
| một token có mấy vector | tuỳ ngữ cảnh | đúng một, cố định |
| trật tự từ | giữ trong encoder, mất khi gộp trung bình | giữ trong cửa sổ 3–5 token, mất khi max-pooling |
| cái gì được huấn luyện | **chỉ** khối dense | bảng nhúng + bộ lọc + khối cuối |
| đệm vector ra đĩa được không | được (`.npy`) — epoch tính bằng giây | không, vì bảng nhúng đổi mỗi epoch |
| cần tải mô hình ngoài | có, 135 triệu tham số | không |
| giải thích được không | rất khó — 768 chiều vô danh | một phần — truy được cụm từ nào kích hoạt bộ lọc nào |
| trạng thái trong dự án | **đã cài, đã đo** → [`docs/06-baseline-dl.md`](../docs/06-baseline-dl.md) | **đề xuất, chưa từng chạy** |

**Vì sao một luận văn có lợi khi có cả hai.** Kiến trúc B là **đối chứng**. Nếu B đạt gần
bằng A thì phần lớn điểm số đến từ chỗ bài toán này vốn đã dễ — tên nghề thường đã nằm
ngay trong tiêu đề. Nếu A bỏ xa B thì đó là bằng chứng đo được rằng tiền huấn luyện
tiếng Việt đóng góp thật, chứ không phải một câu nói cho sang. Cả hai kết cục đều là
kết quả dùng được; không chạy B thì không kết luận được cả hai.

---

## 8. Một câu để nhớ

> Một đường ống học sâu trả lời ba câu hỏi: **văn bản thành số bằng cách nào · câu trả
> lời phải có hình dạng gì · thế nào là sai**. Kiến trúc chỉ trả lời câu đầu tiên.

## Đọc tiếp

- [Bài 6 · chạy T1.3 từng bước](06-chay-t13-tung-buoc.md) — đúng lần chạy sắp tới, số đo thật của máy này
- [`docs/06-baseline-dl.md`](../docs/06-baseline-dl.md) — cùng kiến trúc A, kèm số đo thật
- [`docs/07-bai-toan-luong.md`](../docs/07-bai-toan-luong.md) — các bẫy của bài toán lương
- [`docs/03-protocol.md`](../docs/03-protocol.md) — luật chơi khi so hai mô hình
