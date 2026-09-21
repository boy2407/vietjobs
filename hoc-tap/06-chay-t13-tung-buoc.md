[← Vòng huấn luyện](05-vong-huan-luyen-va-do-luong.md) · [Mục lục](00-index.md)

# 6. Chạy T1.3 — từng bước, trước khi gõ lệnh

Bài 3 giải thích **cơ chế** của kiến trúc A. Bài này khác: nó mô tả **đúng lần chạy sắp
tới** trên máy này, với số đo thật của máy này — hình dạng ma trận, dung lượng tệp,
thời gian, và dấu hiệu để biết mỗi bước đang đúng hay đang hỏng. Đọc xong bài này rồi
mới gõ lệnh.

**Nhắc lại một điều đã chốt:** T1.3 dùng kiến trúc A (PhoBERT đóng băng + dense).
**Không có tầng tích chập (CNN) ở đâu trong lần chạy này.** CNN là kiến trúc B —
[bài 4](04-kien-truc-b-textcnn.md) — một đường ống khác, chưa cài, chưa chạy.

---

## 1. T1.3 thật ra là hai việc, không phải một

| Bước | Lệnh | Làm gì | Chạy mấy lần |
|---|---|---|---|
| T1.2 | `vietjobs.dl.encode --task category --splits train dev` | PhoBERT đọc từng tin, ghi vector ra đĩa | **một lần** |
| T1.3 | `vietjobs.dl.train_dl --task category` | Huấn luyện khối dense trên vector đã ghi | mỗi cấu hình một lần |
| T1.3-cw | thêm cờ `--class-weight` | Chạy lại T1.3, phiên bản cân bằng lớp | một lần |

**Vì sao tách đôi.** PhoBERT bị đóng băng — trọng số không đổi trong suốt quá trình
huấn luyện, nên vector của một tin cũng không đổi giữa các epoch. Tính một lần rồi
dùng lại là đúng, không phải mẹo vặt. Không tách thì mỗi epoch phải mã hoá lại toàn bộ
`train`, và một lần huấn luyện 20 epoch sẽ mất hàng giờ thay vì vài phút.

Máy này (đo thật, phiên trước): **63,5 tin/giây** trên MPS sau khi làm nóng. Mã hoá cả
`train` (34.354 tin) mất khoảng **9 phút**, `dev` (3.812 tin) khoảng **1 phút**.

---

## 2. Bước 1 — một tin tuyển dụng thành một chuỗi

Ví dụ thật, lấy từ dòng số 19.398 của `train` (không phải số bịa):

```
category      : sản_xuất_lao_động_phổ_thông_cơ_khí
job_title_seg : "Chuyên_Viên Kinh_Doanh_Dầu_Mỡ"

chuỗi ghép (300 ký tự đầu):
"Chuyên_Viên Kinh_Doanh_Dầu_Mỡ . Triển_khai kế_hoạch kinh_doanh của công_ty
theo hướng phát_triển kênh phân_phối và đại_lý . Trực_tiếp_nhận đơn hàng ,
bán hàng và theo_dõi đơn hàng . Khảo_sát , nghiên_cứu và báo_cáo tình_hình
thị_trường . Địa_điểm làm_việc : Đức_Hòa - Long_An . . Tốt_nghiệp Trung_cấp"
```

Vì bài toán là `category`, `resolve_column` chọn họ cột **`raw`**:
`job_title_seg`, `description_seg`, `requirements_seg` — ba cột đã tách từ, chưa che
lương. Ba trường được ghép bằng `" . "`, tiêu đề đứng đầu.

---

## 3. Bước 2 — tokenizer: chuỗi thành số

Vẫn ví dụ trên, cho qua tokenizer của PhoBERT:

```
10 token đầu : <s>  Chuyên_@@  Viên  Kinh_@@  Doanh_@@  Dầu_@@  Mỡ  .  Triển_khai  kế_hoạch
10 id đầu    : 0    13826      4364  8360     28223     21130   36728  5  10766      421
```

Hai điều nhìn thấy ngay:
- `<s>` là token khởi đầu câu, luôn có id = 0.
- `Chuyên_Viên` bị **chẻ** thành `Chuyên_@@` + `Viên` — một mảnh từ, đúng như
  [bài 3 §2](03-kien-truc-a-phobert-dense.md#2-tokenizer--chuỗi-ký-tự-thành-dãy-số)
  đã nói. Tin này 466 ký tự nhưng chỉ ra 94 token — không phải một-đối-một.

**Độ dài token trên toàn `train`** (đo trên mẫu 1.000 tin, để biết trước khi cắt còn
bao nhiêu bị mất):

| Phân vị | Số token |
|---|---|
| trung vị (p50) | 173 |
| p75 | 235 |
| p90 | 315 |
| p99 | 521 |
| dài nhất trong mẫu | 657 |

**≈ 19 % số tin dài hơn 256 token** — bị cắt đuôi. Đây chính xác là cái giá của
`max_len=256`, và là lý do [bài 2](02-hai-bai-toan-va-doan-duong-chung.md#2-đoạn-đường-chung--từ-tin-tuyển-dụng-thành-một-chuỗi)
đặt tiêu đề lên đầu chuỗi: phần bị cắt luôn là đuôi mô tả, không bao giờ là tên nghề.

---

## 4. Bước 3 — PhoBERT: chuỗi thành vector, và hình dạng thay đổi thế nào

Đây là câu trả lời trực tiếp cho "sau PhoBERT ta nhận được gì":

```
34.354 chuỗi (mỗi chuỗi độ dài khác nhau)
      │  tokenizer + đệm
      ▼
[34.354, ≤256]            số nguyên — id mảnh từ, phần thừa là <pad>
      │  PhoBERT (đóng băng)
      ▼
[batch , ≤256, 768]       float32 — MỘT vector 768 chiều cho MỖI TOKEN
      │  gộp trung bình có mặt nạ
      ▼
[34.354,      768]        float32 — MỘT vector 768 chiều cho MỖI TIN
      │  ghi ra đĩa
      ▼
train-raw-len256.npy      ≈ 105,5 MB  (34.354 × 768 × 4 byte)
```

Ba điều cần nhớ khi nhìn bảng này:

1. **Kích thước không đổi dù tin dài hay ngắn.** Trước khi gộp, tin dài 94 token và
   tin dài 657 token (bị cắt còn 256) đều đi qua cùng một PhoBERT; sau khi gộp, cả hai
   đều ra đúng 768 số. Đây là việc mà bước gộp trung bình làm được.
2. **Dung lượng tệp.** `train`: 34.354 × 768 × 4 byte ≈ 105,5 MB. `dev`: 3.812 × 768 ×
   4 byte ≈ 11,7 MB. Không lớn — đây là lý do đệm ra đĩa là chuyện hợp lý, không phải
   gánh nặng.
3. **PhoBERT biến mất khỏi câu chuyện sau bước này.** Toàn bộ phần huấn luyện ở dưới
   không bao giờ gọi lại tokenizer hay PhoBERT nữa — nó chỉ đọc tệp `.npy`. Nếu sau
   này bạn thấy vòng huấn luyện chạy rất chậm, nghĩa là có gì đó **sai**: hoặc cache
   chưa tồn tại nên nó tự mã hoá lại (chậm), hoặc đang mã hoá nhầm việc.

---

## 5. Bước 4 — chuẩn hoá: không phải thủ tục cho vui

μ (trung bình) và σ (độ lệch chuẩn) của 768 chiều, tính **trên `train`**, lưu vào
`scaler.npz`. Mọi dữ liệu — kể cả `dev` lúc đánh giá — bị trừ μ rồi chia σ bằng đúng
hai con số đó.

Nhắc lại bằng đúng một câu vì nó quan trọng: **lần chạy đầu tiên của dự án không có
bước này, và mô hình đã phân kỳ** — mất mát tăng từ 2,63 lên tới 1.941,8 thay vì giảm.
Câu chuyện đầy đủ ở [`docs/06-baseline-dl.md` §5.4](../docs/06-baseline-dl.md).

---

## 6. Bước 5 — khối dense: đếm xem có bao nhiêu tham số đang thực sự học

```
LayerNorm(768)            768 × 2       =     1.536 tham số
Linear(768 → 256)         768×256 + 256 =   196.864 tham số
Linear(256 → 128)         256×128 + 128 =    32.896 tham số
Linear(128 → 16)          128×16  + 16  =     2.064 tham số
                                          ───────────
Tổng được huấn luyện                     =   233.360 tham số

PhoBERT (đóng băng, không học)           = 135.000.000 tham số
```

**Thứ đang được huấn luyện nhỏ hơn PhoBERT khoảng 580 lần.** Đây chính là lý do một
epoch trên vector đã đệm chỉ mất vài giây: máy không phải tính lại 135 triệu tham số
mỗi bước, chỉ 233 nghìn.

---

## 7. Bước 6 — vòng huấn luyện sẽ in ra cái gì

`train` có 34.354 dòng, batch mặc định 256 ⇒ **135 batch mỗi epoch**. Tối đa 40 epoch,
dừng sớm sau 8 epoch liền không cải thiện trên `dev`. Mỗi epoch in một dòng dạng:

```
epoch   1  loss 2.4123  val_macro_f1 0.XXXX  val_accuracy 0.XXXX
epoch   2  loss 2.0117  val_macro_f1 0.XXXX  val_accuracy 0.XXXX
...
```

(Không điền số thật ở đây — số thật chỉ xuất hiện khi lần chạy này thực sự diễn ra, và
khi đó nó thuộc về `docs/04-results.md`, không thuộc về bài học này.)

**Ba điều cần nhìn khi đọc log, không phải hoảng loạn khi thấy số lạ:**

| Thấy gì | Bình thường hay bất thường |
|---|---|
| `loss` giảm dần qua các epoch | Bình thường |
| `loss` **tăng vọt** (ví dụ nhảy từ hàng đơn vị lên hàng trăm) | Bất thường — phân kỳ. Xem lại có tắt nhầm chuẩn hoá (`--no-standardize`) hay học suất quá cao không |
| `val_macro_f1` tăng dần rồi chững lại rồi bắt đầu giảm | Bình thường — đúng lúc quá khớp bắt đầu, dừng sớm sẽ bắt đúng đỉnh đó |
| `val_macro_f1` rất thấp ở 2–3 epoch đầu (gần 1/16 ≈ 0,06) | Bình thường — mô hình mới khởi tạo, gần như đoán mò giữa 16 lớp |

---

## 8. Chạy xong thì có gì trong tay

Một thư mục `artifacts/<run_id>/` với 7 tệp — bảng đầy đủ đã có ở
[bài 5 §4](05-vong-huan-luyen-va-do-luong.md#4-mỗi-lần-chạy-để-lại-những-gì). Tệp
đầu tiên nên mở khi có gì bất thường: `history.jsonl` — đường học nằm ở đó, không nằm
ở `metrics.json`.

---

## 9. Đọc kết quả — so với cái gì, và mốc nào là mốc thật

Ba mốc để so, từ thấp đến cao:

1. Mô hình luôn đoán lớp lớn nhất — sàn tuyệt đối.
2. **Phép dò tuyến tính trên chính bộ vector PhoBERT** — mốc thật sự phải vượt: nó
   đọc đúng đầu vào ấy, không có tốc độ học để chỉnh sai, nên nếu mạng dense thua nó
   thì lỗi nằm ở đầu ra chứ không ở biểu diễn.

Con số cụ thể của cả ba mốc: [`docs/06-baseline-dl.md` §3`](../docs/06-baseline-dl.md#3-the-bars-to-beat) —
không chép sang đây vì nó là kết quả đo, không phải khái niệm.

Trước khi kết luận mạng "tệ", chạy que thử tuyến tính
([`scripts/probe_embeddings.py`](../scripts/probe_embeddings.py), xem
[bài 3 §13](03-kien-truc-a-phobert-dense.md#13-linear-probe--que-thử-trước-khi-đổ-lỗi-cho-mạng)).
Nó chạy trong vài giây trên cùng vector đã đệm. Nếu que thử cho điểm khá mà khối dense
cho điểm thấp hơn hẳn, lỗi nằm ở khối dense hoặc siêu tham số — không phải ở PhoBERT.

---

## 10. Những chỗ sẽ hỏng, và vì sao

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `cache {split} có N dòng nhưng split có M dòng` | Bộ đệm `.npy` cũ được tính từ một phiên bản splits khác | Chạy lại `encode` với `--force` |
| Điểm cao bất thường ngay từ đầu | Nạp nhầm tệp `.npy` sang bài toán khác (rò rỉ) | Kiểm tên tệp: `raw` cho category, `masked` cho salary |
| `loss` = `nan` | Học suất quá cao, hoặc chạy với `--no-standardize` | Dùng lại tham số mặc định |
| Vòng huấn luyện chạy rất chậm (nhiều giây/epoch trở lên) | Không tìm thấy cache, đang mã hoá lại; hoặc rơi về CPU | Kiểm `env.json` xem `device`; kiểm `artifacts/embeddings/` đã có tệp chưa |

---

## Đọc tiếp

- [Bài 3 · kiến trúc A, đầy đủ cơ chế](03-kien-truc-a-phobert-dense.md)
- [Bài 5 · vòng huấn luyện và đo lường](05-vong-huan-luyen-va-do-luong.md)
- [`docs/06-baseline-dl.md`](../docs/06-baseline-dl.md) — nơi số thật của lần chạy này sẽ được ghi lại
- [`docs/03-protocol.md`](../docs/03-protocol.md) — luật chơi khi chạy và khi so sánh
