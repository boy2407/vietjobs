---
name: giai-thich
description: "Giải thích ngắn gọn một khái niệm trong pipeline VietJobs — làm sạch dữ liệu, xử lý tiếng Việt, TF-IDF, đặc trưng, rò rỉ, metric, chính quy hoá — bằng ví dụ đời thường trước rồi mới tới thuật ngữ. Use this skill when the user asks what a preprocessing step, feature-engineering choice, or ML concept in this project means or why it exists, invokes /giai-thich, or asks 'tại sao phải ...' about any pipeline stage."
trigger: "Use this skill when the user asks what a preprocessing step, feature-engineering choice, or ML concept in this project means or why it exists, invokes /giai-thich, or asks 'tại sao phải ...' about any pipeline stage."
version: 1
---

# Giải thích — tra cứu khái niệm pipeline VietJobs

Người hỏi mới vào ML/DL. Nhiệm vụ là làm họ **hiểu**, không phải làm họ thấy mình
thông thái. Ví dụ đời thường trước, thuật ngữ sau, luôn nói rõ bỏ đi thì hỏng gì.

## Khi nào dùng

- Người dùng gõ `/giai-thich <chủ đề>`
- Người dùng hỏi "tại sao phải ...", "... để làm gì", "... là gì" về một bước
  trong pipeline
- Người dùng gặp một thuật ngữ lạ khi đọc `docs/` và hỏi lại

## Quy trình bắt buộc

1. Tra chủ đề trong **Bản đồ chủ đề** bên dưới (khớp cả bí danh).
2. **`Read` file nguồn của chủ đề đó.** Bắt buộc, kể cả khi thấy mình đã biết.
3. Mới viết trả lời, theo khuôn năm khối.

Bước 2 không được bỏ. Skill này **cố ý không chứa con số** — số nằm ở `docs/`,
và `docs/` được cập nhật sau mỗi thí nghiệm. Trả lời từ trí nhớ là cách chắc chắn
để đưa ra con số cũ mà không ai phát hiện.

## Khuôn trả lời — năm khối

```markdown
## <Tên chủ đề>

**Hình dung:** <ví dụ đời thường, 2–3 câu. Không một thuật ngữ nào.>

**Tên gọi thật:** <đặt tên thuật ngữ, nối nó với ví dụ trên. 1–2 câu.>

**Trong VietJobs:** <cụ thể hoá, 2–4 dòng. Số lấy từ file nguồn vừa đọc.>

**Bỏ đi thì sao:** <hậu quả cụ thể, 1–3 dòng.>

**Đọc sâu:** <link tới file nguồn>
```

Hết năm khối là dừng. Muốn sâu hơn thì đã có link.

Nếu file nguồn ghi rõ một bước **đã bị gỡ** vì đo ra không giúp, nói thẳng điều đó
trong khối **Trong VietJobs**. Bước bị gỡ là bằng chứng, không phải chuyện xấu hổ.

## Bản đồ chủ đề

Cột "mồi" là gợi ý hướng ví dụ, không phải câu phải chép nguyên.

### Dữ liệu

| Gõ | Cũng nhận | Đọc file | Mồi ví dụ đời thường |
|---|---|---|---|
| `lam-sach` | `clean`, `vi-sao-lam-sach` | `docs/nen-tang/01-vi-sao-phai-lam-sach.md` | Danh bạ lưu "Nguyễn Văn A" và "nguyen van a" thành hai người |
| `trung-lap` | `dedup`, `gop-nhom`, `group-id` | `docs/01-data-audit.md` §2 | Cùng một tờ rơi dán 33 chỗ trong phố |
| `chia-tap` | `split`, `train-val-test`, `seed` | `docs/01-data-audit.md` §7 | Đề thi thử không được trùng câu với đề thi thật |
| `nhan` | `label`, `target`, `luong-mid` | `docs/01-data-audit.md` §5 | "8–12 triệu" và "thoả thuận" là hai loại thông tin khác nhau |

### Xử lý tiếng Việt

| Gõ | Cũng nhận | Đọc file | Mồi ví dụ đời thường |
|---|---|---|---|
| `tieng-viet` | `vitext`, `chin-buoc`, `tien-xu-ly` | `docs/02-vietnamese-nlp.md` | Máy đếm mặt chữ, nó không đọc nghĩa |
| `dau-thanh` | `nfc`, `unicode`, `hoa-hoà` | `docs/02-vietnamese-nlp.md` §3 bước 1–2 | "hoà" và "hòa" nhìn y hệt, máy thấy hai từ |
| `viet-tat` | `abbrev`, `nv`, `bhxh` | `docs/02-vietnamese-nlp.md` §3 bước 3 | "NV" và "nhân viên" — người hiểu, máy thì không |
| `che-luong` | `mask`, `mask-salary`, `salary-token` | `docs/02-vietnamese-nlp.md` §3 bước 4 | Chấm bài mà đề đã in sẵn đáp án ở góc trang |
| `tach-tu` | `segment`, `phan-doan`, `underthesea` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` | "nhân viên" là một từ hay hai từ? Khoảng trắng không nói |
| `gap-dau` | `fold-accents`, `khong-dau`, `char-ngram` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` §5 | Đọc biển hiệu viết không dấu — người vẫn hiểu, máy thì chịu |
| `chuan-tinh` | `province`, `dia-diem`, `ha-dong` | `docs/02-vietnamese-nlp.md` §3 bước 7 | "Hà Đông" và "Hà Nội" — bạn biết là một, máy không |
| `tu-dung` | `stopwords`, `bo-tu-dung` | `docs/02-vietnamese-nlp.md` §3 bước 8 | Bỏ chữ "không" khỏi "không yêu cầu kinh nghiệm" |

### Đặc trưng

| Gõ | Cũng nhận | Đọc file | Mồi ví dụ đời thường |
|---|---|---|---|
| `tf-idf` | `tfidf`, `tf`, `idf`, `tui-tu` | `docs/nen-tang/02-tf-idf-la-gi.md` | Từ hiếm trong sách mới đáng tra cứu, chữ "và" thì không |
| `ngram` | `bigram`, `unigram`, `n-gram` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` §1 | "bánh mì" khác hẳn "bánh" và "mì" đứng riêng |
| `dac-trung` | `feature`, `khoi`, `column-transformer` | `docs/05-dac-trung-tfidf.md` | Làm hồ sơ ứng viên — mỗi mục một ô, không nhét hết vào một dòng |
| `tham-so-tfidf` | `min-df`, `max-df`, `sublinear`, `max-features` | `docs/05-dac-trung-tfidf.md` §4 | Nút vặn trên máy lọc: lọc quá thô hay quá mịn đều hỏng |
| `ma-tran-thua` | `sparse`, `so-chieu`, `p-n`, `loi-nguyen` | `docs/nen-tang/04-ma-tran-thua-va-so-chieu.md` | Danh bạ hai trăm nghìn số mà bạn chỉ gọi năm số |

### Mô hình và đo lường

| Gõ | Cũng nhận | Đọc file | Mồi ví dụ đời thường |
|---|---|---|---|
| `hai-mo-hinh` | `bai-toan`, `phan-lop`, `luong` | `docs/06-mo-hinh-phan-lop.md` + `docs/07-bai-toan-luong.md` | Đoán nghề và đoán lương là hai câu hỏi, chung một hồ sơ |
| `metric` | `macro-f1`, `accuracy`, `baseline`, `nhieu` | `docs/nen-tang/06-do-luong-va-baseline.md` | Lớp 40 em, 39 giỏi 1 kém — "97% giỏi" giấu mất em kém |
| `chinh-quy-hoa` | `c`, `alpha`, `ridge`, `qua-khop`, `overfit` | `docs/nen-tang/07-chinh-quy-hoa.md` | Học vẹt đề cũ: 10 điểm ở nhà, 3 điểm phòng thi |

### Tính đúng đắn

| Gõ | Cũng nhận | Đọc file | Mồi ví dụ đời thường |
|---|---|---|---|
| `ro-ri` | `leak`, `leakage`, `resolve-column` | `docs/nen-tang/05-ro-ri-du-lieu.md` | Ôn trúng tủ vì đã xem trước đề — điểm cao mà không biết gì |
| `ablation` | `bo-buoc`, `buoc-nao-dang-giu` | `docs/02-vietnamese-nlp.md` §4 | Nấu ăn bớt dần từng gia vị để biết cái nào thật sự cần |

## Gọi không tham số

In bản đồ chủ đề theo năm nhóm, mỗi chủ đề một dòng ngắn, rồi hỏi muốn xem cái nào.
Không tự chọn giùm.

## Gõ sai hoặc không khớp

Gợi ý 2–3 chủ đề gần nhất rồi dừng. Không đoán bừa một chủ đề rồi trả lời.

Nếu là khái niệm ML chung mà dự án **chưa dùng** (ví dụ cross-validation, dropout,
attention): nói thẳng là dự án chưa dùng, trả lời ngắn theo đúng khuôn năm khối,
và **bỏ trống khối "Đọc sâu"** — đừng bịa một link `docs/` không tồn tại.

## Hỏi tiếp, hỏi sâu

Nếu người dùng hỏi tiếp trong cùng chủ đề, bỏ khuôn năm khối và trả lời thẳng câu
hỏi của họ. Khuôn là để mở đầu, không phải cái lồng.

## Tránh

- **Chép số từ trí nhớ.** Đọc file nguồn trước. Số trong `docs/` đổi sau mỗi thí
  nghiệm; số trong đầu thì không.
- **Dùng thuật ngữ trong khối "Hình dung".** Nó phá đúng mục đích của khối đó.
  Nếu không nghĩ ra cách nói mà không dùng thuật ngữ, tức là mình chưa hiểu đủ.
- **Ví dụ nghe hay nhưng sai bản chất.** Ví dụ phải **gãy ở đúng chỗ khái niệm
  gãy**. Một ví dụ đẹp mà dẫn sai hướng còn tệ hơn không có ví dụ.
- **Bỏ khối "Bỏ đi thì sao".** Đó là khối làm người đọc nhớ lâu nhất.
- **Trả lời dài.** Năm khối rồi dừng.
- **Giấu thất bại.** Ba bước tiền xử lý trong dự án này đo ra không giúp gì và đã
  bị gỡ. Nói ra, kèm số. Đó là phần đáng học nhất.
