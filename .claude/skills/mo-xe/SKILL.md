---
name: mo-xe
description: "Mổ xẻ một mô hình / thuật toán / bài toán ML-DL bằng một ví dụ tí hon CHẠY THẬT — dựng kho 5-6 mẫu, tính từng bước, in mọi ma trận, rồi đối chiếu với con số thật của VietJobs. Use this skill when the user asks how a model, algorithm, loss, metric, or ML/DL mechanism actually works — 'X hoạt động thế nào', 'sao suy ra được Y', 'giải thích lại từ từ', 'cho ví dụ' — or invokes /mo-xe."
trigger: "Use this skill when the user asks how an ML/DL model, algorithm, loss function, metric, or mechanism works step by step, asks 'sao ra được ...', 'từ A làm sao ra B', asks to slow down or re-explain, asks for a worked example, or invokes /mo-xe."
version: 1
---

# Mổ xẻ — hiểu một cơ chế ML/DL bằng ví dụ chạy thật

Người hỏi mới vào ML/DL và **học bằng con số cụ thể, không học bằng công thức**.
Nhiệm vụ: dựng một ví dụ nhỏ đến mức in ra hết được, chạy nó thật, rồi dẫn họ đi
từng bước cho tới lúc con số cuối cùng hiện ra.

Khác với [`giai-thich`](../giai-thich/SKILL.md): skill đó trả lời *"cái này là gì,
tại sao cần"* bằng ví dụ đời thường. Skill này trả lời *"nó chạy ra sao"* bằng ví
dụ **chạy được**. Hỏi khái niệm → `giai-thich`. Hỏi cơ chế → `mo-xe`.

---

## Luật số 1 — mọi con số phải chạy ra, không được bịa

**Trước khi viết một chữ nào của câu trả lời, chạy `python` qua Bash để lấy số.**

Đây là luật quan trọng nhất, không có ngoại lệ. Một bảng số tự nghĩ ra trong đầu
trông y hệt một bảng số đo được, và người đọc không có cách nào phân biệt. Nếu
không chạy được thì nói thẳng là chưa chạy.

Kèm theo:

- Có mô hình đã huấn luyện trong `artifacts/` thì **mở ra dùng số thật**
  (`joblib.load`), đừng huấn luyện lại một cái giả.
- Số về dự án lấy từ `docs/04-results.md`, `metrics.json`, `manifest.json`, hoặc
  đọc thẳng từ `data/processed/splits/*.parquet`. Không nhớ, không ước lượng.
- Chạy xong thì **đối chiếu**: số in ra bảng phải khớp số terminal vừa trả về.

## Luật số 2 — kho ví dụ phải tí hon và của VietJobs

- **5–6 mẫu**, vừa đủ để in trọn ma trận trong một khối mã.
- Nội dung là **tin tuyển dụng tiếng Việt** — tiêu đề, mô tả, kỹ năng, nghề.
  Không dùng iris, titanic, MNIST, `["the cat sat", "the dog ran"]`. Ví dụ vay
  mượn buộc người đọc bắc cầu hai lần.
- Ít lớp (2–3), ít cột, ít từ. Nếu ma trận không in vừa một màn hình thì kho còn
  quá to — cắt tiếp.
- **Một kho cho cả câu trả lời.** Đổi kho giữa chừng là làm người đọc lạc. Nếu
  một minh hoạ đòi kho khác thật (đã thử gộp mà hỏng) thì nói rõ vì sao phải tách.
- Thiết kế kho **ngược từ điều muốn cho thấy**. Muốn lộ bệnh nào thì dựng kho sao
  cho bệnh đó buộc phải xuất hiện, rồi chạy để xác nhận nó có xuất hiện thật.

## Luật số 3 — đi từng bước, mỗi bước một bảng

Không nhảy cóc. Khuôn thường dùng:

| Bước | In ra cái gì |
|---|---|
| 0 | Kho ví dụ — in cả **cột nhãn**, vì nhãn là thứ hay bị quên nhất |
| 1..n | Mỗi phép biến đổi một bảng, cột trước → cột sau |
| n+1 | **Vật thể cuối cùng** — ma trận / vector trọng số, in trọn |
| n+2 | Một mẫu mới đi qua toàn bộ đường ống, tới tận con số cuối |
| n+3 | Bảng đối chiếu **ví dụ ↔ VietJobs thật** |
| cuối | Đoạn `Tái lập` — mã chạy lại được |

Bảng đối chiếu cuối là bắt buộc. Không có nó, người đọc hiểu ví dụ nhưng không
biết nó liên quan gì tới dự án của mình.

## Luật số 4 — chỉ ra đúng chỗ dễ nhầm

Mỗi cơ chế có một hai chỗ mà người mới gần như chắc chắn hiểu sai. Đoán trước
chúng và đặt một khối cảnh báo ngay tại chỗ:

> ⚠️ L2 **không** làm các giá trị cộng lại bằng 1 — nó làm chúng **cộng bình
> phương** lại bằng 1.

Vài chỗ đã biết trong dự án này: `n` là số cột cố định chứ không phải số từ của
câu vào · `df` đếm **tài liệu** chứ không đếm **lần** · "Inverse" gắn vào
*Document Frequency* chứ không gắn vào *Term Frequency* · TF-IDF khớp **mặt chữ**
chứ không khớp **nghĩa** · nhãn không bao giờ là đặc trưng.

## Luật số 5 — gặp ca sai thì giữ nguyên

Nếu chạy ra kết quả sai — mô hình đoán nhầm, chỉ số tệ, minh hoạ không như mong
đợi — **giữ lại và mổ nó**. Một ca sai mổ kỹ dạy nhiều hơn một ca đúng.

Tuyệt đối không chỉnh ví dụ cho tới khi nó ra đẹp rồi mới trình bày. Đó là bịa số
bằng cách khác.

Và nói rõ giới hạn: một ví dụ không phải một phép đo. Con số đo được của dự án
nằm ở `04-results.md`.

---

## Bản đồ chủ đề

"Ví dụ phải cho thấy" là **mục tiêu** của kho — thiết kế kho ngược từ cột này.

### Biểu diễn dữ liệu

| Chủ đề | Đọc trước | Ví dụ phải cho thấy |
|---|---|---|
| TF-IDF, túi từ | `docs/nen-tang/02-tf-idf-la-gi.md` | Ma trận đầy đủ, `df` khác `tf`, L2 theo hàng |
| n-gram, tách từ | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` | Bigram bắt được cụm mà unigram làm vỡ |
| Nhiều khối đặc trưng | `docs/05-dac-trung-tfidf.md` | Mỗi cột một từ điển riêng, dán ngang, `‖hàng‖ = √(số khối có chữ)` |
| Ma trận thưa, số chiều | `docs/nen-tang/04-ma-tran-thua-va-so-chieu.md` | Đếm ô khác 0 trên tổng số ô |
| One-hot, co giãn số | `docs/05-dac-trung-tfidf.md` §6 | Cột thang đo lớn lấn cột 0/1 khi không co giãn |

### Mô hình

| Chủ đề | Đọc trước | Ví dụ phải cho thấy |
|---|---|---|
| Tuyến tính, SVM, LogReg | `docs/06-mo-hinh-phan-lop.md` | `W` in trọn, `điểm_k = w_k·x + b_k`, `argmax`, ô nào đẩy điểm |
| KNN | `docs/nen-tang/04-ma-tran-thua-va-so-chieu.md` | Hàng xóm gần nhất đổi khi bỏ chuẩn hoá |
| Cây / LightGBM | `docs/07-bai-toan-luong.md` | Một nhánh chia ở đâu và vì sao |
| Hồi quy | `docs/07-bai-toan-luong.md` | Dự đoán so với nhãn, sai số từng mẫu |
| Học sâu | *(dự án chưa chạy — nói rõ)* | Một lượt truyền xuôi bằng số tay |

### Đo lường và tính đúng đắn

| Chủ đề | Đọc trước | Ví dụ phải cho thấy |
|---|---|---|
| macro-F1, accuracy, baseline | `docs/nen-tang/06-do-luong-va-baseline.md` | Lớp lệch làm accuracy nói dối |
| Chính quy hoá, `C` | `docs/nen-tang/07-chinh-quy-hoa.md` | `C` đổi thì trọng số đổi thế nào |
| Rò rỉ dữ liệu | `docs/nen-tang/05-ro-ri-du-lieu.md` | Fit trên cả tập rồi mới chia → điểm đẹp giả |
| Chia tập, `SPLIT_SEED` | `docs/01-data-audit.md` §7 | Nhóm trùng lọt giữa hai tập |

Chủ đề dự án **chưa dùng** (dropout, attention, cross-validation…): vẫn mổ được
bằng ví dụ tự dựng, nhưng **nói thẳng là dự án chưa dùng** và bỏ bảng đối chiếu.

---

## Tránh

- **Số bịa.** Luật 1. Không có gì phá niềm tin nhanh hơn.
- **Công thức không kèm số.** Viết `idf = ln((1+N)/(1+df)) + 1` rồi đi tiếp là
  chưa mổ gì cả. Phải thay số vào và ra con số.
- **Ví dụ sách giáo khoa.** Kho phải là tin tuyển dụng tiếng Việt.
- **Đổi kho giữa chừng** mà không giải thích.
- **Bỏ bảng đối chiếu với VietJobs.** Ví dụ treo lơ lửng thì học xong không dùng
  được.
- **Trả lời dài mà không có bảng nào.** Nếu đoạn văn nhiều hơn bảng, viết sai thể.
- **Tự ý sửa `docs/`.** Mổ xẻ là để trả lời, không phải để sửa tài liệu. Thấy nội
  dung đáng đưa vào note thì **hỏi một dòng ở cuối**, làm khi được đồng ý — và khi
  làm thì theo Quy tắc số 1 trong `CLAUDE.md` (sửa cả mục lục, bảng, `SOURCES`).
