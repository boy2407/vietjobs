# Ôn tập báo cáo — các câu hỏi đã hỏi và cách trả lời

Mỗi mục là một lượt hỏi trong buổi ôn báo cáo: **giải thích đời thường → câu hỏi → bạn đã
trả lời → đáp án**. Đọc lại trước khi bảo vệ. Thêm mục mới sau mỗi lượt.

Như mọi bài trong `hoc-tap/`, ở đây **không có số kết quả thí nghiệm**: chỗ nào cần số
thật thì chỉ sang mục báo cáo (§) chứa nó. Thuật ngữ dùng đúng nghĩa trong
[`GLOSSARY.md`](../GLOSSARY.md).

---

## Chương 1 — Phát biểu bài toán (§1.3)

### 1. Mô hình nhận gì, trả ra gì

**Giải thích.** Như một người làm tuyển dụng lâu năm: che dòng "ngành" và "lương" của một
tin, họ đọc phần còn lại rồi đoán (1) tin thuộc ngành nào trong 16 ngành — **phân loại**,
và (2) bao nhiêu triệu một tháng — **hồi quy**. Mô hình chỉ đọc ba phần chữ: tiêu đề, mô tả
công việc, yêu cầu ứng viên. Ở bài lương, mọi con số lương trong chữ được thay bằng ký hiệu
`<SALARY>` để mô hình không "chép đáp án" — gọi là **chống rò rỉ nhãn**.

**Câu hỏi.** Tin "Kế toán tổng hợp… Thu nhập 12–15 triệu". (a) Bài phân loại có đọc cụm
lương không? (b) Bài lương có đọc không, nếu không thì thấy gì?

**Bạn trả lời.** a có; b không, thấy ký hiệu che. — **Đúng.** Bài lương thấy
"Thu nhập `<SALARY>`".

### 2. Vì sao macro-F1, không phải accuracy

**Giải thích.** Lớp 100 học sinh: 90 Văn, 10 Toán. "Thầy bói lười" gặp ai cũng đoán Văn →
accuracy 90 % nhưng không nhận ra bạn Toán nào. **Macro-F1** chấm từng nhóm riêng rồi lấy
trung bình đều, nên thầy bói lộ ra (≈ 0,47). Dữ liệu đề tài lệch y như vậy; số của mô hình
"luôn đoán lớp lớn nhất" ở §3.5.1.

**Câu hỏi.** A: accuracy 0,70 nhưng không bao giờ đúng 3 ngành nhỏ nhất. B: accuracy 0,66,
đúng một phần ở cả 16 ngành. Theo macro-F1, mô hình nào tốt hơn?

**Bạn trả lời.** B, vì nhận ra cả 16 ngành. — **Đúng.** Accuracy vẫn được báo cáo, nhưng
macro-F1 là thước đo dùng để **chọn** mô hình.

### 3. MAE

**Giải thích.** Ba tin thật 10 / 20 / 8 triệu, đoán 12 / 15 / 8 → lệch 2 / 5 / 0 → MAE =
7 / 3 = 2,33 triệu. Càng nhỏ càng tốt, đơn vị triệu đồng. Chỉ chấm trên tin **có ghi
lương** — vì thế số tin ở bảng lương nhỏ hơn số tin của tập.

**Câu hỏi.** Tin thật 50 đoán 20, tin thật 10 đoán 9. (a) MAE? (b) Tin nào kéo MAE lên?

**Bạn trả lời.** a 15,5 triệu; b tin 50 triệu vì sai 30. — **Đúng.**

### 4. Vì sao học trên log1p(lương)

**Giải thích.** Lương có đuôi dài (vài tin lên hàng trăm triệu). Học trên lương thô thì mô
hình dồn sức chiều vài tin khủng. log1p "nén" số lớn: 5 → 1,8; 13,5 → 2,7; 50 → 3,9;
500 → 6,2. "1p" = cộng 1 để không bao giờ tính log(0). Kết quả đoán được đổi ngược
(`expm1`) về triệu đồng **trước** khi tính MAE.

**Câu hỏi.** (a) Mô hình học trên lương thô hay log1p? (b) MAE trong báo cáo tính bằng
triệu đồng thật hay đơn vị log?

**Bạn trả lời.** a log1p; b triệu đồng thật. — **Đúng.** Đây là câu hội đồng hay bẫy.

### 5. "Lương thật" là điểm giữa khoảng

**Giải thích.** Phần lớn tin ghi khoảng ("12–15 triệu"), nên nhãn là **điểm giữa**:
(12 + 15) / 2 = 13,5. Tin chỉ ghi một đầu thì dùng đầu đó (Bảng 3.3). Hệ quả: đoán 14 cho
tin "12–15" vẫn bị tính sai 0,5 dù nằm trong khoảng — MAE khắt khe hơn cảm giác.

**Câu hỏi tổng kết (chưa trả lời).** Nói lại §1.3 trong 3 câu: đầu vào, hai đầu ra, mỗi
đầu ra chấm bằng thước đo nào.

---

## Chương 2 — Hai bài báo phải đọc (§2.3)

Đọc kèm PDF: [19] arXiv:2603.05262, [20] arXiv:2112.11052. Số liệu của hai bài nằm ở §2.3. Trong danh mục tham khảo mới, hai bài này mang số [13] và [14].

### 9. Bài [19] — bài công bố bộ dữ liệu VietJobs

**Giải thích.** Nhóm tác giả ở VinUni thu tin tuyển dụng từ TopCV và công bố thành bộ
VietJobs: 16 nhóm ngành, có lương, kỹ năng, địa điểm. Họ thử hai bài giống đề tài —
phân loại ngành và ước lượng lương — nhưng chỉ bằng **mô hình ngôn ngữ lớn sinh văn
bản** (Qwen, Llama-SEA-LION…), theo ba cách: hỏi thẳng (zero-shot), hỏi kèm vài ví dụ
(few-shot), tinh chỉnh nhẹ (LoRA). Bài lương của họ **không đọc mô tả công việc**, chỉ
đọc chức danh, loại hợp đồng, địa điểm, quốc gia, kinh nghiệm; và đo bằng RMSE, R². Họ
chia dữ liệu ngẫu nhiên 80/10/10. Phần hạn chế của chính họ nói mô tả công việc có chỗ
**trùng lặp, theo mẫu** — khớp với phát hiện tin gần trùng ở §3.2.

**Câu bảo vệ.** "Số của đề tài không so trực tiếp với [19] được: khác cách chia, khác
đầu vào của bài lương, khác thước đo lương. Đề tài bổ sung cái [19] chưa có: mốc từ bộ
mã hóa và mô hình đọc mô tả để ước lượng lương."

**Câu hỏi.** (a) Bài lương của [19] đọc mô tả công việc không? (b) Vì sao không đặt
MAE của đề tài cạnh RMSE của [19] để nói "hơn"?

### 10. Bài [20] — nguồn của đầu RNN

**Giải thích.** Nhóm tác giả ở UIT dự đoán **chức danh** từ mô tả công việc tiếng Việt
(có lẫn tiếng Anh). Một tin có thể mang nhiều chức danh, nên đây là phân loại **đa
nhãn** (mỗi nhãn một xác suất, quá 0,5 thì chọn), khác bài **một nhãn** của đề tài.
Mô hình: vectơ từ một mô hình tiền huấn luyện → BiGRU và BiLSTM → tích chập → gộp
trung bình và gộp cực đại → nối lại → lớp ra. Họ thử nhiều bộ mã hóa; các mô hình đa
ngữ (mBERT, XLM-R) nhỉnh hơn PhoBERT vì dữ liệu lẫn tiếng Anh. Tập test được thu riêng,
nên điểm test thấp hơn dev khá xa. Thước đo là F1 trung bình theo mẫu, không phải
macro-F1.

**Lưu ý.** Tác giả thứ ba là Son T. Luu (UIT) — rất có thể chính là giảng viên hướng dẫn.
Hãy xác nhận; nếu đúng, hội đồng sẽ coi việc kế thừa kiến trúc này là tự nhiên, nhưng
bạn phải nói rõ đề tài khác ở đâu: một nhãn, thêm đầu lương, bộ mã hóa đóng băng.

**Câu hỏi.** (a) Bài [20] là một nhãn hay đa nhãn? (b) Đề tài lấy gì từ [20], và khác
[20] ở ba điểm nào?

### 11. Mười hai bài nền tảng và bộ mã hoá — đọc tóm tắt là đủ

Số trong ngoặc theo `docs/bao-cao/08-tai-lieu-tham-khao.md`. Mỗi bài: **bài nói gì** →
**đề tài dùng chỗ nào**. Mở phần tóm tắt (abstract) của từng bài, đối chiếu với hai dòng
dưới đây; nói lại được hai dòng đó là đủ để trả lời hội đồng.

| # | Bài | Bài nói gì | Đề tài dùng chỗ nào |
|---|---|---|---|
| [1] | Transformer (Vaswani và cộng sự) | Bỏ RNN, chỉ dùng cơ chế chú ý (attention) để mỗi từ "nhìn" mọi từ khác trong câu | Mọi bộ mã hoá của đề tài đều là Transformer |
| [2] | BERT (Devlin và cộng sự) | Học trước trên kho văn bản lớn bằng cách che từ rồi đoán lại; sau đó dùng lại cho bài khác | Ý tưởng "vectơ ngữ cảnh dùng lại được" mà đề tài dựa vào |
| [3] | RoBERTa (Liu và cộng sự) | Huấn luyện BERT lâu hơn, nhiều dữ liệu hơn, bỏ một nhiệm vụ phụ → tốt hơn | PhoBERT, XLM-R, CafeBERT, ViSoBERT, BamiBERT đều theo kiểu RoBERTa |
| [4] | PhoBERT (Nguyen & Nguyen) | RoBERTa học trên tiếng Việt đã **tách từ**; tối đa 256 token | Bộ mã hoá đầu tiên của đề tài; lý do phải tách từ trước khi đưa vào |
| [5] | underthesea | Bộ công cụ xử lý tiếng Việt (tách từ…) | Tách từ cho đường vào PhoBERT |
| [6] | VnCoreNLP (Vu và cộng sự) | Bộ công cụ khác cho tiếng Việt, tách từ được dùng khi huấn luyện PhoBERT | Chỉ nhắc như lựa chọn thay thế |
| [7] | Multitask learning (Caruana) | Học nhiều bài cùng lúc trên một phần chung giúp các bài hỗ trợ nhau | Mô hình đa nhiệm: một thân chung, hai đầu ngành và lương |
| [8] | Huber | Hàm mất mát phạt bình phương khi sai nhỏ, phạt tuyến tính khi sai lớn → ít bị điểm ngoại lệ kéo | SmoothL1 (β = 0,1) cho đầu lương |
| [15] | VLUE / CafeBERT (Do và cộng sự) | Bộ đánh giá hiểu tiếng Việt, kèm CafeBERT: XLM-R học thêm trên tiếng Việt | Bộ mã hoá của mô hình chốt; không cần tách từ, đọc được 512 token |
| [16] | ViSoBERT (Nguyen và cộng sự) | Bộ mã hoá học trên văn bản mạng xã hội tiếng Việt | Một trong năm bộ mã hoá được so sánh |
| [17] | XLM-R (Conneau và cộng sự) | Bộ mã hoá đa ngữ, học trên khoảng 100 ngôn ngữ | Một trong năm bộ mã hoá; CafeBERT dựng trên bản large của nó, còn đề tài dùng bản base |
| [18] | BamiBERT (Nguyen và cộng sự) | Bộ mã hoá tiếng Việt mới, đọc được tới 2048 token | Một trong năm bộ mã hoá; đề tài cho nó đọc 1024 token |

**Câu hỏi.** (a) Vì sao PhoBERT phải tách từ còn CafeBERT thì không? (b) Học đa nhiệm
[7] là gì, đề tài áp dụng ở đâu? (c) Vì sao dùng hàm Huber [8] cho lương mà không dùng
bình phương sai số?

---

### 12. §2.3 — Biểu diễn văn bản bằng mô hình ngôn ngữ tiền huấn luyện

**Giải thích.** Bộ mã hoá giống một **người phiên dịch đã đọc hàng triệu trang tiếng Việt**.
Người đó học bằng trò chơi điền vào chỗ trống (che một từ, đoán lại từ đó). Sau khi học xong,
đưa cho họ một tin tuyển dụng, họ viết cho **mỗi từ một "thẻ nghĩa"**, tức một dãy số
(vectơ). Thẻ nghĩa phụ thuộc vào câu: "bàn" trong "bàn ghế" và "bàn" trong "bàn bạc" ra hai
thẻ khác nhau. Đó là ý của Transformer và BERT. RoBERTa chỉ là cùng người phiên dịch, nhưng
được học kỹ hơn: đọc nhiều hơn, lâu hơn, và bỏ một bài tập ít ích.

**Đề tài dùng thế nào.** Đề tài **không dạy lại** người phiên dịch (giữ đóng băng). Mỗi tin
chỉ đưa cho họ đọc **một lần**, rồi cất các thẻ nghĩa vào đĩa. Phần được huấn luyện là cái
**đầu mạng nhỏ** phía sau, đọc các thẻ đó để đoán ngành và lương. Nhờ vậy thử được nhiều bộ
mã hoá và nhiều đầu mạng mà không tốn nhiều máy.

**"Trung bình bốn lớp ẩn cuối".** Người phiên dịch không viết thẻ nghĩa trong một lần: họ
sửa qua nhiều bản nháp liên tiếp (mỗi bản nháp là một **lớp**). Cách đơn giản là lấy bản nháp
cuối. Mô hình chốt lấy **trung bình bốn bản nháp cuối**, vì bản cuối cùng thường đã nghiêng
về trò chơi điền chỗ trống, còn mấy bản trước đó giữ nghĩa chung hơn.

**Câu hỏi.** Vì sao đề tài cất thẻ nghĩa vào đĩa rồi mới huấn luyện, thay vì cho người phiên
dịch đọc lại tin ở mỗi vòng học? Đổi lại, đề tài chịu mất điều gì?

**Bạn trả lời.** (chờ)

### 13. Năm bộ mã hoá giống và khác nhau ở đâu

**Giống nhau.** Cả năm đều là "người phiên dịch" cùng một kiểu: kiến trúc RoBERTa, học bằng
trò chơi đoán từ bị che, và trong đề tài đều được giữ nguyên (đóng băng).

**Khác nhau ở bốn chỗ.** (1) **Cỡ người**: CafeBERT có 24 lớp và vectơ 1.024 số, bốn bộ kia
12 lớp và 768 số. (2) **Đã đọc gì**: PhoBERT và BamiBERT chỉ đọc tiếng Việt, ViSoBERT đọc
mạng xã hội tiếng Việt, XLM-R đọc 100 thứ tiếng, CafeBERT là XLM-R bản lớn đọc thêm tiếng
Việt. (3) **Cắt chữ thế nào**: PhoBERT cần văn bản đã tách từ ("nhân_viên"), bốn bộ kia
đọc văn bản thường. (4) **Đọc được bao dài**: PhoBERT 256 token, CafeBERT, ViSoBERT và XLM-R
512, BamiBERT tới 2.048.

**Câu hỏi.** Nếu một tin dài 400 token, bộ mã hoá nào phải cắt bớt tin đó?

**Bạn trả lời.** (chờ)

### 14. SentencePiece là gì

**Giải thích.** Trước khi đọc, bộ mã hoá phải **cắt câu thành từng mảnh** (token), giống như
cắt bánh mì thành lát trước khi ăn. SentencePiece là một cách cắt: nó học từ một kho văn bản
lớn xem mảnh chữ nào hay gặp, rồi cắt câu thô theo các mảnh đó. Nó coi dấu cách cũng là một
chữ (viết là `▁`), nên **không cần ai tách từ trước**, và dùng được cho mọi ngôn ngữ. Từ lạ
bị cắt thành mảnh nhỏ hơn.

**Ví dụ chạy thật** với câu "Nhân viên kinh doanh logistics":

- CafeBERT (SentencePiece), đưa câu thường:
  `▁Nhân · ▁viên · ▁kinh · ▁doanh · ▁logistic · s`
- PhoBERT (BPE), phải đưa câu đã tách từ "Nhân_viên kinh_doanh logistics":
  `Nhân_viên · kinh_doanh · logistics`

PhoBERT cắt theo **từ** nhờ bước tách từ của underthesea làm trước. CafeBERT cắt theo **âm
tiết và mảnh chữ**, và tự học cách ghép nghĩa trong các lớp phía sau.

**Câu hỏi.** Vì sao đưa câu đã có dấu `_` ("nhân_viên") vào CafeBERT lại là sai?

**Bạn trả lời.** (chờ)

## Chương 3 — Dữ liệu

### 6. train, dev, test

**Giải thích.** Bộ đề ôn thi: **train** = câu được xem đáp án để học; **dev** = đề thi thử,
làm bao nhiêu lần cũng được, dùng để chọn cách học; **test** = đề thi thật, cất kín, mở
**đúng một lần** ở cuối. Lén xem đề thật thì điểm cao mà không biết mình giỏi thật không.
Số dòng mỗi tập ở Bảng 3.8.

**Câu hỏi.** Bạn học thuộc đúng 20 câu sẽ ra trong đề thật, được 10 điểm. Điểm đó có cho
thấy bạn giỏi không?

**Bạn trả lời.** Bạn hỏi ngược lại: test có trùng với train không? — **Đúng hướng.** Đó
chính là câu hỏi phải đặt ra; nó dẫn tới mục 7.

### 7. Nhóm tin đăng lại và tin gần trùng

**Giải thích.** Một công ty đăng cùng tin 5 lần, mỗi lần sửa vài chữ. Nếu 3 bản vào train,
2 bản vào test thì mô hình "đã thấy đề". Code gom các bản thành **nhóm tin đăng lại** và
ép cả nhóm vào một tập — nhưng chỉ gom khi ba trường giống hệt nhau sau khi bỏ dấu, dấu câu
và chữ hoa. Khác một từ là khác nhóm, nên vẫn có **tin gần trùng** lọt sang test.

**Đã đo được gì (số ở §3.2).** Ở các cặp gần trùng, nhãn ngành phần lớn **khác** nhau (người
đăng chọn ngành mâu thuẫn), nên học thuộc không giúp phân loại — điểm phân loại trên
**phần sạch** còn cao hơn. Nhưng lương phần lớn **giống hệt**, nên MAE trên toàn bộ test hơi
đẹp hơn MAE trên phần sạch.

**Câu hỏi.** Hai tin giống nhau 100 %, một gắn ngành A, một gắn ngành B. Mô hình có đạt
điểm tuyệt đối trên những cặp này được không?

**Bạn trả lời.** Bỏ tin đó ra. — **Gần đúng, nhưng thiếu một ý.** Mô hình chỉ chọn được
một ngành cho cùng một nội dung, nên **không thể** đúng cả hai — đây là phần lỗi không mô
hình nào sửa được. Bỏ tin khỏi **test** thì bị cấm (test không được biến đổi); bỏ khỏi
**train** là một thí nghiệm mới phải train lại.

### 8. Ba ý của dữ liệu lương

**Giải thích.** (a) Chia theo nhóm, không theo dòng — xem mục 7. (b) **eta²** nhỏ nghĩa là
biết ngành chỉ giảm được một phần rất nhỏ độ khó đoán của lương, vì trong cùng ngành có cả
nhân viên mới lẫn giám đốc → mô hình lương phải đọc nội dung (kinh nghiệm, cấp bậc). Số ở
§3.5.2. (c) Tin không ghi lương **không** gán lương = 0: "không ghi" khác "bằng 0"; gán 0 sẽ
kéo mọi dự đoán xuống. Các tin này chỉ dùng cho bài phân loại.

**Câu hỏi.** Nhắc lại (a), (b), (c), mỗi ý một câu.

**Bạn trả lời.** Chưa hiểu ở lần đầu → đã giảng lại từ gốc qua mục 6 và 7. Cần hỏi lại (b)
và (c) ở ticket "Hiểu §3.5".
