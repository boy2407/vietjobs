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
