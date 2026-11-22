# Cách đưa phần sửa từ repo sang Google Doc

Type: grilling
Status: closed
Blocked by: 08

## Question

Đưa phần sửa từ docs/bao-cao/ sang Google Doc bằng cách nào: sửa tại chỗ từng đoạn qua gws, hay lắp lại cả bản? Ai sửa trực tiếp trên Doc thì làm sao không mất?

## Answer

Tác giả chọn (a), 2026-11-22: sửa trong `docs/bao-cao/` trước, chỉ chép sang Google Doc **một lần** khi xong cả Chương 1–3. Lý do: Doc còn bị sửa nhiều, chép từng lần dễ lệch với repo. Trong thời gian này không ai sửa trực tiếp trên Doc.

**Cập nhật 2026-11-22 (tác giả: "viết trên đó giúp tôi"):** đã chép sớm lên Doc các phần đã chốt, sửa tại chỗ qua Docs API (`gws docs documents batchUpdate`, có khóa revision):
- §2.2 "Các công trình liên quan": bỏ hai mục con và bảy bài chưa đọc, sửa tên tác giả và kết quả lương của VietJobs, bỏ câu về Huynh; Bảng 2.1 còn hai dòng.
- Đánh lại số trích dẫn: [19]→[13], [20]→[14], [22]–[25]→[15]–[18] (khớp `08-tai-lieu-tham-khao.md`).
- Chương 3: Bảng 3.4 số tin có lương 24.622 / 2.698 / 6.773; sửa câu "tin đăng lại tuần sau với một câu sửa khác"; thêm ba đoạn về tin gần trùng, nhãn nhiễu và điểm trên phần sạch (dạng văn xuôi, không thêm bảng để khỏi đánh số lại).
Phần còn lại vẫn theo (a). Mục lục trên Doc cần bấm cập nhật.
