# Xin ý kiến thầy về phạm vi và trình bày khóa luận

**Mục đích:** em cần thầy chốt năm điểm về phạm vi và trình bày để dồn thời gian còn lại vào đúng phần trước hạn nộp 29/12/2026.

**Từ:** Nguyễn Trọng Nghĩa, **Gửi:** TS. Lưu Thanh Sơn, **Câu trả lời của thầy sẽ dùng để:** quyết định viết những gì ở Chương 4–6, có làm thêm phần hệ thống hay không, và có cần thuê GPU chấm thêm hay không.

## Bối cảnh

Đề tài phân loại 16 nhóm ngành và ước lượng lương từ tin tuyển dụng VietJobs (47.707 tin sau loại trùng). Mô hình chốt là mô hình đa nhiệm CafeBERT (đóng băng) với đầu BiGRU‖BiLSTM→CNN: macro-F1 0,6322 trên dev, 0,6278 trên test; MAE 3,70 triệu trên dev, 3,92 triệu trên test. Chương 1–3 đã có trên Google Doc; Chương 4–6 còn ở bản nháp. Khi rà lại, em phát hiện cách gom tin đăng lại chỉ bắt tin giống hệt nhau sau khi bỏ dấu, nên 11,6 % tin dev có một bản giống ≥ 90 % trong train. Đo trên dev, bỏ các tin này ra thì MAE tăng từ 3,70 lên 3,77 triệu, còn macro-F1 tăng từ 0,6322 lên 0,6506. Hiện repo chưa có điểm suy luận (inference) hay giao diện để dùng mô hình; mới có mã huấn luyện và đánh giá.

## Cách trả lời

Mỗi câu chỉ cần vài dòng; trả lời một phần hoặc ghi "chưa chắc" vẫn rất có ích. Em mong nhận phản hồi sớm để kịp hạn 29/12.

## Phạm vi khóa luận

### Tên đề tài có chữ "Xây dựng hệ thống". Khóa luận có bắt buộc phải có phần hệ thống chạy được (API, giao diện hoặc demo) không?

_Vì sao em hỏi: hiện chưa có phần này; nếu bắt buộc, đây là khối việc lớn nhất còn lại và phải bắt đầu ngay._

>

### Nếu cần phần hệ thống, mức tối thiểu thầy chấp nhận là gì (ví dụ: một lệnh dòng lệnh nhận một tin và trả ngành và lương)?

>

### Chương 5 có cần trình bày đủ ba giai đoạn thử nghiệm (Dense trên PhoBERT → so sánh năm bộ mã hóa → đầu RNN và đa nhiệm) không, hay chỉ cần mô hình chốt cùng các mốc so sánh?

_Vì sao em hỏi: Chương 5 hiện có 17 bảng kết quả (Bảng 5.1–5.17); rút gọn sẽ tiết kiệm nhiều thời gian viết._

>

## Số liệu về tin gần trùng

### Thầy chấp nhận em báo ảnh hưởng của tin gần trùng bằng số đo trên dev, hay cần đo cả trên test?

_Vì sao em hỏi: đo trên test cần chấm lại mô hình trên GPU thuê; dev đã đo xong._

>

### Thầy muốn phát hiện này được trình bày ở đâu: một đoạn trong mục chia dữ liệu (§3.2), mục hạn chế (§6.2), hay cả hai?

>

## Chuẩn bị bảo vệ

### Theo kinh nghiệm của thầy, hội đồng thường hỏi sâu nhất vào phần nào: dữ liệu, mô hình, cách đánh giá hay hệ thống?

>

### Có câu hỏi cụ thể nào về đề tài này mà thầy đoán hội đồng sẽ hỏi không?

>

## Trình bày theo mẫu UIT

### Khóa luận phải theo mẫu KLTN UIT chặt đến mức nào (font, cỡ chữ, lề, đánh số mục tới cấp 4)?

>

### Tài liệu tham khảo có bắt buộc theo kiểu IEEE không?

>

### Khóa luận nộp bản Word hay bản LaTeX đều được, hay chỉ chấp nhận một loại?

>

## Còn gì khác không?

Có điều gì em chưa hỏi mà thầy thấy em nên biết trước khi viết nốt và bảo vệ không?

>
