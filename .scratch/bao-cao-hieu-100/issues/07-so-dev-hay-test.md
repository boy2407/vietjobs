# Ghi số dev hay test vào §3.2 và §6.2

Type: grilling
Status: closed
Blocked by: 06

## Question

Với số phần sạch của dev (đã có) và test (từ ticket 06), §3.2 và §6.2 ghi số nào, diễn đạt ra sao để trung thực mà không làm yếu kết luận?

## Answer

Tác giả chọn ghi số **test** (2026-11-22). §3.2 có bảng ba phần (toàn bộ, bỏ tin gần trùng với train, bỏ tin gần trùng với train hoặc dev), nói rõ test không bị sửa và số này không dùng để chọn mô hình. §6.2 thêm hạn chế 7 (MAE trên toàn bộ test thấp hơn phần sạch) và 8 (cùng nội dung khác nhãn). Số toàn bộ test vẫn là số chính.
