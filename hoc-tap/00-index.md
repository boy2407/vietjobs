# Ghi chú tự học — hai đường ống học sâu của VietJobs

Thư mục này **không phải tài liệu của dự án**. Nó là phần ghi chép để người viết
luận văn tự học deep learning từ con số 0, viết bằng tiếng Việt cho dễ vào.

Ba điều cần nói rõ ngay:

1. **Không nằm trong Rule 1.** Không ai phải cập nhật thư mục này sau mỗi lần chạy
   mô hình. Nó nói về **cơ chế**, không nói về kết quả.
2. **Không có số kết quả thí nghiệm.** Ở đây chỉ có các con số *cấu tạo* — 768 chiều,
   16 lớp, 256 token, cửa sổ 3/4/5 — những thứ không đổi khi huấn luyện lại. Mọi
   macro-F1, MAE, R² đều nằm ở [`docs/04-results.md`](../docs/04-results.md) và
   [`docs/06-baseline-dl.md`](../docs/06-baseline-dl.md), và **chỉ** ở đó.
3. **Không trích dẫn vào luận văn.** Chương phương pháp của luận văn là
   [`docs/bao-cao/04-chuong-3-phuong-phap.md`](../docs/bao-cao/04-chuong-3-phuong-phap.md).
   Thư mục này là giàn giáo, không phải công trình.

---

## Đọc theo thứ tự

| # | Bài | Đọc xong thì hiểu được gì |
|---|---|---|
| 1 | [Từ vựng tối thiểu](01-tu-vung-toi-thieu.md) | Mười khái niệm phải có trước khi đọc bất cứ thứ gì khác · bảng tra thuật ngữ Anh–Việt |
| 2 | [Hai bài toán và đoạn đường chung](02-hai-bai-toan-va-doan-duong-chung.md) | Phân loại khác hồi quy ở đúng ba chỗ · văn bản đi từ tin tuyển dụng vào mạng bằng đường nào · vì sao trộn cột là rò rỉ lương |
| 3 | [Kiến trúc A — PhoBERT đóng băng + dense](03-kien-truc-a-phobert-dense.md) | Kiến trúc **đang chạy thật** trong repo, từng mắt xích một, mỗi mắt xích để làm gì |
| 4 | [Kiến trúc B — TextCNN](04-kien-truc-b-textcnn.md) | Một mạng học sâu cơ bản không dùng PhoBERT. **Đề xuất, chưa từng chạy** |
| 5 | [Vòng huấn luyện và đo lường](05-vong-huan-luyen-va-do-luong.md) | "Học" thật ra là làm gì · từng núm vặn · đọc con số nào và vì sao không đọc accuracy |
| 6 | [Chạy T1.3 từng bước](06-chay-t13-tung-buoc.md) | Đúng lần chạy sắp tới trên máy này — hình dạng ma trận, dung lượng tệp, thời gian, dấu hiệu hỏng, **trước khi** gõ lệnh |

Muốn đi nhanh: đọc bài 1 → bài 2 → bài 3. Bài 4 và 5 đọc sau cũng được. Sắp chạy T1.3
thật thì đọc bài 6 ngay trước khi gõ lệnh.

---

## Quan hệ với tài liệu chính

| Cần gì | Đọc ở đâu |
|---|---|
| Vector 768 chiều ở đâu ra, PhoBERT được huấn luyện thế nào | [`docs/nen-tang/09-vector-ngu-nghia.md`](../docs/nen-tang/09-vector-ngu-nghia.md) (tiếng Anh) |
| Kiến trúc A kèm **số đo thật**, và câu chuyện lần chạy đầu bị phân kỳ | [`docs/06-baseline-dl.md`](../docs/06-baseline-dl.md) |
| Bẫy của bài toán lương | [`docs/07-bai-toan-luong.md`](../docs/07-bai-toan-luong.md) |
| Luật chơi: hạt giống, chạm tập `test`, cách ghi log | [`docs/03-protocol.md`](../docs/03-protocol.md) |
| Vì sao đừng đọc accuracy khi lớp lệch | [`docs/nen-tang/06-do-luong-va-baseline.md`](../docs/nen-tang/06-do-luong-va-baseline.md) |

Quy ước số ở thư mục này theo tiếng Việt: **dấu phẩy thập phân** (0,3), dấu chấm
phân nhóm nghìn (34.354). Tài liệu trong `docs/` dùng quy ước ngược lại — đó là
Rule 6, không phải nhầm lẫn.
