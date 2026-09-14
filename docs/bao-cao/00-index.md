# Báo cáo khoá luận tốt nghiệp — bảng điều khiển

Thư mục này là **bản thảo sống** của quyển báo cáo. Nó được cập nhật ngay sau mỗi
thay đổi thật trong dự án (Quy tắc 7 trong [../../AGENTS.md](../../AGENTS.md)), để
đến cuối kỳ việc "viết báo cáo" chỉ còn là **ghép file và định dạng Word**, không
phải viết lại từ đầu.

- **Đề tài:** Xây dựng hệ thống phân loại ngành nghề và ước lượng mức lương từ tin
  tuyển dụng tiếng Việt
- **English title:** Building a System for Job Category Classification and Salary
  Estimation from Vietnamese Job Postings
- **Sinh viên:** Nguyễn Trọng Nghĩa — MSSV 22550011
- **Cán bộ hướng dẫn:** TS. Lưu Thanh Sơn

---

## 1. Trạng thái từng chương

Cột **Nguồn** cho biết chương đó lấy số liệu từ đâu. Cột **Trạng thái** chỉ được
chuyển sang `xong` khi mọi con số trong chương đã có nguồn đo thật.

| # | File | Chương | Trạng thái | Nguồn số liệu |
|---|---|---|---|---|
| 0 | [01-phan-dau.md](01-phan-dau.md) | Bìa · Lời cảm ơn · Danh mục · Tóm tắt | **bản thảo** | tổng hợp từ các chương |
| 1 | [02-chuong-1-gioi-thieu.md](02-chuong-1-gioi-thieu.md) | Giới thiệu đề tài | **xong bản thảo** | đề cương · [01](../01-data-audit.md) · [05](../05-phan-tich-du-lieu.md) |
| 2 | [03-chuong-2-tong-quan.md](03-chuong-2-tong-quan.md) | Tổng quan nghiên cứu | **thiếu khảo sát công trình** | cần tra cứu tài liệu thật |
| 3 | [04-chuong-3-phuong-phap.md](04-chuong-3-phuong-phap.md) | Phương pháp thực hiện | **xong bản thảo** | [01](../01-data-audit.md) · [02](../02-vietnamese-nlp.md) · [03](../03-protocol.md) · [06](../06-baseline-dl.md) |
| 4 | [05-chuong-4-thuc-nghiem.md](05-chuong-4-thuc-nghiem.md) | Thực nghiệm và đánh giá | **số liệu đã cũ** | [04-results.md](../04-results.md) — xem §3 |
| 5 | [06-chuong-5-he-thong.md](06-chuong-5-he-thong.md) | Cài đặt hệ thống minh hoạ | **chưa làm** | hệ thống chưa được xây |
| 6 | [07-chuong-6-ket-luan.md](07-chuong-6-ket-luan.md) | Kết luận và hướng phát triển | **bản thảo một phần** | [09](../09-lo-trinh.md) |
| — | [08-tai-lieu-tham-khao.md](08-tai-lieu-tham-khao.md) | Tài liệu tham khảo (IEEE) | **đang bổ sung** | — |
| — | [09-phu-luc.md](09-phu-luc.md) | Phụ lục | **bản thảo** | — |

---

## 2. Ba quy ước đọc bản thảo

Bản thảo cố tình để lộ chỗ chưa xong thay vì lấp đầy bằng chữ. Ba loại dấu:

| Dấu | Nghĩa | Khi xuất Word |
|---|---|---|
| `> **Nguồn số liệu:** …` | Ghi rõ con số phía trên lấy từ file nào | **Xoá** — chỉ phục vụ kiểm tra nội bộ |
| `> ⛔ **CHƯA CÓ SỐ LIỆU**` | Chỗ này chờ một lần chạy thật, không được đoán | Phải hết sạch trước khi nộp |
| `> ✍️ **CẦN VIẾT TAY**` | Chỗ đòi hỏi tra cứu / nhận định của chính sinh viên | Phải hết sạch trước khi nộp |

Nguyên tắc chống đạo văn được áp ở mức tuyệt đối: **không trích dẫn một công trình
nào mà chưa đọc và chưa kiểm chứng được**. Mọi chỗ cần khảo sát tài liệu đều để dấu
`✍️ CẦN VIẾT TAY` thay vì điền một cái tên nghe có vẻ hợp lý. Chi tiết ở
[08-tai-lieu-tham-khao.md](08-tai-lieu-tham-khao.md).

---

## 3. Cảnh báo quan trọng: số liệu chương 4 đã cũ

Ngày 2026-09-09 lược đồ chia dữ liệu đổi từ 70/15/15 sang hai tầng
`train:test = 8:2` rồi `train:dev = 9:1`. Hệ quả:

| Thứ đã đo trước 2026-09-09 | Còn dùng được không |
|---|---|
| Toàn bộ 10 dòng trong [04-results.md](../04-results.md) | **Không** — đo trên tập dev cũ (7.159 dòng), tập dev mới chỉ có 3.812 dòng |
| Phân tích dữ liệu trong [05](../05-phan-tich-du-lieu.md) | **Có, toàn bộ — đã đo lại 12/09/2026 (T1.1)**. Mọi con số mô tả đo trên **tệp gốc 47.707 dòng** (`AGENTS.md` Rule 8), nên không phụ thuộc lược đồ chia. Cả bảy hình đã sinh lại, kể cả hình 07 (độ dài token); `summary.json` đã sinh lại đầy đủ; các **mốc cơ sở** ở §3.6.6 đã đo lại trên lược đồ v2 |
| Mốc học máy `test macro-F1 0,6112` trong [archive/](../archive/README.md) | **Không** — đo trên lược đồ chia cũ |
| Số liệu làm sạch và chia dữ liệu trong `manifest.json` | **Có** — manifest đã được sinh lại theo lược đồ mới |

Vì vậy chương 4 hiện chép lại các con số cũ **kèm nhãn "lược đồ v1"** và không được
dùng làm kết quả cuối cùng. Chúng ở đó để giữ mạch lập luận và để so sánh khi các
lần chạy mới hoàn tất. Danh sách việc phải chạy lại nằm ở
[09-lo-trinh.md](../09-lo-trinh.md).

---

## 4. Quy cách trình bày (rút từ mẫu báo cáo của trường)

Đo trực tiếp trên các quyển mẫu trong `~/Downloads/report`:

| Thuộc tính | Giá trị |
|---|---|
| Khổ giấy | A4 (21,0 × 29,7 cm) |
| Lề trái | 3,5 cm |
| Lề phải | 2,0 cm |
| Lề trên | 3,0 cm |
| Lề dưới | 3,5 cm |
| Font thân bài | Times New Roman, 13 pt |
| Giãn dòng | 1,5 |
| Thụt đầu dòng | 1,27 cm |
| Heading 1 | Times New Roman, 14–16 pt, đậm, in hoa |
| Heading 2 / 3 | Times New Roman, đậm |
| Trích dẫn | IEEE dạng số — `[1]`, `[2]`, đánh theo thứ tự xuất hiện |
| Đánh số hình | `Hình 3.1: Nội dung chú thích` — dấu hai chấm, đặt **dưới** hình |
| Đánh số bảng | `Bảng 3.1: Nội dung chú thích` — dấu hai chấm, đặt **trên** bảng |
| Xưng hô | **"tác giả"** (đề tài do một sinh viên thực hiện; mẫu hai người dùng "nhóm tác giả") |

Bố cục chương cũng lấy từ mẫu: sáu chương, trước đó là phần đầu (lời cảm ơn, mục
lục, ba danh mục, tóm tắt), sau đó là tài liệu tham khảo và phụ lục.

**Giọng văn** rút từ quyển mẫu gần đề tài nhất (dự đoán giá điện thoại cũ — cũng là
bài toán hồi quy trên dữ liệu rao vặt tiếng Việt). Mỗi kỹ thuật được trình bày theo
đúng ba nhịp, và bản thảo này bám theo:

1. **Nguyên lý hoạt động** — kỹ thuật đó là gì, viết cho người chưa biết.
2. **Cách áp dụng** — nó chạy ở đâu trong hệ thống, trên cột nào, với tham số nào.
3. **Lý do lệch khỏi mặc định** — vì sao không dùng giá trị thông thường, kèm số đo.

Nhịp thứ ba là chỗ quyển báo cáo thể hiện đóng góp riêng. Một bước không giải thích
được vì sao nó ở đó thì theo Quy tắc 2 của dự án phải bị gỡ bỏ, không phải giữ lại
rồi mô tả.

---

## 5. Quy trình ghép ra file Word

1. Chạy `./scripts/render_figures.sh` để xuất mọi sơ đồ mermaid trong `docs/` ra
   `docs/figures/*.png` — Word không đọc được mermaid.
2. Ghép các file theo đúng thứ tự: `01-phan-dau` → `02` → `03` → `04` → `05` →
   `06` → `07` → `08-tai-lieu-tham-khao` → `09-phu-luc`.
3. Xoá mọi khối `> **Nguồn số liệu:**`. Kiểm tra không còn `⛔` hay `✍️` nào.
4. Áp quy cách ở §4, chèn mục lục tự động, chèn danh mục hình và danh mục bảng.
5. Đối chiếu lần cuối: mỗi con số trong quyển phải tìm được trong `manifest.json`,
   `artifacts/eda/summary.json`, `artifacts/<run_id>/metrics.json` hoặc một dòng
   của [04-results.md](../04-results.md).

---

## 6. Danh mục hình dùng trong báo cáo

Báo cáo **dùng lại** các hình đã có trong `docs/figures/`, không tự định nghĩa sơ
đồ mermaid mới. Lý do: `scripts/render_figures.sh` khớp sơ đồ theo vị trí và sẽ
báo lỗi nếu số khối mermaid trong một file không khớp danh sách `SOURCES` của nó.

| Hình | File | Dùng ở chương | Đã render |
|---|---|---|---|
| Sườn tổng thể hệ thống | `figures/01-suon-tong-the.png` | 3 | ✅ |
| Bốn bản sao của mỗi cột văn bản | `figures/02-bon-ban-sao-van-ban.png` | 3 | ✅ |
| Chuỗi xử lý tiếng Việt | `figures/03-xu-ly-tieng-viet.png` | 3 | ✅ |
| Kiến trúc mạng học sâu | `figures/04-kien-truc-hoc-sau.png` | 3 | ✅ |
| Bài toán lương | `figures/05-bai-toan-luong.png` | 3 | ✅ |
| Sơ đồ mã nguồn | `figures/06-ma-nguon.png` | 5 | ✅ |
| Tán xạ lương theo ngành | `figures/eda/eda-04-luong-theo-nganh.png` | 3 | ✅ |
| Lệch lớp | `figures/eda/eda-02-lech-lop.png` | 3 | ✅ |
| Phân bố lương | `figures/eda/eda-05-phan-bo-luong.png` | 3 | ✅ |
| Tỷ lệ công bố lương | `figures/eda/eda-06-cong-bo-luong.png` | 3 | ✅ |
| Tán xạ cỡ lớp | `figures/eda/eda-03-tan-xa-co-lop.png` | 3 | ✅ |
| Độ đầy của 18 trường | `figures/eda/eda-01-do-day-truong.png` | 3 | ✅ |

Cả 12 hình đã render (`./scripts/render_figures.sh`). Chạy lại lệnh đó mỗi khi sửa
một sơ đồ mermaid trong `docs/`.

**Không còn hình thừa.** `scripts/analyze_data.py` sinh ra **bảy** hình, cả bảy đã có
mặt trong §3.6 của chương 3 — `eda-01-do-day-truong.png` đo tỷ lệ điền của 18 trường
và đã trở thành Hình 3.4 (mục 3.6.1); `eda-07-do-dai-token.png` (độ dài token, cái giá
của việc cắt ở 256) đã sinh ngày 12/09/2026 (T1.1) và trở thành Hình 3.10 (mục 3.6.8).
