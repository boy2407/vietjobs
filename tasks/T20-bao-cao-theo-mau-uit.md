# T20 — Viết lại báo cáo theo mẫu KLTN UIT, chốt CafeBERT

**Mục tiêu.** `docs/bao-cao/` theo đúng bố cục quyển mẫu `~/Downloads/Bao cao KLTN.pdf`
(KLTN UIT 2023, LaTeX, 108 trang), lấy CafeBERT làm encoder chốt với cả bốn đầu
(Dense, RNN, MTL Dense, MTL RNN); sau đó xuất bản LaTeX theo định dạng mẫu.

**Quyết định của tác giả (2026-11-05):** encoder chốt là CafeBERT, báo cáo tất cả các
đầu; làm cả hai việc: cấu trúc lại chương và xuất LaTeX.

## Bố cục đích (theo mẫu)

| Chương mẫu | File mới | Lấy nội dung từ |
|---|---|---|
| Phần đầu (+ thông tin hội đồng) | `01-phan-dau.md` | giữ, thêm trang hội đồng |
| 1. Tổng quan | `02-chuong-1-tong-quan.md` | chương 1 cũ + thách thức, phương pháp nghiên cứu, đóng góp |
| 2. Cơ sở lý thuyết | `03-chuong-2-co-so-ly-thuyet.md` | chương 2 cũ + CafeBERT, BiGRU/BiLSTM/CNN |
| 3. Bộ dữ liệu | `04-chuong-3-bo-du-lieu.md` | §3.2, §3.4, §3.5, §3.6 cũ |
| 4. Cài đặt thử nghiệm | `05-chuong-4-cai-dat.md` | §3.1, §3.3, §3.7, §3.8, §4.1, §4.2 cũ + chương 5 cũ (mã nguồn) |
| 5. Kết quả thử nghiệm | `06-chuong-5-ket-qua.md` | §4.3–§4.8 cũ, sắp lại quanh CafeBERT |
| 6. Kết luận | `07-chuong-6-ket-luan.md` | giữ, viết lại theo CafeBERT |

## Mạch kể chương 5 (tác giả chốt 2026-11-05: kể quá trình Dense → RNN dẫn tới CafeBERT)

1. **5.1 Mốc cơ sở** — mốc ngây thơ, dò tuyến tính (`probe-*-0920`).
2. **5.2 Giai đoạn 1 — đầu Dense trên PhoBERT** — `dl-cat-ce-cpu-0920`, `dl-sal-cpu-0920`; focal, class weight; lần chạy phân kỳ `dl-cat-h256`; tái lập trên `cpu`.
3. **5.3 Giai đoạn 2 — đổi encoder, giữ Dense** — PhoBERT vs CafeBERT 5 seed; +lớp 512; nối PhoBERT‖CafeBERT; MTL Dense `-v2` trên 5 encoder → CafeBERT dẫn đầu (dev F1 0,6190; lương đơn nhiệm MAE 4,146).
4. **5.4 Giai đoạn 3 — đầu RNN** — `dl-cat-rnn-ce`, `dl-sal-rnn` (PhoBERT); MTL RNN trên các encoder; `dl-mtl-caf-rnn-b01-cw-s42-v2` tốt nhất (dev F1 0,6322, MAE 3,70).
5. **5.5 Chốt CafeBERT** — bảng bốn đầu CafeBERT (Dense, RNN, MTL Dense, MTL RNN) trên dev / test-informed / ext37k, kèm lý do chọn.
6. **5.6 Phân tích lỗi**, **5.7 Đánh giá ngoài** — chuyển sang mô hình CafeBERT chốt nếu đã có `predictions`, nếu chưa thì `⛔`.

## Checklist

- [ ] Tách/ghép file theo bảng trên, đánh số lại mục, bảng, hình; sửa mọi tham chiếu `§`
- [ ] Chương 1: thêm 1.x Thách thức, Phương pháp nghiên cứu, Đóng góp; số liệu chuyển sang CafeBERT
- [ ] Chương 2: thêm mục CafeBERT và đầu RNN (chỉ ý đã có nguồn; còn lại `✍️`)
- [ ] Chương 5: bảng tổng hợp bốn đầu CafeBERT (dev, test-informed, ext37k), PhoBERT làm đối chiếu
- [ ] Tóm tắt + chương 6 viết theo CafeBERT
- [ ] Cập nhật `00-index.md`, `docs/00-tong-quan.md`, bảng Rule 7 và Rule 10 trong `AGENTS.md`
- [ ] Dựng LaTeX theo mẫu (bìa, mục lục, danh mục, A4, Times 13pt) và biên dịch ra PDF

## Tiêu chí xong

Mọi chương theo bảng trên, không còn liên kết `§` gãy, mọi số có `Nguồn số liệu`;
PDF LaTeX biên dịch được.

## Nhật ký

- 2026-11-05 — tác giả gửi biểu mẫu Word `maubaocaoKLTN_2024.docx` (Google Drive id `1-qUoi2Lur__zFKJdQjRIkwbpXsBBFsb2`): bìa, bìa phụ, hội đồng, lời cảm ơn, mục lục, danh mục hình/bảng/từ viết tắt, tóm tắt, chương đánh số tới cấp 4, chú thích `Hình 1.1:` / `Bảng 1.1:`, tài liệu tham khảo IEEE. Đầu ra có thể đổi từ LaTeX sang Word theo mẫu này — chờ tác giả xác nhận.
- 2026-11-05 — tạo task; đọc mẫu và bản thảo; tác giả chốt CafeBERT (tất cả đầu) và "cả hai".
- 2026-11-05 — đổi đầu ra: viết thẳng vào Google Doc "Báo Cáo Khoá Luận Tốt Nghiệp" (`18Dee72OQjqRkTHYu-vMBZmSZ1FKdpAOzrM3lQlo3pTw`) qua `gws`. Đã ghi mục lục 68 đầu mục; chương 1 (12 mục); §1.4 sửa "khoảng 27 lần" → 25,5 lần (`summary.json` `imbalance_ratio`); chương 2 (15 mục, các trích dẫn chưa có trong danh mục để dấu ✍️). Mỗi cấu hình báo một seed (42) theo yêu cầu tác giả.
- 2026-11-05 — chương 3 vào Google Doc: 5 mục + 4 mục con, 7 bảng (Bảng 3.1–3.7), 4 chỗ chú thích hình (Hình 3.1–3.4) có dòng `[Chèn hình: …]` để tác giả tự chèn ảnh từ `docs/figures/eda/`. Số liệu từ `manifest.json` và `artifacts/eda/summary.json` (tệp gốc 47.707 dòng).
- 2026-11-05 — chương 4 vào Google Doc: 8 mục, 5 bảng (4.1–4.5), 2 chỗ hình (4.1, 4.2). Cấu hình lấy từ `artifacts/{dl-cat-cafe-dense-512,dl-sal-cafe-dense-last4-b01-cw-s42-v2,dl-cat-cafebert-rnn-last4-cweff999-ls05-s42,dl-mtl-caf-dense-last4-cw-s42-v2,dl-mtl-caf-rnn-b01-cw-s42-v2}/config.json`; 220 kiểm thử (`pytest --collect-only`).
- 2026-11-15 — tác giả yêu cầu: báo cáo không dùng icon; tập trung chương 1–3, bỏ các chương sau. Đã xoá chương 4–6 khỏi Google Doc (chỉ số 33099–64867), giữ Phụ lục A, B. Bản JSON trước khi xoá lưu ở scratchpad `doc-backup-truoc-xoa-ch4-6.json`; khôi phục bằng lịch sử phiên bản của Docs. Nội dung chương 5 đã soạn (`c5.py`, Bảng 5.1–5.8) chưa ghi vào Doc. Doc hiện không còn icon.
