# Bản đồ: hiểu 100 % báo cáo Chương 1–3

Label: wayfinder:map

## Destination

Bản báo cáo Chương 1–3 (Google Doc) mà tác giả tự giải thích được mọi đoạn trước hội đồng, mọi con số khớp code và artifacts, kèm danh sách việc còn lại để viết Chương 4–6 trước hạn **29/12/2026**.

## Notes

- Bản gốc là `docs/bao-cao/` trong repo (đọc code; CafeBERT là bản chính). Google Doc `18Dee72OQjqRkTHYu-vMBZmSZ1FKdpAOzrM3lQlo3pTw` là bản lắp, đọc/ghi bằng `gws`.
- Mẫu trình bày UIT: `https://docs.google.com/document/d/1-qUoi2Lur__zFKJdQjRIkwbpXsBBFsb2/edit` (dùng sau, Q20).
- Người đọc: hội đồng và giảng viên hướng dẫn đã biết ML. Thuật ngữ giữ, định nghĩa một lần ở lần dùng đầu.
- Cách học: dạy trước bằng ví dụ đời thường, mỗi lượt một ý, hỏi lại; chỗ tác giả bí thì viết lại đoạn đó trong báo cáo. **Sau mỗi lượt, thêm mục vào `hoc-tap/07-on-tap-bao-cao.md`** (nơi tác giả đọc lại); `## Answer` của ticket chỉ tóm tắt và trỏ tới mục đó.
- Không chạy thí nghiệm mới. Không biến đổi `test` (Rule 5). Mọi số có nguồn (Rule 7).
- Thuật ngữ: `GLOSSARY.md`. Skill nên gọi: `grilling`, `domain-modeling`, `giai-thich`, `mo-xe`.
- Đã sửa trước khi có bản đồ: Bảng 3.8 (số tin có lương theo manifest), §3.2 (định nghĩa nhóm và số đo tin gần trùng) — xem `tasks/T21-trung-gan-test-train.md`.

## Decisions so far

- [Đợt vast dot-24: chấm test phần sạch](issues/06-vast-dot24-cham-test-sach.md): test phần sạch MAE 4,109 (toàn bộ 3,923), macro-F1 0,6488 (toàn bộ 0,6278)

## Not yet specified

- Chương 4–6: viết theo bố cục T20 từ `04-results.md` và `metrics.json`, từng chương một.
- Trình bày theo mẫu UIT: đánh số mục (Doc đang ra "4.", "17." thay vì 2.1.1), chú thích hình/bảng, Phụ lục A (thêm cột tên lớp có dấu).
- Đưa bằng chứng nhãn nhiễu (tin gần trùng khác nhãn 70 %) vào §3.5.1 và `docs/01-data-audit.md`.
- Kết quả các bộ chia lọc trùng của T14 (`splits-dedup-norm-1label-mh`…) có được ghi ở đâu và có nên nhắc trong báo cáo.

## Out of scope

- Train lại hoặc thí nghiệm mới (Q5).
- Lọc tin khỏi `train` để train lại (Q17): là một thí nghiệm mới.
