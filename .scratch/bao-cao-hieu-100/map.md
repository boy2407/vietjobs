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
- Đã sửa (ngoài ticket): Bảng 3.8 (số tin có lương theo manifest), §3.2 (định nghĩa nhóm, số đo tin gần trùng, nhãn nhiễu, điểm phần sạch), §6.2 hạn chế 7–8, `docs/01-data-audit.md` §2 — xem `tasks/T21-trung-gan-test-train.md`.

## Decisions so far

- [Đợt vast dot-24: chấm test phần sạch](issues/06-vast-dot24-cham-test-sach.md): test phần sạch MAE 4,109 (toàn bộ 3,923), macro-F1 0,6488 (toàn bộ 0,6278)
- [Ghi số dev hay test vào §3.2 và §6.2](issues/07-so-dev-hay-test.md): ghi số test, toàn bộ là số chính, phần sạch là số phụ
- [Đối chiếu số liệu Chương 1–3 với nguồn](issues/08-doi-chieu-so-lieu.md): không số đo nào lệch nguồn; 10 câu nói quá, số trích bài [13]–[21] không có nguồn, Chương 1–2 repo đã cũ
- [§2.2: tài liệu đã đọc thật](issues/09-tai-lieu-da-doc.md): giữ công cụ/phương pháp đã dùng và [19], [20]; bỏ các bài liên quan chưa đọc
- [Cách đưa phần sửa từ repo sang Google Doc](issues/10-dong-bo-repo-doc.md): sửa trong repo, chép sang Doc một lần khi xong Chương 1–3

## Not yet specified

- Chương 4–6: viết theo bố cục T20 từ `04-results.md` và `metrics.json`, từng chương một.
- Trình bày theo mẫu UIT: đánh số mục (Doc đang ra "4.", "17." thay vì 2.1.1), chú thích hình/bảng, Phụ lục A (thêm cột tên lớp có dấu; đang lặp lại Bảng 3.5), tiêu đề §2.1 không khớp nội dung.
- Kết quả các bộ chia lọc trùng của T14 (`splits-dedup-norm-1label-mh`…) có được ghi ở đâu và có nên nhắc trong báo cáo.

- `docs/06-baseline-dl.md` dòng 536 vẫn ghi ⛔ cho XLM-R/BamiBERT MTL RNN `-v2`, và `docs/04-results.md` thiếu dòng của 3 run XLM-R, ViSoBERT, BamiBERT: tài liệu chưa theo kịp run (Rule 1).

## Out of scope

- Train lại hoặc thí nghiệm mới (Q5).
- Lọc tin khỏi `train` để train lại (Q17): là một thí nghiệm mới.
