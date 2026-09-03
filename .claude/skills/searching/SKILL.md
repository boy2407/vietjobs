---
name: searching
description: "Tìm thuật toán / phương pháp phù hợp cho một bài toán VietJobs — đọc cái dự án đã đo, điền bảng ràng buộc thật, tra cứu ngoài có ghi nguồn, rồi trả về bảng ứng viên xếp hạng mà mỗi dòng quy về đúng một lệnh chạy được. Use this skill when the user asks which model, algorithm or technique to use or try next — 'nên dùng mô hình nào', 'cải thiện phân lớp bằng gì', 'có nên thử học sâu không' — or invokes /searching."
trigger: "Use this skill when the user asks which algorithm, model family, loss, or technique to try for a VietJobs task, asks whether an approach is worth trying, asks to research a method or read up on the literature, or invokes /searching."
version: 1
---

# Searching — chọn thuật toán bằng ràng buộc, không bằng danh tiếng

Câu hỏi "nên dùng mô hình nào" luôn có một câu trả lời nghe hay và sai: cái đang
hot. Skill này ép câu trả lời đi qua **ràng buộc thật của dự án** — bao nhiêu mẫu,
bao nhiêu chiều, máy nào, đã đo được gì — trước khi nhắc tên bất kỳ thuật toán nào.

Khác với các skill kia: hỏi *khái niệm là gì* → [`giai-thich`](../giai-thich/SKILL.md).
Hỏi *nó chạy ra sao* → [`mo-xe`](../mo-xe/SKILL.md). Hỏi *dữ liệu yếu ở đâu* →
[`dataset-diagnosis`](../dataset-diagnosis/SKILL.md). Hỏi *mô hình sai ở đâu* →
[`model-diagnosis`](../model-diagnosis/SKILL.md). Skill này chỉ trả lời **nên thử
cái gì tiếp theo**.

Skill này **đề xuất**, không chạy. Người chạy là `train.py`.

---

## Luật số 1 — đọc cái đã đo trước khi tra cái mới

Dự án này đã quét **6 mô hình × 6 cấu hình** một cách công bằng. Đề xuất lại một
thứ đã đo mà không nhắc con số cũ là lỗi nặng nhất của skill này.

Bắt buộc `Read` trước khi viết một chữ nào:

| File | Lấy gì ra |
|---|---|
| `docs/10-so-sanh-mo-hinh.md` | Cụm quét công bằng: mô hình nào thắng, thua bao nhiêu, `P(hàng > cột)`, giá của một điểm |
| `docs/04-results.md` (đuôi bảng) | Mọi cấu hình đã chạy thật, kèm ngày và commit |
| `docs/09-lo-trinh.md` | Việc đã được xếp hạng sẵn, và lý do xếp như vậy |
| `docs/03-protocol.md` | Luật quyết định: ngưỡng paired bootstrap, quy tắc gỡ bước |

Nếu câu hỏi thuộc bài toán lương thì đọc thêm `docs/07-bai-toan-luong.md`; nếu
thuộc đặc trưng thì `docs/05-dac-trung-tfidf.md`.

Trả lời từ trí nhớ ở đây tạo ra một đề xuất *nghe hợp lý* cho một thí nghiệm **đã
chạy rồi và đã thua**. Không ai phát hiện, kể cả người hỏi.

## Luật số 2 — điền bảng ràng buộc trước khi mở trình duyệt

Sáu ô này quyết định gần hết câu trả lời. Đọc số từ file, không nhớ:

| Ô | Đọc ở đâu |
|---|---|
| Số mẫu huấn luyện, số nhóm | `data/processed/manifest.json` |
| Số lớp và độ lệch lớp | `manifest.json` + `metrics.json` khoá `metrics.per_class` |
| Số chiều đặc trưng, tỷ lệ p/n | `docs/05-dac-trung-tfidf.md` |
| Ngân sách fit hiện tại | `fit_seconds` trong `artifacts/<run>/metrics.json` |
| Phần cứng | `environment` trong cùng file đó — **không có GPU** thì mọi đề xuất fine-tune phải nói rõ chi phí |
| Điểm đang phải vượt | Dòng tốt nhất trong `docs/04-results.md`, kèm `f1_macro_boot_std` |

Một ứng viên **không điền được ba ô** *chi phí fit ước tính · chạy được trên máy
này không · cần thêm nhãn hay dữ liệu gì* thì bị loại ngay, dù nổi tiếng đến đâu.

## Luật số 3 — tra ngoài có kỷ luật

`WebSearch` và `WebFetch` được dùng, nhưng theo thứ tự: **trong trước, ngoài sau**.
Chỉ tra ngoài khi câu hỏi vượt ra ngoài những gì `docs/` đã đo.

- Ưu tiên nguồn **có số đo**: paper kèm benchmark, tài liệu thư viện, kết quả trên
  bộ dữ liệu tiếng Việt. Blog không có số chỉ dùng để tìm tên phương pháp.
- Mọi khẳng định vay mượn phải **kèm nguồn và năm**. Không có nguồn thì ghi là
  *kinh nghiệm chung*, đừng khoác cho nó vẻ đã đo.
- **Số của bộ dữ liệu khác không phải dự đoán cho bộ này.** Viết rõ nó đo trên đâu,
  bao nhiêu lớp, bao nhiêu mẫu. Một mô hình đạt 0,9 trên tin tuyển dụng tiếng Anh
  nhãn sạch không nói được gì về 16 lớp lệch 27:1 ở đây.
- Thư viện mới phải kiểm được: có bản chạy trên CPU không, có phụ thuộc nặng không,
  còn được bảo trì không.

## Luật số 4 — mỗi ứng viên phải quy về một lệnh

Đề xuất chỉ có giá trị khi người đọc biết gõ gì tiếp theo. Mỗi dòng trong bảng
ứng viên kết thúc bằng **một** trong hai thứ:

- Một lệnh chạy được ngay, ví dụ
  `python -m vietjobs.train --task category --model svm --C 0.02 --province`,
  hoặc thêm `--set KEY=VALUE` cho siêu tham số của estimator.
- Hoặc nói rõ **phải viết thêm gì**: thêm tên vào `CLASSIFIERS`/`REGRESSORS` và
  một nhánh trong `build_estimator` (`src/vietjobs/models.py`); thêm công tắc vào
  `PrepConfig` hoặc sửa khối trong `build_features` (`src/vietjobs/features.py`).
  Kèm ước lượng số dòng code.

Chú ý một giới hạn thật: `--set` chỉ tới được estimator. Muốn đổi tham số TF-IDF
(`min_df`, `max_features`, `sublinear_tf`) thì phải sửa `_word_tfidf` — nói thẳng
điều đó thay vì đưa ra một lệnh không tồn tại.

## Luật số 5 — kỳ vọng không phải kết quả

Số chưa chạy phải ghi rõ **chưa đo**, để trong khối trích dẫn, và **không** được
đưa vào `docs/04-results.md`. Khuôn câu chữ đã có sẵn ở `docs/09-lo-trinh.md`
Ưu tiên 4 — chép đúng giọng đó.

Và luôn báo cáo **hai trục**: điểm số *và* chi phí. Một mô hình hơn 0,05 macro-F1
nhưng chậm gấp 200 lần là một đánh đổi, không phải một chiến thắng.

---

## Khuôn trả lời

```markdown
## <Câu hỏi> — <n> ứng viên

**Ràng buộc siết nhất:** <một câu, kèm số vừa đọc>

| Cách | Ý tưởng một dòng | Hợp bài này vì | Chi phí | Rủi ro | Lệnh đầu tiên |
|---|---|---|---|---|---|

**Làm ngay:** <một việc, kèm lệnh>
**Để dành:** <một việc, kèm điều kiện để nó đáng làm>
**Nên bỏ:** <một việc, kèm số bác bỏ nó>
```

Cột "Chi phí" chỉ nhận bốn giá trị: `<5 phút` · `~1 giờ máy` · `~1 buổi người` ·
`dựng lại splits`. Giá trị cuối là cảnh báo đỏ — nó vô hiệu hoá mọi dòng trong
`04-results.md`.

Khối **Nên bỏ** là bắt buộc. Loại một hướng đi bằng số đo có giá trị ngang việc
chọn được một hướng.

## Bản đồ câu hỏi

| Gõ | Cũng nhận | Đọc trước | Ràng buộc thường siết nhất |
|---|---|---|---|
| `chon-mo-hinh` | `model`, `thuat-toan`, `dung-gi` | `docs/10-so-sanh-mo-hinh.md` | Sáu họ đã đo; chênh lệch nằm trong nhiễu |
| `mat-can-bang` | `imbalance`, `lop-nho`, `class-weight` | `docs/06-mo-hinh-phan-lop.md` | `class_weight="balanced"` đã bật sẵn |
| `giam-chieu` | `min-df`, `max-features`, `svd`, `p-n` | `docs/05-dac-trung-tfidf.md` | p/n ≈ 7; `--set` không tới được TF-IDF |
| `trong-so-khoi` | `block-weight`, `title-x2` | `docs/09-lo-trinh.md` Ưu tiên 3a | Cần sửa `build_features`, chưa có cờ CLI |
| `hoc-sau` | `deep`, `phobert`, `bert`, `transformer` | `docs/09-lo-trinh.md` Ưu tiên 4 | Không GPU; phải đo kèm độ trễ suy luận |
| `hieu-chuan` | `calibration`, `xac-suat`, `top3` | `docs/09-lo-trinh.md` Ưu tiên 5.1 | `LinearSVC` không có `predict_proba` |
| `hoi-quy-luong` | `salary`, `ridge`, `lgbm`, `phan-vi` | `docs/07-bai-toan-luong.md` | Chỉ đọc cột `*_masked`; thiên lệch chọn mẫu |
| `nhieu-nhan` | `label-noise`, `tran`, `gan-lai` | `docs/09-lo-trinh.md` Ưu tiên 2 | Trần thật chưa đo — đo trước khi đổi mô hình |

Câu hỏi không khớp dòng nào: vẫn trả lời theo đúng năm luật, nhưng nói thẳng là
dự án chưa chạm tới hướng đó và **bỏ trống cột "Đọc trước"** thay vì bịa một link.

## Tránh

- **Danh sách mười thuật toán không xếp hạng.** Đó là né trách nhiệm chọn. Xếp
  hạng, rồi nói rõ mình xếp theo tiêu chí nào.
- **Đề xuất lại thứ đã đo mà không nhắc số cũ.** Luật 1. Kiểm `04-results.md` trước.
- **Chép paper mà không quy về lệnh.** Một đoạn tóm tắt phương pháp không giúp ai
  gõ được dòng nào.
- **Trộn số kỳ vọng với số đo được** trong cùng một bảng mà không đánh dấu.
- **Bỏ qua chi phí.** Điểm số không kèm giờ máy là nửa câu trả lời.
- **Tự chạy thí nghiệm.** Skill này dừng ở đề xuất. Muốn chạy thì người dùng gõ
  lệnh — và khi có kết quả thật thì `docs/` phải được cập nhật theo Quy tắc số 1
  trong `CLAUDE.md`.
- **Tự ý sửa `docs/`.** Thấy nội dung đáng đưa vào note thì hỏi một dòng ở cuối.
