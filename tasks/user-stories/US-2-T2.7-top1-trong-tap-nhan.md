# US-2 · T2.7 — Cột "top-1 nằm trong tập nhãn của nhóm"

**Là** tác giả luận văn, **tôi muốn** một trục báo cáo thứ hai bên cạnh macro-F1,
**để** điểm đơn nhãn không bị đọc thành toàn bộ chất lượng khi một tin có nhiều nhãn hợp lệ.

**Trạng thái:** done · **Phụ thuộc:** US-1 · **Chi phí:** <5 phút mỗi run · Hậu xử lý trên `y_pred` đã lưu, chỉ `dev`, không huấn luyện lại.
Mở lại phần T2.2 đã bỏ (`set_prediction.py` đã bị xóa): chỉ dựng lại chỉ số, không dựng lại đầu ra dạng tập nhãn.

## Checklist
- [x] Tính cho `dl-cat-s2` và `dl-cat-cafe-rnn-ce-s43`
- [x] Tính cho 3 run `dl-cat-ce-cpu` và 3 run `dl-cat-soft-cpu` (để đặt cạnh US-3)
- [x] Ghi vào `docs/06-baseline-dl.md` và chương báo cáo 4
- [x] Đổi trạng thái dòng T2.7 ở TASKS.md thành `done`

## Done khi
Cột có mặt cho hai run, giá trị ≥ accuracy, in cạnh macro-F1 và `f1_macro_no_junk`.

## Kết quả (đo 2026-10-17, `dev`, n = 3.812; tập nhãn lấy từ các dòng cùng `group_id` **trong `dev`**)
| Run | accuracy | top-1 trong tập nhãn | Δ | accuracy trên 160 dòng đa nhãn | top-1 trong tập trên 160 dòng đó |
|---|---|---|---|---|---|
| `dl-cat-s2` | 0,6511 | 0,6676 | +0,0165 | 0,4062 | 0,8000 |
| `dl-cat-cafe-rnn-ce-s43` | 0,6666 | 0,6839 | +0,0173 | 0,3938 | 0,8062 |

Vì `dev` chỉ có 160 dòng (4,20 %) nằm trong nhóm đa nhãn (US-1), cột mới chỉ hơn accuracy khoảng 0,017.
Phần chênh lớn nằm ở nhóm đa nhãn: độ đúng tăng gấp đôi. Giá trị này là cận dưới, vì các nhãn còn lại của nhóm
có thể nằm ở dòng thuộc `train` hoặc `test`, không nằm trong `dev`.

## Nhật ký
- 2026-10-17 — Tính xong, hai run đạt tiêu chí (≥ accuracy).
