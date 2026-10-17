# US-3 · T2.8 — Huấn luyện bằng soft target theo nhóm

**Là** tác giả luận văn, **tôi muốn** huấn luyện bằng phân phối nhãn mềm của từng nhóm văn bản trùng nhau,
**để** cùng một văn bản không còn đi vào loss với 2–3 đáp án mâu thuẫn, mà không xóa hay gán lại dòng nào.

**Trạng thái:** done · **Phụ thuộc:** US-1 · **Chi phí:** ~1 phiên · Mở lại hướng xử lý nhãn mà T2.2 / T2.3 / T2.5 đã dừng ngày 2026-09-27, theo "go" của tác giả ngày 2026-10-17.

## Checklist
- [x] Thêm `--soft-target` và `group_soft_targets` vào [src/vietjobs/dl/train_dl.py](../../src/vietjobs/dl/train_dl.py) (+ 1 test trong `tests/test_dl_heads.py`)
- [x] Chạy 3 seed 43–45, `device=cpu`, cùng cấu hình `dl-cat-ce-cpu` (3 dòng mới trong `docs/04-results.md`)
- [x] Chạy `scripts/export_runs.py` (Rule 13)
- [x] Ghi vào `docs/06-baseline-dl.md`, `docs/00-tong-quan.md`, chương báo cáo 4 (Rule 1, 7, 10)
- [x] Đổi trạng thái dòng T2.8 ở TASKS.md thành `done`

## Done khi
3 dòng mới trong `04-results.md` nêu mô hình và loss; kết luận ghi cả khi không tốt hơn;
chênh dưới 0,003 macro-F1 gọi là nhiễu; `f1_macro_no_junk` đi kèm.

## Kết quả (`dev`, n = 3.812, trung bình 3 seed; run `dl-cat-soft-cpu-s43..45` so với `dl-cat-ce-cpu-s43..45`)
| | CE | Soft target | Δ |
|---|---|---|---|
| macro-F1 | 0,6033 | 0,5998 | −0,0035 |
| `f1_macro_no_junk` | 0,6455 | 0,6416 | −0,0039 |
| accuracy | 0,6483 | 0,6470 | −0,0013 |
| top-1 trong tập nhãn (US-2) | 0,6654 | 0,6640 | −0,0014 |
| epoch tốt nhất (TB) | 13,0 | 7,7 | |

Từng seed: macro-F1 CE 0,6070 / 0,6042 / 0,5988; soft 0,6032 / 0,5996 / 0,5965.
Cả ba cặp đều thấp hơn (−0,0038 / −0,0046 / −0,0023), nhưng khoảng dao động giữa các seed của CE là 0,0082,
rộng hơn mức chênh. **Kết luận: soft target không cải thiện, và có xu hướng thấp hơn nhẹ.** Chưa đủ để nói
nó làm hại. Giả thuyết "nhãn mâu thuẫn là nguyên nhân kéo điểm xuống" chưa được ủng hộ.

Lưu ý khi đọc: `dev` gần như đơn nhãn (US-1), nên `dev` ít có khả năng thưởng cho một mô hình
học phân phối đa nhãn.

## Nhật ký
- 2026-10-17 — Thêm cờ + test (`pytest -q`: 198 passed, 2 skipped), chạy 3 seed, ghi kết quả.
