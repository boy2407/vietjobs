# US-1 · T2.6 — Vì sao `train` và `dev` khác nhau về cấu trúc nhóm

**Là** tác giả luận văn, **tôi muốn** biết vì sao `dev` gần như đơn nhãn trong khi `train` thì không,
**để** biết macro-F1 trên `dev` có so sánh công bằng được với dữ liệu mô hình đã học hay không.

**Trạng thái:** done · **Phụ thuộc:** — · **Chi phí:** ~1 phiên, chỉ đọc code · **Không** đổi dữ liệu, không dựng lại split, không chạm `test`.

## Checklist
- [x] Đọc `dataset.build` và `group_stratified_split` ([src/vietjobs/dataset.py](../../src/vietjobs/dataset.py) dòng 186–228)
- [x] Đo kích thước nhóm trên `train` và `dev`
- [x] Đo tỉ lệ nhóm đa nhãn theo từng lớp trên `dev`
- [x] Ghi vào `docs/01-data-audit.md` §7 và `docs/bao-cao/04-chuong-3-phuong-phap.md` (cách chia tách)
- [x] Đổi trạng thái dòng T2.6 ở TASKS.md thành `done`

## Done khi
Có một đoạn nêu file và dòng code gây ra điều đó, kèm số đo theo lớp.

## Kết quả (đo 2026-10-17 trên `train.csv` / `dev.csv`)
**Nguyên nhân:** `group_stratified_split` xếp **nhóm lớn nhất trước**, mỗi nhóm vào giỏ đang thiếu nhiều nhất
(`pick = max(fractions, key=lambda k: targets[k] - filled[k])`). Ở giai đoạn 2 giỏ `train` thiếu nhiều nhất,
nên nhóm lớn đổ hết vào `train`; `dev` chỉ nhận phần nhóm nhỏ còn lại.

| | `train` | `dev` |
|---|---|---|
| Số dòng / số nhóm | 34.354 / 22.026 | 3.812 / 3.728 |
| Số dòng trung bình mỗi nhóm | 1,56 | 1,023 |
| Nhóm lớn nhất | 33 dòng | 2 dòng |
| Nhóm ≥ 2 dòng | 9.097 | 84 |
| Nhóm ≥ 3 dòng | 2.264 | 0 |
| Dòng nằm trong nhóm ≥ 2 dòng | 62,37 % | 4,41 % |
| Dòng nằm trong nhóm đa nhãn | 56,04 % | 4,20 % |

Nhóm `dev` nằm trong `train`: 0,0 %. Tỉ lệ dòng đa nhãn trên `dev` theo lớp: `du_lịch…` 18,1 %,
`sản_xuất…` 7,0 %, `kinh_doanh…` 4,8 %, `marketing…` 1,5 %; 8 lớp đạt 0 %.
Ở `train` cùng chỉ số là 24,5 % (`tài_chính…`) đến 83,6 % (`thiết_kế…`).

**Chưa đo:** `test` có cùng kiểu lệch hay không. Điều này suy ra từ code (cùng giai đoạn 1), chưa đo vì Rule 5.

## Nhật ký
- 2026-10-17 — Đo xong, viết kết quả. T2.3 từng đếm 84 nhóm đa nhãn trên `dev`; đo lại ra 80 nhóm đa nhãn (84 là nhóm ≥ 2 dòng).
