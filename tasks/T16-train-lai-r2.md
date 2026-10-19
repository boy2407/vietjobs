# T16 — Train lại 15 run T11 trên đúng bộ chia của Mac (`-r2`)

## Mục tiêu

15 run T11 train ở vast Đợt 4 (BamiBERT 1024, XLM-R 512, ViSoBERT 512 RNN; cat, sal, mtl) có
`predictions_dev.parquet` lệch `data/processed/splits/dev.csv` của Mac: nhãn nghề khớp 93,1 %,
số tin có lương 2.705 thay vì 2.698 (đo 2026-10-19). Dev lệch nghĩa là train cũng lệch, nên điểm
test (T12) của 15 run này không so được với 46 run còn lại và có thể bị thổi phồng. Train lại cả 15
run, cùng cấu hình, seed 42, run_id đuôi `-r2`; không ghi đè thư mục cũ.

## Checklist

- [x] Đo độ lệch dev của 61 run dashboard (15 lệch, 46 khớp 100 %)
- [x] `scripts/vast_run_r2.sh`: kiểm sha256 splits + manifest trước/sau, xoá token cache theo họ, chấm test + ext37k-sal
- [x] Đợt 8 (XLM-R + ViSoBERT RNN) và Đợt 9 (BamiBERT) khởi chạy, sha256 khớp
- [ ] Kéo 15 thư mục `-r2` (kèm `best.pt`) và log về Mac
- [ ] Kiểm `predictions_dev` của 15 run `-r2` khớp 100 % `dev.csv` Mac
- [ ] Cập nhật sheet Score, `04-results.md`, `export_runs.py`, docs/06, bao-cao ch.4 (Rule 1, 7, 13)
- [ ] Ghi chú 15 run Đợt 4 là "bộ chia khác Mac" ở mọi nơi trích số của chúng

## Tiêu chí xong

15 run `-r2` có `metrics.json`, `best.pt`, `t12_score/{test,ext37k-sal}.json` trên Mac; dev khớp 100 %;
sheet và docs dùng số `-r2`.

## Nhật ký

| Ngày | Việc |
|---|---|
| 2026-10-19 | Phát hiện lệch dev ở 15 run Đợt 4; viết `vast_run_r2.sh`; thuê 2 máy (Đợt 8, 9), khởi chạy |
