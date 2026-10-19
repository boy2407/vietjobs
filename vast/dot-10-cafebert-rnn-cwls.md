# Đợt 10: CafeBERT 512 + RNN, category, cw effective β=0,999 + label smoothing 0,05 — CHƯA THUÊ

Task: chưa có ID trên `TASKS.md`. Hỏi: loss mới (đã tăng Dense CafeBERT 0,6037 → 0,6150, 5 seed) có giúp RNN không. Đổi từ Colab sang vast theo quyết định của tác giả (2026-10-19); notebook `notebooks/rnn_cafebert_cwls.ipynb` giữ làm phương án dự phòng.

Script: `scripts/vast_run_cafe_rnn_cwls.sh` (`all | encode | train`) | Instance: chưa thuê.

Phạm vi: nhúng cache token CafeBERT 512 `category` (`train`, `dev`), train 3 run seed 42–44. **Không đụng `test`.** Head và optimiser giống hệt `dl-cat-cafe-rnn-ce-s42..44` (Bi-GRU ‖ Bi-LSTM → CNN, h=100, c=50, lr 3e-4, wd 0,01, batch 256, patience 8 theo dev macro-F1, `best.pt` ở đỉnh); chỉ thêm `--class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05`.

## Chọn máy (ước tính, chưa đo cho đợt này)

- 1× GPU 16 GB (RTX 5060 Ti / 4070 Ti / 3080), như Đợt 2–5.
- Disk **≥ 100 GB** (cache token category ≈ 40 GB + model ≈ 2,2 GB + run); RAM **≥ 64 GB** để cache train ≈ 36 GB nằm trong page cache (`kinh-nghiem.md` §1–2); CPU ≥ 8 lõi (EPYC / Xeon); NVMe; CUDA ≥ 12.1; image có `/venv/main`.
- Giá ≤ $0,40/h; reliability ≥ 98 %.

## Ước tính thời gian

Mốc **đo được** trên 5060 Ti: nhúng CafeBERT 512 ≈ 30,8 tin/s (Đợt 3); 4 run RNN CE cùng cấu hình mất 27,3 / 40,7 / 40,2 / 48,5 phút (`artifacts/dl-cat-cafe-rnn-ce-*/metrics.json`, `seconds`).

| Phần | Ước tính |
|---|---|
| Nhúng cache token (38.166 tin) | ≈ 21 phút |
| 3 run (seed 42–44) | ≈ 1,5–2,5 giờ |
| Cài đặt, đẩy lên, kéo về | ≈ 45 phút |
| **Tổng thuê** | **≈ 3–4 giờ, ≈ $1,2–1,6** |

## Bảng run (điền từ `artifacts/<run_id>/metrics.json`)

| Run | Seed | macro-F1 | no-junk | accuracy | best / run | Mốc CE cùng seed | Trạng thái |
|---|---|---|---|---|---|---|---|
| `dl-cat-cafe-rnn-cweff999-ls05-s42` | 42 | | | | | 0,6172 | todo |
| `dl-cat-cafe-rnn-cweff999-ls05-s43` | 43 | | | | | 0,6223 | todo |
| `dl-cat-cafe-rnn-cweff999-ls05-s44` | 44 | | | | | 0,6060 | todo |

Nếu đường học cho thấy đỉnh sớm và F1 tụt nhanh hơn run CE, mở một lượt riêng `--lr 2e-4` hoặc `--weight-decay 0.02` (seed 42, run id riêng), không trộn vào 3 seed chính.

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Script + file đợt | done | không cần sửa code (`6d01849`) |
| 1 | Thuê máy theo "Chọn máy" | todo | tác giả |
| 2 | Đẩy code + `splits/train.csv`, `dev.csv` + `manifest.json` | todo | thiếu `manifest.json` thì encode dừng (`kinh-nghiem.md` §5) |
| 3 | Cài thư viện | todo | |
| 3b | `bash scripts/vast_run_cafe_rnn_cwls.sh model`: tải `uitnlp/CafeBERT` về cache HF, nạp thử tokenizer + model | todo | tác giả muốn tải model trước |
| 4 | `bash scripts/vast_run_cafe_rnn_cwls.sh all` trong tmux | todo | |
| 5 | Kéo 3 thư mục run (kèm `best.pt`) + log + `docs/04-results.md` về Mac; so `md5sum` | todo | `scp -r` từng thư mục, không qua `tar` (`kinh-nghiem.md` §5) |
| 5b | `torch.load` từng `best.pt` trên Mac | todo | checklist `README.md` mục 2 |
| 6 | Mac: chép 3 dòng vào `docs/04-results.md`, `export_runs.py`, docs/06, bao-cao ch.4 | todo | Rule 1, 7, 13 |
| 7 | Checklist `README.md`, rồi destroy | todo | tác giả bấm trên web |

## Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml \
  data/processed/splits/train.csv data/processed/splits/dev.csv data/processed/manifest.json \
  root@IP:/workspace/vietjobs/
```

## Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn scipy pyarrow pytest
mkdir -p docs && touch docs/04-results.md
tmux new -s rnn
bash scripts/vast_run_cafe_rnn_cwls.sh model   # tải CafeBERT về trước (~2,2 GB), kiểm nạp được
bash scripts/vast_run_cafe_rnn_cwls.sh all
```

## Kéo về

```bash
for s in 42 43 44; do r=dl-cat-cafe-rnn-cweff999-ls05-s$s
  mkdir -p artifacts/$r && scp -P PORT -r root@IP:/workspace/vietjobs/artifacts/$r/. artifacts/$r/
  ssh -p PORT root@IP "md5sum /workspace/vietjobs/artifacts/$r/best.pt"; md5 artifacts/$r/best.pt
  PYTHONPATH=src .venv-dl/bin/python -c "import torch; torch.load('artifacts/$r/best.pt', map_location='cpu'); print('$r best.pt ok')"
done
scp -P PORT "root@IP:/workspace/vietjobs/logs/dl-cat-cafe-rnn-cweff999-ls05-*.txt" logs/
scp -P PORT root@IP:/workspace/vietjobs/docs/04-results.md logs/dot10-server-04-results.md
```

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
