# Đợt 7: nhúng CafeBERT 512 trung bình 4 lớp cuối (pooled, train + dev) — KHÔNG THUÊ, chuyển sang Colab (2026-10-19)

Task: thử thay vector lớp cuối bằng trung bình 4 lớp ẩn cuối (`output_hidden_states=True`) cho `category`, head Dense. Chưa có ID trên `TASKS.md` (cần mở một task trước khi chạy, theo `tasks/` của CLAUDE.md toàn cục). Hướng này nằm ngoài các cache hiện có vì chúng chỉ lưu lớp cuối.

Script: `scripts/vast_run_cafe_last4.sh` (gọi `python -m vietjobs.dl.encode --encoder cafebert --max-len 512 --layers 4 --splits train dev --device cuda`) | Instance: chưa thuê.

Phạm vi: chỉ **nhúng** `train` (34.354 tin) và `dev` (3.812 tin) → `artifacts/embeddings/<tag>/cafebert-{train,dev}-raw-len512-last4.npy` (+ `.json`, trường `layers: 4`). **Không đụng `test`.** Train 5 seed × 2 cấu hình (CE thường; cw effective β=0.999 + ls 0.05) chạy trên Mac, `--device cpu`, `--layers 4`.

## Chọn máy (ước tính, chưa đo cho đợt này)

- 1× GPU 16 GB (RTX 5060 Ti / 4070 Ti / 3080), như Đợt 3–5; disk ≥ 50 GB (model ≈ 2,2 GB + cache pooled ≈ 160 MB); RAM ≥ 32 GB; CUDA ≥ 12.1; image có `/venv/main`.
- Không cần token cache nên không cần ổ 150 GB như Đợt 5.
- Giá ≤ $0,40/h.

## Ước tính thời gian

Mốc **đo được** (Đợt 3, 5060 Ti): CafeBERT 512 nhúng ≈ 30,8 tin/s. Cho 38.166 tin: ≈ 21 phút; `output_hidden_states=True` giữ 25 activation thay vì 1 nên có thể chậm hơn và tốn VRAM hơn (chưa đo; nếu hết VRAM, giảm `--batch`). Cộng cài đặt và kéo về ⇒ thuê **1–1,5 giờ**, ≈ $0,4–0,6. Trên Mac mps lần thử 2026-10-19 đo ≈ 5,9 tin/s (≈ 97 phút), đã dừng.

## Bảng run

| Cấu hình | Run (seed 42–46) | Encoder · len · layers | Trạng thái |
|---|---|---|---|
| CE thường | `dl-cat-cafe-last4-ce-s{42..46}` | cafebert · 512 · 4 | todo (train trên Mac sau khi kéo cache) |
| cw effective β=0.999 + ls 0.05 | `dl-cat-cafe-last4-cweff999-ls05-s{42..46}` | cafebert · 512 · 4 | todo |

So sánh với: CE lớp cuối 0.6037 ± 0.0037; cw + ls lớp cuối 0.6150 ± 0.0027 (CafeBERT Dense, 5 seed, dev).

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Code `--layers` (`encode.py`, `train_dl.py`), test, script | done | `pytest -q` 213 passed |
| 1 | Thuê máy theo "Chọn máy" | todo | tác giả |
| 2 | Đẩy code + `splits/train.csv`, `dev.csv` + `manifest.json` | todo | thiếu `manifest.json` thì encode dừng (`kinh-nghiem.md` §5) |
| 3 | Cài thư viện, `pytest tests/test_dl_encode.py` | todo | |
| 4 | `bash scripts/vast_run_cafe_last4.sh` trong tmux | todo | |
| 5 | Kéo 4 file (`.npy` + `.json` × 2) về Mac, so `md5sum`, kiểm shape `[n, 1024]` và `n` khớp split | todo | |
| 6 | Train 10 run trên Mac, ghi `04-results.md`, `export_runs.py` | todo | Rule 1, 7, 13 |
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
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn scipy pyarrow
PYTHONPATH=src python -m pytest -q tests/test_dl_encode.py
tmux new -s last4
bash scripts/vast_run_cafe_last4.sh
```

## Kéo về

```bash
ssh -p PORT root@IP "ls -l /workspace/vietjobs/artifacts/embeddings/*/cafebert-*-last4.*"
# scp từng file (cache nhỏ, ≈ 160 MB), rồi so md5sum hai đầu
```

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| 2026-10-19 | Encode trên Mac (mps) ≈ 5,9 tin/s, ước tính ≈ 97 phút | dừng tiến trình, chuyển sang thuê vast; chưa ghi file cache nào |
| 2026-10-19 | Tác giả chọn chạy trên Google Colab, hai tài khoản | không thuê vast; hai notebook `notebooks/last4_cafebert.ipynb` (CafeBERT 512) và `notebooks/last4_phobert.ipynb` (PhoBERT 256), mã `--layers` đưa lên Drive qua `vietjobs_src_last4.zip` vì nhánh GitHub chưa có |
