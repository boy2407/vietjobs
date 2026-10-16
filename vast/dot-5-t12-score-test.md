# Đợt 5: T12.1b — nhúng cache cho test / ext37k-sal và chấm các run RNN — CHƯA THUÊ

Task: **T12.1b** (phụ thuộc T12.1a; T12.1c/d chấm Dense trên Mac sau khi cache về). Quyền: AGENTS.md Rule 5 item 6 (tác giả, 2026-10-15): chấm `best.pt` đã lưu trên `test`, không train lại.

Script: `scripts/vast_run_t12_score.sh` — **CHƯA VIẾT** (bước 0 dưới đây; viết và có test trước khi thuê) | Instance: chưa có (GPU, disk, RAM, IP/port điền khi thuê).

Phạm vi trên máy thuê:
1. Nhúng pooled cho `test` và `ext37k-sal` (danh sách cache bên dưới), bằng `scripts/encode_t12_cpu.py` thêm `--device cuda` (hiện mã cứng `cpu`). Chỉ **đọc** test để nhúng, không ghi `test.csv`.
2. Chấm **14 cấu hình RNN còn checkpoint / 20 run** trên `test` và `ext37k-sal`, nhúng token từng batch rồi chạy head (đường `predict_rnn` của `scripts/eval_external_mtl.py`, không ghi token cache). Mỗi (encoder, họ cột) nhúng một lượt, mọi run cùng encoder đọc chung lượt đó.
3. Loại: `dl-sal-cafe-rnn` (s42–s44 không có `best.pt`). `dl-cat-cafe-rnn-ce` chỉ s42 (s43, s44 không có `best.pt`). Hàng chờ tác giả (T12.1a) không chấm.

Số 14 cấu hình = 15 cấu hình RNN trên dashboard trừ `dl-sal-cafe-rnn`.

## Cache cần tạo (pooled, float32; ước tính từ kích thước mảng, chưa đo)

| Encoder | max_len | `test` raw | `test` masked | `ext37k-sal` masked |
|---|---|---|---|---|
| `phobert` | 256 | cần | cần | **đã có trên Mac** |
| `cafebert` | 256 | cần | — | — |
| `cafebert` | 512 | cần | cần | **đã có trên Mac** |
| `visobert` | 512 | cần | cần | cần |
| `xlmr` | 512 | cần | cần | cần |
| `bamibert` | 1024 | cần | cần | cần |

`ext37k-sal` raw (cho các cấu hình `category` Dense chấm trên ext37k-sal): chờ quyết định T12.1a, chưa nằm trong bảng. Mỗi tệp ≈ 9.541 × (768 hoặc 1024) × 4 B ≈ 29–39 MB; tổng dưới 1 GB.

Phần cache PhoBERT cho `ext37k-sal` cần tệp tách từ `artifacts/t12/ext37k-sal-masked-seg.parquet` (đã tạo trên Mac; `.venv-dl` không có `underthesea`).

## Chọn máy (ước tính, chưa đo)

Không có token cache (nhúng từng batch) nên không còn nút thắt I/O của Đợt 3–4 và không cần RAM lớn.

- 1× GPU 16 GB (RTX 5060 Ti / 4070 Ti / 3080), như Đợt 4.
- Disk **100 GB** (model ≈ 10–15 GB cho 5 encoder + splits + run RNN; không có token cache). Đợt 4 cần 300 GB chỉ vì token cache.
- RAM ≥ 32 GB; CPU ≥ 8 lõi; CUDA ≥ 12.1; image có `/venv/main` (như Đợt 3–4).
- Giá ≤ $0,35/h; reliability ≥ 98 %.

**Ước tính thời gian (chưa đo, ngoại suy từ `kinh-nghiem.md` §5 và Đợt 4):** mỗi encoder ≈ 25.000 lượt nhúng (2 × 9.541 test + 6.161 ext37k-sal). Mốc đo được: CafeBERT 512 ≈ 30,8 tin/s ⇒ ≈ 14 phút; ViSoBERT/XLM-R ≈ 90 tin/s (ước tính) ⇒ ≈ 5 phút mỗi họ; BamiBERT 1024 ≈ 35 tin/s (ước tính) ⇒ ≈ 12 phút mỗi lượt. Tổng nhúng + chấm RNN **≈ 1,5–2,5 giờ**, + ≈ 1 giờ đẩy code, cài đặt, kéo về ⇒ thuê 4 giờ, **≈ $1–2**.

## Bảng run (20 run; số điền từ `artifacts/<run_id>/external_37k/` và file chấm test sau khi chạy)

| Cấu hình dashboard | Run | Task | Encoder · len | Có `best.pt` | Trạng thái |
|---|---|---|---|---|---|
| `dl-cat-cafe-rnn-ce` | `dl-cat-cafe-rnn-ce-s42` | category | cafebert · 512 | có (s43, s44 thiếu) | todo |
| `dl-cat-rnn-ce` | `dl-cat-rnn-ce`, `-s42`, `-s43`, `-s44` | category | phobert · 256 | có | todo |
| `dl-cat-rnn-focal` | `dl-cat-rnn-focal` | category | phobert · 256 | có | todo |
| `dl-cat-bamibert-rnn-1024` | `dl-cat-bamibert-rnn-1024` | category | bamibert · 1024 | có | todo |
| `dl-cat-visobert-rnn-512` | `dl-cat-visobert-rnn-512` | category | visobert · 512 | có | todo |
| `dl-cat-xlmr-rnn-512` | `dl-cat-xlmr-rnn-512` | category | xlmr · 512 | có | todo |
| `dl-mtl-cafe-rnn-a5` | `dl-mtl-cafe-rnn-a5-s42` | mtl | cafebert · 512 | có + `best_sal.pt` | todo |
| `dl-mtl-bamibert-rnn1024-a5` | `dl-mtl-bamibert-rnn1024-a5-s42` | mtl | bamibert · 1024 | có + `best_sal.pt` | todo |
| `dl-mtl-visobert-rnn512-a5` | `dl-mtl-visobert-rnn512-a5-s42` | mtl | visobert · 512 | có + `best_sal.pt` | todo |
| `dl-mtl-xlmr-rnn512-a5` | `dl-mtl-xlmr-rnn512-a5-s42` | mtl | xlmr · 512 | có + `best_sal.pt` | todo |
| `dl-sal-rnn` | `dl-sal-rnn`, `-s42`, `-s43`, `-s44` | salary | phobert · 256 | có | todo |
| `dl-sal-bamibert-rnn-1024` | `dl-sal-bamibert-rnn-1024` | salary | bamibert · 1024 | có | todo |
| `dl-sal-visobert-rnn-512` | `dl-sal-visobert-rnn-512` | salary | visobert · 512 | có | todo |
| `dl-sal-xlmr-rnn-512` | `dl-sal-xlmr-rnn-512` | salary | xlmr · 512 | có | todo |

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Viết `scripts/vast_run_t12_score.sh` + `--device` cho `encode_t12_cpu.py` + script chấm RNN trên test, có test | todo | chặn bước 3 |
| 1 | Thuê máy theo "Chọn máy" | todo | tác giả |
| 2 | Đẩy code + splits + `manifest.json` + `ext37k-sal.csv` + 20 thư mục run lên | todo | lệnh dưới |
| 3 | Cài thư viện, `sha256sum data/processed/splits/test.csv` trên máy thuê | todo | phải bằng `70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047` |
| 4 | `bash scripts/vast_run_t12_score.sh all` trong tmux | todo | |
| 5 | Kéo cache + kết quả chấm + log về Mac; so `md5sum` | todo | |
| 6 | `sha256sum test.csv` trên máy thuê và trên Mac, cả hai bằng giá trị gốc | todo | |
| 7 | Checklist `README.md`, rồi destroy | todo | tác giả bấm trên web |

## Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
shasum -a 256 data/processed/splits/test.csv   # 70ba98d5…fba047
RUNS="dl-cat-cafe-rnn-ce-s42 dl-cat-rnn-ce dl-cat-rnn-ce-s42 dl-cat-rnn-ce-s43 dl-cat-rnn-ce-s44 dl-cat-rnn-focal \
 dl-cat-bamibert-rnn-1024 dl-cat-visobert-rnn-512 dl-cat-xlmr-rnn-512 \
 dl-mtl-cafe-rnn-a5-s42 dl-mtl-bamibert-rnn1024-a5-s42 dl-mtl-visobert-rnn512-a5-s42 dl-mtl-xlmr-rnn512-a5-s42 \
 dl-sal-rnn dl-sal-rnn-s42 dl-sal-rnn-s43 dl-sal-rnn-s44 dl-sal-bamibert-rnn-1024 dl-sal-visobert-rnn-512 dl-sal-xlmr-rnn-512"
FILES=""
for r in $RUNS; do for f in best.pt best_sal.pt config.json scaler.npz; do
  [ -f artifacts/$r/$f ] && FILES="$FILES artifacts/$r/$f"; done; done
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml \
  data/processed/splits data/processed/manifest.json \
  data/external/vietjobs37k/ext37k-sal.csv data/external/vietjobs37k/crosswalk_60_to_16.json \
  artifacts/t12/ext37k-sal-masked-seg.parquet $FILES \
  root@IP:/workspace/vietjobs/
```

Bắt buộc kèm `manifest.json` (`kinh-nghiem.md` §5) và `splits/train.csv`, `dev.csv` (`OurIndex` đọc cả ba split; nhãn lấy từ train). Không đẩy token cache.

## Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn scipy pyarrow underthesea pyvi pytest
sha256sum data/processed/splits/test.csv
tmux new -s t12
bash scripts/vast_run_t12_score.sh all
sha256sum data/processed/splits/test.csv
```

## Kéo về

`tar czf - … | tar xzf -` qua proxy từng treo (`kinh-nghiem.md` §5): dùng `scp` từng file, so `md5sum` hai đầu.

```bash
ssh -p PORT root@IP "cd /workspace/vietjobs && ls artifacts/embeddings/*test* artifacts/embeddings/*ext37k-sal* artifacts/*/external_37k artifacts/*/test_score logs/t12*"
scp -P PORT root@IP:/workspace/vietjobs/artifacts/embeddings/<tên>.npy artifacts/embeddings/
```

Đường dẫn kết quả chấm test (`artifacts/<run_id>/test_score/`) do script bước 0 quy định; điền lại đây khi script có.

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| — | — | — |
