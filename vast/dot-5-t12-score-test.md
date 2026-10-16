# Đợt 5: T12.1b/c/d — nhúng cache cho test / ext37k-sal và chấm các run RNN + Dense — CHƯA THUÊ

Task: **T12.1b** (nhúng + chấm RNN), **T12.1c** (chấm Dense trên `test`), **T12.1d** (chấm Dense trên `ext37k-sal`), tất cả trên máy thuê (tác giả đổi 2026-10-16); phụ thuộc T12.1a. CSV và hàng `04-results.md` vẫn ghi trên Mac sau khi kéo về. Quyền: AGENTS.md Rule 5 item 6 (tác giả, 2026-10-15): chấm `best.pt` đã lưu trên `test`, không train lại.

Script: `scripts/vast_run_t12_score.sh` (gọi `scripts/encode_t12_cpu.py --device cuda` và `scripts/score_t12.py`; đã viết, smoke test trên dev ở Mac) | Instance: chưa có (GPU, disk, RAM, IP/port điền khi thuê).

Phạm vi trên máy thuê:
1. Nhúng pooled cho `test` và `ext37k-sal` (danh sách cache bên dưới), bằng `scripts/encode_t12_cpu.py` thêm `--device cuda` (hiện mã cứng `cpu`). Chỉ **đọc** test để nhúng, không ghi `test.csv`.
2. Chấm **14 cấu hình RNN còn checkpoint / 20 run** trên `test` và `ext37k-sal`, nhúng token từng batch rồi chạy head (đường `predict_rnn` của `scripts/eval_external_mtl.py`, không ghi token cache). Mỗi (encoder, họ cột) nhúng một lượt, mọi run cùng encoder đọc chung lượt đó.
3. Chấm các run **Dense** (đơn nhiệm và MTL dense, 31 thư mục run, bảng dưới) trên cache pooled vừa nhúng, **`--device cpu`** (Rule 13 item 5). Chưa có bằng chứng suy luận trên cuda tất định, nên giữ cpu; head Dense nhỏ, chấm ~25.000 dòng trên cpu tốn vài giây.
4. Loại: `dl-sal-cafe-rnn` (s42–s44 không có `best.pt`). `dl-cat-cafe-rnn-ce` chỉ s42 (s43, s44 không có `best.pt`). Bốn hàng chờ tác giả (`dl-mtl-h256-a1.0`, `dl-cat-focal`, `dl-sal`, `dl-cat-cafe-dense-512-p20`) không chấm và không đẩy lên.

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

### Run Dense chấm trên máy thuê (31 thư mục, cpu)

| Cấu hình dashboard | Run | Task | Encoder · len |
|---|---|---|---|
| `dl-cat-cafe-dense-512` | `dl-cat-cafe-dense-512`, `-s43`, `-s44`, `-s45`, `-s46` | category | cafebert · 512 |
| `dl-cat-cafe-dense` | `dl-cat-cafe-dense` | category | cafebert · 256 |
| `dl-cat-cafe-focal-512` | `dl-cat-cafe-focal-512-s42` … `-s46` | category | cafebert · 512 |
| `dl-cat-ce` | `dl-cat-ce-0920` (khớp accuracy dev, T12.1a) | category | phobert · 256 |
| `dl-cat-bamibert-dense-1024` / `-visobert-dense-512` / `-xlmr-dense-512` | cùng tên | category | bamibert 1024 / visobert 512 / xlmr 512 |
| `dl-mtl-bamibert1024-a5-e80` / `dl-mtl-visobert512-a5-e80` / `dl-mtl-xlmr512-a5-e80` | `…-s42` | mtl | bamibert 1024 / visobert 512 / xlmr 512 |
| `dl-sal-cafe-dense-512` | `dl-sal-cafe-dense-512`, `-s43` … `-s46` | salary | cafebert · 512 |
| `dl-sal-cpu` | `dl-sal-cpu-0920`, `dl-sal-cpu-s43` … `-s46` | salary | phobert · 256 |
| `dl-sal-bamibert-dense-1024` / `-visobert-dense-512` / `-xlmr-dense-512` | cùng tên | salary | bamibert 1024 / visobert 512 / xlmr 512 |

Dense `category` trên `ext37k-sal` cần cache `ext37k-sal` **raw**; quyết định ở T12.1a, chưa nằm trong bảng cache.

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Viết `scripts/vast_run_t12_score.sh`, `--device` cho `encode_t12_cpu.py`, `scripts/score_t12.py` (Dense + RNN), `tests/test_score_t12.py` | done | smoke dev: RNN trùng `predictions_dev` 100 %, Dense khớp `metrics.json` (xem Nhật ký) |
| 1 | Thuê máy theo "Chọn máy" | todo | tác giả |
| 2 | Đẩy code + splits + `manifest.json` + `ext37k-sal.csv` + 20 thư mục run RNN + 31 thư mục run Dense lên | todo | lệnh dưới |
| 3 | Cài thư viện, `sha256sum data/processed/splits/test.csv` trên máy thuê | todo | phải bằng `70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047` |
| 4 | `bash scripts/vast_run_t12_score.sh all` trong tmux (nhúng, chấm RNN cuda, chấm Dense cpu; mỗi lượt kiểm sha256 `test.csv` trước và sau) | todo | chạy từng phần: `encode` / `rnn` / `dense` |
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
RUNS="$RUNS dl-cat-cafe-dense-512 dl-cat-cafe-dense-512-s43 dl-cat-cafe-dense-512-s44 dl-cat-cafe-dense-512-s45 dl-cat-cafe-dense-512-s46 \
 dl-cat-cafe-dense dl-cat-cafe-focal-512-s42 dl-cat-cafe-focal-512-s43 dl-cat-cafe-focal-512-s44 dl-cat-cafe-focal-512-s45 dl-cat-cafe-focal-512-s46 \
 dl-cat-ce-0920 dl-cat-bamibert-dense-1024 dl-cat-visobert-dense-512 dl-cat-xlmr-dense-512 \
 dl-mtl-bamibert1024-a5-e80-s42 dl-mtl-visobert512-a5-e80-s42 dl-mtl-xlmr512-a5-e80-s42 \
 dl-sal-cafe-dense-512 dl-sal-cafe-dense-512-s43 dl-sal-cafe-dense-512-s44 dl-sal-cafe-dense-512-s45 dl-sal-cafe-dense-512-s46 \
 dl-sal-cpu-0920 dl-sal-cpu-s43 dl-sal-cpu-s44 dl-sal-cpu-s45 dl-sal-cpu-s46 \
 dl-sal-bamibert-dense-1024 dl-sal-visobert-dense-512 dl-sal-xlmr-dense-512"
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
PYTHONPATH=src python -m pytest -q tests/test_score_t12.py
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

Kéo thêm kết quả chấm Dense và pooled cache mới (nhỏ, giữ làm bằng chứng). CSV và `04-results.md` ghi trên Mac sau đó.

Kết quả chấm: `artifacts/<run_id>/t12_score/test.json` và `ext37k-sal.json` (kéo cả `artifacts/*/t12_score`); log: `logs/t12-*.txt`.

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| 2026-10-16 | Dev xlmr/bamibert trên Mac không khớp checkpoint (F1 0.5317 so với 0.5953): `predictions_dev.parquet` của run vast có thứ tự dòng khác `dev.csv` trên Mac, nên cache dev xlmr/bamibert đã kéo về xếp theo thứ tự vast | không dùng dev xlmr/bamibert để kiểm; test thì cache nhúng ngay trên máy thuê từ `test.csv` đã đẩy lên (cùng sha256), nên thứ tự khớp |
