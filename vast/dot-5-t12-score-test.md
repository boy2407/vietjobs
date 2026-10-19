# Đợt 5: T12.1b/c/d — nhúng cache, train lại 2 run CafeBERT RNN (seed 42), chấm 61 run trên test và ext37k-sal — CHƯA THUÊ

Task: **T12.1b** (nhúng cache + train lại + chấm RNN), **T12.1c** (chấm Dense trên `test`), **T12.1d** (chấm Dense và phân lớp trên `ext37k-sal`), tất cả trên máy thuê (tác giả đổi 2026-10-16); phụ thuộc T12.1a ([tasks/T12-dashboard-map.md](../tasks/T12-dashboard-map.md)). Quyền: AGENTS.md Rule 5 item 6 (tác giả, 2026-10-15): chấm `best.pt` đã lưu trên `test`. CSV và các hàng `04-results.md` vẫn ghi trên Mac sau khi kéo về.

Script: `scripts/vast_run_t12_score.sh` (đối số `all | encode | train | rnn | dense`; gọi `scripts/encode_t12_cpu.py --device cuda`, `python -m vietjobs.dl.encode`, `python -m vietjobs.dl.train_dl`, `scripts/score_t12.py`) | Instance: chưa có (GPU, disk, RAM, IP/port điền khi thuê).

Phạm vi trên máy thuê (cả 34 hàng dashboard):
1. **Nhúng pooled** `test` (raw, masked) và `ext37k-sal` (masked, raw) cho 6 cặp encoder/len (bảng cache). Chỉ **đọc** test để nhúng, không ghi `test.csv`.
2. **Train lại 2 run CafeBERT 512 + BiGRU-LSTM-CNN, chỉ seed 42** (tác giả, 2026-10-16), cho hai cấu hình không có checkpoint dùng được (`dl-cat-cafe-rnn-ce`: s42 hỏng, s43/s44 thiếu; `dl-sal-cafe-rnn`: s42–s44 thiếu): `dl-cat-cafe-rnn-ce-s42-t12`, `dl-sal-cafe-rnn-s42-t12`, đúng config gốc (`scripts/vast_run_cafe_rnn.sh`); không ghi đè thư mục cũ. Hai hàng dashboard đó chấm với n = 1 ("1 of 3 seeds, retrained -t12", không có sd). Cần token cache CafeBERT 512 (`category` raw và `salary` masked, train + dev).
3. **Chấm 21 run RNN** (19 có sẵn + 2 `-t12`) trên `test` và `ext37k-sal`: nhúng token từng batch, chạy head (đường `predict_rnn` của `scripts/eval_external_mtl.py`, không ghi token cache); mỗi (encoder, họ cột) nhúng một lượt cho mọi run cùng nhóm.
4. **Chấm 40 run Dense** (đơn nhiệm + MTL dense) trên cache pooled, **`--device cpu`** (Rule 13 item 5; chưa có bằng chứng suy luận trên cuda tất định).
5. Trên `ext37k-sal`: lương (silver, tạm thời, T10.1 bị tác giả bỏ qua) cho run `salary`/`mtl`; phân lớp (crosswalk, như `eval_external.py`) cho mọi run có đầu phân lớp, đọc raw cho run `category`, masked cho MTL.

Bốn hàng từng chờ tác giả đã được duyệt cách ánh xạ (xem sheet T12). Số run: 61 = 40 Dense + 21 RNN. Mỗi run chỉ chấm **một lần** trên mỗi tập: `score_t12.py` bỏ qua run đã có `artifacts/<run_id>/t12_score/<dataset>.json` (kiểm bằng test `test_already_scored_blocks_a_second_scoring` và chạy thử trên Mac), nên chạy lại script không chấm `test` lần hai; muốn chấm lại phải xoá file đó có chủ ý.

## Cache cần tạo (pooled, float32; ước tính từ kích thước mảng, chưa đo)

| Encoder | max_len | `test` raw | `test` masked | `ext37k-sal` masked | `ext37k-sal` raw |
|---|---|---|---|---|---|
| `phobert` | 256 | cần | cần | cần (có trên Mac) | cần |
| `cafebert` | 256 | cần | — | — | cần |
| `cafebert` | 512 | cần | cần | cần (có trên Mac) | cần |
| `visobert` | 512 | cần | cần | cần | cần |
| `xlmr` | 512 | cần | cần | cần | cần |
| `bamibert` | 1024 | cần | cần | cần | cần |

Mỗi tệp ≈ 6.161–9.541 × (768 hoặc 1024) × 4 B ≈ 19–39 MB; tổng dưới 1 GB. `ext37k-sal` masked của PhoBERT cần `artifacts/t12/ext37k-sal-masked-seg.parquet` (tạo trên Mac; `.venv-dl` không có `underthesea`). `ext37k-sal` raw đi qua `eval_external.prepare_frame` (preprocess + tách từ, như lúc chấm gold/test của 37K), nên máy thuê cần `underthesea`.

Token cache CafeBERT 512 cho bước train: ≈ 34 GB mỗi task (train + dev), ≈ 68 GB cho hai task (ước tính từ `vast_run_cafe_rnn.sh` và `kinh-nghiem.md` §1; chưa đo lại).

## Chọn máy (ước tính, chưa đo)

- 1× GPU 16 GB (RTX 5060 Ti / 4070 Ti / 3080), như Đợt 3–4 (Đợt 3 đo trên 5060 Ti + EPYC 9534).
- Disk **150 GB** (≈ 68 GB token cache + ≈ 15 GB model + run + cache pooled); RAM **≥ 64 GB, nên 96 GB** (token cache một task ≈ 34 GB phải nằm trong page cache, `kinh-nghiem.md` §1–2); CPU ≥ 8 lõi (EPYC / Xeon); PCIe 4.0 x16; NVMe ≥ 3.000 MB/s; CUDA ≥ 12.1; image có `/venv/main`.
- Giá ≤ $0,40/h; reliability ≥ 98 %.

## Ước tính thời gian và chi phí

Mốc **đo được** (Đợt 3, 5060 Ti): CafeBERT 512 encode ≈ 30,8 tin/s; hai run gốc seed 42 `dl-cat-cafe-rnn-ce-s42` và `dl-sal-cafe-rnn-s42` mất 27,3 và 19,6 phút, tổng 46,9 phút (`artifacts/<run_id>/metrics.json`, `seconds`). Các tốc độ khác dưới đây là **ước tính**.

| Phần | Cách tính | Ước tính |
|---|---|---|
| Nhúng pooled (6 cặp × ≈ 31.400 lượt) | CafeBERT 512 ≈ 17 phút (đo), CafeBERT 256 ≈ 7, PhoBERT ≈ 10, ViSoBERT ≈ 6, XLM-R ≈ 6, BamiBERT 1024 ≈ 15 | ≈ 1 giờ |
| Token cache CafeBERT 512 (2 họ × 38.166 tin) | 76.332 / 30,8 tin/s | ≈ 41 phút |
| Train lại 2 run `-t12` (seed 42) | cùng config nên lấy thời gian gốc | ≈ 47 phút |
| Chấm 21 run RNN (5 encoder × 2 họ × ≈ 15.700 lượt) | nhúng token lại một lượt mỗi nhóm | ≈ 1 giờ |
| Chấm 40 run Dense (cpu) + bootstrap ext37k-sal | mỗi run vài giây | ≈ 10 phút |
| **Tổng tính toán** | | **≈ 3,6 giờ** |

Cộng ≈ 1 giờ đẩy code, cài đặt, kéo về ⇒ thuê **6 giờ** (để dư), **≈ $1,5–2,5**. Run `train` có thể lệch vài chục phút so với số gốc (máy khác, kết quả dev có thể khác dashboard).

## Bảng run (số điền từ `artifacts/<run_id>/metrics.json` và `t12_score/*.json` sau khi chạy)

RNN, 21 run:

| Cấu hình dashboard | Run | Task | Encoder · len | Trạng thái |
|---|---|---|---|---|
| `dl-cat-cafe-rnn-ce` | `dl-cat-cafe-rnn-ce-s42-t12` (1 of 3 seeds) | category | cafebert · 512 | train lại, todo |
| `dl-sal-cafe-rnn` | `dl-sal-cafe-rnn-s42-t12` (1 of 3 seeds) | salary | cafebert · 512 | train lại, todo |
| `dl-cat-rnn-ce` | `dl-cat-rnn-ce`, `-s42`, `-s43`, `-s44` | category | phobert · 256 | todo |
| `dl-cat-rnn-focal` | `dl-cat-rnn-focal` | category | phobert · 256 | todo |
| `dl-cat-bamibert-rnn-1024` / `-visobert-rnn-512` / `-xlmr-rnn-512` | cùng tên | category | bamibert 1024 / visobert 512 / xlmr 512 | todo |
| `dl-mtl-cafe-rnn-a5` / `dl-mtl-bamibert-rnn1024-a5` / `dl-mtl-visobert-rnn512-a5` / `dl-mtl-xlmr-rnn512-a5` | `…-s42` | mtl | cafebert 512 / bamibert 1024 / visobert 512 / xlmr 512 | todo |
| `dl-sal-rnn` | `dl-sal-rnn`, `-s42`, `-s43`, `-s44` | salary | phobert · 256 | todo |
| `dl-sal-bamibert-rnn-1024` / `-visobert-rnn-512` / `-xlmr-rnn-512` | cùng tên | salary | bamibert 1024 / visobert 512 / xlmr 512 | todo |

Dense, 40 run (cpu):

| Cấu hình dashboard | Run | Task | Encoder · len |
|---|---|---|---|
| `dl-cat-cafe-dense-512` | `dl-cat-cafe-dense-512`, `-s43` … `-s46` | category | cafebert · 512 |
| `dl-cat-cafe-dense-512-p20` | `dl-cat-cafe-dense-512-p20-s42` … `-s44` | category | cafebert · 512 |
| `dl-cat-cafe-dense` | `dl-cat-cafe-dense` | category | cafebert · 256 |
| `dl-cat-cafe-focal-512` | `dl-cat-cafe-focal-512-s42` … `-s46` | category | cafebert · 512 |
| `dl-cat-ce` | `dl-cat-ce-0920` | category | phobert · 256 |
| `dl-cat-focal` | `dl-cat-focal-0920`, `dl-cat-focal-s2` | category | phobert · 256 |
| `dl-cat-bamibert-dense-1024` / `-visobert-dense-512` / `-xlmr-dense-512` | cùng tên | category | bamibert 1024 / visobert 512 / xlmr 512 |
| `dl-mtl-h256-a1.0` | `dl-mtl-h256-a1.0-1001-0735`, `-1001-0746` | mtl | phobert · 256 |
| `dl-mtl-bamibert1024-a5-e80` / `dl-mtl-visobert512-a5-e80` / `dl-mtl-xlmr512-a5-e80` | `…-s42` | mtl | bamibert 1024 / visobert 512 / xlmr 512 |
| `dl-sal` | `dl-sal-0920`, `dl-sal-s2` | salary | phobert · 256 |
| `dl-sal-cafe-dense-512` | `dl-sal-cafe-dense-512`, `-s43` … `-s46` | salary | cafebert · 512 |
| `dl-sal-cpu` | `dl-sal-cpu-0920`, `-s43` … `-s46` | salary | phobert · 256 |
| `dl-sal-bamibert-dense-1024` / `-visobert-dense-512` / `-xlmr-dense-512` | cùng tên | salary | bamibert 1024 / visobert 512 / xlmr 512 |

## Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Viết `vast_run_t12_score.sh`, `--device` cho `encode_t12_cpu.py`, `score_t12.py` (Dense + RNN + ext37k-sal raw), stage `train`, `tests/test_score_t12.py` | done | smoke dev: RNN trùng `predictions_dev` 100 %, Dense khớp `metrics.json`; ext37k-sal category chạy trên 147 dòng (xem Nhật ký) |
| 1 | Thuê máy theo "Chọn máy" | todo | tác giả |
| 2 | Đẩy code + splits + `manifest.json` + `ext37k-sal.csv` + 59 thư mục run lên | todo | lệnh dưới; 2 run `-t12` chưa có nên không đẩy |
| 3 | Cài thư viện, pytest `tests/test_score_t12.py`, `sha256sum data/processed/splits/test.csv` | todo | phải bằng `70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047` |
| 4 | `bash scripts/vast_run_t12_score.sh all` trong tmux (encode → train → chấm RNN cuda → chấm Dense cpu; sha256 `test.csv` kiểm trước và sau) | todo | hoặc từng phần: `encode` / `train` / `rnn` / `dense` |
| 5 | Kéo cache, 2 thư mục `-t12` (kèm `best.pt`), kết quả chấm, log về Mac; so `md5sum` | todo | lệnh `scp -r` dưới đây không loại `best.pt` |
| 5b | Trên Mac, **trước khi destroy**: `torch.load` từng `best.pt` của 2 run `-t12` (và kiểm tệp mở được) | todo | checklist `vast/README.md` mục 2; Đợt 5 từng gặp một `best.pt` bị cắt |
| 6 | `sha256sum test.csv` trên máy thuê và trên Mac bằng giá trị gốc | todo | |
| 7 | Mac: thêm 2 dòng train `-t12` vào `docs/04-results.md` (append-only), chạy `scripts/export_runs.py` | todo | Rule 1, Rule 13 |
| 8 | Checklist `README.md`, rồi destroy | todo | tác giả bấm trên web; token cache ≈ 68 GB mất khi destroy |

## Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
shasum -a 256 data/processed/splits/test.csv   # 70ba98d5…fba047
RNN="dl-cat-rnn-ce dl-cat-rnn-ce-s42 dl-cat-rnn-ce-s43 dl-cat-rnn-ce-s44 dl-cat-rnn-focal \
 dl-cat-bamibert-rnn-1024 dl-cat-visobert-rnn-512 dl-cat-xlmr-rnn-512 \
 dl-mtl-cafe-rnn-a5-s42 dl-mtl-bamibert-rnn1024-a5-s42 dl-mtl-visobert-rnn512-a5-s42 dl-mtl-xlmr-rnn512-a5-s42 \
 dl-sal-rnn dl-sal-rnn-s42 dl-sal-rnn-s43 dl-sal-rnn-s44 dl-sal-bamibert-rnn-1024 dl-sal-visobert-rnn-512 dl-sal-xlmr-rnn-512"
DENSE="dl-cat-cafe-dense-512 dl-cat-cafe-dense-512-s43 dl-cat-cafe-dense-512-s44 dl-cat-cafe-dense-512-s45 dl-cat-cafe-dense-512-s46 \
 dl-cat-cafe-dense-512-p20-s42 dl-cat-cafe-dense-512-p20-s43 dl-cat-cafe-dense-512-p20-s44 dl-cat-cafe-dense \
 dl-cat-cafe-focal-512-s42 dl-cat-cafe-focal-512-s43 dl-cat-cafe-focal-512-s44 dl-cat-cafe-focal-512-s45 dl-cat-cafe-focal-512-s46 \
 dl-cat-ce-0920 dl-cat-focal-0920 dl-cat-focal-s2 dl-cat-bamibert-dense-1024 dl-cat-visobert-dense-512 dl-cat-xlmr-dense-512 \
 dl-mtl-h256-a1.0-1001-0735 dl-mtl-h256-a1.0-1001-0746 dl-mtl-bamibert1024-a5-e80-s42 dl-mtl-visobert512-a5-e80-s42 dl-mtl-xlmr512-a5-e80-s42 \
 dl-sal-0920 dl-sal-s2 dl-sal-cafe-dense-512 dl-sal-cafe-dense-512-s43 dl-sal-cafe-dense-512-s44 dl-sal-cafe-dense-512-s45 dl-sal-cafe-dense-512-s46 \
 dl-sal-cpu-0920 dl-sal-cpu-s43 dl-sal-cpu-s44 dl-sal-cpu-s45 dl-sal-cpu-s46 \
 dl-sal-bamibert-dense-1024 dl-sal-visobert-dense-512 dl-sal-xlmr-dense-512"
FILES=""
for r in $RNN $DENSE; do for f in best.pt best_sal.pt config.json scaler.npz; do
  [ -f artifacts/$r/$f ] && FILES="$FILES artifacts/$r/$f"; done; done
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml \
  data/processed/splits data/processed/manifest.json \
  data/external/vietjobs37k/ext37k-sal.csv data/external/vietjobs37k/crosswalk_60_to_16.json \
  artifacts/t12/ext37k-sal-masked-seg.parquet $FILES \
  root@IP:/workspace/vietjobs/
```

Bắt buộc kèm `manifest.json` (`kinh-nghiem.md` §5) và `splits/train.csv`, `dev.csv` (huấn luyện đọc train + dev; `OurIndex` đọc cả ba split). Không đẩy token cache; `docs/` không đẩy (train sẽ tạo `docs/04-results.md` riêng trên máy thuê, chỉ chứa 2 dòng `-t12`).

## Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn scipy pyarrow underthesea pyvi pytest
PYTHONPATH=src python -m pytest -q tests/test_score_t12.py tests/test_dl_encode.py tests/test_dl_heads.py
sha256sum data/processed/splits/test.csv
tmux new -s t12
bash scripts/vast_run_t12_score.sh all
sha256sum data/processed/splits/test.csv
```

## Kéo về

`tar czf - … | tar xzf -` qua proxy từng treo (`kinh-nghiem.md` §5): dùng `scp` từng file, so `md5sum` hai đầu; **kéo `best.pt` của 2 run `-t12`** (Đợt 2 từng mất 5 file vì bỏ bước này).

```bash
ssh -p PORT root@IP "cd /workspace/vietjobs && ls -l artifacts/dl-*-t12/best.pt artifacts/*/t12_score/*.json artifacts/embeddings/*test* artifacts/embeddings/*ext37k-sal* logs/t12* docs/04-results.md"
for r in dl-cat-cafe-rnn-ce-s42-t12 dl-sal-cafe-rnn-s42-t12; do
  mkdir -p artifacts/$r && scp -P PORT -r root@IP:/workspace/vietjobs/artifacts/$r/. artifacts/$r/   # lấy cả best.pt
  ssh -p PORT root@IP "md5sum /workspace/vietjobs/artifacts/$r/best.pt"; md5 artifacts/$r/best.pt
  PYTHONPATH=src .venv-dl/bin/python -c "import torch,sys; torch.load('artifacts/$r/best.pt', map_location='cpu'); print('$r best.pt ok')"
done
scp -P PORT root@IP:/workspace/vietjobs/docs/04-results.md logs/t12-server-04-results.md   # 2 dòng train -t12
```

Lệnh kéo **không** loại `best.pt` (`scp -r` cả thư mục run). Checklist `vast/README.md` mục 2 yêu cầu thêm: `torch.load` mở được từng `best.pt` `-t12` trên Mac (vòng lặp trên) trước khi destroy.

Kết quả chấm: `artifacts/<run_id>/t12_score/test.json` và `ext37k-sal.json`. CSV và `docs/04-results.md` ghi trên Mac sau đó; không sửa dòng nào đã có.

## Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| 2026-10-16 | Dev xlmr/bamibert trên Mac không khớp checkpoint (F1 0.5317 so với 0.5953): `predictions_dev.parquet` của run vast có thứ tự dòng khác `dev.csv` trên Mac, nên cache dev xlmr/bamibert đã kéo về xếp theo thứ tự vast | không dùng dev xlmr/bamibert để kiểm; test thì cache nhúng ngay trên máy thuê từ `test.csv` đã đẩy lên (cùng sha256), nên thứ tự khớp |
| 2026-10-16 | `dl-cat-cafe-rnn-ce-s42/best.pt` (6.641.333 B) không phải zip hợp lệ (bị cắt khi kéo về); quét 59 run có checkpoint chỉ có file này hỏng | train lại seed 42 thành `-t12` (tác giả chỉ chọn một seed mỗi cấu hình); thư mục gốc giữ nguyên |
