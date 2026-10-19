# Đợt 6: bài `salary` trên bộ v8 (MinHash) — PhoBERT và CafeBERT, Dense và RNN — CHƯA THUÊ

Task: **T15** (`tasks/T15-v8-luong.md`), chạy **toàn bộ trên máy thuê, không chạy trên Mac** (tác giả, 2026-10-17).

Script: `scripts/vast_run_v8_salary.sh` (đối số `encode | dense | rnn | all`).

Bộ dữ liệu: `data/processed/splits-minhash-1label` (`v8-minhash-1label`: train 17.350 · dev 1.927 · test 4.815, nhãn mâu thuẫn đã bỏ; báo cáo `docs/models/category/data/v8-minhash-1label.md`). Lương chỉ đọc họ `masked` (Rule 3). `test` không được nhúng, không được chấm (Rule 5).

## Các run

| Run | Encoder | Đầu | Seed | Thiết bị | Ghi chú |
|---|---|---|---|---|---|
| `mh-sal-pho-dense-256-s42..s46` | PhoBERT-v2, 256 token | Dense | 42–46 | `cpu` | Rule 13 item 5 |
| `mh-sal-cafe-dense-512-s42..s46` | CafeBERT, 512 token | Dense | 42–46 | `cpu` | Rule 13 item 5 |
| `mh-sal-pho-rnn-256-s42` | PhoBERT-v2, 256 token | BiGRU-LSTM-CNN | 42 | `cuda` | một seed |
| `mh-sal-cafe-rnn-512-s42` | CafeBERT, 512 token | BiGRU-LSTM-CNN | 42 | `cuda` | một seed |

Tổng 12 run. Run RNN trên `cuda` nên không tái lập được như quy ước Rule 13 (giống các đợt trước).

## Instance

| Mục | Giá trị |
|---|---|
| GPU | RTX 5060 Ti 16 GB (đo bằng `nvidia-smi`; torch 2.11.0+cu128 có sm_120, Python 3.12.15) |
| Disk | **60 GB** (đo bằng `df`, không phải 100 GB như dự định; cần dùng khoảng 52 GB gồm ước 20 GB hệ điều hành, xem bảng dưới. Hệ điều hành nằm ngoài 60 GB này nên còn dư) |
| RAM · CPU | 503 GB (còn trống 408 GB) · 128 luồng (đo bằng `free`, `nproc`) |
| IP / port | `ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p 42754 root@158.181.52.19` (phải chỉ rõ khóa `id_ed25519`, nếu không bị `Permission denied`) |

## Dung lượng ổ đĩa cần dùng

Tính từ số dòng thật (`train` 17.350 + `dev` 1.927 = 19.277 dòng) và công thức `n × token × hidden × 2 byte` (float16). Các mục "ước" là suy ra, chưa đo.

| Mục | Dung lượng |
|---|---|
| Token cache CafeBERT 512 (19.277 × 512 × 1.024 × 2 B) | 20,2 GB |
| Token cache PhoBERT 256 (19.277 × 256 × 768 × 2 B) | 7,6 GB |
| Vector gộp train + dev (CafeBERT 79 MB, PhoBERT 59 MB) | 0,14 GB |
| Vector gộp `ext37k-sal` họ masked (6.161 dòng: 25 MB và 19 MB) | 0,05 GB |
| Cache mô hình Hugging Face (CafeBERT 2,1 GB và PhoBERT-v2 0,5 GB, đo trên Mac) | 2,6 GB |
| Dữ liệu: `splits-minhash-1label` 217 MB, bộ chia cũ 337 MB (để khử trùng `ext37k-sal` và lấy nhãn), `ext37k-sal.csv` 15 MB, file tách từ 2,4 MB | 0,6 GB |
| Run: `best.pt`, `history.jsonl`, log (12 run, mỗi run vài MB) | dưới 0,1 GB |
| Hệ điều hành, torch CUDA, thư viện pip (ước, phụ thuộc image) | khoảng 20 GB |
| **Tổng cần dùng (ước)** | **khoảng 52 GB** |

Thuê **100 GB** để dư gấp đôi: `np.lib.format.open_memmap` cấp phát trước cả file token cache, và nhúng token cache cho cả hai encoder cùng nằm trên đĩa đến hết bước RNN. Script cũ của đợt 2 yêu cầu 150 GB vì token cache `category` + `salary` của 34 nghìn dòng. Lần này không có `category`, `train` chỉ 17.350 dòng, nên 100 GB là đủ.

## Ước tính thời gian (từ các run lương cũ, chưa đo trên bộ này)

- RNN lương cũ: CafeBERT 1.175 giây (17 epoch), PhoBERT 540 giây (13 epoch), mỗi cái trên `train` lớn hơn gấp đôi `v8`. Với `v8`, dự kiến khoảng một nửa nếu thời gian tăng tuyến tính (giả định).
- Nhúng token cache trên GPU: chưa có số đo.

## Các bước

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Chọn và thuê instance | chưa | Cần tác giả duyệt chi phí |
| 2 | Đẩy mã và dữ liệu | chưa | Lệnh bên dưới |
| 3 | Cài thư viện, `pytest` hai file test | chưa | |
| 4 | `bash scripts/vast_run_v8_salary.sh encode` | chưa | Nhúng pooled rồi token, cả hai encoder |
| 5 | `bash scripts/vast_run_v8_salary.sh dense` | chưa | 10 run |
| 6 | `bash scripts/vast_run_v8_salary.sh rnn` | chưa | 2 run |
| 7 | `bash scripts/vast_run_v8_salary.sh score` | chưa | Chấm `ext37k-sal`; nhúng ext họ masked, không nhúng test |
| 8 | Kéo `artifacts/mh-sal-*` (cả `t12_score/`) và `logs` về Mac, kiểm tra `best.pt` mở được | chưa | Checklist trong `vast/README.md` |
| 9 | Destroy instance | chưa | Chỉ sau checklist |
| 10 | Điền bảng run từ `metrics.json` và kết quả `ext37k-sal` | chưa | |

## Kết quả

⛔ Chưa có. Bảng điền từ `artifacts/<run_id>/metrics.json` sau khi kéo về.

## Lệnh đẩy dữ liệu lên (terminal Mac, thay PORT và IP)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests resources pyproject.toml \
  data/processed/manifest.json data/processed/splits-minhash-1label data/processed/splits \
  data/external/vietjobs37k/ext37k-sal.csv data/external/vietjobs37k/crosswalk_60_to_16.json \
  artifacts/t12/ext37k-sal-masked-seg.parquet \
  root@IP:/workspace/vietjobs/
```

Bắt buộc kèm `data/processed/manifest.json` (từ điển kiểu cột của `load_split`; `kinh-nghiem.md` §5). `data/processed/splits` (bộ cũ) chỉ để bước `score` khử trùng `ext37k-sal` với toàn bộ 47.707 dòng và lấy danh sách 16 nhãn; huấn luyện không đọc nó. Bước `score` **không** nhúng và **không** chấm `test` (`--ext-masked-only`). Không đẩy token cache (máy thuê tự nhúng).

## Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn scipy pyarrow underthesea pyvi pytest
tmux new -s v8sal
bash scripts/vast_run_v8_salary.sh all 2>&1 | tee logs/v8_salary_all.txt
```

## Lệnh kéo kết quả về Mac

```bash
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/mh-sal-* logs" | tar xzf - -C .
# kết quả chấm ext37k-sal nằm trong artifacts/mh-sal-*/t12_score/ext37k-sal.json, đã kèm trong lệnh trên
```

Kéo cả `best.pt`. Thư mục cache `artifacts/embeddings/splits-minhash-1label/` không cần kéo.
