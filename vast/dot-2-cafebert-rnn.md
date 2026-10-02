# ✅ Đợt 2: CafeBERT 512-token + RNN (T8.16/T8.17) — HOÀN TẤT

Script: `scripts/vast_run_cafe_rnn.sh` | Instance: **RTX 4090, 24 GB VRAM, 100 GB disk** (m:15969), IP `81.27.69.180`, port `29935`

| Run | macro-F1 / MAE | Ghi chú |
|---|---|---|
| `dl-cat-cafe-rnn-ce-s42` | **0.6172** | gate ✅ (best epoch 9) |
| `dl-cat-cafe-rnn-ce-s43` | **0.6223** 🚀 | kỷ lục mới toàn dự án (best epoch 16) |
| `dl-cat-cafe-rnn-ce-s44` | **0.6060** | hoàn tất (best epoch 17) |
| `dl-sal-cafe-rnn-s42` | **3.89 tr** | gate ✅ (best epoch 9) |
| `dl-sal-cafe-rnn-s43` | **3.84 tr** | hoàn tất (best epoch 5) |
| `dl-sal-cafe-rnn-s44` | **3.84 tr** | hoàn tất (best epoch 8) |

**Trung bình Đợt 2 (CafeBERT 512 + RNN):**
- **Category (Macro-F1):** **`0.6152 ± 0.0084`** (so với PhoBERT RNN `0.6097 ± 0.0071`, CafeBERT Dense `0.6077`)
- **Salary (MAE):** **`3.857 ± 0.029 triệu`** (so với PhoBERT RNN `3.940 ± 0.062 triệu`, baseline `4.10 triệu`)

### Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Destroy instance cũ | done | |
| 2 | Thuê RTX 4090 (m:15969, 100 GB disk) | done | IP 81.27.69.180, port 29935 |
| 3 | Gắn SSH key + kết nối | done | ssh -p 29935 root@81.27.69.180 |
| 4 | Đẩy code + dữ liệu lên | done | 142 files, ~336 MB |
| 5 | Cài thư viện Python | done | transformers 4.46, underthesea, pyvi ok |
| 6 | Chạy gate T8.16 (encode + seed 42) | done | dl-cat F1 0.6172, dl-sal MAE 3.89 tr |
| 7 | Kiểm tra gate | done | Gate PASS: vượt cả PhoBERT RNN lẫn CafeBERT Dense |
| 8 | Chạy T8.17 (seed 43, 44) | done | Hoàn tất 6/6 runs cả 3 seed |
| 9 | Kéo kết quả về Mac | done | rsync macOS treo → `ssh … tar czf - --exclude=best.pt`; đủ 6 run + logs, `best.pt` chỉ có của `dl-cat-cafe-rnn-ce-s42` |
| 10 | Destroy instance | done | 2026-09-30; 5 file `best.pt` không kéo về, đã mất cùng instance |
| 11 | Phân tích: mean±sd 3 seed, so cặp | done | Xem "Kết quả bước 11" bên dưới |

### Kết quả bước 11 (dev, nguồn: `artifacts/<run_id>/metrics.json`)

Cả 12 run RNN trên máy thuê (Đợt 1 + Đợt 2) chạy trên **`cuda`** (`env.json`), không phải `cpu` như quy ước Rule 13 mục 5. Trong `docs/04-results.md` chúng mang nhãn `+cuda` ở cột Prep.

| Họ | n | cat macro-F1 | cat F1 no-junk | sal MAE (tr) | sal R² log |
|---|---|---|---|---|---|
| CafeBERT 512 + RNN | 3 (s42–44) | 0.6152 ± 0.0084 | 0.6555 ± 0.0137 | 3.855 ± 0.030 | 0.5955 ± 0.0017 |
| PhoBERT 256 + RNN | 3 (s42–44) | 0.6097 ± 0.0071 | 0.6527 ± 0.0073 | 3.939 ± 0.058 | 0.5698 ± 0.0133 |
| CafeBERT 512 Dense | 5 (s42–46) | 0.6037 ± 0.0037 | 0.6445 ± 0.0018 | 4.130 ± 0.022 | 0.5127 ± 0.0056 |
| CafeBERT 512 Dense x512 | 5 (s42–46) | 0.6008 ± 0.0061 | 0.6426 ± 0.0047 | 4.117 ± 0.048 | 0.5190 ± 0.0138 |

So cặp cùng seed (CafeBERT RNN trừ đối thủ, s42/s43/s44):

| Đối thủ | Δ cat macro-F1 | Δ sal MAE (tr) |
|---|---|---|
| PhoBERT RNN | +0.0134, +0.0047, −0.0018 → +0.0055 ± 0.0076 | −0.067, −0.037, −0.150 → −0.085 ± 0.058 |
| CafeBERT Dense 512 | +0.0095, +0.0154, +0.0026 → +0.0091 ± 0.0064 | −0.224, −0.303, −0.263 → −0.263 ± 0.039 |

### Lệnh đẩy dữ liệu lên (terminal Mac, thay PORT và IP)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml data/processed/splits \
  artifacts/embeddings/cafebert-dev-masked-len512.{npy,json} \
  artifacts/embeddings/cafebert-dev-raw-len512.{npy,json} \
  artifacts/embeddings/cafebert-train-masked-len512.{npy,json} \
  artifacts/embeddings/cafebert-train-raw-len512.{npy,json} \
  root@IP:/workspace/vietjobs/
```

### Lệnh chạy (trong tmux trên máy thuê)

```bash
cd /workspace/vietjobs
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn pyarrow underthesea pyvi pytest
bash scripts/vast_run_cafe_rnn.sh gate
# Sau khi gate pass:
bash scripts/vast_run_cafe_rnn.sh rest
```

### Lệnh kéo kết quả về Mac

rsync của macOS (openrsync) bị treo khi kéo từ máy thuê. Dùng `tar` qua ssh:

```bash
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - --exclude=best.pt artifacts/dl-*-cafe-rnn-*-s4? logs" | tar xzf - -C .
```

Lệnh rsync cũ (bị treo trên macOS):

```bash
rsync -avz -e "ssh -p PORT" \
  'root@IP:/workspace/vietjobs/artifacts/dl-cat-cafe-rnn-ce-s42' \
  'root@IP:/workspace/vietjobs/artifacts/dl-cat-cafe-rnn-ce-s43' \
  'root@IP:/workspace/vietjobs/artifacts/dl-cat-cafe-rnn-ce-s44' \
  'root@IP:/workspace/vietjobs/artifacts/dl-sal-cafe-rnn-s42' \
  'root@IP:/workspace/vietjobs/artifacts/dl-sal-cafe-rnn-s43' \
  'root@IP:/workspace/vietjobs/artifacts/dl-sal-cafe-rnn-s44' \
  artifacts/
rsync -avz -e "ssh -p PORT" root@IP:/workspace/vietjobs/logs/ logs/
```

