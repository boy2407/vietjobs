# Đợt 17: Huấn luyện 4 MTL RNN còn lại + Chấm điểm toàn diện Test & Ext37K (MTL RNN, MTL Dense, Salary Dense) — ĐÃ TẮT

Task: Hoàn tất toàn bộ các mô hình Đa nhiệm (MTL RNN) và chấm điểm toàn bộ các mô hình Dense trên tập `test` (9.541 tin) và `ext37k-sal` (6.111 tin) để điền trọn vẹn vào `Score_last4` và `Dashboard_last4`.
Script tổng hợp: `scripts/vast_run_dot17.sh`
Instance thực tế: 1x RTX 5090 32 GB, 64 vCPU, RAM 251 GB, đĩa 100 GB, `61.32.91.194:18549`, torch 2.14.1+cu132, transformers 4.46.3. Bắt đầu 2026-10-21 07:17 UTC; chạy `/workspace/run17.sh` (tmux `dot17`) — giống `vast_run_dot17.sh` nhưng xóa cache token sau mỗi mô hình vì 4 cache RNN (~120 GB) vượt đĩa 100 GB, và bỏ giai đoạn xuất Excel (chạy trên Mac).
Thay đổi 2026-10-21 08:50 UTC (tác giả): hôm nay chỉ train CafeBERT (máy này) + ViSoBERT (Đợt 18); **BamiBERT và XLM-R hoãn sang hôm sau**. Trên máy này có `metrics.json` giả trong `artifacts/dl-mtl-{vis,xlm,bam}-rnn-b01-cw-s42/` để `run17.sh` bỏ qua — xóa trước khi kéo về.
Instance đề xuất ban đầu: 1x RTX 4090 24 GB / RTX 3090 24 GB / RTX A5000 / A6000 (>= 24 GB VRAM, RAM >= 64 GB, đĩa >= 80 GB NVMe), giá ~$0.30 – $0.45/giờ.

---

## 1. Danh sách mô hình cần huấn luyện (Train MTL RNN)

PhoBERT RNN MTL (`dl-mtl-pho-rnn-b01-cw-s42`) đã huấn luyện xong trên Dev và chấm điểm xong trên Test. Đợt 17 chỉ huấn luyện **4 mô hình MTL RNN còn lại**:

| STT | Model | Run ID | Cấu hình chính | Thời gian dự kiến (CUDA) |
|:---:|---|---|---|:---:|
| 1 | **CafeBERT** | `dl-mtl-caf-rnn-b01-cw-s42` | len 512, hidden 256, conv 64, cw, $\alpha=5.0$, $\beta=0.1$, seed 42 | ~35–45 phút |
| 2 | **ViSoBERT** | `dl-mtl-vis-rnn-b01-cw-s42` | len 512, hidden 256, conv 64, cw, $\alpha=5.0$, $\beta=0.1$, seed 42 | ~35–45 phút |
| 3 | **XLM-R** | `dl-mtl-xlm-rnn-b01-cw-s42` | len 512, hidden 256, conv 64, cw, $\alpha=5.0$, $\beta=0.1$, seed 42 | ~35–45 phút |
| 4 | **BamiBERT** | `dl-mtl-bam-rnn-b01-cw-s42` | len 1024, hidden 100, conv 50, cw, $\alpha=5.0$, $\beta=0.1$, seed 42 | ~50–60 phút |

*Tổng thời gian huấn luyện 4 MTL RNN:* **~2,5 – 3,0 giờ**.

---

## 2. Danh sách mô hình cần chấm điểm (Score Test 9.541 tin & Ext37K 6.111 tin)

Sau khi train xong, tiến hành chấm điểm tự động trên cả hai tập dữ liệu kiểm thử:

### A. Nhóm MTL RNN (4 mô hình vừa train)
- `dl-mtl-caf-rnn-b01-cw-s42`
- `dl-mtl-vis-rnn-b01-cw-s42`
- `dl-mtl-xlm-rnn-b01-cw-s42`
- `dl-mtl-bam-rnn-b01-cw-s42`

### B. Nhóm MTL Dense (5 mô hình đã train trên Dev, cần chấm Test/Ext37k)
- `dl-mtl-pho-dense-last4-cw-s42`
- `dl-mtl-caf-dense-last4-cw-s42`
- `dl-mtl-vis-dense-last4-cw-s42`
- `dl-mtl-xlm-dense-last4-cw-s42`
- `dl-mtl-bam-dense-last4-cw-s42`

### C. Nhóm Single-task Salary Dense (5 họ encoder × 5 seeds = 25 runs)
- PhoBERT: `dl-sal-pho-dense-last4-b01-cw-s42..s46` (5 seeds)
- CafeBERT: `dl-sal-cafe-dense-last4-b01-cw-s42..s46` (5 seeds)
- ViSoBERT: `dl-sal-visobert-dense-last4-b01-cw-s42..s46` (5 seeds)
- XLM-R: `dl-sal-xlmr-dense-last4-b01-cw-s42..s46` (5 seeds)
- BamiBERT: `dl-sal-bamibert-dense-last4-b01-cw-s42..s46` (5 seeds)

*Tổng thời gian chấm điểm toàn bộ:* **~10–15 phút** (chạy bằng `scripts/eval_dense_all.py` và `scripts/eval_mtl_all.py`).

---

## 3. Danh sách Checkpoints & Artifacts cần kéo về Mac / THor-macbook

Sau khi chạy xong, các thư mục và file sau đây **phải được kéo về an toàn trước khi destroy instance**:

1. **4 thư mục artifacts của MTL RNN:**
   - `artifacts/dl-mtl-caf-rnn-b01-cw-s42/`
   - `artifacts/dl-mtl-vis-rnn-b01-cw-s42/`
   - `artifacts/dl-mtl-xlm-rnn-b01-cw-s42/`
   - `artifacts/dl-mtl-bam-rnn-b01-cw-s42/`
   *(Mỗi thư mục chứa `best.pt`, `best_sal.pt`, `scaler.npz`, `metrics.json`, `metrics_test.json`, `external_37k/metrics.json`, `predictions_dev.parquet`, `predictions_test.parquet`, `history.jsonl`)*

2. **Các file kết quả chấm điểm của nhóm Dense:**
   - `artifacts/dl-mtl-*-dense-last4-*/metrics_test.json`
   - `artifacts/dl-mtl-*-dense-last4-*/external_37k/metrics.json`
   - `artifacts/dl-sal-*-dense-last4-*/metrics_test.json`
   - `artifacts/dl-sal-*-dense-last4-*/external_37k/metrics.json`
   - `artifacts/eval_dense_results_summary.json`
   - `artifacts/eval_mtl_results_summary.json`

3. **Toàn bộ logs:**
   - `logs/dl-mtl-*.txt`
   - `logs/dot17-all.txt`
   - `logs/eval_dense_all.txt`

---

## 4. Quy trình chạy trên Vast.ai

### Bước 1: Khởi tạo trên Vast.ai
- Chọn GPU: RTX 4090 / RTX 3090 / A5000 (24 GB VRAM), image PyTorch / CUDA 12+.
- Mở terminal và clone repo hoặc kết nối qua Tailscale:
  ```bash
  # Tùy chọn: Kết nối Tailscale để truyền file trực tiếp sang THor-macbook (100.92.9.2)
  curl -fsSL https://tailscale.com/install.sh | sh
  tailscale up --authkey <TAILSCALE_AUTHKEY>

  # Clone repo nhánh t8-rnn-colab
  git clone -b t8-rnn-colab https://github.com/boy2407/vietjobs.git /workspace/vietjobs
  cd /workspace/vietjobs
  ```

### Bước 2: Đồng bộ dữ liệu splits & checkpoints Dense từ Mac/Drive (nếu chưa có)
```bash
# Từ máy Mac kéo sang Vast.ai:
rsync -avz -e "ssh -p PORT" data/ external_37k/ artifacts/ root@IP:/workspace/vietjobs/
```

### Bước 3: Khởi chạy trong tmux
```bash
tmux new -s dot17
bash scripts/vast_run_dot17.sh
```

### Bước 4: Theo dõi tiến độ
```bash
tail -f /workspace/vietjobs/logs/dot17-all.txt
```

### Bước 5: Kéo kết quả về Mac (dùng tar qua ssh hoặc Tailscale)
```bash
# Từ máy Mac (hoặc THor-macbook):
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - \
    artifacts/dl-mtl-*-rnn-* \
    artifacts/dl-mtl-*-dense-* \
    artifacts/dl-sal-*-dense-* \
    artifacts/eval_* \
    logs" | tar xzf - -C .
```

---

## 5. Tiến độ thực hiện

| # | Bước công việc | Trạng thái |
|:---:|---|:---:|
| 1 | Chuẩn bị script chạy và tài liệu Đợt 17 | done |
| 2 | Thuê instance vast.ai, cấu hình môi trường | done |
| 3 | Chạy `/workspace/run17.sh` trong tmux `dot17` | doing |
| 4 | Kéo toàn bộ artifacts và logs về Mac, kiểm tra `best.pt` bằng `torch.load` | todo |
| 5 | Chạy `gws` cập nhật tự động vào `Score_last4` và `Dashboard_last4` | todo |
| 6 | Cập nhật kết quả vào `docs/04-results.md` và `docs/06-baseline-dl.md` | todo |
| 7 | Xác nhận checklist, destroy instance trên vast.ai | todo |

---

## 6. Bảng kết quả tổng hợp Đợt 17

### A. Huấn luyện Multi-Task RNN (`dev` 3.812 tin)
| Model | Run ID | Best Epoch | Macro F1 (Cat) | MAE (Sal) | RMSE (Sal) | $R^2_{\log}$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| PhoBERT-v2 | `dl-mtl-pho-rnn-b01-cw-s42` | 64 | 0,6107 | 3,666 tr | 7,215 tr | 0,638 |
| CafeBERT | `dl-mtl-caf-rnn-b01-cw-s42` | ... | ... | ... | ... | ... |
| ViSoBERT | `dl-mtl-vis-rnn-b01-cw-s42` | ... | ... | ... | ... | ... |
| XLM-R | `dl-mtl-xlm-rnn-b01-cw-s42` | ... | ... | ... | ... | ... |
| BamiBERT | `dl-mtl-bam-rnn-b01-cw-s42` | ... | ... | ... | ... | ... |

### B. Chấm điểm Test (9.541 tin) & Ext37K (6.111 tin)
| Model | Nhóm | Test F1 | Test Acc | Test MAE | Test RMSE | Ext37K F1 | Ext37K MAE | Ext37K Spearman |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PhoBERT MTL RNN | MTL RNN | 0,5992 | 0,6316 | 3,683 tr | 6,931 tr | 0,5912 | 4,462 tr | 0,482 |
| CafeBERT MTL RNN | MTL RNN | ... | ... | ... | ... | ... | ... | ... |
| ViSoBERT MTL RNN | MTL RNN | ... | ... | ... | ... | ... | ... | ... |
| XLM-R MTL RNN | MTL RNN | ... | ... | ... | ... | ... | ... | ... |
| BamiBERT MTL RNN | MTL RNN | ... | ... | ... | ... | ... | ... | ... |
| PhoBERT MTL Dense | MTL Dense | ... | ... | ... | ... | ... | ... | ... |
| CafeBERT MTL Dense | MTL Dense | ... | ... | ... | ... | ... | ... | ... |
| ViSoBERT MTL Dense | MTL Dense | ... | ... | ... | ... | ... | ... | ... |
| XLM-R MTL Dense | MTL Dense | ... | ... | ... | ... | ... | ... | ... |
| BamiBERT MTL Dense | MTL Dense | ... | ... | ... | ... | ... | ... | ... |
| PhoBERT Salary Dense | Sal Dense (5s) | — | — | ... | ... | — | ... | ... |
| CafeBERT Salary Dense | Sal Dense (5s) | — | — | ... | ... | — | ... | ... |
| ViSoBERT Salary Dense | Sal Dense (5s) | — | — | ... | ... | — | ... | ... |
| XLM-R Salary Dense | Sal Dense (5s) | — | — | ... | ... | — | ... | ... |
| BamiBERT Salary Dense | Sal Dense (5s) | — | — | ... | ... | — | ... | ... |

## Kết thúc 2026-10-21 ~10:20 UTC (tác giả tắt máy)

- CafeBERT MTL RNN train xong, chấm cùng máy → hợp lệ (dev chấm lại khớp `metrics.json` tuyệt đối). Đã kéo về Mac, 32/32 run kiểm đạt.
- Chấm lần 1 lỗi `KeyError: 'strict_target'` (thiếu crosswalk 60→16) — đã sửa `eval_mtl_all.py`, `eval_dense_all.py`; log ở `logs/vast-dot17/*.fail1`.
- 30 run Dense + PhoBERT MTL RNN chấm ở đây cho số **phồng** (train ở môi trường khác): không dùng. File của PhoBERT giữ ở `*.eval_mtl_all-loi.json`; số test cũ 0,5992 đã khôi phục.
- BamiBERT MTL RNN (loss cũ) bị tắt giữa chừng ở ~epoch 12 — không có kết quả. Làm lại ở Đợt 20.
