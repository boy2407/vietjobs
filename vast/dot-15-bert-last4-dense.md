# Đợt 15: Toàn bộ họ BERT (PhoBERT, CafeBERT, ViSoBERT, XLM-R, BamiBERT) — Encode last4 + Huấn luyện Dense Head (Category & Salary với Improved Loss)

Script: `scripts/vast_run_dot5.sh` (đối số `all` | `encode` | `salary` | `category` | `<model>`) | Instance: **RTX 3090 Ti 24 GB, 36 vCPU, RAM 125 GB, Disk 200 GB NVMe**, SSH `ssh -p 44442 root@171.235.173.155 -L 8080:localhost:8080`, chạy trong tmux `dot5` từ 2026-10-20.

Phạm vi:
1. **Biểu diễn last4 (MEAN pooling qua 4 tầng cuối của Transformer)** cho 5 mô hình BERT:
   - `phobert` (VinAI PhoBERT base v2, max_len=256)
   - `cafebert` (UIT CafeBERT, max_len=512)
   - `visobert` (UIT ViSoBERT, max_len=512)
   - `xlmr` (Facebook XLM-RoBERTa base, max_len=512)
   - `bamibert` (Qualcomm BamiBERT, max_len=1024)
2. **Huấn luyện bài toán Lương (`salary_mid`) với hàm mất mát cải tiến**:
   - `SmoothL1Loss(beta=0.1)` (thay thế mặc định `beta=1.0` để chuyển phạt $L_1$ tuyến tính ở các mẫu lệch $>10\%$)
   - Trọng số mẫu `compute_salary_bin_weights` (Cui et al. 2019 với $\beta_{\text{eff}}=0.999$ trên 7 phân vị lương)
   - 5 seeds: 42, 43, 44, 45, 46.
   - Ablation seed 42 không có Class Weight.
3. **Huấn luyện bài toán Phân loại ngành (`category`) với biểu diễn last4**:
   - Dense Head (MEAN pooling 4 tầng cuối)
   - 5 seeds: 42, 43, 44, 45, 46.
   - Ablation seed 42 có Class Weight.

### Bảng kết quả (cập nhật từ `artifacts/<run_id>/metrics.json`)

#### 1. Bài toán Hồi quy Lương (`salary_mid`)

| Model | Run ID (s42) | MAE (tr) | RMSE (tr) | $R^2_{\log}$ | $R^2_{\text{raw}}$ | $\pm 20\%$ | Ghi chú |
|---|---|---|---|---|---|---|---|
| PhoBERT base v2 (256) | `dl-sal-pho-dense-last4-b01-cw-s42` | 4.29 | 8.19 | 0.555 | 0.432 | 49.2% | baseline last4+cw |
| PhoBERT base v2 (256) | `dl-sal-pho-dense-last4-b01-s42` | 4.23 | 8.78 | 0.563 | 0.347 | 50.5% | ablation no-cw |
| CafeBERT (512) | `dl-sal-cafe-dense-last4-b01-cw-s42` | ... | ... | ... | ... | ... | |
| CafeBERT (512) | `dl-sal-cafe-dense-last4-b01-s42` | ... | ... | ... | ... | ... | ablation no-cw |
| ViSoBERT (512) | `dl-sal-visobert-dense-last4-b01-cw-s42` | ... | ... | ... | ... | ... | |
| ViSoBERT (512) | `dl-sal-visobert-dense-last4-b01-s42` | ... | ... | ... | ... | ... | ablation no-cw |
| XLM-R base (512) | `dl-sal-xlmr-dense-last4-b01-cw-s42` | ... | ... | ... | ... | ... | |
| XLM-R base (512) | `dl-sal-xlmr-dense-last4-b01-s42` | ... | ... | ... | ... | ... | ablation no-cw |
| BamiBERT (1024) | `dl-sal-bamibert-dense-last4-b01-cw-s42` | ... | ... | ... | ... | ... | |
| BamiBERT (1024) | `dl-sal-bamibert-dense-last4-b01-s42` | ... | ... | ... | ... | ... | ablation no-cw |

#### 2. Bài toán Phân loại Ngành nghề (`category`)

| Model | Run ID (s42) | macro-F1 | weighted-F1 | Accuracy | Ghi chú |
|---|---|---|---|---|---|
| PhoBERT base v2 (256) | `dl-cat-pho-dense-last4-s42` | ... | ... | ... | |
| PhoBERT base v2 (256) | `dl-cat-pho-dense-last4-cw-s42` | ... | ... | ... | ablation with cw |
| CafeBERT (512) | `dl-cat-cafe-dense-last4-s42` | ... | ... | ... | |
| CafeBERT (512) | `dl-cat-cafe-dense-last4-cw-s42` | ... | ... | ... | ablation with cw |
| ViSoBERT (512) | `dl-cat-visobert-dense-last4-s42` | ... | ... | ... | |
| ViSoBERT (512) | `dl-cat-visobert-dense-last4-cw-s42` | ... | ... | ... | ablation with cw |
| XLM-R base (512) | `dl-cat-xlmr-dense-last4-s42` | ... | ... | ... | |
| XLM-R base (512) | `dl-cat-xlmr-dense-last4-cw-s42` | ... | ... | ... | ablation with cw |
| BamiBERT (1024) | `dl-cat-bamibert-dense-last4-s42` | ... | ... | ... | |
| BamiBERT (1024) | `dl-cat-bamibert-dense-last4-cw-s42` | ... | ... | ... | ablation with cw |

---

### Tiến độ Đợt 5

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy vast.ai | done | RTX 3090 Ti 24 GB, 125 GB RAM, 200 GB NVMe |
| 2 | SSH Direct + Cài đặt môi trường `/venv/main` | done | Python 3.12, PyTorch 2.14.1+cu132, transformers 4.46.3, underthesea, pyvi |
| 3 | Đồng bộ code, splits (`train.csv`, `dev.csv`), `manifest.json`, cache có sẵn | done | splits và manifest kiểm soát qua Rule 5; test split không động chạm |
| 4 | Kiểm thử pytest đơn vị | done | 22 passed, 14 skipped |
| 5 | Soạn `scripts/vast_run_dot5.sh` và khởi động trong tmux `dot5` | in progress | |
| 6 | Kéo artifacts (`metrics.json`, `best.pt`, `predictions_dev.parquet`) về Mac | todo | |
| 7 | Cập nhật bảng kết quả, đối chiếu và ghi nhận báo cáo | todo | |

### Lệnh đồng bộ & chạy (tham khảo)

```bash
# Đẩy script lên Vast:
rsync -avz -e "ssh -p 44442" scripts/vast_run_dot5.sh root@171.235.173.155:/workspace/vietjobs/scripts/

# Kéo toàn bộ artifacts và logs về Mac khi chạy xong:
ssh -p 44442 root@171.235.173.155 "cd /workspace/vietjobs && tar czf - artifacts/dl-* logs/dl-* logs/dot5*" | tar xzf - -C .
```
