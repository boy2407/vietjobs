# Đợt 16: ViSoBERT encode + train Dense last4, và chấm điểm Dense last4 (CafeBERT, ViSoBERT) — CHỜ THUÊ MÁY

Task: Hoàn thiện toàn bộ các metric còn thiếu ở `Dashboard_last4` và `Score_last4`.
Script: `scripts/vast_run_dot16.sh`
Instance đề xuất: 1x RTX 4090 / RTX 5060 Ti / RTX 3090 / A4000 (16GB+ VRAM, đĩa >= 50 GB), giá ~$0.20 - $0.35/giờ.

## Mục tiêu
1. **ViSoBERT**:
   - Nhúng `train` (34.354 tin) pooled 4 lớp cuối (`--layers 4 --max-len 512`)
   - Train 5 seed Dense (s42..s46) `dl-cat-visobert-dense-last4-cweff999-ls05`
   - Chấm điểm 5 seed trên tập `test` (9.541 tin) và `ext37k-sal` (6.111 tin)
2. **CafeBERT**:
   - Chấm điểm 5 seed Dense `dl-cat-cafe-last4-cweff999-ls05` trên `test` (9.541 tin) và `ext37k-sal` (6.111 tin)

## Ước tính thời gian & Chi phí
| Tác vụ | Thời gian trên CUDA |
|---|---|
| Nhúng ViSoBERT train raw 512 last4 | ~3 phút (~180 tin/s) |
| Train 5 seed ViSoBERT Dense | ~30 giây |
| Chấm CafeBERT Dense 5 seed (Test + Ext37k) | ~3–4 phút (~80 tin/s) |
| Chấm ViSoBERT Dense 5 seed (Test + Ext37k) | ~1.5 phút (~180 tin/s) |
| **Tổng thời gian** | **~10–12 phút** |
| **Chi phí ước tính** | **~$0.05 – $0.08** |

## Quy trình chạy
1. Người dùng thuê máy trên vast.ai, cung cấp lệnh SSH (ví dụ: `ssh -p PORT root@IP`).
2. Đồng bộ mã nguồn và dữ liệu lên máy:
   ```bash
   rsync -avz -e "ssh -p PORT" --exclude '.venv*' --exclude 'data/raw' src scripts tests data artifacts root@IP:/workspace/vietjobs/
   ```
3. Khởi chạy trong tmux:
   ```bash
   ssh -p PORT root@IP "cd /workspace/vietjobs && tmux new -d -s dot16 'bash scripts/vast_run_dot16.sh'"
   ```
4. Theo dõi log:
   ```bash
   ssh -p PORT root@IP "tail -f /workspace/vietjobs/logs/dot16-all.txt"
   ```
5. Khi hoàn tất (khoảng 10-12 phút), kéo kết quả về Mac:
   ```bash
   scp -P PORT -r root@IP:/workspace/vietjobs/artifacts/dl-cat-visobert-dense-last4-* artifacts/
   scp -P PORT -r root@IP:/workspace/vietjobs/artifacts/dl-cat-cafe-last4-*/t12_score artifacts/dl-cat-cafe-last4-*/
   scp -P PORT root@IP:/workspace/vietjobs/artifacts/embeddings/visobert-train-raw-len512-last4.* artifacts/embeddings/
   ```
6. Tự động cập nhật số vào Google Sheets `Dashboard_last4` và `Score_last4`.
7. Kiểm tra `best.pt`, destroy instance trên vast.ai.
