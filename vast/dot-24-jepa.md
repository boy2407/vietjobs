# Đợt 24: T19 JEPA — encoder nhỏ tự học, rồi MTL RNN — CHƯA THUÊ

Task: T19 (chờ tác giả duyệt trên `TASKS.md`). Script: `scripts/vast_run_jepa.sh smoke|full`. Nhánh `t19-jepa`.

## Vì sao có đợt này

- Đo xem tiền huấn luyện JEPA có giúp hơn không tiền huấn luyện, với cùng encoder và cùng head.
- Đầu vào là bảng embedding từ vựng CafeBERT (đóng băng, chưa qua attention), không phải token cache: cache đã đọc cả câu nên token bị che rò qua token lân cận.
- Code chưa chạy trên dữ liệu thật lần nào: bước `smoke` là lần chạy đầu.

## Run

| Run | Cấu hình | Trạng thái | dev F1 | dev MAE |
|---|---|---|---|---|
| `jepa-pre-d256-s42` | pretrain, d 256, 4 tầng, 4 head, predictor 128×2, len 512, 30 epoch, lr 5e-4, τ 0,996→1 | todo | — | — |
| `dl-mtl-jepa-rnn-s42` | encoder trên (đóng băng) + MTL RNN hidden 256, conv 64 | todo | | |
| `dl-mtl-jeparand-rnn-s42` | cùng encoder, khởi tạo ngẫu nhiên (đối chứng) | todo | | |

Chung cho hai finetune: cw effective 0,999 + ls 0,05 + Huber β 0,1, α 5, patience 12, epochs 80, `--device cuda`.
Mốc tham chiếu (không kỳ vọng vượt): `dl-mtl-caf-rnn-b01-cw-s42-v2` F1 0,6322 · MAE 3,70 tr.

## Instance

Tiêu chí (ước, chưa đo): GPU ≥ 16 GB, RAM ≥ 32 GB, đĩa ≥ 40 GB. Không có token cache lớn: token id + mask train+dev ở 512 khoảng 100 MB, bảng embedding float16 khoảng 0,5 GB, mô hình CafeBERT tải về khoảng 2,2 GB.
Thời gian: chưa đo.

## Các bước

| # | Bước | Trạng thái |
|---|---|---|
| 1 | Thuê; `df -h / /workspace`, `free -g`, `nvidia-smi` | todo |
| 2 | `git clone -b t19-jepa` (cần push nhánh trước, hoặc copy repo), `pip install -r requirements-dl.txt` + `pyarrow`; đẩy `data/processed/splits`, `manifest.json` | todo |
| 3 | `bash scripts/vast_run_jepa.sh smoke` — loss giảm, `collapse_std` không về 0, `metrics.json` có đủ khoá | todo |
| 4 | tmux: `bash scripts/vast_run_jepa.sh full` | todo |
| 5 | Kéo `artifacts/jepa-pre-d256-s42`, `artifacts/dl-mtl-jepa*-rnn-s42`, `logs/`; `git diff docs/04-results.md` | todo |
| 6 | Destroy sau checklist `vast/README.md` | todo |

## Lệnh

```bash
ssh -p PORT root@IP
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/jepa-pre-d256-s42 artifacts/dl-mtl-jepa*-rnn-s42 logs" | tar xzf - -C .
```

## Nhật ký

- (chưa thuê)
