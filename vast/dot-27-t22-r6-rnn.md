# Đợt 27 — T22 R6-RNN: BiGRU-LSTM-CNN lương, một text có thêm 6 trường, 5 encoder

**Task:** T22 (R6-RNN). **Script:** `scripts/vast_run_t22_r6rnn.sh`.
**Trạng thái:** ĐÃ DỪNG, R6 bị bỏ (2026-11-24, tác giả). Chờ destroy.

**Máy:** RTX 5060 Ti 16 GB (sm_120), 112 lõi, cgroup RAM 183 GB, ổ 120 GB, Germany. `ssh -p 35187 root@85.10.218.46 -L 8080:localhost:8080`.

**Thiết kế (chốt với tác giả 2026-11-24).** Câu hỏi: với head BiGRU-LSTM-CNN, thêm 6 trường
(cùng `splits-r6text` của Đợt 26) có giảm MAE lương không. Cấu hình giống
`dl-sal-<enc>-rnn-last4-b01-cw-s42-v2` (`scripts/vast_run_sal_rnn.sh`): `--head rnn --last4`,
hidden 100, conv 50, 40 epoch, `patience` 8, SmoothL1 β = 0,1 + trọng số khoảng lương, `cuda`,
**1 seed (42)**. PhoBERT 256, BamiBERT 1024, còn lại 512 token.

**Quy tắc quyết định (chốt trước khi chạy):** so `dev` MAE với RNN cũ cùng encoder.
Một seed nên không tách được nhiễu giữa seed (Dense: độ lệch 0,02–0,05). "Tốt hơn" khi giảm
**≥ 0,05** triệu; giảm dưới 0,05 là chưa kết luận. `test` chỉ để báo cáo, tác giả tự chạy lệnh chấm;
không chấm `ext37k` (thiếu 6 trường).

**Máy:** m:142606, 1× RTX 5060 Ti 16 GB (Blackwell sm_120), EPYC 7663, RAM 64 GB, Germany,
$0,212/giờ. **torch 2.2.2 cu121 không chạy trên sm_120** → dùng torch của image `vastai/pytorch`
(≥ 2.7 cu128), ghi phiên bản vào bảng bên dưới. Ổ ≥ 150 GB.

| Bước | Lệnh / việc | Trạng thái |
|---|---|---|
| 1 | Kiểm máy: `nvidia-smi`, `nproc`, `free -g`, `df -h / /workspace`, `python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"` | xong: torch 2.14.1+cu132 có sm_120 |
| 2 | venv `/root/venv` kế thừa torch image (`--system-site-packages`) + numpy, pandas 2.3.3, sklearn 1.6.1, transformers 4.46.3, `pyarrow` khớp numpy, sentencepiece, protobuf, scipy, underthesea 9.5.0 | xong: cài vào `/venv/main` (venv của image), ghim đúng bản của RNN cũ (`env.json`): numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, transformers 4.46.3 |
| 3 | Đẩy `src/`, `scripts/`, `data/processed/manifest.json`, `splits/{train,dev}.csv`, `splits-r6text/{train,dev}.csv`, `docs/04-results.md` | xong. Đường lên chỉ ~60 KB/s → `splits-r6text` dựng ngay trên máy bằng `build_r6text_split.py`; md5 trùng với bản trên Mac |
| 4 | `bash scripts/vast_run_t22_r6rnn.sh phobert cafebert xlmr visobert bamibert` trong tmux (phiên `rnn`, log `/workspace/rnn.log`), bắt đầu 02:51 UTC | dừng ~07:50 UTC khi đang chạy BamiBERT (tác giả bỏ R6); 4 encoder đã xong, không kéo về |
| 5 | Kéo `artifacts/dl-sal-t22-r6rnn-*` + dòng mới của `04-results.md` + `logs/` về Mac, so md5 | chưa |
| 6 | Chấm `test` (tác giả chạy): đẩy `splits/test.csv`, rồi với mỗi run `VIETJOBS_SPLITS=splits-r6text python scripts/eval_sal_rnn.py dl-sal-t22-r6rnn-<enc>-s42` | chưa |
| 7 | Checklist `vast/README.md`, tác giả destroy | chưa |

| run_id | Dev MAE | RNN cũ Dev MAE | Test MAE | RNN cũ Test MAE |
|---|---|---|---|---|
| `dl-sal-t22-r6rnn-phobert-s42` | — | 4,047 | — | 4,198 |
| `dl-sal-t22-r6rnn-cafebert-s42` | — | 3,840 | — | 4,112 |
| `dl-sal-t22-r6rnn-xlmr-s42` | — | 3,984 | — | 4,174 |
| `dl-sal-t22-r6rnn-visobert-s42` | — | 3,972 | — | 4,207 |
| `dl-sal-t22-r6rnn-bamibert-s42` | — | 4,075 | — | 4,204 |

RNN cũ: `artifacts/dl-sal-<enc>-rnn-last4-b01-cw-s42-v2/{metrics.json,metrics_test.json}` (test n = 6.773 tin có lương, cùng tập con với R6).
