# Chạy trên vast.ai — theo dõi

Mỗi đợt thuê máy là **một file** trong thư mục này (AGENTS.md, Rule 14).
Script chạy nằm ở `scripts/vast_run_*.sh`; kết quả gốc nằm ở `artifacts/<run_id>/`.

| Đợt | File | Task | Script | Trạng thái |
|---|---|---|---|---|
| 1 | [dot-1-phobert-rnn.md](dot-1-phobert-rnn.md) | T8.14/T8.15 | `scripts/vast_run_rnn.sh` | xong |
| 2 | [dot-2-cafebert-rnn.md](dot-2-cafebert-rnn.md) | T8.16/T8.17 | `scripts/vast_run_cafe_rnn.sh` | xong |
| 3 | [dot-3-cafebert-mtl-rnn.md](dot-3-cafebert-mtl-rnn.md) | T4.1 | `scripts/vast_run_cafe_mtl.sh` | xong, chờ destroy |
| 4 | [dot-4-t11-encode-dense.md](dot-4-t11-encode-dense.md) | T11 encode + Dense, RNN, MTL (ViSoBERT/XLM-R 512, BamiBERT 1024) | `scripts/vast_run_t11_encode.sh` | xong, chờ destroy |
| 5 | [dot-5-t12-score-test.md](dot-5-t12-score-test.md) | T12.1b/c/d | `scripts/vast_run_t12_score.sh` (chưa viết) | XONG, CHỜ DESTROY |
| 6 | [dot-6-v8-salary.md](dot-6-v8-salary.md) | T15 | `scripts/vast_run_v8_salary.sh` | ĐANG CHẠY (đang đẩy dữ liệu) |
| 7 | [dot-7-cafebert-last4-encode.md](dot-7-cafebert-last4-encode.md) | chưa có ID — CafeBERT 512 trung bình 4 lớp cuối | `scripts/vast_run_cafe_last4.sh` | KHÔNG THUÊ — chuyển sang Colab |
| 8 | [dot-8-r2-xlmr-visobert.md](dot-8-r2-xlmr-visobert.md) | T16 — train lại XLM-R 512 + ViSoBERT RNN (`-r2`) | `scripts/vast_run_r2.sh xlmr visobert` | ĐANG CHẠY |
| 9 | [dot-9-r2-bamibert.md](dot-9-r2-bamibert.md) | T16 — train lại BamiBERT 1024 (`-r2`) | `scripts/vast_run_r2.sh bamibert` | ĐANG CHẠY |
| 10 | [dot-10-cafebert-rnn-cwls.md](dot-10-cafebert-rnn-cwls.md) | chưa có ID — CafeBERT 512 RNN, cw effective 0,999 + ls 0,05 | `scripts/vast_run_cafe_rnn_cwls.sh` | CHƯA THUÊ |
| 11 | [dot-11-last4-rnn-bamibert.md](dot-11-last4-rnn-bamibert.md) | chưa có ID — BamiBERT RNN last4 + cw + ls, cache pooled last4 | `scripts/vast_run_last4_rnn.sh bamibert 1024 32` | XONG, kết quả đã ở Mac (chờ destroy) |
| 12 | [dot-12-last4-rnn-xlmr-visobert.md](dot-12-last4-rnn-xlmr-visobert.md) | chưa có ID — XLM-R RNN last4 + cw + ls (ViSoBERT chuyển sang dot-14) | `scripts/vast_run_last4_rnn.sh` | ĐANG CHẠY |
| 13 | [dot-13-last4-rnn-cafebert-phobert.md](dot-13-last4-rnn-cafebert-phobert.md) | chưa có ID — CafeBERT + PhoBERT RNN last4 + cw + ls | `scripts/vast_run_last4_rnn.sh` | ĐANG CHẠY |
| 14 | [dot-14-last4-rnn-visobert.md](dot-14-last4-rnn-visobert.md) | chưa có ID — ViSoBERT RNN last4 + cw + ls, cache pooled last4 | `scripts/vast_run_last4_rnn.sh visobert 512 32` | ĐANG CHẠY |
| 15 | [dot-15-bert-last4-dense.md](dot-15-bert-last4-dense.md) | Toàn bộ BERT last4 encode + Dense (Category & Salary với Improved Loss) | `scripts/vast_run_dot5.sh` | đã xong |
| 16 | [dot-16-last4-dense-score.md](dot-16-last4-dense-score.md) | Hoàn thiện metric: ViSoBERT encode+train Dense; chấm Test/Ext37k Dense (CafeBERT, ViSoBERT) | `scripts/vast_run_dot16.sh` | CHỜ THUÊ |
| 17 | [dot-17-mtl-rnn-score-dense.md](dot-17-mtl-rnn-score-dense.md) | Huấn luyện 4 MTL RNN còn lại + Chấm điểm toàn diện Test & Ext37K (MTL RNN, MTL Dense, Salary Dense) | `scripts/vast_run_dot17.sh` | CHƯA THUÊ |

Kinh nghiệm chung cho mọi đợt: [kinh-nghiem.md](kinh-nghiem.md).

## Lệnh dùng chung (thay PORT và IP)

```bash
ssh -p PORT root@IP
# kéo kết quả về Mac — rsync của macOS bị treo, dùng tar qua ssh, giữ cả best.pt
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/<run_glob> logs" | tar xzf - -C .
```

## Checklist trước khi destroy instance

1. Đủ mọi run của đợt có `metrics.json` + `history.jsonl` + log trên Mac.
2. `best.pt` của mọi run đã nằm trên Mac (Đợt 2 mất 5 file vì bỏ qua bước này) và `torch.load` mở được từng file (Đợt 5 gặp một `best.pt` bị cắt khi kéo về).
3. Bảng của file đợt đã điền số từ `metrics.json`.
