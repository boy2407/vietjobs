# T17 — Đợt 17: 4 MTL RNN còn lại + chấm Test/Ext37K (Vast.ai)

Goal: train 4 MTL RNN (CafeBERT, ViSoBERT, XLM-R, BamiBERT), chấm 35 run trên `test` (9.541) và `ext37k-sal` (6.111), ghi `Dashboard_last4` / `Score_last4`. Chi tiết máy: [vast/dot-17-mtl-rnn-score-dense.md](../vast/dot-17-mtl-rnn-score-dense.md).

- [x] Thuê 1x RTX 5090, cài môi trường, đẩy splits + ext37k + artifacts Dense
- [ ] Train 4 MTL RNN (`/workspace/run17.sh`, tmux `dot17`)
- [ ] Chấm MTL RNN + MTL Dense + Salary Dense
- [ ] Kéo `artifacts/` + `best.pt` về Mac, `torch.load` từng file
- [ ] Ghi sheet `Dashboard_last4`, `Score_last4`
- [ ] `export_runs.py`, `04-results.md`, `06-baseline-dl.md`, bao-cao
- [ ] Destroy instance sau checklist `vast/README.md`

Done when: đủ `metrics.json` + `metrics_test.json` + `external_37k/metrics.json` cho 35 run, `best.pt` mở được trên Mac, sheet khớp `metrics*.json`.

## Plan chấm điểm → tải về → ghi sheet

Máy: 5090 `61.32.91.194:18549` (CafeBERT → BamiBERT → tự chấm), 3090-A `38.145.234.214:60880` (ViSoBERT), 3090-B `178.61.243.26:21372` (XLM-R; `/workspace` chỉ 10 GB, cache ở `/root/embeddings`).
Nguồn số duy nhất: `artifacts/<run_id>/metrics.json` (dev), `metrics_test.json`, `external_37k/metrics.json`. Không gõ tay số nào.

### A. Mỗi máy 3090, ngay khi train xong (để destroy sớm)
1. Kiểm tra: `logs/<run_id>.txt` có `=== XONG`, `artifacts/<run_id>/` có `best.pt`, `best_sal.pt`, `scaler.npz`, `config.json`, `metrics.json` thật (có `best_epoch`).
2. Đẩy `data/external/vietjobs37k/` (110 MB) lên máy, chạy `scripts/eval_mtl_all.py` — script tự bỏ qua mô hình không có `best.pt`, nên chỉ chấm mô hình của máy đó (test 9.541 + ext37k 6.111, encode lại tại chỗ, ~5 phút).
3. Kéo về Mac: `ssh … "cd /workspace/vietjobs && tar czf - artifacts/<run_id> logs/<run_id>*.txt logs/eval_mtl_all.txt" | tar xzf - -C .` (đổi tên `logs/eval_mtl_all.txt` → `logs/eval_mtl_all-<máy>.txt` để không đè).
4. Trên Mac: `.venv-dl/bin/python -c "torch.load(...)"` cho `best.pt`, `best_sal.pt`; so `metrics.json` có cùng `best_epoch` với log.
5. Báo tác giả destroy máy đó; cập nhật `vast/` (dot-18 cho 3090-A, dot-19 cho 3090-B).

### B. Máy 5090
1. Trước khi `run17.sh` tới giai đoạn chấm: **xóa 2 thư mục giả** `artifacts/dl-mtl-{vis,xlm}-rnn-b01-cw-s42/` (chỉ chứa `metrics.json` placeholder) — kiểm tra nội dung là placeholder trước khi xóa.
2. `run17.sh` tự chạy `eval_mtl_all.py` (PhoBERT, CafeBERT, BamiBERT; vis/xlm bị bỏ qua vì không có `best.pt`) rồi `eval_dense_all.py` (5 MTL Dense + 25 Salary Dense, tự encode cache last4 cho test/ext37k).
3. Kiểm tra đủ: 3 MTL RNN + 5 MTL Dense + 25 Salary Dense đều có `metrics_test.json` và `external_37k/metrics.json`; log không có `Traceback` (script cũ thiếu `pipefail`, nên phải đọc log, không tin dòng `XONG`).
4. Kéo về: `artifacts/dl-mtl-{pho,caf,bam}-rnn-b01-cw-s42`, `artifacts/dl-mtl-*-dense-last4-cw-s42`, `artifacts/dl-sal-*-dense-last4-b01-cw-s4[2-6]`, `artifacts/eval_*_summary.json`, `logs/` — **không dùng glob `dl-mtl-*-rnn-*`** để tránh kéo nhầm thư mục giả nếu bước B1 bị bỏ sót. Không kéo `artifacts/embeddings`.
5. `torch.load` mọi `best.pt` vừa kéo; rồi mới báo destroy.

### C. Ghi sheet (spreadsheet `14yGfv2RII0q69t-WEvcGtqT2J5guujwsb6Llm7vNuQ8`, locale `vi_VN`)
1. Script tạm (scratchpad, không vào repo) đọc JSON → 15 dòng mỗi sheet:
   - 5 MTL RNN (`mtl`, "<Enc> + BiGRU-LSTM-CNN (MTL)", 1 run), 5 MTL Dense (`mtl`, "<Enc> + Dense (MTL, last4)", 1 run), 5 Salary Dense (`salary`, "<Enc> + Dense (last4)", 5 seed → `mean ± sd`, F1/Acc để trống).
   - `Score_last4` A–O: test F1, Acc, MAE, RMSE, R² log; ext37k F1/Acc (strict), MAE, RMSE, R² log; ghi chú "Vast GPU · test-informed (Rule 5 mục 6) · Spearman ext <x>".
   - `Dashboard_last4` A–L: số `dev` từ `metrics.json` (F1, Acc, RMSE, MAE, R² log), cột Loss/Cấu hình = `cw + α=5 + β=0.1` (MTL) hoặc `last4 + cw + β=0.1` (Salary), ghi chú best epoch + máy.
   - Định dạng như dòng cũ: 1 run → số thô (4 chữ số cho F1/Acc/R², 3 cho MAE/RMSE triệu); nhiều seed → chuỗi `0,6159 ± 0,0040`.
2. **Cho tác giả xem bảng trước khi ghi.** Sau khi đồng ý: `gws sheets spreadsheets values append` vào `Score_last4!A:O` và `Dashboard_last4!A:L`, `valueInputOption=RAW`, `insertDataOption=INSERT_ROWS` — chỉ thêm dòng, không sửa dòng cũ.
3. Đọc lại 2 sheet bằng `values get`, so từng ô với JSON.

### D. Tài liệu & đóng việc (trên Mac)
`export_runs.py` → `04-results.md` (append-only) → `06-baseline-dl.md`, `00-tong-quan.md`, `bao-cao/05-chuong-4` → bảng trong `vast/dot-17*.md` (+ dot-18/19) → `vast/kinh-nghiem.md` (kiểm `df -h / /workspace`; `pipefail`; train bị chặn ở CPU nên 3090 ≈ 5090) → `pytest -q` → commit nhỏ.

### Cần tác giả quyết
- PhoBERT MTL RNN đã có `metrics_test.json`; `eval_mtl_all.py` sẽ chấm lại `test` lần nữa (và tạo `external_37k/metrics.json` đang thiếu trên Mac). Chấm lại (deterministic, số phải trùng 0,5992) hay tạm dời `best.pt` của nó để bỏ qua?
  → 2026-10-21: **chấm lại** (rẻ, ~2–3 phút; tạo nguồn cho số Ext37K đang thiếu; kiểm chứng đường chấm). Nếu test F1 khác 0,5992 thì dừng, không ghi sheet.

## Log
- 2026-10-21: thuê 5090 (`61.32.91.194:18549`), bắt đầu train 07:17 UTC. Cache token RNN ~120 GB > đĩa 100 GB nên script máy thuê xóa cache sau mỗi mô hình.
- 2026-10-21 08:50 UTC: tác giả quyết định hôm nay chỉ CafeBERT + ViSoBERT; BamiBERT + XLM-R hoãn sang hôm sau. Hủy hàng đợi XLM-R trên 3090-A, đặt placeholder `bam` trên 5090. Máy 3090-B (Đợt 19) đã destroy.
- 2026-10-21 ~10:20 UTC: tác giả tắt cả hai máy. Kết luận đợt: chỉ CafeBERT, ViSoBERT MTL RNN có test/ext hợp lệ; PhoBERT có test cũ (0,5992) chưa có ext; Dense chấm lệch môi trường → không dùng.
- 2026-10-21: tác giả yêu cầu mọi MTL dùng loss mới (last4 + cw 0,999 + ls 0,05; lương last4 + cw + β 0,1). Đã thêm cờ vào `train_mtl.py`, RNN đọc cache last4, `eval_mtl_all.py` chấm last4; thêm `scripts/check_mtl_dev.py` (cổng chặn dev) và `scripts/vast_run_dot20.sh`. Kế hoạch: `vast/dot-20-mtl-v2-train-cham.md`. pytest 216 passed.
- 2026-10-21 11:43 UTC: Dense `-v2` (5 MTL + 25 Salary) train+chấm cùng máy 3090 xong, kéo về Mac 30/30 đạt, đã ghi 2 sheet last4 và cập nhật docs/04, 06 §5.9, 00, bao-cao 4.4.5. Còn: CafeBERT + PhoBERT MTL RNN `-v2` trên máy 4090.
- 2026-10-21 13:05 UTC: CafeBERT MTL RNN `-v2` (loss gộp mới, 4090) xong: dừng sớm epoch 42; dev F1 0,6322 · MAE 3,700 (cổng khớp); test F1 0,6278; ext37k F1 0,6090. Kéo về Mac 1/1 đạt; đã ghi 2 sheet last4 (Dashboard dòng 29, Score dòng 26) và docs. PhoBERT `-v2` đang chạy.
- 2026-10-21 13:50 UTC: máy 4090 bị tắt đúng lúc PhoBERT MTL RNN `-v2` train xong (epoch 41) — chưa kịp kéo `best.pt`; mất toàn bộ run, Mac không có file nào của nó. Quyết định của tác giả: train lại PhoBERT sau. Còn lại: `dl-mtl-pho-rnn-b01-cw-s42-v2`, BamiBERT và XLM-R MTL RNN `-v2` (`bash scripts/vast_run_dot20.sh rnn:<enc>`).
- 2026-10-21 15:57 UTC: PhoBERT MTL RNN `-v2` trained and scored on a 3090 Ti (round 21): dev F1 0.6261 / MAE 3.653 (gate matched), test F1 0.6242, ext37k F1 0.5997; pulled to the Mac and verified. Docs updated; sheet rows NOT written yet (`gws` token expired - re-run `gws auth login`). XLM-R `-v2` running on the same machine; BamiBERT not started.
- 2026-10-21 ~16:42 UTC: author stopped the 3090 Ti instance while XLM-R MTL RNN `-v2` was at epoch 24 (best dev F1 0.6220 at epoch 13, MAE 3.810 at epoch 24, still improving) - no checkpoint was written, run lost; partial history kept in `logs/vast-dot21-xlm-partial/`. Remaining MTL RNN `-v2` runs: XLM-R, BamiBERT. Sheet rows for PhoBERT `-v2` still pending (`gws auth login`).
