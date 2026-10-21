# T18 — MTL RNN `-v2` còn lại (XLM-R, BamiBERT), checkpoint an toàn, sheet và tài liệu

Tiếp nối [T17](T17-dot17-mtl-rnn-cham-diem.md). Mục tiêu: có đủ 5 mô hình MTL RNN với hàm mất mát gộp mới (`last4 + cw effective 0,999 + ls 0,05`, lương `cw + β 0,1`, α 5) — mỗi run **huấn luyện và chấm trên cùng một máy**, rồi ghi `Dashboard_last4` / `Score_last4`.

## 1. Hiện trạng (2026-10-21)

| Run `-v2` | Trạng thái | Dev F1 / MAE (triệu) | Test F1 | Ext37K F1 | Về Mac | Lên sheet |
|---|---|---|---|---|---|---|
| CafeBERT 512 | xong (4090) | 0,6322 / 3,700 | 0,6278 | 0,6090 | ✅ | ✅ |
| PhoBERT 256 | xong (3090 Ti) | 0,6261 / 3,653 | 0,6242 | 0,5997 | ✅ | ⏳ chờ `gws auth login` |
| XLM-R 512 | **bị dừng ở epoch 24**, không có checkpoint | (tạm: F1 0,6220 ep 13, MAE 3,810 ep 24, còn cải thiện) | — | — | ✗ | ✗ |
| BamiBERT 1024 | chưa chạy | — | — | — | ✗ | ✗ |
| ViSoBERT 512 | chỉ có bản loss cũ | — | — | — | — | — (chưa thuộc phạm vi) |

Cũng đã xong: 5 MTL Dense `-v2` và 25 Salary Dense `-v2` (đã lên sheet, tài liệu).

## 2. Việc cần bạn làm (mình không làm thay được)

1. `gws auth login` trong terminal (token Google hết hạn, `invalid_grant`) — cần để ghi sheet.
2. Thuê máy và gửi `ssh -p PORT root@IP`; thêm khoá ssh trước khi thuê.
3. Quyết định ngân sách (xem §6) và có cho phép `git push` thêm lần sau hay không.

## 3. Code trước khi thuê (làm trên Mac, không tốn tiền) — ĐÃ XONG 2026-10-21

Lý do: `train_mtl.py` chỉ ghi `best.pt`/`best_sal.pt` **khi train kết thúc** → máy tắt giữa chừng là mất sạch (đã xảy ra với PhoBERT ở máy 4090 và XLM-R ở máy 3090 Ti; ~3 giờ và ~1 giờ GPU).

| Bước | Việc | Kiểm chứng |
|---|---|---|
| 3.1 ✅ | Ghi `best.pt` / `best_sal.pt` **ngay mỗi khi có mức mới** (ghi vào file tạm rồi đổi tên, tránh file hỏng khi tắt máy) | test: sau mỗi epoch cải thiện thì file có mặt và `torch.load` mở được |
| 3.2 ✅ | Thêm `last.pt` (trọng số + optimizer + epoch + bộ đếm patience + `best_*`) và cờ `--resume` đọc lại | test: dừng giữa chừng rồi chạy tiếp cho cùng `history.jsonl` như chạy liền |
| 3.3 ✅ | Không đổi hành vi khi không dùng `--resume` | `pytest -q` xanh; `check_mtl_dev.py` vẫn khớp trên một run nhỏ |

Không cần dòng ablation trong `04-results.md`: đây là thay đổi lưu trữ, không đổi số học (Rule 2 mục 4 chỉ áp cho bước tiền xử lý). Cập nhật `08-ma-nguon.md` nếu đổi tệp (Rule 1).

Kết quả kiểm chứng (đầu Dense, PhoBERT last4, `cpu`, seed 42, `--no-log`): chạy liền 6 epoch so với "mất điện" sau epoch 3 (ngắt bằng lỗi giả ngay sau khi ghi `last.pt`) rồi `--resume` đến epoch 6 — `history.jsonl` trùng ở cả 6 epoch (sai khác < 1e-9) và `metrics.json` cuối trùng (F1, MAE, epoch tốt nhất). Lần đầu epoch 4–6 lệch vì bộ sinh số ngẫu nhiên toàn cục (dropout) chưa được lưu; đã thêm `rng_cpu`/`rng_np`/`rng_cuda` vào `last.pt`. Chưa kiểm trên `cuda` (RNN, máy thuê) — kiểm ở lần thuê tới bằng cách so epoch sau khi chạy tiếp. `pytest -q`: 218 passed (main), 2 test mới trong `tests/test_train_mtl_checkpoint.py` (ghi nguyên tử; lỗi giữa chừng giữ bản cũ). `last.pt` nặng (trọng số + optimizer + hai bản tốt nhất, ước vài chục MB) nên bị xoá khi run kết thúc và không nằm trong danh sách file kéo về.

## 4. Thuê máy và chạy

Tiêu chí máy (đo ở Đợt 17–21): RTX 3090 / 3090 Ti / 4090 24 GB, **CPU mạnh từng lõi** (train bị chặn ở một lõi CPU: PhoBERT 60 s/epoch trên Ryzen 5950X), RAM ≥ 64 GB, **đĩa ≥ 100 GB trên ổ chứa `/workspace`**, tải CPU thấp.

| Mô hình | Lệnh | Ước tính (đo, chưa chắc cho BamiBERT) |
|---|---|---|
| XLM-R | `bash scripts/vast_run_dot20.sh rnn:xlmr` | 127 s/epoch × 30–45 epoch ≈ 1–1,6 giờ + encode/thống kê ~10 phút + chấm ~6 phút |
| BamiBERT | `bash scripts/vast_run_dot20.sh rnn:bamibert` | **chưa đo**: chuỗi 1024 token; ước 2–3 giờ, cache last4 ~60 GB nên đĩa phải ≥ 100 GB |

Mỗi máy: kiểm `df -h / /workspace`, `nvidia-smi`, torch thấy GPU → clone nhánh → đẩy `splits` + `manifest.json` + `ext37k` bằng `cd /workspace/vietjobs && tar xzf -` → chạy trong tmux. Nên chạy hai mô hình trên hai máy riêng hoặc tuần tự; chạy song song trên một máy làm mỗi epoch chậm 1,7 lần khi một bên đang encode.

## 5. An toàn dữ liệu khi chạy

- Watcher kéo về **mỗi 15 giây**, kích hoạt khi `metrics.json` xuất hiện: kéo `best.pt`, `best_sal.pt`, `scaler.npz`, `config.json`, `history.jsonl`, log, rồi `torch.load` + so checksum (`scratchpad/pull3.sh`, `verify.py`).
- Sau khi kiểm `dev` khớp và chấm xong: kéo `metrics_test.json`, `external_37k/metrics.json` + log chấm.
- Lấy các dòng `git diff docs/04-results.md` trên máy thuê **trước khi destroy** (file này không đi cùng `artifacts/`).
- Không dừng/kill tiến trình nào khi chưa hỏi tác giả; nếu cần dừng tạm thì dùng `kill -STOP`/`-CONT` và nói rõ tiền máy vẫn tính.

## 6. Thời gian và chi phí (giá đã thấy: 3090 Ti $0,228/giờ; chưa tính phí băng thông, đĩa)

| Kịch bản | Máy chạy | Tiền thuê ước tính |
|---|---|---|
| Chỉ XLM-R | ~1,5–2,2 giờ | ~$0,35–0,50 |
| Chỉ BamiBERT | ~2,5–3,5 giờ (chưa đo) | ~$0,6–0,8 |
| Cả hai tuần tự, một máy | ~4–5,7 giờ | ~$0,9–1,3 |

Tiền máy rảnh qua đêm không đáng: 8 giờ ≈ $1,8 (xem T17). Luôn kéo mô hình về Mac trước khi destroy; đừng bắt đầu một run dài khi số dư tài khoản không đủ cho cả run.

## 7. Sau khi có kết quả

1. Chấm điểm: `check_mtl_dev.py` khớp → `eval_mtl_all.py` (`ONLY`, `RUN_SUFFIX=-v2`); `dev` lệch thì **dừng**, không chấm `test` (Đợt 17).
2. Sheet (`gws sheets … values append`, chỉ **thêm dòng mới**): `Dashboard_last4` (số `dev`) và `Score_last4` (số `test`, `ext37k`, ghi chú "test-informed, Rule 5 mục 6"); đọc lại so từng ô với JSON.
3. Tài liệu: `04-results.md` (nối dòng), `06-baseline-dl.md` §5.9, `bao-cao/05-chuong-4` §4.4.5 (thay dòng `⛔`), `00-tong-quan.md`, `vast/dot-N-*.md`; `export_runs.py` — lưu ý script hiện bỏ qua dòng `dl-mtl-*` (chỉ nhận 2 dòng) nên workbook không có các run MTL mới: sửa như một task riêng nếu cần.
4. `pytest -q` xanh; commit nhỏ (≈100 dòng), tiếng Anh, không trailer AI.

## 8. Quyết định còn mở

- BamiBERT có cần cho luận văn không, hay dừng ở 4/5 encoder RNN (thiếu BamiBERT)? Sẽ tốn thêm ~$0,6–0,8 và 2,5–3,5 giờ.
- ViSoBERT MTL RNN `-v2` có train lại với loss mới không (hiện chỉ có bản loss cũ, đã loại khỏi sheet)?
- Sửa `export_runs.py` để nhận dòng MTL (task riêng)?
- Có sửa 3 commit tiếng Việt đã push (cần `git push --force`)?

## Nhật ký
- 2026-10-21: tạo task; XLM-R bị dừng ở epoch 24 (người dùng dừng máy); PhoBERT xong, sheet chờ `gws auth login`.
- 2026-10-21: xong §3 (checkpoint nguyên tử, `last.pt`, `--resume` trùng bit với chạy liền trên cpu); `gws auth login` đã xong, hai dòng PhoBERT `-v2` đã lên sheet (Dashboard dòng 30, Score dòng 27).
