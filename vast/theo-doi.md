# Theo dõi các máy vast.ai và thí nghiệm last4 + cw + ls (dành cho người/AI theo dõi, chỉ đọc)

Cập nhật lần cuối: 2026-10-19 (sau khi khởi chạy dot-12, dot-13, dot-14; máy B được dùng). Người theo dõi **chỉ đọc**: không chạy lệnh ghi, không destroy máy, không sửa file. Thấy vấn đề thì báo cho người dùng.

## 1. Dự án

Phân loại ngành nghề (`category`, 16 lớp) và ước lượng lương (`salary`) từ tin tuyển dụng tiếng Việt. Encoder BERT đóng băng (PhoBERT, CafeBERT, XLM-R, BamiBERT, ViSoBERT) + head Dense hoặc RNN (Bi-GRU ‖ Bi-LSTM → CNN). Chỉ số chính: macro-F1 trên `dev` (3.812 dòng). Repo: `/Users/admin/Documents/TrNghia/VietJob/vietjobs`, nhánh `t8-rnn-colab`; luật làm việc ở `AGENTS.md`.

## 2. Mục tiêu

Cấu hình `category`: **4 lớp ẩn cuối** (`--layers 4`) + **class weight effective** (`--cw-mode effective --cw-beta 0.999`) + **label smoothing 0.05**, chạy cho cả 5 encoder × 2 head (Dense, RNN). Mốc so: CE thường, chỉ lớp cuối. RNN chạy 1 seed (42) theo lựa chọn của tác giả; Dense 5 seed (42–46).

## 3. Ma trận tiến độ (dev, macro-F1)

✅ đã chạy và có số · ⏳ đang chạy · ❌ chưa chạy. Số lấy từ `artifacts/<run>/metrics.json`.

| Encoder | Dense last4 + cw + ls (5 seed, cpu, Mac) | RNN last4 + cw + ls (1 seed, cuda, vast) |
|---|---|---|
| CafeBERT 512 | ✅ 0,6176 ± 0,0023 (mốc CE lớp cuối 0,6037) | ❌ (mốc CE 0,6152, 3 seed) |
| PhoBERT 256 | ✅ 0,6055 ± 0,0052 (mốc 0,6032) | ❌ (mốc CE 0,6084, 4 seed) |
| BamiBERT 1024 | ❌ cache pooled last4 đã về Mac, chưa train | ✅ **0,6283** (mốc CE 0,6056, 1 seed) |
| XLM-R 512 | ❌ chưa nhúng | ❌ (mốc CE 0,6134, 1 seed) |
| ViSoBERT 512 | ❌ chưa nhúng | ❌ (mốc CE 0,6046, 1 seed) |

Mốc CE lớp cuối của Dense cho XLM-R (0,5953), BamiBERT (0,5987), ViSoBERT (0,5723) mới có 1 seed; để so cặp cần chạy thêm CE lớp cuối 5 seed (cache lớp cuối đã có trên Mac, mỗi run khoảng 15 giây).

Nguồn các số đã có: `artifacts/dl-cat-cafe-last4-cweff999-ls05-s*`, `artifacts/dl-cat-pho-last4-cweff999-ls05-s*`, `artifacts/dl-cat-bamibert-rnn-last4-cweff999-ls05-s42` (54,5 phút, đỉnh epoch 8/16, không tính lớp rác 0,6606, accuracy 0,6676).

## 4. Các máy (thư mục `/workspace/vietjobs`, GPU RTX 5060 Ti 16 GB, giá $0,25/giờ/máy)

| Máy | Lệnh SSH | Trạng thái lúc kiểm | Của ai |
|---|---|---|---|
| A | `ssh -p 45000 root@85.10.218.46` | **ĐANG CHẠY** [dot-12](dot-12-last4-rnn-xlmr-visobert.md): chỉ **XLM-R 512** (RNN last4 + cw + ls, seed 42, kèm cache pooled last4). ViSoBERT đã chuyển sang máy B; máy A có cờ `logs/SKIP_visobert` nên sẽ thoát ngay khi tới lượt ViSoBERT. tmux `last4b`, log `logs/last4b-all.txt`. Việc BamiBERT trước đó đã xong ở dot-11 | phiên này |
| B | `ssh -p 46062 root@85.10.218.46` | **ĐANG CHẠY** [dot-14](dot-14-last4-rnn-visobert.md): **ViSoBERT 512** (nhúng pooled + token last4, RNN seed 42). tmux `last4d`, log `logs/last4d-all.txt`. Phiên T16 đã xong, 3 run `-r2` còn trên đĩa và đã có trên Mac; tác giả cho phép dùng | phiên này (mượn từ T16) |
| C | `ssh -p 43372 root@158.181.52.18` | **ĐANG CHẠY** [dot-13](dot-13-last4-rnn-cafebert-phobert.md): CafeBERT 512 rồi PhoBERT 256 (RNN last4 + cw + ls, seed 42, bỏ pooled). tmux `last4c`, log `logs/last4c-all.txt` | phiên này (mượn từ T16) |

## 5. Kế hoạch: mỗi máy chạy gì (A, B, C đã khởi chạy 2026-10-19; Mac chưa)

Script chung `scripts/vast_run_last4_rnn.sh <encoder> <max_len> <batch>`: kiểm sha256 → nhúng pooled last4 → nhúng token last4 → train RNN seed 42 → xoá cache token → kiểm sha lần nữa. Thời gian ước tính từ tốc độ đã đo trên cùng loại GPU (nhúng: CafeBERT 30,8 tin/s, XLM-R khoảng 104, ViSoBERT khoảng 90, BamiBERT khoảng 75; RNN CE cũ: PhoBERT 16,5 phút, CafeBERT 27–49, XLM-R 29, ViSoBERT 75); run mới có thể dừng sớm hơn như BamiBERT (54,5 so với 99,9 phút).

| Máy | Việc | Run id dự kiến | Ước tính | Chi phí |
|---|---|---|---|---|
| **A** (đang chạy, dot-12) | XLM-R 512: nhúng pooled + token last4, RNN seed 42 | `dl-cat-xlmr-rnn-last4-cweff999-ls05-s42` | khoảng 1 giờ | khoảng $0,25 |
| **B** (đang chạy, dot-14) | ViSoBERT 512: nhúng pooled + token last4, RNN seed 42 | `dl-cat-visobert-rnn-last4-cweff999-ls05-s42` | khoảng 1,5 giờ | khoảng $0,4 |
| **C** (đang chạy, dot-13) | CafeBERT 512, rồi PhoBERT 256: nhúng token last4 (pooled last4 đã có trên Mac), RNN seed 42 | `dl-cat-cafebert-rnn-last4-cweff999-ls05-s42`, `dl-cat-phobert-rnn-last4-cweff999-ls05-s42` | khoảng 1,5–2 giờ | khoảng $0,4–0,5 |
| (tuỳ chọn, sau khi có máy rảnh) | BamiBERT RNN seed 43–44 để kiểm nhiễu của kết quả 1 seed | `dl-cat-bamibert-rnn-last4-cweff999-ls05-s43`, `-s44` | khoảng 1 giờ mỗi seed | khoảng $0,25 mỗi seed |
| **Mac** (cpu) | Dense last4 + cw + ls: BamiBERT (cache đã có), XLM-R, ViSoBERT (sau khi máy A kéo cache pooled về); CE lớp cuối 5 seed làm mốc | `dl-cat-<enc>-last4-cweff999-ls05-s42..46` | vài phút | $0 |

Lưu ý khi chạy: (1) Script có cờ môi trường `SKIP_POOLED=1` bỏ bước nhúng pooled; máy C dùng nó (pooled CafeBERT/PhoBERT đã có ở Mac), máy A không dùng vì cần pooled cho XLM-R/ViSoBERT. (2) Đợt vast riêng cho CafeBERT RNN chỉ đổi loss ở lớp cuối ([dot-10](dot-10-cafebert-rnn-cwls.md)) nay chỉ còn là phương án dự phòng. (3) Cờ `logs/SKIP_<encoder>` làm script thoát ngay (đã dùng cho ViSoBERT trên máy A). (4) Mỗi máy có file `vast/dot-N-*.md` (dot-11, 12, 13, 14).

## 6. Việc còn nợ về tài liệu và git

- Chưa ghi các run mới vào `docs/04-results.md`, `docs/06-baseline-dl.md`, `docs/bao-cao/05-chuong-4-thuc-nghiem.md` và `data/eda_xlsx/ket_qua_chay.xlsx` (Rule 1, 7, 13).
- Đã push: các notebook Colab. **Chưa commit, chưa push**: code `--layers` cho cache token (`encode.py`, `train_dl.py`, test), `scripts/vast_run_last4_rnn.sh`, `scripts/vast_run_cafe_rnn_cwls.sh`, `vast/dot-10`, `vast/dot-11`, `vast/theo-doi.md`, dòng 10–11 trong `vast/README.md`. Trong `train_dl.py`, `encode.py`, `tests/test_dl_heads.py` còn lẫn thay đổi của việc khác chưa commit, nên cần stage từng hunk.

## 7. Luật không được vi phạm

1. Không đọc `data/processed/splits/test.csv` để huấn luyện hay chỉnh tham số (chỉ chấm một lần, cờ `--confirm-test`).
2. `docs/04-results.md` chỉ thêm dòng, không sửa dòng cũ.
3. Không destroy máy trước khi `best.pt`, `metrics.json`, log đã nằm trên Mac và `torch.load` mở được.
4. Mọi con số phải có nguồn: `artifacts/<run>/metrics.json` hoặc một dòng trong `docs/04-results.md`.
5. Không dùng git worktree. Không push khi chưa được yêu cầu.

## 8. Cách theo dõi (đọc)

```bash
ssh -o BatchMode=yes -p PORT root@IP 'cd /workspace/vietjobs && tail -3 logs/last4-all.txt; \
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h /workspace | tail -1'
ssh -p PORT root@IP 'cat /workspace/vietjobs/artifacts/<run_id>/history.jsonl'   # epoch, train_loss, val_macro_f1
```

Run RNN kết thúc khi 8 epoch liên tiếp không vượt đỉnh (bình thường, không phải lỗi); `best.pt` giữ epoch đỉnh. Log kết thúc đúng có dòng `DONE` và hai dòng `sha256 splits + manifest ok (before/after)`.

## 9. Báo ngay khi

- Log có `Traceback`, `SHA SAI`, `FileNotFound`, `OutOfMemory`.
- Đĩa trống dưới 10 GB (cache token CafeBERT/BamiBERT chiếm 40–80 GB).
- GPU 0 % và không có tiến trình `train_dl`/`encode` trong khi tmux còn sống quá 10 phút.
- Con số Claude báo không khớp `metrics.json`.

## 10. Lưu ý đọc số

- RNN trên `cuda` không tất định, nhiễu giữa các seed khoảng 0,008 macro-F1: **một seed chưa đủ kết luận** (kết quả BamiBERT 0,6283 mới là 1 seed).
- Dense chạy `cpu` (tất định): chênh dưới 0,003 coi là nhiễu.
- "Không tính lớp rác" là `f1_macro_no_junk` (bỏ lớp `nhóm_nghề_khác`).
- Hai ngưỡng để so với mốc: mức tăng của CafeBERT Dense +0,0139 (5/5 seed thắng) đủ tin; mức của PhoBERT Dense +0,0023 nằm trong nhiễu.
