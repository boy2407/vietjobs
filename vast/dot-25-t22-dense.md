# Đợt 25 — T22: CafeBERT 512 Dense, lọc xung đột nhãn, cột mới, lặp tin

**Task:** T22.2–T22.6 ([tasks/T22-cot-moi-loc-xung-dot.md](../tasks/T22-cot-moi-loc-xung-dot.md)).
**Trạng thái:** KHÔNG DÙNG — máy ngắt kết nối khi đang đẩy dữ liệu (chỉ 18 MB lên tới nơi); toàn bộ T22 chạy trên Mac `cpu`. Không có kết quả nào trên máy này; chờ tác giả destroy.

**Máy:** `ssh -p 45233 root@223.166.245.253` (proxy `ssh5.vast.ai:17279`). RTX 3090, 20 lõi, RAM 31 GB, ổ 60 GB. Đã cài venv torch 2.2.2+cpu, numpy 1.26.4, pandas 2.3.3, sklearn 1.6.1, transformers 4.46.3.

**Cách chạy.** Không nhúng gì trên máy này: mọi cache (`cafebert-{train,dev}-{raw,masked}-len512`,
`cafebert-{train,dev}-xf-*-len128`) đẩy từ Mac lên, vì nhúng lại trên máy khác cho vector khác
(`kinh-nghiem.md` §6). Head Dense chạy `--device cpu` (Rule 13). Toàn bộ R0–R4 chạy lại trên máy
này với hậu tố `-vast`, để mọi so sánh nằm trên cùng một máy; R0, R1 đã chạy trên Mac giữ nguyên
trong `04-results.md`.

| Bước | Lệnh / việc | Trạng thái |
|---|---|---|
| 1 | Kiểm máy: `nproc`, `free -g`, `df -h /` | chưa |
| 2 | Đẩy code + `data/processed/splits` + `manifest.json` + 8 cache | chưa |
| 3 | Cài môi trường (giữ torch của image, thêm `pyarrow`) | chưa |
| 4 | Chạy R0–R3 × 2 bài × 3 seed, rồi R4 | chưa |
| 5 | Kéo `artifacts/dl-*-t22-*-vast` và các dòng mới của `04-results.md` về Mac | chưa |
| 6 | Kiểm checklist `vast/README.md`, rồi tác giả destroy | chưa |

| run_id | Kết quả (`metrics.json`) |
|---|---|
| — | — |
