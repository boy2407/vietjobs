# 💡 Đúc kết kinh nghiệm & Hướng tối ưu cho các lần chạy Vast.ai tiếp theo

### 1. Bản chất điểm nghẽn I/O (CPU Single-Thread Fancy Indexing)
- **Hiện tượng:** GPU RTX 4090 chỉ ăn ~30% Util (112W/350W, 40°C) và tốn ~75–90s/epoch cho CafeBERT 512.
- **Nguyên nhân:** 
  - Token cache CafeBERT 512 (512 tokens × 1024 dims × float32) nặng **~72 GB**.
  - Code Python đọc memmap ngẫu nhiên 256 mẫu rời rạc (`arr[rows[order]]`) trên **1 luồng CPU duy nhất** và chuẩn hoá `(xb - mu) / sigma` trên CPU mất ~0.45s/batch ($135 \text{ batches} \approx 65\text{s}$).
  - GPU RTX 4090 tính toán chỉ mất ~0.08s/batch, phần lớn thời gian phải chờ CPU gom dữ liệu.

### 2. Tiêu chí chọn máy tối ưu trên Vast.ai
- **Chọn cấu hình cân đối (tránh lãng phí tiền GPU):**
  - Khi train Head RNN trên Token Cache: **1x hoặc 2x RTX 5060 Ti (16GB) / RTX 3080 / RTX 4070 Ti** ($0.08 – $0.15/h) là lựa chọn tối ưu chi phí nhất.
  - **Phương án 2x GPU:** Chạy song song Task Category (GPU 0) và Task Salary (GPU 1) $\rightarrow$ Giảm 50% thời gian chạy mà giá thuê cực rẻ.
- **Yêu cầu phần cứng nền tảng:**
  - **PCIe:** Bắt buộc chọn máy có `PCIe 4.0 x16` hoặc `5.0 x16` (băng thông $\ge 25\text{ GB/s}$).
  - **CPU:** Ưu tiên dòng Server/Workstation (**AMD EPYC, Threadripper, Intel Xeon**) có 4–8 kênh RAM (băng thông RAM 200–400 GB/s).
  - **RAM:** $\ge 64\text{ GB}$ (để toàn bộ 72 GB cache nằm trọn trong RAM cache).
  - **Ổ cứng:** NVMe SSD tốc độ cao ($\ge 3.000\text{ MB/s}$).

### 3. Về định dạng `float16` (`fp16`)
- **Độ chính xác:** Hoàn toàn **không làm giảm độ chính xác** của mô hình nơ-ron (chênh lệch F1/MAE $\le 0.0001$).
- **Lợi ích:** 
  - Giảm dung lượng cache từ **72 GB $\rightarrow$ 36 GB**.
  - Giảm 50% thời gian truyền dữ liệu qua RAM/PCIe và kích hoạt Tensor Cores trên GPU.

### 4. Hướng nâng cấp code nạp dữ liệu (Rút ngắn thời gian từ 90s $\rightarrow$ 15s/epoch)
1. **PyTorch DataLoader bất đồng bộ:** Dùng `DataLoader(..., num_workers=4, pin_memory=True, prefetch_factor=2)` để 4 core CPU nạp trước các batch tiếp theo vào pinned RAM trong lúc GPU đang tính toán batch hiện tại.
2. **GPU Standardization:** Chuyển phép trừ $\mu$ và chia $\sigma$ lên GPU (dùng CUDA cores tính trong 0.001s thay vì tính trên CPU).
3. **Lưu cache `float16`:** Vừa giữ trọn 512 token vừa giảm 50% kích thước dữ liệu.


### 5. Đợt 3 (đo 2026-10-02)
- **Đẩy lên phải kèm `data/processed/manifest.json`**, không chỉ `splits/`: `load_split` đọc kiểu cột từ đó, thiếu là encode dừng ngay.
- **Kéo về qua SSH proxy (`sshN.vast.ai`): `tar czf - … | tar xzf -` treo** (dừng ở 5,6/14 MB). `scp` từng file chạy được; so `md5sum` hai đầu cho `best*.pt`.
- Image PyTorch của vast: không có `python` ngoài venv → `source /venv/main/bin/activate`; login đã ở trong tmux → dùng `tmux switch-client -t mtl` thay vì `attach`.
- RTX 5060 Ti + EPYC 9534, CafeBERT 512 token-level: encode ~30,8 tin/s; trunk RNN đa nhiệm 142,7 s/epoch.


### 6. Đợt 17–20 (đo 2026-10-21)
- **Kiểm `df -h / /workspace` trước khi chạy bất cứ gì.** Trên một máy 3090 `/workspace` là ổ riêng 10 GB (đã đầy), ổ 100 GB nằm ở `/`; cache token 27 GB không ghi được và tiến trình chết với `Bus error`. `vast_run_dot20.sh` tự dừng nếu ổ chứa `artifacts/` còn dưới 70 GB.
- **Train và chấm phải cùng máy, cùng môi trường (Rule 4).** `best.pt` train ở một môi trường rồi chấm trên máy khác cho `test` cao hơn `dev` ở mọi mô hình Dense (khoảng 0,03–0,09 macro-F1); cosine giữa vectơ cache cũ và vectơ encode lại chỉ 0,8875 (PhoBERT, `dev`). Cổng chặn: `scripts/check_mtl_dev.py` chấm lại `dev` và so với `metrics.json`; CafeBERT, ViSoBERT RNN và 5 MTL Dense `-v2` (train+chấm cùng máy) khớp tuyệt đối.
- **Train bị chặn ở CPU một lõi, không phải GPU.** GPU nhảy 0 % ↔ 100 %; mỗi epoch ~2,5 phút trên 5090, 3090 và 4090 (165 s/epoch đo từ `history.jsonl`); mô hình 512 token cache 27–36 GB. Thuê card rẻ nhất đủ 24 GB, ưu tiên CPU mạnh từng lõi và RAM ≥ 64 GB.
- **Máy chia sẻ CPU có thể chậm gấp ~5 lần ở bước thống kê chuẩn hoá** (load average ~10 trong khi tiến trình dùng một lõi): ~28 phút so với ~5,5 phút trên máy khác.
- **Script train: `set -eo pipefail`.** `set -e` kèm `| tee` che mất lỗi của train; một lần chạy hỏng vẫn in "XONG".
- **Đẩy dữ liệu: `ssh … 'cd /workspace/vietjobs && tar xzf -'`**, không dùng `cd … || (…)` trong subshell — `cd` không có tác dụng và dữ liệu rơi vào `/root`.
- Mỗi encoder 512 token cần cache token: CafeBERT last4 ~40 GB (train) + 4 GB (dev), ViSoBERT/XLM-R ~27 + 3 GB; xoá sau mỗi mô hình (script làm sẵn).
- **File `docs/04-results.md` trên máy thuê chứa các dòng nhật ký của run** — lấy chúng (`git diff docs/04-results.md`) trước khi destroy máy.
- **`train_mtl.py` chỉ ghi `best.pt`/`best_sal.pt` khi train kết thúc** ([train_mtl.py](../src/vietjobs/dl/train_mtl.py) cuối hàm `main`). Máy bị tắt (hết tiền) ngay sau khi PhoBERT `-v2` train xong → mất 41 epoch. Khi tài khoản sắp hết tiền, không bắt đầu một run dài; nếu code ghi checkpoint mỗi lần có mức mới thì kéo về được bất cứ lúc nào.
- **Đã sửa (2026-10-21):** `train_mtl.py` ghi `best.pt`/`best_sal.pt` ngay khi có mức mới (ghi nguyên tử) và hỗ trợ `--resume` từ `last.pt`; chạy tiếp trùng bit với chạy liền trên `cpu`. Máy bị tắt giữa run giờ còn `best.pt` để kéo về và chấm, hoặc chạy tiếp bằng `--resume` cùng `--run-id`.
- (đo, Đợt 23) `pip install -r requirements-dl.txt` trên image vast hạ torch về 2.2.2 (ghim cho Mac) và không có `pyarrow` → cài lại torch của image + `pip install pyarrow` **trước** khi train; thiếu `pyarrow` làm `train_mtl.py` crash ở bước ghi `predictions_dev.parquet`, sau khi đã xoá `last.pt`.
