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
