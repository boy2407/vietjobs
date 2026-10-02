# Đợt 3: CafeBERT 512-token + trunk RNN đa nhiệm (T4.1) — CHƯA THUÊ

Script: `scripts/vast_run_cafe_mtl.sh` | Instance: — (GPU, VRAM, disk, RAM, IP, port, $/h)

Cấu hình: `--head rnn --encoder cafebert --max-len 512 --hidden 100 --conv-channels 50 --alpha-sal 5 --epochs 80 --device cuda`, family `masked` (α chọn từ quét trên trunk dense, xem `docs/06` §5.7).

### Chọn máy

- RAM ≥ 48 GB (nên 64 GB): cache float16 masked train ~36 GB + dev ~4,5 GB phải nằm trong page cache.
- Disk ≥ 100 GB, NVMe; PCIe 4.0 x16; CPU EPYC / Xeon / Threadripper.
- GPU tầm trung là đủ (RTX 4070 Ti / 5060 Ti / 3080): nút thắt là I/O, không phải GPU (xem `kinh-nghiem.md` §1–2).

### Bảng run (số lấy từ `artifacts/<run_id>/metrics.json`)

| Run | cat macro-F1 | sal MAE | best ep cat / sal | epochs_run | s/epoch | Thời gian | Ghi chú |
|---|---|---|---|---|---|---|---|
| `dl-mtl-cafe-rnn-a5-s42` | | | | | | | gate |
| `dl-mtl-cafe-rnn-a5-s43` | | | | | | | |
| `dl-mtl-cafe-rnn-a5-s44` | | | | | | | |

### Gate (seed 42, so với đơn nhiệm cùng seed)

Mốc: `dl-cat-cafe-rnn-ce-s42` macro-F1 0.6172, `dl-sal-cafe-rnn-s42` MAE 3.89 tr.

- **Pass** nếu cat macro-F1 ≥ 0.6172 **hoặc** sal MAE ≤ 3.89 tr, với điều kiện đầu còn lại không tệ hơn quá ngưỡng: Δ macro-F1 > −0.01, Δ MAE < +0.10 tr. Khi pass thì chạy `rest`.
- **Fail** thì không chạy `rest`. Kéo kết quả về, làm checklist rồi destroy.

### Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy theo mục "Chọn máy" | todo | |
| 2 | Gắn SSH key + kết nối | todo | |
| 3 | Đẩy code + splits lên | todo | không cần embeddings |
| 4 | Cài thư viện Python | todo | |
| 5 | `gate`: pytest → encode masked train+dev → seed 42 | todo | |
| 6 | Kiểm tra gate | todo | |
| 7 | `rest` (seed 43, 44) | todo | chỉ khi gate pass |
| 8 | Kéo kết quả **kèm `best.pt` + `best_sal.pt`** về Mac | todo | |
| 9 | Checklist trong `README.md` | todo | |
| 10 | Destroy instance | todo | |
| 11 | Phân tích: mean ± sd 3 seed, so cặp với đơn nhiệm cùng seed và MTL dense α=5 | todo | |

### Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml data/processed/splits \
  root@IP:/workspace/vietjobs/
```

### Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn pyarrow underthesea pyvi pytest
tmux new -s mtl
bash scripts/vast_run_cafe_mtl.sh gate
# Sau khi gate pass:
bash scripts/vast_run_cafe_mtl.sh rest
```

### Theo dõi lúc train (cửa sổ ssh thứ hai)

```bash
tmux attach -t mtl                                            # xem tiến trình chính
tail -f logs/dl-mtl-cafe-rnn-a5-s42.txt                       # dòng epoch: loss, macro-F1, MAE
tail -n 3 artifacts/dl-mtl-cafe-rnn-a5-s42/history.jsonl     # train_loss_cat / train_loss_sal / seconds
watch -n 5 nvidia-smi                                         # GPU util, VRAM
free -g; df -h /workspace                                     # RAM (swap?), disk
```

Ghi vào "Nhật ký sự cố" khi gặp: GPU util < 30 % kéo dài (nghẽn I/O), RAM bị swap, disk > 90 %, MAE không giảm sau 10 epoch, hoặc process chết.

### Lệnh kéo kết quả về Mac

Không dùng rsync (openrsync của macOS bị treo). **Không** thêm `--exclude=best.pt`.

```bash
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/dl-mtl-cafe-rnn-a5-s4? logs" | tar xzf - -C .
ls artifacts/dl-mtl-cafe-rnn-a5-s4?/{metrics.json,history.jsonl,best.pt,best_sal.pt}
```

### Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
| | | |

### Kết quả bước 11

(điền sau khi chạy)
