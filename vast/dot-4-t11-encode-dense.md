# Đợt 4: T11 — ViSoBERT, XLM-R base, BamiBERT: encode + Dense, RNN, MTL — CHƯA THUÊ

Script: `scripts/vast_run_t11_encode.sh` (đối số `all` | `visobert` | `xlmr` | `bamibert`) | Instance: — (chưa thuê)

Phạm vi: trên máy thuê, encode rồi train **mọi run seed 42** của T11 (bảng dưới). Dense và MTL dense chạy `--device cpu` (Rule 13 item 5), RNN và MTL RNN chạy `cuda`. Sau đó kéo cache và kết quả về Mac như các đợt trước.

Độ dài (tác giả chọn 2026-10-12): ViSoBERT và XLM-R **512** (tối đa của model), BamiBERT **1024** cho mọi run. Đo bằng tokenizer trên train + dev: tin dài nhất 1.553 (BamiBERT), 1.609 (XLM-R), 1.641 (ViSoBERT) token; cắt ở 512 là 4,4–6,1 % tin, ở 1024 là 0,1–0,2 %.

ViSoBERT: pooled cache 512 và 3 run Dense đã chạy trên Mac (T11.1, T11.2, T11.14) → trên máy thuê chỉ chạy phần token.

### Cache tạo trên máy thuê (ước tính từ kích thước mảng, chưa đo)

| Encoder | max_len | Pooled (raw + masked, train + dev) | Token cache f16 (raw + masked, train + dev) |
|---|---|---|---|
| `visobert` | 512 | — (đã có trên Mac) | ≈ 60 GB |
| `xlmr` | 512 | ≈ 0,23 GB | ≈ 60 GB |
| `bamibert` | 1024 | ≈ 0,23 GB | ≈ 120 GB |

### Chọn máy

- Disk ≥ 300 GB NVMe (≈ 240 GB token cache + model + splits).
- RAM ≥ 64 GB, nên 96 GB: run RNN BamiBERT đọc cache train một họ ≈ 54 GB, phải nằm được trong page cache (`kinh-nghiem.md` §1–2: nút thắt RNN là I/O).
- GPU 16 GB tầm trung là đủ (RTX 4070 Ti / 5060 Ti / 3080); CPU EPYC / Xeon cho các run Dense cpu.
- PCIe 4.0 x16 trở lên, NVMe ≥ 3.000 MB/s, CUDA ≥ 12.1, image có `/venv/main` (như Đợt 3).
- Không cần 2 GPU: chạy song song hai run RNN thì cần hai cache cùng nằm trong RAM (≥ 128 GB), khó tìm máy rẻ.

**Bộ lọc gợi ý trên vast.ai:** 1× RTX 5060 Ti / 4070 Ti / 3080 (16 GB) · RAM ≥ 96 GB (tối thiểu 64) · disk 300 GB · PCIe 4.0 x16 · CPU EPYC/Xeon ≥ 8 lõi · reliability ≥ 98 % · giá ≤ $0,35/h.

### Ước tính thời gian và chi phí (chưa đo — ngoại suy từ Đợt 3)

Mốc đo được (Đợt 3, RTX 5060 Ti + EPYC 9534, `kinh-nghiem.md` §5): CafeBERT-large 512 encode token 30,8 tin/s; trunk RNN đọc cache ≈ 36 GB → 142,7 s/epoch, 26 epoch. RNN nghẽn ở I/O nên s/epoch được ngoại suy theo kích thước cache; encoder base nhanh ≈ 3 lần large.

| Phần | Cách tính | Ước tính |
|---|---|---|
| Encode ViSoBERT token 512 (2 họ × 38.166 tin) | ≈ 90 tin/s | ≈ 15 phút |
| Encode XLM-R pooled + token 512 (4 lượt) | ≈ 90 tin/s | ≈ 30 phút |
| Encode BamiBERT pooled + token 1024 (4 lượt) | ≈ 35 tin/s | ≈ 75 phút |
| 6 run Dense / MTL dense (cpu) | như trên Mac: 15 s – vài phút/run | ≈ 15 phút |
| 6 run RNN @512 (ViSoBERT, XLM-R) | cache ≈ 27 GB → ≈ 107 s/epoch × ≈ 25 epoch | ≈ 45 phút/run → ≈ 4,5 giờ |
| 3 run RNN @1024 (BamiBERT) | cache ≈ 54 GB → ≈ 215 s/epoch × ≈ 25 epoch | ≈ 90 phút/run → ≈ 4,5 giờ |
| **Tổng** | + ≈ 1 giờ đẩy code, cài đặt, kéo về | **≈ 12 giờ** (để dư: thuê 14–16 giờ) |

Chi phí: máy ≈ $0,22–0,35/h (Đợt 3 trả $0,22/h; disk 300 GB làm giá nhích lên) → **≈ $3–6** cho cả đợt.

### Bảng run (số lấy từ `artifacts/<run_id>/metrics.json`)

| Task | Run | Head · device | cat macro-F1 | sal MAE | Trạng thái |
|---|---|---|---|---|---|
| T11.1 | `dl-cat-visobert-dense-512` | Dense · cpu (Mac) | 0.5723 | — | done |
| T11.2 | `dl-sal-visobert-dense-512` | Dense · cpu (Mac) | — | 4.49 | done |
| T11.14 | `dl-mtl-visobert512-a5-e80-s42` | MTL dense · cpu (Mac) | 0.5578 | 4.45 | done |
| T11.3 | `dl-cat-visobert-rnn-512` | RNN · cuda | | | todo |
| T11.4 | `dl-sal-visobert-rnn-512` | RNN · cuda | | | todo |
| T11.15 | `dl-mtl-visobert-rnn512-a5-s42` | MTL RNN · cuda | | | todo |
| T11.9 | `dl-cat-xlmr-dense-512` | Dense · cpu | | | todo |
| T11.10 | `dl-sal-xlmr-dense-512` | Dense · cpu | | | todo |
| T11.18 | `dl-mtl-xlmr512-a5-e80-s42` | MTL dense · cpu | | | todo |
| T11.11 | `dl-cat-xlmr-rnn-512` | RNN · cuda | | | todo |
| T11.12 | `dl-sal-xlmr-rnn-512` | RNN · cuda | | | todo |
| T11.19 | `dl-mtl-xlmr-rnn512-a5-s42` | MTL RNN · cuda | | | todo |
| T11.5 | `dl-cat-bamibert-dense-1024` | Dense · cpu | | | todo |
| T11.6 | `dl-sal-bamibert-dense-1024` | Dense · cpu | | | todo |
| T11.16 | `dl-mtl-bamibert1024-a5-e80-s42` | MTL dense · cpu | | | todo |
| T11.7 | `dl-cat-bamibert-rnn-1024` | RNN · cuda | | | todo |
| T11.8 | `dl-sal-bamibert-rnn-1024` | RNN · cuda | | | todo |
| T11.17 | `dl-mtl-bamibert-rnn1024-a5-s42` | MTL RNN · cuda | | | todo |

Cấu hình: Dense mặc định của `train_dl` (như `dl-cat-cafe-dense-512`); RNN `--hidden 100 --conv-channels 50` (như T8.16), `category` `--loss ce`; MTL `--alpha-sal 5 --epochs 80` (như T4.1). Run nào đã có `metrics.json` thì script bỏ qua → chạy lại sau sự cố được.

### Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy theo mục "Chọn máy" | todo | tác giả thuê |
| 2 | Gắn SSH key + kết nối | todo | |
| 3 | Đẩy code + splits + `manifest.json` lên | todo | Đợt 3 quên `manifest.json` |
| 4 | Cài thư viện, pytest | todo | script tự chạy pytest dl trước |
| 5 | `bash scripts/vast_run_t11_encode.sh all` trong tmux | todo | |
| 6 | Kéo kết quả **kèm `best.pt` / `best_sal.pt`** + pooled cache + log về Mac | todo | xem "Kéo về" |
| 7 | Checklist `README.md` | todo | |
| 8 | Destroy instance | todo | tác giả bấm trên web |
| 9 | Bootstrap so với PhoBERT/CafeBERT, board, docs/06, ch.4, `export_runs.py` | todo | trên Mac |

### Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml data/processed/splits data/processed/manifest.json \
  root@IP:/workspace/vietjobs/
```

### Lệnh chạy (máy thuê)

```bash
cd /workspace/vietjobs
source /venv/main/bin/activate
pip install "transformers>=4.38,<4.47" sentencepiece protobuf pandas scikit-learn pyarrow underthesea pyvi pytest
tmux new -s t11
bash scripts/vast_run_t11_encode.sh all
```

### Kéo về

Mac còn ≈ 74 GB trống (`df -h`, 2026-10-12) → **token cache ≈ 240 GB không kéo về được**. Kéo: mọi `artifacts/<run_id>/` của bảng trên (cả `best.pt`), pooled cache `xlmr-*-len512.{npy,json}` và `bamibert-*-len1024.{npy,json}` (không có `-tok`/`-mask`), `logs/`. Token cache nằm lại máy thuê; nếu cần giữ, tác giả quyết định chỗ chứa trước bước 8.

```bash
# tar qua proxy từng treo (Đợt 3) → scp từng file
ssh -p PORT root@IP "cd /workspace/vietjobs && ls -d artifacts/dl-*-{visobert,xlmr,bamibert}* artifacts/dl-mtl-{visobert,xlmr,bamibert}*; \
  ls artifacts/embeddings/{xlmr,bamibert}-* | grep -v -e '-tok' -e '-mask'; ls logs/*"
```

### Nhật ký sự cố

| Thời điểm | Hiện tượng | Xử lý |
|---|---|---|
