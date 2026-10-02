# Đợt 3: CafeBERT 512-token + trunk RNN đa nhiệm (T4.1) — CHƯA THUÊ

Script: `scripts/vast_run_cafe_mtl.sh` | Instance: —

| Run | cat macro-F1 | sal MAE | Ghi chú |
|---|---|---|---|
| `dl-mtl-cafe-rnn-a5-s42` | | | gate |
| `dl-mtl-cafe-rnn-a5-s43` | | | |
| `dl-mtl-cafe-rnn-a5-s44` | | | |

Cấu hình: `--alpha-sal 5 --epochs 80` (α chọn từ quét trên trunk dense, xem `docs/06` §5.7).

Ngưỡng gate (đơn nhiệm cùng seed 42): `dl-cat-cafe-rnn-ce-s42` 0.6172, `dl-sal-cafe-rnn-s42` 3.89 tr.

### Tiến độ

| # | Bước | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Thuê máy (disk ≥ 80 GB, RAM ≥ 48 GB) | todo | |
| 2 | Gắn SSH key + kết nối | todo | |
| 3 | Đẩy code + splits lên | todo | không cần embeddings |
| 4 | Cài thư viện Python | todo | |
| 5 | `bash scripts/vast_run_cafe_mtl.sh gate` (encode masked + seed 42) | todo | |
| 6 | Kiểm tra gate | todo | |
| 7 | `bash scripts/vast_run_cafe_mtl.sh rest` (seed 43, 44) | todo | |
| 8 | Kéo kết quả + `best.pt` về Mac | todo | |
| 9 | Destroy instance (sau checklist trong README) | todo | |

### Lệnh đẩy lên (terminal Mac)

```bash
cd ~/Documents/TrNghia/VietJob/vietjobs
rsync -avzR -e "ssh -p PORT" src scripts tests pyproject.toml data/processed/splits \
  root@IP:/workspace/vietjobs/
```
