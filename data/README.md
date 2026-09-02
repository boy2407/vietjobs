# Data

`data/` nằm trong `.gitignore`. Không có gì trong này được commit.

## Lấy dữ liệu thô

Bộ dữ liệu là `dinhieufam/VietJobs` trên Hugging Face Hub — khoảng 103 MB,
**48.092 dòng**, 18 cột. Tải về `data/raw/VietJobs.csv`.

```bash
pip install -r requirements.txt
python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
p = hf_hub_download('dinhieufam/VietJobs', 'VietJobs.csv',
                    repo_type='dataset', cache_dir='data/raw/.cache/huggingface')
pathlib.Path('data/raw').mkdir(parents=True, exist_ok=True)
shutil.copy(p, 'data/raw/VietJobs.csv')
print('ok')"
```

> `scripts/download_data.sh` được nhắc trong bản README cũ nhưng **chưa tồn tại**.
> Đoạn lệnh trên là cách tải thật đang dùng; thư mục `data/raw/.cache/huggingface/`
> là dấu vết của lần tải đó.

Kiểm tra đúng file bằng băm đã chốt trong manifest:

```bash
shasum -a 256 data/raw/VietJobs.csv
# 85862b06fda4e814fe0c1d8622f173d189c92758345f77df16d1232d0c49d477
```

## Dựng splits

```bash
python -m vietjobs.dataset build          # ~26 phút, phần lớn là tách từ
python -m vietjobs.dataset build --no-segment   # nhanh, để gỡ lỗi
```

## Bố cục

| Đường dẫn | Sinh bởi | Nội dung |
|---|---|---|
| `data/raw/VietJobs.csv` | tải từ Hub | Bản gốc, không đụng vào |
| `data/processed/splits/` | `python -m vietjobs.dataset build` | `train/val/test.parquet` — đóng băng, không nhóm nào lọt giữa hai tập |
| `data/processed/manifest.json` | `python -m vietjobs.dataset build` | Số dòng, băm sha256, seed chia tập, thống kê từng tập |
| `data/interim/` | — | Chỗ để file tạm, hiện đang trống |

Kích thước sau khi dựng: train 33.396 dòng · val 7.159 · test 7.152.

## Splits là hợp đồng

Ba file parquet là hợp đồng cho **mọi** thí nghiệm. **Không dựng lại với seed khác**
sau khi đã huấn luyện mô hình đầu tiên — làm vậy là mọi dòng trong
`docs/04-results.md` mất khả năng so sánh với nhau.

Chi tiết: [`docs/03-protocol.md`](../docs/03-protocol.md).
Vì sao làm sạch như vậy: [`docs/01-data-audit.md`](../docs/01-data-audit.md).
