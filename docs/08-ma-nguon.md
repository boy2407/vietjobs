[← Tổng quan](00-tong-quan.md) · [← Đường cơ sở học sâu](06-baseline-dl.md) · [Lộ trình →](09-lo-trinh.md)

# Mã nguồn

10 mô-đun trong `src/vietjobs/` (không tính `__init__.py`) · 2.937 dòng · 162 kiểm thử
(160 chạy ở `.venv`, 2 kiểm thử chạy hết vòng huấn luyện cần `transformers` nên chỉ chạy ở
`.venv-dl`, nơi 29 kiểm thử có torch đều xanh).
Nhánh chính là gói [`dl/`](../src/vietjobs/dl/).

---

```mermaid
%%{init:{'theme':'base','themeVariables':{
  'primaryColor':'#EFF3F1','primaryTextColor':'#141F1D','primaryBorderColor':'#54625E',
  'lineColor':'#54625E','fontSize':'13px','fontFamily':'Be Vietnam Pro, Segoe UI, sans-serif',
  'clusterBkg':'#FFFFFF','clusterBorder':'#C3CFCB','edgeLabelBackground':'#FFFFFF',
  'secondaryColor':'#FFFFFF','tertiaryColor':'#FFFFFF','mainBkg':'#FFFFFF',
  'nodeTextColor':'#141F1D','titleColor':'#141F1D'}}}%%
flowchart LR
    classDef default fill:#FFFFFF,stroke:#54625E,stroke-width:1.5px,color:#141F1D
    CFG["config.py<br/>đường dẫn · seed · hằng số"]
    VT["<b>vitext.py</b><br/>9 bước tiếng Việt"]
    DS["dataset.py<br/>làm sạch · chia tách"]
    FT["features.py<br/>resolve_column"]
    EV["evaluate.py<br/>chỉ số cho 3 tác vụ"]
    RES["resources/<br/>từ viết tắt · từ dừng · tỉnh thành"]
    subgraph DL["src/vietjobs/dl/ — nhánh chính"]
      TX["text.py<br/>ghép 3 trường<br/>qua resolve_column"]
      EN["encode.py<br/>PhoBERT đông cứng<br/>→ bộ nhớ đệm .npy"]
      HD["heads.py<br/>dense 768→h→h/2→out"]
      TD["train_dl.py<br/>CLI · history.jsonl"]
    end
    EX["external.py<br/>VietJobs-37K: tách chuỗi ·<br/>ánh xạ 60→16 · bắt trùng"]

    RES --> VT
    VT --> EX
    EX --> EV
    CFG --> VT
    VT --> DS --> FT
    FT --> TX --> EN --> TD
    HD --> TD
    EV --> TD
```


| Tệp | Số dòng | Vai trò | Tài liệu |
|---|---|---|---|
| [`vitext.py`](../src/vietjobs/vitext.py) | 584 | Toàn bộ xử lý tiếng Việt. Các hàm thuần túy (pure functions), mỗi bước được kiểm thử riêng | [02](02-vietnamese-nlp.md) |
| [`dataset.py`](../src/vietjobs/dataset.py) | 381 | Làm sạch + tập chia cố định (frozen split). Việc tách từ chỉ chạy **một lần** và được lưu vào bộ nhớ đệm trong CSV. `load_split` là hàm đọc duy nhất được hỗ trợ | [01](01-data-audit.md) |
| [`features.py`](../src/vietjobs/features.py) | 75 | `resolve_column` — cánh cửa duy nhất quyết định tác vụ nào đọc cột nào | [05](05-phan-tich-du-lieu.md) |
| [`evaluate.py`](../src/vietjobs/evaluate.py) | 252 | Chỉ số cho cả ba tác vụ + bootstrap và so sánh theo cặp (paired comparison). Mọi lần chạy dùng chung nên các con số so sánh được với nhau | [03](03-protocol.md) |
| [`config.py`](../src/vietjobs/config.py) | 85 | Đường dẫn · `SPLIT_SEED` · nhóm cột · tên tác vụ | — |
| **[`dl/text.py`](../src/vietjobs/dl/text.py)** | 40 | Ghép ba trường văn bản thành đầu vào cho PhoBERT. Không phụ thuộc torch nên kiểm thử của nó chạy được trên mọi máy | [06](06-baseline-dl.md) |
| **[`dl/encode.py`](../src/vietjobs/dl/encode.py)** | 253 | PhoBERT đông cứng, hai mức cache theo họ cột `raw`/`masked`: `--pooling mean` → `[n, 768]` float32; `--pooling none` → `[n, 256, 768]` float16 memmap + mặt nạ, ghi tăng dần (T8.1). Sidecar `.json` ghi tokenizer/`max_len`/cột nguồn/pooling | [06](06-baseline-dl.md) |
| **[`dl/heads.py`](../src/vietjobs/dl/heads.py)** | 104 | `DenseHead` 768 → h → h/2 → out, và `BiGruLstmCnn` (T8.2): SpatialDropout → Bi-GRU ‖ Bi-LSTM → Conv1d → avg/max pool có mặt nạ → dense; `pack_padded_sequence` nên đầu ra bất biến với pad | [06](06-baseline-dl.md) |
| **[`dl/train_dl.py`](../src/vietjobs/dl/train_dl.py)** | 452 | `run(args)` = một lượt chạy = một dòng trong [04-results.md](04-results.md) + `history.jsonl`; `--head {dense,rnn}` đọc cache gộp hay memmap token theo batch; `--max-minutes` trần thời gian (ghi `stopped_by`); `--conv-channels`, `--branches` cho T8.5 | [03](03-protocol.md) · [06](06-baseline-dl.md) |
| [`notebooks/colab_setup.py`](../notebooks/colab_setup.py) | 164 | Khôi phục splits + cache nhỏ từ Google Drive cho một phiên Colab, encode lại train token của họ cần dùng, chạy cổng test. Ba notebook `T8_3_category` / `T8_4_salary` / `T8_5_ablation.ipynb` gọi chung nó | memory `colab-drive-paths` |
| [`external.py`](../src/vietjobs/external.py) | 549 | VietJobs-37K làm tập ngoài: tách `[TITLE]/[REQ]/[DESC]` về ba cột của ta, ánh xạ 60 → 16 (`apply_crosswalk`), bắt tin trùng với ba tập chia bằng băm chính xác **hoặc** cùng tiêu đề + Jaccard ≥ 0,5 (`overlap_mask`), quy ước lenient. Đọc lương bạc từ văn bản: `extract_salary` (mỏ neo từ lương hoặc tiêu đề mục → con số cùng câu), `audit_salary` xếp mọi tin vào một nhóm phủ, `salary_note`, `salary_rule`, `template_ids`. Không phụ thuộc torch | [10](10-danh-gia-ngoai.md) |
| [`scripts/eval_external.py`](../scripts/eval_external.py) | 199 | Chấm một run `category` trên VietJobs-37K: khử trùng → ánh xạ → mã hóa PhoBERT (cache `ext37k-*`) → nạp `best.pt` + `scaler.npz` → strict/lenient + bootstrap → một dòng [04-results.md](04-results.md) mỗi tập | [10](10-danh-gia-ngoai.md) |
| [`scripts/build_ext37k.py`](../scripts/build_ext37k.py) | 146 | Gộp bốn tập con 37K + cột lương bạc → `ext37k.csv`, `ext37k-sal.csv`. `--audit` in bảng nhóm phủ kín (không ghi tệp), `--review N [--rule TÊN]` sinh `review-salary.csv` để người soát — lọc trùng theo cả `template_id` lẫn `salary_text` | [10](10-danh-gia-ngoai.md) §3 |
| [`scripts/analyze_data.py`](../scripts/analyze_data.py) | 415 | Các phép đo + 11 hình cho [05](05-phan-tich-du-lieu.md). Số mô tả đo trên tệp gốc 47.707 dòng; mốc sàn khớp trên `train`, chấm trên `dev` (Rule 8) | [05](05-phan-tich-du-lieu.md) |
| [`scripts/probe_embeddings.py`](../scripts/probe_embeddings.py) | 85 | Một đầu dò tuyến tính (linear probe) trên các vector PhoBERT — tách biệt "đặc trưng kém" khỏi "đầu ra bị hỏng" | [06 §5.4](06-baseline-dl.md#54-thất-bại-đầu-tiên-giữ-lại-làm-bằng-chứng) |
| [`scripts/measure_vitext.py`](../scripts/measure_vitext.py) | 125 | Đo lại bằng chứng cho bảng chín bước | [02](02-vietnamese-nlp.md) |
| [`scripts/export_runs.py`](../scripts/export_runs.py) | 188 | Dựng lại `data/eda_xlsx/ket_qua_chay.xlsx` từ [04-results.md](04-results.md) + `artifacts/<run_id>/`: mọi lần chạy kèm hàm loss, γ, thiết bị, phiên bản thư viện. Bảng là dẫn xuất, không bao giờ sửa tay (Rule 13) | [06 §5](06-baseline-dl.md#5-kết-quả) |

---

## Ba bất biến giữ cho toàn hệ thống đứng vững

**0. Một cánh cửa duy nhất.** Nhánh DL không tự nối vào một cột thô: nó gọi
`features.resolve_column` qua [`dl/text.py`](../src/vietjobs/dl/text.py). Đó là lý
do quy tắc chống rò rỉ lương chỉ cần được canh giữ ở một nơi.

**1. `vitext.py` hoàn toàn là các hàm thuần túy.** Cùng đầu vào, cùng đầu ra, không
đọc trạng thái bên ngoài, không học gì từ dữ liệu. Đó là điều cho phép `dataset.py`,
`features.py` và `dl/text.py` gọi cùng một đoạn mã mà không bị lệch nhau. Đây là
Quy tắc 4 trong [../AGENTS.md](../AGENTS.md).

**2. Chỉ có một cánh cửa duy nhất quyết định tác vụ nào đọc cột nào** —
`features.resolve_column`. Quy tắc 3, được giữ bởi
[`tests/test_no_leak.py`](../tests/test_no_leak.py).

---

## Kiểm thử

123 kiểm thử, đếm bằng `pytest --collect-only -q`:

| Tệp | Số kiểm thử | Giữ điều gì |
|---|---|---|
| [`tests/test_vitext.py`](../tests/test_vitext.py) | 56 | Mọi bước xử lý tiếng Việt, bao gồm bốn trường hợp cụm từ "40 triệu người dùng" phải **không** bị che · cờ chuyển đổi bộ tách từ phải thực sự chuyển được bộ tách từ |
| [`tests/test_no_leak.py`](../tests/test_no_leak.py) | 14 | Các tác vụ lương không bao giờ đọc cột chưa che · **mọi tên trong lá chắn phải là cột có thật** · ba cột kỹ năng không chứa số liệu lương |
| [`tests/test_dl_text.py`](../tests/test_dl_text.py) | 8 | Nhánh DL đọc đúng cột: các tác vụ lương chỉ thấy `*_masked`, phân loại thấy văn bản thô, cả hai đều được tách từ, đầu vào không bị lọc stopword và không bị viết thường |
| [`tests/test_external.py`](../tests/test_external.py) | 46 | Tách chuỗi 37K đúng thứ tự cột của ta, `nan` → rỗng · nhãn `null` bị loại và được đếm · strict chỉ nhận tin đúng một nhãn · băm chính xác bắt tin đăng lại, đường mờ sống sót qua `[COMPANY]`, cùng tiêu đề khác mô tả **không** bị coi là trùng · bảng ánh xạ phủ đủ 60 nhãn và chỉ trỏ tới 16 lớp có thật · lương đọc từ văn bản: mỗi luật một cặp nhận/từ chối, nhóm phủ kín, `salary_note` luôn có giá trị khi không có lương, thân tin giống hệt chung một `template_id` |
| [`tests/test_bootstrap.py`](../tests/test_bootstrap.py) | 9 | Bootstrap tất định khi biết trước seed; hai mô hình giống hệt nhau hòa ở mức 0,5; so sánh theo cặp nhạy hơn so với σ độc lập; F1 theo từng mẫu của arXiv:2112.11052 bằng đúng accuracy khi đơn nhãn |
| [`tests/test_dl_encode.py`](../tests/test_dl_encode.py) | 13 | Cache token (T8.1): trung bình có mặt nạ của `[n, 256, 768]` bằng cache gộp (tol 1e-2), shape khớp `manifest.json`, pad ghi 0, sidecar ghi `pooling: none` — skip theo từng tổ hợp train/dev × raw/masked còn thiếu |
| [`tests/test_dl_heads.py`](../tests/test_dl_heads.py) | 16 | `FocalLoss`; `BiGruLstmCnn` (T8.2): shape cho cả hai bài × ba cấu hình nhánh, **bất biến với pad**, max-pool không đọc pad, số tham số theo nhánh; **hai kiểm thử chạy hết `run()`** — một epoch Dense trọn vẹn và `--max-minutes` cắt đúng chỗ (cần torch + transformers, chạy ở `.venv-dl`) |

---

## Các lệnh dùng thường xuyên

```bash
python scripts/analyze_data.py                      # đo lường + vẽ hình cho tài liệu 05
python -m vietjobs.dl.encode   --task category --splits train dev   # embed, đã lưu vào bộ nhớ đệm
python -m vietjobs.dl.train_dl --task category --class-weight
python -m vietjobs.dl.train_dl --task salary
PYTHONPATH=src .venv-dl/bin/python scripts/probe_embeddings.py --task category   # mốc chẩn đoán
pytest -q

python -m vietjobs.dataset build                    # xây lại các tập chia (hiếm khi cần)
python scripts/measure_vitext.py                    # đo lại bảng chín bước
./scripts/render_figures.sh                         # xuất các sơ đồ ra SVG/PNG

# head đọc token (T8) — chạy trên GPU Colab, xem notebooks/T8_*.ipynb
python -m vietjobs.dl.encode --task category --splits train dev --pooling none --device cuda
python -m vietjobs.dl.train_dl --task category --head rnn --device cuda --hidden 100 --conv-channels 50 --max-minutes 35
python notebooks/colab_setup.py --family raw       # trên Colab: khôi phục từ Drive + cổng test
```

---

## Đọc tiếp

- [Những gì còn lại trên lộ trình](09-lo-trinh.md) — các hạng mục công việc, theo thứ tự
- [Giao thức thí nghiệm](03-protocol.md) — hợp đồng mà `dl/train_dl.py` tuân thủ
- [../AGENTS.md](../AGENTS.md) — các quy tắc làm việc
