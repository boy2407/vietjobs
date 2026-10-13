# Plan — Hệ thống demo VietJobs: đoạn mô tả → nghề + mức lương


## 1. Mục tiêu
Demo không chỉ là một trang web. Nó trả lời ba câu hỏi, theo thứ tự:
1. **Có dựng được hệ thống end-to-end không?** Người dùng nhập một đoạn mô tả, hệ thống trả ra nghề (top-5, có xác suất) và mức lương dự đoán (triệu VND/tháng), chạy local rồi chạy trên VPS.
2. **Input cần gì để mô hình đoán đúng?** Mô hình học trên tin tuyển dụng đầy đủ (tiêu đề + mô tả + yêu cầu, dài, CafeBERT cắt ở 512 token). Một đoạn mô tả ngắn người dùng tự viết thì khác phân bố. Phải đo xem input ngắn hay thiếu tiêu đề làm mô hình tụt bao nhiêu, và cần tối thiểu những gì.
3. **Cải thiện hệ thống dựa trên số đo ở (2).** Ví dụ: gợi ý người dùng viết thêm, cảnh báo khi độ tin cậy thấp, hiển thị khoảng lương thay vì một con số, lượng tử hoá để chạy VPS rẻ.

**Ngoài phạm vi:** train lại mô hình, sửa `vietjobs/`, sửa `vietjobs/TASKS.md`.

## 2. Bối cảnh kỹ thuật (đã kiểm trong repo)
- **Mô hình:** `vietjobs/artifacts/dl-mtl-cafe-rnn-a5-s42/`, gồm `best.pt` (ngành, epoch 18), `best_sal.pt` (lương, epoch 4), `scaler.npz`, `config.json`, `metrics.json` (có `labels`, 16 lớp), `predictions_dev.parquet`.
- **Điểm trên dev (3.812 dòng):** macro-F1 0.606, accuracy 0.658, MAE lương 4.02 triệu, within ±20% 0.52.
- **Encoder `uitnlp/CafeBERT` bị đóng băng, không có trong checkpoint.** Phải tải từ HuggingFace và lấy hidden state từng token (1024 chiều, 512 token) vì head là RNN. Không mean-pool.
- **Kiến trúc head:** `MultiTaskRnn(in_dim=1024, hidden=100, n_classes=16, dropout=0.3, spatial_dropout=0.2, conv_channels=50, branches="both")` (`vietjobs/src/vietjobs/dl/heads.py:145`). `forward(x, mask)` trả `(logits_cat, sal_log)`.
- **Tiền xử lý lúc train:** mỗi trường qua `preprocess(s, tone=True, abbrev=True, mask=True)` (`vitext.py:557`, gọi trong `dataset.py:97`), không tách từ vì dùng CafeBERT. Ghép `title . description . requirements` với `SEP=" . "` (`dl/text.py:29`).
- **Chuẩn hoá:** `(x - mean) / std` lấy từ `scaler.npz`. Mean/std tính theo token có mask (`train_mtl.py:147`, `token_stats`). Lương ra ở dạng `log1p(triệu)`, phải `expm1`.
- **Môi trường có sẵn:** `.venv-dl` (Python 3.9, torch 2.2.2, transformers <4.47).

## 3. Kiến trúc project `VietJob/vietjobs-demo/`
```
vietjobs-demo/
  TASKS.md                 # board riêng (mục 6)
  README.md
  requirements.txt         # local (Mac): torch==2.2.2, transformers>=4.38,<4.47, numpy<2, sentencepiece, protobuf, gradio
  requirements-vps.txt     # Linux: torch không ghim
  model/                   # copy từ run: best.pt, best_sal.pt, scaler.npz, config.json, metrics.json
  data/abbreviations.json  # copy file config.ABBREVIATIONS trỏ tới
  data/dev_sample.parquet  # ~200 dòng dev + dự đoán gốc, dùng để kiểm và nghiên cứu input
  core/
    contract.py            # dataclass Prediction (hợp đồng giữa các luồng)
    textprep.py            # copy rút gọn vitext: normalize_unicode, normalize_tone, expand_abbreviations, mask_salary, preprocess
    heads.py               # copy BiGruLstmCnn + MultiTaskRnn
    predictor.py           # Predictor thật
    stub.py                # StubPredictor trả kết quả giả, để UI làm song song
  app/gradio_app.py        # UI
  api/main.py              # (luồng E) FastAPI
  study/                   # script + báo cáo nghiên cứu input (luồng C)
  deploy/                  # Dockerfile, nginx.conf, docker-compose.yml
  tests/                   # pytest
```

### Hợp đồng chung (`core/contract.py`) — chốt ở task A0, sau đó không đổi
```python
@dataclass
class Prediction:
    top: list[tuple[str, float]]   # 5 (nhãn, xác suất), giảm dần
    salary_trieu: float            # expm1(sal_log)
    clean_text: str                # text sau tiền xử lý (thấy <SALARY>)
    n_tokens: int                  # số token trước khi cắt
    truncated: bool
    seconds: float

class PredictorProtocol(Protocol):
    def predict(self, text: str) -> Prediction: ...
```
Mọi luồng chỉ phụ thuộc hợp đồng này. Nhờ vậy UI, API và deploy làm với `StubPredictor` trong khi luồng A làm `Predictor` thật.

### Pipeline suy luận (`Predictor.predict(text)`)
1. `clean = preprocess(text, tone=True, abbrev=True, mask=True)`, gộp khoảng trắng. Input là 1 đoạn duy nhất; hàm phụ `join_fields(title, desc, req)` chỉ dùng để kiểm với dev.
2. Tokenizer CafeBERT: `max_length=512, truncation=True`. Lấy `last_hidden_state` `[1, L, 1024]` và mask.
3. Chuẩn hoá bằng `scaler.npz`, kiểm shape lúc load.
4. Hai instance `MultiTaskRnn`: một nạp `best.pt` (lấy `logits_cat` → softmax → top-5), một nạp `best_sal.pt` (lấy `sal_log` → `expm1`). Cả hai `eval()`, `torch.no_grad()`.
5. Encoder và hai head chỉ nạp 1 lần, lúc khởi động.

## 4. Luồng công việc (độc lập nhau)
Sau A0, năm luồng A–E chạy song song. Mỗi luồng chỉ sửa thư mục của mình nên không ghi đè nhau.

| Luồng | Thư mục được sửa | Phụ thuộc |
|---|---|---|
| A — Lõi mô hình | `core/` (trừ `stub.py`), `model/`, `data/`, `tests/test_core*` | A0 |
| B — Giao diện Gradio | `app/`, `core/stub.py`, `tests/test_app*` | A0 (dùng stub) |
| C — Nghiên cứu input | `study/` | A2 xong (cần predictor thật) |
| D — Đóng gói và deploy | `deploy/`, `requirements-vps.txt` | A0 (build bằng stub, chạy thật sau A2) |
| E — API (tương lai) | `api/`, `tests/test_api*` | A0 |

Chỉ có hai điểm nối: `core/contract.py` (khoá sau A0) và một dòng chọn `Predictor` hay `StubPredictor` qua biến môi trường `VIETJOBS_STUB=1`.

## 5. Giai đoạn và cổng
- [x] **G1 — Hệ thống chạy được (local):** A1–A3, B1–B3 xong. Gõ một đoạn mô tả → ra nghề và lương, kết quả khớp dự đoán gốc trên dev.
- [x] **G2 — Biết input cần gì:** C1–C4 xong, có báo cáo `study/REPORT.md` kèm số đo.
- [ ] **G3 — Cải thiện:** chỉ làm các task C5/B4/A4 mà số đo ở G2 cho thấy đáng làm.
- [ ] **G4 — Trên VPS:** D1–D4 xong, truy cập qua HTTPS.

## 6. Bảng task (`vietjobs-demo/TASKS.md`)
Chú thích cột ✓: ✅ xong · 🔄 đang làm · ⬜ chưa làm · ❌ bỏ. Khi xong một task: đổi ⬜ → ✅, điền ngày, ghi 1 dòng log ở `vietjobs-demo/TASKS.md`.

Mỗi task nhỏ, có cổng kiểm cụ thể. Trạng thái: `todo` / `doing` / `done` / `dropped`. Cột "Ước lượng" là mức công sức tương đối (S/M/L).

### A — Lõi mô hình
| ✓ | ID | Task | Đầu ra | Cổng kiểm | Ước lượng | Ngày xong |
|---|---|---|---|---|---|---|
| ✅ | A0 | Khởi tạo: tạo thư mục, `git init`, `TASKS.md`, `core/contract.py`, `requirements.txt` | khung project | `python -c "import core.contract"` OK | S | 2026-10-06 |
| ✅ | A1 | Copy model + `heads.py` + `textprep.py` + `abbreviations.json`; trích `data/dev_sample.parquet` (200 dòng, có cột text gốc và dự đoán gốc) | file trong `core/`, `model/`, `data/` | load cả 2 state_dict vào `MultiTaskRnn` không thiếu hay thừa key | S | 2026-10-06 |
| ✅ | A2 | `Predictor` thật theo pipeline ở mục 3 | `core/predictor.py` | `tests/test_core_parity.py`: 200 dòng dev ghép bằng `join_fields`, nhãn top-1 khớp gốc ≥ 99%, lương lệch < 0.05 triệu | M | 2026-10-06 |
| ✅ | A3 | Đo hiệu năng local: thời gian nạp, giây/lần (L=64/256/512), RAM đỉnh | dòng log trong TASKS.md | có số cho cả 3 độ dài | S | 2026-10-06 |
| ⬜ | A4 | (G3, tuỳ chọn) int8 `quantize_dynamic` cho encoder | cờ `VIETJOBS_INT8=1` | macro-F1 trên 200 dòng tụt ≤ 0.01, MAE tăng ≤ 0.2 triệu; ghi tốc độ/RAM | M |  |

### B — Giao diện Gradio
| ✓ | ID | Task | Đầu ra | Cổng kiểm | Ước lượng | Ngày xong |
|---|---|---|---|---|---|---|
| ✅ | B1 | `StubPredictor` trả `Prediction` giả, chọn bằng biến `VIETJOBS_STUB` | `core/stub.py` | `predict("x")` trả đúng kiểu | S | 2026-10-06 |
| ✅ | B2 | UI: 1 `Textbox(lines=12)` cho đoạn mô tả, nút "Dự đoán", `gr.Label(num_top_classes=5)` cho nghề, ô lương (triệu/tháng), accordion "Văn bản sau xử lý", dòng số token/bị cắt/thời gian | `app/gradio_app.py` | chạy với stub tại 127.0.0.1:7860, đủ các ô | M | 2026-10-06 |
| ✅ | B3 | Nối predictor thật; 3 ví dụ mẫu (`gr.Examples`) lấy từ dev; hiển thị tên nghề dễ đọc (bỏ `_`); ghi chú điểm dev | cập nhật app | với predictor thật, nhập "lương 15 triệu" thấy `<SALARY>` trong text đã xử lý | S | 2026-10-06 |
| ⬜ | B4 | (G3) Hiển thị theo kết quả C: cảnh báo độ tin cậy thấp, gợi ý viết thêm, khoảng lương | cập nhật app | ngưỡng lấy đúng từ `study/REPORT.md` | S |  |

### C — Nghiên cứu input (để biết input cần gì)
Tất cả chạy trên `data/dev_sample.parquet`. Mỗi task ghi bảng số vào `study/REPORT.md`.

| ✓ | ID | Task | Câu hỏi | Cổng kiểm | Ước lượng | Ngày xong |
|---|---|---|---|---|---|---|
| ✅ | C1 | Biến thể trường: đủ 3 trường / chỉ tiêu đề / chỉ mô tả / mô tả + yêu cầu (không tiêu đề) | Thiếu trường nào thì tụt nhiều nhất? | bảng macro-F1, accuracy, MAE cho 4 biến thể | S | 2026-10-07 |
| ✅ | C2 | Độ dài: cắt input còn 20/50/100/200/512 token | Cần tối thiểu bao nhiêu chữ? | đường cong accuracy và MAE theo độ dài | S | 2026-10-07 |
| ✅ | C3 | Đoạn mô tả kiểu người dùng: viết tay 20–30 đoạn ngắn (2–4 câu) phủ đủ 16 nghề, gán nhãn và lương kỳ vọng | Input thật sự của demo cho kết quả thế nào? | file `study/handwritten.csv` + bảng kết quả | M | 2026-10-07 |
| ✅ | C4 | Độ tin cậy: xác suất top-1 so với đúng/sai trên C1–C3 | Có ngưỡng nào để cảnh báo người dùng không? | đề xuất ngưỡng kèm precision/coverage | S | 2026-10-07 |
| ⬜ | C5 | (G3) Thử các cải thiện chỉ ở tầng input: thêm tiền tố "Tuyển dụng", lặp tiêu đề, gợi ý mẫu điền | Cải thiện nào rẻ mà có tác dụng? | so với baseline C3 | M |  |

Kết luận của C quyết định B4 và C5. Không làm cải thiện nào chưa có số đo chứng minh.

### D — Đóng gói và deploy VPS
| ✓ | ID | Task | Đầu ra | Cổng kiểm | Ước lượng | Ngày xong |
|---|---|---|---|---|---|---|
| ✅ | D1 | `requirements-vps.txt` + `Dockerfile` (python:3.11-slim, torch CPU, tải sẵn CafeBERT lúc build bằng `snapshot_download`) | `deploy/Dockerfile` | `docker build` OK; `docker run -e VIETJOBS_STUB=1` mở được UI | M | 2026-10-06 |
| ⬜ | D2 | Chạy container với predictor thật, đo RAM đỉnh và giây/lần | dòng log | kết quả khớp A2 trên 5 ví dụ; có số RAM | S |  |
| ⬜ | D3 | Chốt cấu hình VPS theo D2 (dự kiến RAM ≥4 GB, nên 8 GB, ổ ~10 GB, không GPU); nếu RAM chỉ 2 GB thì cần A4 | ghi quyết định | số RAM đo ≤ 70% RAM VPS | S |  |
| ⬜ | D4 | Deploy: `docker-compose.yml`, Nginx reverse proxy, HTTPS (Let's Encrypt), `demo.queue(max_size=16)`, `launch(server_name="0.0.0.0")`, tự khởi động lại | `deploy/nginx.conf`, compose | mở được qua tên miền HTTPS; restart VPS xong app tự lên | M |  |

### E — API (tương lai)
| ✓ | ID | Task | Đầu ra | Cổng kiểm | Ước lượng | Ngày xong |
|---|---|---|---|---|---|---|
| ✅ | E1 | FastAPI `POST /predict {text}` → JSON theo `Prediction`; `GET /health` | `api/main.py` | `tests/test_api.py` với stub | S | 2026-10-06 |
| ⬜ | E2 | Frontend riêng gọi API (thay Gradio nếu cần) | quyết định sau G4 | — | L |  |

## 7. Theo dõi tiến độ
- `vietjobs-demo/TASKS.md` gồm 2 phần:
  1. **Bảng:** `ID | Luồng | Task | Phụ thuộc | Trạng thái | Cổng | Ngày xong`.
  2. **Log:** mỗi task xong ghi 1 dòng `YYYY-MM-DD ID — kết quả + số đo`.
- Mốc G1–G4 nằm đầu file kèm checkbox.
- Quy trình giống board chính: tác giả giao task, agent chỉ làm task được giao rồi cập nhật dòng + log.
- Commit ~100 dòng/commit, message bắt đầu bằng ID task (vd `A2: predictor parity`). Không push nếu chưa được yêu cầu.

## 8. Thứ tự đề xuất
1. **A0** (một mình, chốt hợp đồng).
2. Song song **A1→A2→A3**, **B1→B2**, **D1**, **E1**.
3. Sau A2: **B3**, **C1–C4**, **D2**. → đạt G1, G2.
4. **D3→D4** → đạt G4.
5. Theo kết quả C: **C5, B4, A4** → G3.

## 9. Kiểm tra end-to-end
1. `pytest tests/`: parity với dev (A2), stub (B1), API (E1).
2. `python app/gradio_app.py` → nhập 1 đoạn mô tả → thấy top-5 nghề, lương, text đã che lương.
3. `docker compose up` local → giống bước 2.
4. Trên VPS: `curl https://<tên-miền>/` trả 200; nhập 3 ví dụ mẫu ra kết quả giống local.

## 10. Trạng thái hiện tại (2026-10-07)
Bảng chi tiết và log: `vietjobs-demo/TASKS.md`.
- **Done (2026-10-06):** A0, A1, A2 (parity 100%, lương lệch max 0.0046), A3 (~0.3 s/đoạn CPU), B1, B2, B3, D1, E1.
- **Done (2026-10-07):** C1–C4 — xem `vietjobs-demo/study/REPORT.md`.
- **Todo:** C5, D2–D4, A4, B4, E2.
