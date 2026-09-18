[← Chương 4](05-chuong-4-thuc-nghiem.md) · [Bảng điều khiển](00-index.md) · [Chương 6 →](07-chuong-6-ket-luan.md)

# CHƯƠNG 5. CÀI ĐẶT MÃ NGUỒN THỰC NGHIỆM

---

## 5.1. Các công nghệ sử dụng

**Bảng 5.1: Công nghệ và thư viện đã dùng**

| Nhóm | Công nghệ | Vai trò trong đề tài |
|---|---|---|
| Ngôn ngữ | Python | Toàn bộ mã nguồn |
| Xử lý dữ liệu | pandas · NumPy · PyArrow | Đọc CSV, dựng cột dẫn xuất, ghi parquet |
| Xử lý tiếng Việt | `underthesea` | Tách từ tiếng Việt |
| Mô hình ngôn ngữ | `transformers` · PhoBERT-base-v2 | Sinh vectơ ngữ nghĩa 768 chiều |
| Học sâu | PyTorch | Khối kết nối đầy đủ và vòng huấn luyện |
| Học máy | scikit-learn | Các phép dò tuyến tính (mốc tầng 2) |
| Kiểm thử | pytest | 107 kiểm thử, trong đó 30 kiểm thử canh rò rỉ dữ liệu |

**Bảng 5.2: Hai môi trường ảo và lý do tách riêng**

| Môi trường | Python | Chứa gì | Lý do tách |
|---|---|---|---|
| `.venv` | ≥ 3.10 | dữ liệu, tiếng Việt, kiểm thử | Môi trường chính |
| `.venv-dl` | 3.9 | gói `dl/`, PyTorch, transformers | `torch` không có bản dựng cho macOS x86_64 sau 2.2.2, và không có bản nào cho Python 3.14 |

Ràng buộc này định hình một quyết định thiết kế: **`dl/text.py` không được phép
import torch**. Nhờ đó bộ kiểm thử đường dữ liệu — phần canh rò rỉ lương — vẫn chạy
được ở môi trường chính, nơi không cài torch.

---

## 5.2. Kiến trúc mã nguồn

![Sơ đồ mã nguồn](../figures/06-ma-nguon.png)

**Hình 5.1: Quan hệ phụ thuộc giữa các mô-đun**

Phần mã nguồn trong `src/vietjobs/` mà đề tài dùng gồm **10 mô-đun (không tính `__init__.py`) · 2.379 dòng · 107 kiểm thử** (`wc -l`, `pytest --collect-only`, 17/09/2026).

**Bảng 5.3: Vai trò từng mô-đun**

| Tệp | Dòng | Vai trò |
|---|---|---|
| `vitext.py` | 584 | Toàn bộ xử lý tiếng Việt. Hàm thuần tuý, mỗi bước có kiểm thử riêng |
| `dataset.py` | 381 | Làm sạch + chia tập cố định. Tách từ chạy **một lần** và ghi vào parquet |
| `features.py` | 279 | `resolve_column` — cửa duy nhất quyết định bài toán nào đọc cột nào |
| `evaluate.py` | 267 | Độ đo cho cả ba bài toán + bootstrap và so sánh ghép cặp |
| `config.py` | 85 | Đường dẫn · `SPLIT_SEED` · nhóm cột · tên bài toán |
| `dl/text.py` | 40 | Nối ba trường văn bản thành đầu vào PhoBERT. Không phụ thuộc torch |
| `dl/encode.py` | 153 | PhoBERT đóng băng → vectơ 768 chiều, lưu đệm `.npy` theo họ `raw`/`masked` |
| `dl/heads.py` | 37 | Phần dày đặc: 768 → h → h/2 → out |
| `dl/train_dl.py` | 275 | Một lần chạy = một dòng trong `04-results.md` + `history.jsonl` từng epoch |
| `external.py` | 278 | Đọc VietJobs-37K, ánh xạ 60 → 16 nhãn, lọc tin trùng với ba tập chia (§4.8) |

### 5.2.1. Ba bất biến giữ cả hệ thống đứng vững

**1. Một cánh cửa vào dữ liệu.** Đường học sâu không tự nối vào cột thô: nó gọi
`features.resolve_column` thông qua `dl/text.py`. Nhờ vậy quy tắc chống rò rỉ lương chỉ phải canh ở một chỗ.

**2. `vitext.py` hoàn toàn là hàm thuần tuý.** Cùng đầu vào cho cùng đầu ra, không
đọc trạng thái ngoài, không học gì từ dữ liệu. Đó là thứ cho phép `dataset.py` và
`features.py` gọi cùng một đoạn mã mà không phân kỳ.

**3. Chỉ có một nơi quyết định bài toán nào đọc cột nào** — `features.resolve_column`.

---

## 5.3. Kiểm thử

**Bảng 5.4: Bộ kiểm thử (107 kiểm thử)**

| Tệp | Số kiểm thử | Canh điều gì |
|---|---|---|
| `tests/test_vitext.py` | 56 | Từng bước xử lý tiếng Việt, gồm bốn trường hợp "40 triệu người dùng" **không được** che |
| `tests/test_no_leak.py` | 22 | Hai bài toán lương không bao giờ đọc cột chưa che; mọi tên trong danh sách bảo vệ phải là cột có thật |
| `tests/test_dl_text.py` | 8 | Đường học sâu đọc đúng cột: bài toán lương chỉ thấy `*_masked`; đầu vào không bị lọc stopword, không bị viết thường |
| `tests/test_external.py` | 13 | Tách chuỗi VietJobs-37K đúng thứ tự cột; bảng ánh xạ phủ đủ 60 nhãn; tin trùng bị bắt cả khi văn bản đã bị che tên công ty |
| `tests/test_bootstrap.py` | 8 | Bootstrap tất định theo hạt giống; hai mô hình giống hệt hoà ở 0,5 |

---

## 5.4. Nhận xét

Phần **cơ sở hạ tầng thực nghiệm** của đề tài có bằng chứng: 10 mô-đun, 107 kiểm
thử xanh, ba bất biến chống rò rỉ được canh bằng 30 kiểm thử, mọi lần chạy đều ghi lại
đủ để tái lập (§4.1.3).

---

[← Chương 4](05-chuong-4-thuc-nghiem.md) · [Chương 6 →](07-chuong-6-ket-luan.md)
