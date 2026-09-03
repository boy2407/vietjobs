# Quy tắc làm việc — VietJobs

## Quy tắc số 1 — luôn cập nhật tài liệu

`docs/` là **bản đồ sống** của dự án, không phải tài liệu viết một lần.
`docs/00-tong-quan.md` là trang chủ; mỗi chủ đề có một note riêng.
Sau **mỗi** thay đổi có thật, cập nhật ngay trong cùng lượt làm việc:

| Vừa làm gì | Phải sửa file nào |
|---|---|
| Chạy xong một thí nghiệm mới | `04-results.md` (tự động, chỉ ghi thêm) + bảng bậc thang trong `06-mo-hinh-phan-lop.md` + ô trạng thái trong `00-tong-quan.md` |
| Đổi bước tiền xử lý | `02-vietnamese-nlp.md` — bảng chín bước, bảng ablation, sơ đồ (nét đứt cho bước đã gỡ) |
| Đổi cách làm sạch / chia tập | `01-data-audit.md` |
| Đổi cách dựng đặc trưng (`features.py`) | `05-dac-trung-tfidf.md` |
| Chạy xong bài toán lương | `07-bai-toan-luong.md` + `00-tong-quan.md` (ô cam → xanh) |
| Thêm/sửa/xoá module trong `src/` | `08-ma-nguon.md` — sơ đồ + bảng vai trò file + số dòng/số test |
| Chạy lại cụm quét so sánh mô hình | `10-so-sanh-mo-hinh.md` (bảy bảng, sinh bằng `scripts/report_sweep.py`) + bậc thang trong `06-mo-hinh-phan-lop.md` |
| Xong một hạng mục | Bảng trạng thái `00-tong-quan.md` + `09-lo-trinh.md` |
| Đổi kiến trúc mô hình / hệ thống | Sơ đồ mermaid trong note tương ứng |

Ngoại lệ: **`docs/nen-tang/` không chứa con số kết quả.** Đó là track giải thích
khái niệm cho người mới, và là lý do nó nằm riêng — không phải cập nhật sau mỗi
thí nghiệm. Chỉ sửa khi cách làm thay đổi về bản chất.

Thêm hoặc đổi tên note thì phải sửa **ba** chỗ: mục lục trong `00-tong-quan.md`,
bảng này, và mảng `SOURCES` trong `scripts/render_figures.sh` (nó khớp sơ đồ
mermaid theo vị trí và sẽ thoát nếu lệch số lượng).

**Không được để tài liệu lệch với thực tế.** Nếu sơ đồ nói "chưa chạy" mà
`04-results.md` đã có dòng kết quả, tài liệu đó đang nói dối người đọc.

### Nguyên tắc viết tài liệu

- **Mọi con số phải đo được**, kèm nguồn: `metrics.json`, `manifest.json`, hoặc
  một dòng trong `04-results.md`. Không ước lượng, không "khoảng".
- **Nói rõ từng bước xây dựng** — đây là mục tiêu chính của dự án. Người đọc phải
  hiểu được *tại sao* làm bước đó, *đo bằng gì*, và *kết quả đo ra sao*.
- **Ghi cả thất bại.** Bước nào đo rồi thấy không giúp thì để lại trong bảng với
  kết luận "bỏ" — đó là bằng chứng, không phải rác.
- Tiếng Việt, câu ngắn, một ý một câu. Dấu phẩy thập phân (0,6112).

## Quy tắc số 2 — giao thức thí nghiệm

Chi tiết ở [docs/03-protocol.md](docs/03-protocol.md). Tóm tắt bốn điều tuyệt đối:

1. **Không đổi** `SPLIT_SEED = 20260826` và tỷ lệ chia. Đổi là mọi dòng trong
   `04-results.md` mất khả năng so sánh.
2. **`test` chạm đúng một lần**, ở cuối, có cờ `--confirm-test`. Chọn mô hình dùng `val`.
3. **`04-results.md` chỉ ghi thêm** (append-only). Không sửa dòng đã ghi.
4. **Bước tiền xử lý nào không có dòng ablation chứng minh thì gỡ bỏ.**

## Quy tắc số 3 — chống rò rỉ

Hai bài toán lương chỉ được đọc cột `*_masked`. `features.resolve_column` là nơi
duy nhất quyết định điều đó, `tests/test_no_leak.py` giữ nó. Rò rỉ ở đây **không
báo lỗi** — nó lặng lẽ tạo ra mô hình đọc chính đáp án của mình.

Cảnh báo: `UNMASKED_COLUMNS` là danh sách **liệt kê thủ công**, nên test chỉ canh
được những cột người viết test nghĩ tới. Đã biết `soft_skills_text` và
`qualifications_text` lọt lưới — xem `docs/09-lo-trinh.md` Ưu tiên 0b. Thêm cột
văn bản mới thì phải cập nhật `_MASKABLE` **và** `UNMASKED_COLUMNS`.

## Quy tắc số 4 — huấn luyện và suy luận dùng chung một đường code

`predict.build_frame` phải cho ra đúng bố cục cột mà `dataset.clean` đã tạo.
Lệch nhau là bug im lặng, chỉ lộ ra bằng dự đoán kém dần.

## Lệnh hay dùng

```bash
python -m vietjobs.dataset build                    # dựng lại splits (hiếm khi cần)
python -m vietjobs.train --task category --model svm --C 0.02 --province
python -m vietjobs.train --task disclosed --model logreg --province
python -m vietjobs.train --task salary --model lgbm --province
python -m vietjobs.predict --title "..." --description "..."
pytest -q
```
