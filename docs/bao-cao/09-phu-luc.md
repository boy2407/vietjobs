[← Tài liệu tham khảo](08-tai-lieu-tham-khao.md) · [Bảng điều khiển](00-index.md)

# PHỤ LỤC

---

## Phụ lục A. Danh sách 16 nhóm ngành nghề

Tên lớp giữ nguyên như trong dữ liệu gốc (đây là **dữ liệu**, không phải văn bản báo
cáo, nên không dịch và không sửa chính tả).

> ⛔ **CHƯA CÓ SỐ LIỆU** — bảng đầy đủ 16 dòng phải sinh lại từ
> `artifacts/eda/summary.json` (`class_counts`) sau khi chạy lại
> `scripts/analyze_data.py` trên lược đồ chia hiện tại. Bốn dòng đã biết:

| Nhóm ngành nghề | Số tin (train, lược đồ v1) | Tỷ lệ |
|---|---|---|
| `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` | 5.330 | 16,0 % |
| `nông_nghiệp_năng_lượng_môi_trường` | 197 | 0,6 % |
| `nhóm_nghề_khác` | 250 | 0,7 % |
| … 13 nhóm còn lại | | |

---

## Phụ lục B. Các lệnh tái lập kết quả

Mọi kết quả trong báo cáo tái lập được bằng các lệnh dưới đây, chạy theo đúng thứ tự.

### B.1. Chuẩn bị hai môi trường

```bash
# môi trường chính — dữ liệu, tiếng Việt, học máy, kiểm thử
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# môi trường học sâu — tách riêng vì ràng buộc phiên bản torch
/usr/bin/python3 -m venv .venv-dl
.venv-dl/bin/pip install -r requirements-dl.txt
```

### B.2. Dựng lại dữ liệu và phân tích

```bash
pytest -q                                   # phải xanh trước mọi thứ khác
python -m vietjobs.dataset build            # dựng lại ba tập (hiếm khi cần chạy)
python scripts/analyze_data.py              # số đo + 5 hình cho chương 3
python scripts/measure_vitext.py            # đo lại bảng chín bước
```

### B.3. Huấn luyện mô hình học sâu

```bash
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode   --task category --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode   --task salary   --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --class-weight
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary
PYTHONPATH=src .venv-dl/bin/python scripts/probe_embeddings.py --task category
```

### B.4. Chạy lại mốc học máy

```bash
python -m vietjobs.train --task category --model svm --C 0.02 --province
```

### B.5. Xuất hình cho bản Word

```bash
./scripts/render_figures.sh     # cần Node và Chrome
```

---

## Phụ lục C. Cấu trúc một thư mục kết quả

Mỗi lần chạy sinh ra `artifacts/<run_id>/` gồm:

| Tệp | Nội dung |
|---|---|
| `config.json` | mọi siêu tham số + hạt giống |
| `env.json` | thiết bị, phiên bản thư viện, encoder |
| `history.jsonl` | một dòng mỗi epoch: mất mát, độ đo dev, số giây |
| `metrics.json` | bộ độ đo đầy đủ của lần chạy |
| `best.pt` | trọng số tại epoch tốt nhất trên dev |
| `scaler.npz` | thống kê chuẩn hoá của tập train — **đường suy luận phải dùng đúng tệp này** |
| `predictions_dev.parquet` | dự đoán từng dòng, để đọc lỗi sau khi chạy xong |

Kèm theo là **đúng một dòng** thêm vào `docs/04-results.md`. Tệp đó là **chỉ-thêm**:
một dòng đã viết không bao giờ được sửa, kể cả khi lần chạy đó thất bại.

---

## Phụ lục D. Nhật ký thí nghiệm đầy đủ

Bản đầy đủ nằm ở [docs/04-results.md](../04-results.md) (10 dòng, trục học sâu, bắt
đầu 2026-09-08) và [docs/archive/04-results-ml.md](../archive/04-results-ml.md)
(101 dòng, trục học máy, 2026-08-26 → 09-07).

Khi ghép vào bản Word, chèn cả hai bảng vào phụ lục này dưới dạng bảng, hoặc dẫn
chiếu tới tệp nếu hội đồng chấp nhận phụ lục điện tử.

---

## Phụ lục E. Kế hoạch thực hiện

**Bảng E.1: Kế hoạch 15 tuần**

| STT | Nội dung công việc | Thời gian | Kết quả đạt được | Trạng thái |
|---|---|---|---|---|
| 1 | Khảo sát bài toán, thu thập và xử lý dữ liệu tin tuyển dụng | Tuần 1–2 | Bộ dữ liệu đã chuẩn hoá và chia ba tập | **xong** |
| 2 | Biểu diễn văn bản bằng mô hình ngôn ngữ tiếng Việt và xây nhánh phân loại | Tuần 3–4 | Mô hình phân loại chạy ổn định, vượt các mốc cơ sở | **cần chạy lại** trên lược đồ mới |
| 3 | Bổ sung nhánh ước lượng lương, huấn luyện đa nhiệm | Tuần 5–6 | Mô hình đa nhiệm cho ra ngành nghề và mức lương | nhánh lương **xong**; đa nhiệm **chưa** |
| 4 | Phân tích lỗi và cải tiến mô hình | Tuần 7–9 | Báo cáo nguyên nhân dự đoán sai và kết quả sau cải tiến | phân tích **xong**, cải tiến **chưa** |
| 5 | So sánh học sâu với học máy truyền thống | Tuần 10 | Bảng so sánh độ chính xác và tốc độ xử lý | thiếu **bảng độ trễ suy luận** |
| 6 | Xây dựng hệ thống dự đoán và giao diện | Tuần 11–12 | Hệ thống nhận tin và trả về ngành nghề, mức lương | **chưa bắt đầu** |
| 7 | Hoàn thiện chức năng và kiểm thử hệ thống | Tuần 13 | Hệ thống chạy ổn định, có kết quả kiểm thử | **chưa bắt đầu** |
| 8 | Đánh giá tổng thể trên tập kiểm tra | Tuần 14 | Kết quả đánh giá cuối cùng của hai bài toán | **chưa** — `test` chỉ được chạm một lần |
| 9 | Viết báo cáo và chuẩn bị bảo vệ | Tuần 15 | Quyển báo cáo, slide, bản demo | đang làm liên tục |

> ✍️ **CẦN VIẾT TAY** — ô "Thời gian thực hiện" trong đề cương (ngày bắt đầu và ngày
> kết thúc) vẫn còn để trống. Điền vào đề cương và đồng bộ mốc tuần ở bảng trên.

---

## Phụ lục F. Đối chiếu số liệu — bảng kiểm trước khi nộp

Mỗi con số trong quyển báo cáo phải tìm được ở đúng một trong các nguồn sau:

| Nguồn | Chứa gì |
|---|---|
| `data/processed/manifest.json` | Số dòng, số nhóm, cỡ từng tập, hạt giống, sha256 tệp thô |
| `artifacts/eda/summary.json` | Mọi con số của phân tích khám phá dữ liệu |
| `artifacts/<run_id>/metrics.json` | Bộ độ đo đầy đủ của một lần chạy |
| `docs/04-results.md` | Một dòng mỗi lần chạy — chỉ-thêm |
| `docs/archive/04-results-ml.md` | 101 dòng của trục học máy đã đóng |

**Bảng kiểm cuối cùng trước khi nộp:**

- [ ] Không còn dấu `⛔` nào trong toàn bộ thư mục báo cáo
- [ ] Không còn dấu `✍️` nào
- [ ] Mọi khối `> **Nguồn số liệu:**` đã được xoá khỏi bản Word
- [ ] Ba hình `04`, `05`, `06` đã được render
- [ ] Số thứ tự tài liệu tham khảo đã đánh lại theo thứ tự xuất hiện
- [ ] Mục lục, danh mục hình, danh mục bảng đã cập nhật tự động
- [ ] `pytest -q` xanh
- [ ] `test` được chạm đúng **một lần**, ở bước cuối cùng, có cờ `--confirm-test`

---

[← Tài liệu tham khảo](08-tai-lieu-tham-khao.md) · [Bảng điều khiển](00-index.md)
