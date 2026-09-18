[← Tài liệu tham khảo](08-tai-lieu-tham-khao.md) · [Bảng điều khiển](00-index.md)

# PHỤ LỤC

---

## Phụ lục A. Danh sách 16 nhóm ngành nghề

Tên lớp giữ nguyên như trong dữ liệu gốc (đây là **dữ liệu**, không phải văn bản báo
cáo, nên không dịch và không sửa chính tả).

> **Nguồn số liệu:** `artifacts/eda/summary.json`, hình `docs/figures/eda/eda-02a-lech-lop.png` và
> `eda-03a-co-lop.png` — đo trên **tệp gốc** sau khử trùng lặp, 47.707 dòng
> (`AGENTS.md` Rule 8). Bảng này không phụ thuộc lược đồ chia.

| Nhóm ngành nghề | Số tin | Tỷ lệ |
|---|---|---|
| `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` | 8.213 | 17,2 % |
| `sản_xuất_lao_động_phổ_thông_cơ_khí` | 6.341 | 13,3 % |
| `marketing_truyền_thông_quảng_cáo_nội_dung` | 5.947 | 12,5 % |
| `tài_chính_kế_toán_ngân_hàng_bảo_hiểm` | 5.454 | 11,4 % |
| `du_lịch_nhà_hàng_khách_sạn_dịch_vụ` | 4.198 | 8,8 % |
| `thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí` | 3.406 | 7,1 % |
| `nhân_sự_hành_chính_pháp_chế_tư_vấn` | 3.210 | 6,7 % |
| `xây_dựng_kiến_trúc_bất_động_sản` | 2.809 | 5,9 % |
| `công_nghệ_thông_tin_kỹ_thuật_số` | 1.903 | 4,0 % |
| `logistics_vận_tải_chuỗi_cung_ứng` | 1.811 | 3,8 % |
| `kỹ_thuật_điện_điện_tử_viễn_thông` | 1.236 | 2,6 % |
| `giáo_dục_đào_tạo_nghiên_cứu` | 1.165 | 2,4 % |
| `y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học` | 963 | 2,0 % |
| `ngôn_ngữ_dịch_thuật` | 384 | 0,8 % |
| `nhóm_nghề_khác` | 345 | 0,7 % |
| `nông_nghiệp_năng_lượng_môi_trường` | 322 | 0,7 % |
| **Tổng** | **47.707** | 100 % |

---

## Phụ lục B. Các lệnh tái lập kết quả

Mọi kết quả trong báo cáo tái lập được bằng các lệnh dưới đây, chạy theo đúng thứ tự.

### B.1. Chuẩn bị hai môi trường

```bash
# môi trường chính — dữ liệu, tiếng Việt, kiểm thử
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
python scripts/analyze_data.py              # số đo + 11 hình cho chương 3
PYTHONPATH=src .venv-dl/bin/python scripts/analyze_data.py --tokens   # thêm độ dài token (hình 07)
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

### B.4. Xuất hình cho bản Word

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

Bản đầy đủ nằm ở [docs/04-results.md](../04-results.md) (17 dòng, trục học sâu, bắt
đầu 2026-09-08). Báo cáo chỉ dẫn các dòng đo trên lược đồ chia hiện hành.

Khi ghép vào bản Word, chèn bảng vào phụ lục này dưới dạng bảng, hoặc dẫn
chiếu tới tệp nếu hội đồng chấp nhận phụ lục điện tử.

---

## Phụ lục E. Đối chiếu số liệu — bảng kiểm trước khi nộp

Mỗi con số trong quyển báo cáo phải tìm được ở đúng một trong các nguồn sau:

| Nguồn | Chứa gì |
|---|---|
| `data/processed/manifest.json` | Số dòng, số nhóm, cỡ từng tập, hạt giống, sha256 tệp thô |
| `artifacts/eda/summary.json` | Mọi con số của phân tích khám phá dữ liệu |
| `artifacts/<run_id>/metrics.json` | Bộ độ đo đầy đủ của một lần chạy |
| `docs/04-results.md` | Một dòng mỗi lần chạy — chỉ-thêm |

**Bảng kiểm cuối cùng trước khi nộp:**

- [ ] Không còn dấu `⛔` nào trong toàn bộ thư mục báo cáo
- [ ] Không còn dấu `✍️` nào
- [ ] Mọi khối `> **Nguồn số liệu:**` đã được xoá khỏi bản Word
- [ ] Các hình mermaid đã được render (`bash scripts/render_figures.sh`)
- [ ] Số thứ tự tài liệu tham khảo đã đánh lại theo thứ tự xuất hiện
- [ ] Mục lục, danh mục hình, danh mục bảng đã cập nhật tự động
- [ ] `pytest -q` xanh
- [ ] `test` được chạm đúng **một lần**, ở bước cuối cùng, có cờ `--confirm-test`

---

[← Tài liệu tham khảo](08-tai-lieu-tham-khao.md) · [Bảng điều khiển](00-index.md)
