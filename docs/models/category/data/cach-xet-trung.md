# Cách xét trùng của bộ dữ liệu `category`

Ghi lại các bước đã chạy để từ file gốc đến bộ chia hiện tại (`v8-minhash-1label`). Mọi số lấy từ `data/processed/manifest.json`, `data/processed/splits-minhash-1label/manifest.json`, `artifacts/eda/minhash-all.json` và các báo cáo trong thư mục này. Phạm vi: file gốc `data/raw/VietJobs.csv`.

## Quy trình hiện tại

| Bước | Việc làm | Cách xét trùng | Số còn lại |
|---|---|---|---|
| 0 | File gốc | — | 48.092 dòng |
| 1 | **Lọc thô** (`dataset.clean`, `drop_duplicates()`) | Hai dòng giống hệt ở **cả 18 cột, kể cả nhãn**. Chỉ bỏ bản sao thật, không mất nhãn nào | 47.707 dòng (bớt 385) |
| 2 | **MinHash** trên cả 47.707 dòng | Văn bản = `job_title . description . requirements_text` sau NFC, sửa dấu thanh, mở rộng viết tắt, đưa về chữ thường. Shingle 5 ký tự, 128 hoán vị, ngưỡng Jaccard 0,9. Các cặp giống nhau nối bắc cầu thành cụm | 33.052 cụm |
| 3 | **Bỏ cụm nhiều nhãn** | Đếm nhãn trong từng cụm trên tất cả các dòng của cụm, rồi bỏ hẳn cụm có từ 2 nhãn trở lên (8.960 cụm, 22.126 dòng). Mỗi cụm còn lại giữ dòng đầu trong file gốc | **24.092 tin** |
| 4 | **Chia tập** | `dataset.two_stage_split`, `SPLIT_SEED = 20260826`: train:test = 8:2, rồi train:dev = 9:1. Mỗi cụm là một đơn vị | train 17.350 · dev 1.927 · test 4.815 |

Bộ chia nằm ở `data/processed/splits-minhash-1label/`, giữ 18 cột gốc với tiền tố `raw_`. Bộ cũ `data/processed/splits/` giữ nguyên.

Lệnh dựng lại:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py --minhash --drop-multi
```

`datasketch` cài trong `.venv`, chưa thêm vào `requirements.txt`.

## Thứ tự đếm nhãn và bỏ dòng

Bước 2 chỉ đọc văn bản, không xoá dòng nào. Bước 3 đếm nhãn trước khi bỏ dòng, nên không có nhãn nào biến mất trước khi được đếm. Lọc thô ở bước 1 an toàn vì nhãn nằm trong 18 cột được so sánh: hai dòng khác nhãn không bị coi là bản sao. Nếu lọc theo khoá văn bản trước (như các lần thử cũ), các dòng khác nhãn bị bỏ ngay và MinHash không còn thấy chúng.

## Số đo của MinHash trên 47.707 dòng

| Số đo | Giá trị |
|---|---|
| Số cặp gần-trùng | 25.467 |
| Số cụm | 33.052 |
| Cụm có ≥ 2 dòng · ≥ 5 dòng | 10.294 · 243 |
| Cụm có ≥ 2 nhãn | 8.960 (22.126 dòng) |
| Cụm lớn nhất | 100 dòng |

Bốn cụm lớn nhất (100, 27, 23, 21 dòng) được đọc tay: mỗi cụm là một tin đăng lặp (cùng mô tả, cùng nhà tuyển dụng, tiêu đề khác vài chữ), không phải chuỗi gộp nhầm. Các cụm còn lại chưa đọc.

## Kết quả kiểm tra rò rỉ giữa các tập

| Kiểm tra | `v8-minhash-1label` |
|---|---|
| Tin trùng hệt ở ≥ 2 tập | 0 |
| Cặp MinHash gần-trùng (Jaccard ≥ 0,9) ở hai tập khác nhau | 0 |
| Cụm còn ≥ 2 nhãn | 0 |
| Phân bố nhãn dev và test so với train (PSI) | 0,000 |

## Các lần thử trước đây, đã bỏ

Trước MinHash, ba bước khoá văn bản (chữ thường, bỏ dấu câu, bỏ khoảng trắng) được thử thay cho MinHash. Chúng được giữ ở đây làm bằng chứng vì sao bị bỏ.

| Phiên bản | Cách xét trùng | Số tin | Rò rỉ giữa các tập |
|---|---|---|---|
| `splits-dedup` | Chuỗi BERT đọc, giữ hoa/thường | 36.798 | 15 cặp chỉ khác hoa/thường; 754 khoá khác dấu câu |
| `splits-dedup-ci` | Bỏ qua hoa/thường | 36.753 | 737 khoá |
| `splits-dedup-norm` | Bỏ cả dấu câu và khoảng trắng | 34.742 | 0 theo khoá, nhưng xem dòng dưới |
| `splits-dedup-norm-1label` (`v7`) | Như trên, bỏ tin nhiều nhãn | 26.146 | **664 cặp MinHash** gần-trùng ở hai tập (test–train 456, dev–train 155, dev–test 53) |

Khoá văn bản chỉ bắt được hai tin giống hệt sau chuẩn hoá. Tin chỉ khác vài chữ (thêm "- Sales Manager" vào tiêu đề, thêm "Mô tả công việc:" ở đầu) lọt qua, và chỉ 38,1 % các cặp này cùng nhãn. Đo trên cả 47.707 dòng: khoá cho 34.742 tin, MinHash cho 33.052 tin; sau khi bỏ tin nhiều nhãn, khoá còn 26.146 và MinHash còn 24.092.

## Hạn chế

- **Mất tin mơ hồ.** Các lớp mất nhiều dòng nhất so với bộ 34.742 tin: `du_lịch…` 57,6 %, `nông_nghiệp…` 51,9 %, `giáo_dục…` 46,1 %, `xây_dựng…` 41,1 %, `marketing…` 36,9 %. Bộ dữ liệu nghiêng về tin không mơ hồ, nên điểm đo cao hơn thực tế ở tin khó.
- **Ngưỡng 0,9** là mặc định của công cụ `nm-data`, chưa kiểm chứng riêng cho dữ liệu này. Một số nguồn gián tiếp nhắc tới 0,8.
- **Gộp bắc cầu** có thể nối hai tin không giống nhau qua một tin ở giữa. Mới đọc bốn cụm lớn nhất.
- **MinHash là xấp xỉ** của độ giống Jaccard, không phải phép đo chính xác.
- **Nguồn tài liệu.** Ý tưởng đo độ giống bằng tập các đoạn nhỏ và lấy mẫu là của Broder (1997), đã đọc trang đầu bài gốc. Lee và cộng sự (ACL 2022) và Allamanis (2019) mới đọc phần tóm tắt trên arXiv; cần đọc toàn bài trước khi dẫn vào luận văn (Rule 7). Chưa có nguồn nào cho thứ tự lọc thô rồi MinHash hay cho việc bỏ cụm nhiều nhãn: đó là lựa chọn của dự án.
- Phép đếm trùng giữa các tập đọc cả `test` của bộ mới, chỉ để đếm, không xem mẫu nào.
- Bộ chia mới chưa có run mô hình nào, nên mọi dòng trong `docs/04-results.md` không so sánh được với bộ này (Rule 2).
