[← Tổng quan](00-tong-quan.md) · [← Mốc cơ sở học sâu](06-baseline-dl.md)

# Lộ trình

Nhánh học máy đã đóng vào ngày 2026-09-08 và chuyển vào
[archive/](archive/README.md). Từ đây, nhánh chính của khóa luận là **mạng học sâu
trên biểu diễn PhoBERT**, theo thứ tự: hai bài toán tách riêng
trước, hợp nhất đa nhiệm sau.

Các mốc cần vượt, đo trên cùng tập chia dữ liệu với cùng seed:

| Bài toán | Mốc | Nguồn |
|---|---|---|
| Phân loại ngành nghề | macro-F1 trên test **0,6112** (LinearSVC + TF-IDF) | [archive/04-results-ml.md](archive/04-results-ml.md) |
| Phân loại ngành nghề — sàn | đa số **0,0210** · quy tắc từ khóa **0,4321** | [archive/06-mo-hinh-phan-lop.md](archive/06-mo-hinh-phan-lop.md) |
| Ước lượng lương | MAE trên dev **5,70 triệu** (dự đoán trung vị) | [05 §7](05-phan-tich-du-lieu.md#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev) |
| Có công bố lương hay không | accuracy **0,7078** (luôn đoán "có") | nguồn như trên |

---

## ~~Ưu tiên 1~~ — hai mốc cơ sở học sâu tách riêng · **XONG**

PhoBERT đóng băng → dense → một đầu, mỗi bài toán một mạng riêng. Kiến trúc và kết
quả nằm trong [06-baseline-dl.md](06-baseline-dl.md).

- [x] Phân tích dữ liệu, kèm hình và số liệu ([05](05-phan-tich-du-lieu.md))
- [x] Đường dữ liệu học sâu đi qua `features.resolve_column`, có kiểm thử canh rò rỉ
- [x] Vectơ PhoBERT được lưu đệm ra đĩa, tách theo hai họ `raw` / `masked`
- [x] Mốc cơ sở phân loại — `dl-cat-v2`, macro-F1 trên dev **0,5987** (mốc cũ 0,6050 ± 0,0071)
- [x] Mốc cơ sở hồi quy lương — `dl-sal-v2`, MAE trên dev **4,83 triệu** (mốc 5,86)
- [x] Dò tuyến tính làm mốc chẩn đoán — `probe-cat` 0,5867 · `probe-sal` 6,60

Ba việc kết quả này sinh ra, sắp theo giá trị trên mỗi giờ công bỏ ra:

| Việc | Vì sao | Chi phí |
|---|---|---|
| Đầu ra lương **không dám đi ra dải đuôi phải** — dự đoán tối đa 80,7 trong khi dữ liệu thực lên tới 275 | MAE ở dải trên 30 triệu là **25,58** so với 3,70 ở phần còn lại | thử hàm mất mát bất đối xứng, hoặc chuyển sang dự đoán phân vị |
| Lớp `nhóm_nghề_khác` có F1 = **0,000** | 48 mẫu, nội dung lẫn lộn — nhiều khả năng nên gộp hoặc bỏ, không nên cố học | một quyết định về gán nhãn, không phải về mô hình |
| Ba cặp nhầm lẫn lớn nhất là ranh giới nhãn mờ (`kinh_doanh` ↔ `du_lịch_nhà_hàng`, 382 tin nhầm cả hai chiều) | trần nhiễu nhãn chưa từng được đo | lấy mẫu 100 tin, gán nhãn tay, đo độ đồng thuận |

## Ưu tiên 2 — tinh chỉnh PhoBERT (mức cũ 4b)

Rã đông trọng số encoder, huấn luyện đầu-cuối, 256 token. Chỉ làm **sau khi** mức
đóng băng đã có số: nếu bản đóng băng không vượt nổi 0,6112, lỗi nhiều khả năng nằm
ở đường dữ liệu, và phát hiện điều đó ở mức rẻ tiền chỉ tốn vài phút thay vì hàng
giờ.

Đây là hạng mục còn lại có kỳ vọng cao nhất: bản đóng băng đã ngang mốc TF-IDF **mà
chưa dùng đến khả năng thích nghi của encoder chút nào**. Ràng buộc phần cứng đã đo
được: máy này chạy Intel x86_64, **không có MPS/CUDA**, và `torch` không còn phát
hành bản dựng cho macOS Intel sau bản 2.2.2. Nhúng ở trạng thái đóng băng cho tập
`train` v1 (33.396 tin) mất ~20 phút với tốc độ ~30 tin/giây; tập `train` v2 có
34.354 tin. Tinh chỉnh toàn phần trên máy này không khả thi — cần một GPU trên
Colab/Kaggle, và khi đó phải chép `data/processed/splits/` sang đó cùng với
`manifest.json` để giữ nguyên phép chia.

## Ưu tiên 3 — hợp nhất đa nhiệm

```
PhoBERT → shared dense → head A: 16 classes   (cross-entropy)
                       → head B: salary       (huber, masked)
loss = w_A · loss_A + w_B · loss_B
```

Hai ràng buộc không được vi phạm:

1. **`loss_B` phải được che.** 28,5 % tin không có nhãn lương; với các tin đó,
   `loss_B` bằng 0, chứ không phải nhãn có giá trị 0.
2. **Nhánh lương vẫn phải đọc các cột `*_masked`.** Khi hai đầu dùng chung một
   thân, cái thân đó buộc phải đọc bản đã che — nghĩa là bản đa nhiệm chạy trên
   họ cột `masked`, và cái giá của việc đó (đầu phân loại mất phần văn bản chứa số
   liệu lương) là một con số phải đo, không phải một giả định.

Kỳ vọng nên đặt thấp: ngành nghề chỉ giải thích được **3,2 %** phương sai của
lương trên thang log
([05 §8](05-phan-tich-du-lieu.md#8-phát-hiện-quan-trọng-nhất-ngành-nghề-hầu-như-không-nói-lên-điều-gì-về-mức-lương)),
nên câu chuyện "hai đầu hỗ trợ nhau" gần như không có gì để kể. Giá trị của bản đa
nhiệm vụ nhiều khả năng nằm ở **một mô hình duy nhất thay vì hai**, chứ không phải
ở điểm số.

## Ưu tiên 4 — dữ liệu, các việc mà phân tích đã chỉ ra

- [ ] Chạy lại `scripts/analyze_data.py` trên tệp gốc để `artifacts/eda/summary.json`
      không còn lệch với các hình, và thêm `--tokens` cho hình 07 — cái giá của việc
      cắt ở 256 token chưa từng được đo
- [ ] Các trường hợp biên của lương: 15 tin > 200 triệu, cộng nhóm dưới 2 triệu
      (số lượng chờ chạy lại để có)
- [ ] Đo thiên lệch chọn mẫu: tỷ lệ công bố lương trải từ 58,6 % → 76,5 % tùy ngành
- [x] **Đánh giá chéo bài `category` trên tập ngoài VietJobs-37K** — chạy ngày
      2026-09-17 (T7.3): Gold-1000 strict macro-F1 0,3977 so với 0,6025 trên `dev`.
      Còn chờ người duyệt bảng ánh xạ 60 → 16 (T7.1) trước khi thành số chính thức.
      Xem [10](10-danh-gia-ngoai.md)
- [ ] Kiểm tra chéo bài `salary` — 37K không có lương, nên làm theo nguồn (topcv ↔
      careerviet) hoặc theo thời gian ngay trong `VietJobs.csv`, chỉ chia lại
      `train`/`dev` ([10 §6](10-danh-gia-ngoai.md#6-còn-thiếu-kiểm-tra-chéo-cho-bài-lương))
- [ ] Hướng nghiên cứu ngành–lương cho luận văn: (a) R² log của mô hình chỉ dùng
      nhãn ngành / chỉ dùng văn bản / cả hai — khoảng cách là phần ngành giải thích
      được; (b) phương sai lương trong ngành so với giữa ngành; (c) thiên lệch công
      bố lương theo ngành (mục ngay trên) làm lệch kết luận "ngành X lương cao hơn"
- [ ] `UNMASKED_COLUMNS` vẫn là danh sách viết tay — `soft_skills_text` và
      `qualifications_text` từng lọt qua nó; đường học sâu hiện chỉ đọc ba trường
      văn bản nên chưa bị ảnh hưởng, nhưng thêm một trường nghĩa là phải cập nhật cả
      hai danh sách
- [ ] **Bộ tách từ lệch với nguồn PhoBERT** — dự án dùng underthesea/pyvi, PhoBERT
      được VinAI tách từ bằng RDRSegmenter của VnCoreNLP. Hai bộ tách *trong* dự án
      đã bất đồng trên 84,8 % description ([02 §6](02-vietnamese-nlp.md#6-hai-bộ-tách-từ-khác-nhau-ở-đâu--và-vì-sao-câu-hỏi-mở-lại));
      độ lệch so với RDRSegmenter chưa từng được đo. Kết luận "đừng đổi bộ tách từ"
      ở §6 là kết luận cho TF-IDF (bước tách từ khi đó bị tắt) — với PhoBERT thì
      bước này luôn bật, nên câu hỏi "chọn bộ nào" lại mở ra
- [ ] **2.148 title (4,50 %) viết không dấu chưa có đường xử lý nào** — kênh bỏ dấu
      cũ (bước 6) từng bắt được chúng cho TF-IDF; với BPE của PhoBERT, "Nhan Vien"
      bị băm thành các mảnh hiếm ([02 §4](02-vietnamese-nlp.md#4-luồng-phobert-thực-sự-chạy-những-bước-nào))

## Ưu tiên 5 — hệ thống

- [ ] `predict.py` chưa biết đến đường học sâu. Quy tắc 4 (huấn luyện và suy luận
      dùng chung một đường mã) hiện **không được canh** cho nhánh học sâu
- [ ] `artifacts/PRODUCTION.json` ánh xạ bài toán → run_id, thay cho `_latest_run`,
      vốn đang chọn theo thời gian sửa tệp
- [ ] Đo độ trễ suy luận và báo cáo nó cạnh điểm số: PhoBERT chạy ~30 tin/giây trên
      CPU so với TF-IDF + LinearSVC gần như tức thời — đây là một sự đánh đổi mà báo
      cáo phải trình bày
