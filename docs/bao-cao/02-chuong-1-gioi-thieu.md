[← Phần đầu](01-phan-dau.md) · [Bảng điều khiển](00-index.md) · [Chương 2 →](03-chuong-2-tong-quan.md)

# CHƯƠNG 1. TỔNG QUAN

---

## 1.1. Bối cảnh thị trường tuyển dụng trực tuyến ở Việt Nam

Thị trường tuyển dụng trực tuyến tại Việt Nam vận hành theo cơ chế đăng tin gần như
tự do. Nhà tuyển dụng tự nhập tiêu đề, tự viết mô tả công việc và yêu cầu ứng viên.
Họ cũng tự chọn ngành nghề trong một danh sách thả xuống, và tự quyết định có ghi
mức lương hay không.

Với người tìm việc, hai thông tin quan trọng nhất của một tin tuyển dụng là công việc
thuộc ngành nào và được trả bao nhiêu. Cả hai thông tin này đều do người đăng tin tự
khai, không qua kiểm tra. Vì vậy, khi dữ liệu tuyển dụng được thu thập ở quy mô lớn,
chất lượng của chính hai trường này trở thành vấn đề.

## 1.2. Vấn đề nhãn ngành nghề nhiễu và tin không công bố lương

Thứ nhất, nhãn ngành nghề không nhất quán. Cùng một loại công việc có thể mang nhãn
ngành khác nhau tùy người đăng. Trong tệp gốc, 98 tin có tiêu đề chứa cụm "kinh doanh
bất động sản". 43 tin trong số đó mang nhãn `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng`,
38 tin mang nhãn `xây_dựng_kiến_trúc_bất_động_sản`, số còn lại rơi vào tài chính và
marketing. Tương tự, 5 tin có tiêu đề chứa "quản trị website": 3 tin được gán vào
`nhóm_nghề_khác`, 2 tin vào marketing. Mô hình học từ dữ liệu này sẽ học cả sự không
nhất quán đó.

Thứ hai, hơn một phần tư số tin không công bố mức lương. Đo trên tệp gốc 47.707 tin
sau khi loại trùng, tỷ lệ tin có công bố lương là 71,5 %, tức 28,5 % số tin để trống.

Thứ ba, việc thiếu lương không xảy ra ngẫu nhiên. Tỷ lệ công bố lương dao động từ
58,6 % đến 76,5 % tùy ngành. Hệ số tương quan Pearson giữa tỷ lệ này và logarit cỡ lớp
là 0,610. Ngành càng ít tin thì càng ít công bố lương. Các ngành nhỏ vì vậy vừa có ít
mẫu để học phân loại, vừa có ít nhãn hơn cho bài toán lương.

Tự động phân loại ngành nghề và ước lượng mức lương từ chính nội dung văn bản của tin
đăng vì vậy có ý nghĩa thực tiễn. Nó trả lại cho người tìm việc một điểm tham chiếu mà
thị trường không tự cung cấp.

> **Nguồn số liệu:** `artifacts/eda/title_labels.json` (`scripts/count_title_labels.py`,
> tệp gốc 47.707 dòng, tiêu đề chuẩn hoá NFC và viết thường) ·
> `artifacts/eda/summary.json` (`disclosure_rate` = 0,7146;
> `per_category[*].disclosure_rate` từ 0,5855 đến 0,7649;
> `distribution.size_vs_disclosure_pearson_logn` = 0,6097).

## 1.3. Phát biểu bài toán

Đề tài giải quyết hai bài toán trên cùng một đầu vào là văn bản của một tin tuyển dụng
tiếng Việt. Bài toán thứ nhất là phân loại ngành nghề, một bài toán phân loại đơn nhãn
16 lớp. Bài toán thứ hai là ước lượng mức lương, một bài toán hồi quy. Hai bài toán
được huấn luyện riêng lẻ trước, rồi cùng nhau trong một mô hình đa nhiệm.

### 1.3.1. Đầu vào

Một tin tuyển dụng gồm ba trường văn bản tự do:

- tiêu đề tin (`job_title`), tín hiệu ngành nghề mạnh nhất;
- mô tả công việc (`description`), nguồn thông tin về cấp bậc và quy mô;
- yêu cầu ứng viên (`requirements_text`), chứa kinh nghiệm, bằng cấp, ngoại ngữ.

Mọi con số lương xuất hiện trong văn bản được che đi trước khi đưa vào mô hình lương,
để mô hình không đọc được đáp án ngay trong đầu vào.

### 1.3.2. Đầu ra

Bài toán phân loại trả về một trong 16 nhóm ngành, đánh giá chính bằng macro-F1. Bài
toán ước lượng lương trả về một số thực, đơn vị triệu đồng mỗi tháng, đánh giá chính
bằng MAE. Mô hình học trên `log1p` của mức lương giữa khoảng.

### 1.3.3. Ví dụ minh họa

Đầu vào (tin dựng để minh họa): tiêu đề "Nhân Viên Kinh Doanh Bất Động Sản"; mô tả
"Tìm kiếm khách hàng tiềm năng, tư vấn sản phẩm căn hộ..."; yêu cầu "Tốt nghiệp Cao
đẳng trở lên, có 2 năm kinh nghiệm...".

Đầu ra: một nhóm ngành và một mức lương ước lượng tính bằng triệu đồng mỗi tháng.

Ví dụ này cho thấy bài toán không tầm thường. Ngay trong dữ liệu, các tin có tiêu đề
này chia gần đều giữa nhãn kinh doanh và nhãn xây dựng – bất động sản (§1.2). Còn
"2 năm kinh nghiệm" là một tín hiệu lương nằm trong văn bản, không nằm trong nhãn
ngành.

## 1.4. Những thách thức của đề tài

- **Dữ liệu lệch lớp mạnh.** Lớp lớn nhất (kinh doanh – bán hàng, 8.213 tin) gấp 25,5
  lần lớp nhỏ nhất (nông nghiệp – năng lượng – môi trường). Một mô hình chỉ học các
  lớp lớn vẫn có độ chính xác cao nhưng bỏ rơi phần đuôi.
- **Nhãn nhiễu và các ngành chồng lấn.** Trên tập dev, mô hình chốt nhầm du lịch – nhà
  hàng sang kinh doanh – bán hàng 122 lần, và nhầm theo chiều ngược lại 76 lần. Lỗi
  lớn nhất của lớp thiết kế là sang xây dựng – kiến trúc (71 lần).
- **Nhãn lương thiếu và lệch.** Nhãn lương chỉ có trên 71,5 % số tin. Phân bố lương
  lệch phải mạnh (độ lệch 11,90), với một đuôi dài các tin lương rất cao.
- **Ngành nghề gần như không nói gì về lương.** Nhãn ngành chỉ giải thích 3,2 % biến
  thiên của log-lương. Mô hình lương vì vậy phải đọc tín hiệu từ chính văn bản.
- **Văn bản tiếng Việt không đồng nhất.** 2.148 tiêu đề (4,50 %) không mang dấu thanh
  nào. 3.436 mô tả (7,2 %) chứa ít nhất một từ viết tắt trong bảng viết tắt của đề tài.
  Ranh giới từ tiếng Việt không trùng với khoảng trắng.

> **Nguồn số liệu:** `artifacts/eda/summary.json` (`class_counts`, `imbalance_ratio` =
> 25,51, `smallest_class`, `disclosure_rate`, `salary.skew` = 11,90,
> `variance.eta_squared_log` = 0,0317) ·
> `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/predictions_dev.parquet` (ma trận nhầm lẫn
> trên 3.812 tin dev) · [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) bảng chín bước
> (bước 3 và bước 6).

## 1.5. Mục tiêu của đề tài

**Mục tiêu tổng quát:** xây dựng mô hình học sâu phân loại ngành nghề và ước lượng
mức lương từ nội dung tin tuyển dụng tiếng Việt.

**Mục tiêu cụ thể:**

1. Biểu diễn tin tuyển dụng bằng vectơ ngữ cảnh từ các mô hình ngôn ngữ tiền huấn
   luyện cho tiếng Việt.
2. Thử lần lượt đầu mạng Dense, đầu mạng RNN và mô hình đa nhiệm trên các bộ mã hóa
   khác nhau, để chọn ra cấu hình tốt nhất trên tập dev.
3. Đối chiếu các mô hình với mốc cơ sở không dùng mô hình, và với phép dò tuyến tính
   trên vectơ của PhoBERT và CafeBERT.
4. Đánh giá mô hình chốt trên tập kiểm tra và trên một bộ dữ liệu ngoài.

## 1.6. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu** là nội dung văn bản của tin tuyển dụng tiếng Việt, và mối
liên hệ giữa nội dung đó với ngành nghề và mức lương của vị trí.

**Phạm vi dữ liệu:** 48.092 tin thô, còn 47.707 tin sau khi loại trùng khớp hoàn toàn,
thuộc 16 nhóm ngành nghề. Đơn vị lương là triệu đồng mỗi tháng. Dữ liệu được chia hai
tầng: train:test = 8:2, sau đó train:dev = 9:1, được 34.354 / 3.812 / 9.541 tin.

**Phạm vi mô hình:** các bộ mã hóa được giữ đóng băng trọng số. Chỉ phần đầu mạng phía
sau được huấn luyện. Mô hình chỉ đọc ba trường văn bản ở §1.3.1.

> **Nguồn số liệu:** `data/processed/manifest.json` (`source_rows` = 48.092,
> `rows_after_exact_dedup` = 47.707, `split_test_fraction` = 0,2,
> `split_dev_fraction` = 0,1, `splits.{train,dev,test}.rows`, `salary_unit`).

## 1.7. Phương pháp nghiên cứu

Đề tài đi theo quy trình thực nghiệm ba giai đoạn.

- **Giai đoạn 1** dựng đầu mạng Dense trên vectơ PhoBERT làm mô hình nền.
- **Giai đoạn 2** giữ đầu Dense và thay lần lượt các bộ mã hóa PhoBERT, CafeBERT,
  ViSoBERT, XLM-R, BamiBERT để so sánh chất lượng biểu diễn.
- **Giai đoạn 3** thay đầu Dense bằng đầu RNN đọc trực tiếp chuỗi token: hai nhánh
  BiGRU và BiLSTM song song, mỗi nhánh có một lớp tích chập. Giai đoạn này cũng gộp hai
  bài toán vào một mô hình đa nhiệm.

Mọi lựa chọn mô hình và siêu tham số đều dựa trên tập dev. Tập test không dùng để chọn
mô hình. Các cấu hình trên bảng kết quả được chấm trên test nhiều nhất một lần, từ
checkpoint đã lưu, không huấn luyện lại. Số test được báo cáo kèm số dev, không thay cho số dev.

Mỗi kết quả được so với hai tầng mốc: mốc ngây thơ không dùng mô hình, và phép dò tuyến
tính trên chính vectơ biểu diễn.

## 1.8. Những đóng góp của đề tài

1. **Một mô hình đa nhiệm cho cả hai bài toán.** Mô hình chốt là một lần huấn luyện
   mô hình đa nhiệm CafeBERT với đầu RNN. Lần huấn luyện này lưu hai checkpoint: một
   checkpoint chọn theo macro-F1 trên dev cho đầu phân loại, một checkpoint chọn theo
   MAE trên dev cho đầu lương.
   - Đầu phân loại đạt macro-F1 0,6322 trên dev và 0,6278 trên test.
   - Đầu lương đạt MAE 3,70 triệu trên dev và 3,92 triệu trên test. Trên dev, con số
     này thấp hơn 35,0 % so với MAE 5,70 triệu của mốc đoán trung vị.
2. **So sánh năm bộ mã hóa trên cùng một cấu hình.** Trên dev, CafeBERT có macro-F1
   cao nhất trong năm bộ mã hóa, và được chọn theo tiêu chí đó. Về MAE, CafeBERT đứng
   thứ ba, sau PhoBERT (3,65) và BamiBERT (3,66).
3. **Một quy trình xử lý tiếng Việt có số đo cho từng bước.** Mỗi bước trên đường vào
   bộ mã hóa đều có số đo trên chính dữ liệu, cho biết nó thay đổi bao nhiêu dòng.
4. **Một phép đo giới hạn của bài toán lương.** Nhãn ngành nghề chỉ giải thích 3,2 %
   biến thiên của log-lương.
5. **Một giao thức thực nghiệm chống rò rỉ dữ liệu.** Số lương trong văn bản đầu vào
   được che. Dữ liệu được chia theo nhóm tin đăng lại thay vì theo dòng, nên không nhóm
   nào nằm ở hai tập. Phần tin gần trùng mà cách gom nhóm này bỏ sót được đo và báo cáo
   riêng (§3.2).

> **Nguồn số liệu:** `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/metrics.json`
> (`metrics_cat.f1_macro` = 0,6322, `best_epoch` = 21; `metrics_sal.mae_trieu` = 3,700,
> `best_epoch_sal` = 30; n = 3.812 / 2.698) · `…/metrics_test.json`
> (`metrics_cat.f1_macro` = 0,6278; `metrics_sal.mae_trieu` = 3,923; n = 9.541 / 6.773) ·
> `artifacts/eda/summary.json` (`floors.salary_global_median.mae` = 5,6955,
> `variance.eta_squared_log` = 0,0317) · `metrics.json` của bốn run
> `dl-mtl-{pho,xlm,vis,bam}-rnn-b01-cw-s42-v2` (macro-F1 và MAE trên dev).

## 1.9. Cấu trúc khóa luận

Khóa luận gồm sáu chương.

- **Chương 1 – Tổng quan:** bối cảnh, vấn đề, bài toán, thách thức, mục tiêu, phạm vi,
  phương pháp và đóng góp.
- **Chương 2 – Cơ sở lý thuyết:** mô tả bài toán, các công trình liên quan, các mô
  hình ngôn ngữ tiền huấn luyện, các kiến trúc đầu mạng, học đa nhiệm và hàm mất mát.
- **Chương 3 – Bộ dữ liệu VietJobs:** kiểm toán, làm sạch, chống rò rỉ, phân chia và
  phân tích khám phá dữ liệu.
- **Chương 4 – Cài đặt thử nghiệm:** độ đo, tiền xử lý, mốc cơ sở và cấu hình các mô
  hình.
- **Chương 5 – Các kết quả thử nghiệm:** ba giai đoạn thử nghiệm dẫn tới mô hình chốt,
  phân tích lỗi và đánh giá ngoài.
- **Chương 6 – Kết luận và hướng phát triển.**

---

[← Phần đầu](01-phan-dau.md) · [Chương 2 →](03-chuong-2-tong-quan.md)
