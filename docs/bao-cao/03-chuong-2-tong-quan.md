[← Chương 1](02-chuong-1-gioi-thieu.md) · [Bảng điều khiển](00-index.md) · [Chương 3 →](04-chuong-3-phuong-phap.md)

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

---

## 2.1. Mô tả bài toán

Chương này trình bày nền tảng cho hai bài toán của đề tài: hai công trình liên quan gần
nhất, cách biểu diễn văn bản tiếng Việt thành vectơ, các kiến trúc đầu mạng đọc vectơ
đó, và các hàm mất mát phù hợp với dữ liệu lệch.

### 2.1.1. Phân loại văn bản tin tuyển dụng

Phân loại ngành nghề từ tin tuyển dụng là một trường hợp của phân loại văn bản đa lớp:
cho một đoạn văn bản, gán nó vào đúng một trong K nhãn định trước. Bài toán này khác với
phân loại tin tức hay phân loại cảm xúc ở ba điểm.

Thứ nhất, nhãn do người đăng tin tự chọn chứ không do chuyên gia gán, nên nhiễu nhãn là
bản chất của dữ liệu. Thứ hai, ranh giới giữa các lớp mờ một cách hợp lệ: một tin "Nhân
viên kinh doanh cho khách sạn" có thể thuộc kinh doanh hoặc du lịch – nhà hàng – khách
sạn. Thứ ba, phân bố lớp lệch nặng: tỷ lệ giữa lớp lớn nhất và lớp nhỏ nhất là 25,5:1.

### 2.1.2. Ước lượng mức lương từ văn bản

Ước lượng mức lương là bài toán hồi quy trên đầu vào văn bản, với ba đặc thù.

- **Nhãn khuyết không ngẫu nhiên:** chỉ 71,5 % tin có lương, và tỷ lệ khuyết phụ thuộc
  ngành.
- **Phân bố lệch phải rất nặng:** độ lệch (skewness) của lương thô là 11,90.
- **Nhãn tự nó đã là xấp xỉ:** 92,9 % tin có lương công bố một khoảng chứ không phải một
  con số, và nhãn hồi quy là trung điểm của khoảng đó.

> **Nguồn số liệu:** `artifacts/eda/summary.json` (`imbalance_ratio` = 25,51,
> `disclosure_rate` = 0,7146, `salary.skew` = 11,90, `salary_is_range_share` = 0,9293).

---

## 2.2. Các công trình liên quan

Mục này điểm qua hai công trình gần nhất với đề tài trên dữ liệu tiếng Việt: bài công bố
bộ dữ liệu VietJobs, và bài dự đoán chức danh mà đầu RNN của đề tài kế thừa. Bảng 2.1 ở
cuối mục tóm tắt dữ liệu, phương pháp, kết quả công bố và hạn chế của từng công trình so
với đề tài.

Bộ dữ liệu của đề tài được công bố bởi Phạm Dinh, Nguyễn Huy và El-Haj [13]: 48.092 tin
tuyển dụng thu thập từ TopCV, thuộc 16 nhóm ngành, phủ 34 tỉnh thành. Bài báo chỉ đánh
giá các mô hình ngôn ngữ lớn sinh văn bản (Qwen2.5-7B, Llama-SEA-LION-v3-8B và một số mô
hình khác) ở các chế độ zero-shot, few-shot và tinh chỉnh LoRA. Kết quả tốt nhất cho phân
loại là macro-F1 0,42. Khi chỉ tinh chỉnh trên VietJobs, kết quả tốt nhất cho ước lượng
lương là RMSE 10,60 triệu và R² 0,17. Đầu vào của bài lương chỉ gồm chức danh, loại hợp
đồng, địa điểm, quốc gia và kinh nghiệm, không gồm mô tả công việc. Bài báo không có mốc so
sánh từ bộ mã hóa như PhoBERT, và chia dữ liệu theo tỷ lệ 80/10/10, khác với đề tài, nên
các con số không so sánh trực tiếp được.

Tran, Vo và Luu [14] dự đoán chức danh từ mô tả công việc tiếng Việt dưới dạng phân loại
đa nhãn với 68 chức danh. Họ dùng đầu mạng Bi-GRU‖Bi-LSTM→CNN trên các bộ mã hóa mBERT,
XLM-R, PhoBERT và DistilBERT. Kết quả tốt nhất thuộc về mBERT, với F1 62,20 % trên tập dev
và 47,44 % trên tập test. Đầu RNN của đề tài kế thừa kiến trúc này.

Trong hai công trình trên, chưa công trình nào huấn luyện chung một mô hình cho cả nhóm
ngành lẫn mức lương, so sánh các bộ mã hóa PhoBERT, CafeBERT, ViSoBERT, XLM-R và BamiBERT ở
chế độ đóng băng, hay đọc văn bản mô tả để ước lượng lương trên bộ dữ liệu VietJobs. Đề
tài nhắm vào khoảng trống đó.

**Bảng 2.1: Hệ thống hóa các công trình liên quan**

| STT | Công trình | Dữ liệu & phương pháp | Kết quả công bố | Hạn chế so với đề tài |
|---|---|---|---|---|
| 1 | Phạm Dinh, Nguyễn Huy và El-Haj, 2026 [13] | Bộ VietJobs: 48.092 tin TopCV, 16 nhóm ngành; mô hình ngôn ngữ lớn zero-shot, few-shot và tinh chỉnh LoRA | Phân loại: macro-F1 0,42 (few-shot); lương: RMSE 10,60 triệu, R² 0,17 (tinh chỉnh trên VietJobs) | Không có mốc bộ mã hóa; lương không đọc mô tả công việc; hai bài toán tách riêng |
| 2 | Tran, Vo và Luu, 2021 [14] | Mô tả công việc tiếng Việt, 68 chức danh đa nhãn; Bi-GRU‖Bi-LSTM→CNN trên mBERT, XLM-R, PhoBERT, DistilBERT | F1 62,20 % (dev), 47,44 % (test) với mBERT | Không có lương; đơn nhiệm; không có CafeBERT, ViSoBERT |

> **Nguồn số liệu:** [13] Table 3 (macro-F1 0,42) và bảng kết quả tinh chỉnh, cấu hình
> "Fine-tuned on VietJobs" (RMSE 10,60, R² 0,17); [14] phần tóm tắt, Fig. 3 và Table V.

---

## 2.3. Biểu diễn văn bản bằng mô hình ngôn ngữ tiền huấn luyện

Kiến trúc Transformer [1] và mô hình BERT [2] cho mỗi từ một vectơ phụ thuộc vào các từ
xung quanh nó: cùng một từ ở hai câu khác nhau cho hai vectơ khác nhau. Mô hình được tiền
huấn luyện trên kho văn bản lớn bằng nhiệm vụ đoán từ bị che, rồi dùng lại cho nhiệm vụ
khác. RoBERTa [3] cho thấy chỉ cần điều chỉnh quy trình tiền huấn luyện – bỏ nhiệm vụ đoán
câu kế tiếp, huấn luyện lâu hơn trên nhiều dữ liệu hơn – là đã cải thiện đáng kể chất
lượng biểu diễn.

Trong đề tài, các bộ mã hóa được giữ đóng băng: mỗi tin được mã hóa một lần, vectơ được
lưu lại, và chỉ phần đầu mạng phía sau được huấn luyện. Cách làm này cho phép so sánh
nhiều bộ mã hóa và nhiều đầu mạng với chi phí tính toán nhỏ. Vectơ của mỗi token lấy từ
lớp ẩn cuối của bộ mã hóa, hoặc là trung bình của bốn lớp ẩn cuối. Mô hình chốt dùng
trung bình bốn lớp ẩn cuối.

### 2.3.1. PhoBERT

PhoBERT [4] là mô hình ngôn ngữ đơn ngữ quy mô lớn đầu tiên được tiền huấn luyện riêng cho
tiếng Việt, theo quy trình của RoBERTa. Hai đặc điểm của PhoBERT quyết định cách xử lý dữ
liệu.

Thứ nhất, PhoBERT được huấn luyện trên văn bản đã tách từ: "nhân viên" là một từ và được
viết thành `nhân_viên`. Đưa văn bản chưa tách từ vào PhoBERT là đưa sai phân bố dữ liệu, và
lỗi này không sinh ra thông báo nào. Đề tài dùng bộ tách từ của thư viện underthesea [5];
một lựa chọn thay thế là VnCoreNLP [6]. Thứ hai, PhoBERT cắt văn bản thành đơn vị con theo
BPE và nhận tối đa 256 token. Đề tài dùng bản `phobert-base-v2`, vectơ 768 chiều.

### 2.3.2. CafeBERT

CafeBERT (`uitnlp/CafeBERT`) là mô hình ngôn ngữ cho tiếng Việt xây dựng trên nền XLM-R
[17] bản large, được giới thiệu cùng bộ đánh giá VLUE [15].

So với PhoBERT, CafeBERT khác ở ba điểm có ảnh hưởng trực tiếp tới đề tài. CafeBERT đọc
văn bản chưa tách từ, vì bộ tách token SentencePiece của XLM-R tự xử lý âm tiết. CafeBERT
nhận chuỗi tới 512 token, nên chỉ 5,7 % tin trong tập dev bị cắt bớt khi mã hóa. CafeBERT
là bộ mã hóa được chọn cho mô hình chốt; quá trình so sánh dẫn tới lựa chọn này được trình
bày ở Chương 5.

### 2.3.3. Các bộ mã hóa đối chiếu: ViSoBERT, XLM-R, BamiBERT

Để kiểm tra lựa chọn bộ mã hóa, đề tài so sánh thêm ba mô hình:

- ViSoBERT [16], mô hình tiền huấn luyện trên văn bản mạng xã hội tiếng Việt;
- XLM-R [17] bản base, mô hình đa ngữ. CafeBERT được xây dựng trên bản large của chính
  mô hình này, nên hai mô hình khác nhau về kích thước (768 so với 1.024 chiều);
- BamiBERT [18], mô hình ngôn ngữ cho tiếng Việt.

ViSoBERT và XLM-R được dùng với 512 token, BamiBERT với 1.024 token. Cả ba đọc văn bản chưa
tách từ.

> **Nguồn số liệu:** `src/vietjobs/dl/encode.py` (`ENCODERS`: tên mô hình, số chiều,
> `segmented`) · `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/config.json` (`encoder` =
> cafebert, `max_len` = 512, `last4` = true) ·
> `artifacts/embeddings/cafebert-dev-masked-len512.json` (`truncated_frac` = 0,0572) ·
> `docs/04-results.md` (các run ViSoBERT, XLM-R độ dài 512 và BamiBERT độ dài 1.024).

---

## 2.4. Các kiến trúc đầu mạng

Vectơ từ bộ mã hóa được đưa vào một đầu mạng để ra kết quả cuối. Đề tài thử hai họ đầu
mạng: đầu Dense đọc một vectơ đã gộp cho cả tin, và đầu RNN đọc cả chuỗi vectơ của từng
token. Mỗi đầu có thể dùng riêng cho từng bài toán, hoặc chung cho cả hai trong mô hình đa
nhiệm.

### 2.4.1. Mạng kết nối đầy đủ (Dense)

Mạng kết nối đầy đủ biến đổi vectơ đầu vào qua một chuỗi phép nhân ma trận và hàm kích
hoạt phi tuyến. Trong đề tài, vectơ của tin được gộp bằng trung bình các token, đi qua
LayerNorm, rồi qua hai lớp ẩn 256 và 128 nút với hàm kích hoạt GELU và dropout 0,3. Cuối
cùng là lớp đầu ra 16 nút cho phân loại, hoặc 1 nút cho lương.

Ưu điểm của đầu Dense là nhỏ và nhanh. Hạn chế là phép gộp trung bình xóa mất thứ tự từ và
các cấu trúc như phủ định.

### 2.4.2. BiGRU, BiLSTM và CNN

Mạng hồi quy (RNN) đọc chuỗi theo thứ tự và giữ một trạng thái ẩn tóm tắt phần đã đọc. LSTM
và GRU là hai biến thể có cổng điều khiển, giúp giữ thông tin qua chuỗi dài. Bản hai chiều
(BiLSTM, BiGRU) đọc chuỗi theo cả hai hướng. Lớp tích chập một chiều (Conv1d) sau đó trích
các mẫu cục bộ trên chuỗi trạng thái ẩn.

Kiến trúc của Tran, Vo và Luu [14] gồm hai nhánh BiGRU và BiLSTM song song. Mỗi nhánh có
một lớp tích chập, rồi qua gộp trung bình và gộp cực đại. Kết quả của hai nhánh được ghép
lại trước lớp Dense cuối.

Đầu RNN của đề tài theo đúng kiến trúc đó. Chuỗi vectơ token đi qua SpatialDropout, rồi đi
song song qua hai nhánh BiGRU và BiLSTM. Mỗi nhánh có một lớp Conv1d riêng, sau đó gộp
trung bình và gộp cực đại có mặt nạ để bỏ phần đệm. Kết quả của hai nhánh được ghép lại, qua
LayerNorm và lớp kết nối đầy đủ. Khác với đầu Dense, đầu RNN đọc trực tiếp chuỗi token nên
giữ được thứ tự từ.

### 2.4.3. Học đa nhiệm

Học đa nhiệm [7] huấn luyện một mạng giải đồng thời nhiều nhiệm vụ qua một phần thân dùng
chung. Nếu các nhiệm vụ chia sẻ tín hiệu, phần thân chung buộc phải học biểu diễn tổng quát
hơn, và mỗi nhiệm vụ đóng vai trò chính quy hóa cho nhiệm vụ kia.

Trong đề tài, phần thân chung (Dense hoặc RNN) tách thành hai đầu: một đầu phân loại 16 lớp
và một đầu ước lượng lương. Hàm mất mát là tổng có trọng số:

L = CE(ngành) + α · SmoothL1(lương), với α = 5.

Trong đó, CE dùng trọng số lớp hiệu dụng và làm mịn nhãn 0,05. SmoothL1 dùng β = 0,1, và
nhân mỗi tin có lương với một trọng số theo khoảng lương của nó. Phần lương chỉ tính trên
các tin có công bố lương, nên mô hình vẫn học phân loại từ cả những tin không có lương.

> **Nguồn số liệu:** `src/vietjobs/dl/heads.py` (`DenseHead`, `BiGruLstmCnn`,
> `MultiTaskRnn`) · `src/vietjobs/dl/train_mtl.py` (`loss = l_cat + alpha_sal * l_sal`,
> `l_sal` chỉ trên tin có lương) · `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/config.json`
> (`hidden` = 256, `dropout` = 0,3, `spatial_dropout` = 0,2, `alpha_sal` = 5,
> `beta_sal` = 0,1, `cw_mode` = effective, `label_smoothing` = 0,05).

---

## 2.5. Hàm mất mát cho dữ liệu lệch lớp và lương lệch phải

Với phân loại, hàm Cross-Entropy coi mọi mẫu như nhau nên bị các lớp lớn chi phối. Đề tài
dùng ba điều chỉnh:

- trọng số lớp theo số mẫu hiệu dụng, với β = 0,999;
- làm mịn nhãn (label smoothing) 0,05, để giảm ảnh hưởng của nhãn nhiễu;
- thử Focal Loss với γ = 2. Hàm này nhân mất mát của mỗi mẫu với (1 − p)^γ, để tập trung
  vào mẫu khó.

Với hồi quy, sai số bình phương phạt theo bình phương, nên một điểm ngoại lệ đóng góp
gradient rất lớn. Hàm Huber [8] (SmoothL1) xử lý sai số nhỏ theo bình phương và sai số lớn
theo tuyến tính. Kết hợp với phép biến đổi `log1p` trên nhãn, độ lệch của lương giảm từ
11,90 xuống 0,10.

> **Nguồn số liệu:** `artifacts/dl-mtl-caf-rnn-b01-cw-s42-v2/config.json` (`cw_beta` =
> 0,999, `label_smoothing` = 0,05) · `docs/04-results.md` (các run `focal(g=2)`) ·
> `artifacts/eda/summary.json` (`salary.skew` = 11,90, `salary.skew_log1p` = 0,10).

---

[← Chương 1](02-chuong-1-gioi-thieu.md) · [Chương 3 →](04-chuong-3-phuong-phap.md)
