[← Chương 1](02-chuong-1-gioi-thieu.md) · [Bảng điều khiển](00-index.md) · [Chương 3 →](04-chuong-3-phuong-phap.md)

# CHƯƠNG 2. TỔNG QUAN NGHIÊN CỨU

Chương này trình bày cơ sở lý thuyết mà đề tài dựa vào, khảo sát các công trình liên
quan, và chỉ ra khoảng trống mà đề tài nhắm tới.

---

## 2.1. Tổng quan bài toán

### 2.1.1. Phân loại văn bản tin tuyển dụng

Phân loại ngành nghề từ tin tuyển dụng là một trường hợp của **phân loại văn bản đa
lớp**: cho một đoạn văn bản, gán nó vào đúng một trong `K` nhãn định trước. Điểm
khiến bài toán này khác với phân loại văn bản kinh điển (phân loại tin tức, phân
loại cảm xúc) nằm ở ba đặc điểm của dữ liệu:

- **Nhãn do người dùng tự chọn, không do chuyên gia gán.** Nhiễu nhãn là bản chất
  của dữ liệu chứ không phải sự cố.
- **Ranh giới lớp mờ một cách hợp lệ.** Một tin "Nhân viên kinh doanh cho khách sạn"
  thuộc kinh doanh hay thuộc du lịch — nhà hàng — khách sạn? Cả hai câu trả lời đều
  bảo vệ được.
- **Phân bố lớp lệch nặng.** Trong bộ dữ liệu của đề tài, tỷ lệ giữa lớp lớn nhất và
  lớp nhỏ nhất là **25,5 : 1**.

### 2.1.2. Ước lượng mức lương từ văn bản

Ước lượng mức lương là bài toán **hồi quy trên đầu vào văn bản**. Ba đặc thù:

- **Nhãn khuyết không ngẫu nhiên.** Chỉ 71,5 % tin có nhãn lương, và tỷ lệ khuyết
  phụ thuộc vào ngành nghề. Mô hình hồi quy vì vậy học trên một mẫu đã bị chọn lọc.
- **Phân bố lệch phải rất nặng.** Độ lệch (skewness) của lương thô là **11,90**.
- **Nhãn tự nó đã là xấp xỉ.** Phần lớn tin công bố một *khoảng* lương chứ không phải
  một con số; nhãn hồi quy là trung điểm của khoảng đó. Trên tệp gốc, **92,9 %** tin
  có lương công bố một khoảng (Bảng 3.14).

---

## 2.2. Cơ sở lý thuyết

### 2.2.1. Biểu diễn văn bản bằng vectơ ngữ cảnh

**Vectơ ngữ cảnh.** Kiến trúc Transformer [1] và mô hình BERT [2] thay đổi điều
này: mỗi từ nhận một vectơ **phụ thuộc vào các từ xung quanh nó**. Cùng một từ ở
hai câu khác nhau cho hai vectơ khác nhau. Mô hình được tiền huấn luyện trên kho
văn bản lớn bằng nhiệm vụ đoán từ bị che, rồi được dùng lại cho nhiệm vụ khác.
RoBERTa [3] cho thấy chỉ cần tinh chỉnh quy trình tiền huấn luyện của BERT — bỏ
nhiệm vụ đoán câu kế tiếp, huấn luyện lâu hơn trên nhiều dữ liệu hơn — là đã cải
thiện đáng kể chất lượng biểu diễn.

### 2.2.2. PhoBERT và đặc thù tiếng Việt

**PhoBERT** [4] là mô hình ngôn ngữ đơn ngữ quy mô lớn đầu tiên được tiền huấn
luyện riêng cho tiếng Việt, theo quy trình của RoBERTa. Hai chi tiết của PhoBERT
quyết định trực tiếp cách đề tài xử lý dữ liệu:

**Thứ nhất, PhoBERT được huấn luyện trên văn bản đã tách từ.** Trong tiếng Việt,
đơn vị mang nghĩa thường gồm nhiều âm tiết ngăn cách bằng khoảng trắng: "nhân viên"
là một từ, không phải hai. PhoBERT nhận đầu vào ở dạng đã nối bằng gạch dưới —
`nhân_viên kinh_doanh`. Đưa văn bản chưa tách từ vào PhoBERT là đưa sai phân bố dữ
liệu, và lỗi này **không phát sinh thông báo nào**. Đề tài dùng bộ tách từ của thư
viện `underthesea` [5]; một lựa chọn thay thế phổ biến là VnCoreNLP [6].

**Thứ hai, PhoBERT cắt văn bản thành đơn vị con theo BPE.** Một chuỗi ký tự lạ
không bị báo lỗi; nó chỉ bị vỡ thành nhiều mảnh hiếm, và vectơ ngữ nghĩa thu được
nhạt đi mà không có dấu hiệu nào trên màn hình. Đây là lý do toàn bộ chín bước
chuẩn hoá ở Chương 3 tồn tại: mỗi bước kéo văn bản của đề tài về gần hơn với phân
bố mà PhoBERT đã học.

### 2.2.3. Học đa nhiệm

**Nguyên lý.** Học đa nhiệm (Multi-Task Learning) [7] huấn luyện một mạng giải
đồng thời nhiều nhiệm vụ qua một phần thân dùng chung. Lập luận: nếu các nhiệm vụ
chia sẻ tín hiệu, phần thân chung buộc phải học biểu diễn tổng quát hơn, và mỗi
nhiệm vụ đóng vai trò chính quy hoá cho nhiệm vụ kia.

**Điều đề tài đã đo.** Lập luận "hai nhiệm vụ hỗ trợ nhau" chỉ đứng vững nếu hai
nhãn thực sự chia sẻ tín hiệu. Đề tài đo trực tiếp phần chia sẻ đó bằng phân rã
phương sai và thu được eta² = **0,032**: ngành nghề chỉ giải thích 3,2 % biến thiên
của log-lương (§3.6.7). Hai bài toán được giải bằng hai mạng riêng (§3.7).

### 2.2.4. Hàm mất mát cho dữ liệu lệch phải

**Nguyên lý.** Sai số bình phương trung bình (MSE) phạt sai số theo bình phương,
nên một điểm ngoại lệ đóng góp gradient rất lớn. Hàm mất mát Huber [8] xử lý phần
sai số nhỏ theo bình phương và phần sai số lớn theo tuyến tính, nhờ đó vừa mượt
quanh 0 vừa không để ngoại lệ chi phối.

**Lý do chọn Huber cho đề tài.** Lương có độ lệch 11,90 và giá trị lớn nhất trong tệp
gốc là 500 triệu. Với MSE, mười lăm tin trên 200 triệu sẽ kéo toàn bộ gradient. Kết
hợp thêm phép biến đổi `log1p` trên nhãn, độ lệch giảm từ **11,90** xuống **0,10** —
gần như đối xứng.

---

## 2.3. Các nghiên cứu liên quan

### 2.3.1. Các nghiên cứu trên thế giới

> ✍️ **CẦN VIẾT TAY** — phần này bắt buộc phải do tác giả tự khảo sát và đọc thật.
> Bản thảo cố tình **không** điền sẵn tên công trình, vì một trích dẫn chưa đọc là
> đạo văn dù nội dung có đúng.
>
> Kế hoạch tra cứu, mỗi truy vấn cho một nhóm công trình:
>
> | Chủ đề cần tìm | Truy vấn gợi ý | Nơi tìm |
> |---|---|---|
> | Phân loại tin tuyển dụng theo chuẩn nghề nghiệp | `job posting classification ISCO/O*NET occupation` | ACL Anthology, Google Scholar |
> | Dự đoán lương từ mô tả công việc | `salary prediction job description regression` | ACL Anthology, arXiv |
> | Học đa nhiệm cho phân lớp + hồi quy trên văn bản | `multi-task text classification regression shared encoder` | arXiv |
> | Nhãn khuyết không ngẫu nhiên trong hồi quy | `missing not at random regression selection bias` | Google Scholar |
>
> Với mỗi công trình, ghi lại theo đúng bốn cột của Bảng 2.1 bên dưới. Ba câu là
> đủ, nhưng ba câu đó phải là của người đã đọc bài báo.

### 2.3.2. Các nghiên cứu trong nước

> ✍️ **CẦN VIẾT TAY** — tương tự. Nguồn nên tra: Tạp chí Tin học và Điều khiển học,
> kỷ yếu hội nghị KSE, hội nghị FAIR, các luận văn cùng chủ đề tại các trường trong
> nước. Truy vấn gợi ý: `phân loại ngành nghề tin tuyển dụng tiếng Việt`,
> `dự đoán mức lương tiếng Việt PhoBERT`, `xử lý văn bản tuyển dụng tiếng Việt`.

### 2.3.3. Bảng hệ thống hoá các công trình tham khảo

**Bảng 2.1: Hệ thống hoá các công trình liên quan**

| STT | Công trình | Dữ liệu & phương pháp | Kết quả công bố | Hạn chế / khoảng trống |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

> ⛔ **CHƯA CÓ SỐ LIỆU** (T6.1)
> Mẫu báo cáo của trường đều có bảng này (xem `Bảng 2.1` trong quyển mẫu), nên nó
> không được bỏ.

---

## 2.4. Khoảng trống nghiên cứu và đề xuất hướng tiếp cận

### 2.4.1. Khoảng trống nghiên cứu

Ba khoảng trống dưới đây rút ra từ **chính dữ liệu của đề tài**, nên chúng đứng
vững độc lập với kết quả khảo sát tài liệu ở §2.3.

**Khoảng trống 1 — mức trần thực tế của bài toán hiếm khi được đo.** Các công trình
dự đoán lương thường báo cáo sai số của mô hình mà không báo cáo sai số của quy tắc
ngây thơ tương ứng. Trong đề tài này, chỉ cần đoán trung vị cho mọi tin đã đạt MAE
**5,70 triệu**; biết trước 100 % nhãn ngành nghề cũng chỉ hạ xuống **5,57 triệu**,
tức cải thiện **2,3 %** (học trên `train`, chấm trên `dev`). Nếu không công bố hai con
số này, một mô hình đạt 5,6 triệu trông như một kết quả, trong khi thực chất nó chưa học được gì.

**Khoảng trống 2 — rò rỉ dữ liệu qua tin đăng lại.** Nhà tuyển dụng đăng lại tin
rất nhiều: **12.808 dòng (26,8 %)** trong bộ dữ liệu là tin đăng lại, nhóm lớn nhất
có 33 dòng. Chia dữ liệu theo dòng sẽ đặt cùng một tin vào cả tập huấn luyện lẫn
tập kiểm tra; mô hình ghi nhớ thay vì khái quát hoá, và điểm số thu được không phản
ánh năng lực thật.

**Khoảng trống 3 — rò rỉ nhãn lương nằm ngay trong văn bản đầu vào.** Mô tả công
việc và đặc biệt là phần phúc lợi thường nhắc lại chính con số lương. Đo trên bộ dữ
liệu, **10,65 %** số dòng có con số lương xuất hiện trong trường phúc lợi. Một mô
hình hồi quy đọc trường này là mô hình đang đọc đáp án của chính nó — và lỗi này
**không sinh ra thông báo lỗi nào**, nó chỉ tạo ra một điểm số đẹp vô nghĩa.

### 2.4.2. Đề xuất ý tưởng thực hiện

Từ ba khoảng trống trên, đề tài đề xuất bốn quyết định thiết kế, mỗi quyết định
nhắm vào một khoảng trống và đều kiểm chứng được:

| # | Đề xuất | Nhắm vào | Kiểm chứng bằng |
|---|---|---|---|
| 1 | Hệ thống **mốc cơ sở hai tầng**: mốc ngây thơ không mô hình, mốc dò tuyến tính trên chính vectơ | Khoảng trống 1 | Bảng mốc ở Chương 4 |
| 2 | **Chia dữ liệu theo nhóm tin trùng lặp gần**, kèm khẳng định không nhóm nào bị tách | Khoảng trống 2 | `groups_straddling_splits = 0` trong `manifest.json` |
| 3 | **Che số lương** trong văn bản đầu vào của bài toán lương, qua một cửa duy nhất trong mã nguồn | Khoảng trống 3 | Bộ kiểm thử `tests/test_no_leak.py` |
| 4 | **Đo phần tín hiệu ngành nghề dùng chung với lương**, thay vì giả định nó tồn tại | §2.2.3 | eta² = 0,032 |

Bốn quyết định này chuyển đề tài từ "xây một mô hình và báo cáo điểm số" sang "xây
một quy trình mà mỗi bước đều có bằng chứng đo được" — đúng tinh thần của Quy tắc 2
trong giao thức thực nghiệm: **bước nào không chứng minh được đóng góp thì bị gỡ
bỏ**.

> **Nguồn số liệu:** `manifest.json` (`unique_groups` = 34.899,
> `groups_straddling_splits` = 0) · [01-data-audit.md §2 và §3](../01-data-audit.md)
> (26,8 % đăng lại, 10,65 % phúc lợi có số lương) ·
> [05-phan-tich-du-lieu.md](../05-phan-tich-du-lieu.md) §7 (5,70 · 5,57, `artifacts/eda/summary.json`) · §8 (eta² 0,032).

---

[← Chương 1](02-chuong-1-gioi-thieu.md) · [Chương 3 →](04-chuong-3-phuong-phap.md)
