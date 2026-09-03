[← Chính quy hoá](07-chinh-quy-hoa.md) · [Nền tảng](00-index.md)

# 8. Đọc một bảng so sánh mô hình

Note này không kể kết quả. Nó dạy cách **đọc** một bảng so sánh — và cách nhận ra
lúc nào bảng đó đang nói quá. Con số thật nằm ở
[10-so-sanh-mo-hinh.md](../10-so-sanh-mo-hinh.md).

Đây là note khó nhất trong bảy note nền tảng, vì nó không nói về mô hình mà nói
về **cách ta đánh giá mô hình** — chỗ mà một người đã biết code vẫn có thể sai
suốt nhiều năm mà không ai chỉ ra.

---

## 1. Siêu tham số — nút vặn bạn phải tự đặt

Khi nấu ăn, **nguyên liệu** là thứ bạn có sẵn, nhưng **nhiệt độ lò** là thứ bạn
phải tự chọn. Không công thức nào tính ra nhiệt độ đúng từ nguyên liệu. Cách duy
nhất là nướng thử vài nhiệt độ rồi nếm.

Mô hình học máy có hai loại số hoàn toàn khác nhau:

| Loại | Ai quyết định | Ví dụ |
|---|---|---|
| **Trọng số** | Mô hình **tự học** từ dữ liệu | Mỗi từ nặng bao nhiêu với mỗi nghề |
| **Siêu tham số** | **Bạn** đặt trước khi mô hình bắt đầu học | `C`, `k`, số cây, độ sâu cây |

Mô hình học được trọng số. Nó **không** học được siêu tham số — bạn phải đưa cho
nó trước, rồi nó mới bắt đầu.

> ⚠️ Đây là chỗ nhầm phổ biến nhất của người mới: tưởng "huấn luyện mô hình" là
> tìm ra **mọi** con số. Không phải. Huấn luyện tìm trọng số; siêu tham số là
> việc của bạn, và tìm chúng là một vòng lặp bên ngoài.

---

## 2. Một "ô" là một lần huấn luyện

Vì siêu tham số phải thử, ta lập một **bảng**: cột là mô hình, hàng là cấu hình.

| | Cấu hình 1 | Cấu hình 2 | Cấu hình 3 |
|---|---|---|---|
| **Mô hình A** | ô | ô | ô |
| **Mô hình B** | ô | ô | ô |

**Mỗi ô = một lần huấn luyện mô hình từ đầu.** Bảng 6 mô hình × 6 cấu hình = 36 ô
= 36 lần huấn luyện.

Tên chuyên môn: cái bảng gọi là **lưới siêu tham số**, chạy hết nó gọi là **quét
lưới** (grid search).

Đây là đơn vị để nói về công bằng: *mỗi mô hình được bao nhiêu ô?*

---

## 3. Cùng số ô vẫn chưa công bằng

Cho hai người mỗi người sáu lần ném phi tiêu. Người thứ nhất ném vào bảng có một
vòng tròn; người thứ hai ném vào bảng có tám vòng chồng lên nhau ở tám hướng
khác nhau. Sáu lần ném của người thứ nhất phủ gần hết bảng của họ. Sáu lần ném
của người thứ hai gần như chưa chạm vào đâu cả.

Số nút vặn của mỗi thuật toán rất khác nhau:

| Thuật toán | Số nút vặn thật sự | 6 ô nghĩa là gì |
|---|---|---|
| KNN | **1** (`k`) | Quét gần hết vùng hữu ích |
| SVM · LogReg | **1** (`C`) | Gần hết một đường cong một chiều |
| RandomForest | ~3 | Một phần |
| XGBoost | **~8** | Một mẩu rất nhỏ của không gian 8 chiều |

Cùng sáu ô, nhưng một bên là khảo sát gần đầy đủ, một bên là bốc thăm sáu lần.

**Cách nhận ra ai đã thật sự được khám phá:** nhìn **biên độ** — khoảng cách giữa
ô tốt nhất và ô tệ nhất của cùng một mô hình. Biên độ hẹp nghĩa là sáu ô cho gần
như cùng kết quả, tức ta **chưa di chuyển** trong không gian của nó.

---

## 4. Cực đại ở mép lưới = lưới bị cắt cụt

Bạn đo nhiệt độ trong nhà từ 8h tới 12h và thấy nóng nhất lúc 12h. Kết luận "12h
là lúc nóng nhất trong ngày" **sai** — bạn chỉ chưa đo tới 13h.

Trong bảng siêu tham số cũng vậy. Nếu cấu hình thắng nằm ở **giá trị nhỏ nhất**
hoặc **lớn nhất** mà bạn đã thử, thì rất có thể đỉnh thật nằm ngoài lưới.

| Tình huống | Đọc thế nào |
|---|---|
| Đỉnh nằm **giữa** lưới, hai bên đều thấp hơn | ✅ Đã tìm thấy đỉnh |
| Đỉnh nằm ở **mép** lưới | ⚠️ Lưới cắt cụt — phải nới rộng rồi đo lại |

Đây là phép kiểm rẻ nhất bạn có, và nó chỉ tốn một cái liếc mắt.

---

## 5. Lời nguyền của người thắng

Cho 20 người mỗi người tung một đồng xu 10 lần. Người ra nhiều mặt ngửa nhất có
thể được 9/10. Bạn có kết luận người đó tung xu giỏi không? Không — bạn đã **chọn
người thắng sau khi nhìn kết quả**, nên kết quả của họ luôn đẹp hơn khả năng thật.

Chọn cấu hình tốt nhất trên tập `val` rồi báo cáo nó **trên chính tập `val`** mắc
đúng lỗi đó. Con số thu được **lạc quan hơn** thực tế.

Tên gọi: **lời nguyền của người thắng** (winner's curse).

Và độ thổi phồng **tỉ lệ với biên độ**: mô hình có sáu ô trải rộng có nhiều cơ hội
trúng một ô may hơn mô hình có sáu ô sát nhau. Nên khi hai mô hình gần bằng nhau,
mô hình **trải rộng hơn** đang được thổi phồng nhiều hơn — khoảng cách thật giữa
chúng nhỏ hơn con số bạn nhìn thấy.

---

## 6. So trung vị thay vì so cực đại

Đây là mẹo nghề rẻ nhất trong cả note này.

**Cực đại** (ô tốt nhất) bị lời nguyền của người thắng. **Trung vị** (ô nằm giữa)
thì không — nó bỏ qua cả may lẫn rủi.

Cách dùng: xếp hạng các mô hình **hai lần**, một lần theo cực đại, một lần theo
trung vị.

| Kết quả | Nghĩa là |
|---|---|
| Hai bảng xếp hạng **giống nhau** | Kết luận vững |
| Hai bảng **đảo ngôi** ở vài vị trí | Ở những vị trí đó, khoảng cách **không thật** — chúng thực chất hoà |

Không tốn thêm một giây tính toán nào, vì bạn đã có sẵn cả sáu con số.

---

## 7. σ so với paired bootstrap — hai câu hỏi khác nhau

Đây là mục quan trọng nhất, và cũng là chỗ dễ dùng sai nhất.

### Vấn đề

Mô hình A được 0,61, mô hình B được 0,60. A có thật sự tốt hơn không, hay chênh
lệch đó chỉ là may rủi của việc tập kiểm tra tình cờ có những tin nào?

### Cách thứ nhất — σ (độ lệch chuẩn), **kém nhạy**

Lấy lại mẫu (bootstrap): từ 7.159 tin trong tập val, bốc ngẫu nhiên **có hoàn
lại** ra 7.159 tin, chấm điểm, lặp 1.000 lần. Độ lệch chuẩn của 1.000 điểm đó là σ.

σ trả lời: *"nếu tôi có một tập val khác, điểm này dao động bao nhiêu?"*

Rồi quy tắc thô: chênh lệch nhỏ hơn σ thì coi là nhiễu.

### Cách thứ hai — paired bootstrap, **nhạy hơn nhiều**

Dùng **cùng một** danh sách tin đã bốc cho **cả hai** mô hình, rồi lấy hiệu điểm.
Lặp 1.000 lần, đếm xem A thắng bao nhiêu lần.

Kết quả là `P(A > B)`:

| `P(A > B)` | Đọc thế nào |
|---|---|
| ~0,500 | **Không phân biệt được** — hai mô hình hoà |
| > 0,975 | A hơn B **rõ ràng** |
| < 0,025 | B hơn A rõ ràng |
| 0,7 – 0,9 | Nghiêng về A nhưng **chưa đủ để khẳng định** |

### Vì sao cách thứ hai nhạy hơn

Hai mô hình cùng đọc **cùng những tin** đó, và chúng sai ở **phần lớn cùng những
tin nhập nhằng**. Phần dao động chung ấy có mặt trong σ của **cả hai**, làm cả hai
σ phình to.

Khi lấy hiệu trên cùng một mẫu, phần chung **triệt tiêu**. Chỉ còn lại phần hai
mô hình thật sự khác nhau.

Hình dung: đo chiều cao hai người bằng một cái thước cong. Đo riêng từng người,
sai số của thước làm cả hai con số không đáng tin. Nhưng nếu bắt hai người **đứng
cạnh nhau** rồi so, cái cong của thước không còn ảnh hưởng — bạn vẫn biết chắc ai
cao hơn.

> ⚠️ **Đừng dùng σ để so hai mô hình.** σ là thanh sai số của **một** mô hình
> đứng một mình. Câu bạn cần hỏi là "A có hơn B **trên cùng những dòng đó** không",
> và chỉ paired bootstrap trả lời được. Dùng σ làm ngưỡng là quá bảo thủ — nó
> chôn cả những khác biệt thật.

---

## 8. Viết kết luận có điều kiện

Sau tất cả những điều trên, câu **"thuật toán X là tốt nhất cho bài toán này"**
gần như luôn là câu nói quá. Nó bỏ qua: ngân sách tìm kiếm, cách biểu diễn đặc
trưng, ai đặt lưới, và tập nào được dùng để chọn.

| Đừng viết | Hãy viết |
|---|---|
| "SVM là thuật toán tốt nhất" | "Dưới ngân sách *n* cấu hình mỗi thuật toán, trên biểu diễn *X*, chọn trên `val`: A và B không phân biệt được (`P = ...`), cả hai vượt nhóm C (`P > 0,975`)" |
| "Mô hình đạt 0,61" | "0,61 trên `val`, là cấu hình tốt nhất trong 6 — con số này lạc quan; trung vị 6 ô là 0,59" |
| Giấu ô chạy hỏng | "Hai cấu hình không hoàn thành trong 45 phút; điều đó thuộc cột chi phí" |

Nghe có vẻ dè dặt, nhưng nó **mạnh hơn**. Người chấm đánh giá cao người biết giới
hạn kết quả của mình — vì đó là dấu hiệu bạn hiểu mình đã đo cái gì, chứ không
phải chỉ chạy được code.

---

## Bảng tra nhanh

| Thấy cái này | Nghĩ ngay tới |
|---|---|
| Hai mô hình chênh nhau rất ít | Xem `P(A > B)`, đừng xem σ (mục 7) |
| Một mô hình có biên độ 6 ô rất hẹp | Ta chưa khám phá không gian của nó (mục 3) |
| Cấu hình thắng ở giá trị nhỏ nhất/lớn nhất đã thử | Lưới cắt cụt (mục 4) |
| Xếp hạng đổi khi so trung vị | Khoảng cách không thật (mục 6) |
| Con số đẹp bất thường | Có phải đã chọn trên chính tập dùng để báo cáo? (mục 5) |
| "Mô hình này không có `class_weight`" | Với lớp lệch, đó là bất lợi thật, không phải chi tiết nhỏ |

---

## Quay lại thực tế dự án

- [10-so-sanh-mo-hinh.md](../10-so-sanh-mo-hinh.md) — bảng so sánh thật, đọc bằng đúng những quy tắc trên
- [03-protocol.md](../03-protocol.md) §7 — quy tắc quyết định chính thức của dự án
- [note 6 — đo lường và baseline](06-do-luong-va-baseline.md) — macro-F1 và vì sao accuracy nói dối
- [note 7 — chính quy hoá](07-chinh-quy-hoa.md) — `C` là siêu tham số điển hình
