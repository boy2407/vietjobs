# Giao thức thí nghiệm

Tài liệu này là hợp đồng. Mọi con số trong `04-results.md` chỉ có nghĩa nếu các
quy tắc dưới đây được tôn trọng.

## 1. Tập dữ liệu đã đóng băng

`data/processed/splits/{train,val,test}.parquet` được tạo một lần bởi
`python -m vietjobs.dataset build`, với `SPLIT_SEED = 20260826`.

**Không được đổi seed hoặc đổi tỷ lệ chia sau khi đã huấn luyện mô hình đầu tiên.**
Đổi là mọi dòng trong `04-results.md` không còn so sánh được với nhau, và bảng
ablation — sản phẩm chính của báo cáo — trở nên vô nghĩa.

Nếu bắt buộc phải dựng lại, kiểm tra manifest không đổi:

```bash
python -c "
import json; m=json.load(open('data/processed/manifest.json'))
print(m['unique_groups'], {k:v['rows'] for k,v in m['splits'].items()})"
```

### Chia theo nhóm, không theo dòng

Khoảng 28% mô tả trong bộ dữ liệu này không duy nhất — nhà tuyển dụng đăng lại
nhiều lần. Chia theo dòng sẽ đặt cùng một tin ở cả train và test; mô hình chỉ cần
ghi nhớ là đạt điểm cao, và điểm đó không phản ánh khả năng tổng quát hoá.

Vì vậy: băm `(tiêu đề + mô tả + yêu cầu)` sau khi gấp dấu → `group_id`, và
**cả nhóm** đi cùng một tập. `dataset.build()` có assert kiểm tra không nhóm nào
nằm ở quá một tập; assert này không được phép tắt.

### Phân tầng theo nghề

Ba lớp nhỏ nhất chỉ có 196–258 dòng. Không phân tầng thì chúng có thể vắng mặt
hoàn toàn ở val hoặc test, và macro-F1 tính trên số lớp khác nhau giữa các lần chạy.

## 2. Chính sách chạm tập test

| Tập | Dùng để | Được chạm bao nhiêu lần |
|---|---|---|
| `train` | Huấn luyện | Không giới hạn |
| `val` | Chọn mô hình, chọn siêu tham số, chọn bước tiền xử lý | Không giới hạn |
| `test` | Báo cáo con số cuối cùng | **Đúng một lần**, ở cuối |

`train.py` từ chối chạy `--eval test` nếu không có cờ `--confirm-test`. Cờ này
tồn tại để việc chạm vào test là một hành động có chủ ý, không phải mặc định.

Mỗi lần nhìn vào test rồi quay lại chỉnh mô hình là một lần rò rỉ thông tin từ
test vào quyết định thiết kế. Làm vài lần thì test không còn là ước lượng
không thiên lệch nữa.

## 3. Log chỉ ghi thêm

`docs/04-results.md` là log **append-only**. Mỗi lần `train.py` chạy sẽ ghi đúng
một dòng: run id, thời gian, task, model, scope, cấu hình tiền xử lý, tập đánh giá,
số mẫu, metric, thời gian huấn luyện, git sha.

**Không sửa dòng đã ghi.** Một thí nghiệm thất bại vẫn là dữ liệu — nó ghi lại
những gì đã thử và không hiệu quả, và đó chính là thứ ngăn ta thử lại lần nữa
sau ba tuần.

Mỗi dòng phải tái lập được từ `(run_id, seed, git sha)`. Nếu `git_dirty` là true
thì dòng đó không tái lập được và phải được coi là chỉ mang tính tham khảo.

## 4. Metric

### Phân lớp nghề nghiệp

**Chính: macro-F1.** Lớp lớn nhất gấp 27 lần lớp nhỏ nhất. Accuracy sẽ vui vẻ
che giấu một mô hình bỏ qua hoàn toàn phần đuôi.

Báo cáo kèm:
- `f1_macro_no_junk` — macro-F1 trên 15 lớp, bỏ `nhóm_nghề_khác`. Lớp đó là
  thùng rác chứa "Nhân Viên Seo Web", "Nhân Viên Quản Trị Website" — những tin
  lẽ ra thuộc marketing và IT. Đó là **nhiễu nhãn, không phải lỗi mô hình**;
  báo cáo cả hai con số giúp tách bạch hai chuyện.
- `balanced_accuracy`, `accuracy`, `top3_accuracy`.
- Ma trận nhầm lẫn và 10 cặp nhầm lẫn nhiều nhất.

### Hồi quy lương

Mọi con số **quy về triệu VND/tháng**. MAE trong không gian log không phải là
con số ai có thể hành động được.

## 5. Hai trục thí nghiệm

Ba thuật toán × sáu bước tiền xử lý là 18 lần chạy, mà KNN trên full text mất
10–30 phút mỗi lần. Không khả thi. Vì vậy:

**Trục 1 — bậc thang tiền xử lý.** Cố định *một* thuật toán rẻ và mạnh
(`svm`) làm thước đo, chỉ bật/tắt từng bước xử lý tiếng Việt. Chốt ra cấu hình
tốt nhất, gọi là `PREP*`.

**Trục 2 — so sánh thuật toán.** Cả ba chạy trên đúng `PREP*`, cùng đặc trưng,
cùng seed.

Trộn hai trục lại sẽ không biết cải thiện đến từ tiền xử lý hay từ thuật toán.

## 7. So sánh mô hình — quy tắc quyết định

Thêm sau cụm quét công bằng ([10-so-sanh-mo-hinh.md](10-so-sanh-mo-hinh.md)).
Ba điều ràng buộc mọi so sánh mô hình từ đây:

**a. Cùng số cấu hình cho mọi mô hình.** So "cái tốt nhất trong chín lần thử"
với "lần thử đầu tiên" là so công sức chỉnh tay, không phải so thuật toán. Nếu
một mô hình được quét `n` cấu hình thì mọi mô hình trong cùng bảng phải được `n`.

**b. Mỗi run lưu `predictions.npz`.** `train.py` ghi `row_index`, `y_true`,
`y_pred` trên tập đánh giá. Không có nó thì run đó **không so cặp được** với bất
kỳ run nào khác — đó là lý do 33 run trước cụm phải chạy lại từ đầu.

**c. Quyết định bằng paired bootstrap, không bằng σ.** Dùng **một** ma trận chỉ
số dùng chung (`evaluate.bootstrap_indices`) cho mọi mô hình được so, rồi đọc
`P(A > B)` từ `evaluate.paired_delta`. Ngưỡng: **>0,975 hoặc <0,025** là khác
biệt rõ; quanh 0,5 là không phân biệt được.

σ của từng mô hình (≈ 0,0077) vẫn được báo cáo, nhưng **không phải là cơ sở
quyết định**: nó trả lời "điểm này dao động bao nhiêu nếu đổi tập val", còn câu
cần hỏi là "A có hơn B trên cùng những dòng đó không". Hai mô hình sai ở phần
lớn cùng những tin nhập nhằng, nên phép so cặp nhạy hơn hẳn.

**Chọn mô hình trên `val`, không bao giờ trên `test`** — kể cả để "kiểm chứng"
kết quả một cụm quét. Quy tắc §2 không có ngoại lệ.

---

## 8. Quy tắc gỡ bỏ

Mỗi bước tiền xử lý phải có một dòng ablation chứng minh nó đóng góp.
**Bước nào không cải thiện thì gỡ bỏ**, không giữ lại vì "trông có vẻ đúng" hay
vì "sách bảo thế". Một pipeline ngắn mà mỗi bước đều có bằng chứng thì tốt hơn
một pipeline dài đầy nghi lễ.
