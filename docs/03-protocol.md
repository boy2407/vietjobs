# Giao thức thực nghiệm

Tài liệu này là bản hợp đồng. Mỗi con số trong `04-results.md` chỉ có ý nghĩa
nếu các quy tắc dưới đây được tuân thủ.

## 1. Bộ dữ liệu đóng băng

`data/processed/splits/{train,dev,test}.csv` được tạo một lần bằng
`python -m vietjobs.dataset build`, với `SPLIT_SEED = 20260826` và lược đồ
hai giai đoạn `train:test = 8:2`, sau đó `train:dev = 9:1`.

**Seed và các tỷ lệ chia không được thay đổi sau khi mô hình đầu tiên đã được
huấn luyện.** Thay đổi chúng có nghĩa là không có dòng nào trong `04-results.md`
có thể so sánh được với dòng khác, và bảng ablation — sản phẩm chính của báo
cáo — trở nên vô nghĩa.

Nếu việc xây dựng lại thực sự không thể tránh khỏi, hãy kiểm tra rằng manifest
không thay đổi:

```bash
python -c "
import json; m=json.load(open('data/processed/manifest.json'))
print(m['unique_groups'], {k:v['rows'] for k,v in m['splits'].items()})"
```

### Chia theo nhóm, không theo dòng

Khoảng 28% mô tả trong bộ dữ liệu này không duy nhất — nhà tuyển dụng đăng lại
nhiều lần. Chia theo dòng sẽ đặt cùng một tin tuyển dụng vào cả train và test;
mô hình đạt điểm cao nhờ học thuộc, và điểm số đó không phản ánh khả năng tổng
quát hóa.

Vì vậy: băm `(title + description + requirements)` sau khi bỏ dấu → `group_id`,
và **toàn bộ nhóm** sẽ vào cùng một tập. `dataset.build()` có một assert đảm
bảo không nhóm nào nằm trong nhiều hơn một tập; assert đó không bao giờ được
phép tắt.

### Phân tầng theo ngành nghề

Ba lớp nhỏ nhất chỉ có 196–258 dòng. Nếu không phân tầng, chúng có thể hoàn
toàn vắng mặt trong dev hoặc test, và macro-F1 sẽ được tính trên số lượng lớp
khác nhau giữa các lần chạy.

## 2. Chính sách chạm vào tập kiểm tra

| Tập | Dùng để | Được phép chạm bao nhiêu lần |
|---|---|---|
| `train` | Huấn luyện | Không giới hạn |
| `dev` | Chọn mô hình, siêu tham số, lựa chọn tiền xử lý | Không giới hạn |
| `test` | Báo cáo con số cuối cùng | **Đúng một lần**, ở cuối |

Tập kiểm định cũng được gọi là **`dev`** trên đĩa và trong các cờ CLI
(`--eval dev`), kể từ 2026-09-09.

**`test` không chỉ hiếm khi bị chạm vào — nó không bao giờ bị biến đổi.** Không
lọc, không khử trùng lặp, không cân bằng lớp, không cắt tỉa ngoại lai, không
chia lại. Mọi việc làm sạch hay điều chỉnh dữ liệu chỉ áp dụng cho `train` và
`dev`. Một khi tập test đã bị chỉnh sửa, con số cuối cùng không còn nói lên
điều gì về dữ liệu thực nữa.

`dl/train_dl.py` từ chối chạy `--eval test` nếu không có cờ `--confirm-test`. Cờ đó
tồn tại để việc chạm vào test là một hành động có chủ ý, chứ không phải mặc
định.

Mỗi lần bạn nhìn vào test rồi quay lại điều chỉnh mô hình, thông tin sẽ rò rỉ
từ test vào một quyết định thiết kế. Làm điều đó vài lần và test không còn là
một ước lượng không thiên lệch nữa.

## 3. Nhật ký chỉ được ghi thêm

`docs/04-results.md` là một nhật ký **chỉ được ghi thêm**. Mỗi lần chạy
`dl/train_dl.py` ghi đúng một dòng: run id, timestamp, task, model, scope, cấu hình
tiền xử lý, tập đánh giá, số lượng mẫu, các chỉ số, thời gian huấn luyện, git
sha.

**Không bao giờ sửa một dòng đã được ghi.** Một thí nghiệm thất bại vẫn là dữ
liệu — nó ghi lại những gì đã thử và không hiệu quả, và đó chính xác là điều
ngăn chúng ta thử lại nó ba tuần sau.

Mọi dòng phải có thể tái lập được từ `(run_id, seed, git sha)`. Nếu `git_dirty`
là true, dòng đó không thể tái lập và phải được coi là chỉ mang tính tham
khảo.

## 3b. Học sâu ghi thêm những gì

Một lần chạy học sâu
trải dài qua nhiều epoch, có thể thất bại giữa chừng, và **lý do thất bại nằm
ở đường cong huấn luyện, không nằm ở con số cuối cùng**. Vì vậy
`dl/train_dl.py` ghi thêm các tệp trong `artifacts/<run_id>/`:

| Tệp | Nó chứa gì | Vì sao bắt buộc |
|---|---|---|
| `history.jsonl` | một dòng JSON cho mỗi epoch: loss, chỉ số `val`, số giây | Nếu không có, không thể phân biệt được dưới khớp / quá khớp / lr sai |
| `config.json` | mọi siêu tham số + seed | Một lần chạy không thể tái lập thì con số của nó không có giá trị |
| `env.json` | device, phiên bản torch/transformers, encoder | Hai lần chạy trên hai máy có thời gian không thể so sánh được |
| `best.pt` | trọng số tại epoch `val` tốt nhất | Nếu không, một lần crash ở giờ thứ hai sẽ làm mất tất cả |
| `predictions_<eval>.parquet` | dự đoán theo từng dòng | Đọc các lỗi thực tế sau khi lần chạy đã kết thúc |

**`04-results.md` vẫn chỉ nhận một dòng cho mỗi lần chạy.** Nhồi mọi epoch vào
đó sẽ phá hỏng khả năng đọc của chính nhật ký.

Cột *Headline* của dòng đó, với tác vụ lương, ghi
`MAE · RMSE · R2log · R2raw · ±20%` (từ 2026-09-20; các dòng trước 2026-09-19
không có `RMSE` và `R2raw`, và các dòng trước 2026-09-20 còn có `MedAE`).
`RMSE` và `R2raw` tính trên thang triệu, là hai số mà bài báo giới thiệu bộ dữ
liệu (arXiv 2603.05262) dùng để báo cáo, nên đặt sẵn trên dòng để so sánh
không phải mở tệp. Với phân lớp, dòng ghi `macroF1 · F1 · acc` (từ
2026-09-20; các dòng trước đó ghi `macroF1 · acc · balAcc · top3`).

Các vector PhoBERT được cache trong `artifacts/embeddings/`, thành hai tệp
chia theo nhóm cột: `raw` cho phân lớp, `masked` cho hai tác vụ lương. Trộn
lẫn hai tệp này là một rò rỉ lương âm thầm — chính điều mà Quy tắc 3 đề
phòng.

## 4. Các chỉ số

### Phân lớp ngành nghề

**Chỉ số chính: macro-F1.** Lớp lớn nhất gấp 27× lớp nhỏ nhất. Accuracy sẽ dễ
dàng che giấu một mô hình bỏ qua hoàn toàn phần đuôi.

Được báo cáo kèm theo:
- `f1_macro_no_junk` — macro-F1 trên 15 lớp, bỏ `nhóm_nghề_khác`. Lớp đó là
  một ngăn kéo tạp chứa "Nhân Viên Seo Web", "Nhân Viên Quản Trị Website" —
  những tin tuyển dụng thực chất thuộc về marketing và IT. Đó là **nhiễu
  nhãn, không phải lỗi mô hình**; báo cáo cả hai con số giữ hai điều này tách
  biệt.
- `f1_weighted` — F1 trung bình có trọng số theo số mẫu mỗi lớp, và
  `accuracy`. Từ 2026-09-20 đây là toàn bộ bộ chỉ số của bài phân lớp;
  `balanced_accuracy` và `top3_accuracy` đã bị bỏ khỏi `evaluate.py`, nên các
  run từ ngày đó không còn ghi hai số này vào `metrics.json`.
- Ma trận nhầm lẫn và 10 cặp lớp bị nhầm lẫn nhiều nhất.

### Hồi quy lương

Mọi con số đều được **báo cáo theo triệu VND/tháng**. Một MAE trong không
gian log không phải là con số mà người ta có thể dựa vào để hành động.

## 7. So sánh mô hình — các quy tắc quyết định

Ba ràng buộc áp dụng cho mọi so sánh mô hình:

**a. Cùng một số lượng cấu hình cho mọi mô hình.** So sánh "tốt nhất trong
chín lần thử" với "lần thử đầu tiên" là so sánh công sức tinh chỉnh thủ công,
không phải thuật toán. Nếu một mô hình được sweep `n` cấu hình, mọi mô hình
trong cùng bảng đó cũng phải được sweep `n` cấu hình.

**b. Mọi lần chạy đều lưu dự đoán từng dòng.** `dl/train_dl.py` ghi
`predictions_<eval>.parquet`. Nếu không có, lần chạy đó **không thể được ghép
cặp** với bất kỳ lần chạy nào khác.

**c. Quyết định bằng paired bootstrap, không bằng σ.** Dùng **một** ma trận
chỉ số dùng chung (`evaluate.bootstrap_indices`) cho mọi mô hình được so
sánh, sau đó đọc `P(A > B)` từ `evaluate.paired_delta`. Ngưỡng: **>0,975
hoặc <0,025** là khác biệt rõ ràng; quanh 0,5 là không thể phân biệt được.

σ theo từng mô hình (≈ 0,0077) vẫn được báo cáo, nhưng nó **không phải là căn
cứ để ra quyết định**: nó trả lời câu hỏi "điểm số này thay đổi bao nhiêu nếu
tập dev thay đổi", trong khi câu hỏi thực sự quan trọng là "A có thắng B trên
chính những dòng đó không". Hai mô hình sai trên phần lớn cùng những tin
tuyển dụng mơ hồ, nên so sánh theo cặp nhạy hơn nhiều.

**Chọn mô hình trên `dev`, không bao giờ trên `test`** — kể cả để "xác nhận"
kết quả của một sweep. §2 không có ngoại lệ.

---

## 8. Quy tắc loại bỏ

Mọi bước tiền xử lý phải có một dòng ablation chứng minh nó có đóng góp.
**Một bước không cải thiện gì thì bị loại bỏ**, không được giữ lại chỉ vì nó
"trông có vẻ đúng" hay vì "sách giáo khoa nói vậy". Một pipeline ngắn mà mọi
bước đều có bằng chứng thì tốt hơn một pipeline dài đầy nghi thức.
