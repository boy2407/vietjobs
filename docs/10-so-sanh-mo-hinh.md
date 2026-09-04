[← Tổng quan](00-tong-quan.md) · [← Mô hình phân lớp](06-mo-hinh-phan-lop.md) · [Lộ trình →](09-lo-trinh.md)

# So sánh mô hình — cụm quét công bằng

**Mười một thuật toán, sáu cấu hình mỗi cái**, cùng tập train, cùng đặc trưng,
cùng seed, chấm trên `val`. 63/66 ô chạy xong.

Cụm này tồn tại vì bảng so sánh trước nó **không công bằng**: `svm` từng được quét
chín giá trị `C` rồi lấy cái tốt nhất, còn ba mô hình cây mỗi cái chạy đúng **một**
cấu hình do người viết đoán.

> **Chưa quen các khái niệm bên dưới?** Đọc
> [nền tảng: đọc một bảng so sánh](nen-tang/08-doc-mot-bang-so-sanh.md) trước —
> nó giải thích "ô", siêu tham số, lời nguyền của người thắng, và vì sao
> `P(A > B)` mới là con số để quyết định chứ không phải σ.

---

## 1. Kết luận

**Giữ `LinearSVC C=0,02`.** Không phải vì nó điểm cao nhất — mà vì không mô hình
nào hơn nó một cách phân biệt được, và trong nhóm dẫn đầu nó rẻ nhất.

Năm tầng, tách bằng paired bootstrap:

| Tầng | Mô hình | macro-F1 | Trong tầng |
|---|---|---|---|
| **1** | `logreg` 0,6072 · **`svm` 0,6050** · `sgd` 0,6010 | 0,60–0,61 | hoà nhau |
| **2** | `svm_plain` 0,5927 · `xgb` 0,5912 · `lgbm` 0,5830 | 0,58–0,59 | hoà nhau |
| **3** | `extra` 0,5665 · `rf` 0,5644 · `nb` 0,5535 | 0,55–0,57 | hoà nhau |
| **4** | `knn` 0,4361 | | thua tầng trên rõ |
| **5** | `centroid` 0,3204 | | thua tất cả |

Baseline không dùng ML (trùng từ khoá tiêu đề) đạt **0,4321** — nằm giữa tầng 4 và
tầng 5. **`knn` và `centroid` đều thua nó.**

---

## 2. Cách cụm được dựng

```mermaid
%%{init:{'theme':'base','themeVariables':{
  'primaryColor':'#EFF3F1','primaryTextColor':'#141F1D','primaryBorderColor':'#54625E',
  'lineColor':'#54625E','fontSize':'13px','fontFamily':'Be Vietnam Pro, Segoe UI, sans-serif',
  'clusterBkg':'#FFFFFF','clusterBorder':'#C3CFCB','edgeLabelBackground':'#FFFFFF',
  'secondaryColor':'#FFFFFF','tertiaryColor':'#FFFFFF','mainBkg':'#FFFFFF',
  'nodeTextColor':'#141F1D','titleColor':'#141F1D'}}}%%
flowchart TD
    classDef default fill:#FFFFFF,stroke:#54625E,stroke-width:1.5px,color:#141F1D
    G["<b>Lưới 11 × 6</b><br/>sweep_category.py<br/>mỗi mô hình 6 ô"]
    W["<b>Xếp theo wave</b><br/>ô #j của mọi mô hình<br/>trước ô #j+1"]
    R["<b>63 run tuần tự</b><br/>một tiến trình cha<br/>subprocess.run"]
    P["<b>predictions.npz</b><br/>y_true · y_pred · row_index<br/>7.159 dòng val mỗi run"]
    I["<b>MỘT ma trận chỉ số</b><br/>bootstrap_indices<br/>1.000 × 7.159"]
    SG["σ từng mô hình<br/>0,0056 – 0,0087"]
    D["<b>P(A > B)</b><br/>paired_delta<br/>so trên CÙNG dòng"]
    CH["<b>Chọn mô hình</b>"]

    G --> W --> R --> P
    P --> I
    I --> SG
    I --> D
    SG -.->|"chỉ để báo cáo"| CH
    D ==>|"quyết định"| CH

    classDef acc fill:#E2F0EC,stroke:#0E6B5B,stroke-width:2px,color:#0E6B5B
    classDef warn fill:#F8EDE2,stroke:#9E5C22,stroke-width:2px,color:#9E5C22
    class I,D acc
    class SG warn
```

**Ma trận chỉ số dùng chung là điểm mấu chốt.** Cùng 1.000 mẫu lấy lại được áp cho
cả 63 mô hình, nên mọi so sánh là **so cặp**. Vì hai mô hình sai ở phần lớn cùng
những tin nhập nhằng, phần phương sai chung bị triệt tiêu và phép so nhạy hơn hẳn.

---

## 3. Tái lập — sáu trên sáu

Ô #1 của sáu họ đầu lặp lại một dòng đã có trong [04-results.md](04-results.md).
Lệch một chữ số là cả cụm đáng ngờ.

| Mô hình | mỏ neo (dòng cũ) | đo lại | lệch |
|---|---|---|---|
| `logreg` | 0,6038 | 0,6038 | -0,0000 ✅ |
| `svm` | 0,6050 | 0,6050 | -0,0000 ✅ |
| `xgb` | 0,5861 | 0,5861 | +0,0000 ✅ |
| `lgbm` | 0,5555 | 0,5555 | +0,0000 ✅ |
| `rf` | 0,5630 | 0,5630 | -0,0000 ✅ |
| `knn` | 0,4059 | 0,4059 | -0,0000 ✅ |

Năm họ thêm sau không có mỏ neo vì chưa từng chạy trước đó.

---

## 4. Ma trận P(hàng > cột)

| | `logreg` | `svm` | `sgd` | `svm_plain` | `xgb` | `lgbm` | `extra` | `rf` | `nb` | `knn` | `centroid` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `logreg` | — | 0,685 | 0,987 | 0,982 | 0,995 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| `svm` | 0,315 | — | 0,829 | 0,949 | 0,988 | 0,999 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| `sgd` | 0,013 | 0,171 | — | 0,895 | 0,953 | 0,999 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| `svm_plain` | 0,018 | 0,051 | 0,105 | — | 0,546 | 0,890 | 0,999 | 1,000 | 1,000 | 1,000 | 1,000 |
| `xgb` | 0,005 | 0,012 | 0,047 | 0,454 | — | 0,854 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| `lgbm` | 0,000 | 0,001 | 0,001 | 0,110 | 0,146 | — | 0,989 | 0,996 | 1,000 | 1,000 | 1,000 |
| `extra` | 0,000 | 0,000 | 0,000 | 0,001 | 0,000 | 0,011 | — | 0,697 | 0,946 | 1,000 | 1,000 |
| `rf` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,004 | 0,303 | — | 0,920 | 1,000 | 1,000 |
| `nb` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,054 | 0,080 | — | 1,000 | 1,000 |
| `knn` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | — | 1,000 |
| `centroid` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | — |

> 0,500 = không phân biệt được. >0,975 hoặc <0,025 = khác biệt rõ.

Ba quan sát đáng chú ý nhất:

**a. `svm_plain` ngang `xgb`** (0,546). Tắt `class_weight` làm SVM tụt đúng bằng
khoảng cách từ SVM tới XGBoost. Một cờ boolean đắt ngang cả một họ thuật toán.

**b. `sgd` ngang `svm`** (0,829 — chưa tới ngưỡng 0,975). Học từng mẫu một, không
giải chính xác, mà vẫn không tách được khỏi `LinearSVC`. Nhưng `logreg` **thắng**
`sgd` rõ (0,987), nên `sgd` không thay được vị trí dẫn đầu.

**c. `extra` ngang `rf`** (0,697). Chọn ngưỡng tách **ngẫu nhiên** không tệ hơn đi
tìm ngưỡng tốt nhất — trên ma trận 99,79% số 0, phần lớn ngưỡng ứng viên vô nghĩa
nên việc tìm kiếm chẳng mua được gì.

---

## 5. Giá của điểm số

| Mô hình | macro-F1 | fit | F1 / phút | so với rẻ nhất |
|---|---|---|---|---|
| `knn` | 0,4361 | 17s | 1,567 | 1,0× |
| `centroid` | 0,3204 | 18s | 1,060 | 1,1× |
| `nb` | 0,5535 | 24s | 1,367 | 1,5× |
| `sgd` | 0,6010 | 40s | 0,901 | 2,4× |
| `svm` | 0,6050 | 42s | 0,861 | 2,5× |
| `svm_plain` | 0,5927 | 42s | 0,840 | 2,5× |
| `logreg` | 0,6072 | 4,8 ph | 0,127 | 17,1× |
| `rf` | 0,5644 | 5,3 ph | 0,107 | 18,9× |
| `extra` | 0,5665 | 6,4 ph | 0,089 | 23,0× |
| `lgbm` | 0,5830 | 12,9 ph | 0,045 | 46,2× |
| `xgb` | 0,5912 | 18,7 ph | 0,032 | 67,2× |

`xgb` tốn **67 lần** thời gian của `knn` để về tầng 2. `nb` đạt 0,5535 trong **24
giây** — cùng tầng với `rf` (5,3 phút) và `extra` (6,4 phút).

---

## 6. Độ nhạy siêu tham số

Biên độ trong 6 cấu hình. Biên độ rộng nghĩa là mô hình phụ thuộc nặng
vào việc chỉnh tay — và con số của nhà vô địch lạc quan hơn tương ứng,
vì nó là cái tốt nhất trong 6 lần thử **chọn trên chính tập val**.

| Mô hình | tệ nhất | trung vị | tốt nhất | biên độ |
|---|---|---|---|---|
| `logreg` | 0,5519 | 0,5936 | 0,6072 | 0,0553 |
| `svm` | 0,5865 | 0,5980 | 0,6050 | 0,0185 |
| `sgd` | 0,5234 | 0,5569 | 0,6010 | 0,0776 |
| `svm_plain` | 0,5460 | 0,5766 | 0,5927 | 0,0467 |
| `xgb` | 0,5861 | 0,5874 | 0,5912 | 0,0051 |
| `lgbm` | 0,5555 | 0,5655 | 0,5830 | 0,0275 |
| `extra` | 0,5497 | 0,5642 | 0,5665 | 0,0168 |
| `rf` | 0,5502 | 0,5629 | 0,5644 | 0,0142 |
| `nb` | 0,3582 | 0,5431 | 0,5535 | 0,1954 |
| `knn` | 0,3736 | 0,4235 | 0,4361 | 0,0624 |
| `centroid` | 0,3182 | 0,3200 | 0,3204 | 0,0022 |

Đọc cột cuối, không đọc cột "tốt nhất":

- **`centroid` biên độ 0,0022** — sáu ô gần như trùng nhau. `shrink_threshold`
  **không làm gì cả** trên dữ liệu này.
- **`nb` biên độ 0,1954** — lớn nhất bảng. `alpha` quyết định gần như toàn bộ:
  từ 0,3582 (`alpha=10`) tới 0,5535 (`alpha=0,1`). Một giá trị `alpha` đặt bừa
  biến Naive Bayes từ "dùng được" thành "vô dụng".
- **`xgb` biên độ 0,0051** — sáu ô gần như trùng nhau, tức ta **chưa di chuyển**
  trong không gian 8 nút vặn của nó.
- **`logreg` trải 0,0553, gấp ba `svm` (0,0185)** — nên 0,6072 của nó bị lời
  nguyền của người thắng thổi phồng nhiều hơn 0,6050 của `svm`. Khoảng cách thật
  **nhỏ hơn** +0,0022 đo được.

### Kiểm độ vững: so trung vị thay vì so cực đại

| Mô hình | Tốt nhất | Trung vị | Thứ hạng |
|---|---|---|---|
| `logreg` | **0,6072** ① | 0,5936 ② | ⚠️ **tụt** |
| `svm` | 0,6050 ② | **0,5980** ① | ⚠️ **lên** |
| `sgd` | 0,6010 ③ | 0,5569 ⑧ | ⚠️ **tụt 5 bậc** |
| `xgb` | 0,5912 ⑤ | 0,5874 ③ | lên |
| `svm_plain` | 0,5927 ④ | 0,5766 ④ | giữ |

**Đổi ngôi ở hai vị trí đầu**, và `sgd` tụt năm bậc. Đây là bằng chứng mạnh hơn
con số `P = 0,685`: thứ hạng **không bền** trước cách chọn thống kê, tức ba mô
hình tuyến tính thực sự hoà.

---

## 7. Điểm yếu của từng mô hình

Mỗi mục dựa trên số đo trong chính cụm này, không dựa trên lý thuyết.

| Mô hình | Gãy ở đâu | Bằng chứng |
|---|---|---|
| **`centroid`** | Nén 33.396 tin thành **16 vectơ trung bình**. Một nghề có nhiều kiểu tin khác nhau thì trung bình của chúng không giống tin nào cả | 0,3204 — thua cả baseline từ khoá. Biên độ 0,0022: không nút vặn nào cứu được |
| **`knn`** | Lời nguyền số chiều: ở 236.596 chiều, khoảng cách tới hàng xóm gần nhất và xa nhất hội tụ về nhau | Toàn văn 0,4059 **tệ hơn** chỉ tiêu đề 0,5307. Thua baseline từ khoá |
| **`nb`** | Giả định đặc trưng là **số đếm**, nhưng đây là TF-IDF đã chuẩn hoá L2 — giả định bị vi phạm. Không có `class_weight` | Biên độ `alpha` 0,1954, lớn nhất bảng: mô hình cực kỳ nhạy vì giả định sai |
| **`rf` · `extra`** | Cây tách theo **một chiều mỗi lần**; với 480 ô khác 0 trên 236.596 cột, phần lớn phép tách rơi vào vùng toàn số 0 | Thêm cây gần như vô ích: 100→600 cây chỉ +0,0014. Ngẫu nhiên hoá ngưỡng (`extra`) cũng ngang (P=0,697) |
| **`lgbm` · `xgb`** | Cùng vấn đề của cây, cộng chi phí. `xgb` có cực đại nằm ở **mép lưới** (`depth=4`, `colsample=0,1` đều là giá trị nhỏ nhất đã thử) | `xgb` 18,7 ph để về tầng 2. Ở 400 vòng/depth 8 bị **dừng sau 4h07m chưa xong** |
| **`sgd`** | Rất nhạy với `alpha` và `loss`; trung vị tụt năm bậc so với cực đại | Biên độ 0,0776. `alpha=1e-3` cho 0,5234, `alpha=1e-4` cho 0,6010 |
| **`svm_plain`** | Không có `class_weight` ở lệch 27:1 | balAcc **0,5763** so với 0,6713 của `svm` — mất 0,095. Nhưng accuracy lại **cao hơn** (0,6560 vs 0,6445) |
| **`logreg`** | Đắt (4,8 ph so với 42s) và **nhạy siêu tham số** — biên độ 0,0553 | Trung vị 0,5936, tụt xuống hạng 2 sau `svm` |
| **`svm`** | Không cho xác suất trực tiếp; `predict.py` đang softmax điểm biên và gọi là "score" | Cột `top3` trống ở mọi ô `svm` trong Bảng 1 |

### `class_weight` mua được gì — cặp đối chứng sạch nhất trong bảng

`svm` và `svm_plain` khác **đúng một cờ**:

| | macro-F1 | balanced acc | accuracy |
|---|---|---|---|
| `svm` (`class_weight="balanced"`) | **0,6050** | **0,6713** | 0,6445 |
| `svm_plain` (`class_weight=None`) | 0,5927 | 0,5763 | **0,6560** |

Bỏ `class_weight` làm **accuracy tăng** 0,0115 và **balanced accuracy giảm**
0,0950. Đó chính là hành vi mà [nền tảng note 6](nen-tang/06-do-luong-va-baseline.md)
cảnh báo: mô hình bỏ rơi lớp hiếm để làm đẹp con số tổng. Nếu dự án chọn mô hình
bằng accuracy thì `svm_plain` đã thắng — và 15 nghề hiếm sẽ mờ dần khỏi hệ thống.

---

## 8. σ — lần đầu được tính bằng code

| Mô hình | Cấu hình thắng | macro-F1 | Δ so quán quân | P(Δ>0) | fit |
|---|---|---|---|---|---|
| **`logreg`** | `C=0.5` | **0,6072** | — | — | 4,8 ph |
| `svm` | `C=0.02` | 0,6050 | -0,0024 | 0,315 | 42s |
| `sgd` | `loss=log_loss,alpha=0.0001` | 0,6010 | -0,0062 | 0,013 | 40s |
| `svm_plain` | `C=0.05` | 0,5927 | -0,0150 | 0,018 | 42s |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=4,colsample_bytree=0.1` | 0,5912 | -0,0163 | 0,005 | 18,7 ph |
| `lgbm` | `n_estimators=200,learning_rate=0.15,num_leaves=31` | 0,5830 | -0,0245 | 0,000 | 12,9 ph |
| `extra` | `n_estimators=600` | 0,5665 | -0,0410 | 0,000 | 6,4 ph |
| `rf` | `n_estimators=600` | 0,5644 | -0,0431 | 0,000 | 5,3 ph |
| `nb` | `alpha=0.1` | 0,5535 | -0,0541 | 0,000 | 24s |
| `knn` | `n_neighbors=5` | 0,4361 | -0,1714 | 0,000 | 17s |
| `centroid` | `shrink_threshold=None` | 0,3204 | -0,2866 | 0,000 | 18s |

---

## 9. Toàn bộ 63 cấu hình

| Mô hình | Cấu hình | macro-F1 | σ | F1@15 | balAcc | top3 | fit |
|---|---|---|---|---|---|---|---|
| `logreg` | `C=1.0` | 0,6038 | ±0,0074 | 0,6355 | 0,6419 | 0,9318 | 5,3 ph |
| `logreg` | `C=0.5` | **0,6072** | ±0,0070 | 0,6416 | 0,6554 | 0,9293 | 4,8 ph |
| `logreg` | `C=0.25` | 0,6005 | ±0,0070 | 0,6344 | 0,6590 | 0,9225 | 3,3 ph |
| `logreg` | `C=0.1` | 0,5867 | ±0,0067 | 0,6206 | 0,6569 | 0,9102 | 2,3 ph |
| `logreg` | `C=0.02` | 0,5519 | ±0,0067 | 0,5831 | 0,6370 | 0,8631 | 1,6 ph |
| `logreg` | `C=0.05` | 0,5768 | ±0,0067 | 0,6102 | 0,6535 | 0,8948 | 2,1 ph |
| `svm` | `C=0.02` | **0,6050** | ±0,0071 | 0,6376 | 0,6713 | — | 42s |
| `svm` | `C=0.05` | 0,6030 | ±0,0071 | 0,6374 | 0,6600 | — | 39s |
| `svm` | `C=0.01` | 0,5976 | ±0,0070 | 0,6308 | 0,6711 | — | 26s |
| `svm` | `C=0.1` | 0,5983 | ±0,0074 | 0,6298 | 0,6455 | — | 46s |
| `svm` | `C=0.005` | 0,5865 | ±0,0068 | 0,6180 | 0,6665 | — | 26s |
| `svm` | `C=0.2` | 0,5873 | ±0,0076 | 0,6188 | 0,6231 | — | 47s |
| `sgd` | `loss=hinge,alpha=0.0001` | 0,5618 | ±0,0070 | 0,5932 | 0,6318 | — | 72s |
| `sgd` | `loss=hinge,alpha=1e-05` | 0,5521 | ±0,0077 | 0,5824 | 0,5669 | — | 61s |
| `sgd` | `loss=log_loss,alpha=0.0001` | **0,6010** | ±0,0069 | 0,6371 | 0,6467 | 0,9211 | 40s |
| `sgd` | `loss=modified_huber,alpha=0.0001` | 0,5804 | ±0,0077 | 0,6110 | 0,6071 | 0,8785 | 52s |
| `sgd` | `loss=hinge,alpha=0.0001,penalty=elasticnet` | 0,5455 | ±0,0065 | 0,5785 | 0,6528 | — | 77s |
| `sgd` | `loss=hinge,alpha=0.001` | 0,5234 | ±0,0062 | 0,5531 | 0,6575 | — | 30s |
| `svm_plain` | `C=0.02` | 0,5818 | ±0,0082 | 0,6173 | 0,5608 | — | 58s |
| `svm_plain` | `C=0.05` | **0,5927** | ±0,0078 | 0,6290 | 0,5763 | — | 42s |
| `svm_plain` | `C=0.1` | 0,5902 | ±0,0076 | 0,6222 | 0,5784 | — | 46s |
| `svm_plain` | `C=0.5` | 0,5714 | ±0,0081 | 0,6001 | 0,5669 | — | 80s |
| `svm_plain` | `C=1.0` | 0,5539 | ±0,0082 | 0,5819 | 0,5487 | — | 1,7 ph |
| `svm_plain` | `C=0.005` | 0,5460 | ±0,0065 | 0,5843 | 0,5249 | — | 27s |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=6` | 0,5861 | ±0,0083 | 0,6157 | 0,6105 | 0,9239 | 39,2 ph |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=4` | 0,5895 | ±0,0076 | 0,6221 | 0,6310 | 0,9205 | 20,0 ph |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=4,colsample_bytree=0.1` | **0,5912** | ±0,0076 | 0,6238 | 0,6323 | 0,9184 | 18,7 ph |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=6,colsample_bytree=0.1` | 0,5874 | ±0,0079 | 0,6202 | 0,6068 | 0,9204 | 30,0 ph |
| `xgb` | `n_estimators=100,learning_rate=0.3,max_depth=5,colsample_bytree=0.2` | 0,5861 | ±0,0077 | 0,6191 | 0,6166 | 0,9240 | 24,8 ph |
| `lgbm` | `n_estimators=100,learning_rate=0.3` | 0,5555 | ±0,0082 | 0,5849 | 0,5476 | 0,9163 | 10,5 ph |
| `lgbm` | `n_estimators=100,learning_rate=0.3,num_leaves=31` | 0,5692 | ±0,0083 | 0,5998 | 0,5683 | 0,9169 | 11,1 ph |
| `lgbm` | `n_estimators=100,learning_rate=0.3,colsample_bytree=0.1` | 0,5579 | ±0,0085 | 0,5851 | 0,5458 | 0,9158 | 5,2 ph |
| `lgbm` | `n_estimators=200,learning_rate=0.15,num_leaves=31` | **0,5830** | ±0,0081 | 0,6142 | 0,5821 | 0,9274 | 12,9 ph |
| `lgbm` | `n_estimators=100,learning_rate=0.3,num_leaves=95` | 0,5618 | ±0,0082 | 0,5917 | 0,5533 | 0,9154 | 35,3 ph |
| `lgbm` | `n_estimators=100,learning_rate=0.3,colsample_bytree=0.5` | 0,5708 | ±0,0084 | 0,5992 | 0,5644 | 0,9166 | 14,9 ph |
| `extra` | `n_estimators=300` | 0,5650 | ±0,0079 | 0,5936 | 0,6071 | 0,9133 | 5,6 ph |
| `extra` | `n_estimators=600` | **0,5665** | ±0,0076 | 0,5973 | 0,6096 | 0,9159 | 6,4 ph |
| `extra` | `n_estimators=300,max_features=1000` | 0,5642 | ±0,0077 | 0,5949 | 0,6024 | 0,9124 | 6,0 ph |
| `extra` | `n_estimators=300,criterion=entropy` | 0,5497 | ±0,0080 | 0,5792 | 0,5787 | 0,8969 | 2,8 ph |
| `extra` | `n_estimators=300,max_features=log2` | 0,5507 | ±0,0076 | 0,5750 | 0,6458 | 0,8869 | 30s |
| `rf` | `n_estimators=100` | 0,5630 | ±0,0079 | 0,5935 | 0,5844 | 0,9014 | 55s |
| `rf` | `n_estimators=300` | 0,5629 | ±0,0076 | 0,5932 | 0,5861 | 0,9121 | 3,4 ph |
| `rf` | `n_estimators=600` | **0,5644** | ±0,0078 | 0,5949 | 0,5880 | 0,9166 | 5,3 ph |
| `rf` | `n_estimators=300,max_features=1000` | 0,5606 | ±0,0080 | 0,5909 | 0,5810 | 0,9149 | 2,7 ph |
| `rf` | `n_estimators=300,max_features=log2` | 0,5502 | ±0,0073 | 0,5800 | 0,6369 | 0,8793 | 22s |
| `nb` | `alpha=1.0` | 0,5371 | ±0,0077 | 0,5720 | 0,5246 | 0,9174 | 26s |
| `nb` | `alpha=0.1` | **0,5535** | ±0,0080 | 0,5869 | 0,5437 | 0,9155 | 24s |
| `nb` | `alpha=0.01` | 0,5491 | ±0,0079 | 0,5822 | 0,5414 | 0,9152 | 20s |
| `nb` | `alpha=10.0` | 0,3582 | ±0,0064 | 0,3832 | 0,3561 | 0,7980 | 20s |
| `nb` | `alpha=1.0,norm=True` | 0,4692 | ±0,0069 | 0,5019 | 0,4518 | 0,8856 | 18s |
| `nb` | `alpha=0.1,norm=True` | 0,5532 | ±0,0079 | 0,5866 | 0,5404 | 0,9134 | 17s |
| `knn` | `n_neighbors=30` | 0,4059 | ±0,0081 | 0,4268 | 0,3783 | 0,8127 | 23s |
| `knn` | `n_neighbors=15` | 0,4262 | ±0,0085 | 0,4460 | 0,4014 | 0,8104 | 21s |
| `knn` | `n_neighbors=5` | **0,4361** | ±0,0082 | 0,4582 | 0,4281 | 0,7739 | 17s |
| `knn` | `n_neighbors=50` | 0,3736 | ±0,0082 | 0,3943 | 0,3452 | 0,8058 | 20s |
| `knn` | `n_neighbors=10` | 0,4360 | ±0,0087 | 0,4571 | 0,4167 | 0,8007 | 17s |
| `knn` | `n_neighbors=3` | 0,4208 | ±0,0079 | 0,4418 | 0,4221 | 0,7322 | 17s |
| `centroid` | `shrink_threshold=None` | **0,3204** | ±0,0056 | 0,3392 | 0,3972 | — | 18s |
| `centroid` | `shrink_threshold=0.1` | 0,3203 | ±0,0056 | 0,3390 | 0,3971 | — | 18s |
| `centroid` | `shrink_threshold=0.2` | 0,3203 | ±0,0056 | 0,3390 | 0,3970 | — | 18s |
| `centroid` | `shrink_threshold=0.5` | 0,3198 | ±0,0056 | 0,3385 | 0,3967 | — | 18s |
| `centroid` | `shrink_threshold=1.0` | 0,3196 | ±0,0056 | 0,3383 | 0,3965 | — | 18s |
| `centroid` | `shrink_threshold=2.0` | 0,3182 | ±0,0056 | 0,3368 | 0,3955 | — | 18s |

---

## 10. Ba ô không hoàn thành, và một họ suýt không chạy được

| Ô | Cấu hình | Chuyện gì |
|---|---|---|
| `rf-4` | 300 cây, `min_samples_leaf=1` | DNF ở timeout 45 phút |
| `xgb-5` | 200@0,15 depth4 `colsample=0,2` | DNF ở timeout 45 phút |
| `extra-4` | 300 cây, `min_samples_leaf=1` | DNF ở timeout 45 phút |

Ba họ cây mỗi họ chỉ có **5/6 ô**. Cả ba ô hỏng đều cho cây mọc sâu không giới hạn
(`min_samples_leaf=1`) hoặc dùng nhiều cột — trên ma trận này chúng mọc tới hết
giờ. Không chạy lại: cả ba họ đều thua nhóm tuyến tính rõ, và việc chúng **không
chạy nổi trong 45 phút** là một kết quả thuộc cột chi phí.

`centroid` còn một chuyện khác. Sáu ô đầu tiên bị **hệ điều hành giết (SIGKILL)**,
không phải timeout: `sklearn.NearestCentroid` làm đặc ma trận đánh giá
7.159 × 236.596 = **13,5 GB** trên máy 16 GB. Phải viết lại bằng phép nhân thưa
(`models.CosineNearestCentroid`) mới chạy được — và bản tự viết còn dùng được
**cosine**, thứ sklearn 1.9 không cho (`metric` chỉ nhận euclidean/manhattan).

---

## 11. Điều cụm này *không* trả lời

- **Mọi con số "tốt nhất" đều là ước lượng lạc quan.** Muốn con số không thiên vị
  phải chấm trên tập chưa từng dùng để chọn — mà `test` đã chạm một lần rồi
  ([03-protocol.md](03-protocol.md) §2). **Không được lấy `test` ra "kiểm chứng".**
- **6 ô cho `centroid` (1 nút vặn) không tương đương 6 ô cho `xgb` (8 nút vặn).**
  Bằng nhau về *số cấu hình* không phải bằng nhau về *độ phủ*. Bảng 6 đo đúng
  chênh lệch đó.
- **Lưới `xgb` bị cắt cụt** — cực đại nằm ở mép. Ba ô ở `depth=3`,
  `colsample=0,05` sẽ trả lời dứt điểm.
- **Chưa có mô hình học sâu.** PhoBERT là [Ưu tiên 4](09-lo-trinh.md).
- Cụm chạy trên không gian **236.596 chiều chưa cắt**. `C` tối ưu rơi tận 0,02
  vẫn là triệu chứng thừa chiều — [Ưu tiên 3b](09-lo-trinh.md) còn nguyên.

---

## Đọc tiếp

- [nền tảng: đọc một bảng so sánh](nen-tang/08-doc-mot-bang-so-sanh.md) — khái niệm dùng trong note này
- [06-mo-hinh-phan-lop.md](06-mo-hinh-phan-lop.md) — bậc thang kết quả và mô hình chốt
- [04-results.md](04-results.md) — toàn bộ dòng thí nghiệm
- [03-protocol.md](03-protocol.md) §7 — quy tắc quyết định
