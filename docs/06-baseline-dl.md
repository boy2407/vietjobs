[← Tổng quan](00-tong-quan.md) · [← Phân tích dữ liệu](05-phan-tich-du-lieu.md) · [Bài toán lương →](07-bai-toan-luong.md)

# Baseline học sâu — PhoBERT đóng băng + dense

Hai bài toán, hai mạng riêng biệt.

---

## 1. Kiến trúc

```mermaid
%%{init:{'theme':'base','themeVariables':{
  'primaryColor':'#EFF3F1','primaryTextColor':'#141F1D','primaryBorderColor':'#54625E',
  'lineColor':'#54625E','fontSize':'13px','fontFamily':'Be Vietnam Pro, Segoe UI, sans-serif',
  'clusterBkg':'#FFFFFF','clusterBorder':'#C3CFCB','edgeLabelBackground':'#FFFFFF',
  'secondaryColor':'#FFFFFF','tertiaryColor':'#FFFFFF','mainBkg':'#FFFFFF',
  'nodeTextColor':'#141F1D','titleColor':'#141F1D'}}}%%
flowchart LR
    classDef default fill:#FFFFFF,stroke:#54625E,stroke-width:1.5px,color:#141F1D
    T["<b>tin tuyển dụng</b><br/>tiêu đề . mô tả . yêu cầu<br/>đã tách từ"]
    P["<b>PhoBERT-base-v2</b><br/>135 triệu tham số<br/><i>đóng băng</i> · 256 token<br/>mean pooling → 768 chiều"]
    CR[".npy họ <b>raw</b>"]
    CM[".npy họ <b>masked</b>"]
    DA["<b>dense</b> (mạng phân loại)<br/>LayerNorm → 768→256 → GELU<br/>→ dropout → 256→128 → GELU"]
    DB["<b>dense</b> (mạng lương)<br/>LayerNorm → 768→256 → GELU<br/>→ dropout → 256→128 → GELU"]
    A["<b>head A</b><br/>128 → 16 lớp<br/>cross-entropy"]
    B["<b>head B</b><br/>128 → 1<br/>huber trên log1p"]

    T --> P
    P -->|"cột raw"| CR --> DA --> A
    P -->|"cột *_masked"| CM --> DB --> B

    classDef frozen fill:#EFF3F1,stroke:#54625E,stroke-width:1.5px
    class P,CR,CM frozen
```

**Head A và head B sống trong hai mạng khác nhau**, mỗi mạng có khối dense và
bộ đệm vector riêng: phân loại đọc họ `raw`, lương đọc họ `masked`.

## 2. Năm quyết định, và lý do đo được cho mỗi quyết định

| Quyết định | Lý do |
|---|---|
| **Đóng băng PhoBERT** | Vector tính một lần rồi lưu đệm, mọi lần huấn luyện dense đọc lại. Nếu bản đóng băng không vượt nổi sàn và probe tuyến tính, lỗi nhiều khả năng nằm ở đường dữ liệu — phát hiện ở đây rẻ hơn nhiều |
| **Đầu vào đã tách từ** | PhoBERT được tiền huấn luyện trên văn bản đã tách từ ("nhân_viên kinh_doanh"). Đưa vào văn bản chưa tách từ nghĩa là đưa sai phân phối ngay từ lớp đầu tiên |
| **Mean pooling, không dùng vector `<s>`** | Khi không tinh chỉnh, vector `<s>` của PhoBERT chưa từng được huấn luyện cho bất kỳ tác vụ nào; lấy trung bình các token giữ lại nhiều tín hiệu từ vựng hơn |
| **Cache vector vào `.npy`** | Trọng số đóng băng ⇒ các vector không đổi giữa các epoch. Nhúng 34.354 tin `train` chạy một lần (T1.2); mọi lần huấn luyện sau đọc lại cache. Từ 2026-09-17, mỗi file `.npy` có một sidecar `.json` cùng tên ghi lớp tokenizer (`use_fast=False`), `max_len`, cột nguồn và số dòng — bằng chứng để so hai cache, không phải bắt buộc để đọc lại vector |
| **Dùng Huber cho hồi quy, không dùng MSE** | Lương lệch phải mạnh (skew **11,90**, tối đa 500 triệu trên file gốc — [05 §4](05-phan-tich-du-lieu.md#4-phân-phối-lương--đuôi-dài-và-đó-là-lý-do-dùng-log1p)). MSE để cho một nhúm giá trị ngoại lai kéo lệch toàn bộ gradient |

Hai họ cache được tách riêng — `raw` cho phân lớp, `masked` cho lương. Đó
không phải là chi tiết cài đặt: trộn hai file này là một **rò rỉ (leak) lương
âm thầm**, một mô hình đọc chính đáp án của mình.
[`dl/text.py`](../src/vietjobs/dl/text.py) đi qua
`features.resolve_column` — cổng duy nhất quyết định cột, và
[`tests/test_dl_text.py`](../tests/test_dl_text.py) canh giữ điều đó.

## 3. Các ngưỡng cần vượt qua

| Bài toán | Ngưỡng | Nguồn |
|---|---|---|
| Phân lớp — sàn tuyệt đối | dev macro-F1 **0,0214** (luôn dự đoán lớp lớn nhất) | [05 §7](05-phan-tich-du-lieu.md#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev) |
| Phân lớp — **ngưỡng thật** | dev macro-F1 **0,5898** (probe tuyến tính trên chính bộ vector) | dòng `probe-cat-0920` trong [04-results.md](04-results.md) |
| Lương — **ngưỡng thật** | dev MAE **5,70 triệu** (dự đoán trung vị, 13,0, cho tất cả) | [05 §7](05-phan-tich-du-lieu.md#7-ngưỡng-sàn--các-con-số-quyết-định-khớp-trên-train-chấm-điểm-trên-dev) |
| Lương — trần "biết ngành" | dev MAE **5,57 triệu** (khi biết nhãn ngành thật) | như trên |

Con số cuối cùng gây ấn tượng nhất: nhãn ngành **gần như không nói lên điều gì
về lương** (eta² = 0,032). Nếu head hồi quy chỉ đạt mức 5,70 triệu, nó chưa
học được gì từ văn bản cả — nó chỉ đang dự đoán trung vị bằng một đường vòng.

## 4. Cách chạy

```bash
# 1. nhúng (embed) một lần cho mỗi họ cột (~20 phút cho mỗi 33 nghìn tin trên CPU)
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode --task category --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode --task salary   --splits train dev

# 2. huấn luyện phần dense (vài giây mỗi epoch)
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --class-weight
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary
```

Môi trường được tách riêng — lý do và cách thiết lập nằm trong
[`requirements-dl.txt`](../requirements-dl.txt).

Mỗi lượt chạy ghi ra `artifacts/<run_id>/` chứa `config.json`, `env.json`,
`history.jsonl` (mỗi dòng một epoch), `metrics.json`, `best.pt`,
`predictions_dev.parquet`, và **một** dòng trong [04-results.md](04-results.md).
Chi tiết ở [03-protocol.md §3b](03-protocol.md#3b-học-sâu-ghi-thêm-những-gì).

## 5. Kết quả

Lược đồ chia tách đổi vào ngày 2026-09-09 (`train:test` = 8:2, sau đó
`train:dev` = 9:1 → 34.354 / 3.812 / 9.541). **Các con số đo trước ngày đó
không so sánh được với các con số đo sau đó**, nên mỗi bảng dưới đây ghi rõ nó
thuộc lược đồ nào. Lược đồ v2 tính điểm trên `dev` (3.812 tin cho phân lớp);
lược đồ v1 tính điểm trên tập khi đó gọi là `val` (7.159 tin cho phân lớp;
5.095 cho hồi quy). Mỗi dòng đều có `run_id` trong [04-results.md](04-results.md)
và một thư mục `artifacts/<run_id>/`.

### 5.1 Phân lớp nghề nghiệp

**Lược đồ chia tách v2 — tính điểm trên `dev`, 3.812 tin.** Đây là các con số
hiện tại.

| Run | Cấu hình | macro-F1 | F1 | acc | epoch tốt nhất |
|---|---|---|---|---|---|
| *sàn* | luôn dự đoán lớp đa số | *0,0214* | *0,0705* | *0,2062* | — |
| `probe-cat-0920` | LogReg trên **cùng** bộ vector | 0,5898 | 0,6271 | 0,6388 | — |
| **`dl-cat-ce-cpu-0920`** | CrossEntropyLoss, lr 3e-4 + clipping + chuẩn hóa | **0,6030** | **0,6413** | 0,6501 | 11/19 |
| `dl-cat-focal-cpu-0920` | như trên, thay bằng focal loss (γ=2) | 0,5972 | 0,6343 | 0,6427 | 11/19 |

Ba lần chạy này đo trên `cpu` và **tái lập được từng chữ số** — xem §5.5 về lý
do điều đó không hiển nhiên. `F1` là F1 trung bình có trọng số theo số mẫu mỗi
lớp; `macro-F1` cho mọi lớp trọng số bằng nhau và vẫn là chỉ số dùng để chọn
mô hình.

`f1_macro_no_junk` — macro-F1 khi loại `nhóm_nghề_khác` — là **0,6458** đối với
`dl-cat-ce-cpu-0920` và 0,6395 đối với `dl-cat-focal-cpu-0920`. Riêng lớp 25
mẫu đó làm con số đầu bảng mất 0,043.

Head dense chỉ đạt hơn **0,0132** macro-F1 so với probe tuyến tính trên cùng
bộ vector (0,6030 so với 0,5898). Phần lớn tín hiệu mà một mô hình tuyến tính
trích được, head phi tuyến cũng trích được; dung lượng thêm vào chỉ mua được
một biên độ nhỏ, ổn định, chứ không phải một chế độ khác.

**Focal loss thua CrossEntropy 0,0058 macro-F1** (0,5972 so với 0,6030).
`FocalLoss` trong `dl/focal_loss.py` (bản itakurah/focal-loss-pytorch) nhân cross-entropy từng mẫu với `(1-p_t)^γ` —
cân theo *độ khó* của mẫu thay vì *độ hiếm* của lớp, tức là đúng thứ được kỳ
vọng sẽ giúp trên bộ dữ liệu lệch 27:1. Nó không giúp. Khoảng cách 0,0058 lớn
hơn biên nhiễu 0,003 đo ở §5.5 nên đây là kết luận thật, không phải dao động.

**Lược đồ chia tách v1 — tính điểm trên `val`, 7.159 tin. Dữ liệu lịch sử,
không so sánh được với bảng trên.**

| Run | Cấu hình | macro-F1 | acc | epoch tốt nhất |
|---|---|---|---|---|
| `dl-cat-h256` | lr 1e-3 · không chuẩn hóa · không clip gradient | **0,0420** | 0,2127 | 3/11 — **phân kỳ** |
| `dl-cat-h256-cw` | như trên + trọng số theo lớp | 0,0747 | 0,1904 | 3/11 — **phân kỳ** |
| `probe-cat` | LogReg trên **cùng** bộ vector | 0,5867 | 0,6423 | — |
| `dl-cat-v2-nostd` | lr 3e-4 + clipping · không chuẩn hóa | 0,5934 | 0,6466 | 31/39 |
| **`dl-cat-v2`** | lr 3e-4 + clipping + chuẩn hóa | **0,5987** | 0,6493 | 14/22 |
| `dl-cat-v2-cw` | như trên + trọng số theo lớp | 0,5637 | 0,5913 | 6/14 |

`--class-weight` là một **sự đánh đổi, không phải một cải thiện**: macro-F1
giảm 0,0315 (`dl-cat-s2` 0,6025 → `dl-cat-s2-cw` 0,5710). Nó kéo mô hình về
phía các lớp nhỏ, đúng như thiết kế. Mặc định là tắt. Hai lần chạy này đo trên
`mps` ngày 15/09, nên chênh lệch 0,0315 đáng tin (lớn hơn nhiều biên nhiễu
0,003 ở §5.5) nhưng từng con số riêng lẻ thì không tái lập được.

### 5.2 Ước lượng lương

**Lược đồ chia tách v2 — tính điểm trên `dev`, 2.698 tin có công khai mức
lương.** Đây là các con số hiện tại.

| Run | Phương pháp | MAE (triệu) | RMSE (triệu) | R²(log) |
|---|---|---|---|---|
| *ngưỡng* | dự đoán trung vị, 13,0, cho tất cả | *5,70* | *10,52* | *−0,059* |
| *trần "biết ngành"* | trung vị theo ngành, dùng nhãn thật | *5,57* | *10,33* | *−0,022* |
| `probe-sal-0920` | Ridge trên vector PhoBERT | 4,42 | 8,36 | 0,466 |
| **`dl-sal-cpu-0920`** | SmoothL1Loss (Huber), dense(256), lr 3e-4, đã chuẩn hóa | **4,13** | **8,13** | **0,515** |

RMSE lấy từ `rmse_trieu` trong `artifacts/dl-sal-cpu-0920/metrics.json`, từ `rmse`
trong `artifacts/eda/summary.json` cho hai mốc, và cho `probe-sal-0920` từ dòng
của chính lần chạy đó trong [04-results.md](04-results.md). RMSE gấp đôi MAE vì bị
lỗi lớn ở đuôi lương cao kéo lên: trên `predictions_dev.parquet` của `dl-sal-s2`,
2.571 tin dưới 30 triệu có RMSE 4,49, còn 127 tin từ 30 triệu trở lên có RMSE 32,43.

Head hồi quy **vượt ngưỡng 1,57 triệu (−27,5 %)**: R² trên thang log đi từ âm
lên **0,515**, nghĩa là mô hình đọc được tín hiệu lương từ văn bản mà riêng
nhãn ngành không cung cấp được ([05 §6](05-phan-tich-du-lieu.md)). Đây là biên
độ rõ ràng hơn so với kết quả lược đồ v1 bên dưới (−17,6 %) — tập v2 nhỏ hơn,
tách rời theo nhóm, vẫn cho cùng kiến trúc này tổng quát hóa tốt hơn ở đây.

**Kết quả probe tuyến tính đảo ngược phát hiện ở v1.** Ở v1, Ridge trên cùng
bộ vector (`probe-sal`) đạt điểm **tệ hơn cả dự đoán trung vị** (MAE 6,60) —
kết luận khi đó là tín hiệu lương tồn tại trong các vector nhưng chỉ ở dạng
phi tuyến. Ở v2, `probe-sal-0920` đạt R² **0,466**, đã đi được phần lớn quãng
đường tới 0,515 của head dense, và rõ ràng tốt hơn ngưỡng.

**Lược đồ chia tách v1 — tính điểm trên `val`, 5.095 tin. Dữ liệu lịch sử,
không so sánh được với bảng trên.**

| Run | Phương pháp | MAE (triệu) | R²(log) |
|---|---|---|---|
| *ngưỡng* | dự đoán trung vị, 13,0, cho tất cả | *5,86* | *−0,063* |
| *trần "biết ngành"* | trung vị theo ngành, dùng nhãn thật | *5,75* | *−0,032* |
| `probe-sal` | Ridge trên vector PhoBERT | 6,60 | −0,004 |
| `dl-sal-v2` | dense(256), 40 epoch | 4,83 | 0,381 |
| `dl-sal-v3-long` | cùng cấu hình, patience 15 | 4,88 | 0,357 |

Hai lượt chạy v1 cùng cấu hình cho ra 4,83 và 4,88: **biến thiên giữa các lượt
chạy vào khoảng 0,05 triệu**, nên đừng đọc một chênh lệch nhỏ hơn thế như một
cải thiện.

Đáng chú ý từ v1: Ridge trên cùng bộ vector cho ra **6,60** — *tệ hơn cả dự
đoán trung vị*. Cùng đặc trưng, cùng nhãn; khác biệt là khối dense chuẩn hóa
đầu vào và học một hàm phi tuyến. Ở v1 kết luận là tín hiệu nằm trong các vector
nhưng không ở dạng tuyến tính; ở v2 kết luận này đảo ngược (bảng trên).

### 5.3 Nó sai ở đâu

Phân lớp (`dl-cat-v2`, đọc từ `predictions_dev.parquet`):

| Lớp | F1 | n |
|---|---|---|
| `nhóm_nghề_khác` | **0,000** | 48 |
| `kỹ_thuật_điện_điện_tử_viễn_thông` | 0,411 | 207 |
| `thiết_kế_nghệ_thuật_giải_trí…` | 0,441 | 470 |
| `tài_chính_kế_toán_ngân_hàng_bảo_hiểm` | **0,851** | 713 |

Lớp gộp chung (catch-all) `nhóm_nghề_khác` bị bỏ rơi hoàn toàn — đúng như
[05 §2](05-phan-tich-du-lieu.md) đã dự đoán: 48 mẫu, nội dung lẫn lộn. Ba cặp
bị nhầm lẫn nhiều nhất đều là **ranh giới nhãn thật sự mờ nhòe**, không phải
lỗi mô hình:

| Nhầm lẫn | Số tin |
|---|---|
| `kinh_doanh…` → `du_lịch_nhà_hàng…` | 202 |
| `du_lịch_nhà_hàng…` → `kinh_doanh…` | 180 |
| `thiết_kế_nghệ_thuật…` → `xây_dựng_kiến_trúc…` | 133 |

Hồi quy (`dl-sal-v2`): MAE **3,70 triệu** trên các tin ≤ 30 triệu (4.833 tin)
nhưng **25,58 triệu** trên các tin > 30 triệu (262 tin). Mô hình dự đoán tối
đa **80,7** trong khi dữ liệu có một tin ở mức 275 triệu — nó **không dám đi
vào đuôi phải**, một hệ quả trực tiếp của việc huấn luyện trên `log1p` với hàm
mất mát huber. Ngành khó nhất là `nhóm_nghề_khác` (MAE 11,03), dễ nhất là
`nông_nghiệp_năng_lượng_môi_trường` (2,67).

### 5.4 Thất bại đầu tiên, giữ lại làm bằng chứng

Hai lượt chạy đầu tiên (`dl-cat-h256`, `dl-cat-h256-cw`) cho ra macro-F1
**0,042** — thấp hơn cả việc thay vector PhoBERT bằng số ngẫu nhiên.
`history.jsonl` cho biết lý do chỉ trong bốn dòng:

```
epoch 1  loss 2.63    val_macro_f1 0.028
epoch 2  loss 2.57    val_macro_f1 0.033
epoch 3  loss 2.65    val_macro_f1 0.042
epoch 4  loss 321.6   val_macro_f1 0.011   ← blew up
epoch 5  loss 1941.8  val_macro_f1 0.019
```

Vòng lặp huấn luyện phân kỳ tại epoch 4. Hai nguyên nhân, cả hai đều đo được:

1. **Các vector PhoBERT có tính dị hướng (anisotropic).** Cosin giữa hai tin
   tuyển dụng bất kỳ có trung vị **0,896** (p05 0,838 · p95 0,943) — mọi tin
   đều nằm trong một hình nón hẹp. `LayerNorm` chuẩn hóa *theo từng mẫu*, nên
   nó không thể loại bỏ hướng chung đó.
2. **lr 1e-3 quá lớn** cho loại đầu vào này, khi không có clip gradient.

Cách sửa: chuẩn hóa theo từng chiều bằng thống kê của `train` (lưu thành
`scaler.npz` cùng với lượt chạy, vì đường suy luận phải dùng chính xác các con
số đó), clip chuẩn gradient (gradient norm) ở 1,0, hạ lr xuống 3e-4. Kết quả:
**0,042 → 0,5987**.

Bài học đáng giá hơn cả con số: `probe-cat` (LogReg trên cùng bộ vector,
macro-F1 **0,5867**) chính là thứ phân biệt "đặc trưng kém" với "head mô hình
bị hỏng". Một mô hình tuyến tính có nghiệm lồi và không có tốc độ học nào để
sai — nếu nó đạt 0,59, vấn đề chắc chắn không nằm ở đặc trưng. Chạy lại bằng
`python scripts/probe_embeddings.py --task category`.

> **Về cột thời gian trong [04-results.md](04-results.md):** các lượt chạy
> này chồng lấn nhau trên cùng một máy 6 lõi, nên số giây **không so sánh
> được giữa các dòng**. Một epoch trên các vector đã cache tốn khoảng 10–15
> giây trên máy rảnh.


---

[← Phân tích dữ liệu](05-phan-tich-du-lieu.md) · [Lộ trình →](09-lo-trinh.md)

### 5.5 Tái lập: vì sao mọi lần chạy đều đặt trên `cpu`

Ngày 2026-09-20, khi chạy lại mô hình nền để đo bằng bộ chỉ số mới,
`dl-cat-s2` (chạy 15/09, macro-F1 0,6025) **không tái lập được**: cùng seed,
cùng dữ liệu, cùng mã nguồn, kết quả ra 0,6013.

Truy nguyên bằng cách dựng một git worktree tại đúng commit `12a847c1` của lần
chạy cũ và chạy **mã nguồn nguyên bản** ngày hôm đó: cũng ra 0,6013. Vậy không
phải do thay đổi mã. Loại trừ tiếp: cache vector (không đổi từ 13/09), split và
`manifest.json` (không đổi từ 10/09), kiến trúc, seed, `torch` 2.2.2, cùng chuỗi
`platform`.

Nguyên nhân là **thiết bị**. Trên `mps`, tám lần chạy cùng một cấu hình cho:

| Thiết bị | macro-F1 qua các lần chạy | Thời gian |
|---|---|---|
| `mps` | 0,6012 · 0,6013 · 0,6013 · 0,6013 · 0,6025 · 0,6041 · 0,6041 · 0,6041 | 13–15 s |
| `cpu` | 0,6030 · 0,6030 · 0,6030 (giống từng chữ số) | 12–16 s |

Biên dao động của `mps` là **0,003 macro-F1** — lớn hơn nhiều khác biệt mà một
thí nghiệm nhỏ muốn đo. `cpu` cho đúng một số mỗi lần và **không chậm hơn**:
head dense chỉ 768→256→128→16, quá nhỏ để GPU có lợi ích. Từ đó `train_dl.py`
mặc định `--device cpu` (AGENTS.md Rule 13 điều 5).

Ba hệ quả được ghi lại ở đây vì chúng ràng buộc cách đọc mọi bảng phía trên:

1. Các con số `*-cpu-0920` tái lập được; các con số `-s2` (đo trên `mps`) thì
   không, và được giữ lại như dữ liệu lịch sử.
2. Trên các lần chạy `mps` cũ, **mọi chênh lệch dưới 0,003 macro-F1 là nhiễu**.
   Hai kết luận vẫn đứng vững vì vượt xa biên đó: `--class-weight` (−0,0315) và
   focal loss (−0,0058).
3. `env.json` từ nay ghi thêm `numpy`, `pandas`, `scikit_learn` — ba thư viện
   mà việc không ghi lại đã khiến lần truy nguyên này mất nhiều bước hơn cần
   thiết.

Lần chạy lương không bị ảnh hưởng theo cách này: `dl-sal-s2` tái lập trên `mps`
tới từng chữ số cuối (MAE 4,14868613775178). Nhưng chuyển sang `cpu` vẫn đổi kết
quả (4,13, hội tụ ở epoch 19 thay vì 26) — hai thiết bị là hai đường số học khác
nhau, nên bảng §5.2 dùng lần chạy `cpu` cho nhất quán.

