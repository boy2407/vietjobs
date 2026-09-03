---
name: dataset-diagnosis
description: "Soi bộ dữ liệu VietJobs bằng mười phép đo chạy thật trên splits đã đóng băng — lệch lớp, nhóm đăng lại mâu thuẫn nhãn, trần nhiễu nhãn, ô rỗng, phủ từ vựng, độ tách lớp, lớp rác, lỗ hổng che lương, thiên lệch chọn mẫu, lệch địa lý — rồi xếp hạng thành bảng phát hiện kèm hành động làm sạch. Use this skill when the user asks what is wrong with the data, why the ceiling is low, whether the labels are trustworthy, what to clean or relabel, or invokes /dataset-diagnosis."
trigger: "Use this skill when the user asks about dataset quality, label noise, class imbalance, duplicates, missing fields, vocabulary coverage, selection bias, or what data cleaning to do next, or invokes /dataset-diagnosis."
version: 1
---

# Dataset diagnosis — biến điểm yếu của dữ liệu thành việc phải làm

Mô hình đứng yên ở một mức điểm có hai lý do khác hẳn nhau: mô hình chưa đủ, hoặc
dữ liệu không cho phép hơn. Skill này trả lời vế thứ hai bằng số đo, để không ai
tốn một buổi tinh chỉnh cho một trần mà dữ liệu đã chốt sẵn.

Khác với [`model-diagnosis`](../model-diagnosis/SKILL.md): skill đó soi **một run
đã huấn luyện**. Skill này soi **dữ liệu**, chạy được kể cả khi chưa có mô hình nào.
Vài kết luận phải đọc chéo giữa hai skill — chỗ nào cần thì nói rõ.

---

## Luật số 1 — không con số nào được viết ra mà chưa chạy lệnh sinh ra nó

Mỗi ô trong bảng kết quả phải có khối sinh ra nó ở cột kế bên. Chạy không được thì
viết **chưa đo**. Không có "khoảng 30 %", không có "phần lớn".

Mọi đoạn mã dưới đây chạy nguyên văn từ gốc repo, qua Bash, sau phần mở đầu chung:

```python
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from vietjobs import config as C, vitext as V
tr = pd.read_parquet('data/processed/splits/train.parquet')
va = pd.read_parquet('data/processed/splits/val.parquet')
```

## Luật số 2 — chỉ chẩn đoán trên `train`

`val` chỉ dùng khi câu hỏi **là** về tổng quát hoá (A3, A5). `test` không bao giờ,
trừ đúng một ngoại lệ: đếm tần suất regex để đóng lỗ rò rỉ (A8), vì phép đếm đó
không nhìn nhãn và không dẫn tới lựa chọn mô hình nào.

**Không dựng lại splits.** `SPLIT_SEED = 20260826` bị khoá và tách từ tốn khoảng
26 phút. Mọi đề xuất làm sạch là đề xuất cho **lần dựng lại tương lai**, và phải
kèm dòng cảnh báo: làm việc đó sẽ vô hiệu hoá toàn bộ `docs/04-results.md`.

## Luật số 3 — một phát hiện gồm ba phần

Số đo được · khối tái lập · hành động kèm chi phí. Thiếu vế thứ ba thì đó là một
thống kê, không phải một phát hiện — để nó ở mục "đã kiểm tra và sạch".

Và **một phép đo chưa đủ để xếp hạng cao**. Chỉ nâng một phát hiện lên mức đáng làm
ngay khi có hai chỉ dấu độc lập trỏ cùng một chỗ — ví dụ A6 cho cosine cao *và*
`metrics.top_confusions` của run tốt nhất trùng đúng cặp lớp đó.

---

## Mười khối chẩn đoán

Chạy theo thứ tự. A8 rẻ nhất và gỡ chặn nhiều nhất — chạy trước nếu đang vướng
bài toán lương.

### A1 · Lệch lớp theo **nhóm**, không theo dòng

```python
g = tr.groupby("category").agg(rows=("group_id","size"), groups=("group_id","nunique"))
g["dup_x"] = (g.rows / g.groups).round(2)
print(g.sort_values("rows").to_string(), g.rows.max()/g.rows.min(), g.groups.max()/g.groups.min())
```

Dòng đăng lại không phải mẫu độc lập. Ngưỡng: `groups < 300` cho một lớp → lớp đó
không đủ mẫu để F1 ổn định, σ của nó chi phối macro-F1. `dup_x` lệch nhau giữa các
lớp quá 1,15 → `class_weight="balanced"` đang cân theo **dòng**, tức là cân sai.

### A2 · Nhóm đăng lại có mâu thuẫn nhãn — khối quan trọng nhất

```python
sz = tr.groupby("group_id").size(); nc = tr.groupby("group_id")["category"].nunique()
multi = nc[nc > 1].index
print((sz > 1).sum(), (nc > 1).sum(), tr.group_id.isin(multi).mean())
print(tr[tr.group_id == multi[0]][["job_title","category"]].to_string())
```

`group_id` là băm của tiêu đề + mô tả + yêu cầu sau khi gấp dấu. Cùng một văn bản
mang nhiều nhãn nghĩa là bài toán **thực chất đa nhãn** bị ép về đơn nhãn.

Ngưỡng: quá 5 % nhóm nhiều dòng bị đa nhãn → không phải nhiễu ngẫu nhiên. Quá 50 %
dòng nằm trong nhóm đa nhãn → macro-F1 đơn nhãn **đang đo sai bài toán**; hành động
rẻ là báo cáo thêm *top-1-in-set accuracy*, hành động đắt là chuyển đích sang đa
nhãn — và việc đó phải là **trục thứ hai**, không thay trục cũ.

### A3 · Trần nhiễu nhãn tính được trong ba giây

```python
maj = va.groupby("group_id")["category"].agg(lambda s: s.mode().iat[0])
oracle = va["group_id"].map(maj)
print((oracle == va.category).mean(),
      f1_score(va.category, oracle, average="macro", zero_division=0))
```

Đây là cận trên của **mọi** hàm văn bản → nhãn xác định, tính bằng "đoán nhãn đa số
trong nhóm cùng văn bản". Ngưỡng: trần dưới 0,95 → ghi con số này cạnh mọi macro-F1.
Dùng nó để **chặn** việc đuổi theo vài phần trăm bằng mô hình lớn hơn, không dùng để
tự khen. Chạy khối này **trước** khi bỏ hai giờ gán tay ở Ưu tiên 2.

### A4 · Ô rỗng và độ dài theo từng cột

```python
for c in ["job_title","description","requirements_text","qualifications_text",
          "technical_skills_text","soft_skills_text","benefits_text","languages_text"]:
    L = tr[c].fillna("").astype(str).str.len()
    print(c, f"{(L==0).mean():.2%}", L.quantile(.1), L.median(), L.quantile(.9))
```

Ngưỡng: rỗng quá 60 % → khối TF-IDF của cột đó gần như hằng số; **gỡ khối, đo lại**,
theo đúng quy tắc gỡ bước ở `docs/03-protocol.md`. Rỗng 10–60 % → cột đếm `n_*` đang
trộn "thiếu" với "có nhưng bằng 0"; thêm cờ `has_*`.

### A5 · Phủ từ vựng train → val

```python
cv = CountVectorizer(min_df=3).fit(tr.job_title + " " + tr.description)
an, vocab = cv.build_analyzer(), set(cv.vocabulary_)
r = [(sum(w not in vocab for w in an(t)), len(an(t)))
     for t in (va.job_title + " " + va.description)]
print(sum(a for a, _ in r) / sum(b for _, b in r))
```

Ngưỡng: quá 5 % → từ vựng là nút thắt, đáng thử `min_df=1`, char n-gram hoặc
embedding. **Dưới 1 % → OOV không phải nguyên nhân**, và phát hiện ở đây là một
**loại trừ**: ghi vào bảng để không ai đề xuất PhoBERT với lý do "OOV tiếng Việt".

### A6 · Độ tách lớp — cosine giữa tâm lớp

```python
X = TfidfVectorizer(min_df=3, ngram_range=(1,2), sublinear_tf=True).fit_transform(tr.job_title)
labs = sorted(tr.category.unique())
Cm = normalize(np.vstack([np.asarray(X[(tr.category == l).values].mean(axis=0)) for l in labs]))
S = Cm @ Cm.T; np.fill_diagonal(S, 0)
print(sorted(((S[i,j], labs[i], labs[j]) for i in range(len(labs))
              for j in range(i+1, len(labs))), reverse=True)[:8])
```

Ngưỡng: cosine trên 0,80 → hai lớp **không tách được bằng tiêu đề**. Bắt buộc đối
chiếu với `metrics.top_confusions` của run tốt nhất; trùng cặp thì đó là lỗi phân
loại học (taxonomy), không phải lỗi mô hình → đề xuất gộp lớp hoặc ghi trần.
0,65–0,80 → ứng viên cho một đặc trưng phân biệt riêng. Dưới 0,50 → lớp tách tốt,
F1 thấp ở đây là lỗi mô hình chứ không phải lỗi dữ liệu.

### A7 · Thành phần lớp rác

```python
j = tr[tr.category == C.JUNK_CATEGORY]; print(len(j), j.job_title.head(20).tolist())
```

Ngưỡng: F1 của lớp này (đọc `metrics.per_class` trong `artifacts/<run>/metrics.json`)
dưới 0,25 **và** quá 30 % tiêu đề mẫu gán được tay vào một lớp khác → **nó là nhãn
sai, không phải một lớp**. Khi đó `f1_macro_no_junk` — đã có sẵn trong mọi
`metrics.json` — là con số phải dẫn.

### A8 · Lỗ hổng che lương — `docs/09-lo-trinh.md` Ưu tiên 0b

```python
al = pd.concat([pd.read_parquet(f'data/processed/splits/{s}.parquet')
                for s in ("train", "val", "test")])
for c in ["soft_skills_text","qualifications_text","technical_skills_text","languages_text"]:
    hit = al[c].fillna("").map(lambda t: bool(V._PAT_MILLIONS.search(t)
                               or V._PAT_DONG.search(t) or V._PAT_USD.search(t)))
    print(c, int(hit.sum()), f"{hit.mean():.4%}")
```

Ngưỡng đã ghi sẵn trong lộ trình: **bằng 0 → đóng lại**, chỉ cần thêm hai cột vào
`UNMASKED_COLUMNS` (`src/vietjobs/features.py`) cho chắc. **Khác 0 → rò rỉ thật**:
phải dựng bản `_masked`, thêm vào `_MASKABLE`, và mọi số bài toán lương chạy trước
đó phải bỏ. Đây là khối gỡ chặn cả Ưu tiên 1 — chạy nó trước.

### A9 · Phân bố lương và thiên lệch chọn mẫu

```python
d = tr.groupby("category").salary_disclosed.agg(["mean","size"]).sort_values("mean")
s = tr.loc[tr.salary_disclosed == 1, "salary_mid"]
print(d.to_string(), d["mean"].max() - d["mean"].min(), s.median(), s.quantile(.9), s.max())
```

Ngưỡng: chênh lệch tỷ lệ công bố giữa các lớp quá 0,10 → **thiên lệch chọn mẫu
thật**, không sửa được bằng mô hình tốt hơn. Bảng này phải vào
`docs/07-bai-toan-luong.md` **trước** khi chạy dòng nào của bài toán lương, và mọi
MAE phải báo cáo kèm `evaluate.slice_report` theo `category`.

### A10 · Lệch địa lý

```python
p = tr.province.value_counts(); print(p.size, p.head(8).to_dict(), tr.is_major_city.mean())
```

Ngưỡng: số giá trị `province` nhiều hơn 63 → `normalize_province` chưa gom hết; liệt
kê 20 giá trị hiếm nhất và sửa bảng trong `resources/`. Rẻ, và bảng ablation
`--province` đang đo trên một ánh xạ chưa sạch. Top-2 tỉnh quá 75 % → one-hot
`province` gần như một cờ nhị phân; ghi là **hạn chế phạm vi**, không phải phát hiện
mới.

---

## Khuôn kết quả

```markdown
## Soi dữ liệu — <ngày> · <n> dòng train · manifest <sha8>

| # | Phát hiện | Số đo | Khối | Hành động kéo theo | Chi phí | Chặn gì |
|---|---|---|---|---|---|---|

**Ba việc đáng làm trước:** ...
**Đã kiểm tra và sạch:** <giả thuyết + số bác bỏ nó>
**Phải chờ dựng lại splits:** <khối riêng, không trộn vào bảng chính>
```

Xếp hạng đúng thứ tự này: (1) phát hiện nào **chặn** một hạng mục trong
`docs/09-lo-trinh.md` lên đầu, bất kể độ lớn — kể cả khi số đo bằng 0; (2) rồi tới
biên độ ước tính chia cho chi phí; (3) cuối bảng, tách riêng, mục **đã loại trừ**.

Cột "Chi phí" chỉ nhận bốn giá trị: `<5 phút` · `~1 giờ máy` · `~1 buổi người` ·
`dựng lại splits`.

Mục **đã loại trừ** là bắt buộc. Bác bỏ một giả thuyết bằng số đo là kết quả, không
phải chỗ trống — đúng tinh thần "ghi cả thất bại" của `CLAUDE.md`.

## Tránh

- **Dựng lại splits.** Luật 2. Không chạy `python -m vietjobs.dataset build`.
- **Chạm `test.parquet`** ngoài đúng ngoại lệ A8.
- **Đo trên toàn bộ dữ liệu rồi kết luận cho `train`.** Luôn nói rõ đo trên tập nào.
- **Báo một tỷ lệ mà không kèm mẫu số.** "88 %" của 8.659 khác hẳn "88 %" của 12.
- **Đề xuất xoá dòng mà không kèm hai con số:** bao nhiêu dòng bị xoá, và bao nhiêu
  phần trăm lớp nhỏ nhất bị xoá theo.
- **Kết luận từ một phép đo.** Luật 3.
- **Tự ý ghi vào `docs/`.** In markdown ra stdout, đúng lệ của
  `scripts/measure_vitext.py` và `scripts/report_sweep.py`. Thấy nội dung đáng đưa
  vào note thì hỏi một dòng ở cuối, và khi làm thì theo Quy tắc số 1 trong
  `CLAUDE.md`.
