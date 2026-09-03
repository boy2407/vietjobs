---
name: model-diagnosis
description: "Soi một run đã huấn luyện của VietJobs theo thang tám bậc — per-class F1 so với support, ma trận nhầm lẫn, đọc lỗi thật, chênh train↔val, đường học, trọng số theo khối đặc trưng, độ nhạy siêu tham số, hiệu chuẩn — rồi chọn đúng một trong bốn đòn bẩy: đổi họ mô hình, chỉnh siêu tham số, sửa đặc trưng, làm sạch nhãn. Use this skill when the user asks why the model is wrong, where it fails, whether to tune or switch models, or what to do next about a run — or invokes /model-diagnosis."
trigger: "Use this skill when the user asks why a model underperforms, which classes it confuses, whether it overfits or underfits, whether more data or tuning would help, how to read a confusion matrix or per-class report from an existing run, or invokes /model-diagnosis."
version: 1
---

# Model diagnosis — sai ở đâu, vì sao, và kéo đòn bẩy nào

Điểm số thấp không nói được nên làm gì. Skill này đi từ một `run_id` tới **đúng một
đòn bẩy** trong bốn: đổi họ mô hình · chỉnh siêu tham số · sửa đặc trưng · làm sạch
nhãn. Ba đòn bẩy còn lại phải bị **loại trừ bằng số đo**, không bằng cảm giác.

Khác với [`dataset-diagnosis`](../dataset-diagnosis/SKILL.md): skill đó soi dữ liệu,
chạy được khi chưa có mô hình nào. Skill này cần một run đã lưu trong `artifacts/`.
Bậc B2 và B3 phải đọc chéo sang khối A6 và A3 của skill kia.

---

## Luật số 1 — số đã có sẵn trên đĩa, đọc trước khi chạy lại

`artifacts/<run_id>/metrics.json` đã chứa `per_class`, `confusion_matrix` 16×16,
`top_confusions`, `f1_macro_boot_std` và `f1_macro_boot_ci95` cho **mọi** run. Huấn
luyện lại để lấy thứ đang nằm trên đĩa vừa tốn thời gian vừa làm lệch so sánh.

Cẩn thận: run cũ hơn cụm quét (`cat-T-*`, `cat-P*`, `cat-TREE-*`, `cat-A-*`,
`cat-FINAL-*`) **không có** khoá `model_label` / `overrides` / `estimator_params`.
Dùng `.get()`, đừng `[...]`.

## Luật số 2 — ngưỡng nhiễu, đọc từ chính run đó

σ bootstrap nằm ở `metrics.f1_macro_boot_std` (1.000 lần lấy lại mẫu). Trên bài phân
lớp nghề nó vào cỡ **±0,007**, nên **chênh lệch dưới ~0,014 (2σ) không phải phát
hiện** — không viết vào note, không dựa vào đó để quyết định. Luôn đọc số thật của
run đang soi thay vì nhớ con số này.

Quyết định giữa hai run thì dùng **paired bootstrap**, không dùng σ rời:
`evaluate.bootstrap_indices(n, n_boot=1000, seed=C.RANDOM_SEED)` một ma trận dùng
chung, rồi `evaluate.paired_delta(a, b)["p_gt_0"]`. Ngưỡng theo
`docs/03-protocol.md`: trên 0,975 hoặc dưới 0,025 là khác biệt rõ; quanh 0,5 là hoà.

## Luật số 3 — không nhảy cóc, và không chạm test

Leo thang B0 → B8 theo thứ tự. Ba bậc B4 + B5 + B6 tốn tổng cộng khoảng mười phút
máy; một lần fine-tune tốn một hai giờ. **Đổi họ mô hình luôn là đòn bẩy cuối.**

`artifacts/cat-FINAL-svm-C0.02-test` là lần chạm `test` duy nhất đã tiêu. Không chạy
`--eval test`, không đọc `test.parquet` để chẩn đoán, không "kiểm chứng lại" một kết
luận trên test.

---

## Thang chẩn đoán

Phần mở đầu chung cho mọi khối:

```python
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score
RUN = 'cat-SW-svm-1'
m = json.load(open(f'artifacts/{RUN}/metrics.json'))
va = pd.read_parquet('data/processed/splits/val.parquet')
```

### B0 · Đọc `metrics.json` — mười giây, không được bỏ

```python
print({k: m.get(k) for k in ('run_id','model_label','scope','prep','eval_split',
                             'n_train','n_eval','fit_seconds')})
mm = m['metrics']
print({k: v for k, v in mm.items()
       if k not in ('per_class','confusion_matrix','labels','top_confusions')})
```

Ghi lại ngay `f1_macro`, `f1_macro_no_junk`, `balanced_accuracy` và
`f1_macro_boot_std`. Khoảng cách giữa `f1_macro` và `f1_macro_no_junk` là phần lớp
rác kéo xuống — nếu nó lớn hơn 2σ thì đó đã là một phát hiện, thuộc đòn bẩy **nhãn**.

### B1 · F1 theo lớp so với support — lệch lớp hay khó thật

```python
rows = [(k, v["f1-score"], v["support"]) for k, v in mm["per_class"].items()
        if k not in ("accuracy", "macro avg", "weighted avg")]
df = pd.DataFrame(rows, columns=["class","f1","support"]).sort_values("f1")
print(df.to_string(index=False), df.f1.corr(df.support, method="spearman"))
```

| ρ (spearman) | Kết luận | Đòn bẩy |
|---|---|---|
| trên 0,7 | Mất cân bằng đang chi phối | siêu tham số: `class_weight`, lấy mẫu lại |
| 0,3 – 0,7 | Hỗn hợp | đọc B2 trước khi làm gì |
| dưới 0,3 | **Không phải mất cân bằng — là độ khó của lớp.** `class_weight="balanced"` đã làm xong việc của nó | **đặc trưng** hoặc **nhãn** |

### B2 · Ma trận nhầm lẫn — cặp nào dính, và chúng có thật sự khác nhau không

```python
for c in mm["top_confusions"]:
    print(c["n"], c["true"], "->", c["predicted"])
```

Bước bắt buộc kế tiếp: đối chiếu với khối **A6** của `dataset-diagnosis` (cosine giữa
tâm lớp). Hai phép đo độc lập chỉ cùng một cặp lớp thì phát hiện mới chắc.

| Dấu hiệu | Đọc ra | Đòn bẩy |
|---|---|---|
| Nhầm lẫn **đối xứng** (a→b ≈ b→a) và cosine trên 0,80 | Hai lớp không phân biệt được bằng văn bản | **nhãn** — gộp lớp hoặc ghi trần. Tuyệt đối không phải mô hình |
| Nhầm lẫn **một chiều** lớn | Mô hình đang thiên về lớp đích | **siêu tham số** — trọng số lớp |
| Cặp lớn nhưng cosine dưới 0,60 | Lớp tách được mà mô hình không tách | **đặc trưng** |

### B3 · Đọc N tin sai thật — trần nhiễu nhãn, Ưu tiên 2

`row_index` là **vị trí 0-based trong file split**, không phải nhãn chỉ mục. Nối bằng
`.iloc`, rồi khẳng định lại — nối sai vẫn chạy được và cho ra một bảng lỗi hoàn toàn
bịa.

```python
d = np.load(f'artifacts/{RUN}/predictions.npz', allow_pickle=True)
w = va.iloc[d["row_index"]].copy(); w["y_true"], w["y_pred"] = d["y_true"], d["y_pred"]
assert (w["category"].values == w["y_true"]).all(), "nối sai — dừng lại"
bad = w[w.y_true != w.y_pred]
for r in bad.sample(150, random_state=42).itertuples():
    print(f"\n{r.job_title}\n  thật: {r.y_true}\n  đoán: {r.y_pred}\n  {r.description[:300]}")
```

Gán tay mỗi tin vào đúng **một** trong ba ô — `mô hình sai` · `nhãn gốc sai` ·
`nhập nhằng thật (đa nhãn)` — rồi tính lại trần:
`trần ≈ f1_macro / (1 − tỷ_lệ_nhãn_sai)`.

Trước khi bỏ ra hai giờ người: **chạy khối A3 của `dataset-diagnosis`**. Nó cho trần
trong ba giây. Trần đã thấp thì việc gán tay chỉ để phân giải mịn hơn, không phải để
phát hiện.

### B4 · Chênh train ↔ val — thiên lệch hay phương sai

`train.py --eval` chỉ nhận `val` và `test`, và **đừng** thêm lựa chọn `train` — nó sẽ
ghi một dòng vô nghĩa vào log append-only. Cách đúng là nạp mô hình đã lưu và chấm lại:

```python
p = joblib.load(f'artifacts/{RUN}/model.joblib')
tr = pd.read_parquet('data/processed/splits/train.parquet')
print(f1_score(tr["category"], p.predict(tr), average="macro", zero_division=0))
```

| Chênh train − val | Kết luận | Đòn bẩy |
|---|---|---|
| dưới 2σ | Không phải phát hiện | — |
| 2σ – 0,10 | **Thiên lệch chi phối.** Mô hình chưa khớp nổi cả tập train; chính quy hoá thêm chỉ làm tệ hơn | **đặc trưng** hoặc **họ mô hình** |
| trên 0,15 | Phương sai chi phối | **siêu tham số** — giảm `C`, tăng `min_df` |
| train ≈ 1,0 | Học thuộc | siêu tham số, gấp |

Khối này bác bỏ được một suy diễn sai rất hay gặp: `C` tối ưu rất nhỏ **không** chứng
minh quá khớp. Phải đo chênh mới biết.

### B5 · Đường học theo tỷ lệ tập train — thêm dữ liệu có ăn không

Lấy mẫu **theo `group_id`**, không theo dòng. Lấy theo dòng sẽ để tin đăng lại nằm cả
trong lẫn ngoài mẫu con, và làm đường học phẳng một cách giả tạo.

```python
from sklearn.pipeline import Pipeline
from vietjobs import features as F
from vietjobs.models import build_estimator
prep = F.PrepConfig(province=True)                     # đúng cấu hình của run đang soi
gids = tr["group_id"].unique().copy(); np.random.default_rng(42).shuffle(gids)
for frac in (0.1, 0.25, 0.5, 1.0):                     # xáo MỘT lần -> các mẫu con lồng nhau
    sub = tr[tr["group_id"].isin(set(gids[:int(len(gids)*frac)]))]
    est, _ = build_estimator("category", "svm", 42); est.set_params(C=0.02)
    pipe = Pipeline([("features", F.build_features("category", "full", prep, None)),
                     ("estimator", est)]).fit(sub, sub["category"])
    print(frac, len(sub), f1_score(va["category"], pipe.predict(va),
                                   average="macro", zero_division=0))
```

Bốn điểm mất khoảng ba tới bốn phút, không ghi artifact, không đụng log. Đọc chênh
giữa `frac=0,5` và `frac=1,0`: dưới 2σ → đường đã bão hoà, **thêm dữ liệu không ăn**,
chuyển sang đặc trưng hoặc nhãn. Trên 0,03 → còn dốc, thu thêm dữ liệu là đòn bẩy rẻ
nhất — kèm cảnh báo dựng lại splits sẽ vô hiệu `04-results.md`.

### B6 · Đóng góp theo khối đặc trưng — `Σ|w|`

```python
from vietjobs.features import source_columns
ct, est = p.named_steps["features"], p.named_steps["estimator"]
names = ct.get_feature_names_out()                     # dạng "block__feature"
blocks = np.array([n.split("__", 1)[0] for n in names])
t = pd.DataFrame({"block": blocks, "w": np.abs(est.coef_).sum(axis=0)}) \
      .groupby("block").agg(dims=("w","size"), w=("w","sum"))
t["pct"] = 100 * t.w / t.w.sum()
t["per_dim_x"] = (t.pct / t.dims) / (100 / t.dims.sum())
print(t.sort_values("pct", ascending=False).round(3).to_string(), source_columns(ct))
```

`per_dim_x` là sức nặng trung bình của một chiều trong khối, so với mức trung bình
toàn cục. Đối chiếu bảng này với `docs/05-dac-trung-tfidf.md` §9.

| Điều kiện | Hành động |
|---|---|
| `per_dim_x` của một khối cao gấp hơn 2 lần khối lớn nhất | Khối đó bị **pha loãng** — nhân trọng số nó lên (Ưu tiên 3a: tiêu đề ×2, ×3) |
| Một khối chiếm quá 25 % số chiều mà `per_dim_x` dưới 0,8 | Cắt chiều khối đó bằng `min_df` / `max_features` — Ưu tiên 3b |
| Khối dưới 0,1 % trọng số | Gỡ khối, đo lại. Đối chiếu A4: cột nguồn có rỗng phần lớn không |

Chỉ chạy được với mô hình tuyến tính (`svm`, `logreg`). Với cây thì thay
`np.abs(est.coef_).sum(axis=0)` bằng `est.feature_importances_`, giữ nguyên phần còn lại.

### B7 · Độ nhạy siêu tham số — đọc từ cụm quét đã có, không quét lại

```python
import glob
rows = [(json.load(open(f))["model"], json.load(open(f))["metrics"]["f1_macro"])
        for f in sorted(glob.glob('artifacts/cat-SW-*/metrics.json'))]
df = pd.DataFrame(rows, columns=["model","f1"])
print(df.groupby("model").f1.agg(["min","median","max"])
        .assign(bien_do=lambda d: d["max"] - d["min"]).round(4).to_string())
```

Quy tắc: **biên độ trong sáu cấu hình dưới 2σ → siêu tham số đã cạn cho mô hình đó.**
Quét thêm là tiêu giờ máy mà không đổi kết luận. Muốn dựng lại toàn bộ bảy bảng so
sánh thì chạy `PYTHONPATH=src python scripts/report_sweep.py` — nó tự kiểm
`check_alignment` và có sẵn sáu mỏ neo tái lập.

### B8 · Hiệu chuẩn và top-3 — khi không có `y_proba`

`LinearSVC` không có `predict_proba`, nên `predictions.npz` của mọi run `svm` chỉ có
bốn khoá. Kiểm trước, đừng giả định: `np.load(...).files`.

Thay thế cho `svm` là dùng biên `decision_function`:

```python
M = p.decision_function(va); labs = np.array(p.named_steps["estimator"].classes_)
top3 = labs[np.argsort(-M, axis=1)[:, :3]]
print(np.mean([t in r for t, r in zip(va["category"], top3)]))
S = np.sort(M, axis=1); gap = S[:, -1] - S[:, -2]
ok = (va["category"].values == labs[M.argmax(1)])
for q in (0.10, 0.25, 0.50):
    th = np.quantile(gap, q); print(q, ok[gap <= th].mean(), ok[gap > th].mean())
```

Biên là **thứ tự đúng nhưng không phải xác suất**. Nó dùng được cho ngưỡng từ chối
trả lời (nếu accuracy dưới ngưỡng thấp hơn hẳn phần còn lại), **không** dùng được để
nói "mô hình tự tin 80 %". Muốn xác suất thật thì chạy đúng một lần
`python -m vietjobs.train --task category --model svm_calibrated --scope full --province --C 0.02`
— gấp ba chi phí fit, đổi lại có `y_proba` và `top3_accuracy` (Ưu tiên 5.1).

---

## Bảng quyết định — triệu chứng → đòn bẩy → bước tiếp theo

| Triệu chứng (bậc) | Đòn bẩy | Việc cụ thể |
|---|---|---|
| B1 ρ dưới 0,3 | **loại trừ siêu tham số** | Dừng ý định thêm `class_weight` / lấy mẫu lại. Ghi là đã loại trừ |
| B2 cặp đối xứng + A6 cosine cao | **nhãn** | Gộp lớp thành một trục phụ, hoặc ghi trần. Không chạy mô hình mới |
| B3 quá 30 % lỗi là "nhãn gốc sai" | **nhãn** | Ghi trần đã hiệu chỉnh vào `docs/06-mo-hinh-phan-lop.md`; ngừng so sánh mô hình quanh mức hiện tại |
| B3 quá 40 % lỗi là "nhập nhằng thật" | **định nghĩa bài toán** | Đổi metric sang top-3 hoặc đa nhãn — trục mới, không thay trục cũ |
| B4 chênh trong 2σ–0,10 | **đặc trưng** | Ưu tiên 3a: thêm `transformer_weights` vào `build_features`, chạy lại cùng cấu hình |
| B4 chênh trên 0,15 | **siêu tham số** | `--C` nhỏ hơn, hoặc `min_df` lớn hơn |
| B5 chênh frac 0,5→1,0 dưới 2σ | **loại trừ dữ liệu** | Không thu thêm. Chuyển sang đặc trưng hoặc nhãn |
| B6 một khối `per_dim_x` cao gấp bội | **đặc trưng** | Nhân trọng số khối ×2 rồi ×3, hai run, so paired bootstrap với run gốc |
| B6 một khối nhiều chiều mà nhẹ | **đặc trưng** | Ưu tiên 3b: sửa `_word_tfidf` (`min_df`, `max_features`) — `--set` không tới được TF-IDF |
| B7 biên độ sáu cấu hình dưới 2σ | **đã cạn** | Không quét siêu tham số nữa |
| B7 `p_gt_0` quanh 0,5 giữa hai mô hình | **không đổi họ mô hình** | Hoà → chọn theo chi phí fit |
| B8 top-3 cao hơn hẳn top-1 | **hệ thống, không phải mô hình** | Sản phẩm hiển thị ba gợi ý. Đây là kết quả, không phải thất bại |
| B4 phẳng **và** B5 phẳng **và** B6 cân | **họ mô hình** | Chỉ khi đó mới tới Ưu tiên 4. Bắt buộc đo kèm độ trễ suy luận |

## Khuôn kết quả

```markdown
## Soi mô hình — <run_id> · <model_label> · <scope>/<prep> · eval=<split>

**Chẩn đoán một câu:** <thiên lệch / phương sai / nhãn / đặc trưng> chi phối, vì <số ở bậc Bx>.

| Bậc | Đo được | Ngưỡng | Đọc ra |
|---|---|---|---|

**Đòn bẩy chọn:** <1 trong 4> · **Bước tiếp theo:** `...` · **Kỳ vọng:** <ghi rõ chưa đo>
**Đã loại trừ:** <giả thuyết + số bác bỏ nó>
```

Khối **Đã loại trừ** là bắt buộc. Loại được một đòn bẩy bằng số đo có giá trị ngang
việc chọn được một đòn bẩy.

## Tránh

- **Kết luận từ chênh lệch dưới 2σ.** Luật 2. Đọc σ của chính run đó.
- **So hai run mà một trong hai thiếu `predictions.npz`.** Phần lớn run cũ không có
  file này — chỉ đọc được số đơn lẻ, không so cặp được. Nói thẳng thay vì so bừa.
- **Trừ hai vector chưa kiểm căn hàng.** Chép `report_sweep.check_alignment`:
  `row_index` và `y_true` phải khớp hoàn toàn. Không khớp thì dừng, đừng "sửa cho khớp".
- **So hai run khác `eval_split` / `scope` / `prep`.** Đó là hai thí nghiệm khác nhau.
- **Chạm test.** Luật 3.
- **Huấn luyện lại để lấy số đã có trên đĩa.** Luật 1.
- **Suy diễn từ siêu tham số ra chẩn đoán.** `C` nhỏ không đủ để kết luận quá khớp —
  B4 mới kết luận được.
- **Ghi vào `04-results.md` từ skill này.** Chẩn đoán chạy qua `python - <<PY` hoặc
  `--no-save`. Chỉ khi đã quyết định chạy một cấu hình mới thì mới gọi
  `python -m vietjobs.train` — và khi đó `docs/` phải được cập nhật theo Quy tắc số 1
  trong `CLAUDE.md`.
