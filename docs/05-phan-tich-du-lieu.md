[← Overview](00-tong-quan.md) · [← Vietnamese processing](02-vietnamese-nlp.md) · [Deep-learning baseline →](06-baseline-dl.md)

# Data analysis before any model is built

Every number below is measured on the **`train` split** (33,396 rows) and can be
regenerated with:

```bash
python scripts/analyze_data.py
```

The full output is in [`artifacts/eda/summary.json`](../artifacts/eda/summary.json).
`val` is used only to check whether a number is stable; `test` is not touched
([03-protocol.md](03-protocol.md), item 2).

---

## 1. Five questions, five figures

| Question | Figure | One-sentence answer |
|---|---|---|
| How does salary spread within each sector? | [eda-04](figures/eda/eda-04-luong-theo-nganh.png) | The medians of all 16 sectors span only **11.0 → 16.0**, while each sector spans from 1 to several hundred |
| How skewed are the classes? | [eda-02](figures/eda/eda-02-lech-lop.png) | **27.1:1** between the largest and the smallest class |
| How is salary distributed? | [eda-05](figures/eda/eda-05-phan-bo-luong.png) | Skew **11.84** on the raw scale, **0.12** after `log1p` |
| Where do the salary labels exist? | [eda-06](figures/eda/eda-06-cong-bo-luong.png) | **71.8 %** of postings disclose a salary; the lowest sector is 56.8 % |
| How do the 16 sectors spread out by size? | [eda-03](figures/eda/eda-03-tan-xa-co-lop.png) | A continuous band, Zipf slope **−1.19**; the smaller the class, the less often salary is disclosed (r = **0.670**) |

![Salary spread by sector](figures/eda/eda-04-luong-theo-nganh.png)

---

## 2. Class skew — 27.1:1

| Quantity | Value |
|---|---|
| Number of classes | 16 |
| Largest class | `kinh_doanh_bán_hàng_chăm_sóc_khách_hàng` (sales & customer care) — 5,330 postings (**16.0 %**) |
| Smallest class | `nông_nghiệp_năng_lượng_môi_trường` (agriculture, energy, environment) — 197 postings (0.6 %) |
| Skew ratio | **27.1 : 1** |
| The catch-all class `nhóm_nghề_khác` | 250 postings (0.7 %) |

![Class-size scatter](figures/eda/eda-03-tan-xa-co-lop.png)

One point per sector, x-axis on a log scale. It shows three things the bar chart
in [eda-02](figures/eda/eda-02-lech-lop.png) does not:

| Quantity | Value | How to read it |
|---|---|---|
| An even split over 16 classes | 2,087 postings/class | Only 8 classes reach that level; the other 8 sit below |
| Median class size | 1,690 postings | Below the mean of 2,087 — a right-skewed band |
| The four largest classes | **55.2 %** of the data | Four sectors hold more than half the corpus |
| The four smallest classes | **4.2 %** of the data | All four together are still under a third of the largest class |
| Zipf slope (log n against log rank) | **−1.19** | A long continuous tail, not a handful of isolated rare classes |

The slope of −1.19 is the most important number in the table: it says the class
skew here is **a smooth band**, not two clusters of "normal classes" and "rare
classes". So there is no natural threshold at which to carve off a few rare
classes and fold them into `nhóm_nghề_khác` — any cut point is arbitrary, and
every folded class loses its true label.

The direct consequence for scoring: always predicting the largest class gives
**accuracy 0.2014** on `val` but **macro-F1 0.0210**. That is why the headline
metric is macro-F1 — see
[nen-tang/06-do-luong-va-baseline.md](nen-tang/06-do-luong-va-baseline.md).

The consequence for training: a bare cross-entropy loss will favour the four large
classes (55 % of the data). The `--class-weight` flag in
[`dl/train_dl.py`](../src/vietjobs/dl/train_dl.py) exists to **measure** whether
class balancing helps, not to be on by default.

## 3. Salary distribution — a long tail, and that is why `log1p`

Counting only the 23,965 postings in `train` that disclose a salary, in million
VND/month.

| Quantile | p01 | p05 | p25 | **median** | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| Value | 2.5 | 6.5 | 10.0 | **13.0** | 17.5 | 30.0 | 50.0 | **350.0** |

| Shape | Raw | After `log1p` |
|---|---|---|
| Skew | **11.84** | **0.12** |
| Kurtosis | 255.2 | — |

The mean of 15.5 sits **above the median of 13.0** — the classic signature of a
right tail. Training a regression on the raw scale means letting 13 postings above
200 million drag the entire gradient; training on `log1p` brings the skew to 0.12,
nearly symmetric. So the regression head learns `log1p(salary_mid)` and every
report converts back to million VND
([`evaluate.regression_metrics`](../src/vietjobs/evaluate.py)).

## 4. The edges — what is a real tail and what is junk

| Threshold | Postings | How to read it |
|---|---|---|
| Above the IQR fence (> 28.75) | **1,491** (6.2 %) | Mostly genuine management salaries — **not trimmed** |
| Below the IQR fence | **0** | The distribution is bounded below, there is no left tail |
| < 2 million/month | **103** | Almost certainly hourly or per-shift pay entered in the wrong field |
| > 200 million/month | **13** | Almost certainly annual pay, or the wrong unit |
| Already flagged `salary_extreme` by `dataset.clean` | 76 | The current filter **does not cover all 116 suspicious postings above** |

This is an open work item: the 116 postings at the two edges have not been dealt
with. They are few (0.5 % of the labels) but sit exactly where they do the most
damage to MAE.

One more number for context: **93.0 %** of postings that disclose a salary give a
*range*, not a single figure. `salary_mid` is the midpoint of that range — meaning
the regression label is itself an approximation, and an error under ~1 million
should not be read as signal.

## 5. Salary labels exist for only 71.8 % of the data

| Split | Disclosure rate |
|---|---|
| train | **71.8 %** (23,965 / 33,396) |
| val (dev) | 71.2 % (5,095 / 7,159) |

By sector the rate spans **56.8 %** (`nhóm_nghề_khác`) to **76.4 %**
(`du_lịch…`, `giáo_dục…` — tourism, education), and it **does not spread at
random**: the right panel of
[eda-03](figures/eda/eda-03-tan-xa-co-lop.png) shows that the smaller the class,
the less often a salary is disclosed.

| What class size drags along | Coefficient | p |
|---|---|---|
| Disclosure rate — Pearson(log n) | **0.670** | 0.0045 |
| Median salary — Spearman | −0.234 | 0.383 |

Class size is tightly bound to **whether a salary label exists**, but says nothing
about **how much** the salary is. The five smallest sectors therefore take a double
penalty: fewer postings to learn the class from, and fewer labels still for the
regression head. Three consequences:

1. The regression head has labels for only 7 postings in 10. In the multi-task
   merge, `loss_B` **must be masked** — the remaining 28.2 % contribute 0 to the
   loss, not a contribution of zero.
2. The `disclosed` task (predict whether a posting states a salary) has a majority
   baseline of **0.7117** accuracy. Any model that does not beat that is useless.
3. Disclosure is **sector-dependent**, so ignoring the other 28.2 % is not a random
   omission: the regression model learns on an already-selected sample.

## 6. The most important finding: the sector says almost nothing about the salary

Variance decomposition of `log1p(salary_mid)` across the 16 sectors:

| Quantity | Value |
|---|---|
| eta² (between-sector variance / total) | **0.032** — the sector explains **3.2 %** |
| Lowest median | `nhóm_nghề_khác` — 11.0 |
| Highest median | `xây_dựng_kiến_trúc_bất_động_sản` (construction, architecture, real estate) — 16.0 |

A different measurement reaches the same conclusion, scored on `val` (5,095
postings with a salary):

| Prediction rule | MAE (million) | R² |
|---|---|---|
| The `train` median for every posting (13.0) | **5.86** | −0.063 |
| **Knowing the true sector**, predict its median | **5.75** | −0.032 |

Knowing 100 % of the sector labels reduces MAE by only **1.8 %**. Three
conclusions, and all three shape the rest of the project:

1. **The bar for the regression head is MAE 5.86 million**, not 0. A model that
   produces 5.8 million has learned nothing.
2. The salary signal lives in the **details of the posting** — seniority, years of
   experience, languages, location — not in the sector label. That is exactly what
   PhoBERT has a chance to read and a sector-level TF-IDF does not.
3. Expectations for **multi-task must come down**. The theory that "the two heads
   help each other" rests on the assumption that the two labels share signal; here
   the measurable shared part is 3.2 %. So the roadmap builds **two separate
   networks first**, gets a number for each, and only then merges them — if the
   merged version is not better, we already know why.

---

## 7. Work items that came out of this analysis

| Item | Why | Status |
|---|---|---|
| Deal with the 116 edge-case salaries | 103 postings < 2 million + 13 postings > 200 million, almost certainly the wrong unit | not started |
| Measure `--class-weight` on the classification head | the 27.1:1 skew | waiting on the baseline |
| Mask `loss_B` in the multi-task merge | 28.2 % of postings have no salary label | not yet due |
| Measure the selection bias of the salary head | the disclosure rate varies by sector (56.8 % → 76.4 %) and **with class size** (r = 0.670) | not started |

---

[← Overview](00-tong-quan.md) · [Deep-learning baseline →](06-baseline-dl.md)
