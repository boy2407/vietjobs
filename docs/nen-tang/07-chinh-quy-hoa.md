[← Metrics and baselines](06-do-luong-va-baseline.md) · [Background](00-index.md) · [Reading a comparison table →](08-doc-mot-bang-so-sanh.md)

# 7. Regularisation — `C`, `alpha`, and why a model needs holding back

The note that explains the strangest number in the project: **`C = 0.02`**, fifty
times below the default, and yet the best result.

---

## 1. Overfitting — in numbers, not in words

Two real measurements from this project, same data, same features, one difference:

| Model | R² **train** | R² **dev** | MAE dev |
|---|---|---|---|
| Plain `LinearRegression` | 0.718 | **0.081** | **6.00 M** |
| `Ridge` alpha=1 | 0.657 | **0.507** | **4.25 M** |

Read those two rows carefully — they contain the whole idea of regularisation:

- The first model is **better on data it has already seen** (0.718 > 0.657).
- The first model is **far worse on new data** (0.081 versus 0.507 — **six times**
  worse).

Ridge **deliberately** makes itself worse on train in exchange for being far better
on dev.

One more bitter detail: the plain model's MAE of 6.00 million is worse than the
"median per occupation group" baseline (5.54 million). That is, worse than using no
model at all.

**Overfitting** = the model memorises the small details and the noise of the
training set instead of learning the general rule.

---

## 2. Why overfitting happens here

Back to the p/n ratio from [note 4](04-ma-tran-thua-va-so-chieu.md):

```
p = 236,596 dimensions      n = 33,396 samples      p/n = 7.1
```

There are 7× more "knobs" than examples to learn from. With `p > n`, there always
exists a combination of weights that fits the training data **perfectly** — even if
the labels are entirely random.

The model does not need to *understand* anything. It only needs to *memorise*. And
what it memorises is useless on a new posting.

---

## 3. What regularisation does

Without regularisation, the model optimises one thing only:

```
minimise:   the error on train
```

Regularisation adds a penalty on **the size of the weights**:

```
minimise:   error on train  +  λ × (sum of squared weights)
```

Now the model has to pay for every weight it assigns. It only agrees to pay when the
feature **genuinely** reduces the error by more than the penalty.

The result: hundreds of thousands of noisy dimensions get pushed toward 0, and only
dimensions carrying real signal keep a meaningful weight. **That is exactly how SVM
survives in 236,596 dimensions where KNN does not** — KNN has no mechanism for
ignoring a useless dimension.

---

## 4. `C` and `alpha` — one knob, turned in opposite directions

This is the easiest thing to confuse, so let us be explicit:

| Parameter | Where | Meaning | A **small** value means |
|---|---|---|---|
| `alpha` | `Ridge`, `Lasso` | **Is** λ, the penalty coefficient | a **light** penalty → a freer model |
| `C` | `LinearSVC`, `LogisticRegression` | Is the **inverse** of λ | a **HEAVY** penalty → a tightly held model |

```
alpha ↑   =   stronger regularisation
C     ↓   =   stronger regularisation      ← the other way round!
```

Remember it in one sentence: **`C` is "freedom" (Cost of misclassification),
`alpha` is "restraint".**

---

## 5. The project's `C` curve — read it as a symptom

Sweeping `C` for `LinearSVC`, same `province` configuration, same seed:

| `C` | macro-F1 (dev) | Time | Interpretation |
|---|---|---|---|
| 4.0 | 0.5171 | 313.6 s | Too free — badly overfit |
| 1.0 *(default)* | 0.5618 | 110.6 s | Still too free |
| 0.2 | 0.5873 | 90.9 s | |
| 0.1 | 0.5983 | 37.5 s | |
| 0.05 | 0.6030 | 33.5 s | |
| **0.02** | **0.6050** | **27.8 s** | **The sweet spot** |
| 0.01 | 0.5976 | 34.4 s | Starting to be over-restrained |
| 0.005 | 0.5865 | 44.5 s | Over-restrained — underfitting |

The classic **∩**-shaped curve. Both ends are bad, with a peak in the middle:

- **`C` too large** → overfitting. Memorising the noise in train.
- **`C` too small** → underfitting. Weights pushed to 0 so hard that even real
  signal cannot be learned.

### Three things worth noting besides the peak

**a. The optimal `C` is 50× below the default.** This is not a pretty accident — it
is a **symptom**. It says most of the 236,596 dimensions are noise. The next thing to
do is not another `C` sweep but **cutting dimensions** — sweeping `min_df` and
`max_features`. Recorded in
[09-lo-trinh.md — Priority 3b](../archive/09-lo-trinh-ml.md#ưu-tiên-3--hai-đòn-bẩy-rẻ-cho-mốc-cơ-sở-ml).

**b. Stronger regularisation runs FASTER.** `C=0.02` takes 27.8 s, `C=4.0` takes
313.6 s — **11× faster** and a higher score. A tighter constraint makes the
optimisation converge more easily. There is no trade-off to make here.

**c. One hyper-parameter line beats nine language-processing steps.**

| Work done | macro-F1 contribution |
|---|---|
| All 9 Vietnamese processing steps | **+0.0017** |
| Tuning `C` from 0.5 down to 0.02 | **+0.0287** |

Nearly **17×**. This is the project's most expensive practical lesson: if you have
not swept the hyper-parameters yet, do not spend time writing another preprocessing
step.

---

## 6. `class_weight="balanced"` — a different kind of correction

The final configuration also sets `class_weight="balanced"`. It is not
regularisation, but it shares the spirit of "stop the model from taking the easy
route".

With classes skewed 27:1, the easiest route is to abandon the rare classes —
accuracy still looks fine ([note 6](06-do-luong-va-baseline.md)).
`class_weight="balanced"` scales each class's error weight inversely to its sample
count: getting a posting from a 196-row class wrong is punished far more heavily
than getting one from a 5,000-row class wrong.

This is why `balanced_accuracy` (0.6816) is **higher** than macro-F1 (0.6112) on
test: the model is deliberately spending recall on the rare classes.

---

## 7. Quick decision table

| Symptom | Means | What to do |
|---|---|---|
| High train score, low dev score | Overfitting | Lower `C` / raise `alpha` · raise `min_df` · lower `max_features` |
| Low train score, low dev score too | Underfitting | Raise `C` / lower `alpha` · add features · change the model |
| Both scores close together and both low | The features carry too little signal | Change the representation, do not turn hyper-parameters |
| An unusually low optimal `C` | Too many noisy dimensions | Cut dimensions, stop sweeping `C` |
| Test score **higher** than dev | No sign of overfitting to dev | Normal — in this project 0.6112 > 0.6050 |

---

## Back to the project itself

- [07-bai-toan-luong.md](../07-bai-toan-luong.md#traps-known-in-advance) — the `LinearRegression` vs `Ridge` table
- [06-mo-hinh-phan-lop.md](../archive/06-mo-hinh-phan-lop.md) — the full results ladder
- [note 4 — sparse matrices and dimensionality](04-ma-tran-thua-va-so-chieu.md) — why p/n = 7.1 is a problem
- [04-results.md](../archive/04-results-ml.md) — every `C`-sweep row
