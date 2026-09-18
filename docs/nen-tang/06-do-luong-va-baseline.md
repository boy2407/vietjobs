[← Data leakage](05-ro-ri-du-lieu.md) · [Background](00-index.md) · [Regularisation →](07-chinh-quy-hoa.md)

# 6. Metrics and baselines

Pick the wrong yardstick and everything afterwards is meaningless — you will
optimise diligently in the wrong direction. This note explains why the project
chose macro-F1, and why every model has to clear a floor before it counts as
"having learned anything".

---

## 1. Accuracy lies when the classes are skewed

The occupation task has 16 classes, and they are **very** skewed: the largest is
**27×** the smallest. The three smallest hold only 196–258 rows.

Consider the dumbest possible model — always predict the largest class, never look
at the data:

| Metric | Score |
|---|---|
| **accuracy** | **0.2014** |
| macro-F1 | **0.0210** |
| balanced accuracy | 0.0625 |

Accuracy says "right 20 % of the time". That does not sound terrible. But this model
**ignores 15 of the 16 classes entirely** — it has learned nothing. Macro-F1 tells
the truth: 0.0210.

**Why they differ so much:**

- **Accuracy** = the fraction of correct predictions over **all** samples. Large
  classes contribute many samples, so they dominate the number. Getting the large
  class right is enough for a presentable accuracy.
- **Macro-F1** = compute F1 **per class**, then take the **unweighted** mean. A
  class with 196 rows has **the same** voice as one with 5,000. Abandoning one class
  costs 1/16 of the total, regardless of its size.

> **Rule:** the more skewed the classes, the more accuracy lies. At 27:1 it lies a
> great deal.

---

## 2. Precision, recall, F1 — a quick refresher

For one specific class, say "accounting":

|  | Model says "accounting" | Model says another class |
|---|---|---|
| **Really is accounting** | TP (correct) | FN (missed) |
| **Is not accounting** | FP (false alarm) | TN (correct) |

```
precision = TP / (TP + FP)    Of the postings it CALLS accounting, how many are?
recall    = TP / (TP + FN)    Of the postings that REALLY are accounting, how many did it catch?
F1        = the harmonic mean of the two
```

**Why the harmonic mean and not the arithmetic one.** The harmonic mean punishes
imbalance heavily:

| precision | recall | arithmetic mean | **F1** |
|---|---|---|---|
| 1.00 | 0.10 | 0.55 | **0.18** |
| 0.55 | 0.55 | 0.55 | **0.55** |

A model that only dares say "accounting" when it is dead certain gets precision 1.00
while missing 90 % — the arithmetic mean gives it 0.55, F1 gives it 0.18. F1 is
right.

---

## 3. The four metrics the project uses, and what each answers

| Metric | Question it answers | Why it is there |
|---|---|---|
| **macro-F1** | Does the model do **evenly** well across all 16 classes? | The **headline** metric for model selection |
| `f1_macro_no_junk` | What happens when the junk class `nhóm_nghề_khác` is dropped? | Separates **label noise** from **model error** |
| `balanced_accuracy` | Mean recall across classes | Sensitive to abandoned classes |
| `accuracy` | Raw correctness rate | Only for **comparing against the floor**, never for model selection |
| `top3_accuracy` | Is the right answer among the 3 suggestions? | What a real user actually needs |

**`f1_macro_no_junk` is worth noting.** The class `nhóm_nghề_khác` is a junk drawer:
it holds "Nhân Viên Seo Web", "Nhân Viên Quản Trị Website" — postings that belong in
marketing and IT. Getting them wrong is **not the model's fault**; the original
label is wrong. Reporting both numbers lets the reader keep the two apart.

**`top3_accuracy` = 93 %** while macro-F1 is only 0.61. That gap is not a
contradiction — it says the model nearly always puts the right label in the top 3
and merely orders the intrinsically ambiguous occupations wrongly. For a system that
suggests options to a user, 93 % is the number to report.

---

## 4. Baselines — the floor to clear

A number on its own says nothing. **Is 0.6112 good or bad?** Unanswerable, unless
you know what it is being compared against.

The project builds three floor levels for classification:

| Level | What it is | macro-F1 |
|---|---|---|
| **Floor 1** — random | Always predict the largest class | 0.0210 |
| **Floor 2** — no ML | Keyword matching in the title | **0.4321** |
| The cheapest model | SVM reading titles only | 0.5547 |
| The final model | SVM C=0.02, full text | 0.6050 |

**Floor 2 is the real floor.** It asks: *"if we use no machine learning, just a few
if-else lines matching keywords, how far do we get?"* — 0.4321.

Any machine-learning model that cannot beat 0.4321 **does not deserve to exist**: it
costs training time, is hard to explain, hard to maintain, and loses to a far
simpler piece of code.

In this project, **two full-text KNN models lose to floor 2** (0.4059). That is a
useful result, and it is in the log — see [note 4](04-ma-tran-thua-va-so-chieu.md).

For the salary tasks, the corresponding floors are:

| Task | Floor to clear |
|---|---|
| `disclosed` (does it state a salary) | accuracy **0.7176** · macro-F1 **0.4179** — "always say yes" |
| `salary` (the amount) | MAE **5.54 million** — the median per occupation group |

---

## 5. Noise — when a difference is real

Run two configurations, one gives 0.6050 and the other 0.6033. The first is better,
surely?

**Not necessarily.** The score is measured on a finite dev set (7,159 postings), so
it has random variation of its own. That variation is measured with a **1,000-sample
bootstrap** (`evaluate.bootstrap_scores`, run over all 34 runs of the comparison
cluster):

```
macro-F1 standard deviation  ≈  0.0077      (roughly 0.0067 – 0.0087)
```

Meaning: **a difference smaller than 0.0077 is indistinguishable from noise.**

Applied to the ablation table:

| Comparison | Difference | Verdict |
|---|---|---|
| `province` 0.6050 vs `raw` 0.6033 | +0.0017 | **Within noise.** "Probably helps", P(>0) = 0.97 |
| Stopword removal: 0.5756 vs 0.5754 | +0.0002 | **Within noise.** Indistinguishable |
| `C` 0.02 vs `C` 0.5: 0.6050 vs 0.5763 | +0.0287 | **Real.** Nearly 4× σ |

Without this step it is very easy to spend a week optimising differences of 0.001
and believe you are making progress.

### This σ used to be a number passed by word of mouth

Until the model-comparison cluster, `docs/` cited **σ ≈ 0.009** in five places while
**no `.py` file in the repository computed it**. It was probably computed once in a
REPL and copied into the documentation. The current value (0.0077) is the first time
it came from re-runnable code — and it is about 15 % lower than the old figure,
meaning the old rule was **too conservative**.

The lesson is broader than the number: **a decision threshold nobody can re-derive is
not evidence, however scientific it sounds.** It had been propping up three
"drop this step" conclusions across the whole Vietnamese ablation table.

### An independent σ is not how to compare two models

σ answers: *"how much does this score move if the dev set changes?"*
The question that actually matters is: *"does A beat B on **the same** rows?"*

Two different questions, and the second is far more sensitive. The reason: two models
are wrong on largely the same ambiguous postings, so if you resample **the same** set
of rows for both and subtract, the shared variation cancels out. That is the **paired
bootstrap**, and it is the basis for decisions in
[10-so-sanh-mo-hinh.md §4](../archive/10-so-sanh-mo-hinh.md#4-ma-trận-phàng--cột).

A real example from that cluster: `logreg` 0.6072 versus `svm` 0.6050 — a gap of
+0.0022, smaller than σ, so the old rule called it noise. The paired bootstrap gives
`P(logreg > svm) = 0.685`, i.e. **still indistinguishable**, but now we know that
from a correct measurement rather than from a crude threshold.

> **Principle:** before celebrating an improvement, ask *"is it larger than the
> noise?"* If you do not know how large the noise is, you may not conclude anything.

---

## Back to the project itself

- [03-protocol.md §4](../03-protocol.md) — the project's official metric definitions
- [06-mo-hinh-phan-lop.md](../archive/06-mo-hinh-phan-lop.md) — the results ladder and how to read it
- [04-results.md](../archive/04-results-ml.md) — the experiment log
