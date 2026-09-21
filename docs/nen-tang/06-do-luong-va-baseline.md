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
| F1 (weighted) | 0.0705 |

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
| `f1_weighted` (reported as **F1**) | Same F1, averaged by how many rows each class has | Shows the large classes' view; always higher than macro-F1 here |
| `accuracy` | Raw correctness rate | Only for **comparing against the floor**, never for model selection |

**`f1_macro_no_junk` is worth noting.** The class `nhóm_nghề_khác` is a junk drawer:
it holds "Nhân Viên Seo Web", "Nhân Viên Quản Trị Website" — postings that belong in
marketing and IT. Getting them wrong is **not the model's fault**; the original
label is wrong. Reporting both numbers lets the reader keep the two apart.

**The gap between F1 and macro-F1 is the imbalance, made visible.** On
`dl-cat-ce-cpu-0920` the two are 0.6413 and 0.6030: weighting by class size flatters
the model, because the classes it handles best are also the biggest. Report macro-F1
as the headline for exactly that reason, and read F1 beside it as "what a user drawn
uniformly from the corpus experiences".

**Two metrics were removed on 2026-09-20**: `balanced_accuracy` and `top3_accuracy`
are no longer computed by `evaluate.py`, so runs from that date carry neither. Older
rows in `04-results.md` still show them and are left as written (the log is
append-only).

---

## 4. Baselines — the floor to clear

A number on its own says nothing. **Is 0.6112 good or bad?** Unanswerable, unless
you know what it is being compared against.

The project builds three floor levels for classification:

| Level | What it is | macro-F1 |
|---|---|---|
| **Floor 1** — no model | Always predict the largest class | 0.0214 |
| **Floor 2** — linear probe | A linear model on the same PhoBERT vectors | 0.5898 |
| The network | Dense head on those vectors (`dl-cat-ce-cpu-0920`) | 0.6030 |

**Floor 2 is the floor that matters.** It asks: *"is the representation weak, or is
the head broken?"* A linear model has a convex solution and no learning rate to get
wrong — if it already reaches 0.5898 on those vectors, the representation is fine
and any failure is in the head.

A network that cannot beat its linear probe **does not deserve its extra capacity**:
it costs training time, is harder to explain and harder to maintain, and loses to a
far simpler piece of code.

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
bootstrap**, implemented in `evaluate.paired_delta`, and it is how comparisons are
decided here.

A gap smaller than σ is not automatically noise, and a gap larger than σ is not
automatically real: σ answers *"how much would this score move on a different dev
set?"*, while the paired bootstrap answers the question actually being asked.

> **Principle:** before celebrating an improvement, ask *"is it larger than the
> noise?"* If you do not know how large the noise is, you may not conclude anything.

---

## Back to the project itself

- [03-protocol.md §4](../03-protocol.md) — the project's official metric definitions
- [06-baseline-dl.md](../06-baseline-dl.md) — the results ladder and how to read it
- [04-results.md](../04-results.md) — the experiment log
