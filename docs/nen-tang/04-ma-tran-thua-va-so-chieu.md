[← n-grams and word boundaries](03-ngram-va-ranh-gioi-tu.md) · [Background](00-index.md) · [Data leakage →](05-ro-ri-du-lieu.md)

# 4. Sparse matrices and dimensionality

> **A note from the closed track.** The 236,596-dimensional matrix belongs to
> TF-IDF. PhoBERT produces a **768-dimensional dense** vector — nearly 310× fewer
> dimensions and not an empty cell in it. Read this note to see why switching to
> dense vectors is a change of kind, not a matter of taste
> ([note 9](09-vector-ngu-nghia.md)).

This project builds a **236,596-dimensional** matrix over **33,396** training
samples. This note explains what that number means, why it fits in memory, and why
it makes KNN collapse while SVM survives.

---

## 1. What "236,596 dimensions" means

Each job posting becomes a **list of 236,596 numbers**. Each position in the list
corresponds to one specific feature: a word, a word pair, a province, a count
column.

```
posting #1  =  [0, 0, 0.42, 0, 0, 0, 0.31, 0, ... , 0, 0.07, 0]
                ↑            ↑                          ↑
            "kế toán"    "kinh doanh"              experience_months
```

Where the 236,596 dimensions come from — they add up exactly:

| Block | Dimensions |
|---|---|
| description | 103,459 |
| requirements | 65,737 |
| benefits | 29,936 |
| technical skills | 14,163 |
| soft skills | 10,747 |
| title | 6,835 |
| qualifications | 5,626 |
| one-hot (province, contract, experience) | 57 |
| languages | 22 |
| numeric | 14 |
| **total** | **236,596** |

---

## 2. Sparse — only 492 non-zero cells

A job posting uses a few hundred words, not two hundred thousand. So most cells in
the list are **zero**.

On average each row has only **492 non-zero cells**:

```
492 / 236,596  =  0.21 %      →  99.79 % of the matrix is zero
```

A smaller, verifiable example: the title "Nhân viên kinh doanh" activates only
**7 cells** out of the title block's 6,835 dimensions — the table in
[note 2](02-tf-idf-la-gi.md#6-a-worked-example--on-the-real-data).

### Sparse or dense storage — the numbers

| Storage | Required |
|---|---|
| **Dense** — store all 236,596 × 33,396 eight-byte floats | ~**63 GB** |
| **Sparse** — store only (row, column, value) for non-zero cells | ~**130 MB** |

A factor of roughly **480×**. That is why the project forces
`sparse_threshold=1.0` and uses `MaxAbsScaler` instead of `StandardScaler` —
`StandardScaler` subtracts the mean, which turns every zero into a non-zero and
**destroys sparsity** immediately. Details in
[05-dac-trung-tfidf.md §6–7](../archive/05-dac-trung-tfidf.md#6-khối-số--vì-sao-log1p-rồi-maxabsscaler).

---

## 3. The p/n ratio — the most important number to remember

```
p = number of dimensions (features)  = 236,596
n = number of training samples       =  33,396
p / n                                =       7.1
```

**There are 7× more features than examples to learn from.**

Why that is dangerous: with `p > n`, there always exists a combination of weights
that fits the training data **perfectly** — even if the labels are entirely random.
The model does not need to *learn a rule*, it only needs to *memorise*. And what it
memorises does not generalise to new postings.

The p/n table for this project:

| Configuration | p | n | p/n | Note |
|---|---|---|---|---|
| `scope=full` | 236,596 | 33,396 | **7.1** | The real configuration |
| `scope=title` | 6,835 | 33,396 | 0.20 | |
| `scope=structured` | ~60 | 23,965 | 0.0025 | The only fair playing field for plain `LinearRegression` |

This is why unregularised `LinearRegression` is **guaranteed to break** at
`scope=full`, and why the project created `scope=structured` specifically to
measure it fairly — [note 7](07-chinh-quy-hoa.md).

---

## 4. The curse of dimensionality — why KNN collapses

This is the most shocking result in the project's table:

| Model | Features | macro-F1 (dev) |
|---|---|---|
| Keyword baseline, **no ML at all** | title | 0.4321 |
| KNN k=15 | **title only** | 0.5307 |
| KNN k=30 | **full text** | **0.4059** ← below the baseline |
| SVM C=0.02 | full text | 0.6050 |

**Giving KNN more features makes it worse**, so much worse that it loses to a
baseline that uses no machine learning. SVM on exactly the same features is
markedly better.

### Why

KNN relies entirely on **distance**: find the k nearest postings, take the majority
label. In a very high-dimensional space, distance loses its discriminating power:

> As the dimensionality grows, the distance from a point to its **nearest**
> neighbour and to its **farthest** neighbour converge. "Nearest" stops meaning
> "most similar".

Concretely with text: two postings in the same sector may **share no word at all**
apart from stopwords. Under normalised TF-IDF they are nearly orthogonal — all the
distances are the same. KNN has nothing to hold onto.

### Why SVM does not collapse

A linear SVM does not measure distances between points. It looks for a
**hyperplane** — a weighted combination of the dimensions. A noisy dimension only
has to receive a weight near 0 to be neutralised. Adding useless dimensions makes
the problem harder, but does not **break** the way it works.

The condition for that to happen: most dimensions have to be pushed toward zero
weight. That is exactly the job of the `C` parameter — and also why the optimal `C`
falls all the way to 0.02.

---

## 5. Reading `C = 0.02` as a symptom

The `LinearSVC` default for `C` is 1.0. This project swept it and found:

| `C` | macro-F1 (dev) |
|---|---|
| 4.0 | 0.5171 |
| 1.0 (default) | 0.5618 |
| 0.1 | 0.5983 |
| **0.02** | **0.6050** |
| 0.01 | 0.5976 |
| 0.005 | 0.5865 |

Small `C` = strong regularisation = weights pushed toward zero. An optimal `C`
**50× below** the default is not a pretty accident. It is **the model calling for
help because it has too many dimensions**.

The right reading: if it takes that much regularisation to work at all, then most
of the 236,596 dimensions are noise.

---

## 6. Three things to remember

1. **More features is not automatically better.** KNN on full text at 0.4059 versus
   KNN on titles alone at 0.5307 is measured evidence, from this very project.
2. **Different algorithms tolerate dimensionality differently.** Do not conclude
   "this feature set is bad" from a single algorithm — KNN said bad, SVM said good,
   and SVM was right.
3. **An unusually low optimal `C` is a message**, not a number to copy into the
   report and forget.

---

## Back to the project itself

- [06-mo-hinh-phan-lop.md](../archive/06-mo-hinh-phan-lop.md) — the full results ladder
- [05-dac-trung-tfidf.md](../archive/05-dac-trung-tfidf.md) — how the 236,596 dimensions are assembled
- [note 7 — regularisation](07-chinh-quy-hoa.md) — what `C` actually does
