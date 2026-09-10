[← Overview](../00-tong-quan.md)

# Background — for newcomers to ML/DL

Nine notes explaining **concepts**; they contain no experiment results. They do not
have to be updated after every model run — which is exactly why they sit apart
here.

Notes 1–8 were written during the TF-IDF phase, closed on 2026-09-08. They are
**not deleted**: to understand why the main track is now PhoBERT, you have to
understand what it replaced. [Note 9](09-vector-ngu-nghia.md) is the bridge between
the two phases — a newcomer should read note 1, then 9, and come back to 2–8 when
depth is needed.

Every example is taken from the VietJobs corpus itself, not from a textbook.

---

## Read in this order

| # | Note | What you will understand | Pairs with |
|---|---|---|---|
| 1 | [Why clean the data at all](01-vi-sao-phai-lam-sach.md) | The real reason is not "garbage in, garbage out". It is **dimension splitting** and **leakage** | [01-data-audit](../01-data-audit.md) |
| 2 | [What TF-IDF is](02-tf-idf-la-gi.md) | How words become numbers. Why L2 normalisation is required. With the full matrix printed and a worked example | [05-dac-trung-tfidf](../archive/05-dac-trung-tfidf.md) |
| 3 | [n-grams and word boundaries](03-ngram-va-ranh-gioi-tu.md) | Why Vietnamese breaks the "whitespace separates words" assumption | [02-vietnamese-nlp](../02-vietnamese-nlp.md) |
| 4 | [Sparse matrices and dimensionality](04-ma-tran-thua-va-so-chieu.md) | What 236,596 dimensions means. Why KNN collapses and SVM does not | [06-mo-hinh-phan-lop](../archive/06-mo-hinh-phan-lop.md) |
| 5 | [Data leakage](05-ro-ri-du-lieu.md) | Three kinds of leak, and why **none of them raises an error** | [03-protocol](../03-protocol.md) |
| 6 | [Metrics and baselines](06-do-luong-va-baseline.md) | Why accuracy lies when the classes are skewed 27:1 | [04-results](../archive/04-results-ml.md) |
| 7 | [Regularisation](07-chinh-quy-hoa.md) | What `C` is. Why an optimal `C` of 0.02 is a model calling for help | [07-bai-toan-luong](../07-bai-toan-luong.md) |
| 8 | [Reading a model comparison table](08-doc-mot-bang-so-sanh.md) | Hyper-parameters · what a "cell" is · why equal cell counts are still not fair · the winner's curse · σ versus the paired bootstrap | [10-so-sanh-mo-hinh](../archive/10-so-sanh-mo-hinh.md) |
| **9** | [**Semantic vectors**](09-vector-ngu-nghia.md) | Why counting was abandoned. Subwords · context · 768 dense dimensions · frozen versus fine-tuned | [02-vietnamese-nlp](../02-vietnamese-nlp.md) · [06-baseline-dl](../06-baseline-dl.md) |

---

## One sentence summarising notes 1–8 — and the sentence that replaces it

> **Notes 1–8:** machine learning on text is the problem of **counting character
> sequences and then finding weights**. Every preprocessing step has exactly one
> job: make that counting count the thing we meant to count.

> **Note 9:** stop counting — place each posting at **a point in a 768-dimensional
> space**, so that two postings in the same occupation land near each other even
> when they share no words at all.

The second sentence is the current track. The first still has to be understood,
because the four consequences below explain exactly what counting **cannot** do —
and that is the reason the second sentence exists.

Four consequences, one note each above:

1. One concept written two ways → counted as two things → a weaker signal (notes 1, 3).
2. Once counted, the counts have to be weighted, because "appears often" ≠ "matters" (note 2).
3. Counting produces very many dimensions, and the geometry of the space changes character (note 4).
4. If the answer is accidentally counted into the input, every measured number is meaningless (notes 5, 6, 7).

---

## Three ideas to discard right away

**"More features is better."** False. In this project, KNN over the full text
(0.4059) is **worse** than KNN over the title alone (0.5307), and worse than the
baseline that uses no machine learning at all (0.4321). See
[note 4](04-ma-tran-thua-va-so-chieu.md).

**"More preprocessing makes a better model."** False. The nine Vietnamese
processing steps in this project contribute **+0.0017** macro-F1 in total. One
hyper-parameter line contributes **+0.0287** — nearly 17× more. See
[02-vietnamese-nlp §5](../02-vietnamese-nlp.md#5-the-closed-tracks-ablation--evidence-not-direction).

**"A high score means a good model."** Not necessarily. A high score is usually a
sign of leakage before it is a sign of a good model. See
[note 5](05-ro-ri-du-lieu.md).

These three are not opinions. They are **measured** conclusions from this project,
and each has a row in [04-results.md](../archive/04-results-ml.md) standing behind
it.
