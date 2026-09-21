[← Overview](../00-tong-quan.md)

# Background — for newcomers to deep learning

Five notes explaining **concepts**; they contain no experiment results. They do not
have to be updated after every model run — which is exactly why they sit apart
here.

Every example is taken from the VietJobs corpus itself, not from a textbook.

---

## Read in this order

| # | Note | What you will understand | Pairs with |
|---|---|---|---|
| 1 | [Why clean the data at all](01-vi-sao-phai-lam-sach.md) | The real reason is not "garbage in, garbage out". It is **dimension splitting** and **leakage** | [01-data-audit](../01-data-audit.md) |
| 2 | [Data leakage](05-ro-ri-du-lieu.md) | Three kinds of leak, and why **none of them raises an error** | [03-protocol](../03-protocol.md) |
| 3 | [Metrics and baselines](06-do-luong-va-baseline.md) | Why accuracy lies when the classes are skewed 27:1 | [04-results](../04-results.md) |
| 4 | [Regularisation](07-chinh-quy-hoa.md) | Why a model that needs heavy regularisation is a model calling for help | [07-bai-toan-luong](../07-bai-toan-luong.md) |
| **5** | [**Semantic vectors**](09-vector-ngu-nghia.md) | Subwords · context · 768 dense dimensions · frozen versus fine-tuned | [02-vietnamese-nlp](../02-vietnamese-nlp.md) · [06-baseline-dl](../06-baseline-dl.md) |

The file names keep their original numbers so that existing links do not break.

---

## The one sentence these notes serve

> Place each posting at **a point in a 768-dimensional space**, so that two
> postings in the same occupation land near each other even when they share no
> words at all.

Everything else follows from what that sentence does **not** protect you from:

1. One concept written two ways lands in two different places → a weaker signal
   (note 1).
2. If the answer is accidentally present in the input, every measured number is
   meaningless (notes 2, 3, 4).

---

## Two ideas to discard right away

**"More input is better."** False, and measured here: `province` covers 217 values
of which 101 appear once, 80 % of postings are Hà Nội or HCM, and it carries almost
no signal for either task. More columns is not more information.

**"A high score means a good model."** Not necessarily. A high score is usually a
sign of leakage before it is a sign of a good model. See
[note 2](05-ro-ri-du-lieu.md).

Neither is an opinion; both are measured conclusions from this project, and each
has a row in [04-results.md](../04-results.md) or a figure in
[05](../05-phan-tich-du-lieu.md) standing behind it.
