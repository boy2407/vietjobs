[← Why clean the data](01-vi-sao-phai-lam-sach.md) · [Background](00-index.md) · [n-grams and word boundaries →](03-ngram-va-ranh-gioi-tu.md)

# 2. What TF-IDF is

> **A note from the closed track.** Since 2026-09-08 the main track of this project
> is semantic vectors from PhoBERT ([note 9](09-vector-ngu-nghia.md)), not TF-IDF.
> This note is kept because it explains what *counting words* means — and without
> understanding the counting you cannot see what makes a semantic vector different.

A machine-learning model can only eat **numbers**. TF-IDF is a way of turning a
passage of text into a list of numbers. The whole idea fits in one sentence:

> A word matters to this document if it **appears often in this document** but is
> **rare in the other documents**.

Those two clauses are exactly TF and IDF.

**Full name: Term Frequency – Inverse Document Frequency.** Four words; read them
one at a time and you get the formula:

| Word | Meaning | In the formula |
|---|---|---|
| **T**erm | the *word* (or n-gram) — the unit being counted | `t` |
| **F**requency | how many **times** it occurs | `tf(t, d)` |
| **I**nverse | the more there are, the smaller it gets | the division in `N / df` |
| **D**ocument **F**requency | how many **documents** contain the word | `df(t)` |

Two things that are easy to misread:

- **"Inverse" attaches to "Document Frequency", not to "Term Frequency".**
  Read it as `TF × I(DF)`. Term frequency keeps its direction — repeat more, score
  higher; only the document frequency gets flipped.
- **"Document Frequency" is not "frequency within the document".** It counts how
  many **postings** contain the word, not how many times the word appears.

---

## 1. Bag of words — the first step, and its price

Before TF-IDF there is a brutal simplification to accept: the **bag of words**.
Text is treated as a bag holding words, **with no order**.

```
"Nhân viên kinh doanh"  →  {nhân: 1, viên: 1, kinh: 1, doanh: 1}
"Doanh kinh viên nhân"  →  {nhân: 1, viên: 1, kinh: 1, doanh: 1}   ← identical!
```

The model **cannot tell** those two apart. That is a real loss, and we pay it
knowingly in exchange for two things:

- A sparse matrix that computes very fast — 27 experiments in an afternoon.
- A linear model whose per-word weights are inspectable, so its predictions can be
  explained.

The partial patch for this loss is **n-grams** — see
[note 3](03-ngram-va-ranh-gioi-tu.md). The complete patch is a neural network that
reads in order (PhoBERT), recorded in
[09-lo-trinh.md — Priority 4](../archive/09-lo-trinh-ml.md#ưu-tiên-4--chuyển-trục-chính-sang-học-sâu-đa-nhiệm).

---

## 2. TF — frequency inside the document

`tf(t, d)` = how many times the term `t` appears in document `d`.

But raw counting has a problem: a word appearing 10 times is **not** ten times as
important as a word appearing once. Job postings repeat sector keywords for SEO
reasons, not because the posting is "more sales" than another.

So the project sets `sublinear_tf=True`, changing the formula to:

```
tf = 1 + log(number of occurrences)
```

| Occurrences | Raw `tf` | `1 + log(tf)` |
|---|---|---|
| 1 | 1 | 1.00 |
| 2 | 2 | 1.69 |
| 5 | 5 | 2.61 |
| 10 | 10 | 3.30 |
| 100 | 100 | 5.61 |

The second occurrence adds a lot of information; the hundredth adds almost nothing.
A log scale says exactly that.

---

## 3. IDF — rarity across the corpus

The second clause fixes a large hole in TF: the word *"công ty"* (company) appears
in nearly **every** job posting. Its TF is high, but it distinguishes no occupation
from any other.

`idf` downweights words that are everywhere. The formula sklearn uses:

```
idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1
```

- `N` = the total number of documents (here: **33,396** postings in train)
- `df(t)` = the number of documents that **contain** `t` (how many postings, not how many times)

The more common a word, the larger `df`, the smaller `idf`.

| If a word appears in… | `df` | `idf` |
|---|---|---|
| 1 % of postings | 334 | 5.60 |
| 10 % of postings | 3,340 | 3.30 |
| 50 % of postings | 16,698 | 1.69 |
| 90 % of postings | 30,056 | 1.11 |

**TF-IDF = tf × idf**. One more step remains — L2 normalisation — in the next
section.

---

## 4. L2 normalisation — why every posting must have length exactly 1

Multiplying `tf × idf` still does not give something usable. One disease is
untreated: **long postings automatically beat short ones**.

### The example corpus

One corpus serves both this section and [§5](#5-the-matrix-made-visible).

```
D1 = "tuyển nhân viên kinh doanh"                      sales · short
D2 = "nhân viên kinh doanh phụ trách tìm kiếm khách     sales · LONG
      hàng và chăm sóc khách hàng cũ"
D3 = "kế toán tổng hợp"                                 accounting
D4 = "kế toán trưởng có kinh nghiệm"                    accounting
D5 = "kỹ sư xây dựng công trình"                        engineering
```

Five postings, three occupations, `N = 5`. **D1 and D2 are the same occupation**,
and D2 is the long one.

### The disease

A long posting has more non-zero entries, so its vector is **longer** — in the
geometric sense. And every calculation downstream reads that length: KNN's
distance, SVM's `w · x` score. The model mislearns that "long postings are
special", when the length only reflects whether that company writes a lot of
marketing copy.

Measuring the distance from **D1** to the others:

| | `‖D1‖` | `‖D2‖` | `‖D3‖` | `d(D1,D2)` | `d(D1,D3)` | Nearest to D1 |
|---|---|---|---|---|---|---|
| **No L2** | 3.870 | **8.430** | 3.813 | 8.055 | 5.433 | **D3 — wrong occupation** ✗ |
| **With L2** | 1.000 | 1.000 | 1.000 | 1.163 | 1.414 | **D2 — right occupation** ✓ |

Look closely at the first row, because it is absurd enough to be memorable:

- D1 and D2 **share four words** (`nhân`, `viên`, `kinh`, `doanh`) → distance **8.055**
- D1 and D3 **share no word at all** → distance **5.433**

Two unrelated postings end up closer together than two in the same occupation. The
reason is trivial: `‖D1‖ = 3.870` and `‖D3‖ = 3.813` are nearly equal — **both
short**. The distance is measuring a difference in text length, not content.

Distance-based models take the hardest hit — see
[note 4](04-ma-tran-thua-va-so-chieu.md) on why KNN collapses.

Reproduce the table above (the corpus is declared in [§5 · Reproduce](#reproduce)):

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

for norm in (None, "l2"):
    X = TfidfVectorizer(sublinear_tf=True, norm=norm).fit_transform(corpus.values()).toarray()
    print(norm, [round(np.linalg.norm(X[0] - X[i]), 3) for i in range(1, 5)])
```

### What "length exactly 1" means

Divide the whole vector by its own length. But "length" here is **Pythagoras**, not
addition.

Take the 2-dimensional vector `(3, 4)`. Its length is not 3 + 4 = 7, it is:

```
√(3² + 4²) = √25 = 5      →      (3, 4) / 5 = (0.6 , 0.8)
```

Check the result:

- sum of **squares**: 0.6² + 0.8² = 0.36 + 0.64 = **1** ✓
- plain sum: 0.6 + 0.8 = **1.4** ✗

> ⚠️ This is very easy to misremember. L2 does **not** make the values sum to 1 —
> it makes their **squares** sum to 1.

The D1 vector after normalisation, checked against the real numbers:

| Word | TF-IDF | Squared |
|---|---|---|
| tuyển | 0.5422 | 0.2940 |
| nhân | 0.4375 | 0.1914 |
| viên | 0.4375 | 0.1914 |
| doanh | 0.4375 | 0.1914 |
| kinh | 0.3631 | 0.1318 |
| **Sum** | **2.2178** ✗ | **1.0000** ✓ |

After L2, each posting is **an arrow of length exactly 1** from the origin. Postings
now differ only in **direction** — the proportions among their words — not in how
much text they contain.

### Why L2 and not L1

Normalising so that the values **sum** to 1 is also a real thing, called **L1**.

| | Constraint | What it lets you read |
|---|---|---|
| **L1** | Σ value = 1 | What **percentage** of the posting each word is |
| **L2** | Σ (value)² = 1 | Each posting is an arrow of length 1 |

L1 sounds more intuitive, and it also cures the long-posting disease. The project
still picks L2 for a geometric reason: **after L2, the dot product of two postings
is exactly the cosine of the angle between them.** That relation only holds for L2,
because Pythagoras is the geometry of Euclidean distance — and `LinearSVC` (`w · x`)
and KNN (distance) both work in exactly that geometry. Choose L1 and the downstream
measurements lose this tidy interpretation.

### The price

**1. L2 erases length information entirely.** And length is sometimes genuinely
useful — a senior-position posting usually has a longer description than a seasonal
one. The project recovers it by another route: the numeric columns `desc_len`,
`req_len`, `title_len` do not pass through TF-IDF, so L2 never touches them. This
is a design decision worth noticing — **throw information away in one channel and
reintroduce it in another, in a cleaner form**, instead of letting it contaminate a
distance measurement. See [§7](#7-what-tf-idf-cannot-see).

**2. Long postings have their keywords diluted.** D1 and D2 both have length 1 after
L2, but they divide that budget differently:

| | Non-zero cells | `nhân` | `viên` | `doanh` | `kinh` |
|---|---|---|---|---|---|
| D1 — short posting | 5 | **0.4375** | **0.4375** | **0.4375** | **0.3631** |
| D2 — long posting, **same occupation** | 14 | 0.2009 | 0.2009 | 0.2009 | 0.1667 |

D2 has to spread its length of 1 over 14 dimensions, so each keyword fades by more
than half; D1 concentrates everything into 5 dimensions and stays strong. Mostly
this is the **correct** behaviour — inside a long posting, `kinh doanh` genuinely is
a small part of the content. But it has a downside: a posting stuffed with generic
marketing copy ("young environment", "attractive benefits") dilutes its own
occupation signal. The compensation is **block weighting**, to restore what was
spread thin —
[09-lo-trinh.md Priority 3a](../archive/09-lo-trinh-ml.md#ưu-tiên-3--hai-đòn-bẩy-rẻ-cho-mốc-cơ-sở-ml).

---

## 5. The matrix made visible

The four sections above describe **each step**. This one prints the **final
object** the model actually receives — still on the 5-posting corpus from
[§4](#4-l2-normalisation--why-every-posting-must-have-length-exactly-1), small
enough to take in at a glance.

### Table 1 · The vocabulary — learned once, shared by every posting

Count `df` = **the number of postings containing** the word, then
`idf = ln((1+5)/(1+df)) + 1`.

| `df` | `idf` | Which words |
|---|---|---|
| 3 | ln(6/4) + 1 = **1.4055** | `kinh` |
| 2 | ln(6/3) + 1 = **1.6931** | `doanh`, `kế`, `nhân`, `toán`, `viên` |
| 1 | ln(6/2) + 1 = **2.0986** | the remaining 22 words |

There are only **three** `idf` values, because this tiny corpus has only three `df`
levels. The rarer the word, the larger the `idf` — `kinh` is in 3 of 5 postings, so
it is pushed lowest.

This is a **vector of 28 numbers**, not a matrix. It does not depend on any one
posting.

### Table 2 · The computation chain, for D1 alone

| Word | 1 · raw `tf` | 2 · `1+ln(tf)` | 3 · `df` | 4 · `idf` | 5 · `tf × idf` | 6 · `÷ ‖D1‖` |
|---|---|---|---|---|---|---|
| tuyển | 1 | 1.0 | 1 | 2.0986 | 2.0986 | **0.5422** |
| nhân | 1 | 1.0 | 2 | 1.6931 | 1.6931 | **0.4375** |
| doanh | 1 | 1.0 | 2 | 1.6931 | 1.6931 | **0.4375** |
| viên | 1 | 1.0 | 2 | 1.6931 | 1.6931 | **0.4375** |
| kinh | 1 | 1.0 | 3 | 1.4055 | 1.4055 | **0.3631** |

```
‖D1‖ = √(2.0986² + 1.6931² + 1.6931² + 1.6931² + 1.4055²) = √14.9800 = 3.8704
```

The sum of squares of the last row = **1.000000** ✓

Columns 1 and 2 are identical — **every `tf` in D1 equals 1**, because D1 repeats no
word. So for D1 specifically, TF-IDF really reduces to `idf` after normalisation.
Table 4 uses D2 to make TF do something.

### Table 3 · The final matrix — 5 rows × 28 columns

Each **row** is a posting, each **column** a word. A `·` is a zero.

```
    chăm    có  công    cũ doanh  dựng  hàng   hợp khách  kinh  kiếm    kế    kỹ nghiệm
D1     ·     ·     ·     ·  0.44     ·     ·     ·     ·  0.36     ·     ·     ·      ·
D2  0.25     ·     ·  0.25  0.20     ·  0.42     ·  0.42  0.17  0.25     ·     ·      ·
D3     ·     ·     ·     ·     ·     ·     ·  0.55     ·     ·     ·  0.44     ·      ·
D4     ·  0.46     ·     ·     ·     ·     ·     ·     ·  0.31     ·  0.37     ·   0.46
D5     ·     ·  0.41     ·     ·  0.41     ·     ·     ·     ·     ·     ·  0.41      ·

    nhân   phụ   sóc    sư  toán trách trình trưởng tuyển   tìm  tổng  viên    và   xây
D1  0.44     ·     ·     ·     ·     ·     ·      ·  0.54     ·     ·  0.44     ·     ·
D2  0.20  0.25  0.25     ·     ·  0.25     ·      ·     ·  0.25     ·  0.20  0.25     ·
D3     ·     ·     ·     ·  0.44     ·     ·      ·     ·     ·  0.55     ·     ·     ·
D4     ·     ·     ·     ·  0.37     ·     ·   0.46     ·     ·     ·     ·     ·     ·
D5     ·     ·     ·  0.41     ·     ·  0.41      ·     ·     ·     ·     ·     ·  0.41
```

**Read by row = one posting.** D1 has exactly 5 non-zero cells — its 5 words. The
other 23 columns are 0: D1 "does not contain" every other word. Each row's squares
sum to 1.

**Read by column = one word, across the corpus.** The `kinh` column is non-zero in
D1, D2 and D4. The `sư` column is non-zero only in D5.

**The same word takes different values across postings.** `nhân` is 0.44 in D1 but
only 0.20 in D2 — same `idf`, same `tf`, differing **purely because of L2**. This is
the dilution from
[§4](#4-l2-normalisation--why-every-posting-must-have-length-exactly-1), visible
directly in the matrix: row D2 spreads thin across 14 columns, row D1 concentrates
into 5.

**Rows D1 and D3 do not intersect in a single column.** That is exactly the pair
[§4](#the-disease) showed being ranked *closest together* without L2 — visual proof
that the distance was not measuring content at all.

**The matrix does not know about occupations.** D1 and D2 are both sales postings,
and that only shows through four shared columns. D3 and D4 are both accounting, with
two shared columns. There is no column called "occupation" — knowing the occupation
is the job of the label and of the classifier, not of this matrix. See
[§7](#7-what-tf-idf-cannot-see).

**35 non-zero cells out of 140 — 25 %.** On real data this ratio is hundreds of
times smaller; that is what "sparse" means, see
[note 4](04-ma-tran-thua-va-so-chieu.md).

### Table 4 · TF actually doing something — when a word repeats

D2 repeats `khách` and `hàng` twice ("tìm kiếm **khách hàng** mới và chăm sóc
**khách hàng** cũ" — find new *customers* and care for old *customers*). This is the
only place in the corpus with `tf > 1`:

| Word in D2 | `tf` | `1+ln(tf)` | `idf` | `tf × idf` | After L2 |
|---|---|---|---|---|---|
| khách | **2** | **1.6931** | 2.0986 | 3.5533 | **0.4215** |
| hàng | **2** | **1.6931** | 2.0986 | 3.5533 | **0.4215** |
| nhân | 1 | 1.0000 | 1.6931 | 1.6931 | 0.2009 |
| kinh | 1 | 1.0000 | 1.4055 | 1.4055 | 0.1667 |

`khách` is **more than twice** as strong as `nhân` inside the same posting. Two
forces add up: it repeats (high TF) **and** it is rare in the corpus (high IDF).
That is precisely the definition of "a keyword characteristic of this document".

With a raw `tf` instead of `1+ln(tf)`, `khách` would get `2 × 2.0986 = 4.1972`
instead of `3.5533` — 18 % stronger. And a posting stuffing an SEO keyword twenty
times would push all of its own other words toward 0, because L2 forces the row to
length exactly 1. That is why the project sets `sublinear_tf=True`.

### Reproduce

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = {"D1": "tuyển nhân viên kinh doanh",
          "D2": "nhân viên kinh doanh phụ trách tìm kiếm khách hàng "
                "và chăm sóc khách hàng cũ",
          "D3": "kế toán tổng hợp",
          "D4": "kế toán trưởng có kinh nghiệm",
          "D5": "kỹ sư xây dựng công trình"}
V = TfidfVectorizer(sublinear_tf=True, norm="l2")
X = V.fit_transform(corpus.values())
print(pd.DataFrame(X.toarray().round(2), index=corpus,
                   columns=V.get_feature_names_out()).replace(0, "·").to_string())
```

---

## 6. A worked example — on the real data

Take the most common title in the corpus: **"Nhân viên kinh doanh"** (sales staff,
754 postings in train). With `ngram_range=(1,2)` it generates **7 features**:

| Feature | Kind | `df` (of 33,396 postings) | `idf` | **TF-IDF** |
|---|---|---|---|---|
| `viên kinh` | bigram | 3,321 · 9.94 % | 3.308 | **0.491** |
| `kinh doanh` | bigram | 4,764 · 14.27 % | 2.947 | **0.437** |
| `doanh` | unigram | 4,899 · 14.67 % | 2.919 | **0.433** |
| `kinh` | unigram | 5,118 · 15.33 % | 2.876 | **0.427** |
| `nhân viên` | bigram | 14,494 · 43.40 % | 1.835 | **0.272** |
| `nhân` | unigram | 14,951 · 44.77 % | 1.804 | **0.268** |
| `viên` | unigram | 19,181 · **57.44 %** | 1.554 | **0.231** |

Reproduced with `_word_tfidf(PrepConfig(), max_features=60_000)` fitted on the
`job_title` column of `train.parquet`. All 7 cells have `tf = 1` (each word appears
once), so the TF-IDF column here is just `idf` after L2 normalisation.

### Four things to read off this table

**a. The vector has 6,835 dimensions but only 7 non-zero cells.** The title
vocabulary is exactly **6,835** words — the number in the diagram in
[06-mo-hinh-phan-lop.md](../archive/06-mo-hinh-phan-lop.md). The other 6,828
dimensions are zero. That is what "sparse" means —
[note 4](04-ma-tran-thua-va-so-chieu.md).

**b. `viên` is weakest (0.231) even though it is the most common word.** That is
IDF working: `viên` appears in 57 % of postings, so it distinguishes almost nothing.
It occurs in nhân viên, chuyên viên, kỹ thuật viên, giáo viên — staff, specialist,
technician, teacher: four different occupations.

**c. `viên kinh` is strongest (0.491) — and it is a grammatically meaningless
bigram.** "viên kinh" is not a Vietnamese word. But it is **rare** (9.94 %), so it
is an excellent fingerprint for exactly the phrase "nhân viên kinh doanh". The
machine does not need grammar — it needs a character sequence that discriminates.

**d. `viên` was nearly dropped.** The project sets `max_df=0.6` — any word present
in more than 60 % of documents is thrown out. `viên` sits at **57.44 %**, right at
the threshold. The phrase `công ty`, more common in *descriptions*, is cut by
`max_df` in that block. This is a **stopword filter learned automatically from the
data itself** — and the reason the manual stopword step measured only **+0.0002**.

---

## 7. What TF-IDF **cannot** see

Knowing a tool's limits matters as much as knowing how to use it.

| What it cannot see | The consequence in this project |
|---|---|
| **Word order** | "không yêu cầu kinh nghiệm" and "yêu cầu kinh nghiệm" differ by a single `không` token. Bigrams patch part of it |
| **Meaning / synonyms** | `NV` and `nhân viên` are two unrelated dimensions until the abbreviation-expansion step merges them |
| **Negation, irony, long-range context** | This is why the bag-of-words ceiling sits below the PhoBERT ceiling |
| **Text length** | [L2 normalisation](#4-l2-normalisation--why-every-posting-must-have-length-exactly-1) erases it. To keep it you have to add a separate column — exactly what `desc_len`, `req_len` and `title_len` do |

This table explains why the project **adds** 14 numeric columns and 3 one-hot
columns alongside TF-IDF: they carry precisely what TF-IDF discards. And the weight
table in
[05-dac-trung-tfidf.md §9](../archive/05-dac-trung-tfidf.md#9-khối-nào-thật-sự-được-dùng)
shows the model takes those columns **very** seriously.

---

## Back to the project itself

- [05-dac-trung-tfidf.md](../archive/05-dac-trung-tfidf.md) — every TF-IDF parameter in this project, and what happens without it
- [note 3 — n-grams and word boundaries](03-ngram-va-ranh-gioi-tu.md) — why `(1,2)` and not `(1,1)`
- [note 4 — sparse matrices](04-ma-tran-thua-va-so-chieu.md) — what 7 non-zero cells out of 6,835 dimensions means
