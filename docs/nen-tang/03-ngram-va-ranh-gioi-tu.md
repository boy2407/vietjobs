[← What TF-IDF is](02-tf-idf-la-gi.md) · [Background](00-index.md) · [Sparse matrices →](04-ma-tran-thua-va-so-chieu.md)

# 3. n-grams and word boundaries

> **The conclusion in this note holds for TF-IDF only.** With PhoBERT the opposite
> is true: word segmentation is **mandatory**, because the model was pre-trained on
> segmented text. One step, two opposite verdicts — and *why* that happens is the
> most worthwhile part of this note. See [note 9](09-vector-ngu-nghia.md) and
> [02 §4](../02-vietnamese-nlp.md#4-which-steps-the-phobert-path-actually-runs).

This note explains a surprising result in the project: **the Vietnamese word
segmentation step — the most linguistically "correct" step of all — measured as no
help and was switched off.**

The reason is not that segmentation is meaningless. The reason is that
`ngram_range=(1,2)` already solves the same problem.

---

## 1. What an n-gram is

An n-gram is a sequence of `n` adjacent tokens.

```
"Nhân viên kinh doanh"

unigrams (1-grams):  nhân · viên · kinh · doanh
bigrams  (2-grams):  nhân viên · viên kinh · kinh doanh
trigrams (3-grams):  nhân viên kinh · viên kinh doanh
```

`ngram_range=(1,2)` means: take **both** unigrams **and** bigrams. The four-word
phrase above yields 4 + 3 = **7 features**, exactly as in the worked table in
[note 2](02-tf-idf-la-gi.md#6-a-worked-example--on-the-real-data).

Bigrams are how a bag of words patches back some of the ordering information it
threw away: it does not know the whole sentence order, but it knows **which two
words stand next to each other**.

---

## 2. Vietnamese's own problem

Most NLP tooling defaults to an assumption borrowed from English:

> **Whitespace separates words.**

For English that is roughly true: `"salesperson"` is one word, one token.
For Vietnamese it is **false**. Vietnamese is an **isolating** language: whitespace
separates **syllables**, not **words**.

```
English:     salesperson              →  1 token,  and indeed 1 word
Vietnamese:  nhân viên kinh doanh     →  4 tokens, but really 2 words
```

### The consequence: the syllable "viên" swallows four occupations

Measured in the VietJobs corpus: the syllable `viên` appears **27,053 times** in
titles, following **70 different syllables**:

| Phrase | Count | Which occupation |
|---|---|---|
| nhân **viên** | 20,138 | staff |
| chuyên **viên** | 4,858 | specialist |
| kỹ thuật **viên** | 498 | technician |
| giáo **viên** | … | teacher |

With **unigrams only**, all four occupations pour into the single `viên` dimension.
That dimension occurs in 57.44 % of postings, so it is nearly useless — it
distinguishes nothing.

---

## 3. Two fixes — and they do the same job

### Option A — word segmentation (`underthesea`)

Run a Vietnamese segmentation model and join the syllables of one word with an
underscore:

```
"Nhân viên kinh doanh"  →  "Nhân_viên kinh_doanh"
```

Now `nhân_viên` is its own token, and `chuyên_viên` is its own token.

### Option B — bigrams (`ngram_range=(1,2)`)

Requires knowing nothing about Vietnamese. Just pair every two adjacent words:

```
"Nhân viên kinh doanh"  →  ... + "nhân viên" + "viên kinh" + "kinh doanh"
```

Now `nhân viên` is also its own feature, and so is `chuyên viên`.

### Comparison

| | Segmentation | Bigrams |
|---|---|---|
| Gives `nhân_viên` / `nhân viên` its own feature | ✅ | ✅ |
| Needs a language model | ✅ underthesea | ❌ nothing |
| Cost | **1,533.7 seconds** over 48k postings | effectively zero |
| Generates junk features (`viên kinh`) | ❌ | ✅ yes |
| What happens when it is wrong | A bad cut → a completely wrong token | There is no notion of "wrong" |

**Both options solve the same problem.** And bigrams were on from the start.

---

## 4. The measured result

Same `C = 0.02`, only the segmentation step toggled:

| Configuration | macro-F1 (dev) | Run |
|---|---|---|
| Province normalisation only | **0.6050** | `cat-T-svm-C0.02` |
| Province + **segmentation** | 0.5998 | `cat-R-svm-C0.02-seg` |

Segmentation **lowers** the score by 0.0052 — inside the noise (σ ≈ 0.0077), so the
correct verdict is "no help", not "harmful".

**Why it does not help:** bigrams already catch `nhân viên`. Step 5 solves a problem
the vectoriser had already solved — the only difference being that it costs 25
minutes and an external library.

**Why it is mildly harmful:** segmentation **reduces** the number of discriminative
features. After segmentation, `nhân_viên` is one token, and the bigram of the
segmented text is `nhân_viên kinh_doanh` — `viên kinh` no longer exists. And by the
table in [note 2](02-tf-idf-la-gi.md), `viên kinh` was the **strongest** of all
seven features (TF-IDF 0.491), precisely because it is rare. Segmentation deleted
the strongest feature of the most common title.

---

## 5. What about accent-less postings — the character channel

**2,148 titles (4.50 %)** are written entirely without diacritics:

```
"Nhan Vien Kinh Doanh"    ← the word channel can never match "Nhân Viên Kinh Doanh"
```

This cannot be fixed by stripping diacritics from the whole corpus, because
**Vietnamese diacritics carry meaning** — `má / mà / mả / mã / mạ` are five
different words. Stripping them everywhere merges five meaningful dimensions into
one meaningless one.

The solution: a **second channel** running in parallel, using **character** n-grams
over the accent-stripped copy:

```
analyzer="char_wb", ngram_range=(3, 5), preprocessor=fold_accents

"nhan vien"  →  nha · han · nhan · vie · ien · vien · ...
"nhân viên"  →  (after folding) nha · han · nhan · vie · ien · vien · ...
                                 ↑ they match
```

The word channel keeps the meaning of the diacritics; the character channel bridges
the accent-less postings. **Additive, not a replacement.**

The measured result: 0.6024 versus 0.6050 — also **no help**, and also switched off.
4.50 % of postings is too few to pay for the 60,000 noisy dimensions this channel
adds.

---

## 6. The lesson

**Before adding a language-processing step, ask: has the vectoriser already done
it?**

Three real overlaps in this project:

| Manual step | Already done by | Measured result |
|---|---|---|
| Word segmentation | `ngram_range=(1,2)` | −0.0052 |
| Stopword removal | `max_df=0.6` and `idf` | +0.0002 |
| Accent-folded channel | (no overlap, but too little data) | −0.0026 |

A preprocessing step that is "theoretically right" is not automatically a useful
one. **Only an ablation row can decide** — that is the removal rule,
[03-protocol.md §8](../03-protocol.md#8-the-removal-rule).

---

## Back to the project itself

- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — the nine steps and the full ablation table
- [05-dac-trung-tfidf.md](../archive/05-dac-trung-tfidf.md#4-từng-tham-số-tf-idf--và-bỏ-đi-thì-sao) — `ngram_range` and its sibling parameters
- [note 2 — what TF-IDF is](02-tf-idf-la-gi.md) — the worked table of 7 features
