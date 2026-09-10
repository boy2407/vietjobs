[← Reading a comparison table](08-doc-mot-bang-so-sanh.md) · [Background](00-index.md)

# 9. Semantic vectors — what PhoBERT returns

The previous eight notes are built on **one** sentence: *machine learning on text is
counting character sequences and finding weights*. This note replaces that sentence
with another:

> Stop counting words. Instead, place each passage at **a point** in a
> 768-dimensional space, so that two postings in the same occupation land near each
> other — even when they share no words at all.

After this note you will understand why
[02-vietnamese-nlp](../02-vietnamese-nlp.md) reverses two earlier verdicts, and why
the TF-IDF track had to close.

---

## 1. The problem counting cannot solve

Two job postings, the same occupation:

```
A:  Nhân viên bán hàng tại cửa hàng        (in-store sales staff)
B:  Chuyên viên kinh doanh khu vực         (regional sales specialist)
```

Their bags of words:

```
A:  {nhân_viên, bán_hàng, tại, cửa_hàng}
B:  {chuyên_viên, kinh_doanh, khu_vực}
```

**Not one word in common.** To TF-IDF these two vectors are orthogonal — cosine zero,
i.e. "entirely unrelated". A linear model can only learn that they share a class if it
**sees each word separately, often enough**, in the training set.

That is the hard limit of counting: it has no notion of *meaning*. `bán_hàng` and
`kinh_doanh` are two unrelated dimensions, exactly as far apart as `bán_hàng` and
`hàn_xì` (welding). The matrix holds no information that could say otherwise.

No amount of extra preprocessing fixes it — tone normalisation, abbreviation
expansion and segmentation all just make *the same word* land in *the same cell*.
They cannot bring **two different words** close together.

---

## 2. The idea: let context define the word

Every modern language model rests on an old observation:

> A word is defined by the words that tend to stand beside it.

`bán_hàng` and `kinh_doanh` rarely co-occur, but they share **neighbours**:
`khách_hàng`, `doanh_số`, `chỉ_tiêu`, `hoa_hồng` (customers, revenue, targets,
commission). `hàn_xì` has entirely different neighbours: `công_trường`, `bảo_hộ`,
`thép` (construction site, protective gear, steel).

So: read an enormous corpus, give each word a list of numbers, then **adjust those
numbers until words sharing neighbours receive similar lists**. That is the whole job
of the *pre-training* phase.

The result: `bán_hàng` and `kinh_doanh` sit close together in the space, not because
someone declared it, but because the corpus uses them the same way.

The key point for understanding this project: **that phase is already done, by
someone else, on a different corpus.** VinAI trained PhoBERT on tens of gigabytes of
Vietnamese. We just download the weights and use them. That is why a 135-million
parameter model can run on 33 thousand job postings without instantly overfitting.

---

## 3. Three differences from TF-IDF

| | TF-IDF | PhoBERT |
|---|---|---|
| Dimensions | 236,596 — **depends on the corpus** | 768 — fixed, whatever the corpus |
| What a dimension means | One specific word. Dimension 8,412 = the word "kế_toán" | **No name.** No dimension corresponds to a word |
| How many cells one posting fills | A few dozen non-zero cells, the rest zeros — **sparse** | All 768 cells non-zero — **dense** |
| Learned from | Only the 33 thousand postings in train | Tens of GB of Vietnamese, before it ever saw this project |
| Word order | Discarded (bag of words) | Kept — the model reads in sequence |
| A never-seen word | Ignored entirely | Broken into small pieces, still gets a vector |

The first two rows are the most disappointing. With TF-IDF we can **open the lid**:
print the weight of each word and see immediately which class the model scores
`kế_toán` highly for. With 768 dense dimensions we cannot. No dimension has a name,
and there is no way to read one out in words. We trade explainability for
generalisation.

---

## 4. Subwords — why there is no longer an "unknown word"

PhoBERT does not cut text into words. It cuts into **subwords**, with the BPE
algorithm: character sequences that frequently occur together are merged into one
piece, and the rest is chopped small.

```
nhân_viên        →  [nhân_viên]                one piece, very common
lập_trình_viên   →  [lập_trình] [_viên]        two pieces
xúc_tác_quang    →  [xúc] [_tác] [_quang]      three pieces, rare
Nhan Vien        →  [Nh] [an] [Vi] [en]        four fragments, nearly meaningless
```

The good consequence: there is no longer any "out-of-vocabulary word". Every
character sequence can be split into pieces, so every posting has a vector.

The bad consequence, and this is what to remember: **the rarer the piece, the fainter
the vector.** The model has an excellent vector for `[nhân_viên]` because it met that
piece millions of times during pre-training; it knows almost nothing about `[Nh]`
standing next to `[an]`.

That is exactly why the preprocessing steps in
[02-vietnamese-nlp](../02-vietnamese-nlp.md) are **still needed**, even though the
reason has changed:

| Step | Old reason (TF-IDF) | New reason (PhoBERT) |
|---|---|---|
| Unicode and tone-mark normalisation | `hoà` and `hòa` occupy two dimensions | `hoà` and `hòa` cut into two different piece sequences |
| Abbreviation expansion | `NV` and `nhân viên` are two unrelated dimensions | `NV` is a rare piece, `nhân_viên` is a common one |
| Word segmentation | redundant — bigrams already solved it | **mandatory** — PhoBERT's piece table was built on segmented text |

The last row is the reversal. PhoBERT learned on *segmented* text, so its piece table
contains `nhân_viên` as **one** unit. Feed it unsegmented "nhân viên" and the model
must use two other pieces — not wrong, but two pieces it has seen far less often.

For the same reason, the **2,148 titles written without diacritics** in this corpus
are a problem with no answer yet: no step restores the diacritics, and without them
every piece is rare.

---

## 5. What "context" means — one word, two vectors

The generation before PhoBERT (word2vec, GloVe) gave each word **exactly one** vector:
a table lookup and done. PhoBERT does not: a piece's vector depends on the whole
sentence containing it.

```
"quản lý cửa hàng"        →  quản_lý here is a JOB TITLE (store manager)
"quản lý kho hàng tồn"    →  quản_lý here is a TASK (managing stock)
```

Two sentences give two different `quản_lý` vectors. The model reads the whole sequence
before deciding, so it can tell the two senses apart — something neither a bag of
words nor word2vec can do.

This is why **stopword removal is dropped entirely** on the new track. With TF-IDF,
deleting "của", "và", "tại" (of, and, at) was merely redundant. With PhoBERT, those
very words build the relations among the remaining ones — deleting them presents a
kind of sentence the model has never read.

---

## 6. From 256 vectors to 1 vector

Feed in a posting and PhoBERT returns **one vector per piece** — at most 256 pieces,
with anything longer truncated. But the output we need is *one* vector for the whole
posting.

How this project pools them: a **masked mean** — sum the vectors of the real pieces
and divide by the number of real pieces, ignoring the padding.

There is a more common alternative: take the vector of the opening `<s>` token. The
project does **not** use it, for a memorable reason: the `<s>` vector only becomes
meaningful **after** the model has been trained for a specific task. With an
un-fine-tuned PhoBERT it was never trained for anything. Averaging the pieces retains
more lexical signal — exactly what the occupation label depends on.

Implementation details in [`dl/encode.py`](../../src/vietjobs/dl/encode.py).

---

## 7. Frozen or fine-tuned — two levels, not two options

**Frozen.** The PhoBERT weights do not change. It is just a machine turning text into
768 numbers. Because the weights are fixed, a posting's vector is fixed too — compute
it once, store it on disk, and every subsequent training loop runs over a numeric
matrix, hundreds of times faster.

**Fine-tuned.** Unlock the weights and let the whole model keep learning on our data.
The vectors are no longer fixed, so nothing can be cached; every epoch has to push all
the data through 135 million parameters again.

The order is **always frozen first**, and the reason is diagnostic rather than
economic: if the frozen version performs abnormally badly, the fault is almost
certainly in the data path — the wrong column, misaligned labels, a leak. Finding that
out at a level that runs in minutes is far cheaper than finding it after hours on a
GPU.

---

## 8. Three things semantic vectors do **not** give for free

1. **They do not know our labels.** PhoBERT has never heard of this project's 16
   occupation classes. It only supplies a good representation; mapping that
   representation onto labels is still a model that has to be trained.
2. **They do not fix wrong labels.** If two classes have a blurred boundary and the
   annotators were inconsistent, no vector however good gets past that ceiling.
3. **They do not explain themselves.** With TF-IDF we can point at the word that drove
   a decision. With 768 anonymous dimensions we need dedicated tooling — and in a
   thesis, losing explainability is a price that must be stated, not a detail to skip.

---

## One sentence to replace the old one

> The TF-IDF track asks: **which words appear?**
> The PhoBERT track asks: **which postings is this one like?**
>
> The second question can be answered even for postings that share no words — and that
> is the entire reason for the change of track.

---

## Read next

- [Vietnamese processing](../02-vietnamese-nlp.md) — the nine steps, and which ones PhoBERT actually needs
- [Deep-learning baseline](../06-baseline-dl.md) — the real architecture, and the measured numbers
- [note 3 · n-grams and word boundaries](03-ngram-va-ranh-gioi-tu.md) — why the same
  segmentation step yields two opposite verdicts
- [note 4 · sparse matrices](04-ma-tran-thua-va-so-chieu.md) — 236,596 sparse
  dimensions versus 768 dense ones
