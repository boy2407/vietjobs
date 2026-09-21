[← Regularisation](07-chinh-quy-hoa.md) · [Background](00-index.md)

# 9. Semantic vectors — what PhoBERT returns

One sentence carries this whole note:

> Stop counting words. Instead, place each passage at **a point** in a
> 768-dimensional space, so that two postings in the same occupation land near each
> other — even when they share no words at all.

After this note, the preprocessing choices in
[02-vietnamese-nlp](../02-vietnamese-nlp.md) stop looking arbitrary.

---

## 1. The problem word matching cannot solve

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

**Not one word in common.** Represent each posting by the words it contains and these
two are orthogonal — "entirely unrelated". Such a representation has no notion of
*meaning*: `bán_hàng` and `kinh_doanh` are two unrelated dimensions, exactly as far
apart as `bán_hàng` and
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

## 3. Six properties of the representation

| Property | PhoBERT vectors |
|---|---|
| Dimensions | 768 — fixed, whatever the corpus |
| What a dimension means | **Nothing nameable.** No dimension corresponds to a word |
| How many cells one posting fills | All 768 are non-zero — **dense** |
| Learned from | Tens of GB of Vietnamese, before it ever saw this project |
| Word order | Kept — the model reads in sequence |
| A never-seen word | Broken into small pieces, still gets a vector |

The second row is the expensive one. A representation whose dimensions are words can
be opened up: print the weight of each word and read off what drove a decision. With
768 anonymous dimensions that is not available. Explainability is traded for
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

| Step | Why PhoBERT needs it |
|---|---|
| Unicode and tone-mark normalisation | `hoà` and `hòa` cut into two different piece sequences |
| Abbreviation expansion | `NV` is a rare piece, `nhân_viên` is a common one |
| Word segmentation | **mandatory** — PhoBERT's piece table was built on segmented text |

The last row matters most. PhoBERT learned on *segmented* text, so its piece table
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

This is why **stopword removal is dropped entirely**. Words like "của", "và", "tại"
(of, and, at) are exactly what builds the relations among the remaining ones —
deleting them presents a kind of sentence the model has never read.

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
3. **They do not explain themselves.** With 768 anonymous dimensions, pointing at what
   drove a decision needs dedicated tooling — and in a thesis, losing explainability
   is a price that must be stated, not a detail to skip.

---

## The sentence to keep

> The question is not **which words appear** but **which postings is this one like**.
>
> The second question can be answered even for postings that share no words — and that
> is the entire reason this representation is worth its cost.

---

## Read next

- [Vietnamese processing](../02-vietnamese-nlp.md) — the nine steps, and which ones PhoBERT actually needs
- [Deep-learning baseline](../06-baseline-dl.md) — the real architecture, and the measured numbers
- [note 7 · regularisation](07-chinh-quy-hoa.md) — what holds the dense head back
