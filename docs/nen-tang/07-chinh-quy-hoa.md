[← Metrics and baselines](06-do-luong-va-baseline.md) · [Background](00-index.md) · [Semantic vectors →](09-vector-ngu-nghia.md)

# 7. Regularisation — why a model has to be held back

A network that is free to do anything will memorise its training set. This note
explains the four mechanisms this project uses to stop that, all of them visible in
`artifacts/<run_id>/config.json`.

---

## 1. Overfitting, stated precisely

**Overfitting** = the model memorises the particular details and the noise of the
training set instead of learning the general rule. Its signature is a single
pattern: the score on data it has already seen keeps improving while the score on
held-out data stops improving and then gets worse.

That is why every run here writes `history.jsonl` — one line per epoch with both
numbers. A final score alone cannot tell you whether this happened.

---

## 2. Why it happens on this project

The dense head sits on top of a **768-dimensional** vector and has a hidden layer of
256 and another of 128. That is on the order of 200,000 free parameters learning
from 34,354 training postings. Nothing stops such a model from finding a weight
combination that fits the training rows exactly — including the rows that are simply
mislabelled or ambiguous.

The model does not need to *understand* anything to do that. It only needs to
*memorise*, and what it memorises is useless on a new posting.

---

## 3. The four mechanisms in use

| Mechanism | Setting | What it does |
|---|---|---|
| **Dropout** | `dropout 0.3` | Each training step randomly zeroes 30 % of the hidden units, so no single unit can become a private shortcut to the answer. The network is forced to spread the representation out |
| **Weight decay** | `weight_decay 0.01` | Adds a penalty on the size of the weights: `error + λ × (sum of squared weights)`. The model now pays for every weight it assigns, and only pays when the feature genuinely reduces the error |
| **Early stopping** | `patience 8` | Training stops once the dev score has not improved for 8 epochs, and the weights kept are those of the best epoch — not the last one |
| **Gradient clipping** | `clip 1.0` | Caps the size of a single update. Without it one pathological batch can throw the weights somewhere they never recover from |

The first two shape *what* the model may learn; the last two shape *how far* it is
allowed to travel while learning it.

---

## 4. Reading the penalty term

Without regularisation the model optimises one thing:

```
minimise:   error on train
```

Weight decay changes the objective to:

```
minimise:   error on train  +  λ × (sum of squared weights)
```

Larger λ means a more tightly held model. Dimensions carrying no real signal get
pushed toward zero, and only dimensions that earn their keep retain a meaningful
weight.

Note the direction, because libraries disagree on it: some expose λ directly (bigger
= stronger restraint), others expose its inverse (smaller = stronger restraint).
Always check which one you are turning.

---

## 5. Standardisation is not regularisation — but it belongs here

PhoBERT vectors are **anisotropic**: every posting sits inside a narrow cone, so the
cosine between two arbitrary postings is high even when they have nothing to do with
each other. `LayerNorm` normalises *per sample*, so it cannot remove a direction
shared by all samples.

The fix is per-dimension standardisation using statistics taken from `train` only,
saved as `scaler.npz` next to the model — inference must use exactly those numbers
(Rule 4). Fitting those statistics on all the data instead would be a quiet leak:
the model would be told something about the evaluation set before being scored on
it.

---

## 6. Quick decision table

| Symptom | Means | What to do |
|---|---|---|
| Train score high, dev score low | Overfitting | More dropout · more weight decay · stop earlier |
| Train score low, dev score low too | Underfitting | Less restraint · a bigger head · a better representation |
| Both close together and both low | The representation carries too little signal | Change the input, do not turn hyper-parameters |
| Dev score collapses in the first epochs | Updates too large | Check gradient clipping and the learning rate |
| A linear probe on the same vectors nearly matches the network | The extra capacity is buying little | The ceiling is in the representation, not in the head |

---

## Back to the project itself

- [06-baseline-dl.md](../06-baseline-dl.md) — the five architecture decisions and
  what each was measured against
- [note 6 — metrics and baselines](06-do-luong-va-baseline.md) — why a score needs a
  floor before it can be read
- [note 9 — semantic vectors](09-vector-ngu-nghia.md) — what the 768 dimensions are
