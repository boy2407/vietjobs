[← Regularisation](07-chinh-quy-hoa.md) · [Background](00-index.md) · [Semantic vectors →](09-vector-ngu-nghia.md)

# 8. Reading a model comparison table

This note reports no results. It teaches how to **read** a comparison table — and
how to spot when that table is overstating things. The real numbers are in
[10-so-sanh-mo-hinh.md](../archive/10-so-sanh-mo-hinh.md).

This is the hardest of the background notes, because it is not about models but
about **how we evaluate models** — the place where someone who already writes code
can stay wrong for years without anyone pointing it out.

---

## 1. Hyper-parameters — the knobs you have to set yourself

When cooking, the **ingredients** are what you have, but the **oven temperature** is
what you choose. No formula derives the right temperature from the ingredients. The
only way is to bake at a few temperatures and taste.

A machine-learning model has two completely different kinds of number:

| Kind | Who decides | Example |
|---|---|---|
| **Weights** | The model **learns** them from the data | How much each word weighs for each occupation |
| **Hyper-parameters** | **You** set them before learning starts | `C`, `k`, number of trees, tree depth |

The model learns weights. It **cannot** learn hyper-parameters — you have to hand
them over first, and only then does it begin.

> ⚠️ This is the most common beginner's confusion: thinking "training a model" finds
> **all** the numbers. It does not. Training finds weights; hyper-parameters are your
> job, and finding them is an outer loop.

---

## 2. A "cell" is one training run

Because hyper-parameters have to be tried, we lay out a **table**: models as
columns, configurations as rows.

| | Config 1 | Config 2 | Config 3 |
|---|---|---|---|
| **Model A** | cell | cell | cell |
| **Model B** | cell | cell | cell |

**Each cell = one model trained from scratch.** A table of 6 models × 6
configurations = 36 cells = 36 training runs.

The technical names: the table is a **hyper-parameter grid**, and running all of it
is a **grid search**.

This is the unit in which fairness is discussed: *how many cells does each model
get?*

---

## 3. Equal cell counts are still not fair

Give two people six darts each. The first throws at a board with one ring; the second
throws at a board with eight rings stacked in eight different directions. Six throws
cover most of the first board. Six throws have barely touched the second.

The number of real knobs differs enormously between algorithms:

| Algorithm | Real knobs | What 6 cells means |
|---|---|---|
| KNN | **1** (`k`) | Covers most of the useful range |
| SVM · LogReg | **1** (`C`) | Most of a one-dimensional curve |
| RandomForest | ~3 | A portion |
| XGBoost | **~8** | A very small piece of an 8-dimensional space |

Six cells each, but one is a near-complete survey and the other is six lottery
tickets.

**How to spot who was actually explored:** look at the **spread** — the gap between
that model's best and worst cell. A narrow spread means the six cells gave nearly the
same result, i.e. we **never moved** in its space.

---

## 4. A maximum at the grid edge = a truncated grid

Measure the indoor temperature from 8am to noon and you find noon is the hottest.
Concluding "noon is the hottest time of day" is **wrong** — you simply never measured
1pm.

The same holds in a hyper-parameter table. If the winning configuration sits at the
**smallest** or **largest** value you tried, the true peak is very likely outside the
grid.

| Situation | How to read it |
|---|---|
| The peak is **inside** the grid, lower on both sides | ✅ The peak has been found |
| The peak is at an **edge** of the grid | ⚠️ The grid is truncated — widen it and measure again |

This is the cheapest check you have, and it costs one glance.

---

## 5. The winner's curse

Have 20 people each flip a coin 10 times. The one with the most heads might get 9/10.
Do you conclude that person is a skilled flipper? No — you **picked the winner after
looking at the results**, so their result always looks better than their true ability.

Selecting the best configuration on `val` and then reporting it **on that same `val`**
makes exactly that mistake. The number you get is **more optimistic** than reality.

The name: the **winner's curse**.

And the inflation is **proportional to the spread**: a model whose six cells are
spread out has more chances to hit a lucky one than a model whose six cells sit close
together. So when two models are nearly equal, the **more spread-out** one is being
inflated more — the true gap between them is smaller than the one you see.

---

## 6. Compare medians instead of maxima

This is the cheapest trick of the trade in this whole note.

The **maximum** (the best cell) suffers the winner's curse. The **median** (the middle
cell) does not — it ignores both the luck and the misfortune.

How to use it: rank the models **twice**, once by maximum and once by median.

| Result | Meaning |
|---|---|
| The two rankings **agree** | The conclusion is solid |
| The two rankings **swap** at some positions | At those positions the gap is **not real** — they are effectively tied |

It costs no extra computation, because you already have all six numbers.

---

## 7. σ versus the paired bootstrap — two different questions

This is the most important section, and also the easiest to get wrong.

### The problem

Model A scores 0.61, model B scores 0.60. Is A genuinely better, or is that gap just
the luck of which postings happened to be in the evaluation set?

### Method one — σ (standard deviation), **the less sensitive one**

Resample (bootstrap): from the 7,159 postings in the dev set, draw 7,159 **with
replacement**, score, and repeat 1,000 times. The standard deviation of those 1,000
scores is σ.

σ answers: *"if I had a different dev set, how much would this score move?"*

Then the crude rule: a gap smaller than σ counts as noise.

### Method two — the paired bootstrap, **far more sensitive**

Use **the same** drawn list of postings for **both** models, then take the difference
of their scores. Repeat 1,000 times and count how often A wins.

The result is `P(A > B)`:

| `P(A > B)` | How to read it |
|---|---|
| ~0.500 | **Indistinguishable** — the two models tie |
| > 0.975 | A beats B **clearly** |
| < 0.025 | B beats A clearly |
| 0.7 – 0.9 | Leaning toward A but **not enough to assert it** |

### Why the second method is more sensitive

Both models read **the same postings**, and they are wrong on **largely the same
ambiguous ones**. That shared variation is present in **both** σ values, inflating
both.

Take the difference on the same sample and the shared part **cancels**. What remains
is only where the two models genuinely differ.

A picture: measure two people's heights with a bent ruler. Measured separately, the
ruler's error makes both numbers untrustworthy. But stand the two people **side by
side** and compare, and the bend no longer matters — you still know for certain who is
taller.

> ⚠️ **Do not use σ to compare two models.** σ is the error bar of **one** model
> standing alone. The question you need is "does A beat B **on those same rows**", and
> only the paired bootstrap answers it. Using σ as a threshold is too conservative — it
> buries real differences too.

---

## 8. Write conditional conclusions

After all of the above, the sentence **"algorithm X is the best for this problem"** is
almost always an overstatement. It ignores: the search budget, the feature
representation, who laid out the grid, and which split was used to select.

| Do not write | Write instead |
|---|---|
| "SVM is the best algorithm" | "Under a budget of *n* configurations per algorithm, on representation *X*, selecting on `val`: A and B are indistinguishable (`P = ...`), and both beat group C (`P > 0.975`)" |
| "The model reaches 0.61" | "0.61 on `val`, the best of 6 configurations — this number is optimistic; the median of the 6 cells is 0.59" |
| Hiding the runs that failed | "Two configurations did not finish within 45 minutes; that belongs in the cost column" |

It sounds more cautious, but it is **stronger**. Reviewers value someone who knows the
limits of their own result — because that is the sign you understand what you measured
rather than merely having got the code to run.

---

## Quick lookup

| When you see this | Think of |
|---|---|
| Two models very close together | Look at `P(A > B)`, not σ (section 7) |
| One model's 6 cells have a very narrow spread | We never explored its space (section 3) |
| The winning configuration is at the smallest/largest value tried | A truncated grid (section 4) |
| The ranking changes when comparing medians | The gap is not real (section 6) |
| An unusually pretty number | Was it selected on the very split it is reported on? (section 5) |
| "This model has no `class_weight`" | With skewed classes that is a real disadvantage, not a detail |

---

## Back to the project itself

- [10-so-sanh-mo-hinh.md](../archive/10-so-sanh-mo-hinh.md) — the real comparison table, read with exactly these rules
- [03-protocol.md](../03-protocol.md) §7 — the project's official decision rules
- [note 6 — metrics and baselines](06-do-luong-va-baseline.md) — macro-F1 and why accuracy lies
- [note 7 — regularisation](07-chinh-quy-hoa.md) — `C`, the archetypal hyper-parameter
