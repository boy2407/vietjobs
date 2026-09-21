---
name: mo-xe
description: "Dissect a model / algorithm / ML-DL problem with a tiny example that ACTUALLY RUNS — build a 5–6 sample corpus, compute every step, print every matrix, then line it up against the real VietJobs numbers. Use this skill when the user asks how a model, algorithm, loss, metric, or ML/DL mechanism actually works — 'X hoạt động thế nào', 'sao suy ra được Y', 'giải thích lại từ từ', 'cho ví dụ' — or invokes /mo-xe."
trigger: "Use this skill when the user asks how an ML/DL model, algorithm, loss function, metric, or mechanism works step by step, asks 'sao ra được ...', 'từ A làm sao ra B', asks to slow down or re-explain, asks for a worked example, or invokes /mo-xe."
version: 1
---

# Dissect — understand an ML/DL mechanism with an example that runs

The person asking is new to ML/DL and **learns from concrete numbers, not from
formulas**. The job: build an example small enough to print in full, actually run
it, then walk them through it step by step until the final number appears.

Different from [`giai-thich`](../giai-thich/SKILL.md): that skill answers *"what is
this, why is it needed"* with an everyday analogy. This one answers *"how does it
run"* with an example that **executes**. A concept question → `giai-thich`. A
mechanism question → `mo-xe`.

---

## Rule 1 — every number must be produced by a run, never invented

**Before writing a single word of the answer, run `python` through Bash to get the
numbers.**

This is the most important rule, with no exceptions. A table of numbers made up in
your head looks exactly like a table of measured ones, and the reader has no way to
tell. If it cannot be run, say plainly that it was not run.

Alongside that:

- If a trained model exists in `artifacts/`, **open it and use the real numbers**
  (`joblib.load`); do not retrain a fake one.
- Project numbers come from `docs/04-results.md`, `metrics.json`, `manifest.json`,
  or by reading `data/processed/splits/*.parquet` directly. Never from memory,
  never estimated.
- After running, **cross-check**: the numbers printed in the table must match what
  the terminal just returned.

## Rule 2 — the example corpus must be tiny, and it must be VietJobs

- **5–6 samples**, just enough to print the whole matrix in one code block.
- The content is **Vietnamese job postings** — title, description, skills,
  occupation. Do not use iris, titanic, MNIST, or `["the cat sat", "the dog ran"]`.
  A borrowed example forces the reader to cross two bridges.
- Few classes (2–3), few columns, few words. If the matrix does not fit on one
  screen, the corpus is still too big — cut further.
- **One corpus for the whole answer.** Switching corpora midway loses the reader.
  If an illustration genuinely needs a different one (you tried merging and it
  broke), say why the split was necessary.
- Design the corpus **backwards from what you want to show**. To expose a symptom,
  build the corpus so the symptom is forced to appear, then run it to confirm that
  it really does.

## Rule 3 — go step by step, one table per step

No skipping. The usual shape:

| Step | What to print |
|---|---|
| 0 | The example corpus — including the **label column**, because that is what gets forgotten most |
| 1..n | One table per transformation, before column → after column |
| n+1 | **The final object** — the matrix / weight vector, printed in full |
| n+2 | One new sample travelling the whole pipeline, all the way to the final number |
| n+3 | A comparison table, **example ↔ real VietJobs** |
| last | A `Reproduce` block — code that can be re-run |

That final comparison table is mandatory. Without it the reader understands the
example but does not know what it has to do with their own project.

## Rule 4 — point at the exact place people get it wrong

Every mechanism has one or two places a newcomer is almost certain to misread.
Anticipate them and put a warning block right there:

> ⚠️ L2 does **not** make the values sum to 1 — it makes their **squares** sum to 1.

A few known ones in this project: `n` is a fixed number of columns, not the number
of words in the input sentence · `df` counts **documents**, not **occurrences** ·
**spelling**, not **meaning** · the label is never a feature.

## Rule 5 — when a case comes out wrong, keep it

If the run produces a wrong result — the model misclassifies, the metric is bad, the
illustration is not what you hoped — **keep it and dissect it**. One wrong case
dissected carefully teaches more than one correct case.

Never tune the example until it comes out pretty and only then present it. That is
inventing numbers by another route.

And state the limit: an example is not a measurement. The project's measured numbers
are in `04-results.md`.

---

## Topic map

"What the example must show" is the **goal** of the corpus — design the corpus
backwards from this column.

### Data representation

| Topic | Read first | What the example must show |
|---|---|---|
| n-grams, segmentation | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` | A bigram catching a phrase that unigrams shatter |
| Semantic vectors, PhoBERT | `docs/nen-tang/09-vector-ngu-nghia.md` | Two postings with no shared word landing close together |

### Models

| Topic | Read first | What the example must show |
|---|---|---|
| The dense head | `docs/06-baseline-dl.md` | 768 → 256 → 128 → out, what each layer does to one vector |
| KNN | `docs/nen-tang/04-ma-tran-thua-va-so-chieu.md` | The nearest neighbour changing when normalisation is removed |
| Regression | `docs/07-bai-toan-luong.md` | Prediction versus label, per-sample error |
| Deep learning | `docs/06-baseline-dl.md` | One forward pass computed by hand |

### Measurement and correctness

| Topic | Read first | What the example must show |
|---|---|---|
| macro-F1, accuracy, baselines | `docs/nen-tang/06-do-luong-va-baseline.md` | Skewed classes making accuracy lie |
| Regularisation, `C` | `docs/nen-tang/07-chinh-quy-hoa.md` | How the weights change as `C` changes |
| Data leakage | `docs/nen-tang/05-ro-ri-du-lieu.md` | Fitting on the whole set before splitting → a falsely pretty score |
| Splitting, `SPLIT_SEED` | `docs/01-data-audit.md` §7 | A duplicate group straddling two splits |
| Reading a comparison table | `docs/nen-tang/08-doc-mot-bang-so-sanh.md` | A paired bootstrap reversing a verdict that σ got wrong |

For a topic the project **does not use** (dropout, attention, cross-validation…):
it can still be dissected with a purpose-built example, but **say plainly that the
project does not use it** and drop the comparison table.

---

## Avoid

- **Invented numbers.** Rule 1. Nothing destroys trust faster.
- **A formula with no numbers in it.** Writing `idf = ln((1+N)/(1+df)) + 1` and
  moving on has dissected nothing. Substitute the numbers and produce a result.
- **Textbook examples.** The corpus must be Vietnamese job postings.
- **Switching corpora midway** without explaining.
- **Dropping the VietJobs comparison table.** An example left dangling cannot be
  used once the reading is over.
- **A long answer with no tables.** If there is more prose than table, the form is
  wrong.
- **Editing `docs/` on your own initiative.** Dissecting is for answering, not for
  changing documentation. If something deserves to go into a note, **ask in one line
  at the end** and do it once agreed — and when you do, follow Rule 1 in `AGENTS.md`
  (update the table of contents, the mapping table, and `SOURCES` too).
