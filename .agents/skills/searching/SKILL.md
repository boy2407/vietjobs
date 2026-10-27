---
name: searching
description: "Find a suitable algorithm or method for a VietJobs task — read what the project has already measured, fill in the real constraint table, research outside with sources cited, then return a ranked candidate table where every row reduces to exactly one runnable command. Use this skill when the user asks which model, algorithm or technique to use or try next — 'nên dùng mô hình nào', 'cải thiện phân lớp bằng gì', 'có nên thử học sâu không' — or invokes /searching."
trigger: "Use this skill when the user asks which algorithm, model family, loss, or technique to try for a VietJobs task, asks whether an approach is worth trying, asks to research a method or read up on the literature, or invokes /searching."
version: 1
---

# Searching — pick an algorithm by constraints, not by reputation

The question "which model should we use" always has an answer that sounds good and
is wrong: whatever is currently fashionable. This skill forces the answer through
**the project's real constraints** — how many samples, how many dimensions, which
machine, what has already been measured — before any algorithm is named.

How it differs from the other skills: *what is this concept* →
[`giai-thich`](../giai-thich/SKILL.md). *How does it run* →
[`mo-xe`](../mo-xe/SKILL.md). *Where is the data weak* →
[`dataset-diagnosis`](../dataset-diagnosis/SKILL.md). *Where is the model wrong* →
[`model-diagnosis`](../model-diagnosis/SKILL.md). This skill answers only **what to
try next**.

This skill **proposes**; it does not run. `dl/train_dl.py` runs.

---

## Rule 1 — read what has been measured before researching anything new

This project already swept **6 models × 6 configurations** fairly. Re-proposing
something already measured without citing the old number is this skill's worst
failure mode.

Mandatory `Read` before writing a single word:

| File | What to take from it |
|---|---|
| `docs/04-results.md` | Every configuration actually run, with date and commit |
| `docs/06-baseline-dl.md` | The two baseline runs, the floors, and what each decision was measured against |
| `docs/09-lo-trinh.md` | The work items already ranked, and why they are ranked that way |
| `docs/03-protocol.md` | The decision rules: the paired-bootstrap threshold, the removal rule |

If the question is about the salary task, also read `docs/07-bai-toan-luong.md`; if
it is about how the text reaches the model, `docs/02-vietnamese-nlp.md`.

Answering from memory here produces a *plausible-sounding* proposal for an
experiment that **has already been run and already lost**. Nobody catches it, not
even the person asking.

## Rule 2 — fill in the constraint table before opening a browser

These six cells decide most of the answer. Read the numbers from the files, do not
recall them:

| Cell | Where to read it |
|---|---|
| Training samples, number of groups | `data/processed/manifest.json` |
| Number of classes and the class skew | `manifest.json` + the `metrics.per_class` key of `metrics.json` |
| Input dimensionality | `docs/06-baseline-dl.md` — 768 dense dimensions per posting |
| Current fit budget | `fit_seconds` in `artifacts/<run>/metrics.json` |
| Hardware | `environment` in that same file — **no GPU**, so every fine-tuning proposal must state its cost |
| The score to beat | The best row in `docs/04-results.md`, with `f1_macro_boot_std` |

A candidate that **cannot fill three cells** — *estimated fit cost · does it run on
this machine · what extra labels or data it needs* — is eliminated on the spot,
however famous it is.

## Rule 3 — disciplined outside research

`WebSearch` and `WebFetch` are allowed, but in order: **inside first, outside
second**. Only research outside when the question goes beyond what `docs/` has
measured.

- Prefer sources **with measurements**: papers with benchmarks, library
  documentation, results on Vietnamese datasets. A blog with no numbers is only good
  for finding a method's name.
- Every borrowed claim comes **with a source and a year**. Without a source, label
  it *general experience*; do not dress it up as measured.
- **A number from another dataset is not a prediction for this one.** State where it
  was measured, on how many classes, on how many samples. A model reaching 0.9 on
  clean-labelled English job postings says nothing about 16 classes skewed 27:1 here.
- A new library must be checkable: does it run on CPU, does it pull heavy
  dependencies, is it still maintained.

## Rule 4 — every candidate reduces to one command

A proposal is only worth something if the reader knows what to type next. Every row
of the candidate table ends in **one** of two things:

- A command that runs right now, e.g.
  `PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category`,
  with the flags that differ from the default.
- Or an explicit statement of **what has to be written**: which columns
  `dl/text.py` would have to join (and that changing them means re-encoding);
  which architecture to add to `src/vietjobs/dl/heads.py`. With an estimate of the
  lines of code.

Note one real limitation: a flag reaches the head only. Changing what goes **into**
the vector means editing `dl/text.py` and re-running `dl/encode.py` — say so plainly
instead of offering a command that does not exist.

## Rule 5 — an expectation is not a result

A number that has not been run must be marked **not measured**, kept inside a block
quote, and **must not** be written into `docs/04-results.md`. The phrasing template
is already in `docs/09-lo-trinh.md` — match that tone.

And always report **both axes**: the score *and* the cost. A model that gains 0.05
macro-F1 while being 200× slower is a trade-off, not a victory.

---

## Answer template

```markdown
## <The question> — <n> candidates

**The tightest constraint:** <one sentence, with the number you just read>

| Approach | One-line idea | Fits this problem because | Cost | Risk | First command |
|---|---|---|---|---|---|

**Do now:** <one item, with the command>
**Park it:** <one item, with the condition that would make it worth doing>
**Drop it:** <one item, with the number that rules it out>
```

The "Cost" column takes four values only: `<5 min` · `~1 machine-hour` ·
`~1 person-session` · `rebuild the splits`. The last one is a red flag — it
invalidates every row in `04-results.md`.

The **Drop it** block is mandatory. Ruling a direction out with a measurement is
worth as much as picking one.

## Question map

| Type | Also accepts | Read first | The constraint that usually binds |
|---|---|---|---|
| `pick-a-model` | `chon-mo-hinh`, `model`, `architecture` | `docs/06-baseline-dl.md` | The dense head beats its linear probe by only 0.0127 — capacity is not the binding constraint |
| `imbalance` | `mat-can-bang`, `small-classes`, `class-weight` | `docs/05-phan-tich-du-lieu.md` §2 | Skew 27.1:1; `--class-weight` measured as a trade, not a gain |
| `more-input-columns` | `them-cot`, `them-dac-trung`, `skills`, `province` | `docs/05-phan-tich-du-lieu.md` | Adding a column means editing `dl/text.py` and re-encoding, or concatenating a block onto the vector |
| `deep-learning` | `hoc-sau`, `phobert`, `bert`, `transformer` | `docs/06-baseline-dl.md` + `docs/09-lo-trinh.md` Priority 2 | No GPU; inference latency must be reported alongside the score |
| `calibration` | `hieu-chuan`, `probabilities`, `top3` | `docs/06-baseline-dl.md` | `top3_accuracy` is already logged; softmax scores are not calibrated probabilities |
| `salary-regression` | `hoi-quy-luong`, `salary`, `quantile` | `docs/07-bai-toan-luong.md` | Reads `*_masked` columns only; selection bias |
| `label-noise` | `nhieu-nhan`, `ceiling`, `relabel` | `docs/viec-du-lieu/00-index.md` D1 | The real ceiling is unmeasured — measure it before switching models |
| `multi-task` | `da-nhiem`, `shared-trunk` | `docs/09-lo-trinh.md` Priority 3 | The sector explains 3.2 % of log-salary variance — expect little |

A question matching no row: still answer under all five rules, but say plainly that
the project has not touched that direction and **leave the "Read first" column
empty** rather than inventing a link.

## Avoid

- **A list of ten unranked algorithms.** That is dodging the responsibility to
  choose. Rank them, and state the criterion you ranked by.
- **Re-proposing something already measured without citing the old number.** Rule 1.
  Check `04-results.md` first.
- **Summarising a paper without reducing it to a command.** A method summary helps
  nobody type a line.
- **Mixing expected numbers with measured ones** in one table without marking them.
- **Ignoring cost.** A score with no machine-hours attached is half an answer.
- **Running the experiment yourself.** This skill stops at the proposal. If the user
  wants it run, they type the command — and once there is a real result, `docs/`
  must be updated under Rule 1 in `AGENTS.md`.
- **Editing `docs/` on your own initiative.** If something deserves to go into a
  note, ask in one line at the end.
