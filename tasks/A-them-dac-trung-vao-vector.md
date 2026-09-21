# Proposal A — concatenate a structured feature block onto the PhoBERT vector

**This is a proposal, not a board section.** No row for it exists in
[TASKS.md](../TASKS.md), and none is created here: tasks are written and assigned by
the author (AGENTS.md §2, TASKS.md §1). Read it, cut what you disagree with, and open
a row when it is worth running. Written in English (AGENTS.md Rule 6).

---

## 1. The question this answers

The baseline reads **three text columns** (`job_title`, `description`,
`requirements_text` — `dl/text.py:21`), pooled by a frozen PhoBERT into **768
numbers** per posting. Everything else `dataset.clean` derives — skills counts,
province, experience, language flags — is loaded into the frame and then **never
reaches the model** (`train_dl.py:75-90` returns the embedding matrix verbatim).

The author's question: those columns describe the job too, so should they be in the
input? The honest answer has two halves, and only one of them is settled.

**Settled, measured on `data/processed/splits/train.csv` (34,354 rows; 24,622 with a
disclosed salary):**

| Column | Fill rate | Distinct values | Net MI/H with `category` | Net η² with `salary_mid` |
|---|---|---|---|---|
| `technical_skills_text` | 86.4 % | 17,903 | 0.064 | 0.166 |
| `qualifications_text` | 99.2 % | 12,150 | 0.075 | −0.012 |
| `province` | 100 % | 217 | **0.016** | **0.006** |

Technical skills carry real signal. **`province` carries almost none**: 80.1 % of
postings are Hà Nội or HCM, the median salary across the big provinces varies by about
±20 %, and the column is dirty — 101 values occur exactly once, and it contains
districts, wards, industrial parks and one `berlin`.

Two methodological warnings that came out of the same measurement:

- **Raw association statistics on the skill strings are inflated by cardinality.**
  Raw η² of the skill string against salary is 0.69; with the labels randomly
  permuted the same statistic still returns 0.52. The "net" column above already
  subtracts that permutation baseline. Any new number here must report one too.
- **`benefits_text` leaks the salary target** — 10.26 % of rows restate a pay figure,
  correlation 0.408 with `salary_mid`, 4.9 % an exact match. It is excluded from this
  proposal except as the count `n_benefits`. `technical_skills_text` and
  `qualifications_text` measure **0** pay figures, which is what
  `tests/test_no_leak.py` pins.

**Not settled, and the reason to run this at all:** how much these columns add *on
top of* `description` + `requirements_text`, which usually restate the same skills in
prose. Nothing in this repository measures that, and no run has ever fed the network
anything but the three text columns.

---

## 2. What is proposed

Four runs. **PhoBERT stays frozen and is not re-run at all** — every run reloads the
existing embedding cache from T1.2, so each costs seconds, not minutes.

| run_id | Task | Input width |
|---|---|---|
| `dl-cat-tab` | category | 768 ⊕ 47 = **815** |
| `dl-cat-num` | category | 768 ⊕ 9 = **777** (no `province`) |
| `dl-sal-tab` | salary | 768 ⊕ 47 = **815** |
| `dl-sal-num` | salary | 768 ⊕ 9 = **777** (no `province`) |

The `-num` pair exists because **38 of the 47 dimensions are `province`**, the column
measured above as carrying almost nothing. Without that pair, a gain could not be
attributed and a loss could not be explained.

**The block, 47 dimensions, counted on `train`:**

- `province` one-hot, **38 dims** — the 37 provinces with ≥ 20 rows (covering 98.7 %
  of postings) plus one `other` cell absorbing the remaining 180 values.
- **9 numeric columns**, `log1p` then standardised: `n_technical_skills`,
  `n_qualifications`, `n_benefits`, `n_locations`, `n_languages`, `requires_english`,
  `is_major_city`, `experience_months`, `has_working_hours`.

**Soft skills are excluded entirely**, as the author decided: used alone the column
reaches macro-F1 0.159 against a 0.160 majority floor, adding it on top of technical
skills moves 0.299 → 0.301, its vocabulary is nearly closed (20 items cover 71.6 % of
rows), and it has no `_seg` mirror — adding one would mean rebuilding the splits,
which rewrites `test.csv` and breaks Rule 5.

Everything that must be fitted — which provinces are kept, the mean and standard
deviation of each numeric column — is **fitted on `train` only** and then applied to
`dev`. Fitting on both would leak through the scaler.

---

## 3. Code changes

Three edits. `DenseHead` needs no change at all: it takes `in_dim` from
`Xtr.shape[1]` (`train_dl.py:176`), so the first Linear simply becomes 815×256.

1. **`src/vietjobs/dl/tabular.py` (new).** `fit_block(train_df) -> stats` and
   `apply_block(df, stats) -> np.ndarray[n, k]`, plus a `--features` vocabulary of
   `none` / `tabular` / `numeric`. A shared module, not inline code, because Rule 4
   requires any future inference path to rebuild exactly this layout.
2. **`train_dl.py`** — add `--features {none,tabular,numeric}`; when it is not `none`,
   `np.hstack` the block onto `X` in `load_task` (`:75-90`), **before** the
   standardisation at `:166-173` so the new dimensions are standardised in the same
   pass with the same saved statistics.
3. **Persist what was fitted.** The province vocabulary and the block statistics go
   next to `scaler.npz` (`:256`), and `metrics.json`'s `columns` key (`:270`) must
   list the block, not just the three text columns — that key is the only record of
   what a model was fitted on.

**Tests to add:** block shape is `[n, k]` and `X` becomes `768 + k` wide; statistics
are fitted on `train` and never see `dev`; a province unseen in `train` falls into
`other` instead of creating a column; and the salary task with `--features tabular`
still touches nothing in `UNMASKED_COLUMNS`.

---

## 4. How to run it

```bash
source .venv/bin/activate && pytest -q                 # 132 passed before any of this

PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --features tabular --run-id dl-cat-tab
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --features numeric --run-id dl-cat-num
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary   --features tabular --run-id dl-sal-tab
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary   --features numeric --run-id dl-sal-num
```

`dev` only; `test` is not touched (Rule 5). After each run, check `metrics.json`: `n`
matches `manifest.json` (3,812 / 2,698) and `columns` lists the new block.

---

## 5. The decision rule — fixed before looking at any number

Compare against the most recent same-configuration runs, **not** against the older
`*-s2` rows:

| Task | Baseline to beat | Source |
|---|---|---|
| `category` | macro-F1 **0.6030** (`dl-cat-ce-cpu-0920`) | `04-results.md` |
| `salary` | MAE **4.13 tr** · R²log 0.515 (`dl-sal-cpu-0920`) | `04-results.md` |

**The noise floor is already measured, and it is the strongest argument in this
sheet.** Three runs of the *same* configuration are now in the log — `dl-cat-s2`
0.6025, `dl-cat-ce-0920` 0.6013, `dl-cat-ce-cpu-0920` 0.6030 — a spread of **0.0017**
with nothing changed but the machine and the day. Any gain below roughly **0.015**
(≈ 2σ of the bootstrap, and nearly ten times that observed spread) is not a result.

If the gain is below that threshold: **delete the code** and log the negative
ablation row anyway. A measured "it did not help" is evidence, and it is what stops
the same idea being tried again in three weeks (Rule 2 item 4, Rule 1).

---

## 6. What this proposal does **not** include

- **No fine-tuning of PhoBERT.** The encoder stays frozen; only the dense head trains.
  Fine-tuning is Priority 2 of the roadmap and a different task.
- **Appending skill text into the encoder input** (editing `FIELDS` in `dl/text.py`).
  Still frozen PhoBERT, but it forces a full re-encode (~15 minutes per family) and a
  cache-naming fix at `dl/encode.py:39-40`, where the field list is **not** part of
  the cache key — a silent-stale-cache trap. It deserves its own row, with one
  measurement worth knowing first: naively concatenating skills and province onto the
  title *reduced* classification quality in the probe (0.434 → 0.388), because the
  extra tokens dilute the part that identifies the occupation. That is exactly the
  mechanism `dl/text.py` uses today, which is why the cheap version is the risky one
  and the vector-concatenation version is proposed first.

---

## 7. Honest expectation

`job_title` alone reaches macro-F1 0.434 and R²log 0.498 in a linear probe; the whole
768-dimensional vector reaches 0.6030 and 0.515. The title dominates both tasks, and
the description almost certainly restates most of what the skills columns say. The
most likely outcome of this experiment is **a gain inside the noise** — which is
still worth one afternoon, because it converts an open assumption into a logged
ablation row, and because `province` at 38 of 47 dimensions is a concrete hypothesis
that deserves to be killed with a number rather than an opinion.
