---
name: dataset-diagnosis
description: "Examine the VietJobs dataset with ten measurements run for real on the frozen splits — class skew, repost groups with conflicting labels, the label-noise ceiling, empty fields, vocabulary coverage, class separability, the junk class, salary-masking holes, selection bias, geographic skew — then rank them into a findings table with the cleaning action each implies. Use this skill when the user asks what is wrong with the data, why the ceiling is low, whether the labels are trustworthy, what to clean or relabel, or invokes /dataset-diagnosis."
trigger: "Use this skill when the user asks about dataset quality, label noise, class imbalance, duplicates, missing fields, vocabulary coverage, selection bias, or what data cleaning to do next, or invokes /dataset-diagnosis."
version: 1
---

# Dataset diagnosis — turning the data's weaknesses into work items

A model stalling at some score has two entirely different explanations: the model is
not good enough, or the data does not allow more. This skill answers the second with
measurements, so nobody spends a session tuning against a ceiling the data has
already fixed.

Different from [`model-diagnosis`](../model-diagnosis/SKILL.md): that skill examines
**one trained run**. This one examines **the data**, and runs even when no model
exists yet. A few conclusions have to be cross-read between the two skills — this
note says where.

---

## Rule 1 — no number is written down before the command that produced it has run

Every cell in the results table has the block that produced it in the next column.
If it cannot be run, write **not measured**. No "about 30 %", no "most of them".

Every snippet below runs verbatim from the repo root, through Bash, after this
shared preamble:

```python
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from vietjobs import config as C, vitext as V
tr = pd.read_parquet('data/processed/splits/train.parquet')
dv = pd.read_parquet('data/processed/splits/dev.parquet')
```

## Rule 2 — diagnose on `train` only

`dev` is used only when the question **is** about generalisation (A3, A5). `test`
never, with exactly one exception: counting regex frequencies to close a leak (A8),
because that count does not look at labels and does not lead to any model choice.

**Never rebuild the splits.** `SPLIT_SEED = 20260826` is locked and segmentation
costs roughly 26 minutes. Every cleaning proposal is a proposal for **some future
rebuild**, and must carry the warning line: doing it invalidates all of
`docs/04-results.md`.

## Rule 3 — a finding has three parts

The measurement · the reproduction block · the action with its cost. Without the
third part it is a statistic, not a finding — put it under "checked and clean".

And **one measurement is not enough to rank something highly**. Only raise a finding
to do-this-now when two independent indicators point at the same place — for
instance A6 giving a high cosine *and* the best run's `metrics.top_confusions`
naming exactly that class pair.

---

## The ten diagnostic blocks

Run them in order. A8 is the cheapest and unblocks the most — run it first if the
salary task is what is stuck.

### A1 · Class skew by **group**, not by row

```python
g = tr.groupby("category").agg(rows=("group_id","size"), groups=("group_id","nunique"))
g["dup_x"] = (g.rows / g.groups).round(2)
print(g.sort_values("rows").to_string(), g.rows.max()/g.rows.min(), g.groups.max()/g.groups.min())
```

Repost rows are not independent samples. Threshold: `groups < 300` for a class →
that class has too few samples for a stable F1, and its σ dominates macro-F1. A
`dup_x` differing by more than 1.15 across classes → `class_weight="balanced"` is
balancing by **row**, i.e. balancing wrongly.

### A2 · Repost groups with conflicting labels — the most important block

```python
sz = tr.groupby("group_id").size(); nc = tr.groupby("group_id")["category"].nunique()
multi = nc[nc > 1].index
print((sz > 1).sum(), (nc > 1).sum(), tr.group_id.isin(multi).mean())
print(tr[tr.group_id == multi[0]][["job_title","category"]].to_string())
```

`group_id` is the hash of title + description + requirements after accent folding.
The same text carrying several labels means the problem is **really multi-label**
and has been forced into single-label.

Threshold: over 5 % of multi-row groups being multi-label → not random noise. Over
50 % of rows sitting in a multi-label group → single-label macro-F1 **is measuring
the wrong problem**; the cheap action is to also report *top-1-in-set accuracy*, the
expensive one is to move the target to multi-label — and that must be a **second
axis**, not a replacement for the first.

### A3 · The label-noise ceiling, computable in three seconds

```python
maj = dv.groupby("group_id")["category"].agg(lambda s: s.mode().iat[0])
oracle = dv["group_id"].map(maj)
print((oracle == dv.category).mean(),
      f1_score(dv.category, oracle, average="macro", zero_division=0))
```

This is the upper bound of **any** deterministic text → label function, computed as
"predict the majority label within the group of identical text". Threshold: a ceiling
below 0.95 → record this number next to every macro-F1. Use it to **stop** the chase
for a few percent with a bigger model, not to congratulate yourself. Run this block
**before** spending two hours hand-labelling.

### A4 · Empty fields and length per column

```python
for c in ["job_title","description","requirements_text","qualifications_text",
          "technical_skills_text","soft_skills_text","benefits_text","languages_text"]:
    L = tr[c].fillna("").astype(str).str.len()
    print(c, f"{(L==0).mean():.2%}", L.quantile(.1), L.median(), L.quantile(.9))
```

Threshold: over 60 % empty → that column carries almost nothing; **do not feed it in
without measuring**, following the removal rule in `docs/03-protocol.md`.
10–60 % empty → the `n_*` count column is conflating "missing" with "present but
zero"; add a `has_*` flag.

### A5 · Vocabulary coverage, train → dev

```python
cv = CountVectorizer(min_df=3).fit(tr.job_title + " " + tr.description)
an, vocab = cv.build_analyzer(), set(cv.vocabulary_)
r = [(sum(w not in vocab for w in an(t)), len(an(t)))
     for t in (dv.job_title + " " + dv.description)]
print(sum(a for a, _ in r) / sum(b for _, b in r))
```

Threshold: over 5 % → the vocabulary is the bottleneck, and `min_df=1`, character
n-grams or embeddings are worth trying. **Under 1 % → OOV is not the cause**, and the
finding here is an **exclusion**: record it so nobody proposes PhoBERT with the
reason "Vietnamese OOV".

### A6 · Class separability — cosine between class centroids

```python
X = np.load('artifacts/embeddings/train-raw-len256.npy')   # the vectors the model actually reads
labs = sorted(tr.category.unique())
Cm = normalize(np.vstack([np.asarray(X[(tr.category == l).values].mean(axis=0)) for l in labs]))
S = Cm @ Cm.T; np.fill_diagonal(S, 0)
print(sorted(((S[i,j], labs[i], labs[j]) for i in range(len(labs))
              for j in range(i+1, len(labs))), reverse=True)[:8])
```

Threshold: cosine above 0.80 → the two classes are **not separable from the title**.
Cross-check against the best run's `metrics.top_confusions` is mandatory; if the same
pair appears, this is a taxonomy error, not a model error → propose merging the
classes or recording the ceiling. 0.65–0.80 → a candidate for a dedicated
discriminating feature. Below 0.50 → the classes separate well, and a low F1 here is
a model error, not a data error.

### A7 · What the junk class is made of

```python
j = tr[tr.category == C.JUNK_CATEGORY]; print(len(j), j.job_title.head(20).tolist())
```

Threshold: this class's F1 (read `metrics.per_class` in
`artifacts/<run>/metrics.json`) below 0.25 **and** over 30 % of the sampled titles
being hand-assignable to another class → **it is a wrong label, not a class**. In
that case `f1_macro_no_junk` — already present in every `metrics.json` — is the number
to lead with.

### A8 · Salary-masking holes

```python
al = pd.concat([pd.read_parquet(f'data/processed/splits/{s}.parquet')
                for s in ("train", "dev", "test")])
for c in ["soft_skills_text","qualifications_text","technical_skills_text","languages_text"]:
    hit = al[c].fillna("").map(lambda t: bool(V._PAT_MILLIONS.search(t)
                               or V._PAT_DONG.search(t) or V._PAT_USD.search(t)))
    print(c, int(hit.sum()), f"{hit.mean():.4%}")
```

The thresholds are already written into the roadmap: **zero → close it**, just add
the two columns to `UNMASKED_COLUMNS` (`src/vietjobs/features.py`) for safety.
**Non-zero → a real leak**: a `_masked` copy has to be built, added to `_MASKABLE`,
and every salary number produced before that has to be discarded. This block unblocks
the whole salary task — run it first.

### A9 · Salary distribution and selection bias

```python
d = tr.groupby("category").salary_disclosed.agg(["mean","size"]).sort_values("mean")
s = tr.loc[tr.salary_disclosed == 1, "salary_mid"]
print(d.to_string(), d["mean"].max() - d["mean"].min(), s.median(), s.quantile(.9), s.max())
```

Threshold: a disclosure-rate gap between classes above 0.10 → **real selection
bias**, which a better model cannot fix. This table must go into
`docs/07-bai-toan-luong.md` **before** any salary-task run, and every MAE must be
reported with `evaluate.slice_report` by `category`.

### A10 · Geographic skew

```python
p = tr.province.value_counts(); print(p.size, p.head(8).to_dict(), tr.is_major_city.mean())
```

Threshold: more than 63 distinct `province` values → `normalize_province` is not
collapsing everything; list the 20 rarest values and fix the table in `resources/`.
Cheap, and the `--province` ablation table is currently measured on an unclean
mapping. The top 2 provinces above 75 % → the `province` one-hot is nearly a binary
flag; record that as a **scope limitation**, not a new finding.

---

## Result template

```markdown
## Dataset diagnosis — <date> · <n> train rows · manifest <sha8>

| # | Finding | Measurement | Block | Action it implies | Cost | What it blocks |
|---|---|---|---|---|---|---|

**The three worth doing first:** ...
**Checked and clean:** <hypothesis + the number that refutes it>
**Waiting on a split rebuild:** <a separate block, not mixed into the main table>
```

Rank in exactly this order: (1) findings that **block** a work item in
`docs/09-lo-trinh.md` go first, whatever their size — including when the measurement
is zero; (2) then estimated effect divided by cost; (3) at the bottom, kept separate,
the **ruled out** section.

The "Cost" column takes four values only: `<5 min` · `~1 machine-hour` ·
`~1 person-session` · `rebuild the splits`.

The **ruled out** section is mandatory. Refuting a hypothesis with a measurement is a
result, not an empty space — exactly the "record the failures too" spirit of
`AGENTS.md`.

## Avoid

- **Rebuilding the splits.** Rule 2. Do not run `python -m vietjobs.dataset build`.
- **Touching `test.parquet`** outside the single A8 exception.
- **Measuring on all the data and concluding for `train`.** Always name the split.
- **Reporting a percentage without its denominator.** "88 %" of 8,659 is nothing like
  "88 %" of 12.
- **Proposing to delete rows without two numbers:** how many rows are deleted, and
  what percentage of the smallest class goes with them.
- **Concluding from a single measurement.** Rule 3.
- **Writing into `docs/` on your own initiative.** Print markdown to stdout, the way
  `scripts/measure_vitext.py` does. If something
  deserves to go into a note, ask in one line at the end, and when you do it, follow
  Rule 1 in `AGENTS.md`.
