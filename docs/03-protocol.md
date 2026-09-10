# Experiment protocol

This document is the contract. Every number in `04-results.md` means something
only if the rules below are respected.

## 1. The frozen dataset

`data/processed/splits/{train,dev,test}.csv` is created once by
`python -m vietjobs.dataset build`, with `SPLIT_SEED = 20260826` and the
two-stage scheme `train:test = 8:2`, then `train:dev = 9:1`.

**The seed and the split fractions must not change once the first model has been
trained.** Changing them means no row in `04-results.md` is comparable with any
other, and the ablation table — the main product of the report — becomes
meaningless.

If a rebuild is truly unavoidable, check that the manifest is unchanged:

```bash
python -c "
import json; m=json.load(open('data/processed/manifest.json'))
print(m['unique_groups'], {k:v['rows'] for k,v in m['splits'].items()})"
```

### Split by group, not by row

About 28 % of the descriptions in this dataset are not unique — employers repost
many times. Splitting by row puts the same advert in both train and test; the
model scores high by memorising, and that score does not reflect generalisation.

So: hash `(title + description + requirements)` after accent folding → `group_id`,
and **the whole group** goes into one split. `dataset.build()` has an assert that
no group sits in more than one split; that assert may never be disabled.

### Stratify by occupation

The three smallest classes hold only 196–258 rows. Without stratification they
could be entirely absent from dev or test, and macro-F1 would be computed over a
different number of classes from run to run.

## 2. Test-touching policy

| Split | Used for | How often it may be touched |
|---|---|---|
| `train` | Training | Unlimited |
| `dev` | Model selection, hyper-parameters, preprocessing choices | Unlimited |
| `test` | Reporting the final number | **Exactly once**, at the end |

The development set is called **`dev`** on disk as well, since 2026-09-09. The
CLI flags of the older entry points (`--eval val`) have **not** been renamed yet —
that is a loose end, tracked in [09-lo-trinh.md](09-lo-trinh.md).

**`test` is not merely rarely touched — it is never transformed.** No filtering,
no de-duplication, no class balancing, no outlier trimming, no re-splitting. Every
cleaning or data adjustment applies to `train` and `dev` only. Once a test set has
been edited, the final number no longer says anything about real data.

`train.py` refuses to run `--eval test` without the `--confirm-test` flag. That
flag exists so that touching test is a deliberate act, not a default.

Every time you look at test and then go back and adjust the model, information
leaks from test into a design decision. Do that a few times and test is no longer
an unbiased estimate.

## 3. The log is append-only

`docs/04-results.md` is an **append-only** log. Each `train.py` run writes exactly
one row: run id, timestamp, task, model, scope, preprocessing configuration,
evaluation split, sample count, metrics, training time, git sha.

**Never edit a row already written.** A failed experiment is still data — it
records what was tried and did not work, and that is precisely what stops us
trying it again three weeks later.

Every row must be reproducible from `(run_id, seed, git sha)`. If `git_dirty` is
true, that row is not reproducible and must be treated as indicative only.

## 3b. What deep learning logs on top

A machine-learning run ends in a single number. A deep-learning run spans many
epochs, can fail halfway, and **the reason it failed is in the learning curve, not
in the final number**. So `dl/train_dl.py` writes extra files in
`artifacts/<run_id>/`:

| File | What it holds | Why it is mandatory |
|---|---|---|
| `history.jsonl` | one JSON line per epoch: loss, `val` metric, seconds | Without it, underfitting / overfitting / a wrong lr cannot be told apart |
| `config.json` | every hyper-parameter + the seed | A run that cannot be reproduced has no value as a number |
| `env.json` | device, torch/transformers versions, encoder | Two runs on two machines have incomparable timings |
| `best.pt` | the weights at the best `val` epoch | A crash in hour two would otherwise lose everything |
| `predictions_<eval>.parquet` | per-row predictions | Reading real errors after the run has ended |

**`04-results.md` still takes one row per run.** Stuffing every epoch into it
destroys the readability of the log itself.

PhoBERT vectors are cached in `artifacts/embeddings/`, in two files split by
column family: `raw` for classification, `masked` for the two salary tasks. Mixing
the two files is a silent salary leak — the very thing Rule 3 guards.

## 4. Metrics

### Occupation classification

**Headline: macro-F1.** The largest class is 27× the smallest. Accuracy would
happily hide a model that ignores the tail entirely.

Reported alongside:
- `f1_macro_no_junk` — macro-F1 over 15 classes, dropping `nhóm_nghề_khác`. That
  class is a junk drawer holding "Nhân Viên Seo Web", "Nhân Viên Quản Trị Website"
  — postings that belong in marketing and IT. That is **label noise, not model
  error**; reporting both numbers keeps the two things apart.
- `balanced_accuracy`, `accuracy`, `top3_accuracy`.
- The confusion matrix and the 10 most-confused class pairs.

### Salary regression

Every number is **reported in million VND/month**. An MAE in log space is not a
number anyone can act on.

## 5. Two experiment axes

Three algorithms × six preprocessing steps is 18 runs, and KNN on full text takes
10–30 minutes each. Not feasible. So:

**Axis 1 — the preprocessing ladder.** Fix *one* cheap, strong algorithm (`svm`)
as the yardstick and toggle the Vietnamese-processing steps one at a time. Settle
on the best configuration, call it `PREP*`.

**Axis 2 — algorithm comparison.** All three run on exactly `PREP*`, the same
features, the same seed.

Mix the two axes and you cannot tell whether an improvement came from the
preprocessing or from the algorithm.

## 7. Model comparison — the decision rules

Added after the fair sweep cluster
([10-so-sanh-mo-hinh.md](archive/10-so-sanh-mo-hinh.md)). Three constraints bind
every model comparison from here on:

**a. The same number of configurations for every model.** Comparing "the best of
nine attempts" against "the first attempt" compares hand-tuning effort, not
algorithms. If one model gets `n` configurations swept, every model in the same
table gets `n`.

**b. Every run saves `predictions.npz`.** `train.py` writes `row_index`, `y_true`
and `y_pred` on the evaluation split. Without it, that run **cannot be paired**
with any other — which is why the 33 runs before the cluster had to be re-run from
scratch.

**c. Decide with a paired bootstrap, not with σ.** Use **one** shared index matrix
(`evaluate.bootstrap_indices`) for every model being compared, then read
`P(A > B)` from `evaluate.paired_delta`. Threshold: **>0.975 or <0.025** is a clear
difference; around 0.5 is indistinguishable.

The per-model σ (≈ 0.0077) is still reported, but it is **not the basis for a
decision**: it answers "how much does this score move if the dev set changes",
whereas the question that matters is "does A beat B on those same rows". Two
models are wrong on largely the same ambiguous postings, so the paired comparison
is far more sensitive.

**Select models on `dev`, never on `test`** — not even to "confirm" the result of
a sweep. §2 has no exceptions.

---

## 8. The removal rule

Every preprocessing step must have an ablation row proving it contributes.
**A step that does not improve anything gets removed**, not kept because it "looks
right" or because "the textbook says so". A short pipeline where every step has
evidence beats a long pipeline full of ritual.
