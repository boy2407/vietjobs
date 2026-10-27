---
name: model-diagnosis
description: "Examine one trained VietJobs run on an eight-rung ladder — per-class F1 versus support, confusion matrix, reading real errors, the train↔dev gap, learning curves, weight per feature block, hyper-parameter sensitivity, calibration — then pick exactly one of four levers: change model family, tune hyper-parameters, fix features, clean labels. Use this skill when the user asks why the model is wrong, where it fails, whether to tune or switch models, or what to do next about a run — or invokes /model-diagnosis."
trigger: "Use this skill when the user asks why a model underperforms, which classes it confuses, whether it overfits or underfits, whether more data or tuning would help, how to read a confusion matrix or per-class report from an existing run, or invokes /model-diagnosis."
version: 1
---

# Model diagnosis — where it is wrong, why, and which lever to pull

A low score does not tell you what to do. This skill goes from a `run_id` to
**exactly one lever** out of four: change the model family · tune hyper-parameters ·
fix the features · clean the labels. The other three must be **ruled out by
measurement**, not by feel.

Different from [`dataset-diagnosis`](../dataset-diagnosis/SKILL.md): that skill
examines the data and runs before any model exists. This one needs a run already
saved in `artifacts/`. Rungs B2 and B3 must cross-read blocks A6 and A3 of that
skill.

---

## Rule 1 — the numbers are already on disk; read before re-running

`artifacts/<run_id>/metrics.json` already contains `per_class`, the 16×16
`confusion_matrix`, `top_confusions`, `f1_macro_boot_std` and `f1_macro_boot_ci95`
for **every** run. Retraining to obtain something already on disk wastes time and
skews the comparison.

Careful: runs older than the sweep cluster (`cat-T-*`, `cat-P*`, `cat-TREE-*`,
`cat-A-*`, `cat-FINAL-*`) **do not have** the `model_label` / `overrides` /
`estimator_params` keys. Use `.get()`, not `[...]`.

## Rule 2 — the noise threshold, read from that same run

The bootstrap σ is in `metrics.f1_macro_boot_std` (1,000 resamples). On the
occupation task it is about **±0.007**, so **a gap under ~0.014 (2σ) is not a
finding** — do not write it into a note, do not base a decision on it. Always read
the real number from the run being examined rather than recalling this one.

To decide between two runs use a **paired bootstrap**, not separate σ values: one
shared matrix from
`evaluate.bootstrap_indices(n, n_boot=1000, seed=C.RANDOM_SEED)`, then
`evaluate.paired_delta(a, b)["p_gt_0"]`. The threshold, per `docs/03-protocol.md`:
above 0.975 or below 0.025 is a clear difference; around 0.5 is a tie.

## Rule 3 — no skipping rungs, and no touching test

Climb B0 → B8 in order. Rungs B4 + B5 + B6 together cost about ten machine-minutes;
one fine-tune costs an hour or two. **Changing the architecture is always the last
lever.**

No run has been scored on `test` yet — that single touch is still unspent. Do not
run `--eval test`, do not read the test split for diagnosis, do not "re-confirm" a
conclusion on test.

---

## The diagnostic ladder

The shared preamble for every block:

```python
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
RUN = 'dl-cat-s2'
m = json.load(open(f'artifacts/{RUN}/metrics.json'))
dv = pd.read_parquet('data/processed/splits/dev.parquet')
```

### B0 · Read `metrics.json` — ten seconds, not skippable

```python
print({k: m.get(k) for k in ('run_id','model_label','scope','prep','eval_split',
                             'n_train','n_eval','fit_seconds')})
mm = m['metrics']
print({k: v for k, v in mm.items()
       if k not in ('per_class','confusion_matrix','labels','top_confusions')})
```

Write down `f1_macro`, `f1_macro_no_junk`, `balanced_accuracy` and
`f1_macro_boot_std` immediately. The gap between `f1_macro` and `f1_macro_no_junk`
is how much the junk class drags the score down — if it exceeds 2σ that is already a
finding, and it belongs to the **label** lever.

### B1 · Per-class F1 versus support — skew or genuine difficulty

```python
rows = [(k, v["f1-score"], v["support"]) for k, v in mm["per_class"].items()
        if k not in ("accuracy", "macro avg", "weighted avg")]
df = pd.DataFrame(rows, columns=["class","f1","support"]).sort_values("f1")
print(df.to_string(index=False), df.f1.corr(df.support, method="spearman"))
```

| ρ (Spearman) | Conclusion | Lever |
|---|---|---|
| above 0.7 | Imbalance dominates | hyper-parameters: `class_weight`, resampling |
| 0.3 – 0.7 | Mixed | read B2 before doing anything |
| below 0.3 | **Not imbalance — class difficulty.** `class_weight="balanced"` has already done its job | **features** or **labels** |

### B2 · Confusion matrix — which pairs stick, and are they genuinely different

```python
for c in mm["top_confusions"]:
    print(c["n"], c["true"], "->", c["predicted"])
```

The mandatory next step: cross-check against block **A6** of `dataset-diagnosis`
(cosine between class centroids). A finding is only solid when two independent
measurements point at the same class pair.

| Sign | Read as | Lever |
|---|---|---|
| **Symmetric** confusion (a→b ≈ b→a) with cosine above 0.80 | The two classes are not separable from text | **labels** — merge the classes or record the ceiling. Definitely not the model |
| Large **one-way** confusion | The model leans toward the target class | **hyper-parameters** — class weights |
| A large pair with cosine below 0.60 | The classes are separable but the model does not separate them | **features** |

### B3 · Read N real errors — the label-noise ceiling

`row_index` is a **0-based position in the split file**, not an index label. Join
with `.iloc`, then assert — a wrong join still runs and produces an entirely
fabricated error table.

```python
d = np.load(f'artifacts/{RUN}/predictions.npz', allow_pickle=True)
w = dv.iloc[d["row_index"]].copy(); w["y_true"], w["y_pred"] = d["y_true"], d["y_pred"]
assert (w["category"].values == w["y_true"]).all(), "bad join — stop"
bad = w[w.y_true != w.y_pred]
for r in bad.sample(150, random_state=42).itertuples():
    print(f"\n{r.job_title}\n  true: {r.y_true}\n  pred: {r.y_pred}\n  {r.description[:300]}")
```

Assign each posting by hand to exactly **one** of three buckets — `model error` ·
`wrong original label` · `genuinely ambiguous (multi-label)` — then recompute the
ceiling: `ceiling ≈ f1_macro / (1 − wrong_label_fraction)`.

Before spending two person-hours: **run block A3 of `dataset-diagnosis`**. It gives
the ceiling in three seconds. If the ceiling is already low, hand-labelling is only
for finer resolution, not for discovery.

### B4 · The train ↔ dev gap — bias or variance

`dl/train_dl.py --eval` only accepts `dev` and `test`, and **do not** add a `train`
option — it would write a meaningless row into the append-only log. Read the gap
from `history.jsonl`, which already records both numbers per epoch:

```python
h = [json.loads(l) for l in open(f'artifacts/{RUN}/history.jsonl')]
print(pd.DataFrame(h)[["epoch", "train_loss", "dev_f1_macro"]].to_string())
```

| train − dev gap | Conclusion | Lever |
|---|---|---|
| under 2σ | Not a finding | — |
| 2σ – 0.10 | **Bias dominates.** The model cannot even fit the training set; more regularisation only makes it worse | **features** or **model family** |
| above 0.15 | Variance dominates | **hyper-parameters** — lower `C`, raise `min_df` |
| train ≈ 1.0 | Memorising | hyper-parameters, urgently |

This block refutes a very common false inference: a very small optimal `C` does
**not** prove overfitting. Only the gap tells you.

### B5 · Learning curve over the training fraction — will more data pay

Sample **by `group_id`**, not by row. Sampling by row puts reposts both inside and
outside the subsample, and flattens the learning curve artificially.

```python
from sklearn.linear_model import LogisticRegression     # the linear probe, not the network
Xtr = np.load('artifacts/embeddings/train-raw-len256.npy')
Xdv = np.load('artifacts/embeddings/dev-raw-len256.npy')
gids = tr["group_id"].unique().copy(); np.random.default_rng(42).shuffle(gids)
for frac in (0.1, 0.25, 0.5, 1.0):                     # shuffle ONCE -> nested subsamples
    keep = tr["group_id"].isin(set(gids[:int(len(gids)*frac)])).values
    clf = LogisticRegression(max_iter=2000).fit(Xtr[keep], tr["category"][keep])
    print(frac, keep.sum(), f1_score(dv["category"], clf.predict(Xdv),
                                     average="macro", zero_division=0))
```

The probe stands in for the network here on purpose: it reads the same vectors, has
no learning rate to get wrong, and four points cost minutes instead of an hour. The
**shape** of the curve is what is being read, not its height.

Four points cost about three to four minutes, write no artifact and touch no log.
Read the gap between `frac=0.5` and `frac=1.0`: under 2σ → the curve has saturated,
**more data will not pay**, move to features or labels. Above 0.03 → still climbing,
and collecting more data is the cheapest lever — with the warning that rebuilding
the splits invalidates `04-results.md`.

### B6 · Which input columns the run actually read

```python
print(m["columns"])          # exactly what dl/text.py joined for this run
```

| Condition | Action |
|---|---|
| A salary run lists a column without `_masked` | **Stop** — that is a leak (Rule 3). Re-run after fixing `dl/text.py` |
| The list differs from another run being compared | The two runs are not comparable; re-encode before comparing |
| A column is mostly empty in the split (cross-check A4) | Its contribution cannot be large; do not spend a lever on it |

### B7 · Hyper-parameter sensitivity — read the runs already on disk

```python
import glob
rows = [(json.load(open(f))["run_id"], json.load(open(f))["metrics"]["f1_macro"])
        for f in sorted(glob.glob('artifacts/dl-cat-*/metrics.json'))]
print(pd.DataFrame(rows, columns=["run", "f1"]).round(4).to_string())
```

Pair each row with its `config.json`: the deviation from the default is the whole
explanation of the gap. **A spread under 2σ across several configurations means
hyper-parameters are exhausted for that architecture** — sweeping further burns
machine-hours without changing the conclusion.

`history.jsonl` matters more than the final number here: a run whose dev metric
peaked early and then decayed is over-trained, and no further sweeping fixes that.

### B8 · Calibration and top-3

The classification head emits logits over 16 classes, so probabilities come from a
softmax and `top3_accuracy` is already in `metrics.json`. Read it before computing
anything.

```python
pred = pd.read_parquet(f'artifacts/{RUN}/predictions_dev.parquet')
print(pred.columns.tolist())      # check what was actually saved, do not assume
```

A large gap between top-1 and top-3 accuracy means the model **orders** the classes
well while failing to separate the top pair — that is a label-boundary problem
(B2, B3), not a capacity problem. Do not pull the architecture lever for it.

---

## Decision table — symptom → lever → next step

| Symptom (rung) | Lever | Concrete action |
|---|---|---|
| B1 ρ below 0.3 | **rule out hyper-parameters** | Drop the idea of adding `class_weight` / resampling. Record it as ruled out |
| B2 symmetric pair + high A6 cosine | **labels** | Merge the classes into a secondary axis, or record the ceiling. Do not run a new model |
| B3 over 30 % of errors are "wrong original label" | **labels** | Record the corrected ceiling in the matching `docs/` note; stop comparing models around the current level |
| B3 over 40 % of errors are "genuinely ambiguous" | **problem definition** | Move the metric to top-3 or multi-label — a new axis, not a replacement for the old one |
| B4 gap between 2σ and 0.10 | **features** | The representation is the ceiling — change what goes into the vector, then re-encode |
| B4 gap above 0.15 | **hyper-parameters** | A smaller `--C`, or a larger `min_df` |
| B5 gap from frac 0.5→1.0 under 2σ | **rule out data** | Do not collect more. Move to features or labels |
| B6 one block's `per_dim_x` is several times higher | **features** | Weight that block ×2 then ×3, two runs, paired-bootstrap against the original |
| B6 the run read a column it should not have | **stop** | Fix `dl/text.py` and re-encode; the number is not usable until then |
| B7 spread across six configurations under 2σ | **exhausted** | Stop sweeping hyper-parameters |
| B7 `p_gt_0` around 0.5 between two models | **do not change model family** | A tie → choose by fit cost |
| B8 top-3 clearly higher than top-1 | **the system, not the model** | Have the product show three suggestions. That is a result, not a failure |
| B4 flat **and** B5 flat **and** B6 balanced | **model family** | Only then move to a different family. Inference latency must be measured alongside |

## Result template

```markdown
## Model diagnosis — <run_id> · <model_label> · <scope>/<prep> · eval=<split>

**One-sentence diagnosis:** <bias / variance / labels / features> dominates, because <the number at rung Bx>.

| Rung | Measured | Threshold | Read as |
|---|---|---|---|

**Lever chosen:** <1 of 4> · **Next step:** `...` · **Expectation:** <marked not measured>
**Ruled out:** <hypothesis + the number that refutes it>
```

The **Ruled out** block is mandatory. Eliminating a lever with a measurement is
worth as much as choosing one.

## Avoid

- **Concluding from a gap under 2σ.** Rule 2. Read that run's own σ.
- **Comparing two runs when one lacks `predictions.npz`.** Most old runs do not have
  this file — you can read their individual numbers but cannot pair them. Say so
  plainly instead of comparing anyway.
- **Subtracting two vectors without checking alignment.** Copy
  `report_sweep.check_alignment`: `row_index` and `y_true` must match exactly. If
  they do not, stop; do not "fix" them into matching.
- **Comparing runs with different `eval_split` / `scope` / `prep`.** Those are two
  different experiments.
- **Touching test.** Rule 3.
- **Retraining to obtain a number already on disk.** Rule 1.
- **Inferring a diagnosis from a hyper-parameter.** A small `C` is not enough to
  conclude overfitting — only B4 concludes that.
- **Writing into `04-results.md` from this skill.** Diagnostics run through
  `python - <<PY` or `--no-save`. Only once a new configuration has been decided on
  do you call `python -m vietjobs.train` — and then `docs/` must be updated under
  Rule 1 in `AGENTS.md`.
