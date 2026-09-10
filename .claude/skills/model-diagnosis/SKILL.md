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
one fine-tune costs an hour or two. **Changing the model family is always the last
lever.**

`artifacts/cat-FINAL-svm-C0.02-test` is the single `test` touch that has been spent.
Do not run `--eval test`, do not read `test.parquet` for diagnosis, do not
"re-confirm" a conclusion on test.

---

## The diagnostic ladder

The shared preamble for every block:

```python
import sys, json; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score
RUN = 'cat-SW-svm-1'
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

`train.py --eval` only accepts `dev` and `test`, and **do not** add a `train` option
— it would write a meaningless row into the append-only log. The right way is to
load the saved model and score again:

```python
p = joblib.load(f'artifacts/{RUN}/model.joblib')
tr = pd.read_parquet('data/processed/splits/train.parquet')
print(f1_score(tr["category"], p.predict(tr), average="macro", zero_division=0))
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
from sklearn.pipeline import Pipeline
from vietjobs import features as F
from vietjobs.models import build_estimator
prep = F.PrepConfig(province=True)                     # exactly the config of the run being examined
gids = tr["group_id"].unique().copy(); np.random.default_rng(42).shuffle(gids)
for frac in (0.1, 0.25, 0.5, 1.0):                     # shuffle ONCE -> nested subsamples
    sub = tr[tr["group_id"].isin(set(gids[:int(len(gids)*frac)]))]
    est, _ = build_estimator("category", "svm", 42); est.set_params(C=0.02)
    pipe = Pipeline([("features", F.build_features("category", "full", prep, None)),
                     ("estimator", est)]).fit(sub, sub["category"])
    print(frac, len(sub), f1_score(dv["category"], pipe.predict(dv),
                                   average="macro", zero_division=0))
```

Four points cost about three to four minutes, write no artifact and touch no log.
Read the gap between `frac=0.5` and `frac=1.0`: under 2σ → the curve has saturated,
**more data will not pay**, move to features or labels. Above 0.03 → still climbing,
and collecting more data is the cheapest lever — with the warning that rebuilding
the splits invalidates `04-results.md`.

### B6 · Contribution per feature block — `Σ|w|`

```python
from vietjobs.features import source_columns
ct, est = p.named_steps["features"], p.named_steps["estimator"]
names = ct.get_feature_names_out()                     # shaped "block__feature"
blocks = np.array([n.split("__", 1)[0] for n in names])
t = pd.DataFrame({"block": blocks, "w": np.abs(est.coef_).sum(axis=0)}) \
      .groupby("block").agg(dims=("w","size"), w=("w","sum"))
t["pct"] = 100 * t.w / t.w.sum()
t["per_dim_x"] = (t.pct / t.dims) / (100 / t.dims.sum())
print(t.sort_values("pct", ascending=False).round(3).to_string(), source_columns(ct))
```

`per_dim_x` is the average weight of one dimension in the block, relative to the
global average. Cross-check this table against
`docs/archive/05-dac-trung-tfidf.md` §9.

| Condition | Action |
|---|---|
| A block's `per_dim_x` is more than 2× that of the largest block | That block is **diluted** — weight it up (archive roadmap Priority 3a: title ×2, ×3) |
| A block holds over 25 % of the dimensions with `per_dim_x` under 0.8 | Cut its dimensions with `min_df` / `max_features` — archive roadmap Priority 3b |
| A block under 0.1 % of the weight | Remove the block and re-measure. Cross-check A4: is the source column mostly empty |

Only works for linear models (`svm`, `logreg`). For trees, replace
`np.abs(est.coef_).sum(axis=0)` with `est.feature_importances_` and keep the rest.

### B7 · Hyper-parameter sensitivity — read the existing sweep, do not re-sweep

```python
import glob
rows = [(json.load(open(f))["model"], json.load(open(f))["metrics"]["f1_macro"])
        for f in sorted(glob.glob('artifacts/cat-SW-*/metrics.json'))]
df = pd.DataFrame(rows, columns=["model","f1"])
print(df.groupby("model").f1.agg(["min","median","max"])
        .assign(spread=lambda d: d["max"] - d["min"]).round(4).to_string())
```

The rule: **a spread under 2σ across six configurations means hyper-parameters are
exhausted for that model.** Sweeping further burns machine-hours without changing
the conclusion. To rebuild all seven comparison tables, run
`PYTHONPATH=src python scripts/archive/report_sweep.py` — it runs `check_alignment`
itself and carries six reproduction anchors.

### B8 · Calibration and top-3 — when there is no `y_proba`

`LinearSVC` has no `predict_proba`, so `predictions.npz` for every `svm` run has
only four keys. Check first, do not assume: `np.load(...).files`.

The substitute for `svm` is the `decision_function` margin:

```python
M = p.decision_function(dv); labs = np.array(p.named_steps["estimator"].classes_)
top3 = labs[np.argsort(-M, axis=1)[:, :3]]
print(np.mean([t in r for t, r in zip(dv["category"], top3)]))
S = np.sort(M, axis=1); gap = S[:, -1] - S[:, -2]
ok = (dv["category"].values == labs[M.argmax(1)])
for q in (0.10, 0.25, 0.50):
    th = np.quantile(gap, q); print(q, ok[gap <= th].mean(), ok[gap > th].mean())
```

The margin gives **the right ordering but not a probability**. It can be used for a
refuse-to-answer threshold (if accuracy below the threshold is clearly lower than
above it), but it **cannot** be used to say "the model is 80 % confident". For real
probabilities, run exactly once
`python -m vietjobs.train --task category --model svm_calibrated --scope full --province --C 0.02`
— three times the fit cost, in exchange for `y_proba` and `top3_accuracy`.

---

## Decision table — symptom → lever → next step

| Symptom (rung) | Lever | Concrete action |
|---|---|---|
| B1 ρ below 0.3 | **rule out hyper-parameters** | Drop the idea of adding `class_weight` / resampling. Record it as ruled out |
| B2 symmetric pair + high A6 cosine | **labels** | Merge the classes into a secondary axis, or record the ceiling. Do not run a new model |
| B3 over 30 % of errors are "wrong original label" | **labels** | Record the corrected ceiling in the matching `docs/` note; stop comparing models around the current level |
| B3 over 40 % of errors are "genuinely ambiguous" | **problem definition** | Move the metric to top-3 or multi-label — a new axis, not a replacement for the old one |
| B4 gap between 2σ and 0.10 | **features** | Archive roadmap Priority 3a: add `transformer_weights` to `build_features`, re-run the same configuration |
| B4 gap above 0.15 | **hyper-parameters** | A smaller `--C`, or a larger `min_df` |
| B5 gap from frac 0.5→1.0 under 2σ | **rule out data** | Do not collect more. Move to features or labels |
| B6 one block's `per_dim_x` is several times higher | **features** | Weight that block ×2 then ×3, two runs, paired-bootstrap against the original |
| B6 a block with many dimensions but little weight | **features** | Archive roadmap Priority 3b: edit `_word_tfidf` (`min_df`, `max_features`) — `--set` cannot reach TF-IDF |
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
