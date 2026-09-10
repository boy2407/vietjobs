[← Overview](00-tong-quan.md) · [← Deep-learning baseline](06-baseline-dl.md)

# Roadmap

The machine-learning track closed on 2026-09-08 and moved into
[archive/](archive/README.md). From here, the main track of the thesis is a **deep
network on PhoBERT representations**, in this order: the two tasks separately
first, the multi-task merge second.

The bars to beat, measured on the same splits with the same seed:

| Task | Bar | Source |
|---|---|---|
| Occupation classification | test macro-F1 **0.6112** (LinearSVC + TF-IDF) | [archive/04-results-ml.md](archive/04-results-ml.md) |
| Occupation classification — floors | majority **0.0210** · keyword rules **0.4321** | [archive/06-mo-hinh-phan-lop.md](archive/06-mo-hinh-phan-lop.md) |
| Salary estimation | dev MAE **5.86 million** (predict the median) | [05-phan-tich-du-lieu.md](05-phan-tich-du-lieu.md#6-the-most-important-finding-the-sector-says-almost-nothing-about-the-salary) |
| Salary disclosed or not | accuracy **0.7117** (always predict "yes") | same source |

---

## ~~Priority 1~~ — two separate deep-learning baselines · **DONE**

Frozen PhoBERT → dense → one head, one network per task. The architecture and the
results are in [06-baseline-dl.md](06-baseline-dl.md).

- [x] Data analysis, with figures and numbers ([05](05-phan-tich-du-lieu.md))
- [x] The DL data path goes through `features.resolve_column`, with a leak test
- [x] PhoBERT embeddings cached to disk, split into the `raw` / `masked` families
- [x] Classification baseline — `dl-cat-v2`, dev macro-F1 **0.5987** (old bar 0.6050 ± 0.0071)
- [x] Salary regression baseline — `dl-sal-v2`, dev MAE **4.83 million** (bar 5.86)
- [x] Linear probes as a diagnostic bar — `probe-cat` 0.5867 · `probe-sal` 6.60

Three items this result generated, ordered by value per hour spent:

| Item | Why | Cost |
|---|---|---|
| The salary head **dares not go out into the right tail** — it predicts at most 80.7 where the data reaches 275 | MAE above 30 million is **25.58** versus 3.70 on the rest | try an asymmetric loss, or predict quantiles |
| The class `nhóm_nghề_khác` has F1 = **0.000** | 48 samples, mixed content — most likely it should be merged or dropped, not learned | a labelling decision, not a modelling one |
| The three largest confusions are blurred label boundaries (`kinh_doanh` ↔ `du_lịch_nhà_hàng`, 382 postings both ways) | the label-noise ceiling has never been measured | sample 100 postings, label them by hand, measure agreement |

## Priority 2 — fine-tune PhoBERT (old level 4b)

Unfreeze the encoder weights, train end to end, 256 tokens. Only **after** the
frozen level has numbers: if the frozen version cannot beat 0.6112, the fault
mostly lies in the data path, and finding that out at the cheap level costs minutes
instead of hours.

This is the remaining step with the highest expected value: the frozen version
already matches the TF-IDF bar **without using the encoder's ability to adapt at
all**. The measured hardware constraint: this machine is Intel x86_64, with **no
MPS/CUDA**, and `torch` no longer publishes wheels for macOS Intel after 2.2.2.
Frozen embedding of 33,396 postings takes ~20 minutes at ~30 postings/s. A full
fine-tune on this machine is not realistic — it needs a Colab/Kaggle GPU, and then
`data/processed/splits/` has to be copied there together with `manifest.json` to
keep the splits identical.

## Priority 3 — merge into multi-task

```
PhoBERT → shared dense → head A: 16 classes   (cross-entropy)
                       → head B: salary       (huber, masked)
loss = w_A · loss_A + w_B · loss_B
```

Two constraints that may not be violated:

1. **`loss_B` must be masked.** 28.2 % of postings have no salary label; for them
   `loss_B` is 0, not a label of 0.
2. **The salary branch still reads the `*_masked` columns.** Once the two heads
   share a trunk, that trunk has to read the masked copy — meaning the multi-task
   version runs on the `masked` family, and its cost (the classification head loses
   the text that contains salary figures) is a number to measure, not an assumption.

Expectations should be low: the sector explains only **3.2 %** of the log-salary
variance
([05 §6](05-phan-tich-du-lieu.md#6-the-most-important-finding-the-sector-says-almost-nothing-about-the-salary)),
so the "two heads help each other" story has very little to share. The value of the
multi-task version most likely lies in **one model instead of two**, not in the
score.

## Priority 4 — data, the items the analysis pointed at

- [ ] The 116 edge-case salaries (103 postings < 2 million, 13 postings > 200 million)
- [ ] Measure the selection bias: the disclosure rate spans 56.8 % → 74.6 % by sector
- [ ] `UNMASKED_COLUMNS` is still a hand-written list — `soft_skills_text` and
      `qualifications_text` slipped through it once; the DL path currently reads
      only three text fields so it is not affected, but adding a field means
      updating both lists

## Priority 5 — the system

- [ ] `predict.py` does not know about the DL path. Rule 4 (train and serve share
      one code path) is currently **not guarded** for the deep-learning branch
- [ ] `artifacts/PRODUCTION.json` mapping task → run_id, replacing `_latest_run`,
      which currently picks by file modification time
- [ ] Measure inference latency and report it next to the score: PhoBERT at ~30
      postings/s on CPU versus TF-IDF + LinearSVC being effectively instant — that
      is a trade-off the report has to present
