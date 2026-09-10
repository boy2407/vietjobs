# VietJobs

Occupation classification and salary estimation from Vietnamese job postings.
48,092 raw postings → 47,707 clean rows → 16 occupation classes, split by group
and frozen.

**Main track:** a deep network on **PhoBERT** representations — the two tasks
separately first (occupation classification, salary estimation), the multi-task
merge second.

**Where it stands**, scored on `dev`, on the same splits and the same seed as every
older bar:

| Task | Deep-learning baseline | Bar to beat |
|---|---|---|
| Occupation classification | macro-F1 **0.5987** · top-3 **0.9257** | 0.6050 ± 0.0071 — TF-IDF + LinearSVC ([archive/](docs/archive/README.md)) |
| Salary estimation | MAE **4.83 million** · R²log **0.381** | 5.86 million — predict the median ([docs/05](docs/05-phan-tich-du-lieu.md)) |

The salary head beats its bar by 17.6 %. The classification head is 0.0063 below
its bar — less than half that bar's own confidence interval, i.e. **not
distinguishable yet**. The details, including the first run that diverged, are in
[docs/06](docs/06-baseline-dl.md).

## Install

Two environments, deliberately separate. The main one runs the data pipeline,
the Vietnamese NLP and the tests; `.venv-dl` runs everything under
`src/vietjobs/dl/`. They do not share packages and neither replaces the other.

### 1. Main environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt && pip install -e .
pytest -q
```

### 2. `.venv-dl` — the deep-learning environment

**Why it is separate.** `torch` publishes no wheel for macOS x86_64 after 2.2.2,
and none at all for Python 3.14 — the version the rest of the project runs on.
So the deep-learning package gets its own interpreter, pinned to `torch==2.2.2`.

**What it needs:**

| Requirement | Value on this machine | Why |
|---|---|---|
| Interpreter | **Python 3.9.6** (`/usr/bin/python3`) | The last version with a `torch` 2.2.2 wheel here. Any 3.9–3.11 works; 3.12+ does not, on macOS Intel |
| Disk — venv | **1.1 GB** | `torch` alone is most of it |
| Disk — model cache | **1.0 GB** in `~/.cache/huggingface` | `vinai/phobert-base-v2`, downloaded on the first `encode` run |
| Disk — vector cache | **238 MB** per split scheme | `artifacts/embeddings/`, four `.npy` files: two column families × two splits. Not in git, regenerated locally |
| Network | Once, on first run | To pull PhoBERT from the Hugging Face Hub. Everything after that is offline |
| Accelerator | Optional | `encode` picks MPS → CUDA → CPU automatically (`--device` overrides). On CPU it is **~20 minutes per 33k postings** |

```bash
/usr/bin/python3 -m venv .venv-dl                    # Python 3.9
.venv-dl/bin/pip install -r requirements-dl.txt
```

`pip install -e .` is **not** run in this environment — the deep-learning
commands reach the package through `PYTHONPATH=src` instead. That is why every
one of them starts with it; dropping it gives `ModuleNotFoundError: vietjobs`.

Resolved versions, for reproducing the runs in
[docs/04-results.md](docs/04-results.md): `torch 2.2.2` · `transformers 4.46.3` ·
`tokenizers 0.20.3` · `numpy 1.26.4` · `pandas 2.3.3` · `scikit-learn 1.6.1`.
`numpy` must stay on 1.x — torch 2.2.2 is built against it.

**Check the environment is live:**

```bash
PYTHONPATH=src .venv-dl/bin/python -c "
import torch, transformers
from vietjobs.dl import text
print(torch.__version__, transformers.__version__, torch.backends.mps.is_available())"
# 2.2.2 4.46.3 True
```

A `NotOpenSSLWarning` from urllib3 on that line is expected — the system Python
3.9 links LibreSSL — and harmless.

The data-path tests run in the **main** environment, not this one —
`dl/text.py` must not import torch, which is what keeps `pytest -q` green on a
machine with no torch at all.

### 3. Data

`data/` holds the frozen splits — **34,354 / 3,812 / 9,541 rows** over 16
occupation classes — and is **not in the repository**: 717 MB, with one file above
GitHub's 100 MB limit. You rebuild it instead, and the rebuild is exact — the same
postings land in the same splits on every machine:

```bash
# 1. download the raw dump and check its sha256   (data/README.md has the snippet)
# 2. confirm the segmenter is installed
python -c "from underthesea import word_tokenize; print('segmenter ok')"
# 3. build — ~26 minutes, mostly segmentation
python -m vietjobs.dataset build
```

Then **verify the split fingerprints** against the table in
[data/README.md](data/README.md#4-verify-you-got-the-same-splits) before training
anything. Row counts alone can match while the membership differs. Do **not**
rebuild with a different seed — that makes every row in `docs/04-results.md`
incomparable (Rule 2 in [AGENTS.md](AGENTS.md)).

Read a split with `dataset.load_split`, never with a bare `pd.read_csv` — the
types live in `manifest.json`, not in the CSV:

```python
from vietjobs import dataset as D
df = D.load_split("dev")      # (3812, 50)
```

Do **not** rebuild the splits with a different seed — that makes every row in
`docs/04-results.md` incomparable (Rule 2 in [AGENTS.md](AGENTS.md)).

## Run

Which environment each command belongs to is not interchangeable.

```bash
# main environment
python scripts/analyze_data.py                      # data analysis + figures
pytest -q                                           # must be green before a commit

# .venv-dl — embed once per column family (~20 minutes per 33k postings on CPU),
# then train the dense head on the cached vectors (seconds per epoch)
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode   --task category --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --class-weight
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary
```

Running `encode` first is not strictly required — `train_dl` embeds on the fly
when the cache is missing — but then a run that should take seconds silently
takes ~20 minutes instead. The `category` and `salary` tasks read **different**
column families, raw text and salary-masked text, so each needs its own `encode`
pass; the two caches are separate files on purpose and mixing them is a silent
salary leak (Rule 3).

## Documentation

[docs/00-tong-quan.md](docs/00-tong-quan.md) is the home page: the pipeline
diagram, the status table, the table of contents. Start at
[docs/05](docs/05-phan-tich-du-lieu.md) to understand the data, or
[docs/06](docs/06-baseline-dl.md) to understand the model. Newcomers to ML should
read [docs/nen-tang/](docs/nen-tang/00-index.md) first. Every run is logged in
[docs/04-results.md](docs/04-results.md).

## Contributing (humans and AI)

The mandatory rules are in [AGENTS.md](AGENTS.md): living documentation, the
experiment protocol (frozen seed, test touched once, an append-only log), salary
leak prevention, and one shared code path for training and serving. All Markdown
in this repository is written in English (Rule 6). `CLAUDE.md` and `GEMINI.md` are
symlinks to that file; `.cursor/rules/` and `.github/copilot-instructions.md` are
pointers.
