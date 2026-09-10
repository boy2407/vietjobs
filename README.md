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

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt && pip install -e .
pytest -q
```

The deep-learning package runs in its own environment (torch has no wheel for
macOS Intel + Python 3.14) — see [requirements-dl.txt](requirements-dl.txt).

`data/` is not in git — see [data/README.md](data/README.md) for how to obtain the
raw data.

## Run

```bash
python scripts/analyze_data.py                      # data analysis + figures

PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode   --task category --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --class-weight
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary
```

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
