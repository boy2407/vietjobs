# AGENTS.md — how to work on VietJobs

This file is the **single source** of the rules, for every AI agent (Claude Code,
Cursor, Copilot, Codex, Gemini CLI…) and for humans too. `CLAUDE.md` and
`GEMINI.md` are symlinks to it; `.cursor/rules/` and
`.github/copilot-instructions.md` are only pointers. To change a rule, edit
**this file**, never a copy.

---

## 0. What the project is

Occupation classification and salary estimation from Vietnamese job postings
(48,092 raw postings → 47,707 clean rows, 16 occupation classes). Three tasks:
`category` (classification), `disclosed` (does the posting state a salary?),
`salary` (regression on `log1p(salary_mid)`).

**The main track is deep learning** (since 2026-09-08): PhoBERT → dense → a
single head, one task at a time first, multi-task later. The machine-learning
track is closed and lives in `docs/archive/` and `scripts/archive/`; it is the
**bar to beat** (test macro-F1 0.6112 for classification; dev MAE 5.86 million
VND for salary), not junk — do not extend that branch, and do not delete it.

The project map is [docs/00-tong-quan.md](docs/00-tong-quan.md). Read it before
touching the code.

## 1. Environment and commands

**Two environments, deliberately separate.** `torch` has no wheel for macOS
x86_64 after 2.2.2 and none at all for Python 3.14, so the `dl/` package runs in
its own venv.

```bash
# main environment (data, Vietnamese NLP, machine learning, tests)
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt && pip install -e .
pytest -q                                            # must be green before a commit
python scripts/analyze_data.py                       # measures + figures for docs/05

# deep-learning environment
/usr/bin/python3 -m venv .venv-dl                    # Python 3.9
.venv-dl/bin/pip install -r requirements-dl.txt
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode   --task category --splits train dev
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task category --class-weight
PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.train_dl --task salary
PYTHONPATH=src .venv-dl/bin/python scripts/probe_embeddings.py --task category  # diagnostic bar

# machine-learning path — only to re-run the old benchmark
python -m vietjobs.dataset build                     # rebuild the splits (rarely needed)
python -m vietjobs.train --task category --model svm --C 0.02 --province
python -m vietjobs.predict --title "..." --description "..."
bash scripts/render_figures.sh                       # render the mermaid diagrams in docs/
```

`underthesea`, `pyvi`, `lightgbm` and `xgboost` are optional at import time —
without them the pipeline still runs, it just loses one measurement axis. Do not
wrap them in new `try/except` blocks: `vitext.py` already handles that.

## 2. Layout

| Path | What it is |
|---|---|
| `src/vietjobs/` | 8 shared modules. Role of each file: [docs/08-ma-nguon.md](docs/08-ma-nguon.md) |
| `src/vietjobs/dl/` | The main track: `text.py` (input) · `encode.py` (PhoBERT) · `heads.py` (dense) · `train_dl.py` (CLI) |
| `tests/` | 5 test files. `pytest -q` runs all of them — it works even without torch |
| `scripts/` | Measurement and reporting — not a library, never imported back into `src/` |
| `docs/archive/`, `scripts/archive/` | The closed machine-learning phase. **Do not edit** |
| `docs/` | The living map of the project (see Rule 1) |
| `resources/` | Abbreviations, stopwords, province list — Vietnamese data, committed |
| `data/`, `artifacts/` | **Never committed** (already in `.gitignore`) |

Do not add a new module under `src/vietjobs/` without updating
[docs/08-ma-nguon.md](docs/08-ma-nguon.md) in the same working session.

---

## Rule 1 — always keep the documentation current

`docs/` is the **living map** of the project, not write-once documentation.
`docs/00-tong-quan.md` is the home page; every topic has its own note.
After **every** real change, update it in the same working session:

| What you just did | Which file you must edit |
|---|---|
| Finished a training run | `04-results.md` (automatic, append-only) + the results table in `06-baseline-dl.md` + the status cell in `00-tong-quan.md` |
| Changed the network architecture / default hyper-parameters | `06-baseline-dl.md` — diagram + the "five decisions" table |
| Added a data measurement, changed a figure | `05-phan-tich-du-lieu.md` + re-run `python scripts/analyze_data.py` |
| Changed a preprocessing step | `02-vietnamese-nlp.md` — the nine-step table, the ablation table, the diagram (dashed for a removed step) |
| Changed the cleaning / splitting | `01-data-audit.md` |
| Changed how a run is logged | `03-protocol.md` §3b |
| Added / changed / removed a module in `src/` | `08-ma-nguon.md` — diagram + file-role table + line and test counts |
| Completed a work item | The status table in `00-tong-quan.md` + `09-lo-trinh.md` |
| Changed the model / system architecture | The mermaid diagram in the matching note |
| **Any of the above** | **Also the matching chapter in `docs/bao-cao/` — see Rule 7** |

Exception: **`docs/nen-tang/` holds no result numbers.** It is the
concept-explanation track for newcomers, which is exactly why it sits apart — it
does not have to be updated after every experiment. Edit it only when the method
itself changes.

Adding or renaming a note means editing **three** places: the table of contents
in `00-tong-quan.md`, the table above, and the `SOURCES` array in
`scripts/render_figures.sh` (it matches mermaid diagrams by position and exits if
the counts disagree).

**Documentation must never drift from reality.** If a diagram says "not run yet"
while `04-results.md` already has a result row, that document is lying to its
reader.

### Principles for writing documentation

- **Every number must be measured**, with its source: `metrics.json`,
  `manifest.json`, or a row in `04-results.md`. No estimates, no "about".
- **Spell out every construction step** — this is the main goal of the project.
  The reader must understand *why* a step was taken, *what measured it*, and
  *what the measurement said*.
- **Record the failures too.** A step that was measured and did not help stays in
  the table with the verdict "dropped" — that is evidence, not junk.
- Short sentences, one idea per sentence. Decimal point (0.6112).

## Rule 2 — experiment protocol

Details in [docs/03-protocol.md](docs/03-protocol.md). Four absolutes:

1. **Never change** `SPLIT_SEED = 20260826` or the split fractions. Changing them
   makes every row in `04-results.md` **and every benchmark in `docs/archive/`**
   incomparable.
2. **`test` is touched exactly once**, at the end, behind `--confirm-test`. Model
   selection uses `dev` — and `test` is never *transformed*, see Rule 5.
3. **`04-results.md` is append-only.** Never edit a row that is already written.
4. **Any preprocessing step without an ablation row proving it gets removed.**

## Rule 3 — leak prevention

The two salary tasks may only read `*_masked` columns. `features.resolve_column`
is the only place that decides this, and `tests/test_no_leak.py` and
`tests/test_dl_text.py` hold the line. The deep-learning path **must not** wire
itself to a raw column: it goes through `dl/text.py` → `resolve_column`, and the
vector cache keeps the `raw` and `masked` families in separate files under
`artifacts/embeddings/` — mixing the two files is a silent leak. A leak here
**raises no error** — it quietly produces a model that reads its own answer key.

Warning: `UNMASKED_COLUMNS` is a **hand-written list**, so the test only guards
the columns whoever wrote it thought of. `soft_skills_text` and
`qualifications_text` are known to slip through — see `docs/09-lo-trinh.md`
Priority 4. Adding a new text column means updating `_MASKABLE` **and**
`UNMASKED_COLUMNS`.

## Rule 4 — training and inference share one code path

`predict.build_frame` must produce exactly the column layout `dataset.clean`
creates. A mismatch is a silent bug that only shows up as slowly worsening
predictions.

## Rule 5 — `test` is untouchable, `dev` is the only place to tune

1. **Do not transform `test` in any way.** No filtering, no de-duplication, no
   class balancing, no outlier trimming, no re-splitting, no overwriting
   `data/processed/splits/test.csv`. It stays exactly as `dataset.build`
   produced it: **9,541 rows**. The only permitted operation is **reading** it to
   report the final number — exactly once, behind `--confirm-test`.
2. **Every transformation and adjustment happens on `train` and `dev` only**:
   cleaning, filtering, de-duplication, class balancing, changing preprocessing
   steps, tuning hyper-parameters, selecting models. Every decision reads numbers
   on `dev`.
3. **The `val` split is called `dev` from now on** in the documentation and in
   every discussion. It is the development set — it may be looked at without
   limit, precisely because it is not `test`.
4. The identifier in the code and on disk is **`dev` as well**, since
   2026-09-09: `splits/dev.csv`, `--eval dev`, `--splits train dev`, the
   `"dev"` key in `manifest.json`. The blanket rename happened together with the
   new split scheme, because that scheme had already invalidated every number
   measured before it — so there was nothing left to stay name-compatible with.
   Rows already written in `docs/archive/04-results-ml.md` still say `val`; that
   file is append-only and describes the old splits, so leave it alone.
5. **The split scheme is two-stage** (`config.SPLIT_TEST_FRACTION`,
   `SPLIT_DEV_FRACTION`): `train:test` = 8:2, then that train pool split again
   into `train:dev` = 9:1 → 34,354 / 3,812 / 9,541 rows. `dev` is small on
   purpose; the smallest class has 25 rows there, so read `f1_macro_no_junk`
   next to any per-class number and confirm on `test` at the end.

## Rule 6 — every Markdown file is written in English

1. **All `.md` in this repository is English**: `docs/` (including
   `docs/nen-tang/`), `README.md`, this file, and every `SKILL.md` under
   `.claude/skills/`. Body text, headings, table cells, mermaid diagram labels
   and code comments inside fenced blocks — all of it.
2. **Numbers follow English convention**: decimal point and thousands comma
   (`0.6112`, `47,707`), not `0,6112` / `47.707`.
3. **Vietnamese stays only where it is data or a quotation**: sample postings,
   examples of raw text, class names as they appear in the dataset, entries in
   `resources/`, and identifiers in the code. Gloss such a quotation in English
   when its meaning carries the argument.
4. **Filenames do not change.** The note slugs (`00-tong-quan.md`,
   `05-phan-tich-du-lieu.md`, …) are wired into `scripts/render_figures.sh`, into
   the cross-links between notes, and into the git history. Translate the
   content, keep the path. Heading anchors *do* move with the translation, so a
   `file.md#heading` link must be updated in the same session as the heading it
   points at.
5. `docs/04-results.md` and `docs/archive/` are frozen (Rule 2 item 3, and the
   layout table above). A row already written stays exactly as it was written,
   Vietnamese included; only new rows follow this rule.
6. **Exception — `docs/bao-cao/` is written in Vietnamese.** That directory is the
   thesis manuscript, submitted to a Vietnamese university and read by the
   supervisor; English there would have to be translated back before submission.
   It follows Vietnamese number convention (decimal comma, thousands dot:
   `0,6112`, `47.707`) — the opposite of item 2 — because the printed report must
   match the university template. Every other `.md` in the repository stays
   English. See Rule 7.

---

## Rule 7 — the thesis report is written as the work happens

`docs/bao-cao/` is the **living manuscript** of the graduation report, structured
to be assembled into the final Word document. Its control panel —
chapter status, presentation spec, assembly procedure — is
[docs/bao-cao/00-index.md](docs/bao-cao/00-index.md).

**After every real change, update the matching chapter in the same working
session**, exactly as Rule 1 requires for `docs/`:

| What you just did | Which chapter you must edit |
|---|---|
| Finished a training run | `05-chuong-4-thuc-nghiem.md` — the results table + §4.8 summary |
| Changed the architecture / hyper-parameters | `04-chuong-3-phuong-phap.md` §3.7 |
| Changed cleaning, splitting, or preprocessing | `04-chuong-3-phuong-phap.md` §3.2–§3.5 |
| Re-ran the data analysis | `04-chuong-3-phuong-phap.md` §3.6 |
| Added / changed a module in `src/` | `06-chuong-5-he-thong.md` §5.2 |
| Completed a roadmap item | `07-chuong-6-ket-luan.md` §6.1 + the plan table in `09-phu-luc.md` |
| Read a paper worth citing | `08-tai-lieu-tham-khao.md` **and** Table 2.1 in `03-chuong-2-tong-quan.md` |

Six absolutes:

1. **Every number traces to a source**, named in a `> **Nguồn số liệu:**` block:
   `manifest.json`, `artifacts/eda/summary.json`, `artifacts/<run_id>/metrics.json`,
   or a row in `04-results.md`. No estimates, no "khoảng".
2. **Never write a citation for something not read.** A plausible-looking
   reference is plagiarism even when the content happens to be right. Anything
   needing a literature search gets a `> ✍️ **CẦN VIẾT TAY**` marker instead of a
   guess — never a fabricated author, title, venue, or year.
3. **Never present a number as current when it is stale.** Results measured under
   an older split scheme carry the label `(lược đồ v1)` and the chapter says so at
   the top. A missing measurement gets `> ⛔ **CHƯA CÓ SỐ LIỆU**`, never a
   plausible-sounding sentence.
4. **Report failures too.** The diverged run at macro-F1 0.042 stays in the report
   with its cause — it is evidence, and it is what stops the same mistake twice.
5. **The report reuses figures already in `docs/figures/`**; it does not define new
   mermaid blocks, because `scripts/render_figures.sh` matches blocks by position
   and aborts when a file's count disagrees with its `SOURCES` entry.
6. **Both marker types must be gone before submission.** `⛔` and `✍️` are the
   to-do list; the final checklist is in `09-phu-luc.md` Phụ lục F.

Adding a chapter file means editing **three** places: the status table in
`docs/bao-cao/00-index.md`, the table above, and the table of contents in
`docs/00-tong-quan.md`.

---

## 3. Code style

- Plain Python, `from __future__ import annotations`, with type hints.
- `vitext.py` is **pure functions**: string in, string out, no file reads, no
  global state. Keep it that way — every step has its own test.
- Shared constants live in `config.py`; do not scatter magic numbers.
- Comments explain *why*, they do not restate *what* — read the existing code to
  match its tone.
- Adding a preprocessing step means adding a test in `tests/test_vitext.py`
  **and** an ablation row; what cannot be measured is not kept (Rule 2, item 4).
- Code under `dl/` runs on **Python 3.9** (the torch limit on this machine): use
  `from __future__ import annotations`, no 3.10+ syntax.
- `dl/text.py` must not import torch — that is what lets the data-path tests run
  in the main environment, where torch is absent.

## 4. What "done" means

1. `pytest -q` is green.
2. The real change is reflected in the right file under `docs/` (the Rule 1
   table).
3. Every number written down traces back to `manifest.json`, `metrics.json`, or a
   row in `04-results.md`.
4. `test` was not touched unless `--confirm-test` was given and it was the final
   step, and nothing wrote to `test.csv` (Rule 5).

## 5. Git

- No `Co-Authored-By: Claude…` trailer, no "Generated with…" line, no mention of
  AI anywhere in the git history.
- Commit only when asked. Never commit `data/`, `artifacts/`, `.venv/`.
