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

**The only track is deep learning**: PhoBERT → dense → a single head, one task at
a time first, multi-task later. The bars to clear are the no-model floors and the
linear probe on the same vectors — see Rule 12.

The project map is [docs/00-tong-quan.md](docs/00-tong-quan.md). Read it before
touching the code.

## 1. Environment and commands

**Two environments, deliberately separate.** `torch` has no wheel for macOS
x86_64 after 2.2.2 and none at all for Python 3.14, so the `dl/` package runs in
its own venv.

```bash
# main environment (data, Vietnamese NLP, metrics, tests)
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

python -m vietjobs.dataset build                     # rebuild the splits (rarely needed)
bash scripts/render_figures.sh                       # render the mermaid diagrams in docs/
```

`underthesea` and `pyvi` are optional at import time — without them the pipeline
still runs, it just loses one measurement axis. Do not wrap them in new
`try/except` blocks: `vitext.py` already handles that.

## 2. Layout

| Path | What it is |
|---|---|
| `TASKS.md` | **The work board** — read it first in every session: active task, dependencies, evidence, log. [docs/09-lo-trinh.md](docs/09-lo-trinh.md) is the strategy; this is the operations |
| `tasks/` | Detail sheets for board sections that outgrew `TASKS.md` (one file per section, TASKS.md §3 Rule 7). `TASKS.md` keeps the pointer; status lives in the sheet |
| `src/vietjobs/` | 5 shared modules. Role of each file: [docs/08-ma-nguon.md](docs/08-ma-nguon.md) |
| `src/vietjobs/dl/` | The main track: `text.py` (input) · `encode.py` (PhoBERT) · `heads.py` (dense) · `train_dl.py` (CLI) |
| `tests/` | 5 test files (132 tests). `pytest -q` runs all of them — it works even without torch |
| `scripts/` | Measurement and reporting — not a library, never imported back into `src/` |
| `docs/` | The living map of the project (see Rule 1) |
| `hoc-tap/` | **Personal study notes, written in Vietnamese** — how the two deep-learning pipelines work, for a reader with no DL background. Outside Rule 1, Rule 6 and Rule 7: it holds no result numbers, nobody updates it after a run, and it is never cited in the thesis |
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
| Finished a training run | `04-results.md` (automatic, append-only) + the results table in `06-baseline-dl.md` + the status cell in `00-tong-quan.md` + regenerate `data/eda_xlsx/ket_qua_chay.xlsx` with `scripts/export_runs.py` (Rule 13) |
| Changed the network architecture / default hyper-parameters | `06-baseline-dl.md` — diagram + the "five decisions" table |
| Added a data measurement, changed a figure | `05-phan-tich-du-lieu.md` + re-run `python scripts/analyze_data.py` — measured on the **original file**, see Rule 8 |
| Changed a preprocessing step | `02-vietnamese-nlp.md` — the nine-step table, the ablation table, the diagram (only the steps on the path into PhoBERT, Rule 11) |
| Changed the cleaning / splitting | `01-data-audit.md` |
| Changed how a run is logged | `03-protocol.md` §3b |
| Added / changed / removed a module in `src/` | `08-ma-nguon.md` — diagram + file-role table + line and test counts |
| Completed a work item | The status table in `00-tong-quan.md` + `09-lo-trinh.md` |
| Re-ran the external evaluation, or reviewed/changed the 60 → 16 crosswalk | `10-danh-gia-ngoai.md` — the result tables + the crosswalk status line at the top |
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
  the ablation table of `02-vietnamese-nlp.md` with the verdict "dropped" — that
  is evidence, not junk. It is not retold elsewhere (Rule 11).
- **Only what was done** (Rule 11). Plans live in `TASKS.md` and
  `09-lo-trinh.md`, nowhere else.
- Short sentences, one idea per sentence. Decimal point (0.6025).

## Rule 2 — experiment protocol

Details in [docs/03-protocol.md](docs/03-protocol.md). Four absolutes:

1. **Never change** `SPLIT_SEED = 20260826` or the split fractions. Changing them
   makes every row in `04-results.md` incomparable.
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
the columns whoever wrote it thought of. `soft_skills_text`,
`qualifications_text` and `technical_skills_text` are not in it; they were
measured to hold **0** salary figures, and `tests/test_no_leak.py` keeps that
check — that test is now the source of the claim. Adding a new text column means updating `_MASKABLE` **and**
`UNMASKED_COLUMNS`.

## Rule 4 — training and inference share one code path

Any inference path must build exactly the column layout `dataset.clean` creates,
and read its columns through `features.resolve_column` like `dl/text.py` does. A
mismatch is a silent bug that only shows up as slowly worsening predictions.
There is **no inference entry point right now** — T5.1 owns adding one.

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
5. **The split scheme is two-stage** (`config.SPLIT_TEST_FRACTION`,
   `SPLIT_DEV_FRACTION`): `train:test` = 8:2, then that train pool split again
   into `train:dev` = 9:1 → 34,354 / 3,812 / 9,541 rows. `dev` is small on
   purpose; the smallest class has 25 rows there, so read `f1_macro_no_junk`
   next to any per-class number and confirm on `test` at the end.

## Rule 6 — language is per file, not per repository

1. **`docs/00-tong-quan.md` through `docs/09-lo-trinh.md` (the ten numbered
   top-level notes) are written in Vietnamese**, translated from English on
   2026-09-16. **Everything else stays English**: `docs/nen-tang/`,
   `README.md`, this file, and every `SKILL.md` under
   `.claude/skills/`. Two further notes were already Vietnamese before this
   change: `docs/bao-cao/` (the thesis manuscript, see item 6) and `hoc-tap/`
   (personal study notes — outside the documentation tree, never cited, see §2).
   In whichever language a file uses, translate all of it: body text, headings,
   table cells, mermaid diagram labels, and code comments inside fenced blocks.
   Code itself — commands, identifiers, file paths — is never translated.
2. **Numbers in the ten Vietnamese notes follow Vietnamese convention**:
   decimal comma, thousands dot (`0,6112`, `47.707`), matching
   `docs/bao-cao/` — including when a sentence quotes a result that lives in
   `04-results.md` under the English convention; only the log rows themselves
   stay as originally measured (see item 5). English-language files
   (`docs/nen-tang/`, `README.md`, this file, `SKILL.md`) keep
   English convention: decimal point, thousands comma.
3. **English stays only where it is a code identifier, file path, column name,
   commit hash, or a literal quotation** (sample postings, raw text examples,
   class names as they appear in the dataset, entries in `resources/`). Gloss
   such a quotation in Vietnamese when its meaning carries the argument.
4. **Filenames do not change.** The note slugs (`00-tong-quan.md`,
   `05-phan-tich-du-lieu.md`, …) are wired into `scripts/render_figures.sh`, into
   the cross-links between notes, and into the git history. Translate the
   content, keep the path. Heading anchors *do* move with a translation, so a
   `file.md#heading` link must be updated in the same session as the heading it
   points at.
5. **`docs/04-results.md`'s log table is frozen**
   (Rule 2 item 3, and the layout table above). The RunID rows already written —
   text and number formatting both — stay exactly as written; only the prose
   around the table (e.g. the log's opening paragraph) follows item 1. New rows
   appended to `04-results.md` keep the log's existing English-convention,
   code-like format (`macroF1=0.6112`) regardless of item 2, since that format is
   what scripts and other docs parse and quote from.
6. **`docs/bao-cao/` is written in Vietnamese** for an unrelated reason: it is
   the thesis manuscript, submitted to a Vietnamese university and read by the
   supervisor, so it must match the university template independently of the
   choice made in item 1. See Rule 7.

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
| Completed a roadmap item | `07-chuong-6-ket-luan.md` §6.1 (and drop the item from §6.3) |
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
   This covers runs that were done and failed; steps never run, or removed
   preprocessing steps, are not retold in the report (Rule 11).
5. **The report reuses figures already in `docs/figures/`**; it does not define new
   mermaid blocks, because `scripts/render_figures.sh` matches blocks by position
   and aborts when a file's count disagrees with its `SOURCES` entry.
6. **Both marker types must be gone before submission.** `⛔` and `✍️` are the
   to-do list; the final checklist is in `09-phu-luc.md` Phụ lục E.

Adding a chapter file means editing **three** places: the status table in
`docs/bao-cao/00-index.md`, the table above, and the table of contents in
`docs/00-tong-quan.md`.

---

## Rule 8 — data analysis is measured on the original file

Every **descriptive** number about the dataset is measured on
`data/raw/VietJobs.csv` after exact de-duplication — **47,707 rows** — and never
on a split. `scripts/analyze_data.py` is the only producer of those numbers.

| Kind of number | Measured on | Examples |
|---|---|---|
| **Descriptive** — what the dataset *is* | the **original file**, after `drop_duplicates()` | class skew, salary distribution and its tails, disclosure rate, field completeness, text length, eta² |
| **Decision** — what to *build* | `train` and `dev` only | the no-model floors, the `--max-len` choice, hyper-parameters, model selection |

Three reasons the descriptive numbers belong on the original file:

1. A chapter describing the data has to describe **the corpus as collected**. A
   reader needs to know what the dataset is before knowing how it was split.
2. The raw file **does not move when the split scheme changes**. Every descriptive
   number measured this way survived the 2026-09-09 re-split; every number
   measured on the old `train` did not.
3. It keeps Rule 5 intact without pretending `test` does not exist. `test` is not
   *read* as a split here — the original file is read as one corpus, and no number
   from it selects a model or a hyper-parameter.

Four obligations that come with it:

1. **Every table and every figure states its scope**, in the caption or in the
   first line of the section: *original file (47,707)* or *fitted on `train`,
   scored on `dev`*. A number with no scope is a number nobody can check.
2. **The text and the figure beside it come from the same run** of
   `scripts/analyze_data.py`. If the figures were regenerated and the prose was
   not, the page contradicts itself — that is worse than being stale, because both
   numbers look measured.
3. **When a figure is regenerated, `docs/05-phan-tich-du-lieu.md` and
   `docs/bao-cao/04-chuong-3-phuong-phap.md` §3.6 are rewritten in the same
   working session** — they are the two places that quote those figures.
4. **A descriptive number that cannot be re-measured yet gets `⛔`**, never a
   leftover value from an older scope. Decision numbers measured under the old
   split scheme carry `(lược đồ v1)` as Rule 7 item 3 requires.

---

## Rule 9 — figures are split small for report readability

1. **Each subplot gets its own PNG file** (not one wide figure with three panels).
   A single multi-panel figure is hard to resize and cite in `docs/bao-cao/`.
2. **Naming convention**: if a logical group has sub-figures, use `-05a-`, `-05b-`,
   `-05c-` instead of cramming them into `-05-`. The number stays the same; the
   suffix shows the split.
3. **Size**: each individual figure ~6–7 inches wide, readable at 100% on print.
   The report can then cite `eda-05a`, `eda-05b`, and `eda-05c` side by side or in
   sequence without rescaling artefacts.
4. **Examples**: salary shape (now `-05a-thang-tho`, `-05b-sau-log1p`, `-05c-duoi-iqr`)
   replaces the old three-column layout. Any other multi-panel figure added should
   follow this pattern.

---

## Rule 10 — the report describes every task completely

When a task in `TASKS.md` moves to `done`, the matching chapter in `docs/bao-cao/`
must let a reader **reproduce that run without opening `TASKS.md` or
`04-results.md`**. Rule 7 says *when* and *where* to edit the manuscript; this
rule says *what* the manuscript must contain. A results table with a number and
a one-line remark is not a description of a run.

Five things every run must be told:

1. **What was run** — the task ID (`T1.3`), the `run_id`, the exact command and
   environment (`.venv` / `.venv-dl`, device), the date, which splits were read
   and which one was scored (`eval=dev`, n rows).
2. **What the model is** — the architecture and every hyper-parameter that
   deviates from the default (lr, hidden size, patience, class weighting, loss),
   which embedding family was loaded (`raw` / `masked`, `len256`), the best
   epoch. For a floor or a probe, the exact estimator.
3. **How the data was handled** — the de-duplication tiers, the salary-label
   repair, the split scheme (`v2`: 8:2 then 9:1, seed 20260826), the row count
   per split, and which columns the task reads (`*_masked` for the salary tasks,
   Rule 3).
4. **How the Vietnamese text was processed** — which of the nine `vitext` steps
   were on, which were dropped and the ablation number proving it, the
   tokenizer / word segmentation PhoBERT expects, and `--max-len 256` with its
   measured cost (19.8 % of postings truncated, 91.5 % of tokens kept).
5. **What the preprocessing analysis said** — the EDA measurement behind each
   choice above (class skew, disclosure rate, salary tails, token length…), its
   scope stated as Rule 8 requires, and the no-model floor the run is compared
   against.

| Required content | Where it lives in `docs/bao-cao/` |
|---|---|
| 1 — what was run | `05-chuong-4-thuc-nghiem.md` §4.1.3 (per-run record) + the results table in §4.3 / §4.4 |
| 2 — the model | `04-chuong-3-phuong-phap.md` §3.7; per-run deviations in the §4.3 / §4.4 prose |
| 3 — data handling | `04-chuong-3-phuong-phap.md` §3.2, §3.4, §3.5 |
| 4 — Vietnamese processing | `04-chuong-3-phuong-phap.md` §3.3 |
| 5 — preprocessing analysis | `04-chuong-3-phuong-phap.md` §3.6 + §3.8 (metric and floor) |

Two obligations that come with it:

1. **Each item is written in the three beats** fixed in `docs/bao-cao/00-index.md`
   §4 — principle → how it is applied here → why it deviates from the default,
   with the number that justifies it. An item that cannot be measured yet gets
   a one-line `> ⛔ **CHƯA CÓ SỐ LIỆU** (task ID)`, never a generic sentence
   (Rule 11).
2. **A task is not `done` while any of the five is missing** from its chapter.
   This is part of the definition of done in §4 below.

Changing this list means editing **three** places: this rule, §7 of
`docs/bao-cao/00-index.md`, and item 5 of `TASKS.md` §3.

---

## Rule 11 — the report tells only what was done, and tells it once

1. **`docs/bao-cao/` and the numbered notes in `docs/` describe only work that
   was actually run.** A step that was never run is not written as method,
   architecture, system, or result — no "proposed", "sẽ", "dự kiến" prose in
   the body, no diagram node for something that does not exist.
2. **Work still to do is kept**, but only in the places made for it:
   `TASKS.md`, `docs/09-lo-trinh.md`, the objectives in `02-chuong-1` §1.2 and
   the future-work section `07-chuong-6` §6.3. Inside any other section, a
   missing number is a **one-line** `⛔` marker naming its task ID — never a
   paragraph describing the undone step. Move a to-do item to one of those
   places before deleting it from a note.
3. **Done work is written without excess.** Describe the steps that ran on the
   path the model actually reads, in order, once. Do not list steps that were
   removed or deliberately not done (their ablation evidence lives in
   `docs/02-vietnamese-nlp.md`), do not repeat what another
   section already says (link to it), do not restate a table in prose.
   A run that failed is still a run that was done: it stays (Rule 7 item 4).
4. **Example:** bao-cao §3.3 lists only NFC → tone marks → abbreviations →
   salary mask → word segmentation → field join, on the three columns PhoBERT
   reads — not the removed steps, not columns PhoBERT never reads.

Changing this rule means editing **three** places: this rule, §8 of
`docs/bao-cao/00-index.md`, and item 6 of `TASKS.md` §3.

---

## Rule 12 — the repository has one track, and does not name the closed one

1. **Do not mention the earlier sparse-feature / classical-machine-learning
   track anywhere** — not in `docs/`, `docs/bao-cao/`, `docs/nen-tang/`,
   `README.md`, `TASKS.md`, source comments, or any `SKILL.md`. Its files, its
   numbers and its dependencies were removed on 2026-09-19; writing them back in
   as context, comparison, or history re-creates the thing that was removed.
2. **The only bars are measured on this track**: the no-model floors
   (`artifacts/eda/summary.json`, key `floors`) and the linear probe on the same
   PhoBERT vectors (`probe-cat-s2`, `probe-sal-s2`). A new model is judged
   against those and against the two baseline runs `dl-cat-s2` and `dl-sal-s2`,
   never against a number that no longer has a file behind it.
3. **The exception is the work log.** `TASKS.md` §5 and the rows already written
   in `docs/04-results.md` are an immutable record of what was done on which day
   (Rule 2 item 3). They are not edited to match this rule — rewriting a log to
   fit a later decision is falsifying it, which is worse than the mention.
4. **A number with no live source is deleted, not relabelled.** If a measurement
   cannot be pointed at a file in this repository — `metrics.json`,
   `summary.json`, `manifest.json`, or a row in `04-results.md` — it does not
   appear in any document (Rule 1, Rule 7 item 1).

Changing this rule means editing **two** places: this rule and the status line in
`docs/00-tong-quan.md`.

---

## Rule 13 — every run is recorded, and names its model and its loss

1. **After every training or probe run, regenerate the results workbook**:

   ```bash
   .venv/bin/python scripts/export_runs.py     # -> data/eda_xlsx/ket_qua_chay.xlsx
   ```

   It rebuilds three sheets from the sources — `00_Runs` (every run, every
   column), `01_Phan_Lop` and `02_Luong` (the two report tables). Run it in
   `.venv`; `.venv-dl` has no `openpyxl`.

2. **The workbook is derived, never authored.** Its sources are
   `docs/04-results.md` (the append-only log) and `artifacts/<run_id>/`. Never
   edit the `.xlsx` by hand: the next regeneration silently discards the edit,
   and a hand-typed number has no source (Rule 7 item 1). If a number in it is
   wrong, the run is wrong, not the workbook.

3. **Every row must name the model and the loss, with the loss's own
   hyper-parameters** — γ for focal, whether class weighting was on. Two runs
   that differ only in loss are otherwise indistinguishable in the log, which is
   exactly the comparison the classification task is being judged on.

4. **A run's row also carries the device and the library versions**
   (`torch`, `numpy`, `pandas`, `scikit_learn`, from `env.json`). This is not
   bookkeeping: on 2026-09-20 `dl-cat-s2` could not be reproduced, and the cause
   could not be diagnosed because those versions were never recorded. See item 5.

5. **Training runs on `cpu` by default, and that default is load-bearing.**
   Measured 2026-09-20: on `mps`, the same seed and the same code give macro-F1
   anywhere in **0,6012–0,6041** across eight runs; on `cpu` the same
   configuration gives one number every time, at the same wall-clock cost
   (12–16 s). Any run that reports `device=mps` for the dense head is a run whose
   number cannot be checked — treat a difference below **0,003** macro-F1 between
   two such runs as noise, not as a result. `encode.py` may stay on `mps`: its
   output is written to `artifacts/embeddings/` once and reused byte-for-byte.

Changing this rule means editing **three** places: this rule, the "Finished a
training run" row of the Rule 1 table, and item 8 of `TASKS.md` §3.

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
   table), and the matching `docs/bao-cao/` chapter tells the five contents of
   Rule 10.
3. Every number written down traces back to `manifest.json`, `metrics.json`, or a
   row in `04-results.md`.
4. `test` was not touched unless `--confirm-test` was given and it was the final
   step, and nothing wrote to `test.csv` (Rule 5).

## 5. Git

- No `Co-Authored-By: Claude…` trailer, no "Generated with…" line, no mention of
  AI anywhere in the git history.
- Commit only when asked. Never commit `data/`, `artifacts/`, `.venv/`.
