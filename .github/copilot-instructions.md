# Copilot instructions — VietJobs

The full rules are in [`AGENTS.md`](../AGENTS.md). Read that file before proposing
a change. In summary:

- **Context:** occupation classification (16 classes) and salary estimation from
  Vietnamese job postings. Module layout: `docs/08-ma-nguon.md`. Project map:
  `docs/00-tong-quan.md`.
- **Living documentation:** every real change comes with an update to the matching
  note in `docs/` (the mapping table is in `AGENTS.md`).
- **Complete report:** every finished task is described in `docs/bao-cao/` with
  what was run, the model, data handling, Vietnamese processing, and the
  preprocessing analysis behind it (Rule 10).
- **Only what was done:** the report and `docs/` describe work that was run,
  once and without excess; plans live only in `TASKS.md` and
  `docs/09-lo-trinh.md` (Rule 11).
- **Experiment protocol:** `SPLIT_SEED = 20260826` is immutable; the `test` split
  is touched once, behind `--confirm-test`; `docs/04-results.md` is append-only.
- **Leak prevention:** the two salary tasks read only `*_masked` columns, decided
  in `features.resolve_column`, guarded by `tests/test_no_leak.py`.
- **Train = serve:** `predict.build_frame` must match `dataset.clean`.
- **English Markdown:** all `.md` in this repository is written in English
  (Rule 6). Vietnamese stays only in quoted data and examples.
- **Testing:** `pytest -q` green before a commit. A new preprocessing step needs a
  test and an ablation row, otherwise it is removed.
- **Git:** no AI co-author trailer, no "Generated with…" line.
