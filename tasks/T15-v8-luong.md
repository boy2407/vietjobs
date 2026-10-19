# T15 — salary on dataset v8 (MinHash), PhoBERT and CafeBERT, Dense and RNN

Detail sheet for new work asked by the author (Nghĩa) on 2026-10-17. Everything runs on a rented vast.ai GPU, **not on the Mac** (author). Rental file: [vast/dot-6-v8-salary.md](../vast/dot-6-v8-salary.md). `test` is never encoded or scored (Rule 5). Salary reads only `*_masked` columns (Rule 3). Written in English.

## 1. Board

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T15.1 | Rent an instance, push code and `splits-minhash-1label`, run `encode` | todo | Nghĩa (rent) + Claude | — | token caches for PhoBERT 256 and CafeBERT 512 exist on the instance; two test files green |
| T15.2 | Dense, 5 seeds (42–46) per encoder, `--device cpu` | todo | Claude | T15.1 | 10 runs `mh-sal-pho-dense-256-s*`, `mh-sal-cafe-dense-512-s*` with `metrics.json`, `best.pt` on the Mac |
| T15.3 | RNN, 1 seed (42) per encoder, `--device cuda` | todo | Claude | T15.1 | runs `mh-sal-pho-rnn-256-s42`, `mh-sal-cafe-rnn-512-s42` with `metrics.json`, `best.pt` on the Mac |
| T15.4 | Pull results, pass the `vast/README.md` checklist, destroy the instance | todo | Nghĩa | T15.2, T15.3 | every `best.pt` opens with `torch.load` |
| T15.5 | Docs: `04-results.md` rows, `ket_qua_chay.xlsx` (Rule 13), `docs/06`, bao-cao ch.4 (Rules 1, 7, 10); score on `ext37k-sal` | todo | Claude | T15.4 | Rule 1 and 7 tables followed; `pytest -q` green |

## 2. Log (append-only)

| Date | Who | What happened |
|---|---|---|
| 2026-10-17 | Claude | Created the sheet, `scripts/vast_run_v8_salary.sh` and `vast/dot-6-v8-salary.md` (`CHƯA THUÊ`). Nothing rented, nothing run. Salary results on v8 are not comparable to any earlier row of `04-results.md`: v8 drops texts with two or more labels and the `dev` is different |
