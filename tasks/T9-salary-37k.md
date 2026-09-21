# T9 — salary column for the external data (VietJobs-37K): read a salary for every posting from its text

Detail sheet for board section **T9** of [TASKS.md](../TASKS.md). Split out under TASKS.md §3 Rule 7. Same rules as the board:
tasks are written and assigned by the author (Nghĩa); an agent works only the task it is
assigned, fills *Evidence* before `done`, and appends to the log **here** and to
[TASKS.md](../TASKS.md) §5 in the same session. Status vocabulary: `todo` · `doing` ·
`blocked (why)` · `done` · `dropped (why)`. Written in English (AGENTS.md Rule 6).

**Scope: external data only.** Everything here runs on `data/external/vietjobs37k/`. The project's own dataset `data/raw/VietJobs.csv` / `data/processed/` already has platform-given salary columns and is not read, rewritten or re-split by any T9 task. "VietJobs-37K" is the Zenodo release (PROVENANCE.md), a different dataset from `VietJobs.csv` despite the name.

Files this sheet is about: `data/external/vietjobs37k/ext37k.csv` (all 38,274 rows),
`ext37k-sal.csv` (labelled rows, deduplicated), `scripts/build_ext37k.py`,
`src/vietjobs/external.py::extract_salary`, `tests/test_external.py`.

---

## 1. Goal and current state

**Goal.** The 37K release has no salary field. For each of the 38,274 postings, read the advertised pay from `job_title` + `description` + `requirements_text` whenever the text states one, so that `salary_min` / `salary_max` / `salary_mid` are filled for as many rows as the text allows — and every empty row has a stated reason. Same column meaning as `VietJobs.csv`: the range the ad advertises, in triệu VND / month.

**Where we are (measured 2026-09-18).** Round 1 (`external.extract_salary`, log 2026-09-17/18: pay word → figure within 40 chars, ladders chained) fills 5,289 rows (13.8 %). Of the 32,985 empty rows, 3,899 still carry a money figure: 1,085 also contain a pay word (refused by gap, unit or range), 721 have the figure right after a section header with no pay word (`quyền lợi` / `chế độ` / `chính sách` / `phúc lợi`; 710 of them are one careerviet bank template `3. Quyền lợi- 3.000.000 - 16.000.000 VNĐ, theo năng suất lao động` reposted per branch), 1,766 have a figure and no salary signal (ăn trưa, gửi xe, KPI, ngân sách…). The remaining ~29,000 rows state no figure at all — the ceiling for this task is about 9,200 rows (24 %), not 38,274. Out of scope: training on the result (text is unmasked; needs `mask_salary` first, Rule 3/4).


## 2. Board

| ID | Task | Status | Owner | Depends on | Done when (evidence) |
|---|---|---|---|---|---|
| T9.1 | **Coverage audit** — `scripts/build_ext37k.py --audit`: put every row in exactly one bucket: `read` (round 1) · `pay word + figure, refused` (sub-bucket by reason: gap > 40 / unit / out of [2, 200] / hourly-annual) · `header + figure` · `figure only` · `negotiable` (`thoả thuận`, `thỏa thuận`, `cạnh tranh`, `theo năng lực` with no figure) · `no figure`. Counts by `source`. This table is the baseline every later step is measured against | done | Claude | —| Bucket table in [docs/10](../docs/10-danh-gia-ngoai.md) §3.1 and in §4 below, before/after columns. `scripts/build_ext37k.py --audit`, run 2026-09-18: the 12 buckets sum to 38,274 exactly. Largest refusal is `payword_unit` 2,261 — reading 30 of them showed it is almost entirely "Lương tháng 13" / "lương 2 lần năm", i.e. correctly refused, so no rule was written for it |
| T9.2 | **Extractor round 2** — in `external.extract_salary`, one rule per bucket, each with a test: (a) section headers (`quyền lợi`, `phúc lợi`, `chế độ`, `chính sách`, `benefit`) anchor a figure like a pay word does, same `_GAP_BREAK` stop words; the bank template must read 3–16tr; (b) widen the pay-word gap / unit / range rules only where a 30-row read of that sub-bucket says it is safe — decision and count per rule in the log; (c) `figure only`: read a range `X - Y triệu` / `X - Y tr` when the clause has no `_GAP_BREAK` word — refuse single figures there. Precision stays ahead of recall: a rule that reads a lunch allowance as pay is rolled back | done | Claude | T9.1| 10 new tests (one accept + one refuse per rule, plus the tone guard) in `tests/test_external.py` (46 in that file); `pytest -q` was 156 passed when this ran, 132 after the author removed the ML branch on 2026-09-19 — no T9 test was touched. Rows read 5,289 → 6,328 (+1,041, −2). Per rule in the file: `header` 717 · `dong_range` 223 · `m_unit` 194 · `bare_range` 116 · `payword` 5,078. The 2 lost rows are both correct refusals (annual pay `130-160tr năm`; "lương doanh thu"). Reading 46 new rows by hand found 3 false positives, all fixed before this row was written: `Gói Benefit 18M năm`, `Khoảng 100 triệu năm`, `ngân sách 100 - 200 triệu` |
| T9.3 | **Explain every empty row** — new column `salary_note` ∈ {`negotiable`, `figure_unread`, `no_figure`} on the rows still without a salary, so "no salary" is never silent; `salary_text` keeps the negotiable phrase when there is one | done | Claude | T9.2| `salary_note` in `ext37k.csv`: `no_figure` 23,850 · `figure_unread` 4,850 · `negotiable` 3,246. Verified on the file: 0 rows with `salary_disclosed=0` and an empty note, 0 rows with a salary and a note. `test_salary_note_explains_every_empty_row` (4 cases) |
| T9.4 | **`template_id`** — one id per identical `description` in both `ext37k.csv` and `ext37k-sal.csv` (title ignored: templates differ only by branch). Needed before anyone splits this data: 859 / 5,153 rows already repeat a description, and the bank template alone becomes ~13 % of the labelled file after T9.2 | done | Claude | T9.2| `template_id` in both files, from `external.template_ids`. Full file: 34,949 groups, largest 99, 4,820 rows in a group > 1. `ext37k-sal.csv`: 4,639 groups, largest 98 (1.6 %). `test_template_ids_group_reposted_ads`. **Key correction:** the first attempt keyed on `description` alone and put the bank template in 711 separate groups — 710 of its rows have an *empty* description and repeat in `requirements_text`. The key is now the whole body. Limitation kept and written up (§4, docs/10 §3.3): the bank ad still spans 34 groups because its requirements vary per branch, so a group split on `template_id` does not fully isolate it |
| T9.5 | **Human review** — 200 rows sampled at random from the labelled rows after T9.2, stratified by `source` and by the rule that read them, each marked `correct` / `wrong` / `ambiguous` with a note, in `data/external/vietjobs37k/review-salary.csv`. **Human only** — a labelling decision (Rule 7) | done | Nghĩa | T9.2| `review-salary.csv`, 199 rows, all verdicts filled by the author 2026-09-19. **Precision 97.0 % on the 165 rows that carry distinct evidence** (160 correct · 2 wrong · 3 ambiguous, ambiguous counted against). Per rule, all clear the 90 % bar: `payword` 100 % (40/40) · `header` 100 % (8/8) · `bare_range` 97.4 % (37/38) · `m_unit` 95.0 % (38/40) · `dong_range` 94.9 % (37/39). **No rule rolled back.** Claude re-checked the review on request: read 24 `correct` rows in full context (all held) and disputed two verdicts, both accepted by the author — id 19184 `ambiguous` → `wrong` (2.000.000 is a KPI allowance, not the low end of the pay), id 15489 `correct` → `ambiguous` (read 30–30, dropped the 8–15M base, same shape as id 10506) |
| T9.6 | **Rebuild and report** — one run of `scripts/build_ext37k.py` rewrites `ext37k.csv` + `ext37k-sal.csv`; the T9.1 audit re-run gives before/after coverage; PROVENANCE.md, docs/10 §4 and the Backlog "cross-check for `salary`" item updated | done | Claude | T9.1–T9.5| Both files rebuilt in one run: `ext37k.csv` 38,274 rows / 6,328 with a salary (16.5 %) / median 15.0tr; `ext37k-sal.csv` 6,161 rows. Coverage table before/after in §4 and docs/10 §3.1; PROVENANCE.md has its own section on the three derived files, the review result and the three caveats; docs/08 rows updated for `external.py`, `tests/test_external.py` and the new `scripts/build_ext37k.py`. `pytest -q` green (`tests/test_external.py` 46 passed) |

## 3. Rules that bind every task here

1. Precision before recall: a rule that reads an allowance, bonus, KPI target or budget as pay is rolled back, whatever it recovers.
2. `salary_min` / `salary_max` keep the one meaning they have in `VietJobs.csv`: the range the ad advertises, in triệu VND / month (author's decision 2026-09-18, log). Hourly, daily, annual and USD figures are refused, not converted.
3. Every rule is one commit with one accept test and one refuse test; the count of rows it recovers goes in its *Evidence* cell.
4. The text stays unmasked in these files. Nothing in T9 trains a model; that needs `mask_salary` first (TASKS.md Rule 3/4) and is its own task.
5. Human review (T9.5) is the only ground truth; agent spot-checks are evidence for a rule, not for the file.

## 4. Coverage, before and after

`scripts/build_ext37k.py --audit`, both columns measured 2026-09-18. Every row
falls in exactly one bucket, so the coverage number reads on: of the rows
without a salary, how many state a figure we refuse, and how many state none.

| Bucket | Round 1 | Round 2 | What it is |
|---|---|---|---|
| `read` | 5,289 | **6,328** | A salary was read |
| `payword_unit` | 2,261 | 1,846 | Pay word + figure, unit unreadable — nearly all "Lương tháng 13" |
| `payword_gap` | 909 | 1,723 | Pay word, figure more than 40 chars away |
| `payword_break_word` | 435 | 676 | A stop word (thưởng, phụ cấp…) sits between |
| `payword_not_monthly` | 0 | 179 | Hourly / daily / annual pay |
| `payword_far` | 101 | 73 | Pay word and a figure, not in the same clause |
| `payword_loose_range` | 88 | 53 | "9 15 triệu", no dash, not enough evidence |
| `payword_out_of_range` | 29 | 18 | Outside [2, 200] triệu |
| `header_figure` | 717 | 0 | Figure after a section header — round 2 reads all of these |
| `figure_only` | 376 | 282 | A figure with no anchor at all |
| `negotiable` | 3,431 | 3,246 | "Thỏa thuận", "cạnh tranh", no figure |
| `no_figure` | 24,638 | 23,850 | States no figure |
| **Total** | 38,274 | 38,274 | |

Known limitations, measured and **not** fixed:

1. **`template_id` groups exact bodies only.** The bank ad (711 rows, 694 of
   them in `ext37k-sal.csv` = 11.3 % of that file) varies its requirements per
   branch and so sits in 34 groups. A group split on `template_id` still leaks
   it. A fuzzy key (Jaccard) would be needed.
2. **`payword_gap` grew 909 → 1,723** — not a regression: section headers are
   anchors now, so rows that used to have no anchor at all are bucketed by the
   rule that refused them instead.
3. **`bare_range` is the least safe rule.** It reads a range with no pay word
   anywhere ("Kế Toán Trưởng (45 - 60 Triệu)"). One false positive survived the
   46-row spot check by the agent: a tax refund, "tiền thuế từ 80-100 triệu".
   The review found the same row and nothing worse — 97.4 %, the rule stays.
4. **The sampler's dedup was wrong, and the review exposed it.** `review_sample`
   deduplicated on `template_id` alone, so 33 of the 39 `header` rows came back
   as the same bank ad (it holds 33 template ids while stating one salary). The
   sampler now deduplicates on `template_id` **and** `salary_text`. Two
   consequences: the 199-row sample is really 165 rows of distinct evidence, and
   `header` turns out to have only **8 distinct snippets in the whole corpus**
   (710 of its 717 rows are that one bank ad) — all 8 were reviewed, so its
   100 % is complete coverage, not a thin sample. A rule that adds 717 rows on
   the strength of 7 advertisers is worth knowing about before anyone trains on
   this file.

## 5. Log (append-only, newest last)

| Date | Who | What happened |
|---|---|---|
| 2026-09-18 | Claude | Sheet opened at the author's request; T9 section moved here from TASKS.md unchanged. Board only |
| 2026-09-18 | Claude | **T9.1–T9.4 done, T9.5 ready for the author, T9.6 blocked on it.** `_scan` factored out of `extract_salary` so the audit and the extractor walk the same anchors (one code path, TASKS.md Rule 4) and every refusal carries a code; `audit_salary` + `BUCKETS` put all 38,274 rows in one bucket each. Four widening rules (`header`, `m_unit`, `dong_range`, `bare_range`), each with an accept and a refuse test: 5,289 → 6,328 rows (+1,041, −2, both losses correct refusals). Three false positives found by reading 46 new rows and fixed before the rules were called done — `Gói Benefit 18M năm`, `Khoảng 100 triệu năm` (de-identification stripped the "/" in "triệu/năm", so the bare period word now refuses), `ngân sách 100 - 200 triệu` (budget words added to `_GAP_BREAK`). Two self-inflicted bugs found and fixed while measuring: widening `_NOT_MONTHLY` to bare period words also matched toneless "nam" and dropped two real salaries ("Thu nhập 8tr Nam nữ…"), and `template_ids` keyed on `description` alone put the bank template in 711 separate groups because its description is empty. New columns `salary_note`, `salary_rule`, `template_id`; `review-salary.csv` (199 rows, 40 per rule) written for T9.5. `pytest -q` 156 passed (was 140; 132 from 2026-09-19 after the author removed the ML branch — no T9 test touched). Docs: docs/10 new §3 (+ old §3–§5 renumbered), PROVENANCE.md new section, docs/08 rows for `external.py` (278 → 549 lines, count was already stale), `tests/test_external.py` (13 → 46) and a new row for `scripts/build_ext37k.py` |
| 2026-09-19 | Nghĩa + Claude | **T9.5 and T9.6 done — T9 closed.** The author reviewed all 199 rows. Claude re-checked on request: read 24 `correct` rows in full context (all held) and disputed two verdicts, both accepted — id 19184 `ambiguous` → `wrong`, id 15489 `correct` → `ambiguous`. **Precision 97.0 % on 165 rows of distinct evidence** (`payword` 100 % · `header` 100 % · `bare_range` 97.4 % · `m_unit` 95.0 % · `dong_range` 94.9 %); every rule clears its 90 % bar, none rolled back. Two findings the review produced: (1) the sampler deduplicated on `template_id` alone, so 33 of 39 `header` rows were the same bank ad — fixed, it now also deduplicates on `salary_text`, and `--review N --rule NAME` can top up one rule while skipping ids already reviewed; (2) `header` has only 8 distinct snippets in the whole corpus, so a top-up was impossible and unnecessary — all 8 are reviewed. Claude first proposed sampling 30 more `header` rows and withdrew it once the count was measured. Docs: docs/10 new §3.4 with the precision table and the sampling caveat, PROVENANCE.md caveat 1 rewritten from "chưa ai soát" to the measured number. `tests/test_external.py` 46 passed; the suite total moved 156 → 132 because the author removed the ML branch the same day, no T9 test touched |
