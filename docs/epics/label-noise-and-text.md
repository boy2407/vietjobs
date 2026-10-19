# Epic: label-noise-and-text

Label noise, Vietnamese text handling and salary follow-ups for the CafeBERT path. Source: author's user stories US1–US12, board rows T14.1–T14.12 in `tasks/T14-nhieu-nhan-va-van-ban.md`. No story reads or transforms `test` (Rule 5); no label or split change without the author's explicit approval (Rule 2).

## A — labels (category)

- ~~`label-noise-ceiling`~~ (dropped 2026-10-17) · feature · depends_on: [] · Estimate the label-noise ceiling on `dev`, so the author knows how much of the macro-F1 gap a better model could still buy back. (US1 · T14.1)
- `hand-label-dev-errors` · feature · depends_on: [] · Let the author hand-label 150 `dev` errors as model error / wrong original label / ambiguous, so the ceiling rests on a human verdict. Human only. (US2 · T14.2)
- `class-centroid-cosine` · feature · depends_on: [] · Measure cosine between the centroids of three confused class pairs, so each pair gets a verdict: taxonomy error or model error. (US3 · T14.3)
- `junk-class-decision` · feature · depends_on: [`hand-label-dev-errors`, `class-centroid-cosine`] · Record the author's decision on `nhom_nghe_khac` (F1 0.000, n = 25 on `dev`), so the class is no longer an unexplained zero. Human decision. (US4 · T14.4)

## B — Vietnamese text, CafeBERT path (gated on `class-centroid-cosine`)

- `strip-company-intro` · feature · depends_on: [`class-centroid-cosine`] · Test whether removing company-intro boilerplate from the CafeBERT input raises macro-F1. (US5 · T14.5)
- `field-order-512` · feature · depends_on: [`class-centroid-cosine`] · Find which field order loses the least under the 512-token limit, and how much of each field is truncated. (US6 · T14.6)
- `cafebert-preprocess-ablation` · feature · depends_on: [`class-centroid-cosine`] · Give NFC, tone marks and abbreviations one on/off ablation row each on the CafeBERT path. (US7 · T14.7)

## C — salary

- `salary-province-field` · feature · depends_on: [] · Add a province field to the `salary` input (`*_masked` only), aiming at the >20M quintile where the residual is −8.97. (US8 · T14.8)
- `salary-model-comparison-docs` · feature · depends_on: [] · Record CafeBERT+RNN vs Dense and vs PhoBERT+RNN on `salary`, each p-value traced to its file. Documentation only. (US9 · T14.9)

## D — consistency and product

- `cafebert-rnn-mean-reconcile` · feature · depends_on: [] · Reduce the two CafeBERT+RNN `category` means (0.6152 vs 0.6097) to one sourced number. (US10 · T14.10)
- `top3-product-result` · feature · depends_on: [T5.1 — not a story yet] · Report top-3 accuracy 0.9373 reproduced by the inference entry point. (US11 · T14.11)

## Not a story

- US12 · T14.12 (upkeep after each story: docs, bao-cao, `export_runs.py`) is the definition of done of every story above (Rules 1, 7, 10, 13), so it becomes an AC line in each story and not a slug of its own.
