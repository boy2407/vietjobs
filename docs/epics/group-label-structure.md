# Epic: group-label-structure

Same posting text carries several labels in `train` but almost never in `dev`. Explain it, measure it, and test whether training on it helps. Source: dataset diagnosis of `category`, 2026-10-17; board rows T2.6–T2.8 in `TASKS.md`.

- `split-group-structure` · feature · depends_on: [] · Explain why `train` and `dev` differ in group structure, so the author knows whether `dev` macro-F1 is comparable with what the model learns from.
- `top1-in-label-set` · feature · depends_on: [`split-group-structure`] · Report "top-1 inside the group's label set" next to macro-F1, so a single-label score is not read as the whole quality when a posting has several valid labels.
- `group-soft-targets` · feature · depends_on: [`split-group-structure`] · Train on each group's label distribution instead of one hard label, so identical text no longer enters the loss with contradicting answers (no row removed or relabelled).
