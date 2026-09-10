[← Sparse matrices](04-ma-tran-thua-va-so-chieu.md) · [Background](00-index.md) · [Metrics and baselines →](06-do-luong-va-baseline.md)

# 5. Data leakage

Leakage is when the model sees — directly or indirectly — information it would not
have at real prediction time.

What makes leakage more dangerous than any other kind of bug:

> **It raises no error. It makes the score LOOK BETTER.**

An ordinary bug crashes the program or makes the score worse — you notice it
straight away. Leakage makes everything look wonderful. You only find out when the
model goes into the world and breaks, and by then the report is already printed.

---

## Three kinds of leakage in this project

### Kind 1 — the label is inside the input

**The task:** predict the salary from the posting's content.
**The problem:** a great many postings **state the salary figure right in the
description and the benefits**.

Measured:

| Text field | Rows restating the salary figure |
|---|---|
| benefits | **5,081 (10.65 %)** |
| description | 250 (0.52 %) |
| requirements | 103 (0.22 %) |
| title | 121 (0.25 %) |

Without handling, the model only has to learn one rule: *"find the number next to
the word 'salary' and print it"*. The R² looks great. But it is not **predicting**
anything — it is **copying**. On a real posting that does not state a salary, it is
useless.

**How it is blocked — mask, do not delete:**

```
"Lương 15 - 22 triệu/tháng"   →   "Lương <SALARY>/tháng"
```

The figure disappears, the word "Lương" (salary) **stays**. That is deliberate:
*"this posting mentions pay"* is a legitimate signal; *"how much the pay is"* is the
answer that must be hidden.

**The hardest part is telling money from counts:**

| Sentence | Mask? | Why |
|---|---|---|
| "thu nhập 15 - 22 triệu" (income 15–22 million) | ✅ | money |
| "40 triệu người dùng" (40 million users) | ❌ | counting people |
| "500 triệu đồng doanh thu" (500 million in revenue) | ❌ | counting revenue |
| "thưởng 5tr mỗi quý" (5M bonus per quarter) | ✅ | money |

The comment in `tests/test_vitext.py:210` says it plainly: *"this was a real bug"*.
Over-masking is also a way of destroying data.

### Kind 2 — the same posting is in both train and test

Employers repost an advert many times with a few words changed. Measured:
**12,808 rows (26.8 %)** are reposts; the 47,707 rows contain only **34,899** truly
distinct groups.

Splitting randomly **by row** almost guarantees that a posting has one copy in train
and a near-copy in test. The model scores points purely by **memorising** — and that
score says nothing about how it handles a posting it has never seen.

**How it is blocked:** hash the content → `group_id`, and force **the whole group**
into one split.

```python
straddling = int((df.groupby("group_id")["split"].nunique() > 1).sum())
assert straddling == 0, f"{straddling} groups straddle splits — split is leaking"
```

That `assert` may never be disabled. The current result: **0 straddling groups**.

> Note: reposts are **not deleted** — they are still real data. Only the **way the
> split is made** changes. See
> [01-data-audit.md §2](../01-data-audit.md#2-de-duplication--two-layers-two-different-purposes).

### Kind 3 — touching the test set repeatedly

This kind is the subtlest, because it is not in the code — it is in the **workflow**.

Every time you score test → see a low number → adjust the model → score test again,
information flows from test into a design decision. Do it ten times and test is no
longer an unbiased estimate — it has become a second dev set, and you have nothing
left to tell you how good the model really is.

**How it is blocked — by rule, not by code:**

| Split | Used for | How often it may be touched |
|---|---|---|
| `train` | Training | Unlimited |
| `dev` | Model selection, hyper-parameters, preprocessing steps | Unlimited |
| `test` | Reporting the final number | **Exactly once**, at the end |

`train.py` refuses `--eval test` without the `--confirm-test` flag. That flag exists
so that touching test is **a deliberate act**, not a default.

Across the project's 27 experiments, exactly **one** row was scored on test:
`cat-FINAL-svm-C0.02-test`.

---

## The defence mechanism: a single door

Kind 1 is blocked by an architectural rule, not by carefulness:

**Every text column exists in two copies — raw and masked — and only ONE function
decides which task reads which.**

```python
def resolve_column(name, *, task, segmented):
    col = _MASKABLE[name] if (task in C.MASKED_TASKS and name in _MASKABLE) else name
    ...
```

Why concentrate it in one function instead of scattering `if` statements: **one
door can be tested.** `tests/test_no_leak.py` needs a single line to guard the lot:

```python
assert set(F.source_columns(ct)) & F.UNMASKED_COLUMNS == set()
```

---

## The most expensive lesson: a test only protects what you thought of

While writing this documentation, a careful read of `features.py` turned up:

**`soft_skills_text` and `qualifications_text` reach the salary model unmasked.**
Neither column has a `_masked` copy, and neither is **in `UNMASKED_COLUMNS`** — so
`tests/test_no_leak.py` passes green without ever checking them.

Whether those two columns restate salary figures is still unknown, because
`scripts/measure_vitext.py` does not measure them. The work item is recorded in
[09-lo-trinh.md — Priority 0b](../archive/09-lo-trinh-ml.md#ưu-tiên-0b--xong-không-phải-rò-rỉ).

This is a living example of the most memorable thing in this note:

> A leak test does **not** prove there is no leak.
> It only proves that the leak paths **the test's author thought of** are blocked.
>
> A list like `UNMASKED_COLUMNS` is a list that **permits omissions**: forget to add
> a column and the test silently skips it. A safer design would invert the default —
> list the columns that are **allowed** to be read, and reject everything else.

---

## A checklist for when you suspect a leak

1. **Unusually high score?** Suspect first, celebrate later. Leakage always looks
   like an excellent model.
2. **Which feature is strongest?** If the top feature is something you *cannot have*
   at real prediction time, that is a leak.
3. **Does this feature exist at prediction time?** The golden question for every
   column.
4. **Are there near-duplicates between train and test?** Hash the content and count.
5. **How many times have you touched test?** If you have to think about it, too many.
6. **Which columns are NOT in the test's checklist?** — the lesson above.

---

## Back to the project itself

- [03-protocol.md](../03-protocol.md) — the full train/dev/test contract
- [01-data-audit.md](../01-data-audit.md#3-four-copies-of-every-text-column) — the four copies of every column
- [05-dac-trung-tfidf.md](../archive/05-dac-trung-tfidf.md#1-resolve_column--cửa-duy-nhất) — `resolve_column` and the unpatched hole
- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — step 4 (salary masking) and step 9 (the grouping key)
