[← Background](00-index.md) · [What TF-IDF is →](02-tf-idf-la-gi.md)

# 1. Why clean the data at all

The familiar answer is "garbage in, garbage out". That answer is **not wrong but
useless** — it does not say which step is worth doing and which is ritual.

There are two real reasons, both concrete and both measurable.

---

## Reason 1 — the machine counts characters, it does not read them

This is the hardest thing for a newcomer to accept, so let us say it plainly:

> The model does not understand Vietnamese. It **counts character sequences**.

To `TfidfVectorizer`, `"hoà"` and `"hòa"` are two different character sequences,
so they are **two separate features**, exactly as `"hoà"` and `"máy xúc"`
(excavator) are two separate features. The machine has no way of knowing the first
two are the same word.

### What happens when one concept gets split in two

Suppose the word "hoà" appears in 1,000 postings, written two ways:

| | **Without** normalisation | **With** normalisation |
|---|---|---|
| Dimension `hoà` | 600 postings | **1,000 postings** |
| Dimension `hòa` | 400 postings | — |

Three consequences, each heavier than the last:

**a. Each half is weaker.** The model learns a weight per dimension. A dimension
that sees 600 examples learns less than one that sees 1,000.

**b. The two halves learn two different weights** — possibly of opposite sign, if
the `hòa` spelling happens to be more common in some sector. The model learns a
**spurious** correlation that comes from typing habits, not from content.

**c. At prediction time, a new posting activates only one of the two dimensions.**
If the user types the less common style, the model uses the weaker dimension — and
answers worse, entirely silently.

### How big this is in the VietJobs corpus

All real measurements, from `python scripts/measure_vitext.py`:

| Form of "dimension splitting" | Scale |
|---|---|
| Uncomposed Unicode letters (`ế` written as 2–3 code points) | **529 rows (1.11 %)** |
| Descriptions using both tone-mark placements `hoà`/`hòa` | 76.6 % use one style, 30.3 % the other — **most postings mix both** |
| Descriptions with unexpanded abbreviations (`NV`, `BHXH`, `CSKH`) | **3,436 descriptions (7.2 %)** |
| Fragmented locations (`hà đông` cannot be linked to `hà nội`) | 984 strings → only **265** after normalisation; **7,481 rows (15.7 %)** change value |
| Titles written entirely **without diacritics** | **2,148 titles (4.50 %)** |

Each number looks small on its own. But they stack, and each one splits precisely
the **most common** words — the ones the model relies on most.

---

## Reason 2 — stopping the model from seeing the answer

This reason is far more important than reason 1, and completely different in kind.

Reason 1 is about **performance**: skip it and the model is slightly worse.
Reason 2 is about **correctness**: skip it and every number you report is **wrong**,
and wrong in the flattering direction.

This project has two leak paths, both blocked during cleaning:

**a. The answer sits inside the input itself.** Task 2 predicts the salary. But
**10.65 %** of rows restate the salary figure right there in the *benefits* field.
Without masking, the model just reads that figure back — a pretty R² in the report,
broken in the world.

**b. The same posting is in both train and test.** Employers repost adverts many
times: **12,808 rows (26.8 %)** are reposts. Split by row and the model scores high
on test purely by **memorising** — and that score says nothing about generalisation.

The full detail is in [note 5 — data leakage](05-ro-ri-du-lieu.md).

---

## The consequence: two kinds of preprocessing step

The two reasons above yield a very useful classification, and it is the backbone of
the whole project:

| Kind | Purpose | Proved by | Removable |
|---|---|---|---|
| **Performance step** | Raise the score | An ablation row in [04-results.md](../archive/04-results-ml.md) | **Yes** — no improvement, no keeping |
| **Correctness step** | Make the measured number meaningful | Cannot be proved by a score | **Never** |

Of the nine Vietnamese processing steps, four are of the second kind (NFC,
tone-mark normalisation, salary masking, the grouping key) and **none** of them
raises the score appreciably. They stay anyway. Three of the first kind measured as
no help and **have been switched off**.

This is what separates a disciplined pipeline from a long ritual one: not "many
steps", but **knowing which reason each step is there for**.

---

## The opposite warning: over-cleaning also breaks things

Cleaning is not better the more you do. Three real examples from this project:

**Stripping Vietnamese diacritics.** `strip_accents` is a common default for
English TF-IDF. In Vietnamese it destroys: `má` (mother), `mà` (but), `mả` (grave),
`mã` (code), `mạ` (to plate metal) are **five different words**. Stripping the
diacritics merges five meaningful dimensions into one meaningless one.

**Mechanical stopword removal.** Dropping the word "không" turns *"không yêu cầu
kinh nghiệm"* (no experience required) into *"yêu cầu kinh nghiệm"* (experience
required) — **the opposite meaning**. This project's stopword list deliberately
spares `không`, `chưa`, `trên/dưới`, `tối thiểu`.

**Careless abbreviation expansion.** Expanding `TP` into `trưởng phòng` (department
head) inside `"TP HCM"` (Ho Chi Minh City) manufactures a **false** signal — the
model would think every posting in Saigon is hiring a department head. **A wrong
expansion is worse than none**, so ambiguous abbreviations are expanded only when a
context pattern matches.

The rule that follows: **every cleaning operation is a merge of information, and a
merge is an irreversible loss.** Merge only when you are certain the two things
being merged really are one.

---

## Back to the project itself

- [01-data-audit.md](../01-data-audit.md) — how the cleaning is actually done
- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — the nine steps, and which ones survived
- [note 5 — data leakage](05-ro-ri-du-lieu.md) — reason 2, in detail
