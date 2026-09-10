---
name: giai-thich
description: "Briefly explain one concept in the VietJobs pipeline — data cleaning, Vietnamese processing, TF-IDF, features, leakage, metrics, regularisation — starting from an everyday analogy and only then naming the term. Use this skill when the user asks what a preprocessing step, feature-engineering choice, or ML concept in this project means or why it exists, invokes /giai-thich, or asks 'tại sao phải ...' about any pipeline stage."
trigger: "Use this skill when the user asks what a preprocessing step, feature-engineering choice, or ML concept in this project means or why it exists, invokes /giai-thich, or asks 'tại sao phải ...' about any pipeline stage."
version: 1
---

# Explain — a concept lookup for the VietJobs pipeline

The person asking is new to ML/DL. The job is to make them **understand**, not to
make yourself look clever. Everyday analogy first, terminology second, and always
say what breaks without the step.

## When to use it

- The user types `/giai-thich <topic>`
- The user asks "tại sao phải ...", "... để làm gì", "... là gì" ("why do we have
  to", "what is it for", "what is it") about a pipeline step
- The user hits an unfamiliar term while reading `docs/` and asks about it

## Mandatory procedure

1. Look the topic up in the **Topic map** below (aliases count too).
2. **`Read` the source file for that topic.** Mandatory, even when you are sure you
   already know it.
3. Only then write the answer, in the five-block template.

Step 2 may not be skipped. This skill **deliberately contains no numbers** — the
numbers live in `docs/`, and `docs/` is updated after every experiment. Answering
from memory is the surest way to hand over a stale number that nobody catches.

## Answer template — five blocks

```markdown
## <Topic name>

**Picture it:** <everyday analogy, 2–3 sentences. Not one piece of jargon.>

**What it is called:** <name the term, tie it to the analogy above. 1–2 sentences.>

**In VietJobs:** <make it concrete, 2–4 lines. Numbers come from the source file you just read.>

**What breaks without it:** <concrete consequence, 1–3 lines.>

**Read deeper:** <link to the source file>
```

Five blocks and stop. Anyone who wants more depth has the link.

If the source file says a step **was removed** because it measured as no help, say
so plainly in the **In VietJobs** block. A removed step is evidence, not something
to be embarrassed about.

## Topic map

The "analogy seed" column is a direction for the example, not a sentence to copy.

### Data

| Type | Also accepts | Read this file | Everyday analogy seed |
|---|---|---|---|
| `cleaning` | `lam-sach`, `clean`, `why-clean` | `docs/nen-tang/01-vi-sao-phai-lam-sach.md` | A contacts app storing "Nguyễn Văn A" and "nguyen van a" as two people |
| `duplicates` | `trung-lap`, `dedup`, `grouping`, `group-id` | `docs/01-data-audit.md` §2 | The same flyer stuck up in 33 places along one street |
| `splitting` | `chia-tap`, `split`, `train-dev-test`, `seed` | `docs/01-data-audit.md` §7 | A mock exam must not share questions with the real one |
| `labels` | `nhan`, `label`, `target`, `salary-mid` | `docs/01-data-audit.md` §5 | "8–12 million" and "negotiable" are two different kinds of information |

### Vietnamese processing

| Type | Also accepts | Read this file | Everyday analogy seed |
|---|---|---|---|
| `vietnamese` | `tieng-viet`, `vitext`, `nine-steps`, `preprocessing` | `docs/02-vietnamese-nlp.md` | The machine counts letter shapes; it does not read meaning |
| `tone-marks` | `dau-thanh`, `nfc`, `unicode`, `hoa-hoà` | `docs/02-vietnamese-nlp.md` §3 steps 1–2 | "hoà" and "hòa" look identical; the machine sees two words |
| `abbreviations` | `viet-tat`, `abbrev`, `nv`, `bhxh` | `docs/02-vietnamese-nlp.md` §3 step 3 | "NV" and "nhân viên" — a person understands, a machine does not |
| `salary-masking` | `che-luong`, `mask`, `mask-salary`, `salary-token` | `docs/02-vietnamese-nlp.md` §3 step 4 | Marking an exam whose answer key is printed in the corner |
| `segmentation` | `tach-tu`, `segment`, `underthesea` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` | Is "nhân viên" one word or two? The space does not say |
| `accent-folding` | `gap-dau`, `fold-accents`, `no-diacritics`, `char-ngram` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` §5 | Reading a sign written without diacritics — a person copes, a machine does not |
| `provinces` | `chuan-tinh`, `province`, `location`, `ha-dong` | `docs/02-vietnamese-nlp.md` §3 step 7 | "Hà Đông" and "Hà Nội" — you know they are one place, the machine does not |
| `stopwords` | `tu-dung`, `stopwords` | `docs/02-vietnamese-nlp.md` §3 step 8 | Deleting "không" from "không yêu cầu kinh nghiệm" |

### Features

| Type | Also accepts | Read this file | Everyday analogy seed |
|---|---|---|---|
| `tf-idf` | `tfidf`, `tf`, `idf`, `bag-of-words`, `tui-tu` | `docs/nen-tang/02-tf-idf-la-gi.md` | A rare word in a book is worth looking up; the word "and" is not |
| `ngram` | `bigram`, `unigram`, `n-gram` | `docs/nen-tang/03-ngram-va-ranh-gioi-tu.md` §1 | "bánh mì" is nothing like "bánh" and "mì" standing apart |
| `features` | `dac-trung`, `feature`, `blocks`, `column-transformer` | `docs/archive/05-dac-trung-tfidf.md` (closed track) | Filling in an application form — one field per item, not everything on one line |
| `tfidf-params` | `tham-so-tfidf`, `min-df`, `max-df`, `sublinear`, `max-features` | `docs/archive/05-dac-trung-tfidf.md` §4 (closed track) | The dials on a filter: too coarse and too fine are both wrong |
| `sparse-matrix` | `ma-tran-thua`, `sparse`, `dimensions`, `p-n`, `curse` | `docs/nen-tang/04-ma-tran-thua-va-so-chieu.md` | A phone book with two hundred thousand numbers when you only call five |
| `semantic-vectors` | `vector-ngu-nghia`, `phobert`, `embedding`, `subword` | `docs/nen-tang/09-vector-ngu-nghia.md` | Two job ads with no words in common that mean the same job |

### Models and measurement

| Type | Also accepts | Read this file | Everyday analogy seed |
|---|---|---|---|
| `two-tasks` | `hai-mo-hinh`, `bai-toan`, `classification`, `salary` | `docs/06-baseline-dl.md` + `docs/07-bai-toan-luong.md` | Guessing the occupation and guessing the pay are two questions about one CV |
| `metrics` | `metric`, `macro-f1`, `accuracy`, `baseline`, `noise` | `docs/nen-tang/06-do-luong-va-baseline.md` | A class of 40 with 39 top students and 1 struggling — "97 % excellent" hides that one |
| `regularisation` | `chinh-quy-hoa`, `c`, `alpha`, `ridge`, `overfit` | `docs/nen-tang/07-chinh-quy-hoa.md` | Rote-learning last year's paper: 10/10 at home, 3/10 in the exam hall |
| `comparison-table` | `bang-so-sanh`, `paired-bootstrap`, `winners-curse` | `docs/nen-tang/08-doc-mot-bang-so-sanh.md` | Twenty coin flippers — the best one is not skilled, just lucky |

### Correctness

| Type | Also accepts | Read this file | Everyday analogy seed |
|---|---|---|---|
| `leakage` | `ro-ri`, `leak`, `resolve-column` | `docs/nen-tang/05-ro-ri-du-lieu.md` | Acing an exam because you saw the paper — a high mark, no knowledge |
| `ablation` | `bo-buoc`, `which-steps-to-keep` | `docs/02-vietnamese-nlp.md` §5 | Cooking a dish with one spice removed at a time to see which one matters |

## Called with no argument

Print the topic map in its five groups, one short line per topic, then ask which
one they want. Do not choose for them.

## A typo, or no match

Suggest the 2–3 nearest topics and stop. Do not guess a topic and answer anyway.

If it is a general ML concept the project **does not use** (cross-validation,
dropout, attention, say): say plainly that the project does not use it, answer
briefly in the same five-block template, and **leave "Read deeper" empty** — do not
invent a `docs/` link that does not exist.

## Follow-up questions

If the user asks a follow-up within the same topic, drop the five-block template
and answer their actual question. The template is an opening, not a cage.

## Avoid

- **Quoting numbers from memory.** Read the source file first. The numbers in
  `docs/` change after every experiment; the numbers in your head do not.
- **Using jargon in the "Picture it" block.** It defeats the entire purpose of that
  block. If you cannot phrase it without jargon, you do not understand it well
  enough yet.
- **An analogy that sounds good but is wrong underneath.** The analogy must **break
  where the concept breaks**. A pretty analogy that leads the wrong way is worse
  than none.
- **Dropping the "What breaks without it" block.** It is the block readers remember
  longest.
- **A long answer.** Five blocks and stop.
- **Hiding failures.** Three preprocessing steps in this project measured as no help
  and were removed. Say so, with the numbers. That is the most instructive part.
