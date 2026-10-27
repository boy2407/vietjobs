---
name: tach-task
description: "Turn a new idea or experiment for VietJobs into small tasks on TASKS.md: one run or one comparison per task, the cheapest one first, each with its own run_id, baseline and evidence. The author approves the rows before they are written. Use this skill when the author proposes something new to try, such as 'thử X xem có hơn Y không', 'chạy thử mô hình Z' or 'thêm thí nghiệm', asks to split or plan a piece of work, or invokes /tach-task."
trigger: "Use this skill when the author proposes a new experiment, model, encoder, loss or preprocessing change to try, asks to plan or split work into tasks, or invokes /tach-task."
version: 1
---

# Tách task: one idea becomes several small tasks

Large tasks hide their cost and their result. An example is "try CafeBERT", which
bundles an encode, two heads, an 18 GB cache and two comparisons into one row.
When one part fails, the row stays `in progress` forever, and the part that did
finish has no evidence cell of its own. Small tasks fix this. Each one can end,
each one has one number, and the author can stop after any of them.

This skill **only proposes** tasks. The author creates and assigns them
(`TASKS.md` "How to use it"). No row is written before the author says yes.

## Step 1: read before splitting

1. Read `TASKS.md` §1, §2 and the last few rows of §5.
2. Find the run the idea is compared against: its `run_id`, its number and its
   config (`artifacts/<run_id>/config.json`). If there is none, the first task
   is to measure that run.
3. List what the idea needs that does not exist yet:
   - code (a flag, a module, a test);
   - caches (and their size on disk);
   - hardware (cpu, mps or a Colab GPU);
   - a notebook.

## Step 2: split by these rules

A task is small enough when **all** of these hold:

| Rule | Why |
|---|---|
| One run, or one comparison, per task | Its *Evidence* cell holds one `run_id` and one number |
| Only one thing changes against the baseline | Otherwise the difference cannot be attributed to anything |
| Cheapest first: pooled/dense before token-level/rnn, cpu before GPU, `category` before `salary` | A cheap negative result can cancel the expensive task |
| Each later, costlier task has a **gate** on the earlier one: `Depends` plus a stated condition in *Evidence* (e.g. "run only if T8.6 beats the dense baseline, bootstrap CI above 0") | If the cheap task shows no gain, the expensive one becomes `dropped (reason)` instead of being run |
| One Colab session (≤ 35 min of training, the free-tier cap) per task | A dropped session loses at most one task |
| Code and runs split when the run needs a machine the agent cannot use | Code can be `done` with `pytest` as its evidence while the run waits for Colab |
| `test` is never a task of its own until a model has won on `dev` | Rule 5 |

Typical order for a new encoder or model:
1. code + tests;
2. dense baseline + linear probe;
3. rnn head;
4. `salary`;
5. ablation.

Stop proposing after the first task whose outcome decides whether the rest are
worth running.

## Step 3: propose, then write

1. Show the author the proposed rows in the §2 column layout: ID, what, status
   `todo`, owner `—`, depends, and *Evidence* stating the exact `run_id` and the
   number it is compared against. Use the next free IDs in the right section.
   Say in one line what each task costs (time, disk, hardware).
2. Wait for approval. The author may merge, drop or reorder rows.
3. Write only the approved rows, then add one §5 log line saying who asked for
   them and why they were split that way. Everything in `TASKS.md` is in English
   (Rule 6).
4. If a notebook is part of the work, give each task its own notebook, or its own
   clearly separated cells, so it can run alone.

## What this skill does not do

- It does not pick which idea to try. That is the job of
  [`searching`](../searching/SKILL.md).
- It does not write results into `docs/` or `docs/bao-cao/`. That happens only
  after a run (Rules 1, 7 and 11).
- It does not assign an owner or set a task to `doing` unless the author says so.
