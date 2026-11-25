# VietJobs

Occupation classification and salary estimation from Vietnamese job postings. These are the
terms the thesis and its defence use with one fixed meaning.

## Language

**Mô hình chốt** (final model):
The single training run of the multi-task CafeBERT model with an RNN body that the thesis
reports as its result. The run keeps two checkpoints: one picked by macro-F1 on dev for the
classification head, one picked by MAE on dev for the salary head.
_Avoid_: "CafeBERT với bốn đầu", best model

**Cấu hình CafeBERT** (CafeBERT configuration):
One of the four CafeBERT set-ups compared before choosing the final model: Dense for
classification, Dense for salary, multi-task Dense, multi-task RNN.
_Avoid_: đầu (when a whole configuration is meant)

**Nhóm tin đăng lại** (repost group):
Postings whose title, description and requirements are identical after lower-casing and
removing accents and punctuation; a group always lands in one split.
_Avoid_: nhóm trùng gần (it does not catch postings that differ by a word)

**Tin xung đột nhãn** (conflicting-label posting):
A train posting whose title, description and requirements are character-for-character
identical to another posting's, while its occupation label differs.
_Avoid_: tin trùng, nhãn nhiễu (without qualification)

**Tin gần trùng** (near-duplicate posting):
A dev or test posting that has a train posting with word-set Jaccard of at least 0.9 over
title, description and requirements, while being in a different repost group.
_Avoid_: tin trùng, leak (without qualification)

**Phần sạch** (clean subset):
The dev or test split with its near-duplicate postings left out, used only to measure how much
they shift a score.
_Avoid_: clean test, filtered test (test itself is never filtered)

**Cắt bỏ** (truncation):
Reading only the first tokens of a posting, up to the encoder's length limit, and discarding
the rest.
_Avoid_: cắt (without qualification), truncate

**Cắt đoạn** (chunking):
Splitting a posting that is longer than the encoder's length limit into consecutive,
non-overlapping pieces so that every token of the posting is read; a posting that fits in one
piece is read exactly as under cắt bỏ.
_Avoid_: cắt trượt, sliding window (the pieces do not overlap)
