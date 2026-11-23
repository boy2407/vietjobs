#!/usr/bin/env bash
# T22 R6-L4 trên máy thuê: dựng splits-r6text → nhúng CafeBERT last4 (GPU) → Dense 5 seed (cpu).
# Chạy trong /workspace/vietjobs sau khi đã đẩy src/, scripts/, data/processed/{manifest.json,
# splits/train.csv, splits/dev.csv}, docs/04-results.md. Không đọc test.
set -eo pipefail
cd /workspace/vietjobs
PY=/root/venv/bin/python
export PYTHONPATH=src

df -h / | tail -1
$PY scripts/build_r6text_split.py
VIETJOBS_SPLITS=splits-r6text $PY -m vietjobs.dl.encode --task salary --encoder cafebert \
  --max-len 512 --layers 4 --splits train dev --device cuda --batch 32
for s in 42 43 44 45 46; do
  VIETJOBS_SPLITS=splits-r6text $PY -m vietjobs.dl.train_dl --task salary --encoder cafebert \
    --max-len 512 --layers 4 --class-weight --beta-sal 0.1 --device cpu --seed $s \
    --run-id dl-sal-t22-r6l4-s$s 2>&1 | grep -E "^\[dl-|Error|error"
done
echo XONG
