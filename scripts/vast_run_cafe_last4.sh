#!/usr/bin/env bash
# Đợt 7: nhúng CafeBERT 512 pooled, trung bình 4 lớp cuối (encode.py --layers 4), train + dev.
# Chỉ nhúng trên máy thuê; train Dense chạy trên Mac, --device cpu (Rule 13 item 5).
# Không đụng test (Rule 5): chỉ nhúng train và dev.
set -euo pipefail
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs
$PYTHON -m vietjobs.dl.encode --task category --encoder cafebert --max-len 512 --layers 4 \
  --splits train dev --device cuda --batch 32 2>&1 | tee logs/cafe-last4-encode-vast.txt
ls -l artifacts/embeddings/*/cafebert-*-raw-len512-last4.*
md5sum artifacts/embeddings/*/cafebert-*-raw-len512-last4.npy
