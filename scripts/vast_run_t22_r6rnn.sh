#!/usr/bin/env bash
# T22 R6-RNN: BiGRU-LSTM-CNN lương, một text có thêm 6 trường, seed 42, 5 encoder.
# Cấu hình giống dl-sal-<enc>-rnn-last4-b01-cw-s42-v2 (scripts/vast_run_sal_rnn.sh), chỉ đổi split.
# splits-r6text dựng sẵn trên Mac (PhoBERT cần cột tách từ) rồi đẩy lên. Chỉ train + dev;
# chấm test là bước riêng do tác giả chạy (eval_sal_rnn.py, Rule 5).
#
#   bash scripts/vast_run_t22_r6rnn.sh phobert cafebert xlmr visobert bamibert
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src VIETJOBS_SPLITS=splits-r6text
PY=/venv/main/bin/python
mkdir -p logs artifacts
LOG=logs/r6rnn.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a $LOG; }

for e in "$@"; do
  case $e in phobert) L=256 ;; bamibert) L=1024 ;; *) L=512 ;; esac
  rid=dl-sal-t22-r6rnn-$e-s42
  say "TRAIN $rid ($(df -h / | tail -1 | awk '{print $4}') trống)"
  $PY -m vietjobs.dl.train_dl --task salary --encoder $e --max-len $L --head rnn --last4 \
    --hidden 100 --conv-channels 50 --epochs 40 --patience 8 --beta-sal 0.1 --class-weight \
    --device cuda --seed 42 --run-id $rid 2>&1 | tee logs/$rid.txt | grep -aE "^\[dl-|Error|Traceback"
  find artifacts/embeddings \( -name "*${e}*-tok*.npy" -o -name "*-len${L}-last4-tok*.npy" \) -delete
  say "XONG $rid"
done
say "=== XONG R6-RNN: $* ==="
