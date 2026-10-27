#!/usr/bin/env bash
# Salary RNN đơn nhiệm -v2 (last4 + cw + beta 0.1): train rồi chấm NGAY trên cùng máy (Rule 4).
#
#   bash scripts/vast_run_sal_rnn.sh phobert cafebert xlmr visobert bamibert
#
# Cấu hình như các run dl-sal-<enc>-rnn-b01-cw-s42 cũ (hidden 100, conv 50, 40 epoch, patience 8),
# chỉ thêm --last4. Mỗi encoder: train -> eval_sal_rnn.py (cổng dev rồi test + ext37k) -> xoá cache token.
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src
PY=/venv/main/bin/python
mkdir -p logs artifacts
LOG=logs/sal-rnn-v2.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a $LOG; }

for e in "$@"; do
  case $e in phobert) L=256 ;; bamibert) L=1024 ;; *) L=512 ;; esac
  rid=dl-sal-$e-rnn-last4-b01-cw-s42-v2
  say "TRAIN $rid"
  $PY -m vietjobs.dl.train_dl --task salary --encoder $e --max-len $L --head rnn --last4 \
    --hidden 100 --conv-channels 50 --epochs 40 --patience 8 --beta-sal 0.1 --class-weight \
    --device cuda --seed 42 --run-id $rid 2>&1 | tee logs/$rid.txt
  say "CHAM $rid"
  $PY scripts/eval_sal_rnn.py $rid 2>&1 | tee logs/eval-$rid.txt | grep -a -E "DEV|TEST" | tee -a $LOG
  rm -f artifacts/embeddings/*${e}*-tok*.npy artifacts/embeddings/*-len${L}-last4-tok*.npy
  say "XONG $rid"
done
say "=== XONG SAL RNN: $* ==="
