#!/usr/bin/env bash
# T22 R6-L4 cho 4 encoder còn lại. splits-r6text dựng sẵn trên Mac (cần bộ tách từ cho PhoBERT) rồi đẩy lên.
set -eo pipefail
cd /workspace/vietjobs
PY=/root/venv/bin/python
export PYTHONPATH=src VIETJOBS_SPLITS=splits-r6text
for spec in phobert:256:pho:32 visobert:512:visobert:32 xlmr:512:xlmr:32 bamibert:1024:bamibert:16; do
  IFS=: read enc len tag bs <<< "$spec"
  $PY -m vietjobs.dl.encode --task salary --encoder $enc --max-len $len --layers 4 \
    --splits train dev --device cuda --batch $bs
  for s in 42 43 44 45 46; do
    $PY -m vietjobs.dl.train_dl --task salary --encoder $enc --max-len $len --layers 4 \
      --class-weight --beta-sal 0.1 --device cpu --seed $s --run-id dl-sal-t22-r6l4-$tag-s$s 2>&1 \
      | grep -E "^\[dl-|Error|error"
  done
done
echo XONG
