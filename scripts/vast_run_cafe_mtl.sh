#!/usr/bin/env bash
# T4.1: trunk RNN chung (CafeBERT 512, family `masked`) → hai đầu category + salary, trên vast.ai GPU.
#
#   bash scripts/vast_run_cafe_mtl.sh gate   # encode token cache masked, smoke 1 epoch, chạy seed 42
#   bash scripts/vast_run_cafe_mtl.sh rest   # seed 43 44 (chỉ sau khi gate pass)
#
# Chạy trong tmux từ thư mục gốc repo. Chỉ cần cache `masked` (task salary):
# train ~36 GB + dev ~4,5 GB float16 → RAM ≥ 48 GB, disk ≥ 100 GB.
# Cấu hình head giống T8.16 (hidden 100, conv 50) để so với run đơn nhiệm.
# Ngưỡng gate (seed 42): dl-cat-cafe-rnn-ce-s42 macro-F1 0.6172, dl-sal-cafe-rnn-s42 MAE 3.89 tr.
set -euo pipefail
PYTHON=${PYTHON:-$(command -v python)}
export PYTHONPATH=src
mkdir -p logs

run() {  # seed run_id [extra args]
  local seed=$1 rid=$2; shift 2
  $PYTHON -m vietjobs.dl.train_mtl \
    --head rnn --encoder cafebert --max-len 512 \
    --device cuda --hidden 100 --conv-channels 50 \
    --seed "$seed" --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}

case "${1:-}" in
  gate)
    nvidia-smi
    free -g
    $PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py

    echo "=== Encode CafeBERT 512 token cache (masked = task salary, train + dev) ==="
    $PYTHON -m vietjobs.dl.encode --task salary --splits train dev \
      --encoder cafebert --max-len 512 --pooling none --device cuda
    du -sh artifacts/embeddings

    echo "=== Smoke: 1 epoch, không ghi 04-results ==="
    $PYTHON -m vietjobs.dl.train_mtl --head rnn --encoder cafebert --max-len 512 \
      --device cuda --hidden 100 --conv-channels 50 --epochs 1 --no-log --run-id smoke-mtl-cafe-rnn
    rm -rf artifacts/smoke-mtl-cafe-rnn

    seeds="42" ;;
  rest) seeds="43 44" ;;
  *) echo "usage: $0 gate|rest" >&2; exit 2 ;;
esac

for s in $seeds; do
  run "$s" "dl-mtl-cafe-rnn-s$s"
done
