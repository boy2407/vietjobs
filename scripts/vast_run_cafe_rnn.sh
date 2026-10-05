#!/usr/bin/env bash
# CafeBERT 512-token + RNN head cho `category` + `salary` trên vast.ai GPU.
#
#   bash scripts/vast_run_cafe_rnn.sh gate   # T8.16: encode cache, chạy seed 42
#   bash scripts/vast_run_cafe_rnn.sh rest   # T8.17: seed 43 44 (chỉ sau khi gate pass)
#
# Chạy trong tmux từ thư mục gốc repo. Encoder: CafeBERT (XLM-R-large, hidden=1024).
# max-len 512 (CafeBERT hỗ trợ, PhoBERT không). Cache token-level ~34 GB/task/split.
# Disk yêu cầu: ≥ 150 GB trên máy thuê.
# Ngưỡng gate (seed 42, so với CafeBERT Dense 512 seed 42):
#   dl-cat-cafe-rnn-ce-s42  → macro-F1 cần cải thiện so với dl-cat-cafe-dense-512 (0.6077)
#   dl-sal-cafe-rnn-s42     → MAE cần cải thiện so với dl-sal-cafe-dense-512
set -euo pipefail
PYTHON=/venv/main/bin/python
export PYTHONPATH=src
mkdir -p logs

run() {  # task seed run_id [extra args]
  local task=$1 seed=$2 rid=$3; shift 3
  $PYTHON -m vietjobs.dl.train_dl \
    --task "$task" --head rnn --encoder cafebert --max-len 512 \
    --device cuda --hidden 100 --conv-channels 50 \
    --seed "$seed" --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}

case "${1:-}" in
  gate)
    nvidia-smi
    $PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py

    echo "=== Encode CafeBERT 512 token cache (category, train + dev) ==="
    $PYTHON -m vietjobs.dl.encode --task category --splits train dev \
      --encoder cafebert --max-len 512 --pooling none --device cuda

    echo "=== Encode CafeBERT 512 token cache (salary, train + dev) ==="
    $PYTHON -m vietjobs.dl.encode --task salary --splits train dev \
      --encoder cafebert --max-len 512 --pooling none --device cuda

    seeds="42" ;;
  rest) seeds="43 44" ;;
  *) echo "usage: $0 gate|rest" >&2; exit 2 ;;
esac

for s in $seeds; do
  run category "$s" "dl-cat-cafe-rnn-ce-s$s" --loss ce
  run salary   "$s" "dl-sal-cafe-rnn-s$s"
done
