#!/usr/bin/env bash
# RNN head (PhoBERT token cache) for `category` + `salary` on a rented vast.ai GPU.
#
#   bash scripts/vast_run_rnn.sh gate   # T8.14: test gate, encode train token caches, seed 42
#   bash scripts/vast_run_rnn.sh rest   # T8.15: seeds 43 44 (only after the seed-42 gate passes)
#
# Run inside tmux from the repo root. Config matches T8.3/T8.4 (hidden 100, conv 50).
# max-len stays at the default 256: PhoBERT's max_position_embeddings is 258 (~256 real
# tokens), so 512 is not reachable with this encoder — only CafeBERT (XLM-R-large) goes to
# 512 (T8.6/T8.8). No --max-minutes (no Colab cap). Each run logs to logs/<run_id>.txt.
set -euo pipefail
export PYTHONPATH=src
mkdir -p logs

run() {  # task seed run_id [extra args]
  local task=$1 seed=$2 rid=$3; shift 3
  python -m vietjobs.dl.train_dl --task "$task" --head rnn --device cuda \
    --hidden 100 --conv-channels 50 --seed "$seed" --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}

case "${1:-}" in
  gate)
    nvidia-smi
    python -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py
    # same command as notebooks/colab_setup.py train_token_cache: skipped when tok + mask exist
    python -m vietjobs.dl.encode --task category --splits train --pooling none --device cuda
    python -m vietjobs.dl.encode --task salary   --splits train --pooling none --device cuda
    seeds="42" ;;
  rest) seeds="43 44" ;;
  *) echo "usage: $0 gate|rest" >&2; exit 2 ;;
esac

for s in $seeds; do
  run category "$s" "dl-cat-rnn-ce-s$s" --loss ce
  run salary   "$s" "dl-sal-rnn-s$s"
done
