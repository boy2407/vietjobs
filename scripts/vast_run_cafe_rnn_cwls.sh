#!/usr/bin/env bash
# Đợt 10: CafeBERT 512 + RNN, category, class weight effective (β=0.999) + label smoothing 0.05.
# Cùng head/optimiser với dl-cat-cafe-rnn-ce-s42..44 (scripts/vast_run_cafe_rnn.sh); chỉ đổi loss.
#
#   bash scripts/vast_run_cafe_rnn_cwls.sh model    # chỉ tải + nạp thử CafeBERT (~2,2 GB)
#   bash scripts/vast_run_cafe_rnn_cwls.sh all      # test → encode → seed 42 43 44
#   bash scripts/vast_run_cafe_rnn_cwls.sh encode   # chỉ nhúng cache token (train + dev)
#   bash scripts/vast_run_cafe_rnn_cwls.sh train    # chỉ train (bỏ qua seed đã có metrics.json)
#
# Không đụng test (Rule 5). Cache token category ≈ 40 GB.
set -euo pipefail
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs

model() {  # tải trước để lỗi mạng/HF lộ ra trước khi nhúng
  $PYTHON - <<'PY'
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("uitnlp/CafeBERT", use_fast=True)
m = AutoModel.from_pretrained("uitnlp/CafeBERT")
print("CafeBERT ok:", type(tok).__name__, m.config.num_hidden_layers, "lớp", m.config.hidden_size, "chiều")
PY
}
encode() {
  nvidia-smi
  $PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py
  $PYTHON -m vietjobs.dl.encode --task category --splits dev train \
    --encoder cafebert --max-len 512 --pooling none --device cuda 2>&1 | tee -a logs/cafe-rnn-cwls-encode.txt
}
train() {
  for s in 42 43 44; do
    rid=dl-cat-cafe-rnn-cweff999-ls05-s$s
    [ -f "artifacts/$rid/metrics.json" ] && { echo "skip $rid"; continue; }
    $PYTHON -m vietjobs.dl.train_dl --task category --head rnn --encoder cafebert --max-len 512 \
      --device cuda --hidden 100 --conv-channels 50 --seed "$s" \
      --class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 \
      --run-id "$rid" 2>&1 | tee "logs/$rid.txt"
  done
}

case "${1:-}" in
  model) model ;;
  all) model; encode; train ;;
  encode) encode ;;
  train) train ;;
  *) echo "usage: $0 model|all|encode|train" >&2; exit 2 ;;
esac
