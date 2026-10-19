#!/usr/bin/env bash
# Bài `salary` trên bộ v8 (MinHash, splits-minhash-1label), chạy HẾT trên vast.ai GPU, không chạy trên Mac.
#
#   bash scripts/vast_run_v8_salary.sh encode   # nhúng pooled + token cho PhoBERT 256 và CafeBERT 512 (train, dev)
#   bash scripts/vast_run_v8_salary.sh dense    # Dense: 5 seed (42-46) mỗi encoder, --device cpu (Rule 13)
#   bash scripts/vast_run_v8_salary.sh rnn      # RNN: 1 seed (42) mỗi encoder, --device cuda
#   bash scripts/vast_run_v8_salary.sh score    # chấm 12 run trên ext37k-sal (lương, silver, tạm thời)
#   bash scripts/vast_run_v8_salary.sh all      # lần lượt cả bốn
#
# Chạy trong tmux từ thư mục gốc repo. Họ cột `masked` (Rule 3): lương chỉ đọc cột *_masked.
# `test` không được nhúng và không được chấm (Rule 5). VIETJOBS_SPLITS chọn bộ chia và thư mục cache riêng.
# Disk: token cache CafeBERT 512 ~20 GB + PhoBERT 256 ~8 GB (train + dev); cần khoảng 52 GB, thuê 100 GB (vast/dot-6-v8-salary.md).
set -euo pipefail
PYTHON=/venv/main/bin/python
export PYTHONPATH=src
export VIETJOBS_SPLITS=splits-minhash-1label
mkdir -p logs

encode() {  # encoder max_len
  echo "=== Encode $1 len $2 (salary, train + dev): pooled rồi token cache ==="
  $PYTHON -m vietjobs.dl.encode --task salary --splits train dev --encoder "$1" --max-len "$2" --device cuda
  $PYTHON -m vietjobs.dl.encode --task salary --splits train dev --encoder "$1" --max-len "$2" --pooling none --device cuda
}

train() {  # encoder max_len head device seed run_id [extra]
  local enc=$1 len=$2 head=$3 dev=$4 seed=$5 rid=$6; shift 6
  $PYTHON -m vietjobs.dl.train_dl --task salary --encoder "$enc" --max-len "$len" --head "$head" \
    --device "$dev" --seed "$seed" --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}

do_encode() { nvidia-smi; $PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py; encode phobert 256; encode cafebert 512; }
do_dense() {
  for s in 42 43 44 45 46; do
    train phobert 256 dense cpu "$s" "mh-sal-pho-dense-256-s$s"
    train cafebert 512 dense cpu "$s" "mh-sal-cafe-dense-512-s$s"
  done
}
do_rnn() {
  train phobert 256 rnn cuda 42 "mh-sal-pho-rnn-256-s42" --hidden 100 --conv-channels 50
  train cafebert 512 rnn cuda 42 "mh-sal-cafe-rnn-512-s42" --hidden 100 --conv-channels 50
}

do_score() {
  # Chấm với VIETJOBS_SPLITS bỏ đặt: khử trùng ext37k-sal với bộ chia cũ đủ 47.707 dòng, nên cùng các dòng ext
  # (6.111 dòng) với mọi lần chấm trước; phải có data/processed/splits (train, dev, test) trên máy thuê.
  # Chỉ nhúng ext37k-sal họ masked, không nhúng test (Rule 5).
  unset VIETJOBS_SPLITS
  $PYTHON scripts/encode_t12_cpu.py --device cuda --ext-masked-only --jobs phobert:256 cafebert:512 2>&1 | tee -a logs/v8-sal-ext-encode.txt
  local dense="" i
  for s in 42 43 44 45 46; do dense="$dense,mh-sal-pho-dense-256-s$s,mh-sal-cafe-dense-512-s$s"; done
  $PYTHON scripts/score_t12.py --dataset ext37k-sal --device cpu --runs "${dense#,}" 2>&1 | tee -a logs/v8-sal-ext-score-dense.txt
  $PYTHON scripts/score_t12.py --dataset ext37k-sal --device cuda --runs mh-sal-pho-rnn-256-s42,mh-sal-cafe-rnn-512-s42 2>&1 | tee -a logs/v8-sal-ext-score-rnn.txt
}

case "${1:-}" in
  encode) do_encode ;;
  dense) do_dense ;;
  rnn) do_rnn ;;
  score) do_score ;;
  all) do_encode; do_dense; do_rnn; do_score ;;
  *) echo "usage: $0 encode|dense|rnn|score|all" >&2; exit 2 ;;
esac
