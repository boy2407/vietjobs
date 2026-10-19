#!/usr/bin/env bash
# Bài `category` trên bộ v8 (MinHash, splits-minhash-1label), đầu RNN, 1 seed (42), chạy HẾT trên vast.ai GPU.
#
#   bash scripts/vast_run_v8_category_rnn.sh all     # encode token cache (raw) -> train 2 run -> chấm ext37k-sal
#
# Cấu hình giống `dl-cat-cafe-rnn-ce-s42-t12` (CafeBERT 512, --loss ce, hidden 100, conv 50, cuda).
# Họ cột `raw` (phân loại). `test` không được nhúng và không được chấm (Rule 5).
# Disk: token cache raw CafeBERT 512 ~20 GB + PhoBERT 256 ~8 GB (train + dev); vast/dot-7-v8-category-rnn.md.
set -euo pipefail
PYTHON=/venv/main/bin/python
export PYTHONPATH=src
export VIETJOBS_SPLITS=splits-minhash-1label
mkdir -p logs

encode() {  # encoder max_len
  echo "=== Encode $1 len $2 (category, train + dev): token cache ==="
  $PYTHON -m vietjobs.dl.encode --task category --splits train dev --encoder "$1" --max-len "$2" --pooling none --device cuda
}
train() {  # encoder max_len run_id
  $PYTHON -m vietjobs.dl.train_dl --task category --encoder "$1" --max-len "$2" --head rnn --loss ce \
    --hidden 100 --conv-channels 50 --device cuda --seed 42 --run-id "$3" 2>&1 | tee "logs/$3.txt"
}

nvidia-smi
$PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py
encode phobert 256
train phobert 256 mh-cat-pho-rnn-256-s42
encode cafebert 512
train cafebert 512 mh-cat-cafe-rnn-512-s42

# Chấm với VIETJOBS_SPLITS bỏ đặt (khử trùng ext với bộ cũ đủ 47.707 dòng, như mọi lần chấm trước).
unset VIETJOBS_SPLITS
$PYTHON scripts/score_t12.py --dataset ext37k-sal --device cuda \
  --runs mh-cat-pho-rnn-256-s42,mh-cat-cafe-rnn-512-s42 2>&1 | tee -a logs/v8-cat-rnn-ext-score.txt
