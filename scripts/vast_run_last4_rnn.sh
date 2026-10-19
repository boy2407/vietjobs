#!/usr/bin/env bash
# Đợt 11: RNN category với 4 lớp ẩn cuối + class weight effective (β=0.999) + label smoothing 0.05, seed 42.
# Cũng nhúng cache pooled last4 (train + dev) để huấn luyện Dense last4 trên Mac (Rule 13 item 5).
#
#   bash scripts/vast_run_last4_rnn.sh bamibert 1024 32
#   bash scripts/vast_run_last4_rnn.sh xlmr 512 32
#
# SKIP_POOLED=1 bỏ bước nhúng pooled (khi cache pooled last4 đã có trên Mac).
# Không đụng test (Rule 5). Cache token category xoá ngay sau run để vừa đĩa 120 GB.
set -euo pipefail
ENC_NAME=${1:?encoder}; MAXLEN=${2:?max_len}; BATCH=${3:-32}
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs

# logs/SKIP_<encoder> có mặt thì bỏ qua encoder này (dùng khi chuyển việc sang máy khác giữa chừng).
if [ -f "logs/SKIP_$ENC_NAME" ]; then echo "SKIP $ENC_NAME (logs/SKIP_$ENC_NAME)"; echo DONE; exit 0; fi

check_sha() {
  ( cd data/processed && sha256sum -c --quiet - ) <<'EOF' || { echo "SHA SAI ($1) — dừng"; exit 1; }
ee457e6b47b2c72a5be6e9bfeaf80bf41f4e6e3e909abbb2e59dbaa6020a1758  splits/train.csv
7b88f3bb627d38b30e6e265a677f089a94a6b4210d6e224105f4019ce1131d9d  splits/dev.csv
70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047  splits/test.csv
1f326f3bba678529c64acb735da75874e6d8109ab07a9c6f2c71e25576b8ac18  manifest.json
EOF
  echo "sha256 splits + manifest ok ($1)"
}

check_sha before
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
$PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py

if [ "${SKIP_POOLED:-0}" = "1" ]; then echo "=== bỏ qua pooled last4 (SKIP_POOLED=1) ==="; else
echo "=== $ENC_NAME pooled last4 (train + dev) ==="
$PYTHON -m vietjobs.dl.encode --task category --splits dev train --encoder "$ENC_NAME" \
  --max-len "$MAXLEN" --batch "$BATCH" --layers 4 --device cuda 2>&1 | tee -a "logs/last4-encode-$ENC_NAME.txt"
fi

echo "=== $ENC_NAME token last4 (train + dev) ==="
$PYTHON -m vietjobs.dl.encode --task category --splits dev train --encoder "$ENC_NAME" \
  --max-len "$MAXLEN" --batch "$BATCH" --layers 4 --pooling none --device cuda 2>&1 | tee -a "logs/last4-encode-$ENC_NAME.txt"
df -h /workspace | tail -1

rid=dl-cat-$ENC_NAME-rnn-last4-cweff999-ls05-s42
if [ -f "artifacts/$rid/metrics.json" ]; then echo "skip $rid"; else
  $PYTHON -m vietjobs.dl.train_dl --task category --head rnn --encoder "$ENC_NAME" --max-len "$MAXLEN" \
    --device cuda --hidden 100 --conv-channels 50 --seed 42 --layers 4 \
    --class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 \
    --run-id "$rid" 2>&1 | tee "logs/$rid.txt"
fi
rm -f artifacts/embeddings/*/"$ENC_NAME"-*-raw-len"$MAXLEN"-last4-tok.f16.npy artifacts/embeddings/"$ENC_NAME"-*-raw-len"$MAXLEN"-last4-tok.f16.npy
check_sha after
ls -l artifacts/embeddings/ | grep "$ENC_NAME-.*raw-len$MAXLEN-last4"
echo DONE
