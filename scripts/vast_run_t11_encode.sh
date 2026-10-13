#!/usr/bin/env bash
# T11: ViSoBERT / XLM-R base / BamiBERT — encode + toàn bộ Dense, RNN, MTL (seed 42), trên vast.ai.
#
#   bash scripts/vast_run_t11_encode.sh            # cả ba encoder
#   bash scripts/vast_run_t11_encode.sh xlmr       # một encoder
#
# ViSoBERT: pooled + 3 run Dense đã chạy trên Mac (2026-10-12) → chỉ phần token.
# Dense (đơn nhiệm + MTL dense) chạy --device cpu (Rule 13 item 5), RNN chạy cuda.
# Cấu hình giống T8.6/T8.8 (Dense), T8.16 (RNN h=100 c=50), T4.1 (MTL α=5, epochs 80).
# Độ dài: ViSoBERT/XLM-R 512 (tối đa của model). BamiBERT 1024 cho mọi run (tác giả chọn
# 2026-10-12; tin dài nhất 1,553 token, chỉ 0.1 % tin bị cắt ở 1024).
set -euo pipefail
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs

enc() {  # encoder task max_len batch [--pooling none]
  local e=$1 t=$2 L=$3 b=$4; shift 4
  $PYTHON -m vietjobs.dl.encode --task "$t" --splits train dev --encoder "$e" \
    --max-len "$L" --batch "$b" --device cuda "$@" 2>&1 | tee -a "logs/t11-encode-$e.txt"
}
st() {  # run_id task encoder max_len device [extra]
  local rid=$1 t=$2 e=$3 L=$4 d=$5; shift 5
  [ -f "artifacts/$rid/metrics.json" ] && { echo "skip $rid"; return; }
  $PYTHON -m vietjobs.dl.train_dl --task "$t" --encoder "$e" --max-len "$L" \
    --device "$d" --seed 42 --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}
mtl() {  # run_id encoder max_len device head [extra]
  local rid=$1 e=$2 L=$3 d=$4 h=$5; shift 5
  [ -f "artifacts/$rid/metrics.json" ] && { echo "skip $rid"; return; }
  $PYTHON -m vietjobs.dl.train_mtl --head "$h" --encoder "$e" --max-len "$L" \
    --alpha-sal 5 --epochs 80 --device "$d" --seed 42 --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
}
RNN="--head rnn --hidden 100 --conv-channels 50"

one() {  # encoder pooled_len rnn_len short [skip_pooled]
  local e=$1 P=$2 R=$3 s=$4
  if [ "${5:-}" != skip_pooled ]; then
  echo "=== $e: pooled @$P ==="
  for t in category salary; do enc "$e" "$t" "$P" 32; done
  st  "dl-cat-$e-dense-$P" category "$e" "$P" cpu
  st  "dl-sal-$e-dense-$P" salary   "$e" "$P" cpu
  mtl "dl-mtl-$s$P-a5-e80-s42" "$e" "$P" cpu dense
  fi
  echo "=== $e: token cache @$R ==="
  for t in category salary; do enc "$e" "$t" "$R" 16 --pooling none; done
  st  "dl-cat-$e-rnn-$R" category "$e" "$R" cuda $RNN --loss ce
  st  "dl-sal-$e-rnn-$R" salary   "$e" "$R" cuda $RNN
  mtl "dl-mtl-$e-rnn$R-a5-s42" "$e" "$R" cuda rnn --hidden 100 --conv-channels 50
}

nvidia-smi; free -g; df -h /workspace
$PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py
case "${1:-all}" in
  visobert) one visobert 512 512 visobert skip_pooled ;;
  xlmr)     one xlmr 512 512 xlmr ;;
  bamibert) one bamibert 1024 1024 bamibert ;;
  all)      one visobert 512 512 visobert skip_pooled
            one xlmr 512 512 xlmr
            one bamibert 1024 1024 bamibert ;;
  *) echo "usage: $0 [all|visobert|xlmr|bamibert]" >&2; exit 2 ;;
esac
echo ALL_DONE
