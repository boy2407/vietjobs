#!/usr/bin/env bash
# Đợt 20 — train lại MTL với loss mới rồi chấm NGAY trên cùng máy (Rule 4; bài học Đợt 17).
#
#   bash scripts/vast_run_dot20.sh rnn:bamibert dense      # máy 1
#   bash scripts/vast_run_dot20.sh rnn:cafebert rnn:phobert  # máy 2
#
# Mỗi job rnn:<enc>: train MTL RNN (last4 + cw effective 0.999 + ls 0.05 + beta 0.1)
#   (--resume: máy bị ngắt thì chạy lại lệnh, train tiếp từ last.pt)
#   -> check_mtl_dev.py (dev phải khớp metrics.json) -> chấm test + ext37k -> xoá cache token.
# Job dense: encode pooled last4 5 encoder -> 5 MTL Dense (loss mới) + 25 Salary Dense
#   (last4 + cw + beta 0.1, lệnh gốc Đợt 5) -> kiểm dev MTL Dense -> chấm test + ext37k.
# Run mới mang đuôi -v2; run cũ không bị đụng.
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src
PY=/venv/main/bin/python
SUF=-v2
LOSS="--class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 --beta-sal 0.1 --alpha-sal 5.0"
mkdir -p logs artifacts
LOG=logs/dot20-all.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a $LOG; }

# Bài học Đợt 19: /workspace có thể là ổ riêng 10 GB — cache phải nằm trên ổ đủ lớn.
free_gb=$(df -BG --output=avail artifacts 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 70 ] || { say "DUNG: chi con ${free_gb} GB trong o chua artifacts/ (can >= 70)"; exit 1; }

spec() { # encoder -> short max_len extra
  case $1 in
    phobert)  echo "pho 256" ;;
    cafebert) echo "caf 512" ;;
    visobert) echo "vis 512" ;;
    xlmr)     echo "xlm 512" ;;
    bamibert) echo "bam 1024" ;;
  esac
}

run_rnn() {
  local e=$1; read -r s L extra <<<"$(spec $e)"
  local rid=dl-mtl-$s-rnn-b01-cw-s42$SUF
  extra="--conv-channels 64"  # cùng đầu với 4 encoder kia (hidden 256 mặc định), so sánh công bằng
  say "TRAIN $rid"
  $PY -m vietjobs.dl.train_mtl --encoder $e --max-len $L --head rnn --last4 $LOSS $extra \
    --epochs 80 --patience 12 --device cuda --seed 42 --run-id $rid --resume 2>&1 | tee logs/$rid.txt
  say "KIEM DEV $rid"
  $PY scripts/check_mtl_dev.py $rid 2>&1 | grep -a DEV | tee -a $LOG
  say "CHAM $rid"
  ONLY=$rid RUN_SUFFIX=$SUF $PY scripts/eval_mtl_all.py 2>&1 | tee logs/eval-$rid.txt
  rm -f artifacts/embeddings/*${e}*-tok*.npy artifacts/embeddings/*-len${L}-last4-tok*.npy
  say "XONG $rid"
}

run_dense() {
  for e in phobert cafebert visobert xlmr bamibert; do
    read -r s L _ <<<"$(spec $e)"
    local b=32; [ $e = phobert ] && b=64; [ $e = bamibert ] && b=16
    say "ENCODE pooled last4 $e"
    $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $L --batch $b \
      --last4 --device cuda 2>&1 | tee logs/dot20-encode-$e.txt
    local rid=dl-mtl-$s-dense-last4-cw-s42$SUF
    $PY -m vietjobs.dl.train_mtl --encoder $e --max-len $L --head dense --last4 $LOSS \
      --epochs 80 --patience 15 --device cuda --seed 42 --run-id $rid 2>&1 | tee logs/$rid.txt
    $PY scripts/check_mtl_dev.py $rid 2>&1 | grep -a DEV | tee -a $LOG
    local ss=$s; [ $s = caf ] && ss=cafe; [ $s = vis ] && ss=visobert; [ $s = xlm ] && ss=xlmr; [ $s = bam ] && ss=bamibert
    for seed in 42 43 44 45 46; do
      rid=dl-sal-$ss-dense-last4-b01-cw-s$seed$SUF
      $PY -m vietjobs.dl.train_dl --task salary --encoder $e --max-len $L --head dense --last4 --device cpu \
        --run-id $rid --beta-sal 0.1 --class-weight --seed $seed 2>&1 | tee logs/$rid.txt
    done
  done
  say "CHAM DENSE"
  RUN_SUFFIX=$SUF $PY scripts/eval_dense_all.py 2>&1 | tee logs/eval-dense$SUF.txt
  say "XONG DENSE"
}

for job in "$@"; do
  case $job in
    rnn:*) run_rnn "${job#rnn:}" ;;
    dense) run_dense ;;
    *) say "job la: $job"; exit 2 ;;
  esac
done
say "=== XONG DOT 20: $* ==="
