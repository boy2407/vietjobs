#!/usr/bin/env bash
# Đợt 28 — T23: train lại 6 cấu trúc last4 × 5 encoder trên splits-sal17-11c (90 run).
# Bảng quyết định: tasks/T23-train-lai.md. Một bộ tham số chung cho cả 6 cấu trúc.
#
#   bash scripts/vast_run_t23.sh pooled dense rnn     # cả ba giai đoạn, theo thứ tự
#   bash scripts/vast_run_t23.sh rnn:cafebert          # một encoder của giai đoạn RNN
#
# pooled: encode cache pooled last4 họ masked (train + dev), 5 encoder (T23.2).
# dense : Dense cat, Dense sal, MTL Dense × seed 42–46, cpu (T23.3–T23.5).
# rnn   : mỗi encoder encode cache token → RNN cat, RNN sal, MTL RNN seed 42, cuda → xoá cache token
#         (T23.7–T23.10).
# Chạy lại cùng lệnh là chạy tiếp: run đã có metrics.json bị bỏ qua; MTL RNN dùng --resume.
# Không đọc test: chấm test + ext37k là bước riêng, tác giả gõ lệnh (Rule 5).
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src VIETJOBS_SPLITS=splits-sal17-11c
PY=${PY:-/venv/main/bin/python}
mkdir -p logs artifacts
LOG=logs/t23.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a $LOG; }

COMMON="--last4 --lr 3e-4 --weight-decay 0.01 --dropout 0.3 --epochs 80 --patience 12"
CAT="--family masked --class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05"
SAL="--beta-sal 0.1 --class-weight"
MTL="--class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 --beta-sal 0.1 --alpha-sal 5.0"
RNN="--hidden 100 --conv-channels 50"
ENCS="phobert cafebert xlmr visobert bamibert"

free_gb=$(df -BG --output=avail artifacts 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 70 ] || { say "DUNG: chi con ${free_gb} GB trong o chua artifacts/ (can >= 70)"; exit 1; }

len() { case $1 in phobert) echo 256 ;; bamibert) echo 1024 ;; *) echo 512 ;; esac; }
bs()  { case $1 in phobert) echo 64 ;; bamibert) echo 16 ;; *) echo 32 ;; esac; }
done_() { [ -f artifacts/$1/metrics.json ] && say "BO QUA $1 (da co metrics.json)"; }

train() { # rid module args...
  local rid=$1 mod=$2; shift 2
  done_ $rid && return 0
  say "TRAIN $rid"
  $PY -m vietjobs.dl.$mod "$@" --run-id $rid 2>&1 | tee logs/$rid.txt | grep -aE "^\[dl-|Error|Traceback" || true
  [ -f artifacts/$rid/metrics.json ] || { say "LOI $rid: khong co metrics.json"; exit 1; }
}

stage_pooled() {
  for e in $ENCS; do
    say "ENCODE pooled last4 masked $e"
    $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $(len $e) \
      --batch $(bs $e) --last4 --device cuda 2>&1 | tee logs/t23-encode-$e.txt
  done
}

stage_dense() {
  for e in $ENCS; do
    L=$(len $e)
    for s in 42 43 44 45 46; do
      train dl-cat-$e-dense-sal17-s$s train_dl --task category --encoder $e --max-len $L --head dense \
        $COMMON $CAT --device cpu --seed $s
      train dl-sal-$e-dense-sal17-s$s train_dl --task salary --encoder $e --max-len $L --head dense \
        $COMMON $SAL --device cpu --seed $s
      train dl-mtl-$e-dense-sal17-s$s train_mtl --encoder $e --max-len $L --head dense \
        $COMMON $MTL --device cpu --seed $s
    done
  done
}

stage_rnn() { # encoder
  local e=$1 L=$(len $1)
  say "ENCODE token last4 masked $e ($(df -h artifacts | tail -1 | awk '{print $4}') trong)"
  $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $L \
    --batch $(bs $e) --last4 --pooling none --device cuda 2>&1 | tee logs/t23-encode-tok-$e.txt
  train dl-cat-$e-rnn-sal17-s42 train_dl --task category --encoder $e --max-len $L --head rnn \
    $COMMON $CAT $RNN --device cuda --seed 42
  train dl-sal-$e-rnn-sal17-s42 train_dl --task salary --encoder $e --max-len $L --head rnn \
    $COMMON $SAL $RNN --device cuda --seed 42
  say "TRAIN dl-mtl-$e-rnn-sal17-s42"
  done_ dl-mtl-$e-rnn-sal17-s42 || $PY -m vietjobs.dl.train_mtl --encoder $e --max-len $L --head rnn \
    $COMMON $MTL $RNN --device cuda --seed 42 --run-id dl-mtl-$e-rnn-sal17-s42 --resume 2>&1 \
    | tee -a logs/dl-mtl-$e-rnn-sal17-s42.txt | grep -aE "^\[dl-|Error|Traceback" || true
  [ -f artifacts/dl-mtl-$e-rnn-sal17-s42/metrics.json ] || { say "LOI MTL RNN $e"; exit 1; }
  # Cache token chỉ là trung gian (encode lại được từ splits); ext37k token cache (T23.11b) là bước sau.
  # Encoder chạy lần lượt nên chỉ cache token của encoder này còn trên đĩa (PhoBERT không có tiền tố tên).
  find artifacts/embeddings -name "*masked-len${L}-last4-tok*.npy" -delete
  say "XONG RNN $e"
}

for job in "$@"; do
  case $job in
    pooled) stage_pooled ;;
    dense)  stage_dense ;;
    rnn)    for e in $ENCS; do stage_rnn $e; done ;;
    rnn:*)  stage_rnn ${job#rnn:} ;;
    *) echo "job la: $job"; exit 2 ;;
  esac
done
say "=== XONG T23: $* ==="
