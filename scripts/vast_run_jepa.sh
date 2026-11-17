#!/usr/bin/env bash
# Đợt 24 — T19 JEPA: pretrain encoder nhỏ trên text train, rồi MTL RNN trên encoder đó
# và trên cùng encoder khởi tạo ngẫu nhiên (đối chứng).
#
#   bash scripts/vast_run_jepa.sh smoke     # vài phút: 512 dòng, 1 epoch, không ghi 04-results
#   bash scripts/vast_run_jepa.sh full      # pretrain 30 epoch + 2 finetune (80 epoch, patience 12)
#
# Không cần token cache CafeBERT: chỉ tải tokenizer + bảng embedding từ vựng (~2,2 GB mô hình).
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src
PY=${PY:-/venv/main/bin/python}
SEED=${SEED:-42}
mkdir -p logs artifacts
LOG=logs/jepa-all.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a $LOG; }

# Bài học Đợt 23: thiếu pyarrow thì run chết lúc ghi predictions_dev.parquet, sau khi train xong.
$PY -c "import pyarrow, transformers, torch; assert torch.cuda.is_available()" \
  || { say "DUNG: thieu pyarrow/transformers hoac khong co CUDA"; exit 1; }

J="$PY -m vietjobs.dl.train_jepa"
case ${1:-} in
  smoke)
    $J pretrain --run-id jepa-smoke --limit 512 --epochs 1 --device cuda 2>&1 | tee logs/jepa-smoke.txt
    $J finetune --run-id jepa-smoke-ft --init jepa --from-run jepa-smoke --limit 512 --epochs 1 \
      --no-log --device cuda 2>&1 | tee -a logs/jepa-smoke.txt
    $J finetune --run-id jepa-smoke-rand --init random --limit 512 --epochs 1 \
      --no-log --device cuda 2>&1 | tee -a logs/jepa-smoke.txt
    say "SMOKE XONG"
    ;;
  full)
    PRE=jepa-pre-d256-s$SEED
    say "PRETRAIN $PRE"
    $J pretrain --run-id $PRE --seed $SEED --device cuda 2>&1 | tee logs/$PRE.txt
    say "FINETUNE jepa"
    $J finetune --run-id dl-mtl-jepa-rnn-s$SEED --init jepa --from-run $PRE --seed $SEED \
      --device cuda 2>&1 | tee logs/dl-mtl-jepa-rnn-s$SEED.txt
    say "FINETUNE random"
    $J finetune --run-id dl-mtl-jeparand-rnn-s$SEED --init random --seed $SEED \
      --device cuda 2>&1 | tee logs/dl-mtl-jeparand-rnn-s$SEED.txt
    say "XONG full s$SEED"
    ;;
  *) echo "dung: bash scripts/vast_run_jepa.sh smoke|full"; exit 2 ;;
esac
