#!/usr/bin/env bash
# Chấm test + ext37k cho các run MTL RNN đã huấn luyện ở máy khác (best.pt đã đẩy lên máy này).
#
#   bash scripts/vast_score_mtl_rnn.sh dl-mtl-vis-rnn-b01-cw-s42-v2 dl-mtl-xlm-rnn-b01-cw-s42-v2
#
# Cổng: chấm lại `dev` bằng đúng đường chấm và so với metrics.json (check_mtl_dev.py). Lệch thì
# DỪNG, không chạm `test` — Đợt 17 cho thấy chấm ở môi trường khác môi trường huấn luyện có thể
# làm số phồng (Rule 4). Mỗi run cần: best.pt, best_sal.pt, scaler.npz, config.json, metrics.json.
set -eo pipefail
cd /workspace/vietjobs
export PYTHONPATH=src
PY=/venv/main/bin/python
SUF=-v2
mkdir -p logs
LOG=logs/score-mtl-rnn.txt
say() { echo "[$(date -u +%H:%M)] $*" | tee -a "$LOG"; }

for rid in "$@"; do
  for f in best.pt best_sal.pt scaler.npz config.json metrics.json; do
    [ -f "artifacts/$rid/$f" ] || { say "DUNG: thieu artifacts/$rid/$f"; exit 1; }
  done
  say "KIEM DEV $rid"
  gate=$($PY scripts/check_mtl_dev.py "$rid" 2>&1 | grep -a "DEV" || true)
  say "$gate"
  if ! grep -q "KHOP" <<<"$gate"; then
    say "DUNG: dev cham lai khong khop metrics.json -> khong cham test cho $rid"
    exit 1
  fi
  say "CHAM test + ext37k $rid"
  ONLY=$rid RUN_SUFFIX=$SUF $PY scripts/eval_mtl_all.py 2>&1 | tee "logs/eval-$rid.txt"
  say "XONG CHAM $rid"
done
say "=== XONG: $* ==="
