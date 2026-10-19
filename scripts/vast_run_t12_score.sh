#!/usr/bin/env bash
# T12.1b/c/d: nhúng cache cho test + ext37k-sal, chấm lại best.pt (RNN trên cuda, Dense trên cpu).
#
#   bash scripts/vast_run_t12_score.sh all      # encode + rnn + dense
#   bash scripts/vast_run_t12_score.sh encode   # chỉ nhúng
#   bash scripts/vast_run_t12_score.sh train    # chỉ train lại 2 run CafeBERT RNN seed 42 (-t12)
#   bash scripts/vast_run_t12_score.sh rnn      # chỉ chấm RNN
#   bash scripts/vast_run_t12_score.sh dense    # chỉ chấm Dense
#
# AGENTS.md Rule 5 item 6 (tác giả, 2026-10-15): chỉ ĐỌC test để chấm checkpoint có sẵn;
# không train lại, không ghi test.csv, không ghi gì dưới data/. Dense chấm --device cpu
# (Rule 13 item 5). Danh sách run = vast/dot-5-t12-score-test.md: 60 run = 34 hàng dashboard
# (2 run train lại -t12 thay cho checkpoint thiếu/hỏng của dl-sal-cafe-rnn và dl-cat-cafe-rnn-ce).
set -euo pipefail
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs
TEST_SHA=70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047

# Train lại (tác giả, 2026-10-16): CafeBERT RNN không có checkpoint dùng được (s42 hỏng, s43/s44
# thiếu; dl-sal-cafe-rnn không có). Chỉ MỘT seed (42) mỗi cấu hình, đúng config gốc
# (vast_run_cafe_rnn.sh), run_id mới đuôi -t12; hai hàng dashboard đó chấm với n=1.
T12_CAT=dl-cat-cafe-rnn-ce-s42-t12
T12_SAL=dl-sal-cafe-rnn-s42-t12
RNN_RUNS=dl-cat-rnn-ce,dl-cat-rnn-ce-s42,dl-cat-rnn-ce-s43,dl-cat-rnn-ce-s44,dl-cat-rnn-focal,\
dl-cat-bamibert-rnn-1024,dl-cat-visobert-rnn-512,dl-cat-xlmr-rnn-512,\
dl-mtl-cafe-rnn-a5-s42,dl-mtl-bamibert-rnn1024-a5-s42,dl-mtl-visobert-rnn512-a5-s42,dl-mtl-xlmr-rnn512-a5-s42,\
dl-sal-rnn,dl-sal-rnn-s42,dl-sal-rnn-s43,dl-sal-rnn-s44,dl-sal-bamibert-rnn-1024,dl-sal-visobert-rnn-512,dl-sal-xlmr-rnn-512,\
$T12_CAT,$T12_SAL
DENSE_RUNS=dl-cat-cafe-dense-512-p20-s42,dl-cat-cafe-dense-512-p20-s43,dl-cat-cafe-dense-512-p20-s44,\
dl-cat-focal-0920,dl-cat-focal-s2,dl-sal-0920,dl-sal-s2,dl-mtl-h256-a1.0-1001-0735,dl-mtl-h256-a1.0-1001-0746,\
dl-cat-cafe-dense-512,dl-cat-cafe-dense-512-s43,dl-cat-cafe-dense-512-s44,dl-cat-cafe-dense-512-s45,dl-cat-cafe-dense-512-s46,\
dl-cat-cafe-dense,dl-cat-cafe-focal-512-s42,dl-cat-cafe-focal-512-s43,dl-cat-cafe-focal-512-s44,dl-cat-cafe-focal-512-s45,dl-cat-cafe-focal-512-s46,\
dl-cat-ce-0920,dl-cat-bamibert-dense-1024,dl-cat-visobert-dense-512,dl-cat-xlmr-dense-512,\
dl-mtl-bamibert1024-a5-e80-s42,dl-mtl-visobert512-a5-e80-s42,dl-mtl-xlmr512-a5-e80-s42,\
dl-sal-cafe-dense-512,dl-sal-cafe-dense-512-s43,dl-sal-cafe-dense-512-s44,dl-sal-cafe-dense-512-s45,dl-sal-cafe-dense-512-s46,\
dl-sal-cpu-0920,dl-sal-cpu-s43,dl-sal-cpu-s44,dl-sal-cpu-s45,dl-sal-cpu-s46,\
dl-sal-bamibert-dense-1024,dl-sal-visobert-dense-512,dl-sal-xlmr-dense-512

check_sha() {
  local got; got=$(sha256sum data/processed/splits/test.csv | cut -d' ' -f1)
  [ "$got" = "$TEST_SHA" ] || { echo "test.csv sha256 sai: $got"; exit 1; }
  echo "test.csv sha256 ok ($1)"
}

encode() {
  $PYTHON scripts/encode_t12_cpu.py --device cuda \
    --jobs phobert:256 cafebert:256 cafebert:512 visobert:512 xlmr:512 bamibert:1024 2>&1 | tee -a logs/t12-encode.txt
}
train() {  # 2 run CafeBERT 512 + RNN (seed 42), đúng config gốc (T8.16); không ghi đè thư mục cũ
  for t in category salary; do
    $PYTHON -m vietjobs.dl.encode --task "$t" --splits train dev --encoder cafebert \
      --max-len 512 --pooling none --device cuda 2>&1 | tee -a logs/t12-train-encode.txt
  done
  for s in 42; do
    for spec in "category dl-cat-cafe-rnn-ce-s$s-t12 --loss ce" "salary dl-sal-cafe-rnn-s$s-t12"; do
      set -- $spec; local task=$1 rid=$2; shift 2
      [ -f "artifacts/$rid/metrics.json" ] && { echo "skip $rid"; continue; }
      $PYTHON -m vietjobs.dl.train_dl --task "$task" --head rnn --encoder cafebert --max-len 512 \
        --device cuda --hidden 100 --conv-channels 50 --seed "$s" --run-id "$rid" "$@" 2>&1 | tee "logs/$rid.txt"
    done
  done
}
score() {  # runs device
  for ds in test ext37k-sal; do
    $PYTHON scripts/score_t12.py --dataset "$ds" --confirm-test --device "$2" --runs "$1" 2>&1 | tee -a "logs/t12-score-$2-$ds.txt"
  done
}

check_sha before
case "${1:-all}" in
  encode) encode ;;
  train)  train ;;
  rnn)    score "$RNN_RUNS" cuda ;;
  dense)  score "$DENSE_RUNS" cpu ;;
  all)    encode; train; score "$RNN_RUNS" cuda; score "$DENSE_RUNS" cpu ;;
  *) echo "dùng: all | encode | train | rnn | dense"; exit 2 ;;
esac
check_sha after
echo ALL_DONE
