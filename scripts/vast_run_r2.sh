#!/usr/bin/env bash
# Train lại 15 run T11 trên ĐÚNG bộ chia của Mac (đuôi -r2), rồi chấm test + ext37k-sal.
#
#   bash scripts/vast_run_r2.sh xlmr visobert   # máy B: 6 run XLM-R + 3 run ViSoBERT RNN
#   bash scripts/vast_run_r2.sh bamibert        # máy A: 6 run BamiBERT 1024
#
# Lý do: 15 run của Đợt 4 có predictions_dev lệch dev.csv của Mac (nhãn khớp 93,1 %,
# 2.705 tin có lương thay vì 2.698) — máy thuê đã train trên một bộ chia khác.
# Script dừng ngay nếu sha256 của splits/manifest khác Mac, kiểm cả trước và sau.
# Cấu hình giữ nguyên vast_run_t11_encode.sh; Dense --device cpu (Rule 13 item 5), RNN cuda.
# Đĩa chỉ cần ≈ 60 GB: token cache của một encoder bị xoá ngay sau các run RNN của nó.
set -euo pipefail
PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs

check_sha() {
  ( cd data/processed && sha256sum -c --quiet - ) <<'EOF' || { echo "SHA SAI ($1) — dừng"; exit 1; }
ee457e6b47b2c72a5be6e9bfeaf80bf41f4e6e3e909abbb2e59dbaa6020a1758  splits/train.csv
7b88f3bb627d38b30e6e265a677f089a94a6b4210d6e224105f4019ce1131d9d  splits/dev.csv
70ba98d58397c8454f04153ace602bc8ed760acdcf2b5ad65d197af572fba047  splits/test.csv
1f326f3bba678529c64acb735da75874e6d8109ab07a9c6f2c71e25576b8ac18  manifest.json
EOF
  echo "sha256 splits + manifest ok ($1)"
}

enc() {  # encoder task max_len batch [--pooling none]
  local e=$1 t=$2 L=$3 b=$4; shift 4
  $PYTHON -m vietjobs.dl.encode --task "$t" --splits train dev --encoder "$e" \
    --max-len "$L" --batch "$b" --device cuda "$@" 2>&1 | tee -a "logs/r2-encode-$e.txt"
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
DENSE_RUNS=""; RNN_RUNS=""

one() {  # encoder len short [rnn_only]
  local e=$1 L=$2 s=$3
  if [ "${4:-}" != rnn_only ]; then
    for t in category salary; do enc "$e" "$t" "$L" 32; done
    st  "dl-cat-$e-dense-$L-r2" category "$e" "$L" cpu
    st  "dl-sal-$e-dense-$L-r2" salary   "$e" "$L" cpu
    mtl "dl-mtl-$s$L-a5-e80-s42-r2" "$e" "$L" cpu dense
    DENSE_RUNS+="dl-cat-$e-dense-$L-r2,dl-sal-$e-dense-$L-r2,dl-mtl-$s$L-a5-e80-s42-r2,"
  fi
  # Từng họ một (raw rồi masked), xoá ngay sau dùng: BamiBERT 1024 ≈ 60 GB mỗi họ, đĩa 120 GB.
  enc "$e" category "$L" 16 --pooling none
  st  "dl-cat-$e-rnn-$L-r2" category "$e" "$L" cuda $RNN --loss ce
  rm -f artifacts/embeddings/"$e"-*-raw-len"$L"-tok.f16.npy artifacts/embeddings/"$e"-*-raw-len"$L"-mask.npy
  enc "$e" salary "$L" 16 --pooling none
  st  "dl-sal-$e-rnn-$L-r2" salary   "$e" "$L" cuda $RNN
  mtl "dl-mtl-$e-rnn$L-a5-s42-r2" "$e" "$L" cuda rnn --hidden 100 --conv-channels 50
  rm -f artifacts/embeddings/"$e"-*-masked-len"$L"-tok.f16.npy artifacts/embeddings/"$e"-*-masked-len"$L"-mask.npy
  RNN_RUNS+="dl-cat-$e-rnn-$L-r2,dl-sal-$e-rnn-$L-r2,dl-mtl-$e-rnn$L-a5-s42-r2,"
  df -h /workspace | tail -1
}
score() {  # runs device
  for ds in test ext37k-sal; do
    $PYTHON scripts/score_t12.py --dataset "$ds" --confirm-test --device "$2" --runs "${1%,}" 2>&1 \
      | tee -a "logs/r2-score-$2-$ds.txt"
  done
}

[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0  # source để dùng hàm (vast_run_r2_c.sh)

nvidia-smi; free -g; df -h /workspace
check_sha before
$PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py
JOBS=()
for e in "$@"; do
  case "$e" in
    xlmr)     one xlmr 512 xlmr;              JOBS+=(xlmr:512) ;;
    visobert) one visobert 512 visobert rnn_only ;;
    bamibert) one bamibert 1024 bamibert;     JOBS+=(bamibert:1024) ;;
    *) echo "dùng: $0 [xlmr] [visobert] [bamibert]" >&2; exit 2 ;;
  esac
done
check_sha after-train
[ ${#JOBS[@]} -gt 0 ] && $PYTHON scripts/encode_t12_cpu.py --device cuda --jobs "${JOBS[@]}" 2>&1 | tee -a logs/r2-encode-t12.txt
[ -n "$RNN_RUNS" ] && score "$RNN_RUNS" cuda
[ -n "$DENSE_RUNS" ] && score "$DENSE_RUNS" cpu
check_sha after
echo ALL_DONE
