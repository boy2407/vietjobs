#!/usr/bin/env bash
# Đợt 5: Toàn bộ họ BERT (PhoBERT, CafeBERT, ViSoBERT, XLM-R, BamiBERT) — CHỈ BÀI TOÁN SALARY
# Cấu hình: last4 (MEAN pooling 4 tầng cuối) + Improved Loss: SmoothL1(beta=0.1) + Binned CW(0.999)
#
# Cách dùng:
#   bash scripts/vast_run_dot5.sh all          # chạy Dense (5 seeds) trước, sau đó chạy RNN (1 seed)
#   bash scripts/vast_run_dot5.sh dense        # chỉ chạy 5 mô hình Dense (5 seeds)
#   bash scripts/vast_run_dot5.sh rnn          # chỉ chạy 5 mô hình RNN (1 seed s42)
set -euo pipefail

PYTHON=${PYTHON:-/venv/main/bin/python}
export PYTHONPATH=src
mkdir -p logs artifacts/embeddings

echo "=== KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN SALARY (ĐỢT 5) ==="
nvidia-smi
free -h
df -h /workspace

# 1. Hàm Encode last4 pooled (cho Dense)
enc_pooled_last4() { # encoder max_len batch
  local e=$1 L=$2 b=$3
  local prefix=""
  [ "$e" != "phobert" ] && prefix="${e}-"
  
  local train_cache="artifacts/embeddings/${prefix}train-masked-len${L}-last4.npy"
  local dev_cache="artifacts/embeddings/${prefix}dev-masked-len${L}-last4.npy"

  if [ -f "$train_cache" ] && [ -f "$dev_cache" ]; then
    echo ">> [Pooled Last4] Cache đã tồn tại cho $e @ len $L -> BỎ QUA"
    return
  fi

  echo ">> [Pooled Last4] Đang encode $e (salary, masked, len=$L, batch=$b)..."
  $PYTHON -m vietjobs.dl.encode --task salary --splits train dev --encoder "$e" \
    --max-len "$L" --batch "$b" --last4 --device cuda 2>&1 | tee -a "logs/dot5-encode-pooled-$e.txt"
}

# 2. Hàm Train Dense Head (5 seeds)
train_dense_salary() { # encoder short_name max_len
  local e=$1 s=$2 L=$3
  echo "--- [Dense Salary] Bắt đầu $e ($s) @ len $L ---"

  # 5 seeds với Improved Loss (beta=0.1, CW=0.999)
  for seed in 42 43 44 45 46; do
    local rid="dl-sal-$s-dense-last4-b01-cw-s$seed"
    if [ -f "artifacts/$rid/metrics.json" ]; then
      echo ">> Đã có $rid -> BỎ QUA"
      continue
    fi
    echo ">> Train $rid..."
    $PYTHON -m vietjobs.dl.train_dl --task salary --encoder "$e" --max-len "$L" \
      --head dense --last4 --device cpu --run-id "$rid" \
      --beta-sal 0.1 --class-weight --seed "$seed" 2>&1 | tee "logs/$rid.txt"
  done

  # Ablation seed 42 không có CW
  local rid_ab="dl-sal-$s-dense-last4-b01-s42"
  if [ ! -f "artifacts/$rid_ab/metrics.json" ]; then
    echo ">> Train ablation $rid_ab..."
    $PYTHON -m vietjobs.dl.train_dl --task salary --encoder "$e" --max-len "$L" \
      --head dense --last4 --device cpu --run-id "$rid_ab" \
      --beta-sal 0.1 --seed 42 2>&1 | tee "logs/$rid_ab.txt"
  fi
}

# 3. Hàm Encode token cache (cho RNN)
enc_tokens() { # encoder max_len batch
  local e=$1 L=$2 b=$3
  local prefix=""
  [ "$e" != "phobert" ] && prefix="${e}-"
  
  local train_tok="artifacts/embeddings/${prefix}train-masked-len${L}-tok.f16.npy"
  local dev_tok="artifacts/embeddings/${prefix}dev-masked-len${L}-tok.f16.npy"

  if [ -f "$train_tok" ] && [ -f "$dev_tok" ]; then
    echo ">> [Token Cache] Đã tồn tại cho $e @ len $L -> BỎ QUA"
    return
  fi

  echo ">> [Token Cache] Đang encode $e (salary, masked, len=$L, batch=$b)..."
  $PYTHON -m vietjobs.dl.encode --task salary --splits train dev --encoder "$e" \
    --max-len "$L" --batch "$b" --pooling none --device cuda 2>&1 | tee -a "logs/dot5-encode-tok-$e.txt"
}

# 4. Hàm Train RNN Head (1 seed s42)
train_rnn_salary() { # encoder short_name max_len
  local e=$1 s=$2 L=$3
  local rid="dl-sal-$s-rnn-b01-cw-s42"
  if [ -f "artifacts/$rid/metrics.json" ]; then
    echo ">> Đã có $rid -> BỎ QUA"
    return
  fi
  echo "--- [RNN Salary] Bắt đầu $rid ($e @ len $L, CUDA) ---"
  $PYTHON -m vietjobs.dl.train_dl --task salary --encoder "$e" --max-len "$L" \
    --head rnn --hidden 100 --conv-channels 50 --device cuda --run-id "$rid" \
    --beta-sal 0.1 --class-weight --seed 42 2>&1 | tee "logs/$rid.txt"
}

run_all_dense() {
  echo "=================================================="
  echo "=== GIAI ĐOẠN 1: ENCODE POOLED LAST4 & TRAIN DENSE ==="
  echo "=================================================="

  # PhoBERT (len 256)
  enc_pooled_last4 phobert 256 64
  train_dense_salary phobert pho 256

  # CafeBERT (len 512)
  enc_pooled_last4 cafebert 512 32
  train_dense_salary cafebert cafe 512

  # ViSoBERT (len 512)
  enc_pooled_last4 visobert 512 32
  train_dense_salary visobert visobert 512

  # XLM-R (len 512)
  enc_pooled_last4 xlmr 512 32
  train_dense_salary xlmr xlmr 512

  # BamiBERT (len 1024)
  enc_pooled_last4 bamibert 1024 16
  train_dense_salary bamibert bamibert 1024
}

run_all_rnn() {
  echo "=================================================="
  echo "=== GIAI ĐOẠN 2: ENCODE TOKEN CACHE & TRAIN RNN (1 SEED S42) ==="
  echo "=================================================="

  # PhoBERT (len 256)
  enc_tokens phobert 256 64
  train_rnn_salary phobert phobert 256

  # CafeBERT (len 512)
  enc_tokens cafebert 512 32
  train_rnn_salary cafebert cafebert 512

  # ViSoBERT (len 512)
  enc_tokens visobert 512 32
  train_rnn_salary visobert visobert 512

  # XLM-R (len 512)
  enc_tokens xlmr 512 32
  train_rnn_salary xlmr xlmr 512

  # BamiBERT (len 1024)
  enc_tokens bamibert 1024 16
  train_rnn_salary bamibert bamibert 1024
}

TARGET=${1:-all}

case "$TARGET" in
  dense)
    run_all_dense
    ;;
  rnn)
    run_all_rnn
    ;;
  all)
    run_all_dense
    run_all_rnn
    ;;
  *)
    echo "Lựa chọn không hợp lệ: $TARGET" >&2
    echo "Cách dùng: $0 [all|dense|rnn]" >&2
    exit 2
    ;;
esac

echo "=================================================="
echo "=== HOÀN TẤT TOÀN BỘ TIẾN TRÌNH SALARY ĐỢT 5 (ALL_DONE) ==="
echo "=================================================="
