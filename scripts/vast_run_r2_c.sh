#!/usr/bin/env bash
# Máy C (Đợt 10): 3 run RNN masked chuyển từ máy A và B để xong sớm hơn ~1 giờ.
#   bamibert 1024: dl-mtl-bamibert-rnn1024-a5-s42-r2
#   visobert 512:  dl-sal-visobert-rnn-512-r2, dl-mtl-visobert-rnn512-a5-s42-r2
# Trên A và B, ba run này có metrics.json giữ chỗ nên script ở đó tự skip.
set -euo pipefail
source scripts/vast_run_r2.sh

nvidia-smi; free -g; df -h /workspace
check_sha before
$PYTHON -m pytest -q tests/test_dl_encode.py tests/test_dl_heads.py

enc visobert salary 512 16 --pooling none
st  dl-sal-visobert-rnn-512-r2 salary visobert 512 cuda $RNN
mtl dl-mtl-visobert-rnn512-a5-s42-r2 visobert 512 cuda rnn --hidden 100 --conv-channels 50
rm -f artifacts/embeddings/visobert-*-masked-len512-tok.f16.npy artifacts/embeddings/visobert-*-masked-len512-mask.npy

enc bamibert salary 1024 16 --pooling none
mtl dl-mtl-bamibert-rnn1024-a5-s42-r2 bamibert 1024 cuda rnn --hidden 100 --conv-channels 50
rm -f artifacts/embeddings/bamibert-*-masked-len1024-tok.f16.npy artifacts/embeddings/bamibert-*-masked-len1024-mask.npy

check_sha after-train
score dl-sal-visobert-rnn-512-r2,dl-mtl-visobert-rnn512-a5-s42-r2,dl-mtl-bamibert-rnn1024-a5-s42-r2 cuda
check_sha after
echo ALL_DONE
