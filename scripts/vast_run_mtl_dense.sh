#!/usr/bin/env bash
set -e
cd /workspace/vietjobs
export PYTHONPATH=src
mkdir -p logs

echo "=== [1/5] PhoBERT Dense MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder phobert --max-len 256 --head dense --last4 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 15 --device cuda \
    --seed 42 --run-id dl-mtl-pho-dense-last4-cw-s42 \
    2>&1 | tee logs/dl-mtl-pho-dense-last4-cw-s42.txt

echo "=== [2/5] CafeBERT Dense MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder cafebert --max-len 512 --head dense --last4 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 15 --device cuda \
    --seed 42 --run-id dl-mtl-caf-dense-last4-cw-s42 \
    2>&1 | tee logs/dl-mtl-caf-dense-last4-cw-s42.txt

echo "=== [3/5] ViSoBERT Dense MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder visobert --max-len 512 --head dense --last4 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 15 --device cuda \
    --seed 42 --run-id dl-mtl-vis-dense-last4-cw-s42 \
    2>&1 | tee logs/dl-mtl-vis-dense-last4-cw-s42.txt

echo "=== [4/5] XLM-R Dense MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder xlmr --max-len 512 --head dense --last4 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 15 --device cuda \
    --seed 42 --run-id dl-mtl-xlm-dense-last4-cw-s42 \
    2>&1 | tee logs/dl-mtl-xlm-dense-last4-cw-s42.txt

echo "=== [5/5] BamiBERT Dense MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder bamibert --max-len 1024 --head dense --last4 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 15 --device cuda \
    --seed 42 --run-id dl-mtl-bam-dense-last4-cw-s42 \
    2>&1 | tee logs/dl-mtl-bam-dense-last4-cw-s42.txt

echo "=== ALL 5 MTL DENSE RUNS COMPLETED SUCCESSFULLY! ==="
