#!/usr/bin/env bash
set -e
cd /workspace/vietjobs
export PYTHONPATH=src
mkdir -p logs

echo "=== [1/5] PhoBERT RNN MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder phobert --max-len 256 --head rnn --conv-channels 64 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 12 --device cuda \
    --seed 42 --run-id dl-mtl-pho-rnn-b01-cw-s42 \
    2>&1 | tee logs/dl-mtl-pho-rnn-b01-cw-s42.txt

echo "=== [2/5] CafeBERT RNN MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder cafebert --max-len 512 --head rnn --conv-channels 64 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 12 --device cuda \
    --seed 42 --run-id dl-mtl-caf-rnn-b01-cw-s42 \
    2>&1 | tee logs/dl-mtl-caf-rnn-b01-cw-s42.txt

echo "=== [3/5] ViSoBERT RNN MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder visobert --max-len 512 --head rnn --conv-channels 64 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 12 --device cuda \
    --seed 42 --run-id dl-mtl-vis-rnn-b01-cw-s42 \
    2>&1 | tee logs/dl-mtl-vis-rnn-b01-cw-s42.txt

echo "=== [4/5] XLM-R RNN MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder xlmr --max-len 512 --head rnn --conv-channels 64 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 12 --device cuda \
    --seed 42 --run-id dl-mtl-xlm-rnn-b01-cw-s42 \
    2>&1 | tee logs/dl-mtl-xlm-rnn-b01-cw-s42.txt

echo "=== [5/5] BamiBERT RNN MTL ==="
/venv/main/bin/python -m vietjobs.dl.train_mtl \
    --encoder bamibert --max-len 1024 --head rnn --conv-channels 50 --hidden 100 --class-weight \
    --alpha-sal 5.0 --beta-sal 0.1 --epochs 80 --patience 12 --device cuda \
    --seed 42 --run-id dl-mtl-bam-rnn-b01-cw-s42 \
    2>&1 | tee logs/dl-mtl-bam-rnn-b01-cw-s42.txt

echo "=== ALL 5 MTL RNN RUNS COMPLETED SUCCESSFULLY! ==="
