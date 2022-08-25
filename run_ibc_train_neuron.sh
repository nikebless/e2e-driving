#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --model-type pilotnet-classifier \
    --loss ce \
    --loss-variant ce-proximity-aware \
    --dataset-folder /data/Bolt/end-to-end/rally-estonia-cropped \
    --batch-size 512 \
    --num-workers 16 \
    --ebm-train-samples 256 \
    --ebm-inference-samples 256 \
    --temporal-regularization 0 \
    --debug \
    &> runs/$(date +%s)-run.txt &
