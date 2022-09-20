#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python -u train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --loss ce \
    --dataset-folder /data/Bolt/end-to-end/rally-estonia-cropped \
    --batch-size 512 \
    --num-workers 16 \
    --wandb-entity nikebless \
    --wandb-project ebm-driving \
    --ebm-train-samples 512 \
    --ebm-inference-samples 512 \
    --model-type pilotnet-ebm \
    --loss-variant ce-proximity-aware \
    --temporal-regularization 0.3 \
    &> runs/$(date +%s)-run.txt &


