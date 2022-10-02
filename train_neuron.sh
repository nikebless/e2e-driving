#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python -u train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --loss mae \
    --dataset-folder /data/Bolt/end-to-end/rally-estonia-cropped \
    --batch-size 256 \
    --num-workers 16 \
    --wandb-entity nikebless \
    --wandb-project ebm-driving \
    --ebm-train-samples 512 \
    --ebm-inference-samples 512 \
    --model-type pilotnet-mdn \
    --temporal-regularization 0 \
    --mdn-n-components 3 \
    --mdn-init-biases -90 0 90 \
    --mdn-lambda-sigma 0 \
    --mdn-lambda-pi 0 \
    --mdn-lambda-mu 0 \
    --debug \
    &> runs/$(date +%s)-run.txt &


