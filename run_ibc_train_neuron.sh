#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python train.py \
    --input-modality nvidia-camera \
    --output-modality steering_angle \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --model-type pilotnet-ebm \
    --loss ebm \
    --dataset-folder /data/Bolt/dataset-new-small/summer2021 \
    --batch-size 512 \
    --num-workers 16 \
    --wandb-project ibc \
    --stochastic-optimizer-train-samples 32 \
    &> runs/$(date +%s)-run.txt &
