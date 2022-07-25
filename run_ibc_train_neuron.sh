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
    --stochastic-optimizer-train-samples 512 \
    --stochastic-optimizer-inference-samples 512 \
    --stochastic-optimizer-iters 3 \
    --steering-bound 4.5 \
    --use-constant-samples \
    --temporal-regularization 100 \
    --temporal-group-size 2 \
    --temporal-regularization-ignore-target \
    --temporal-regularization-type emd \
    --temporal-regularization-schedule exponential \
    --temporal-regularization-schedule-k 0.003 \
    --temporal-regularization-schedule-n 400 \
    --debug \
    &> runs/$(date +%s)-run.txt &
