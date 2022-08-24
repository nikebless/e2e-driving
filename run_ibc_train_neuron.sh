#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --model-type pilotnet-ebm \
    --loss ebm \
    --dataset-folder /data/Bolt/dataset-cropped \
    --batch-size 512 \
    --num-workers 16 \
    --stochastic-optimizer-train-samples 512 \
    --stochastic-optimizer-inference-samples 512 \
    --stochastic-optimizer-iters 3 \
    --steering-bound 4.5 \
    --use-constant-samples \
    --temporal-group-size 2 \
    --temporal-regularization 0 \
    --temporal-regularization-ignore-target \
    --temporal-regularization-type l2 \
    --ebm-loss-type ce-proximity-aware \
    --ce-proximity-aware-temperature 0.5 \
    --debug \
    &> runs/$(date +%s)-run.txt &
