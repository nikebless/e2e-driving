#!/usr/bin/env bash

set -e
set -x

mkdir -p runs

nohup python -u train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --loss mae \
    --dataset-folder `/data/Bolt/end-to-end/rally-estonia-cropped-antialias \`
    --batch-size 256 \
    --num-workers 16 \
    --wandb-entity nikebless \
    --wandb-project ebm-driving \
    --model-type pilotnet \
    --debug \
    &> runs/$(date +%s)-run.txt &


`