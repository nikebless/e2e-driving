#!/bin/bash
# NOTE: before running this do `conda activate itn2`.
# run as `sbatch run_ibc_train_hpc.sh`

# Job Details
#SBATCH --partition=gpu
#SBATCH -J train
#SBATCH -o ./runs/%j-slurm-run.txt # STDOUT/STDERR

# Resources
#SBATCH -t 23:59:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2,falcon3

# Actual job command(s)

srun python -u train.py \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --model-type pilotnet-ebm \
    --loss ce \
    --loss-variant default \
    --dataset-folder /gpfs/space/projects/Bolt/end-to-end/rally-estonia-cropped \
    --batch-size 512 \
    --num-workers 16 \
    --wandb-project ebm-e2e \
    --ebm-train-samples 512 \
    --ebm-inference-samples 512 \
    --temporal-regularization 0 \

