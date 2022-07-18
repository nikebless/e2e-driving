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
##SBATCH --exclude=falcon1,falcon2,falcon3

# Actual job command(s)

srun python -u train.py \
    --input-modality nvidia-camera \
    --output-modality steering_angle \
    --patience 10 \
    --max-epochs 100 \
    --model-name steering-angle \
    --model-type pilotnet-ebm \
    --loss ebm \
    --dataset-folder /gpfs/space/projects/Bolt/dataset-new-small/summer2021 \
    --batch-size 512 \
    --num-workers 16 \
    --wandb-project ibc \
    --stochastic-optimizer-train-samples 1024 \
    --stochastic-optimizer-inference-samples 1024 \
    --stochastic-optimizer-iters 3 \
