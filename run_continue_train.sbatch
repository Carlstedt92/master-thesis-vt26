#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 240:00:00
#SBATCH -p haswell
#SBATCH -c 16
#SBATCH --gpus=t4:1
#SBATCH -J VAL_TESTING_CONTINUE_TRAINING
#SBATCH -o logs/VAL_TESTING_continue_training_%j.out
#SBATCH -e logs/VAL_TESTING_continue_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run continue_training.py --model VAL_TESTING --checkpoint-name checkpoint_epoch_200.pth --target-epochs 400