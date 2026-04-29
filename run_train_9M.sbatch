#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 120:00:00
#SBATCH -p haswell
#SBATCH -c 16
#SBATCH --gpus=t4:1
#SBATCH -J 512_DIM_GINE_training
#SBATCH -o 512_DIM_GINE_training_%j.out
#SBATCH -e 512_DIM_GINE_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run train.py --config configs/TEST.json